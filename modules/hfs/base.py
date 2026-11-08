"""
Level-1 subsystem contract of the hierarchical fuzzy system (A1 / TOD-41).

A subsystem is a Mamdani fuzzy system with its own rule base ``config/fuzzy/<name>.yaml`` (triangular / trapezoidal
terms) and its own update rate (``update_sec`` in that file). It learns from the event stream (``observe``), turns
what it saw in a location into crisp inputs (``features``) and infers that location's risk (``infer``)::

    {"risk": 0.84, "level": "critical", "interval": [0.79, 1.0], "fired_rules": [...], "inputs": {...},
     "partial": False}

``features`` returns None when the subsystem has no data for the location (no device / device offline / stale
readings) — the top level then works on the remaining subsystems and marks its result partial.

A5 (TOD-45): ``features`` also returns ``_deltas`` — the measured error δ of every input (``uncertainty.py``,
``config/uncertainty.yaml``). The crisp risk is the type-1 inference; the **corridor** ``interval`` = [y_l, y_r] of
the interval type-2 inference with every membership edge moved by ±δ (``core.fuzzy.engine2``, Karnik–Mendel,
area-weighted: δ = 0 gives exactly the type-1 risk). ``widened_by`` ranks the inputs by their share of the width.
"""
import os
import time
from typing import Any, Dict, Iterable, List, Optional

from core.events import Event
from core.fuzzy import FuzzyEngine, load_yaml
from core.fuzzy.engine import location_overrides
from core.fuzzy.engine2 import FuzzyEngine2
from modules.hfs.uncertainty import DEFAULT_PATH, Uncertainty

CONFIG_DIR = os.path.join("config", "fuzzy")
FIRED_RULES_KEPT = 5            # strongest fired rules carried in every result (explanation)
ALPHA = 0.5                     # α-cut of the type-1 output set (reported as ``alpha_cut``; the corridor is IT2)


def level_of(engine: FuzzyEngine, crisp: float) -> str:
    """:return: the output term with the largest membership at the crisp value (low / medium / high / critical)"""
    terms = engine.fuzzify(engine.output, crisp)
    return max(terms, key=lambda term: terms[term])


def summarise(engine: FuzzyEngine, result: Dict[str, Any], it2: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    :param result: one ``FuzzyEngine.infer`` (the crisp risk, rules)
    :param it2: the ``FuzzyEngine2.infer`` of the same inputs with the measured δ (None: no error → no corridor)
    :return: risk, level, interval (= the IT2 corridor [y_l, y_r], widened to contain the risk), rules, inputs
    """
    crisp = float(result["crisp"])
    low, high = it2["interval"] if it2 is not None else (crisp, crisp)
    low, high = min(low, crisp), max(high, crisp)
    out = {"risk": round(crisp, 4), "level": level_of(engine, crisp),
           "interval": [round(low, 4), round(high, 4)], "corridor_width": round(high - low, 4),
           "fired_rules": [{k: rule[k] for k in ("id", "term", "strength", "explanation")}
                           for rule in result["fired_rules"][:FIRED_RULES_KEPT]],
           "inputs": {name: info["value"] for name, info in result["inputs"].items() if not info["defaulted"]}}
    if it2 is not None:
        out["widened_by"] = it2["widened_by"][:3]
    return out


class FuzzySubsystem:
    """Base class: rule base + update rate; subclasses implement ``observe`` / ``features``."""
    name = "subsystem"
    device_kind: Optional[str] = None     # devices.db kind whose online device in a location means "data available"
    event_types: tuple = ()

    def __init__(self, config: Dict[str, Any], path: Optional[str] = None, clock=time.time):
        """
        :param config: Full merged config (``hfs.<name>`` section + what the subsystem needs, e.g. hfs.gas.gases)
        :param path: Rule base (``hfs.<name>.rules`` or ``config/fuzzy/<name>.yaml``)
        """
        self.config = config
        self.cfg = (config.get("hfs") or {}).get(self.name) or {}
        self.path = path or self.cfg.get("rules") or os.path.join(CONFIG_DIR, f"{self.name}.yaml")
        self.data = load_yaml(self.path)
        self.update_sec = float(self.cfg.get("update_sec", self.data.get("update_sec", 1.0)))
        self.stale_sec = float(self.cfg.get("stale_sec", 30))
        self.clock = clock
        self._engines: Dict[Optional[str], FuzzyEngine] = {}
        self._engines2: Dict[Optional[str], FuzzyEngine2] = {}
        self.engine = self.engine_for(None)
        self.last_seen: Dict[str, float] = {}         # location → last event of this subsystem
        self.sources: Dict[str, Dict[str, float]] = {}   # location → device → its last event (TOD-51: one device)
        hfs = config.get("hfs") or {}
        self.uncertainty = Uncertainty.load(hfs.get("uncertainty", DEFAULT_PATH) if hfs.get("it2", True) else None,
                                            getattr(self, "table", None))
        self.uncertainty_scale = float(hfs.get("uncertainty_scale", 1.0))   # A6 experiments: corridor vs δ
        self.it2 = bool(hfs.get("it2", True))            # false: the type-1 system, every δ ignored

    def engine_for(self, location_id: Optional[str]) -> FuzzyEngine:
        """Per-location overrides from the ``locations:`` block of the rule base (same format as fuzzy.yaml)."""
        key = location_id if location_id in (self.data.get("locations") or {}) else None
        if key not in self._engines:
            self._engines[key] = FuzzyEngine.from_dict(self.data, key, location_overrides(self.data, key))
            self._engines2[key] = FuzzyEngine2(self._engines[key], area_weighted=True)
        return self._engines[key]

    # --- input ------------------------------------------------------------------------------------------------------

    def wants(self, event: Event) -> bool:
        return event.type in self.event_types

    def observe(self, event: Event, now: float) -> None:
        """Updates the per-location evidence from one event (called for ``event_types`` only)."""
        self.last_seen[event.location_id] = now
        self.sources.setdefault(event.location_id, {})[event.device_id] = now

    def devices_of(self, location_id: str, now: float) -> set:
        """:return: the devices whose evidence of this location is still fresh (``stale_sec``)"""
        return {d for d, t in (self.sources.get(location_id) or {}).items() if now - t <= self.stale_sec}

    def forget_device(self, device_id: str) -> None:
        """A device went offline: its evidence no longer counts."""

    def locations(self) -> Iterable[str]:
        return list(self.last_seen)

    # --- inference --------------------------------------------------------------------------------------------------

    def features(self, location_id: str, now: float, available: bool) -> Optional[Dict[str, Any]]:
        """
        :param available: an online device of ``device_kind`` is in the location
        :return: crisp inputs of the rule base, None when the subsystem has no data for the location
        """
        raise NotImplementedError

    def infer(self, inputs: Optional[Dict[str, Any]], location_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        :param inputs: crisp inputs; keys starting with ``_`` are not inputs but are carried into the result without
                       the underscore (``_intervals`` → ``input_intervals``: input uncertainty for the FOU, TOD-45)
        :return: {risk, level, interval, fired_rules, inputs, partial, …} or None without inputs
        """
        if inputs is None:
            return None
        deltas = {k: float(v) * self.uncertainty_scale for k, v in (inputs.get("_deltas") or {}).items()
                  if v and self.it2}
        extras = {("input_intervals" if k == "_intervals" else k[1:]): v for k, v in inputs.items()
                  if k[:1] == "_" and k != "_deltas"}
        crisp = {k: v for k, v in inputs.items() if k[:1] != "_"}
        engine = self.engine_for(location_id)
        key = location_id if location_id in (self.data.get("locations") or {}) else None
        it2 = self._engines2[key].infer(crisp, deltas=deltas) if any(d > 0 for d in deltas.values()) else None
        return {**summarise(engine, engine.infer(crisp, alpha=ALPHA), it2), "partial": False,
                "deltas": {k: round(v, 4) for k, v in deltas.items()}, **extras}

    def evaluate(self, location_id: str, now: float, available: bool) -> Optional[Dict[str, Any]]:
        return self.infer(self.features(location_id, now, available), location_id)

    def recent(self, location_id: str, now: float) -> bool:
        return now - self.last_seen.get(location_id, float("-inf")) <= self.stale_sec


def window(items: List[Any], now: float, seconds: float) -> List[Any]:
    """Drops ``(ts, …)`` items older than ``seconds`` in place. :return: the list"""
    while items and now - items[0][0] > seconds:
        items.pop(0)
    return items
