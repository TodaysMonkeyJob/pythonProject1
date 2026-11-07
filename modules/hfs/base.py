"""
Level-1 subsystem contract of the hierarchical fuzzy system (A1 / TOD-41).

A subsystem is a Mamdani fuzzy system with its own rule base ``config/fuzzy/<name>.yaml`` (triangular / trapezoidal
terms) and its own update rate (``update_sec`` in that file). It learns from the event stream (``observe``), turns
what it saw in a location into crisp inputs (``features``) and infers that location's risk (``infer``)::

    {"risk": 0.84, "level": "critical", "interval": [0.79, 1.0], "fired_rules": [...], "inputs": {...},
     "partial": False}

``features`` returns None when the subsystem has no data for the location (no device / device offline / stale
readings) — the top level then works on the remaining subsystems and marks its result partial.
"""
import os
import time
from typing import Any, Dict, Iterable, List, Optional

from core.events import Event
from core.fuzzy import FuzzyEngine, load_yaml
from core.fuzzy.engine import location_overrides

CONFIG_DIR = os.path.join("config", "fuzzy")
FIRED_RULES_KEPT = 5            # strongest fired rules carried in every result (explanation)
ALPHA = 0.5                     # the interval = α-cut of the aggregated output set (A5 / TOD-45: IT2 corridor)


def level_of(engine: FuzzyEngine, crisp: float) -> str:
    """:return: the output term with the largest membership at the crisp value (low / medium / high / critical)"""
    terms = engine.fuzzify(engine.output, crisp)
    return max(terms, key=lambda term: terms[term])


def summarise(engine: FuzzyEngine, result: Dict[str, Any]) -> Dict[str, Any]:
    """:return: the subsystem / top-level result of one ``FuzzyEngine.infer`` (risk, level, interval, rules)"""
    crisp = float(result["crisp"])
    low, high = result["cut"] or (crisp, crisp)
    return {"risk": round(crisp, 4), "level": level_of(engine, crisp),
            "interval": [round(min(low, crisp), 4), round(max(high, crisp), 4)],
            "fired_rules": [{k: rule[k] for k in ("id", "term", "strength", "explanation")}
                            for rule in result["fired_rules"][:FIRED_RULES_KEPT]],
            "inputs": {name: info["value"] for name, info in result["inputs"].items() if not info["defaulted"]}}


class FuzzySubsystem:
    """Base class: rule base + update rate; subclasses implement ``observe`` / ``features``."""
    name = "subsystem"
    device_kind: Optional[str] = None     # devices.db kind whose online device in a location means "data available"
    event_types: tuple = ()

    def __init__(self, config: Dict[str, Any], path: Optional[str] = None, clock=time.time):
        """
        :param config: Full merged config (``hfs.<name>`` section + what the subsystem needs, e.g. gas.thresholds)
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
        self.engine = self.engine_for(None)
        self.last_seen: Dict[str, float] = {}         # location → last event of this subsystem

    def engine_for(self, location_id: Optional[str]) -> FuzzyEngine:
        """Per-location overrides from the ``locations:`` block of the rule base (same format as fuzzy.yaml)."""
        key = location_id if location_id in (self.data.get("locations") or {}) else None
        if key not in self._engines:
            self._engines[key] = FuzzyEngine.from_dict(self.data, key, location_overrides(self.data, key))
        return self._engines[key]

    # --- input ------------------------------------------------------------------------------------------------------

    def wants(self, event: Event) -> bool:
        return event.type in self.event_types

    def observe(self, event: Event, now: float) -> None:
        """Updates the per-location evidence from one event (called for ``event_types`` only)."""
        self.last_seen[event.location_id] = now

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
        extras = {("input_intervals" if k == "_intervals" else k[1:]): v for k, v in inputs.items() if k[:1] == "_"}
        crisp = {k: v for k, v in inputs.items() if k[:1] != "_"}
        engine = self.engine_for(location_id)
        return {**summarise(engine, engine.infer(crisp, alpha=ALPHA)), "partial": False, **extras}

    def evaluate(self, location_id: str, now: float, available: bool) -> Optional[Dict[str, Any]]:
        return self.infer(self.features(location_id, now, available), location_id)

    def recent(self, location_id: str, now: float) -> bool:
        return now - self.last_seen.get(location_id, float("-inf")) <= self.stale_sec


def window(items: List[Any], now: float, seconds: float) -> List[Any]:
    """Drops ``(ts, …)`` items older than ``seconds`` in place. :return: the list"""
    while items and now - items[0][0] > seconds:
        items.pop(0)
    return items
