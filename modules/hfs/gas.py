"""
Gas and PM2.5 subsystem (A1 / TOD-41, A2 / TOD-42) → risk 1. Rule base ``config/fuzzy/gas.yaml``.

Every substance is compared as a share of its critical concentration, c_norm = c / c_crit (``config/gases.yaml``:
NIOSH IDLH, EPA AQI Hazardous for PM2.5; ``gas_normalize.py``):

* ``max_norm`` — max c_norm over the substances read in the location in the last ``stale_sec``;
* A8 (TOD-50) multi-gas escalation (``gas_escalation.py``): ``second_norm`` / ``third_norm`` — the second / third
  highest c_norm of the *active* gases (lower bound of the reading's interval above ``activity.calm_norm``), and
  ``group_index`` — max over effect groups of Σ c_norm inside the group (additive mixture rule); readings only from
  cross-sensitive sensors without a selective one do not escalate, benign co-occurrences (``benign`` of
  config/gases.yaml, location rule ``gas_benign``; e.g. CO₂ while people are in view) are not counted. Every
  decision with an active gas carries a Ukrainian ``explanation`` (logged when it changes) and ``escalation``;
* ``dynamics`` — least-squares slope of max_norm per minute over the last ``window_sec``.

Every ``gas_reading`` event is the current state of its sensor: it replaces that sensor's previous substances (the
TOD-19 graphene classifier reports one gas per reading, so a classifier flipping between two gases never adds them
up); a sensor measuring several substances at once sends ``readings: [...]``. Unknown substances (no c_crit) are
ignored with a warning; clean air reads 0. No fresh reading in the location → no data (None).

TOD-48: a reading that carries its sensor's ``accuracy`` / ``range`` / ``resolution`` (the gas drivers send them)
uses them for δ (``gas_model_builder.reading_interval``: δ(c) = max(absolute, relative · c, resolution / 2)) instead
of ``config/uncertainty.yaml``; a reading at full scale of a sensor whose range ends below C_crit is an open upper
bound — the interval reaches 1.2 C_crit, so the corridor shows that a critical level cannot be excluded, and the
result lists it under ``saturated``.

Sampling and inference periods (``hfs.gas`` in config.yaml, see docs/sensors.md): the graphene board is polled every
``gas.gas_poll_sec`` (1 s), electrochemical / particle modules report about every 1 s; the subsystem infers every
``update_sec`` (2 s); the dynamics window is 60 s — longer than the T90 response of electrochemical sensors (≤ 30 s CO,
≤ 150 s NH₃ for the DFRobot modules), so the slope follows the gas, not sensor noise, and holds ≥ 30 samples.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from core.events import Event
from modules.hfs.base import FuzzySubsystem, window
from modules.hfs.gas_escalation import GasState, analyse, explain
from modules.hfs.gas_normalize import GasTable, UnitError, reading_items, slope_per_minute
from modules.hfs.gas_model_builder import reading_interval
from system_setup.logger_setup import init_logger

logger = init_logger()

CLEAN = ("clean_air", "air", "none")


PEOPLE_PRESENT = "відомий нешкідливий збіг — у приміщенні є люди"
EXCEPTION = "виняток для цієї локації"


@dataclass
class SensorState:
    """The latest reading of one sensor (a gas_reading is the sensor's whole current state)."""
    ts: float
    norms: Dict[str, float] = field(default_factory=dict)       # substance → c / c_crit
    deltas: Dict[str, float] = field(default_factory=dict)      # substance → δ of c / c_crit (half-width, A5)
    lows: Dict[str, float] = field(default_factory=dict)        # substance → lower bound of c / c_crit
    selectivity: Optional[str] = None                           # selective | cross_sensitive | None
    saturated: set = field(default_factory=set)


class GasSubsystem(FuzzySubsystem):
    name = "gas"
    device_kind = "gas"
    event_types = ("gas_reading", "object_detected")

    def __init__(self, config: Dict[str, Any], path: Optional[str] = None, table: Optional[GasTable] = None,
                 rules: Callable[[str], Dict[str, Any]] = lambda location_id: {}, **kwargs):
        """:param rules: location_id → location rules (``gas_benign``)"""
        super().__init__(config, path, **kwargs)
        self.table = table or GasTable.load(self.cfg.get("gases", "config/gases.yaml"))
        self.uncertainty.table = self.table                 # A5: reading errors are converted per substance
        self.rules = rules
        self.window_sec = float(self.cfg.get("window_sec", 60))
        self.min_samples = int(self.cfg.get("min_samples", 3))
        self.min_span_sec = float(self.cfg.get("min_span_sec", 10))
        self.readings: Dict[str, Dict[str, SensorState]] = {}         # location → device → its latest reading
        self.history: Dict[str, List[tuple]] = {}                     # location → [(ts, max_norm)]
        self.people: Dict[str, set] = {}                              # location → persons in view (benign rule)
        self.analysis: Dict[str, Any] = {}                            # location → last escalation analysis
        self.explained: Dict[str, str] = {}                           # location → last logged explanation
        self._warned: set = set()

    def normalise(self, payload: Dict[str, Any], state: Optional[SensorState] = None) -> Dict[str, float]:
        """
        :param state: filled with the δ, lower bounds and saturation of every substance (A5 / TOD-48)
        :return: {substance: c_norm} of one reading (clean air → {})
        """
        out: Dict[str, float] = {}
        for item in reading_items(payload):
            gas, value, unit = item["gas"], item["value"], item["unit"]
            if str(gas or "").lower() in CLEAN:
                continue
            substance = self.table.get(gas)
            if substance is None:
                if gas not in self._warned:
                    self._warned.add(gas)
                    logger.warning(f"HFS gas: no critical concentration for {gas!r} in config/gases.yaml — ignored")
                continue
            try:
                norm = substance.norm(value, unit)
                out[substance.name] = max(out.get(substance.name, 0.0), norm)
                if state is None:
                    continue
                if "accuracy" in item or "range" in item:
                    low, high, is_open = reading_interval(value, unit, item, substance)
                    if is_open:
                        state.saturated.add(substance.name)
                        self._saturated(payload, substance.name, value, unit, item)
                else:
                    delta = self.uncertainty.norm_delta(substance.name, value, unit, payload.get("sensor_model"))
                    low, high = max(0.0, norm - delta), norm + delta
                state.deltas[substance.name] = max(norm - low, high - norm)
                state.lows[substance.name] = low
            except UnitError as e:
                logger.warning(f"HFS gas: {gas} {value} {unit}: {e}")
        return out

    def _saturated(self, payload: Dict[str, Any], gas: str, value: float, unit: str, item: Dict[str, Any]) -> None:
        key = (payload.get("sensor_model") or payload.get("driver"), gas)
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(f"HFS gas: {gas} {value} {unit} is the full scale of {key[0] or 'the sensor'} (range "
                           f"{item.get('range')}): at least this much — the sensor cannot confirm a critical "
                           "concentration")

    @staticmethod
    def selectivity_of(payload: Dict[str, Any]) -> Optional[str]:
        """TOD-48 descriptor flag carried by the readings: selective wins (one selective quantity confirms)."""
        found = {item.get("selectivity") for item in payload.get("readings") or [payload]}
        found |= {payload.get("selectivity")}
        return "selective" if "selective" in found else "cross_sensitive" if "cross_sensitive" in found else None

    def observe(self, event: Event, now: float) -> None:
        if event.type == "object_detected":                 # only for the benign rule "people_present"
            if event.payload.get("class") == "person":
                people = self.people.setdefault(event.location_id, set())
                key = (event.device_id, event.payload.get("track_id"))
                (people.discard if event.payload.get("state") == "disappeared" else people.add)(key)
            return
        super().observe(event, now)
        state = SensorState(now, selectivity=self.selectivity_of(event.payload))
        state.norms = self.normalise(event.payload, state)
        self.readings.setdefault(event.location_id, {})[event.device_id] = state
        self.history.setdefault(event.location_id, []).append((now, self._max_norm(event.location_id, now)))

    def forget_device(self, device_id: str) -> None:
        for per_location in self.readings.values():
            per_location.pop(device_id, None)
        for people in self.people.values():
            for key in [k for k in people if k[0] == device_id]:
                people.discard(key)

    def _fresh_states(self, location_id: str, now: float) -> Optional[Dict[str, GasState]]:
        """:return: substance → its strongest fresh reading (+ δ, lower bound, selectivity), None without data"""
        fresh = [s for s in (self.readings.get(location_id) or {}).values() if now - s.ts <= self.stale_sec]
        if not fresh:
            return None
        merged: Dict[str, GasState] = {}
        for state in fresh:
            for name, value in state.norms.items():
                substance = self.table.substances[name]
                current = merged.get(name)
                if current is None or value >= current.norm:
                    previous = current.selectivity if current else None
                    merged[name] = GasState(name, value, state.lows.get(name, value), substance.group,
                                            substance.name_uk, state.selectivity, delta=state.deltas.get(name, 0.0))
                    if previous == "selective":
                        merged[name].selectivity = "selective"
                elif state.selectivity == "selective":
                    current.selectivity = "selective"   # a selective sensor confirms the substance
        return merged

    def _fresh(self, location_id: str, now: float) -> Optional[Dict[str, float]]:
        """:return: {substance: c_norm} over fresh sensors (max per substance), None without a fresh reading"""
        states = self._fresh_states(location_id, now)
        return None if states is None else {name: g.norm for name, g in states.items()}

    def _max_norm(self, location_id: str, now: float) -> float:
        return max((self._fresh(location_id, now) or {}).values(), default=0.0)

    def benign(self, location_id: str, gases: Dict[str, GasState]) -> Dict[str, str]:
        """:return: gas → reason, for the gases a benign exception excludes from the escalation now"""
        rule = self.rules(location_id).get("gas_benign")
        entries = rule if rule is not None else self.table.benign
        out = {}
        for entry in entries or []:
            when = entry.get("when", "always")
            if when == "people_present" and not self.people.get(location_id):
                continue
            for gas in entry.get("gases") or []:
                substance = self.table.get(gas)
                if substance is not None and substance.name in gases:
                    out[substance.name] = entry.get("reason") or (PEOPLE_PRESENT if when == "people_present"
                                                                   else EXCEPTION)
        return out

    def level_of(self, norm: float) -> str:
        terms = self.engine.fuzzify("max_norm", norm)
        return max(terms, key=terms.get)

    def features(self, location_id: str, now: float, available: bool) -> Optional[Dict[str, Any]]:
        gases = self._fresh_states(location_id, now)
        if gases is None:
            return None
        a = analyse(list(gases.values()), self.table.calm_norm, self.level_of, self.benign(location_id, gases))
        self.analysis[location_id] = a
        samples = window(self.history.get(location_id, []), now, self.window_sec)
        by_name = {g.name: g for g in a.gases}
        top = a.gases[0] if a.gases else None
        top_delta = top.delta if top else 0.0
        counted = [by_name[n] for n in a.counted]
        delta_of = lambda i: counted[i].delta if len(counted) > i else 0.0   # noqa: E731
        group_delta = sum(g.delta for g in counted if (g.group or g.name) == a.group)
        return {"max_norm": round(a.max_norm, 4), "second_norm": round(a.second_norm, 4),
                "group_index": round(a.group_index, 4), "third_norm": round(a.third_norm, 4),
                "dynamics": round(slope_per_minute(samples, self.min_samples, self.min_span_sec), 4),
                # A5: δ of each gas; Σ δ for the group sum (worst case); slope error over the window's times
                "_deltas": {"max_norm": top_delta, "second_norm": delta_of(1), "third_norm": delta_of(2),
                            "group_index": group_delta,
                            "dynamics": self.uncertainty.dynamics_delta(top_delta, [t for t, _ in samples])},
                "_escalation": a.payload()}

    def infer(self, inputs: Optional[Dict[str, Any]], location_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        result = super().infer(inputs, location_id)
        if result is None or location_id is None:
            return result
        now = self.clock()
        gases = self._fresh_states(location_id, now) or {}
        result["substances"] = {name: round(g.norm, 4) for name, g in sorted(gases.items())}
        fresh = [s for s in (self.readings.get(location_id) or {}).values() if now - s.ts <= self.stale_sec]
        saturated = sorted({g for s in fresh for g in s.saturated})
        if saturated:
            result["saturated"] = saturated
        a = self.analysis.get(location_id)
        if a is not None:
            dyn = self.engine.fuzzify("dynamics", float(inputs.get("dynamics", 0.0)))
            third = [g for g in a.gases if g.name in a.counted][2:3]
            text = explain(a, result["level"], {k: v.get("name_uk", k) for k, v in self.table.groups.items()},
                           dynamics_sharp=max(dyn, key=dyn.get) == "sharp_rise",
                           third_high=bool(third) and third[0].level in ("високий", "критичний"))
            if text:
                result["explanation"] = text
                if self.explained.get(location_id) != text:
                    self.explained[location_id] = text
                    logger.info(f"HFS gas {location_id}: {text}")
        return result
