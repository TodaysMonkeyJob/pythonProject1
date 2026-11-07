"""
Gas and PM2.5 subsystem (A1 / TOD-41, A2 / TOD-42) → risk 1. Rule base ``config/fuzzy/gas.yaml``.

Every substance is compared as a share of its critical concentration, c_norm = c / c_crit (``config/gases.yaml``:
NIOSH IDLH, EPA AQI Hazardous for PM2.5; ``gas_normalize.py``):

* ``max_norm`` — max c_norm over the substances read in the location in the last ``stale_sec``;
* ``mixture_index`` — Σ c_norm (additive mixture rule): two gases at 0.6 are more dangerous than one;
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
from typing import Any, Dict, List, Optional, Tuple

from core.events import Event
from modules.hfs.base import FuzzySubsystem, window
from modules.hfs.gas_normalize import GasTable, UnitError, reading_items, slope_per_minute
from modules.hfs.gas_model_builder import reading_interval
from system_setup.logger_setup import init_logger

logger = init_logger()

CLEAN = ("clean_air", "air", "none")


class GasSubsystem(FuzzySubsystem):
    name = "gas"
    device_kind = "gas"
    event_types = ("gas_reading",)

    def __init__(self, config: Dict[str, Any], path: Optional[str] = None, table: Optional[GasTable] = None,
                 **kwargs):
        super().__init__(config, path, **kwargs)
        self.table = table or GasTable.load(self.cfg.get("gases", "config/gases.yaml"))
        self.uncertainty.table = self.table                 # A5: reading errors are converted per substance
        self.window_sec = float(self.cfg.get("window_sec", 60))
        self.min_samples = int(self.cfg.get("min_samples", 3))
        self.min_span_sec = float(self.cfg.get("min_span_sec", 10))
        # location → {device: (ts, {substance: c_norm}, {substance: δ of c_norm})} — the latest state of each sensor
        self.readings: Dict[str, Dict[str, Tuple[float, Dict[str, float], Dict[str, float]]]] = {}
        self.history: Dict[str, List[Tuple[float, float]]] = {}       # location → [(ts, max_norm)]
        self._warned: set = set()
        self._open: set = set()
        self.open_bounds: Dict[str, Dict[str, set]] = {}             # location → device → saturated substances

    def normalise(self, payload: Dict[str, Any], deltas: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        :param deltas: filled with {substance: δ of c_norm} (A5: the sensor's accuracy, config/uncertainty.yaml)
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
                if deltas is not None and ("accuracy" in item or "range" in item):
                    low, high, is_open = reading_interval(value, unit, item, substance)
                    deltas[substance.name] = max(norm - low, high - norm)
                    if is_open:
                        self._saturated(payload, substance.name, value, unit, item)
                elif deltas is not None:
                    deltas[substance.name] = self.uncertainty.norm_delta(substance.name, value, unit,
                                                                         payload.get("sensor_model"))
            except UnitError as e:
                logger.warning(f"HFS gas: {gas} {value} {unit}: {e}")
        return out

    def _saturated(self, payload: Dict[str, Any], gas: str, value: float, unit: str, item: Dict[str, Any]) -> None:
        self._open.add(gas)
        key = (payload.get("sensor_model") or payload.get("driver"), gas)
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(f"HFS gas: {gas} {value} {unit} is the full scale of {key[0] or 'the sensor'} (range "
                           f"{item.get('range')}): at least this much — the sensor cannot confirm a critical "
                           "concentration")

    def observe(self, event: Event, now: float) -> None:
        super().observe(event, now)
        per_location = self.readings.setdefault(event.location_id, {})
        deltas: Dict[str, float] = {}
        self._open: set = set()
        norms = self.normalise(event.payload, deltas)
        per_location[event.device_id] = (now, norms, deltas)
        self.open_bounds.setdefault(event.location_id, {})[event.device_id] = set(self._open)
        self.history.setdefault(event.location_id, []).append((now, self._max_norm(event.location_id, now)))

    def forget_device(self, device_id: str) -> None:
        for per_location in (*self.readings.values(), *self.open_bounds.values()):
            per_location.pop(device_id, None)

    def _fresh_with_deltas(self, location_id: str, now: float):
        """:return: ({substance: c_norm}, {substance: its δ}) over fresh sensors (max per substance), None if none"""
        fresh = [(norms, deltas) for ts, norms, deltas in (self.readings.get(location_id) or {}).values()
                 if now - ts <= self.stale_sec]
        if not fresh:
            return None
        merged: Dict[str, float] = {}
        errors: Dict[str, float] = {}
        for norms, deltas in fresh:
            for name, value in norms.items():
                if value >= merged.get(name, -1.0):
                    merged[name], errors[name] = value, deltas.get(name, 0.0)
        return merged, errors

    def _fresh(self, location_id: str, now: float) -> Optional[Dict[str, float]]:
        """:return: {substance: c_norm} over fresh sensors (max per substance), None without a fresh reading"""
        found = self._fresh_with_deltas(location_id, now)
        return None if found is None else found[0]

    def _max_norm(self, location_id: str, now: float) -> float:
        return max((self._fresh(location_id, now) or {}).values(), default=0.0)

    def features(self, location_id: str, now: float, available: bool) -> Optional[Dict[str, Any]]:
        found = self._fresh_with_deltas(location_id, now)
        if found is None:
            return None
        norms, errors = found
        samples = window(self.history.get(location_id, []), now, self.window_sec)
        top = max(norms, key=norms.get) if norms else None
        top_delta = errors.get(top, 0.0) if top else 0.0
        return {"max_norm": round(max(norms.values(), default=0.0), 4),
                "mixture_index": round(sum(norms.values()), 4),
                "dynamics": round(slope_per_minute(samples, self.min_samples, self.min_span_sec), 4),
                # A5: δ of the main substance; Σ δ for the sum (worst case); slope error over the window's times
                "_deltas": {"max_norm": top_delta, "mixture_index": sum(errors.values()),
                            "dynamics": self.uncertainty.dynamics_delta(top_delta, [t for t, _ in samples])}}

    def infer(self, inputs: Optional[Dict[str, Any]], location_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        result = super().infer(inputs, location_id)
        if result is not None and location_id is not None:
            norms = self._fresh(location_id, self.clock()) or {}
            result["substances"] = {name: round(value, 4) for name, value in sorted(norms.items())}
            now = self.clock()
            readings = self.readings.get(location_id) or {}
            fresh = {d for d, (ts, _, _) in readings.items() if now - ts <= self.stale_sec}
            open_ = sorted({g for d, gases in (self.open_bounds.get(location_id) or {}).items() if d in fresh
                            for g in gases})
            if open_:
                result["saturated"] = open_
        return result
