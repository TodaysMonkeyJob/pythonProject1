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

Sampling and inference periods (``hfs.gas`` in config.yaml, see docs/sensors.md): the graphene board is polled every
``gas.gas_poll_sec`` (1 s), electrochemical / particle modules report about every 1 s; the subsystem infers every
``update_sec`` (2 s); the dynamics window is 60 s — longer than the T90 response of electrochemical sensors (≤ 30 s CO,
≤ 150 s NH₃ for the DFRobot modules), so the slope follows the gas, not sensor noise, and holds ≥ 30 samples.
"""
from typing import Any, Dict, List, Optional, Tuple

from core.events import Event
from modules.hfs.base import FuzzySubsystem, window
from modules.hfs.gas_normalize import GasTable, UnitError, readings_of, slope_per_minute
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
        self.window_sec = float(self.cfg.get("window_sec", 60))
        self.min_samples = int(self.cfg.get("min_samples", 3))
        self.min_span_sec = float(self.cfg.get("min_span_sec", 10))
        # location → {device: (ts, {substance: c_norm})} — the latest state of each sensor
        self.readings: Dict[str, Dict[str, Tuple[float, Dict[str, float]]]] = {}
        self.history: Dict[str, List[Tuple[float, float]]] = {}       # location → [(ts, max_norm)]
        self._warned: set = set()

    def normalise(self, payload: Dict[str, Any]) -> Dict[str, float]:
        """:return: {substance: c_norm} of one reading (clean air → {})"""
        out: Dict[str, float] = {}
        for gas, value, unit in readings_of(payload):
            if str(gas or "").lower() in CLEAN:
                continue
            substance = self.table.get(gas)
            if substance is None:
                if gas not in self._warned:
                    self._warned.add(gas)
                    logger.warning(f"HFS gas: no critical concentration for {gas!r} in config/gases.yaml — ignored")
                continue
            try:
                out[substance.name] = max(out.get(substance.name, 0.0), substance.norm(value, unit))
            except UnitError as e:
                logger.warning(f"HFS gas: {gas} {value} {unit}: {e}")
        return out

    def observe(self, event: Event, now: float) -> None:
        super().observe(event, now)
        per_location = self.readings.setdefault(event.location_id, {})
        per_location[event.device_id] = (now, self.normalise(event.payload))
        self.history.setdefault(event.location_id, []).append((now, self._max_norm(event.location_id, now)))

    def forget_device(self, device_id: str) -> None:
        for per_location in self.readings.values():
            per_location.pop(device_id, None)

    def _fresh(self, location_id: str, now: float) -> Optional[Dict[str, float]]:
        """:return: {substance: c_norm} over fresh sensors (max per substance), None without a fresh reading"""
        fresh = [norms for ts, norms in (self.readings.get(location_id) or {}).values() if now - ts <= self.stale_sec]
        if not fresh:
            return None
        merged: Dict[str, float] = {}
        for norms in fresh:
            for name, value in norms.items():
                merged[name] = max(merged.get(name, 0.0), value)
        return merged

    def _max_norm(self, location_id: str, now: float) -> float:
        return max((self._fresh(location_id, now) or {}).values(), default=0.0)

    def features(self, location_id: str, now: float, available: bool) -> Optional[Dict[str, Any]]:
        norms = self._fresh(location_id, now)
        if norms is None:
            return None
        samples = window(self.history.get(location_id, []), now, self.window_sec)
        return {"max_norm": round(max(norms.values(), default=0.0), 4),
                "mixture_index": round(sum(norms.values()), 4),
                "dynamics": round(slope_per_minute(samples, self.min_samples, self.min_span_sec), 4)}

    def infer(self, inputs: Optional[Dict[str, Any]], location_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        result = super().infer(inputs, location_id)
        if result is not None and location_id is not None:
            norms = self._fresh(location_id, self.clock()) or {}
            result["substances"] = {name: round(value, 4) for name, value in sorted(norms.items())}
        return result
