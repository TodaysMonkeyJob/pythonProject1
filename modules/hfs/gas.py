"""
Gas subsystem (A1 / TOD-41) → risk 1. Rule base ``config/fuzzy/gas.yaml``.

Inputs from ``gas_reading`` events, normalised to each gas's critical limit L (``hfs.gas.limits`` or the placeholder
``gas.thresholds``; A2 / TOD-42 brings NIOSH IDLH / EPA limits and PM2.5):

* ``gas_level`` — max over fresh readings of C / L;
* ``gas_dynamics`` — rise rate of the latest reading as a share of its limit per minute;
* ``mixture_index`` — Σ C_i / L_i over the sensors of the location (additive mixture rule). One sensor reports one
  gas class per reading (the TOD-19 classifier), so only its latest reading counts — a classifier flipping between
  two gases must not add them up; A2 brings per-gas channels.

No fresh reading in the location → no data (None).
"""
from typing import Any, Dict, Optional, Tuple

from core.events import Event
from modules.hfs.base import FuzzySubsystem

CLEAN = ("clean_air", "air", "none", None)
DEFAULT_LIMIT_PCT = 3.0          # unknown gas: the stricter placeholder limit


class GasSubsystem(FuzzySubsystem):
    name = "gas"
    device_kind = "gas"
    event_types = ("gas_reading",)

    def __init__(self, config: Dict[str, Any], path: Optional[str] = None, **kwargs):
        super().__init__(config, path, **kwargs)
        self.limits = {str(k): float(v) for k, v in
                       (self.cfg.get("limits") or (config.get("gas") or {}).get("thresholds") or {}).items()}
        # location → {device: (ts, C/L, rate / L per minute)} — the latest reading of each sensor
        self.readings: Dict[str, Dict[str, Tuple[float, float, float]]] = {}

    def limit(self, gas: str) -> float:
        return self.limits.get(gas, DEFAULT_LIMIT_PCT)

    def observe(self, event: Event, now: float) -> None:
        super().observe(event, now)
        p = event.payload
        gas = p.get("gas")
        per_location = self.readings.setdefault(event.location_id, {})
        if gas in CLEAN:
            per_location[event.device_id] = (now, 0.0, 0.0)
            return
        limit = self.limit(gas)
        level = max(0.0, float(p.get("concentration_pct") or 0.0)) / limit
        rate = p.get("rise_rate_pct_per_min")
        per_location[event.device_id] = (now, level, 0.0 if rate is None else float(rate) / limit)

    def forget_device(self, device_id: str) -> None:
        for per_location in self.readings.values():
            per_location.pop(device_id, None)

    def features(self, location_id: str, now: float, available: bool) -> Optional[Dict[str, Any]]:
        fresh = [(ts, level, rate) for ts, level, rate in (self.readings.get(location_id) or {}).values()
                 if now - ts <= self.stale_sec]
        if not fresh:
            return None
        latest = max(fresh)
        return {"gas_level": round(max(level for _, level, _ in fresh), 4),
                "gas_dynamics": round(latest[2], 4),
                "mixture_index": round(sum(level for _, level, _ in fresh), 4)}
