"""
Experiment v2 variants (F29 / TOD-40) that are not configuration of the live system.

* ``StaticTrustRegistry`` — the old design's constant trust (TOD-27 ``trust.base``: physical 0.95, scene text 0.1,
  identity 0.8) instead of the measured, dynamic registry (TOD-34): no priors from held-out data, no confirmations /
  refutations, no device health, no uncertainty (u = 0 → the IT2 corridor collapses to its type-1 centre).
* ``with_gas_failure`` — the sensor-failure block: a scenario replayed with its gas sensor broken — the clean-air
  baseline drifts up by ``drift_pct`` (every reading reads higher) and the sensor's self-test reports
  ``baseline_drift`` with health ``health`` (0.3) from the start. With the registry the broken sensor's channels
  lose trust (τ·h, u_eff = 1 − h(1 − u)) and widen the corridor; with static trust it keeps holding the corridor at
  0.95.
"""
import json
import hashlib
import uuid
from typing import Any, Dict, List, Optional

from modules.trust.registry import TrustRegistry

STATIC_TRUST = {"scene_text": 0.1, "face_identity": 0.8, "person_registry": 0.8}   # everything else: physical
STATIC_PHYSICAL = 0.95


class StaticTrustRegistry(TrustRegistry):
    """The TOD-27 constants behind the registry interface (channels exist, their trust never moves)."""

    def static(self, channel_id: str) -> Optional[float]:
        ch = self.channels.get(channel_id)
        return None if ch is None else STATIC_TRUST.get(ch.channel_kind, STATIC_PHYSICAL)

    def tau(self, channel_id: str, now: float) -> Optional[float]:
        return self.static(channel_id)

    def uncertainty(self, channel_id: str, now: float) -> Optional[float]:
        return None if channel_id not in self.channels else 0.0

    def get(self, channel_id: str, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        info = super().get(channel_id, now)
        if info is not None:
            info.update(tau=self.static(channel_id), h=1.0, u=0.0, method="static")
        return info

    def update(self, channel_id: str, dr: float, ds: float, reason: str, now: float, source: str = "evidence") -> bool:
        return channel_id in self.channels                  # evidence is ignored: the trust is a constant

    def set_health(self, device_id: str, failing: Dict[str, Dict[str, Any]], now: float,
                   reporter: str = "device") -> List[str]:
        return []                                           # no self-diagnostics in the old design


def static_registry(registry_cfg: Dict[str, Any]) -> StaticTrustRegistry:
    return StaticTrustRegistry(registry_cfg, ":memory:")


class FailedScenario:
    """A scenario whose event log is transformed on read (same id / labels → paired statistics stay valid)."""

    def __init__(self, scenario, transform):
        self._scenario, self._transform = scenario, transform

    def __getattr__(self, name):
        return getattr(self._scenario, name)

    def events(self) -> List[str]:
        return self._transform(self._scenario.events())


def event_uuid(key: str) -> str:
    """A deterministic UUID4-formatted id (replays stay reproducible)"""
    return str(uuid.UUID(bytes=hashlib.sha256(key.encode()).digest()[:16], version=4))


def gas_failure_lines(lines: List[str], drift_pct: float = 1.5, health: float = 0.3) -> List[str]:
    """:return: the event log with drifted gas readings and a ``device_health`` report right after the gas device"""
    out: List[str] = []
    reported = set()
    for line in lines:
        event = json.loads(line)
        payload = event.get("payload") or {}
        if event.get("type") == "gas_reading" and payload.get("concentration_pct") is not None:
            payload["concentration_pct"] = round(float(payload["concentration_pct"]) + drift_pct, 3)
            event["payload"] = payload
            line = json.dumps(event, ensure_ascii=False)
        out.append(line if line.endswith("\n") else line + "\n")
        device = payload.get("device_id") or event.get("device_id")
        if event.get("type") == "device_online" and payload.get("kind") == "gas" and device not in reported:
            reported.add(device)
            report = {**event, "event_id": event_uuid(f"health:{event.get('event_id')}"),
                      "type": "device_health",
                      "severity": "warning",
                      "payload": {"device_id": device, "kind": "gas", "reporter": "device",
                                  "failing": {"baseline_drift": {"factor": health, "value": drift_pct,
                                                                 "detail": f"simulated drift +{drift_pct} %"}}}}
            out.append(json.dumps(report, ensure_ascii=False) + "\n")
    return out


def with_gas_failure(scenario, drift_pct: float = 1.5, health: float = 0.3) -> FailedScenario:
    return FailedScenario(scenario, lambda lines: gas_failure_lines(lines, drift_pct, health))


__all__ = ["StaticTrustRegistry", "static_registry", "with_gas_failure", "gas_failure_lines", "FailedScenario",
           "STATIC_TRUST", "STATIC_PHYSICAL"]
