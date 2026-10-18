"""
Trusted evidence per location (F17 / TOD-28): the input of the risk corridor and of the safety confirmation.

Only events whose trust degree (F16, ``payload.trust``) is ≥ θ count — θ = ``envelope.theta`` (0.6), overridable per
location. Events without a trust degree (trust scoring off) are treated as untrusted, so the corridor never rests on
unscored evidence. Low-trust claims that would lower risk (``claim_type`` de-escalation / instruction to the system)
are remembered separately: they are what the envelope ignores, and they appear in ``envelope_reason``.
"""
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.events import Event

LOWERING_CLAIMS = ("deescalation", "instruction_to_system")
TREND_RATE = {"falling": -1.0, "stable": 0.0, "rising": 1.0, "sharply_rising": 3.0}   # %/min stand-ins


def epoch(timestamp: Optional[str]) -> float:
    try:
        return datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return time.time()


@dataclass
class PersonTrack:
    person_id: Optional[str] = None
    known: bool = False
    trust: float = 0.0
    last: float = 0.0


@dataclass
class LocationTrust:
    gas: Optional[Dict[str, Any]] = None                       # latest trusted reading
    motion_level: float = 0.0
    motion_at: float = 0.0
    tracks: Dict[Tuple[str, Any], PersonTrack] = field(default_factory=dict)
    actions: Dict[Tuple[str, Any], Tuple[str, float]] = field(default_factory=dict)    # track → (action, time)
    ignored: List[Dict[str, Any]] = field(default_factory=list)                        # low-trust lowering claims
    ack: Optional[Dict[str, Any]] = None                       # {"by", "at"}
    last_alarm: float = 0.0                                    # latest trusted critical alert


class TrustedState:
    def __init__(self, theta: float = 0.6, theta_by_location: Optional[Dict[str, float]] = None,
                 gas_fresh_sec: float = 120.0, action_fresh_sec: float = 60.0, motion_fresh_sec: float = 30.0,
                 claim_memory_sec: float = 600.0, presence_stale_sec: float = 600.0):
        self.theta = theta
        self.theta_by_location = theta_by_location or {}
        self.gas_fresh_sec, self.action_fresh_sec = gas_fresh_sec, action_fresh_sec
        self.motion_fresh_sec, self.claim_memory_sec = motion_fresh_sec, claim_memory_sec
        self.presence_stale_sec = presence_stale_sec
        self.locations: Dict[str, LocationTrust] = {}

    def at(self, location_id: Optional[str]) -> LocationTrust:
        return self.locations.setdefault(location_id or "unassigned", LocationTrust())

    def theta_for(self, location_id: Optional[str]) -> float:
        return float(self.theta_by_location.get(location_id or "", self.theta))

    # --- feeding ------------------------------------------------------------------------------------------------

    def observe(self, event: Event) -> None:
        p, now = event.payload, epoch(event.timestamp)
        loc = self.at(event.location_id)
        if event.type == "notification_acked":
            loc.ack = {"by": p.get("by") or "operator", "at": now}
            return
        trust = p.get("trust")
        trusted = trust is not None and float(trust) >= self.theta_for(event.location_id)
        if event.type in ("speech", "scene_text"):
            if not trusted and p.get("claim_type") in LOWERING_CLAIMS:
                loc.ignored.append({"at": now, "provenance": p.get("provenance") or event.type,
                                    "claim_type": p.get("claim_type"), "trust": trust,
                                    "text": str(p.get("text", ""))[:80]})
            loc.ignored = [c for c in loc.ignored if now - c["at"] <= self.claim_memory_sec][-10:]
            return
        if not trusted:
            return
        key = (event.device_id, p.get("track_id"))
        if event.type == "gas_reading":
            loc.gas = {"gas": p.get("gas"), "concentration": p.get("concentration_pct"),
                       "rate": p.get("rise_rate_pct_per_min"), "trend": p.get("trend"), "at": now}
        elif event.type == "motion":
            area = float(p.get("area_ratio") or 0.0)
            loc.motion_level = 0.0 if p.get("state") == "end" else min(1.0, max(0.2, area * 4))
            loc.motion_at = now
        elif event.type == "object_detected" and p.get("class") == "person":
            if p.get("state") == "disappeared":
                loc.tracks.pop(key, None)
                loc.actions.pop(key, None)
            else:
                loc.tracks.setdefault(key, PersonTrack()).last = now
        elif event.type == "person_identified" and p.get("track_id") is not None:
            track = loc.tracks.setdefault(key, PersonTrack())
            track.person_id, track.known = p.get("person_id"), bool(p.get("known"))
            track.trust, track.last = float(trust), now
        elif event.type == "action" and p.get("track_id") is not None:
            loc.actions[key] = (str(p.get("action")), now)
        elif event.type == "alert" and event.severity == "critical":
            loc.last_alarm = now

    # --- features -----------------------------------------------------------------------------------------------

    def features(self, location_id: Optional[str], now: float, action_map: Dict[str, float],
                 local_hour: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        :return: (risk FIS inputs, extra facts for the confirmation) from trusted evidence only. Gas is omitted when
                 there is no fresh trusted reading (the risk rules then use their default; the confirmation does not).
        """
        loc = self.at(location_id)
        for key in [k for k, t in loc.tracks.items() if now - t.last > self.presence_stale_sec]:
            del loc.tracks[key]
        people = list(loc.tracks.values())
        identified = [t for t in people if t.person_id]
        inputs: Dict[str, Any] = {"people_count": float(len(people)), "time_of_day_risk": local_hour,
                                  "motion_level": loc.motion_level if now - loc.motion_at <= self.motion_fresh_sec
                                  else 0.0}
        inputs["unknown_share"] = (sum(not t.known for t in identified) / len(identified)) if identified else 0.0
        actions = [a for key, (a, t) in loc.actions.items() if now - t <= self.action_fresh_sec and key in loc.tracks]
        inputs["action_risk"] = max((float(action_map.get(a, 0.0)) for a in actions), default=0.0)
        gas = loc.gas if loc.gas and now - loc.gas["at"] <= self.gas_fresh_sec else None
        if gas and gas["concentration"] is not None:
            inputs["gas_concentration"] = float(gas["concentration"])
            if gas["gas"]:
                inputs["gas"] = gas["gas"]
            if gas["rate"] is not None:
                inputs["gas_trend"] = float(gas["rate"])
            elif gas["trend"] in TREND_RATE:            # no rate yet (short history): the label is enough
                inputs["gas_trend"] = TREND_RATE[gas["trend"]]
        known = [t.trust for t in identified if t.known]
        facts = {"known_present": max(known, default=0.0), "has_gas": gas is not None,
                 "ack": loc.ack, "last_alarm": loc.last_alarm,
                 "ignored": [c for c in loc.ignored if now - c["at"] <= self.claim_memory_sec]}
        return inputs, facts
