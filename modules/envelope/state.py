"""
Evidence per location (F17 / TOD-28, F24 / TOD-35): the input of the risk corridor and of the safety confirmation.

F24: there is no trust cut for physical evidence any more — every gas reading, motion, object, face and action enters
with its uncertainty ``u`` (``payload.trust_u`` from the trust registry, 1 when unscored), which widens the footprint
of uncertainty of the FIS input it feeds (``features`` → ``facts["uncertainty"]``). Language and scene text never feed
the FIS: a claim that would lower risk (``claim_type`` de-escalation / instruction to the system) with τ below the
cost-fitted ``trust.claim_threshold`` is remembered as ignored — what the envelope ignores, shown in
``envelope_reason``.
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
    u_presence: float = 1.0                                    # objects channel behind "a person is here"
    u_identity: float = 1.0                                    # face_identity channel behind known / unknown


@dataclass
class LocationTrust:
    gas: Optional[Dict[str, Any]] = None                       # latest reading (+ its u)
    motion_level: float = 0.0
    motion_at: float = 0.0
    motion_u: float = 1.0
    tracks: Dict[Tuple[str, Any], PersonTrack] = field(default_factory=dict)
    actions: Dict[Tuple[str, Any], Tuple[str, float, float]] = field(default_factory=dict)  # → (action, time, u)
    ignored: List[Dict[str, Any]] = field(default_factory=list)                        # low-trust lowering claims
    ack: Optional[Dict[str, Any]] = None                       # {"by", "at"}
    last_alarm: float = 0.0                                    # latest critical alert


class TrustedState:
    def __init__(self, claim_threshold: float = 0.6, gas_fresh_sec: float = 120.0, action_fresh_sec: float = 60.0,
                 motion_fresh_sec: float = 30.0, claim_memory_sec: float = 600.0, presence_stale_sec: float = 3600.0):
        """:param claim_threshold: τ a lowering language / scene-text claim needs to count (trust.claim_threshold)"""
        self.claim_threshold = claim_threshold
        self.gas_fresh_sec, self.action_fresh_sec = gas_fresh_sec, action_fresh_sec
        self.motion_fresh_sec, self.claim_memory_sec = motion_fresh_sec, claim_memory_sec
        # Cameras report changes only (a person lying still sends nothing); tracks leave on "disappeared" — this only
        # guards against a missed one (module restart)
        self.presence_stale_sec = presence_stale_sec
        self.locations: Dict[str, LocationTrust] = {}

    def at(self, location_id: Optional[str]) -> LocationTrust:
        return self.locations.setdefault(location_id or "unassigned", LocationTrust())


    # --- feeding ------------------------------------------------------------------------------------------------

    def observe(self, event: Event) -> None:
        p, now = event.payload, epoch(event.timestamp)
        loc = self.at(event.location_id)
        if event.type == "notification_acked":
            loc.ack = {"by": p.get("by") or "operator", "at": now}
            return
        trust = p.get("trust")
        u = float(p["trust_u"]) if p.get("trust_u") is not None else 1.0       # unscored → nothing known
        if event.type in ("speech", "scene_text"):
            trusted = trust is not None and float(trust) >= self.claim_threshold
            if not trusted and p.get("claim_type") in LOWERING_CLAIMS:
                loc.ignored.append({"at": now, "provenance": p.get("provenance") or event.type,
                                    "claim_type": p.get("claim_type"), "trust": trust,
                                    "text": str(p.get("text", ""))[:80]})
            loc.ignored = [c for c in loc.ignored if now - c["at"] <= self.claim_memory_sec][-10:]
            return
        key = (event.device_id, p.get("track_id"))
        if event.type == "gas_reading":
            loc.gas = {"gas": p.get("gas"), "concentration": p.get("concentration_pct"),
                       "rate": p.get("rise_rate_pct_per_min"), "trend": p.get("trend"), "at": now, "u": u}
        elif event.type == "motion":
            area = float(p.get("area_ratio") or 0.0)
            loc.motion_level = 0.0 if p.get("state") == "end" else min(1.0, max(0.2, area * 4))
            loc.motion_at, loc.motion_u = now, u
        elif event.type == "object_detected" and p.get("class") == "person":
            if p.get("state") == "disappeared":
                loc.tracks.pop(key, None)
                loc.actions.pop(key, None)
            else:
                track = loc.tracks.setdefault(key, PersonTrack())
                track.last, track.u_presence = now, u
        elif event.type == "person_identified" and p.get("track_id") is not None:
            track = loc.tracks.setdefault(key, PersonTrack())
            track.person_id, track.known = p.get("person_id"), bool(p.get("known"))
            track.trust, track.last = float(trust or 0.0), now
            track.u_identity = u
            if p.get("person_id") is None:              # an unidentified person: the objects channel speaks
                track.u_presence = min(track.u_presence, u)
        elif event.type == "action" and p.get("track_id") is not None:
            loc.actions[key] = (str(p.get("action")), now, u)
        elif event.type == "alert" and event.severity == "critical":
            loc.last_alarm = now

    # --- features -----------------------------------------------------------------------------------------------

    def features(self, location_id: Optional[str], now: float, action_map: Dict[str, float],
                 local_hour: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        :return: (risk FIS inputs, extra facts for the confirmation). Gas is omitted when there is no fresh reading
                 (the risk rules then use their default; the confirmation does not). ``facts["uncertainty"]`` =
                 {FIS variable: u} — the most uncertain evidence behind each input (F24 FOU width).
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
        # The action module publishes only CHANGES: a person lying still sends nothing more, so the current action of
        # a track holds until it changes or the track disappears (no time expiry)
        actions = [(a, au) for key, (a, _, au) in loc.actions.items() if key in loc.tracks]
        inputs["action_risk"] = max((float(action_map.get(a, 0.0)) for a, _ in actions), default=0.0)
        risky = [au for a, au in actions if float(action_map.get(a, 0.0)) >= inputs["action_risk"] - 1e-9]
        uncertainty = {"people_count": max((t.u_presence for t in people), default=0.0),
                       "unknown_share": max((t.u_identity for t in identified), default=0.0),
                       "action_risk": max(risky, default=0.0),
                       "motion_level": loc.motion_u if now - loc.motion_at <= self.motion_fresh_sec else 0.0}
        gas = loc.gas if loc.gas and now - loc.gas["at"] <= self.gas_fresh_sec else None
        if gas and gas["concentration"] is not None:
            inputs["gas_concentration"] = float(gas["concentration"])
            if gas["gas"]:
                inputs["gas"] = gas["gas"]
            if gas["rate"] is not None:
                inputs["gas_trend"] = float(gas["rate"])
            elif gas["trend"] in TREND_RATE:            # no rate yet (short history): the label is enough
                inputs["gas_trend"] = TREND_RATE[gas["trend"]]
            uncertainty["gas_concentration"] = uncertainty["gas_trend"] = float(gas.get("u", 1.0))
        known = [t.trust for t in identified if t.known]
        facts = {"known_present": max(known, default=0.0), "has_gas": gas is not None,
                 "ack": loc.ack, "last_alarm": loc.last_alarm,
                 "ignored": [c for c in loc.ignored if now - c["at"] <= self.claim_memory_sec],
                 "uncertainty": uncertainty}
        return inputs, facts
