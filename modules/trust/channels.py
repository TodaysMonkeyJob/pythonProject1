"""
Evidence channels and trust inputs (F16 / TOD-27).

``EvidenceState`` follows, from the routed event stream itself, what the trust scorer needs per location: who is
visible (person tracks + their identified person_id), which cameras exist / are healthy, the latest gas reading per
sensor, recent physical alerts and risky actions. ``channel_of`` maps an event to its evidence channel:

* ``physical``   — gas readings, motion, object detections, scene_active, actions (pose), sound, sensor alerts;
* ``identity``   — face identifications (``person_identified``);
* ``speech``     — Whisper transcripts with speaker attribution;
* ``scene_text`` — text seen in the frame (signs, screens; optional OCR / captioning, always weak evidence).

System events (heartbeats, decisions, device / location / identity bookkeeping) are not evidence and are not scored.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from core.events import Event

PHYSICAL_TYPES = ("gas_reading", "motion", "object_detected", "scene_active", "action", "sound")
PHYSICAL_SOURCES = ("gas", "camera", "microphone", "audio")
RISKY_ACTIONS = ("lying", "possible_smoking")


def epoch(timestamp: Optional[str]) -> Optional[float]:
    """``2026-10-18T12:30:05.123Z`` → seconds since the epoch (None if missing / malformed)"""
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def channel_of(event: Event) -> Optional[str]:
    if event.type == "speech":
        return "speech"
    if event.type == "scene_text":
        return "scene_text"
    if event.type == "person_identified":
        return "identity"
    if event.source in PHYSICAL_SOURCES and (event.type in PHYSICAL_TYPES or event.type == "alert"):
        return "physical"
    return None


@dataclass
class Track:
    person_id: Optional[str] = None
    last: float = 0.0


@dataclass
class LocationEvidence:
    tracks: Dict[Tuple[str, Any], Track] = field(default_factory=dict)     # (device, track id) → visible person
    last_person_seen: Optional[float] = None
    cameras: Set[str] = field(default_factory=set)
    gas: Dict[str, Dict[str, Any]] = field(default_factory=dict)            # sensor device → latest reading
    alarms: List[Tuple[float, str]] = field(default_factory=list)          # (time, what) recent physical alarms


class EvidenceState:
    def __init__(self, presence_stale_sec: float = 600.0, alarm_memory_sec: float = 300.0):
        self.presence_stale_sec = presence_stale_sec
        self.alarm_memory_sec = alarm_memory_sec
        self.locations: Dict[str, LocationEvidence] = {}
        self.unhealthy: Set[str] = set()                     # devices offline / in error
        self.camera_location: Dict[str, str] = {}

    def at(self, location_id: Optional[str]) -> LocationEvidence:
        return self.locations.setdefault(location_id or "unassigned", LocationEvidence())

    # --- feeding ------------------------------------------------------------------------------------------------

    def seed_devices(self, devices: List[Dict[str, Any]]) -> None:
        """Cameras known to the device registry at start (they may stay silent for a long time)."""
        for device in devices:
            self._device(device.get("device_id"), device.get("kind"), device.get("location_id"),
                         device.get("status", "online"))

    def _device(self, device_id: Optional[str], kind: Optional[str], location_id: Optional[str], status: str) -> None:
        if not device_id:
            return
        if status == "online":
            self.unhealthy.discard(device_id)
        else:
            self.unhealthy.add(device_id)
        if kind == "camera":
            old = self.camera_location.get(device_id)
            if old and old != location_id:
                self.at(old).cameras.discard(device_id)
            if location_id:
                self.camera_location[device_id] = location_id
                self.at(location_id).cameras.add(device_id)

    def observe(self, event: Event) -> None:
        now = epoch(event.timestamp) or 0.0
        p = event.payload
        if event.type in ("device_online", "device_updated", "device_offline"):
            status = "online" if event.type != "device_offline" else p.get("status", "offline")
            if event.type == "device_updated":
                status = p.get("status", "online")
            self._device(p.get("device_id"), p.get("kind"), p.get("location_id"), status)
            return
        loc = self.at(event.location_id)
        if event.source == "camera" and event.device_id and event.device_id != "unknown":
            loc.cameras.add(event.device_id)
            self.camera_location.setdefault(event.device_id, event.location_id)
        key = (event.device_id, p.get("track_id"))
        if event.type == "object_detected" and p.get("class") == "person":
            if p.get("state") == "disappeared":
                loc.tracks.pop(key, None)
            else:
                loc.tracks.setdefault(key, Track()).last = now
            loc.last_person_seen = now
        elif event.type == "person_identified" and p.get("track_id") is not None:
            track = loc.tracks.setdefault(key, Track())
            track.person_id, track.last = p.get("person_id"), now
            loc.last_person_seen = now
        elif event.type == "gas_reading" and event.device_id:
            loc.gas[event.device_id] = {"gas": p.get("gas"), "concentration": p.get("concentration_pct"),
                                        "trend": p.get("trend"), "rate": p.get("rise_rate_pct_per_min"), "t": now}
        elif event.type == "alert" and event.source in PHYSICAL_SOURCES and event.severity == "critical":
            loc.alarms.append((now, str(p.get("reason") or p.get("message") or "alert")))
        elif event.type == "action" and p.get("action") in RISKY_ACTIONS:
            loc.alarms.append((now, str(p.get("action"))))
        if loc.alarms and now - loc.alarms[0][0] > self.alarm_memory_sec:
            loc.alarms = [a for a in loc.alarms if now - a[0] <= self.alarm_memory_sec]

    # --- queries ------------------------------------------------------------------------------------------------

    def visible(self, location_id: Optional[str], now: float) -> List[Optional[str]]:
        """:return: person_ids (None = not identified) of the person tracks currently in view"""
        loc = self.at(location_id)
        stale = [k for k, t in loc.tracks.items() if now - t.last > self.presence_stale_sec]
        for key in stale:
            del loc.tracks[key]
        return [t.person_id for t in loc.tracks.values()]

    def has_working_camera(self, location_id: Optional[str], degraded: Optional[Set[str]] = None) -> bool:
        """:param degraded: cameras the trust registry reports unhealthy (covered lens, frozen, …)"""
        return any(c not in self.unhealthy and c not in (degraded or ()) for c in self.at(location_id).cameras)

    def person_seen_between(self, location_id: Optional[str], start: float, end: float, grace: float) -> bool:
        loc = self.at(location_id)
        return bool(self.visible(location_id, end)) or (loc.last_person_seen is not None
                                                        and loc.last_person_seen >= start - grace)

    def latest_gas(self, location_id: Optional[str]) -> Optional[Dict[str, Any]]:
        readings = list(self.at(location_id).gas.values())
        return max(readings, key=lambda r: r["t"]) if readings else None

    def recent_alarms(self, location_id: Optional[str], now: float) -> List[str]:
        return [what for t, what in self.at(location_id).alarms if now - t <= self.alarm_memory_sec]
