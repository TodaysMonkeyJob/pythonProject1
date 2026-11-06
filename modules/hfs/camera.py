"""
Camera subsystem (A1 / TOD-41) → risk 2. Rule base ``config/fuzzy/camera.yaml``.

* ``unknown_share`` — persons in view that are not identified: ``unknown_N`` at once, a track still without a
  decision (``unidentified`` / ``not_visible`` / nothing yet) after ``unidentified_after_sec``; 0 with nobody in view;
* ``motion_intensity`` — share of the last ``motion_window_sec`` with motion (the busiest camera of the location);
* ``off_hours`` — hours from now to the location's activity hours (rule ``activity_hours: ["08:00", "19:00"]`` of
  the location, else ``hfs.camera.activity_hours``); 0 inside them, at most 12.

Data available = an online camera in the location, or camera events from it in the last ``stale_sec``. A3 / TOD-43
refines the inputs (adaptive motion, optional actions mode).
"""
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.events import Event
from modules.hfs.base import FuzzySubsystem

NOT_DECIDED = ("unidentified", "not_visible", None, "")
DEFAULT_HOURS = ("08:00", "19:00")


def hours_outside(now_local: datetime, start: str, end: str) -> float:
    """:return: hours from ``now_local`` to the nearest edge of the [start, end) window (0 inside), wraps midnight"""
    def minutes(text: str) -> int:
        hours, _, mins = str(text).partition(":")
        return int(hours) * 60 + int(mins or 0)

    a, b = minutes(start) % 1440, minutes(end) % 1440
    t = now_local.hour * 60 + now_local.minute + now_local.second / 60
    inside = a <= t < b if a <= b else (t >= a or t < b)
    if inside or a == b:
        return 0.0
    after_end, before_start = (t - b) % 1440, (a - t) % 1440
    return round(min(12.0, min(after_end, before_start) / 60), 3)


class CameraSubsystem(FuzzySubsystem):
    name = "camera"
    device_kind = "camera"
    event_types = ("motion", "object_detected", "person_identified")

    def __init__(self, config: Dict[str, Any], path: Optional[str] = None,
                 rules: Callable[[str], Dict[str, Any]] = lambda location_id: {},
                 local_now: Callable[[], datetime] = datetime.now, **kwargs):
        """
        :param rules: location_id → location rules (``activity_hours``)
        """
        super().__init__(config, path, **kwargs)
        self.rules, self.local_now = rules, local_now
        self.hours = tuple(self.cfg.get("activity_hours") or DEFAULT_HOURS)
        self.motion_window = float(self.cfg.get("motion_window_sec", 30))
        self.unidentified_after = float(self.cfg.get("unidentified_after_sec", 10))
        # location → {(device, track): {"since", "status": known / unknown / pending}}
        self.people: Dict[str, Dict[Tuple[str, Any], Dict[str, Any]]] = {}
        # location → {device: [(start, end or None)]} motion intervals
        self.motion: Dict[str, Dict[str, List[List[Optional[float]]]]] = {}

    def observe(self, event: Event, now: float) -> None:
        super().observe(event, now)
        p = event.payload
        if event.type == "motion":
            spans = self.motion.setdefault(event.location_id, {}).setdefault(event.device_id, [])
            if p.get("state") == "start" and not (spans and spans[-1][1] is None):
                spans.append([now, None])
            elif p.get("state") == "end" and spans and spans[-1][1] is None:
                spans[-1][1] = now
            return
        if p.get("track_id") is None:
            return
        people = self.people.setdefault(event.location_id, {})
        key = (event.device_id, p.get("track_id"))
        if event.type == "object_detected":
            if p.get("class") != "person":
                return
            if p.get("state") == "disappeared":
                people.pop(key, None)
            else:
                people.setdefault(key, {"since": now, "status": "pending"})
            return
        person = people.setdefault(key, {"since": now, "status": "pending"})
        identity = p.get("identity")
        if identity in NOT_DECIDED:
            if person["status"] != "known":
                person["status"] = "pending"
        elif str(identity).startswith("unknown_") or not p.get("person_id"):
            person["status"] = "unknown"
        else:
            person["status"] = "known"

    def forget_device(self, device_id: str) -> None:
        for people in self.people.values():
            for key in [k for k in people if k[0] == device_id]:
                del people[key]
        for cameras in self.motion.values():
            cameras.pop(device_id, None)

    def motion_share(self, location_id: str, now: float) -> float:
        start_of_window, best = now - self.motion_window, 0.0
        for spans in (self.motion.get(location_id) or {}).values():
            spans[:] = [s for s in spans if s[1] is None or s[1] > start_of_window]
            covered = sum((now if end is None else end) - max(start, start_of_window) for start, end in spans)
            best = max(best, covered / self.motion_window)
        return round(min(1.0, max(0.0, best)), 4)

    def unknown_share(self, location_id: str, now: float) -> float:
        people = list((self.people.get(location_id) or {}).values())
        if not people:
            return 0.0
        unidentified = sum(p["status"] == "unknown" or
                           (p["status"] == "pending" and now - p["since"] >= self.unidentified_after)
                           for p in people)
        return round(unidentified / len(people), 4)

    def off_hours(self, location_id: str) -> float:
        start, end = (self.rules(location_id) or {}).get("activity_hours") or self.hours
        return hours_outside(self.local_now(), start, end)

    def features(self, location_id: str, now: float, available: bool) -> Optional[Dict[str, Any]]:
        if not (available or self.recent(location_id, now)):
            return None
        return {"unknown_share": self.unknown_share(location_id, now),
                "motion_intensity": self.motion_share(location_id, now),
                "off_hours": self.off_hours(location_id)}
