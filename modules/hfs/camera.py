"""
Camera subsystem (A1 / TOD-41, A3 / TOD-43) → risk 2. Rule base ``config/fuzzy/camera.yaml``.

Inputs (supervisor, slides 9–11: no people count, no action classes — both are scene-specific or too unreliable):

* ``unknown_share`` = N_unknown / (N_unknown + N_known) over the persons in view of the location's cameras (0 when
  nobody is there): 1 known + 1 unknown = 0.5. A person whose face gives no decision for ``unidentified_after_sec``
  counts as unknown (a hidden face is no reason to trust). Its interval (the future FOU, TOD-45): each identity is
  wrong with probability 1 − p (p = identification accuracy, ``identification_accuracy`` below), so the number of
  wrong identities is Binomial(n, 1 − p); with its ``interval_quantile`` quantile f the share lies in
  [(k − f) / n, (k + f) / n];
* ``motion_intensity`` — smoothed moving share of the frame (``motion_level`` events of the adaptive detector, the
  busiest camera of the location); without them the area of the motion in progress;
* ``activity_time`` — signed hours to the nearest change of the location's activity schedule (``core/schedule.py``):
  negative inside, positive outside, ±1 h = transition.

Data available = an online camera in the location, or camera events from it in the last ``stale_sec``.
"""
import json
import math
import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

from core.events import Event
from core.schedule import Schedule
from modules.hfs.base import FuzzySubsystem

NOT_DECIDED = ("unidentified", "not_visible", None, "")
DEFAULT_HOURS = (("08:00", "19:00"),)


def identification_accuracy(report_path: str, threshold: float, z: float = 1.96) -> Tuple[float, str]:
    """
    :param report_path: ``tools/eval_faces.py`` report (``reports/identity_eval_lfw.json``)
    :param threshold: the deployed ``identity.face_threshold``
    :return: (p, provenance) — the Wilson lower bound of the share of correct decisions (known people recognised +
             strangers rejected) at the threshold: 650 / 650 correct gives p ≈ 0.994, not an overconfident 1.0
    """
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    row = min(report["by_threshold"], key=lambda r: abs(r["threshold"] - threshold))
    known, strangers = int(report["known_test_images"]), int(report["stranger_images"])
    n = known + strangers
    correct = row["known_accuracy"] * known + (1 - row["false_accept_rate"]) * strangers
    phat = correct / n
    centre = (phat + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(centre - half, 4), (f"Wilson 95 % lower bound of {correct:.0f}/{n} correct decisions at threshold "
                                     f"{row['threshold']} ({os.path.basename(report_path)})")


def binomial_quantile(n: int, error: float, q: float) -> int:
    """:return: the smallest f with P(Binomial(n, error) ≤ f) ≥ q"""
    cumulative = 0.0
    for f in range(n + 1):
        cumulative += math.comb(n, f) * error ** f * (1 - error) ** (n - f)
        if cumulative >= q - 1e-12:
            return f
    return n


def share_interval(unknown: int, total: int, error: float, q: float = 0.95) -> Tuple[float, float]:
    """:return: [lo, hi] of the unknown share when each of ``total`` identities is wrong with probability ``error``"""
    if total <= 0:
        return 0.0, 0.0
    f = binomial_quantile(total, error, q)
    return round(max(0, unknown - f) / total, 4), round(min(total, unknown + f) / total, 4)


class CameraSubsystem(FuzzySubsystem):
    name = "camera"
    device_kind = "camera"
    event_types = ("motion", "motion_level", "object_detected", "person_identified")

    def __init__(self, config: Dict[str, Any], path: Optional[str] = None,
                 rules: Callable[[str], Dict[str, Any]] = lambda location_id: {},
                 local_now: Callable[[], datetime] = datetime.now, **kwargs):
        """
        :param rules: location_id → location rules (``activity_hours`` / ``activity_days`` / ``object_type``)
        """
        super().__init__(config, path, **kwargs)
        self.rules, self.local_now = rules, local_now
        self.default_hours = self.cfg.get("activity_hours") or [list(w) for w in DEFAULT_HOURS]
        self.presets = self.cfg.get("schedules") or {}
        self.unidentified_after = float(self.cfg.get("unidentified_after_sec", 10))
        self.level_stale = float(self.cfg.get("motion_level_stale_sec", 10))
        self.quantile = float(self.cfg.get("interval_quantile", 0.95))
        self.accuracy, self.accuracy_source = self._accuracy(config)
        # location → {(device, track): {"since", "status": known / unknown / pending}}
        self.people: Dict[str, Dict[Tuple[str, Any], Dict[str, Any]]] = {}
        # location → {device: (ts, smoothed intensity)} / {device: area ratio of the motion in progress}
        self.levels: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self.active: Dict[str, Dict[str, float]] = {}
        self._schedules: Dict[str, Tuple[str, Schedule]] = {}

    def _accuracy(self, config: Dict[str, Any]) -> Tuple[float, str]:
        if self.cfg.get("identification_accuracy") is not None:
            return float(self.cfg["identification_accuracy"]), "hfs.camera.identification_accuracy"
        path = self.cfg.get("identification_report", "reports/identity_eval_lfw.json")
        threshold = float((config.get("identity") or {}).get("face_threshold", 0.45))
        try:
            return identification_accuracy(path, threshold)
        except (OSError, KeyError, ValueError):
            return 0.95, "default 0.95 (no identification report)"

    # --- evidence ---------------------------------------------------------------------------------------------------

    def observe(self, event: Event, now: float) -> None:
        super().observe(event, now)
        p = event.payload
        if event.type == "motion_level":
            self.levels.setdefault(event.location_id, {})[event.device_id] = (now, float(p.get("intensity") or 0.0))
            return
        if event.type == "motion":
            active = self.active.setdefault(event.location_id, {})
            if p.get("state") == "start":
                active[event.device_id] = float(p.get("area_ratio") or 0.0)
            else:
                active.pop(event.device_id, None)
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
        for store in (*self.levels.values(), *self.active.values()):
            store.pop(device_id, None)

    # --- inputs -----------------------------------------------------------------------------------------------------

    def counts(self, location_id: str, now: float) -> Tuple[int, int]:
        """:return: (unknown, total) persons in view; undecided persons count as unknown after a grace period"""
        people = list((self.people.get(location_id) or {}).values())
        unknown = sum(p["status"] == "unknown" or
                      (p["status"] == "pending" and now - p["since"] >= self.unidentified_after) for p in people)
        decided = sum(p["status"] == "known" for p in people) + unknown
        return unknown, decided

    def unknown_share(self, location_id: str, now: float) -> float:
        unknown, total = self.counts(location_id, now)
        return round(unknown / total, 4) if total else 0.0

    def motion_intensity(self, location_id: str, now: float) -> float:
        levels = [value for ts, value in (self.levels.get(location_id) or {}).values() if now - ts <= self.level_stale]
        if levels:
            return round(min(1.0, max(levels)), 4)
        return round(min(1.0, max((self.active.get(location_id) or {}).values(), default=0.0)), 4)

    def schedule(self, location_id: str) -> Schedule:
        rules = self.rules(location_id) or {}
        key = json.dumps([rules.get(k) for k in ("activity_hours", "activity_days", "object_type")], sort_keys=True)
        cached = self._schedules.get(location_id)
        if cached is None or cached[0] != key:
            cached = (key, Schedule.from_rules(rules, self.presets, self.default_hours))
            self._schedules[location_id] = cached
        return cached[1]

    def activity_time(self, location_id: str) -> float:
        return self.schedule(location_id).activity_time(self.local_now())

    def features(self, location_id: str, now: float, available: bool) -> Optional[Dict[str, Any]]:
        if not (available or self.recent(location_id, now)):
            return None
        unknown, total = self.counts(location_id, now)
        lo, hi = share_interval(unknown, total, 1 - self.accuracy, self.quantile)
        return {"unknown_share": round(unknown / total, 4) if total else 0.0,
                "motion_intensity": self.motion_intensity(location_id, now),
                "activity_time": self.activity_time(location_id),
                "_intervals": {"unknown_share": [lo, hi]},
                "_persons": {"unknown": unknown, "total": total}}
