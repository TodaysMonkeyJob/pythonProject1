"""
Activity schedules (A3 / TOD-43): when a location is normally in use, the same notion for every object type —
apartment, office, warehouse, factory. The HFS camera subsystem uses ``activity_time`` (position relative to the
schedule) instead of the clock.

A schedule is a list of daily windows ``[["08:00", "19:00"]]`` (shifts: several; a window may cross midnight,
``"24:00"`` = end of day) and optional weekdays (0 = Monday … 6 = Sunday or ``mon`` … ``sun``; a window that crosses
midnight belongs to the day it starts on). Location rules (``core/locations.py``): ``activity_hours``,
``activity_days``, or ``object_type`` picking a preset from ``hfs.camera.schedules``.

``activity_time`` = signed hours to the nearest change of state: negative inside the schedule (−3 = three hours before
it ends), positive outside (+3 = three hours after it ended / before it starts), clipped to ±12; 24/7 → −12, never →
+12. The camera terms: working ≤ −1 h, transition within ±1 h of an edge, off ≥ +1 h.
"""
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
WEEK_MIN = 7 * 1440
LIMIT_H = 12.0


class ScheduleError(ValueError):
    pass


def minutes(text: Any) -> int:
    """:return: minutes since midnight of "HH:MM" ("24:00" = 1440)"""
    try:
        hours, _, mins = str(text).strip().partition(":")
        value = int(hours) * 60 + int(mins or 0)
    except ValueError as e:
        raise ScheduleError(f"time must be HH:MM, got {text!r}") from e
    if not 0 <= value <= 1440:
        raise ScheduleError(f"time must be within 00:00–24:00, got {text!r}")
    return value


def parse_windows(value: Any) -> List[List[str]]:
    """
    :param value: ``["08:00", "19:00"]``, ``[["06:00", "14:00"], ["14:00", "22:00"]]``, ``"08:00-19:00"`` or
                  ``"06:00-14:00, 14:00-22:00"``
    :return: [[start, end]] (validated)
    """
    if isinstance(value, str):
        value = [part.replace("–", "-").split("-") for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, str) for v in value):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ScheduleError(f"activity hours must be [\"HH:MM\", \"HH:MM\"] or a list of them, got {value!r}")
    windows = []
    for window in value:
        if not (isinstance(window, (list, tuple)) and len(window) == 2):
            raise ScheduleError(f"a window needs a start and an end, got {window!r}")
        start, end = (str(w).strip() for w in window)
        minutes(start), minutes(end)
        windows.append([start, end])
    return windows


def parse_days(value: Any) -> Optional[List[int]]:
    """:return: weekday numbers 0..6, None = every day"""
    if value in (None, "", []):
        return None
    items = value.split(",") if isinstance(value, str) else value
    out = []
    for item in items:
        text = str(item).strip().lower()[:3]
        if text.isdigit() and 0 <= int(text) <= 6:
            out.append(int(text))
        elif text in DAYS:
            out.append(DAYS.index(text))
        else:
            raise ScheduleError(f"weekday must be 0..6 or mon..sun, got {item!r}")
    return sorted(set(out))


class Schedule:
    def __init__(self, windows: Sequence[Sequence[str]], days: Optional[Iterable[int]] = None):
        self.windows = parse_windows(list(windows))
        self.days = None if days is None else sorted(set(days))
        self.week = np.zeros(WEEK_MIN, dtype=bool)          # minute of the week (Monday 00:00 = 0) → in use
        for day in (self.days if self.days is not None else range(7)):
            for start, end in self.windows:
                a, b = minutes(start), minutes(end)
                length = b - a if b > a else b + 1440 - a            # crosses midnight; equal = the whole day
                index = (day * 1440 + a + np.arange(length)) % WEEK_MIN
                self.week[index] = True
        changes = np.flatnonzero(self.week != np.roll(self.week, 1))
        self._changes = changes                              # minutes where the state flips

    @classmethod
    def from_rules(cls, rules: Dict[str, Any], presets: Optional[Dict[str, Any]] = None,
                   default: Sequence[Sequence[str]] = (("08:00", "19:00"),)) -> "Schedule":
        """Location rules → schedule: ``activity_hours`` / ``activity_days``, else the ``object_type`` preset."""
        preset = (presets or {}).get(str(rules.get("object_type") or "")) or {}
        windows = rules.get("activity_hours") or preset.get("hours") or [list(w) for w in default]
        days = rules.get("activity_days") if rules.get("activity_hours") else preset.get("days")
        return cls(parse_windows(windows), parse_days(days))

    def activity_time(self, moment: datetime) -> float:
        """:return: signed hours to the nearest change (negative inside, positive outside), clipped to ±12"""
        t = moment.weekday() * 1440 + moment.hour * 60 + moment.minute
        inside = bool(self.week[t])
        if len(self._changes) == 0:
            return -LIMIT_H if inside else LIMIT_H
        frac = moment.second / 60
        forward = (self._changes - t) % WEEK_MIN - frac          # minutes until the next flip
        backward = (t - self._changes) % WEEK_MIN + frac         # minutes since the last flip
        distance = min(float(forward.min()), float(backward.min())) / 60
        distance = min(LIMIT_H, max(0.0, distance))
        return round(-distance if inside else distance, 3)


def windows_text(windows: Sequence[Sequence[str]]) -> str:
    return ", ".join(f"{a}–{b}" for a, b in windows)


def describe(schedule: Schedule) -> Tuple[str, str]:
    days = "every day" if schedule.days is None else ", ".join(DAYS[d] for d in schedule.days)
    return windows_text(schedule.windows), days
