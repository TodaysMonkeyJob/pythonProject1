"""A3 (TOD-43) activity schedules → activity_time."""
from datetime import datetime

import pytest

from core.locations import clean_rules
from core.schedule import Schedule, ScheduleError, parse_days, parse_windows

THU = datetime(2026, 11, 5)          # a Thursday


def at(day, hh, mm=0):
    return day.replace(hour=hh, minute=mm)


def test_activity_time_signed_hours_to_the_nearest_change():
    office = Schedule([["08:00", "19:00"]])
    assert office.activity_time(at(THU, 12)) == -4          # inside: 4 h after opening
    assert office.activity_time(at(THU, 18, 30)) == -0.5    # transition before closing
    assert office.activity_time(at(THU, 19, 30)) == 0.5     # transition after closing
    assert office.activity_time(at(THU, 2)) == 6            # night
    assert Schedule([["22:00", "06:00"]]).activity_time(at(THU, 23)) == -1   # a night shift wraps midnight


def test_weekdays_shifts_and_always():
    office = Schedule([["08:00", "19:00"]], [0, 1, 2, 3, 4])
    assert office.activity_time(datetime(2026, 11, 7, 12)) == 12          # Saturday noon: > 12 h to Monday
    shifts = Schedule([["06:00", "14:00"], ["14:00", "22:00"]])
    assert shifts.activity_time(at(THU, 14)) == -8          # back-to-back shifts: no change at 14:00
    assert Schedule([["00:00", "24:00"]]).activity_time(at(THU, 3)) == -12
    assert Schedule([["06:00", "14:00"], ["14:00", "22:00"], ["22:00", "06:00"]]).activity_time(at(THU, 3)) == -12


def test_parsing_and_location_rules():
    assert parse_windows("06:00-14:00, 14:00–22:00") == [["06:00", "14:00"], ["14:00", "22:00"]]
    assert parse_windows(["08:00", "19:00"]) == [["08:00", "19:00"]]
    assert parse_days("mon,tue,6") == [0, 1, 6] and parse_days(None) is None
    assert parse_windows("8-19") == [["8", "19"]]           # hours without minutes are fine
    for bad in ("8-19-20", [["08:00"]], "25:00-26:00", "noon-night"):
        with pytest.raises(ScheduleError):
            parse_windows(bad)
    rules = clean_rules({"activity_hours": "08:00-19:00", "activity_days": "mon,fri", "object_type": "office",
                         "actions_enabled": "true", "unrelated": 1})
    assert rules == {"activity_hours": [["08:00", "19:00"]], "activity_days": [0, 4], "object_type": "office",
                     "actions_enabled": True}
    presets = {"office": {"hours": [["09:00", "18:00"]], "days": [0, 1, 2, 3, 4]}}
    assert Schedule.from_rules({"object_type": "office"}, presets).windows == [["09:00", "18:00"]]
    assert Schedule.from_rules({"object_type": "office", "activity_hours": [["07:00", "15:00"]]}, presets).days is None
    assert Schedule.from_rules({}, presets).windows == [["08:00", "19:00"]]
