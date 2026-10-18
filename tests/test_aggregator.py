import json
import os
import time
from datetime import datetime

import jsonschema
import pandas as pd
import pytest

from core.events import Event, make_event
from modules.aggregator.aggregator import SCHEMA_PATH, Aggregator, in_hours
from modules.aggregator.outbox import Outbox

SCHEMA = json.load(open(SCHEMA_PATH, encoding="utf-8"))


class Clock:
    def __init__(self, now=1_792_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def make(tmp_path, local_hour=12, **agg):
    clock = Clock()
    config = {"aggregator": {"export_dir": str(tmp_path / "exports"), **agg}}
    aggregator = Aggregator(config, clock=clock, local_now=lambda: datetime(2026, 10, 14, local_hour, 0))
    return aggregator, clock


def feed(aggregator, clock, events, dt=1.0):
    produced = []
    for event in events:
        snapshot = aggregator.on_event(event)
        produced += [snapshot] if snapshot else []
        clock.now += dt
    return produced


def scene_events():
    return [
        make_event("system", "mic_state", {"state": "active", "reason": "scene_active", "module": "audio"}),
        make_event("camera", "scene_active", {"active": True}),
        make_event("camera", "object_detected", {"track_id": 7, "class": "person", "bbox": [1, 2, 3, 4],
                                                 "state": "appeared"}, confidence=0.9),
        make_event("camera", "object_detected", {"track_id": 8, "class": "backpack", "bbox": [5, 6, 7, 8],
                                                 "state": "appeared"}),
        make_event("camera", "person_identified", {"track_id": 7, "identity": "unknown_2", "known": False,
                                                   "similarity": 0.12, "previous": None}, severity="warning"),
        make_event("camera", "action", {"track_id": 7, "action": "walking", "previous": None, "duration_sec": 0}),
        make_event("audio", "speech", {"text": "Привіт, є тут хтось?", "language": "uk", "duration_sec": 2.1}),
        make_event("audio", "sound", {"rms_db": -12.5, "state": "active"}),
        make_event("gas", "gas_reading", {"gas": "acetone", "gas_confidence": 0.93, "concentration_pct": 4.1,
                                          "trend": "rising", "sensor_id": "gas-1"}, confidence=0.93),
        make_event("camera", "action", {"track_id": 7, "action": "sitting", "previous": "walking",
                                        "duration_sec": 0}),
    ]


# --- snapshots --------------------------------------------------------------------------------------------------

def test_interval_snapshot_from_events(tmp_path):
    aggregator, clock = make(tmp_path, snapshot_interval_sec=60)
    assert feed(aggregator, clock, scene_events()) == []                  # nothing critical
    clock.now += 60
    snapshot = aggregator.tick()

    jsonschema.validate(snapshot, SCHEMA)
    size = len(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).encode())
    assert size <= 2048
    assert snapshot["trigger"] == "interval" and "reason" not in snapshot
    (person,) = snapshot["people"]
    assert person == {"track_id": 7, "identity": "unknown_2", "known": False, "actions": ["walking", "sitting"],
                      "present": True, "in_frame_sec": 68}
    assert snapshot["objects"] == ["person", "backpack"]
    assert snapshot["speech"] == [{"t": snapshot["speech"][0]["t"], "text": "Привіт, є тут хтось?", "language": "uk"}]
    assert snapshot["sounds"][0]["label"] == "loud_noise" and snapshot["sounds"][0]["db"] == -12.5
    assert snapshot["gas"] == {"gas": "acetone", "concentration_ppm": 41000, "trend": "rising", "confidence": 0.93}
    assert snapshot["system"]["mic_state"] == "active" and snapshot["system"]["scene_active"] is True
    assert "bbox" not in json.dumps(snapshot)                             # no raw coordinates


def test_snapshot_written_to_file_and_outbox(tmp_path):
    aggregator, clock = make(tmp_path, snapshot_interval_sec=10)
    feed(aggregator, clock, scene_events())
    snapshot = aggregator.tick()
    (file,) = os.listdir(tmp_path / "exports" / "snapshots")
    assert json.load(open(tmp_path / "exports" / "snapshots" / file, encoding="utf-8")) == snapshot
    (row,) = aggregator.outbox.pending()
    assert row["id"] == snapshot["snapshot_id"] and row["payload"] == snapshot and row["status"] == "pending"


def test_no_snapshot_for_an_idle_or_gas_only_window(tmp_path):
    aggregator, clock = make(tmp_path, snapshot_interval_sec=60)
    clock.now += 61
    assert aggregator.tick() is None                                      # nothing happened
    reading = make_event("gas", "gas_reading", {"gas": "clean_air", "concentration_pct": 0.0, "trend": "stable"})
    feed(aggregator, clock, [reading] * 61)                               # routine readings only
    assert aggregator.tick() is None
    assert aggregator.outbox.counts() == {}


def test_critical_gas_alert_snapshot_immediately(tmp_path):
    aggregator, clock = make(tmp_path)
    feed(aggregator, clock, scene_events())
    alert = make_event("gas", "alert", {"reason": "gas_threshold", "gas": "acetone", "concentration_pct": 6.2,
                                        "threshold_pct": 5.0, "sensor_id": "gas-1"}, severity="critical")
    snapshot = aggregator.on_event(alert)
    assert snapshot is not None and snapshot["trigger"] == "critical_event"
    assert snapshot["reason"] == "alert gas_threshold"
    assert snapshot["gas"]["alerts"] == ["gas_threshold acetone"]
    assert aggregator.outbox.counts() == {"pending": 1}
    jsonschema.validate(snapshot, SCHEMA)


@pytest.mark.parametrize("payload,severity", [
    ({"track_id": 3, "action": "possible_smoking", "gas_confirmed": False, "wrist_to_mouth_count": 3}, "warning"),
    ({"track_id": 3, "action": "lying", "previous": "standing"}, "warning"),
])
def test_immediate_actions(tmp_path, payload, severity):
    aggregator, clock = make(tmp_path)
    snapshot = aggregator.on_event(make_event("camera", "action", payload, severity=severity))
    assert snapshot["trigger"] == "critical_event" and snapshot["reason"] == f"{payload['action']} #3"
    assert snapshot["alerts"] and snapshot["alerts"][0].startswith(payload["action"])


def test_critical_events_within_cooldown_are_batched(tmp_path):
    aggregator, clock = make(tmp_path, immediate_cooldown_sec=3)
    smoking = make_event("camera", "action", {"track_id": 1, "action": "possible_smoking"}, severity="critical")
    lying = make_event("camera", "action", {"track_id": 2, "action": "lying"})
    assert aggregator.on_event(smoking) is not None
    clock.now += 1
    assert aggregator.on_event(lying) is None                             # within the cooldown
    clock.now += 2
    late = aggregator.tick()
    assert late["trigger"] == "critical_event" and late["reason"] == "lying #2"
    assert aggregator.outbox.counts() == {"pending": 2}


def test_unknown_person_at_night(tmp_path):
    unknown = make_event("camera", "person_identified", {"track_id": 4, "identity": "unknown_9", "known": False,
                                                         "similarity": 0.1, "previous": None}, severity="warning")
    night, _ = make(tmp_path / "n", local_hour=23, unknown_alert_hours=["22:00", "06:00"])
    day, _ = make(tmp_path / "d", local_hour=12, unknown_alert_hours=["22:00", "06:00"])
    assert night.on_event(unknown)["reason"] == "unknown person unknown_9 at 23:00"
    assert day.on_event(unknown) is None


def test_in_hours_wraps_midnight():
    hours = ["22:00", "06:00"]
    assert in_hours(datetime(2026, 1, 1, 23, 30), hours) and in_hours(datetime(2026, 1, 1, 5, 59), hours)
    assert not in_hours(datetime(2026, 1, 1, 12, 0), hours) and not in_hours(datetime(2026, 1, 1, 6, 0), hours)
    assert in_hours(datetime(2026, 1, 1, 10, 0), ["09:00", "18:00"]) and not in_hours(datetime(2026, 1, 1, 1), None)


def test_rename_notice_updates_people(tmp_path):
    aggregator, clock = make(tmp_path, snapshot_interval_sec=5)
    feed(aggregator, clock, scene_events()[:6])
    feed(aggregator, clock, [make_event("camera", "person_identified", {
        "track_id": None, "identity": "Oleh", "known": True, "similarity": None, "previous": "unknown_2",
        "renamed_from": "unknown_2"})])
    assert aggregator.tick()["people"][0]["identity"] == "Oleh"


def test_people_who_left_and_window_reset(tmp_path):
    aggregator, clock = make(tmp_path, snapshot_interval_sec=10)
    feed(aggregator, clock, scene_events())
    feed(aggregator, clock, [make_event("camera", "object_detected", {"track_id": 7, "class": "person",
                                                                      "bbox": [0, 0, 1, 1], "state": "disappeared"})])
    first = aggregator.tick()
    assert first["people"][0]["present"] is False
    feed(aggregator, clock, [make_event("camera", "motion", {"state": "start", "area_ratio": 0.1})])
    clock.now += 10
    second = aggregator.tick()
    assert second["people"] == [] and second["speech"] == [] and second["window"]["from"] == first["window"]["to"]


def test_large_window_is_truncated_to_budget(tmp_path):
    aggregator, clock = make(tmp_path, snapshot_interval_sec=5, max_snapshot_bytes=2048)
    events = []
    for i in range(20):
        events.append(make_event("camera", "object_detected", {"track_id": i, "class": "person", "bbox": [0, 0, 1, 1],
                                                               "state": "appeared"}))
        events.append(make_event("camera", "person_identified", {"track_id": i, "identity": f"unknown_{i}",
                                                                 "known": False, "similarity": 0.1}))
        for action in ("walking", "standing", "sitting", "walking", "lying"):
            events.append(make_event("camera", "action", {"track_id": i, "action": action}))
    events += [make_event("audio", "speech", {"text": "дуже довга фраза " * 30, "language": "uk"})] * 30
    events += [make_event("audio", "sound", {"rms_db": -10.0})] * 30
    aggregator.cfg.immediate_actions = ()                                 # keep the lying actions non-critical
    feed(aggregator, clock, events, dt=0.01)
    clock.now += 5
    snapshot = aggregator.tick()
    record = {k: v for k, v in snapshot.items() if k != "llm_view"}         # the budget is for the v1 record …
    assert len(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()) <= 2048
    (row,) = aggregator.outbox.pending()
    assert len(row["digest"].encode()) <= 2048                               # … and for what GPT gets (F18 view)
    assert snapshot["truncated"] is True and snapshot["llm_view"]["truncated"] is True
    jsonschema.validate(snapshot, SCHEMA)
    assert snapshot["people"]                                             # people kept, details dropped first


# --- CSV --------------------------------------------------------------------------------------------------------

def at(ts, event):
    return Event(**{**json.loads(event.to_json()), "timestamp": ts})


def test_csv_rows_header_rotation_and_pandas(tmp_path):
    aggregator, clock = make(tmp_path)
    feed(aggregator, clock, [at("2026-10-14T23:59:58.000Z", e) for e in scene_events()])
    feed(aggregator, clock, [at("2026-10-15T00:00:01.000Z", make_event("audio", "speech", {"text": "Добраніч"}))])
    aggregator.close()

    day1 = pd.read_csv(tmp_path / "exports" / "events_2026-10-14.csv", encoding="utf-8")
    day2 = pd.read_csv(tmp_path / "exports" / "events_2026-10-15.csv", encoding="utf-8")
    assert list(day1.columns) == ["timestamp", "source", "type", "severity", "track_id", "identity", "action",
                                  "text", "gas_ppm", "confidence", "person_id", "speaker_name", "speaker_score",
                                  "location_id", "device_id"]
    assert len(day1) == len(scene_events()) and len(day2) == 1
    assert day2.loc[0, "text"] == "Добраніч"
    speech = day1[day1["type"] == "speech"].iloc[0]
    assert speech["text"] == "Привіт, є тут хтось?"
    assert day1[day1["type"] == "gas_reading"].iloc[0]["gas_ppm"] == 41000
    assert day1[day1["type"] == "person_identified"].iloc[0]["identity"] == "unknown_2"
    assert day1[day1["type"] == "object_detected"].iloc[0]["text"] == "person appeared"

    again, clock = make(tmp_path)                                         # restart: appends without a 2nd header
    feed(again, clock, [at("2026-10-15T08:00:00.000Z", make_event("system", "mic_state",
                                                                  {"state": "standby", "reason": "idle"}))])
    again.close()
    day2 = pd.read_csv(tmp_path / "exports" / "events_2026-10-15.csv", encoding="utf-8")
    assert len(day2) == 2 and day2.loc[1, "text"] == "standby (idle)"


# --- outbox -----------------------------------------------------------------------------------------------------

def test_outbox_lifecycle_and_persistence(tmp_path):
    path = str(tmp_path / "outbox.db")
    box = Outbox(path)
    for i in range(3):
        box.add({"snapshot_id": f"id-{i}", "n": i}, "interval", created_at=100 + i)
    assert [r["id"] for r in box.pending()] == ["id-0", "id-1", "id-2"]
    box.mark_sent("id-0")
    box.mark_failed("id-1", "timeout")
    box.close()

    box = Outbox(path)                                                    # survives a restart
    assert box.counts() == {"sent": 1, "failed": 1, "pending": 1}
    assert [r["id"] for r in box.pending()] == ["id-1", "id-2"]          # failed rows are retried
    for _ in range(4):
        box.mark_failed("id-1", "timeout")
    assert [r["id"] for r in box.pending(max_attempts=5)] == ["id-2"]    # gave up after 5 attempts
    box.close()


# --- through the orchestrator -----------------------------------------------------------------------------------

def test_critical_event_reaches_outbox_within_two_seconds(tmp_path):
    from core.orchestrator import ModuleSpec, Orchestrator
    from tests.test_orchestrator import pump

    config = {"orchestrator": {"startup_grace_sec": 30}, "logging": {"dir": str(tmp_path / "logs")},
              "aggregator": {"export_dir": str(tmp_path / "exports")}}
    orch = Orchestrator(config, [ModuleSpec("aggregator", "modules.aggregator:run", ["*"]),
                                 ModuleSpec("steady", "tests.orchestrator_dummies:steady", [])])
    orch.start()
    try:
        assert pump(orch, 30, until=lambda: orch.modules["aggregator"].last_heartbeat)
        alert = make_event("gas", "alert", {"reason": "gas_threshold", "gas": "toluol", "concentration_pct": 3.4},
                           severity="critical")
        orch.emit(alert)
        sent_at = time.time()
        box = Outbox(str(tmp_path / "exports" / "outbox.db"))
        assert pump(orch, 10, until=lambda: box.counts().get("pending"))
        (row,) = box.pending()
        box.close()
    finally:
        orch.shutdown()
    assert row["trigger"] == "critical_event"
    assert row["created_at"] - sent_at <= 2.0


def test_csv_speaker_columns_and_old_header_is_not_mixed(tmp_path):
    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "events_2026-10-14.csv").write_text("timestamp,source,type\n2026-10-14T01:00:00.000Z,gas,x\n")
    aggregator, clock = make(tmp_path)
    speech = make_event("audio", "speech", {"text": "Close the door", "person_id": "p1", "speaker_name": "Oleh",
                                            "speaker_score": 0.62, "location_id": "warehouse", "device_id": "mic-mac"})
    feed(aggregator, clock, [at("2026-10-14T12:00:00.000Z", speech)])
    aggregator.close()
    row = pd.read_csv(exports / "events_2026-10-14_v2.csv", encoding="utf-8").iloc[0]
    assert (row["person_id"], row["speaker_name"], row["speaker_score"], row["location_id"], row["device_id"]) == \
        ("p1", "Oleh", 0.62, "warehouse", "mic-mac")
    assert (exports / "events_2026-10-14.csv").read_text().count("\n") == 2            # old file untouched
