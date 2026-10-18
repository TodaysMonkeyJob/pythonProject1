import json
import os
import uuid
import zipfile
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from core.events import make_event
from modules.aggregator.aggregator import Aggregator, AggregatorModule
from modules.aggregator.digest import snapshot_digest
from modules.aggregator.incidents import IncidentRecorder
from tests.test_aggregator import Clock, feed, make, scene_events

KYIV = timezone(timedelta(hours=3))


def critical_gas():
    return make_event("gas", "alert", {"reason": "gas_threshold", "gas": "acetone", "concentration_pct": 6.2},
                      severity="critical")


def jpeg(value):
    return cv2.imencode(".jpg", np.full((48, 64, 3), value, np.uint8))[1].tobytes()


# --- digest -----------------------------------------------------------------------------------------------------

def test_digest_is_compact_text_with_the_same_facts(tmp_path):
    aggregator, clock = make(tmp_path, snapshot_interval_sec=10)
    feed(aggregator, clock, scene_events())
    snapshot = aggregator.tick()
    digest = snapshot_digest(snapshot, local_now=datetime(2026, 10, 14, 15, 0, tzinfo=KYIV))
    lines = digest.splitlines()
    assert lines[0].endswith("UTC+3")
    assert "people: #7 unknown_2 walking>sitting 8s" in digest
    assert "objects: person backpack" in digest
    assert 'uk "Привіт, є тут хтось?"' in digest
    assert "gas: acetone 41000ppm rising conf0.93" in digest
    assert "system: mic active, scene active" in digest
    assert "{" not in digest and "snapshot_id" not in digest and "null" not in digest


def test_digest_uses_fewer_tokens_than_json(tmp_path):
    tiktoken = pytest.importorskip("tiktoken")
    encoding = tiktoken.get_encoding("o200k_base")
    aggregator, clock = make(tmp_path, snapshot_interval_sec=10)
    feed(aggregator, clock, scene_events())
    snapshot = aggregator.tick()
    json_tokens = len(encoding.encode(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))))
    digest_tokens = len(encoding.encode(snapshot_digest(snapshot)))
    assert digest_tokens < 0.6 * json_tokens


def test_digest_mentions_critical_reason_and_incident():
    snapshot = {"window": {"from": "2026-10-14T21:00:00Z", "to": "2026-10-14T21:00:10Z"},
                "trigger": "critical_event", "reason": "alert gas_threshold", "people": [], "objects": [],
                "speech": [], "sounds": [], "gas": {"gas": "acetone", "alerts": ["gas_threshold acetone"]},
                "alerts": [], "system": {}}
    digest = snapshot_digest(snapshot, "20261015-000000-ab12", local_now=datetime(2026, 10, 15, tzinfo=KYIV))
    assert digest.splitlines()[0] == ("00:00:00-00:00:10 UTC+3 CRITICAL alert gas_threshold"
                                      " | incident 20261015-000000-ab12")
    assert digest.splitlines()[1] == "gas: acetone alerts gas_threshold"


def test_digest_respects_max_chars():
    snapshot = {"window": {"from": "2026-10-14T21:00:00Z", "to": "2026-10-14T21:00:10Z"}, "trigger": "interval",
                "people": [], "objects": [], "sounds": [], "gas": {}, "alerts": [], "system": {},
                "speech": [{"t": "21:00:0%d" % i, "text": "x" * 150, "language": "uk"} for i in range(9)]}
    digest = snapshot_digest(snapshot, max_chars=500)
    assert len(digest) <= 500 and digest.count("speech:") < 9


def test_outbox_row_has_digest(tmp_path):
    aggregator, clock = make(tmp_path)
    feed(aggregator, clock, scene_events())
    snapshot = aggregator.on_event(critical_gas())
    (row,) = aggregator.outbox.pending()
    assert json.loads(row["digest"])["incident"] is True                    # F18: the v2 view says an archive exists
    assert row["payload"]["incident_id"] == snapshot["incident_id"]


# --- incident archives ------------------------------------------------------------------------------------------

def recorder(tmp_path, clock, **cfg):
    return IncidentRecorder({"dir": str(tmp_path / "incidents"), "context_sec": 60, "post_sec": 10,
                             "pre_frames_sec": 6, "frame_interval_sec": 2, **cfg}, clock,
                            local_now=lambda: datetime(2026, 10, 14, 15, 30, tzinfo=KYIV))


def test_incident_archive_contents(tmp_path):
    clock = Clock()
    rec = recorder(tmp_path, clock)
    for i in range(10):                                                   # 20 s before the trigger
        rec.add_event(make_event("camera", "motion", {"state": "start", "area_ratio": 0.1}))
        rec.add_frame(jpeg(i))
        clock.now += 2
    incident_id = rec.start("alert gas_threshold")
    rec.attach({"snapshot_id": str(uuid.uuid4())}, "digest text")
    for i in range(5):                                                    # 10 s after
        rec.add_event(make_event("camera", "action", {"track_id": 1, "action": "lying"}, severity="warning"))
        rec.add_frame(jpeg(100 + i))
        clock.now += 2
    assert rec.tick() is not None and rec.active is None

    with zipfile.ZipFile(tmp_path / "incidents" / f"{incident_id}.zip") as archive:
        names = archive.namelist()
        summary = archive.read("summary.txt").decode()
        meta = json.loads(archive.read("incident.json"))
        events = archive.read("events.jsonl").decode().splitlines()
    frames = sorted(n for n in names if n.startswith("frames/"))
    assert len([f for f in frames if f.endswith("_before.jpg")]) == 4     # pre_frames_sec 6 s → last 4 frames
    assert len([f for f in frames if f.endswith("_after.jpg")]) == 5
    assert len(events) == 10 + 5
    assert meta["reasons"] == ["alert gas_threshold"] and meta["frames"] == 9
    assert "digest text" in summary and "#1 lying [warning]" in summary
    assert incident_id.startswith("20261014-153000-")


def test_repeated_critical_events_extend_one_incident(tmp_path):
    clock = Clock()
    rec = recorder(tmp_path, clock)
    first = rec.start("alert gas_rising")
    clock.now += 8
    assert rec.start("possible_smoking #3") == first                     # still open → same archive
    clock.now += 8
    assert rec.tick() is None                                             # deadline moved to +10 s from 2nd
    clock.now += 3
    path = rec.tick()
    assert path and len(os.listdir(tmp_path / "incidents")) == 1
    with zipfile.ZipFile(path) as archive:
        assert json.loads(archive.read("incident.json"))["reasons"] == ["alert gas_rising", "possible_smoking #3"]


def test_old_archives_removed_above_size_cap(tmp_path):
    clock = Clock()
    rec = recorder(tmp_path, clock, max_total_mb=0.05)                    # ~50 KB
    big = os.urandom(30_000)
    ids = []
    for _ in range(3):
        ids.append(rec.start("alert"))
        rec.add_frame(big)
        clock.now += 11
        rec.tick()
        clock.now += 100
    left = sorted(os.listdir(tmp_path / "incidents"))
    assert left == [f"{ids[-1]}.zip"]


def test_open_incident_written_on_close(tmp_path):
    aggregator, clock = make(tmp_path)
    snapshot = aggregator.on_event(critical_gas())
    aggregator.close()
    assert os.path.exists(tmp_path / "exports" / "incidents" / f"{snapshot['incident_id']}.zip")


def test_interval_snapshots_open_no_incident(tmp_path):
    aggregator, clock = make(tmp_path, snapshot_interval_sec=10)
    feed(aggregator, clock, scene_events())
    snapshot = aggregator.tick()
    assert "incident_id" not in snapshot and aggregator.incidents.active is None


def test_incidents_can_be_disabled(tmp_path):
    aggregator = Aggregator({"aggregator": {"export_dir": str(tmp_path), "incidents": {"enabled": False}}},
                            clock=Clock())
    snapshot = aggregator.on_event(critical_gas())
    assert "incident_id" not in snapshot and not os.path.exists(tmp_path / "incidents")


def test_module_samples_frames_from_shared_buffer(tmp_path):
    from modules.vision.shared_frame import SharedFrameBuffer

    name = f"tagg_{uuid.uuid4().hex[:8]}"
    buffer = SharedFrameBuffer.create(name, 64, 48)
    try:
        config = {"aggregator": {"export_dir": str(tmp_path), "incidents": {"frame_interval_sec": 0}},
                  "vision": {"shared_buffer": name, "resolution": [64, 48]}}
        module = AggregatorModule(None, None, config)
        module.setup()
        buffer.write(jpeg(50), np.zeros((48, 64, 3), np.uint8), {})
        module._sample_frame()
        module._sample_frame()                                            # same frame id: not added twice
        assert len(module.aggregator.incidents.frames) == 1
        module.teardown()
    finally:
        buffer.unlink()
        buffer.close()
