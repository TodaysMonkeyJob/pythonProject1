import json
import os
import queue

from modules.vision.identity import FaceDatabase, IdentityConfig, IdentityEngine, IdentityRecognizer
from modules.vision.identity_store import UnknownStore
from tests.test_identity import ANNA, FRAME, OLEH, STRANGER, FakeFaces, face_row, near, person, vec


def make_engine(store=None, **cfg):
    database = FaceDatabase()
    database.add("Oleh", [OLEH])
    database.add("Anna", [ANNA])
    faces = FakeFaces()
    config = IdentityConfig(retry_sec=2.0, verify_sec=5.0, switch_confirmations=2, **cfg)
    return IdentityEngine(faces, database, config, store), faces


def run(engine, faces, embedding, seconds, t0=0.0, track_id=7, step=1.0):
    """The same track for `seconds`, face showing `embedding` (None = face not visible)."""
    events, t = [], t0
    while t < t0 + seconds:
        faces.rows, faces.embedding = ([face_row()], embedding) if embedding is not None else ([], None)
        events += engine.update(t, FRAME, [person(track_id)])
        t += step
    return events, t


def changes(events):
    return [(p["previous"], p["identity"]) for p, _, _ in events]


def test_other_known_person_takes_the_same_track():
    engine, faces = make_engine()
    events, t = run(engine, faces, near(OLEH, 1), 10)
    assert changes(events) == [(None, "Oleh")]
    events, _ = run(engine, faces, near(ANNA, 2), 15, t0=t)        # Anna sits where Oleh was, same track id
    assert changes(events) == [("Oleh", "Anna")]
    assert events[0][0]["known"] is True


def test_stranger_replaces_known_person():
    engine, faces = make_engine()
    _, t = run(engine, faces, near(OLEH, 1), 10)
    events, _ = run(engine, faces, near(STRANGER, 2), 15, t0=t)
    assert changes(events) == [("Oleh", "unknown_1")]
    assert events[0][1] == "warning"


def test_one_stranger_replaces_another():
    engine, faces = make_engine()
    _, t = run(engine, faces, near(STRANGER, 1), 6)
    events, _ = run(engine, faces, vec(500), 10, t0=t)
    assert changes(events) == [("unknown_1", "unknown_2")]


def test_single_bad_view_does_not_switch():
    engine, faces = make_engine()
    _, t = run(engine, faces, near(OLEH, 1), 10)
    events, t = run(engine, faces, vec(321), 1, t0=t)               # one odd check
    events2, _ = run(engine, faces, near(OLEH, 3), 20, t0=t + 5)
    assert events == [] and events2 == []
    assert engine.tracks[7].identity == "Oleh"


def test_same_person_checks_are_cheap_and_spaced():
    engine, faces = make_engine()
    run(engine, faces, near(OLEH, 1), 60)
    assert faces.detect_calls == 12                                 # once + every verify_sec (5 s) for 60 s


def test_turning_away_keeps_the_label():
    engine, faces = make_engine()
    _, t = run(engine, faces, near(OLEH, 1), 6)
    events, _ = run(engine, faces, None, 20, t0=t)
    assert events == [] and engine.tracks[7].identity == "Oleh"


def test_unknown_track_upgraded_after_enrollment():
    engine, faces = make_engine()
    stranger = vec(77)
    _, t = run(engine, faces, near(stranger, 1), 4)
    engine.database.add("Ivan", [stranger])                         # enrolled meanwhile (rename / CLI)
    events, _ = run(engine, faces, near(stranger, 2), 4, t0=t)
    assert changes(events) == [("unknown_1", "Ivan")]


def test_same_stranger_keeps_collecting_samples(tmp_path):
    store = UnknownStore(str(tmp_path / "unknown"), str(tmp_path), max_images=10)
    engine, faces = make_engine(store=store)
    faces.aligned_crop = lambda image, face: __import__("numpy").zeros((112, 112, 3), "uint8")
    run(engine, faces, near(STRANGER, 1), 10)                        # 10 s, retries every 2 s
    folder = tmp_path / "unknown" / "unknown_1"
    assert len([f for f in os.listdir(folder) if f.startswith("face_")]) == 5
    assert sorted(os.listdir(tmp_path / "unknown")) == ["_counter.json", "unknown_1"]   # no extra unknowns


def test_identity_events_written_to_jsonl(tmp_path):
    engine, faces = make_engine()
    faces.rows, faces.embedding = [face_row()], near(OLEH, 1)
    config = {"identity": {}, "vision": {}, "logging": {"dir": str(tmp_path / "logs")}}
    module = IdentityRecognizer(queue.Queue(), queue.Queue(), config, engine=engine)
    module.process(1, 0.0, FRAME, {"persons": [person(3)]})
    module.teardown()
    (day,) = os.listdir(tmp_path / "logs")
    lines = (tmp_path / "logs" / day / "identity.jsonl").read_text().splitlines()
    assert [json.loads(line)["payload"]["identity"] for line in lines] == ["Oleh"]
