import json
import os
import queue

from core.identity import IdentityRegistry
from modules.vision.identity import IdentityConfig, IdentityEngine, IdentityRecognizer
from tests.test_identity import (ANNA, FRAME, OLEH, STRANGER, FakeFaces, face_crop, face_row, make_registry, near,
                                 person, vec)


def make_engine(faces_dir=None, **cfg):
    faces = FakeFaces()
    config = IdentityConfig(retry_sec=2.0, verify_sec=5.0, switch_confirmations=2, **cfg)
    return IdentityEngine(faces, make_registry(faces_dir), config), faces


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
    # The track starts collecting again; the stranger becomes unknown_1 only after 3 good faces
    assert changes(events) == [("Oleh", "unidentified"), ("unidentified", "unknown_1")]
    assert events[-1][1] == "warning"


def test_one_stranger_replaces_another():
    engine, faces = make_engine()
    _, t = run(engine, faces, near(STRANGER, 1), 6)
    events, _ = run(engine, faces, vec(500), 10, t0=t)
    assert changes(events) == [("unknown_1", "unidentified"), ("unidentified", "unknown_2")]


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
    ivan = engine.registry.create_person("Ivan")                    # enrolled meanwhile (tools.enroll)
    engine.registry.add_embedding(ivan, "face", stranger)
    events, _ = run(engine, faces, near(stranger, 2), 4, t0=t)
    assert changes(events) == [("unknown_1", "Ivan")]


def test_same_stranger_keeps_collecting_samples(tmp_path):
    engine, faces = make_engine(faces_dir=str(tmp_path / "persons"))
    faces.aligned_crop = lambda image, face: face_crop()
    faces.rows = [face_row()]
    for t in range(20):                                              # different photos of one stranger
        faces.embedding = near(STRANGER, t, noise=1.2)
        engine.update(float(t), FRAME, [person(7)])
    (person_id,) = os.listdir(tmp_path / "persons")                  # no extra unknowns
    assert engine.registry.name(person_id) == "unknown_1"
    assert len(os.listdir(tmp_path / "persons" / person_id)) >= 2
    assert 3 <= engine.registry.counts[person_id]["face"] <= 10      # the gallery grows (non-redundant faces)


def test_identical_faces_are_not_stored_twice():
    engine, faces = make_engine()
    run(engine, faces, near(STRANGER, 1), 30)                        # the very same embedding every time
    (stranger,) = [p for p in engine.registry.list_persons() if not p["is_known"]]
    assert stranger["face_samples"] == 1


# --- registry changes from other processes (dashboard / CLI) --------------------------------------------------------

def shared_engine(tmp_path):
    path = str(tmp_path / "identity.db")
    registry = IdentityRegistry(path)
    for name, embedding in (("Oleh", OLEH), ("Anna", ANNA)):
        registry.add_embedding(registry.create_person(name), "face", embedding)
    faces = FakeFaces()
    engine = IdentityEngine(faces, registry, IdentityConfig(retry_sec=2.0, verify_sec=5.0))
    return engine, faces, IdentityRegistry(path)                     # the second one = the dashboard process


def test_rename_elsewhere_relabels_live_track(tmp_path):
    engine, faces, dashboard = shared_engine(tmp_path)
    events, _ = run(engine, faces, near(STRANGER, 1), 3)
    event = events[-1]
    person_id = event[0]["person_id"]
    assert event[0]["identity"] == "unknown_1" and dashboard.name(person_id) is None   # not polled yet
    dashboard.poll()
    dashboard.rename(person_id, "Petro")
    (renamed,) = engine.sync()
    payload, severity, _ = renamed
    assert (payload["track_id"], payload["identity"], payload["known"]) == (7, "Petro", True)
    assert payload["renamed_from"] == "unknown_1" and payload["person_id"] == person_id and severity == "info"
    assert engine.sync() == []                                       # nothing changed since
    events, _ = run(engine, faces, near(STRANGER, 2), 20, t0=5)
    assert events == []                                              # verified as Petro, no flip back


def test_merge_and_delete_elsewhere(tmp_path):
    engine, faces, dashboard = shared_engine(tmp_path)
    events, _ = run(engine, faces, near(STRANGER, 1), 3)
    stranger = events[-1][0]["person_id"]
    dashboard.poll()
    oleh = dashboard.find("Oleh")
    dashboard.merge(stranger, oleh)                                  # it was Oleh with a bad camera angle
    (merged,) = engine.sync()
    assert merged[0]["person_id"] == oleh and merged[0]["identity"] == "Oleh"
    dashboard.delete(oleh)
    assert engine.sync() == [] and engine.tracks[7].person_id is None   # identified again on the next frame
    events, _ = run(engine, faces, near(STRANGER, 3), 3, t0=10)
    assert events[-1][0]["identity"] == "unknown_2"                   # numbers are never reused


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
