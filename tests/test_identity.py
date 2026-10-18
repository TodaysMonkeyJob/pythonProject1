import os
import queue
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.events import Event, make_event
from core.identity import IdentityRegistry
from modules.vision.identity import (NOT_VISIBLE, UNIDENTIFIED, FaceDatabase, IdentityConfig, IdentityEngine,
                                     IdentityRecognizer, largest_face, normalize)
from modules.vision.shared_frame import SharedFrameBuffer

REPO = Path(__file__).resolve().parent.parent
DETECTOR = REPO / "models" / "face_detection_yunet_2023mar.onnx"
RECOGNIZER = REPO / "models" / "face_recognition_sface_2021dec.onnx"
FRAME = np.zeros((480, 640, 3), np.uint8)
PERSON_BOX = [100, 50, 300, 450]


def vec(seed):
    return normalize(np.random.default_rng(seed).normal(size=128))


def near(base, seed, noise=0.3):
    """Embedding close to `base` (same person, different photo)."""
    # Offset seed: noise must not be generated like vec(seed), or it would point along the base vector
    return normalize(base + noise * np.random.default_rng(10_000 + seed).normal(size=128) / np.sqrt(128))


# Frontal 5-point layout (right eye, left eye, nose tip, right / left mouth corner) as fractions of the face box
FRONTAL = np.array([[0.31, 0.40], [0.69, 0.40], [0.50, 0.58], [0.35, 0.78], [0.65, 0.78]], np.float32)


def face_row(x=150, y=70, size=100, score=0.95, layout=FRONTAL):
    """YuNet row in frame coordinates (a good frontal face); the default sits at the head of PERSON_BOX."""
    row = np.zeros(15, np.float32)
    row[:4] = (x, y, size, size)
    row[4:14] = (np.array([x, y], np.float32) + layout * size).reshape(-1)
    row[-1] = score
    return row


def face_crop(seed=0):
    """A textured mid-grey 112×112 crop: sharp and well exposed for the quality gate."""
    return np.random.default_rng(seed).integers(60, 200, (112, 112, 3)).astype(np.uint8)


class FakeFaces:
    """Returns the configured faces / embedding; counts calls."""

    def __init__(self):
        self.rows = []
        self.embedding = None
        self.detect_calls = 0

    def detect(self, image):
        self.detect_calls += 1
        return np.array(self.rows, np.float32).reshape(-1, 15)

    def embed(self, image, face):
        return self.embedding


OLEH, ANNA, STRANGER = vec(1), vec(2), vec(3)


def make_registry(faces_dir=None):
    """In-memory registry with two known people."""
    registry = IdentityRegistry(":memory:", faces_dir=faces_dir)
    for name, embedding in (("Oleh", OLEH), ("Anna", ANNA)):
        registry.add_embedding(registry.create_person(name), "face", embedding)
    return registry


def make_engine(**cfg):
    faces = FakeFaces()
    return IdentityEngine(faces, make_registry(), IdentityConfig(**cfg)), faces


def person(track_id, bbox=PERSON_BOX):
    return {"track_id": track_id, "bbox": bbox}


def labels(events):
    return [(p["track_id"], p["identity"]) for p, _, _ in events]


def show(engine, track_id, t0, frames=3, step=0.5):
    """The same face for `frames` samples `step` s apart (an unknown needs 3 good faces over ≥ 1 s)."""
    events = []
    for i in range(frames):
        events += engine.update(t0 + i * step, FRAME, [person(track_id)])
    return events


# --- database ---------------------------------------------------------------------------------------------------

def test_database_match_and_merge(tmp_path):
    db = FaceDatabase(str(tmp_path / "faces.npz"))
    db.add("Oleh", [near(OLEH, 10), near(OLEH, 11)])
    db.add("Anna", [ANNA])
    name, similarity = db.match(near(OLEH, 12))
    assert name == "Oleh" and similarity > 0.8
    assert db.match(STRANGER)[1] < 0.45

    db.add("Oleh", [near(OLEH, 13)])                       # merged, not duplicated
    assert db.names == ["Oleh", "Anna"] and db.counts == [3, 1]
    db.save()
    again = FaceDatabase(str(tmp_path / "faces.npz"))
    assert again.names == ["Oleh", "Anna"] and np.allclose(again.embeddings, db.embeddings)
    assert again.remove("Anna") and not again.remove("Anna")
    assert FaceDatabase().match(OLEH) == (None, 0.0)
    with pytest.raises(ValueError):
        db.add("X", [])


def test_largest_face_respects_min_size():
    rows = np.array([face_row(size=30), face_row(size=80), face_row(size=50)])
    assert largest_face(rows, min_size=40)[2] == 80
    assert largest_face(rows[:1], min_size=40) is None


# --- engine -----------------------------------------------------------------------------------------------------

def test_known_person_identified_once_then_only_verified():
    engine, faces = make_engine()
    faces.rows, faces.embedding = [face_row()], near(OLEH, 5)
    events = []
    for t in range(20):
        events += engine.update(float(t), FRAME, [person(7)])
    assert labels(events) == [(7, "Oleh")]
    payload, severity, confidence = events[0]
    assert payload["known"] is True and payload["previous"] is None and payload["similarity"] > 0.8
    assert severity == "info" and confidence == payload["similarity"]
    assert faces.detect_calls == 4                          # once + a cheap re-check every verify_sec (5 s)


def test_face_not_visible_then_recognised():
    engine, faces = make_engine(retry_sec=2.0)
    events = engine.update(0.0, FRAME, [person(1)])           # facing away
    assert labels(events) == [(1, NOT_VISIBLE)]
    assert events[0][0]["known"] is False and events[0][2] is None
    assert engine.update(1.0, FRAME, [person(1)]) == []       # before retry_sec: no attempt
    assert faces.detect_calls == 1
    assert engine.update(2.0, FRAME, [person(1)]) == []       # retried, still not visible: no repeat event
    faces.rows, faces.embedding = [face_row()], near(ANNA, 6)
    events = engine.update(4.0, FRAME, [person(1)])
    assert labels(events) == [(1, "Anna")] and events[0][0]["previous"] == NOT_VISIBLE


def test_unknown_labels_are_stable_and_reused():
    engine, faces = make_engine(retry_sec=1.0)
    faces.rows, faces.embedding = [face_row()], near(STRANGER, 1)
    events = show(engine, 1, 0.0)
    assert labels(events) == [(1, UNIDENTIFIED), (1, "unknown_1")]          # only after 3 good faces over 1 s
    assert events[-1][1] == "warning" and events[-1][0]["known"] is False

    faces.embedding = near(STRANGER, 2)                       # same stranger, new track after leaving
    engine.drop(1)
    assert labels(engine.update(5.0, FRAME, [person(2)])) == [(2, "unknown_1")]

    faces.embedding = vec(99)                                 # another stranger
    assert labels(show(engine, 3, 6.0))[-1] == (3, "unknown_2")

    faces.embedding = vec(123)                                # bad view of track 3: keeps its label
    assert engine.update(8.0, FRAME, [person(3)]) == []
    assert engine.tracks[3].identity == "unknown_2"


def test_unknown_upgraded_to_known():
    engine, faces = make_engine(retry_sec=1.0)
    faces.rows, faces.embedding = [face_row()], near(OLEH, 1, noise=3.0)   # blurry: below threshold
    assert labels(show(engine, 4, 0.0))[-1][1].startswith("unknown_")
    faces.embedding = near(OLEH, 2)
    events = engine.update(7.0, FRAME, [person(4)])
    assert labels(events) == [(4, "Oleh")] and events[0][0]["previous"].startswith("unknown_")


def test_known_label_kept_when_face_turns_away():
    engine, faces = make_engine(retry_sec=1.0, face_threshold=0.99)        # never "known" → keeps retrying
    faces.rows, faces.embedding = [face_row()], near(STRANGER, 1)
    show(engine, 1, 0.0)
    faces.rows = []
    assert engine.update(2.0, FRAME, [person(1)]) == []                    # not_visible does not overwrite
    assert engine.tracks[1].identity == "unknown_1"


def test_strangers_are_not_matched_to_known_people():
    engine, faces = make_engine()
    faces.rows = [face_row()]
    for i in range(50):
        faces.embedding = vec(1000 + i)
        payload = engine.update(0.0, FRAME, [person(100 + i)])[0][0]
        assert payload["known"] is False


def test_face_filters():
    engine, faces = make_engine()
    faces.embedding = near(OLEH, 1)
    faces.rows = [face_row(y=400)]                           # centre low in the box: someone else's face
    assert labels(engine.update(0.0, FRAME, [person(1)])) == [(1, NOT_VISIBLE)]
    faces.rows = [face_row(size=20)]                         # too small
    assert labels(engine.update(0.0, FRAME, [person(2)])) == [(2, NOT_VISIBLE)]
    faces.rows = [face_row(score=0.5)]                       # low detector score
    assert labels(engine.update(0.0, FRAME, [person(3)])) == [(3, NOT_VISIBLE)]
    faces.rows = [face_row(size=60)]                         # detected, but below the quality gate's 80 px
    assert labels(engine.update(0.0, FRAME, [person(4)])) == [(4, NOT_VISIBLE)]
    assert engine.stats["rejected:small"] == 1
    faces.rows = [face_row(size=20), face_row(x=155, size=90)]
    assert labels(engine.update(0.0, FRAME, [person(5)])) == [(5, "Oleh")]


def test_tracks_expire():
    engine, faces = make_engine(track_lost_sec=10)
    engine.update(0.0, FRAME, [person(1)])
    engine.update(11.0, FRAME, [])
    assert engine.tracks == {}


def test_config_from_dict():
    cfg = IdentityConfig.from_dict({"face_threshold": 0.5, "debug_dir": "/tmp/x", "unrelated": 1})
    assert cfg.face_threshold == 0.5 and cfg.debug_dir == "/tmp/x"


# --- module -----------------------------------------------------------------------------------------------------

def _drain(q):
    events = []
    while True:
        try:
            events.append(Event.from_json(q.get_nowait()))
        except queue.Empty:
            return events


def test_recognizer_publishes_and_drops_tracks():
    engine, faces = make_engine()
    faces.rows, faces.embedding = [face_row()], near(OLEH, 1)
    module = IdentityRecognizer(queue.Queue(), queue.Queue(), {"identity": {}, "vision": {}}, engine=engine)
    module.process(1, 0.0, FRAME, {"persons": []})
    assert faces.detect_calls == 0
    module.process(2, 0.5, FRAME, {"persons": [person(9)]})
    (event,) = _drain(module.event_queue)
    assert (event.source, event.type, event.payload["identity"]) == ("camera", "person_identified", "Oleh")

    gone = make_event("camera", "object_detected",
                      {"track_id": 9, "class": "person", "bbox": PERSON_BOX, "state": "disappeared"})
    module.handle_control(gone.to_json())
    assert 9 not in engine.tracks


def test_recognizer_reads_shared_buffer():
    name = f"ti_{uuid.uuid4().hex[:10]}"
    buffer = SharedFrameBuffer.create(name, 640, 480)
    try:
        engine, faces = make_engine()
        faces.rows, faces.embedding = [face_row()], near(ANNA, 1)
        config = {"identity": {"identity_fps": 1000}, "vision": {"shared_buffer": name, "resolution": [640, 480]}}
        module = IdentityRecognizer(queue.Queue(), queue.Queue(), config, engine=engine)
        buffer.write(b"jpeg", FRAME, {"persons": [person(3)]})
        for _ in range(3):
            module._next_poll = 0
            module.step()
        (event,) = _drain(module.event_queue)
        assert event.payload["identity"] == "Anna"
        module.teardown()
    finally:
        buffer.unlink()
        buffer.close()


def test_debug_mode_saves_crops_only_when_enabled(tmp_path):
    engine, faces = make_engine(debug_dir=str(tmp_path / "debug"))
    faces.aligned_crop = lambda image, face: face_crop()
    faces.rows, faces.embedding = [face_row()], near(OLEH, 1)
    engine.update(0.0, FRAME, [person(1)])
    assert len(os.listdir(tmp_path / "debug")) == 1
    engine, faces = make_engine()                     # the crop is still needed by the quality gate…
    faces.aligned_crop = lambda image, face: face_crop()
    faces.rows, faces.embedding = [face_row()], near(OLEH, 1)
    engine.update(0.0, FRAME, [person(1)])
    assert os.listdir(tmp_path) == ["debug"]          # …but nothing new is written without debug_dir


# --- real models (skipped unless downloaded: python -m modules.vision.face_models) -------------------------------

@pytest.mark.skipif(not (DETECTOR.exists() and RECOGNIZER.exists()), reason="face models not downloaded")
def test_real_models_recognise_flipped_photo():
    from ultralytics.utils import ASSETS

    from modules.vision.identity import FaceEngine

    engine = FaceEngine(str(DETECTOR), str(RECOGNIZER))
    image = cv2.imread(str(ASSETS / "zidane.jpg"))
    faces = engine.detect(image)
    assert len(faces) >= 2
    db = FaceDatabase()
    for i, face in enumerate(faces[:2]):
        db.add(f"person_{i}", [engine.embed(image, face)])

    # Mirrored photo: each person must be recognised at the mirrored position (other detections are ignored)
    flipped = cv2.flip(image, 1)
    flipped_faces = engine.detect(flipped)
    for i, face in enumerate(faces[:2]):
        mirrored_x = image.shape[1] - (face[0] + face[2] / 2)
        match = min(flipped_faces, key=lambda f: abs(f[0] + f[2] / 2 - mirrored_x))
        name, similarity = db.match(engine.embed(flipped, match))
        assert name == f"person_{i}" and similarity >= IdentityConfig().face_threshold
