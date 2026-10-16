import json
import os
import queue
import time

import numpy as np
import pytest

from core.events import Event
from modules.vision.identity import FaceDatabase, IdentityConfig, IdentityEngine, IdentityRecognizer
from modules.vision.identity_store import AliasRegistry, UnknownStore, valid_name
from tests.test_identity import FRAME, OLEH, STRANGER, FakeFaces, face_row, near, person, vec


@pytest.fixture
def paths(tmp_path):
    faces_dir = tmp_path / "faces"
    return {"faces_dir": str(faces_dir), "unknown_dir": str(faces_dir / "unknown"),
            "db": str(faces_dir / "faces.npz"), "aliases": str(faces_dir / "aliases.json")}


class CropFaces(FakeFaces):
    def aligned_crop(self, image, face):
        return np.full((112, 112, 3), 128, np.uint8)


def make(paths, known=True, **cfg):
    database = FaceDatabase(paths["db"])
    if known and "Oleh" not in database.names:
        database.add("Oleh", [OLEH])
        database.save()
    store = UnknownStore(paths["unknown_dir"], paths["faces_dir"], max_images=3)
    faces = CropFaces()
    engine = IdentityEngine(faces, database, IdentityConfig(retry_sec=1.0, **cfg), store,
                            AliasRegistry(paths["aliases"]))
    return engine, faces, store


def see(engine, faces, embedding, track_id, t):
    faces.rows, faces.embedding = [face_row()], embedding
    return engine.update(t, FRAME, [person(track_id)])


# --- store ------------------------------------------------------------------------------------------------------

def test_unknown_is_saved_with_crops_and_embeddings(paths):
    engine, faces, _ = make(paths)
    for t in range(6):
        engine.drop(1)
        see(engine, faces, near(STRANGER, t), 1, float(t))
    folder = os.path.join(paths["unknown_dir"], "unknown_1")
    files = sorted(os.listdir(folder))
    assert files == ["embeddings.npy", "face_001.jpg", "face_002.jpg", "face_003.jpg", "meta.json"]  # max 3 crops
    meta = json.load(open(os.path.join(folder, "meta.json")))
    assert meta["label"] == "unknown_1" and meta["sightings"] == 6
    assert np.load(os.path.join(folder, "embeddings.npy")).shape == (6, 128)


def test_known_people_are_not_saved(paths):
    engine, faces, _ = make(paths)
    see(engine, faces, near(OLEH, 1), 1, 0.0)
    assert [n for n in os.listdir(paths["unknown_dir"]) if not n.startswith("_")] == []


def test_unknown_labels_survive_restart(paths):
    engine, faces, _ = make(paths)
    see(engine, faces, near(STRANGER, 1), 1, 0.0)
    see(engine, faces, vec(77), 2, 0.0)

    engine, faces, _ = make(paths)                                          # new process
    assert labels_of(see(engine, faces, near(STRANGER, 2), 5, 0.0)) == ["unknown_1"]
    assert labels_of(see(engine, faces, vec(55), 6, 0.0)) == ["unknown_3"]


def labels_of(events):
    return [p["identity"] for p, _, _ in events]


def test_label_numbers_are_never_reused(paths):
    store = UnknownStore(paths["unknown_dir"], paths["faces_dir"])
    store.save(store.next_label(), vec(1))
    store.promote("unknown_1", "Oleh")
    assert store.next_label() == "unknown_2"


def test_retention_purges_old_unknowns(paths):
    store = UnknownStore(paths["unknown_dir"], paths["faces_dir"], retention_days=30)
    store.save("unknown_1", vec(1), now=time.time() - 31 * 86400)
    store.save("unknown_2", vec(2))
    assert store.purge() == ["unknown_1"]
    assert sorted(store.load()[i][0] for i in range(len(store.load()))) == ["unknown_2"]


def test_valid_names():
    assert valid_name("Oleh") and valid_name("Oleh Osadchuk")
    assert not any(valid_name(n) for n in ("unknown_4", "unknown", ".tmp", "_x", "", "a/b"))


def test_alias_registry_follows_chains(paths):
    aliases = AliasRegistry(paths["aliases"])
    aliases.add("unknown_1", "Olег")
    aliases.add("Olег", "Oleh")
    assert AliasRegistry(paths["aliases"]).resolve("unknown_1") == "Oleh"
    assert aliases.resolve("unknown_9") == "unknown_9"


# --- renames ----------------------------------------------------------------------------------------------------

def test_renamed_folder_is_promoted_everywhere(paths):
    engine, faces, store = make(paths)
    anna = vec(42)
    see(engine, faces, near(anna, 1), 1, 0.0)
    see(engine, faces, near(anna, 2), 2, 0.0)                               # second live track, same person
    assert {t.identity for t in engine.tracks.values()} == {"unknown_1"}

    os.rename(os.path.join(paths["unknown_dir"], "unknown_1"), os.path.join(paths["unknown_dir"], "Anna"))
    events = engine.sync()

    notice, *per_track = events
    assert notice[0] == {"track_id": None, "identity": "Anna", "known": True, "similarity": None,
                         "previous": "unknown_1", "renamed_from": "unknown_1"}
    assert sorted(p["track_id"] for p, _, _ in per_track) == [1, 2]
    assert all(p["identity"] == "Anna" and p["known"] for p, _, _ in per_track)
    # database (in memory and on disk), image folder, aliases, unknown cache
    assert "Anna" in engine.database.names and "Anna" in FaceDatabase(paths["db"]).names
    folder = os.path.join(paths["faces_dir"], "Anna")
    assert sorted(os.listdir(folder)) == ["unknown_1_face_001.jpg", "unknown_1_face_002.jpg", "unknown_1_meta.json"]
    assert not os.path.exists(os.path.join(paths["unknown_dir"], "Anna"))
    assert json.load(open(paths["aliases"])) == {"unknown_1": "Anna"}
    assert all(u.label != "unknown_1" for u in engine.unknowns)
    assert engine.sync() == []                                              # applied once

    # recognised as known from now on, also on a new track
    events = see(engine, faces, near(anna, 3), 9, 5.0)
    assert labels_of(events) == ["Anna"] and events[0][0]["known"] is True


def test_folder_moved_into_faces_dir_is_promoted(paths):
    engine, faces, _ = make(paths)
    see(engine, faces, near(STRANGER, 1), 1, 0.0)
    os.rename(os.path.join(paths["unknown_dir"], "unknown_1"), os.path.join(paths["faces_dir"], "Ivan"))
    assert labels_of(engine.sync())[0] == "Ivan"
    assert sorted(os.listdir(os.path.join(paths["faces_dir"], "Ivan"))) == [
        "unknown_1_face_001.jpg", "unknown_1_meta.json"]
    assert engine.database.match(near(STRANGER, 2))[0] == "Ivan"


def test_rename_to_existing_person_merges(paths):
    engine, faces, _ = make(paths)
    see(engine, faces, near(OLEH, 1, noise=4.0), 1, 0.0)                    # bad photo of Oleh → unknown
    os.rename(os.path.join(paths["unknown_dir"], "unknown_1"), os.path.join(paths["unknown_dir"], "Oleh"))
    engine.sync()
    assert engine.database.names == ["Oleh"] and engine.database.counts == [2]


def test_recognizer_picks_up_cli_rename_and_reloads_database(paths):
    from tools import enroll_face

    engine, faces, _ = make(paths)
    see(engine, faces, near(STRANGER, 1), 3, 0.0)                           # live track 3 = unknown_1
    config = {"identity": {"sync_sec": 0}, "vision": {}}
    module = IdentityRecognizer(queue.Queue(), queue.Queue(), config, engine=engine)
    module.setup()

    cfg_path = os.path.join(paths["faces_dir"], "config.yaml")
    import yaml
    sections = {s: {} for s in ("vision", "actions", "audio", "gas", "aggregator", "llm", "dashboard")}
    sections["identity"] = {"db_path": paths["db"], "unknown_dir": paths["unknown_dir"],
                            "faces_dir": paths["faces_dir"], "aliases_path": paths["aliases"]}
    with open(cfg_path, "w") as f:
        yaml.safe_dump(sections, f)
    time.sleep(0.01)                                                        # distinct mtimes
    enroll_face.main(["--from-unknown", "unknown_1", "--name", "Petro", "--config", cfg_path])

    module.sync()
    events = []
    while not module.event_queue.empty():
        events.append(Event.from_json(module.event_queue.get()))
    assert "Petro" in engine.database.names                                 # faces.npz reloaded
    assert {(e.payload["track_id"], e.payload["identity"]) for e in events} == {(None, "Petro"), (3, "Petro")}
    assert engine.tracks[3].known


def test_save_unknowns_can_be_disabled(paths):
    database = FaceDatabase()
    faces = CropFaces()
    engine = IdentityEngine(faces, database, IdentityConfig())
    see(engine, faces, STRANGER, 1, 0.0)
    assert not os.path.exists(paths["unknown_dir"])
    assert engine.sync() == []
