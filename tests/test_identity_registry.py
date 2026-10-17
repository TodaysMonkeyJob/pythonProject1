import threading

import numpy as np
import pytest

from core.cooccurrence import PresenceTracker, VoiceFaceLinker
from core.events import make_event
from core.identity import IdentityRegistry, NameTaken, normalize


def face(seed, base=None, noise=0.3):
    v = np.random.default_rng(seed).normal(size=128)
    return normalize(v if base is None else base + noise * v / np.sqrt(128))


def voice(seed, base=None, noise=0.3):
    v = np.random.default_rng(1000 + seed).normal(size=192)
    return normalize(v if base is None else base + noise * v / np.sqrt(192))


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def bus():
    return []


@pytest.fixture
def registry(tmp_path, bus):
    reg = IdentityRegistry(str(tmp_path / "identity.db"), {"max_embeddings": 5}, publish=bus.append,
                           faces_dir=str(tmp_path / "persons"), clock=Clock())
    yield reg
    reg.close()


OLEH_FACE, OLEH_VOICE = face(1), voice(1)


# --- matching -----------------------------------------------------------------------------------------------------

def test_create_unknown_uses_one_global_counter_and_matches(registry, bus):
    a = registry.create_unknown("face", OLEH_FACE)
    b = registry.create_unknown("voice", OLEH_VOICE)
    c = registry.create_unknown("face", face(2))
    assert [registry.name(p) for p in (a, b, c)] == ["unknown_1", "unknown_2", "unknown_3"]
    assert registry.match_face(face(3, OLEH_FACE))[0] == a
    assert registry.match_face(face(99))[0] is None                                   # a stranger
    assert registry.match_voice(voice(5, OLEH_VOICE))[0] == b
    assert registry.match_voice(OLEH_VOICE, known=True) == (None, 0.0)
    assert [e.type for e in bus] == ["identity_created"] * 3
    assert bus[0].source == "system" and bus[0].payload["display_name"] == "unknown_1"


def test_add_embedding_keeps_k_best_and_recomputes_centroid(registry):
    pid = registry.create_unknown("face", face(10), quality=0.1)
    for i in range(8):
        registry.add_embedding(pid, "face", face(20 + i, OLEH_FACE), quality=0.5 + i / 100)
    assert registry.counts[pid]["face"] == 5                                         # K = 5
    qualities = [q for (q,) in registry.db.execute("SELECT quality FROM face_embeddings")]
    assert min(qualities) >= 0.53                                                    # the worst ones dropped
    assert float(registry.centroid(pid, "face") @ OLEH_FACE) > 0.95                 # the outlier is gone
    with pytest.raises(ValueError):
        registry.add_embedding(pid, "face", np.ones(192))                           # wrong size


# --- rename / merge / delete ----------------------------------------------------------------------------------------

def test_rename_emits_event_and_is_seen_by_other_processes(tmp_path, registry, bus):
    pid = registry.create_unknown("face", OLEH_FACE)
    reader = IdentityRegistry(str(tmp_path / "identity.db"))                    # e.g. the dashboard process
    assert reader.name(pid) == "unknown_1" and not reader.poll()
    event = registry.rename(pid, "Oleh")
    assert (event.type, event.payload["display_name"], event.payload["previous"]) == \
        ("identity_renamed", "Oleh", "unknown_1")
    assert bus[-1] is event and registry.is_known(pid)
    assert reader.handle_event(event) and reader.name(pid) == "Oleh"
    other = registry.create_unknown("face", face(2))
    with pytest.raises(NameTaken):
        registry.rename(other, "oleh")                                            # case-insensitive
    with pytest.raises(ValueError):
        registry.rename(other, "unknown_7")
    assert registry.find("Oleh") == pid and registry.find(pid[:8]) == pid and registry.find("nobody") is None
    reader.close()


def test_merge_moves_embeddings_and_keeps_old_ids_resolvable(registry, bus, tmp_path):
    face_id = registry.create_unknown("face", OLEH_FACE)
    voice_id = registry.create_unknown("voice", OLEH_VOICE)
    registry.save_face_crop(voice_id, np.zeros((8, 8, 3), np.uint8))
    registry.rename(voice_id, "Oleh")
    event = registry.merge(voice_id, face_id)
    assert event.type == "identity_merged" and event.payload["source_id"] == voice_id
    assert registry.counts[face_id] == {"face": 1, "voice": 1}                      # both embedding types
    assert registry.name(face_id) == "Oleh" and registry.is_known(face_id)          # the known name wins
    assert registry.resolve(voice_id) == face_id and registry.name(voice_id) == "Oleh"
    assert registry.match_voice(voice(3, OLEH_VOICE))[0] == face_id
    assert [p["person_id"] for p in registry.list_persons()] == [face_id]
    assert registry.thumbnail(face_id) is not None                                  # crops moved too
    with pytest.raises(ValueError):
        registry.merge(face_id, voice_id)                                           # same person now


def test_delete_removes_everything(registry, bus, tmp_path):
    pid = registry.create_unknown("face", OLEH_FACE)
    registry.save_face_crop(pid, np.zeros((8, 8, 3), np.uint8))
    event = registry.delete(pid)
    assert event.type == "identity_deleted" and event.payload["display_name"] == "unknown_1"
    assert registry.name(pid) is None and registry.match_face(OLEH_FACE)[0] is None
    assert registry.db.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0] == 0
    assert not (tmp_path / "persons" / pid).exists()


def test_every_mutation_is_audited(registry):
    a = registry.create_unknown("face", OLEH_FACE)
    b = registry.create_unknown("voice", OLEH_VOICE)
    registry.rename(a, "Oleh")
    registry.link(a, "voice", voice(2, OLEH_VOICE), duration_sec=2.0)
    registry.merge(b, a)
    registry.delete(a)
    actions = [h["action"] for h in reversed(registry.history())]
    assert actions == ["create", "create", "rename", "link_voice", "merge", "delete"]


def test_purge_unknowns_after_retention(registry):
    old = registry.create_unknown("face", face(5))
    known = registry.create_person("Anna")
    registry.clock.now += 31 * 86400
    fresh = registry.create_unknown("face", face(6))
    assert registry.purge_unknowns(30) == [old]
    assert registry.name(known) == "Anna" and registry.name(fresh) == "unknown_2"


def test_concurrent_creates_get_distinct_labels(tmp_path):
    path = str(tmp_path / "identity.db")
    registries = [IdentityRegistry(path) for _ in range(4)]

    def work(reg, seed):
        for i in range(5):
            reg.create_unknown("face", face(seed * 100 + i))
    threads = [threading.Thread(target=work, args=(r, i)) for i, r in enumerate(registries)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    names = [p["display_name"] for p in IdentityRegistry(path).list_persons()]
    assert len(names) == 20 and len(set(names)) == 20


# --- auto-link ----------------------------------------------------------------------------------------------------

def seen(tracker, track_id, person_id, at):
    tracker.on_event(make_event("camera", "object_detected", {"track_id": track_id, "class": "person",
                                                               "state": "appeared"}), now=at)
    tracker.on_event(make_event("camera", "person_identified", {"track_id": track_id, "person_id": person_id,
                                                                 "identity": "x"}), now=at)


def gone(tracker, track_id, at):
    tracker.on_event(make_event("camera", "object_detected", {"track_id": track_id, "class": "person",
                                                               "state": "disappeared"}), now=at)


def linker_setup(registry, min_co=3):
    tracker = PresenceTracker("home")
    return tracker, VoiceFaceLinker(registry, tracker, min_cooccurrences=min_co)


def test_alone_in_view_links_voice_after_three_segments(registry, bus):
    face_id = registry.create_unknown("face", OLEH_FACE)
    registry.rename(face_id, "Oleh")
    voice_id = registry.create_unknown("voice", OLEH_VOICE)
    tracker, linker = linker_setup(registry)
    seen(tracker, 7, face_id, at=100)
    assert linker.on_segment(voice_id, "home", 110, 113) is None
    assert linker.on_segment(voice_id, "home", 120, 122) is None
    event = linker.on_segment(voice_id, "home", 130, 134)
    assert event.type == "identity_linked" and event.payload["person_id"] == face_id
    assert registry.resolve(voice_id) == face_id and registry.counts[face_id]["voice"] == 1
    assert registry.match_voice(voice(9, OLEH_VOICE))[0] == face_id                 # off camera: still Oleh
    assert registry.history()[0]["action"] == "link_voice"


def test_two_people_visible_never_links(registry):
    a = registry.create_unknown("face", OLEH_FACE)
    b = registry.create_unknown("face", face(2))
    voice_id = registry.create_unknown("voice", OLEH_VOICE)
    tracker, linker = linker_setup(registry, min_co=1)
    seen(tracker, 1, a, at=100)
    seen(tracker, 2, b, at=100)
    for start in (110, 120, 130, 140):
        assert linker.on_segment(voice_id, "home", start, start + 3) is None
    assert registry.resolve(voice_id) == voice_id


def test_presence_must_cover_80_percent_and_face_must_be_identified(registry):
    a = registry.create_unknown("face", OLEH_FACE)
    voice_id = registry.create_unknown("voice", OLEH_VOICE)
    tracker, linker = linker_setup(registry, min_co=1)
    tracker.on_event(make_event("camera", "object_detected", {"track_id": 1, "class": "person",
                                                               "state": "appeared"}), now=100)
    assert linker.on_segment(voice_id, "home", 101, 104) is None                    # face not identified yet
    tracker.on_event(make_event("camera", "person_identified", {"track_id": 1, "person_id": a}), now=105)
    gone(tracker, 1, at=106)
    assert linker.on_segment(voice_id, "home", 104, 109) is None                    # alone only 20 %
    assert tracker.sole_person("home", 105, 106) == a
    assert tracker.sole_person("garage", 105, 106) is None                          # other location


def test_conflicting_faces_block_the_link(registry):
    a = registry.create_unknown("face", OLEH_FACE)
    b = registry.create_unknown("face", face(2))
    voice_id = registry.create_unknown("voice", OLEH_VOICE)
    tracker, linker = linker_setup(registry, min_co=2)
    seen(tracker, 1, a, at=100)
    linker.on_segment(voice_id, "home", 101, 103)
    gone(tracker, 1, at=104)
    seen(tracker, 2, b, at=105)
    for start in (106, 110, 114):
        assert linker.on_segment(voice_id, "home", start, start + 2) is None
    assert registry.resolve(voice_id) == voice_id


def test_known_voice_is_never_reassigned(registry):
    a = registry.create_unknown("face", OLEH_FACE)
    voice_id = registry.create_unknown("voice", OLEH_VOICE)
    registry.rename(voice_id, "Anna")
    tracker, linker = linker_setup(registry, min_co=1)
    seen(tracker, 1, a, at=100)
    assert linker.on_segment(voice_id, "home", 101, 104) is None
    assert registry.resolve(voice_id) == voice_id and registry.name(voice_id) == "Anna"
