"""F14 (TOD-25): quality gate, track voting, multi-prototype gallery, dedup job, undo, review queue, metrics."""
import json

import numpy as np
import pytest

from core.gallery import Prototype, Quotas, admit
from core.identity import IdentityRegistry, normalize
from modules.vision.dedup import DedupConfig, DedupJob
from modules.vision.face_quality import QualityConfig, assess, pose
from modules.vision.identity import UNIDENTIFIED, IdentityConfig, IdentityEngine, TrackIdentity
from modules.vision.identity_metrics import MetricsReport
from tests.test_identity import FRAME, FRONTAL, FakeFaces, face_crop, face_row, person


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def base(seed):
    return normalize(np.random.default_rng(seed).normal(size=128))


def photo(seed, around, noise=0.8):
    """A different photo of the person `around` (noise 0.8 → ≈ 0.78 to the base, ≈ 0.6 to each other)."""
    v = np.random.default_rng(50_000 + seed).normal(size=128)
    return normalize(around + noise * v / np.sqrt(128))


GOOD = {"tier": "good"}
OLEH = base(1)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def registry(tmp_path, clock):
    reg = IdentityRegistry(str(tmp_path / "identity.db"), faces_dir=str(tmp_path / "persons"), clock=clock)
    yield reg
    reg.close()


def person_with(registry, name, around, n, seed0=0):
    pid = registry.create_person(name) if name else registry.create_unknown("face", photo(seed0, around), meta=GOOD)
    for i in range(n - (0 if name else 1)):
        registry.add_embedding(pid, "face", photo(seed0 + 1 + i, around), meta=GOOD)
    return pid


# --- Part 1: quality gate -------------------------------------------------------------------------------------------

def side_view(yaw_deg):
    layout = FRONTAL.copy()
    layout[2, 0] = 0.5 + np.tan(np.radians(yaw_deg)) / 2 * 0.38      # nose offset along the eye line
    return layout


def test_quality_gate_tiers():
    box = [100, 50, 300, 450]
    assert assess(FRAME.shape, face_row(), box, face_crop())["tier"] == "good"
    assert abs(pose(face_row())["yaw"]) < 1 and abs(pose(face_row(layout=side_view(30)))["yaw"] - 30) < 1
    slight = assess(FRAME.shape, face_row(layout=side_view(42)), box, face_crop())
    assert slight["tier"] == "usable" and slight["reasons"] == ["pose"]
    cases = {
        "small": face_row(size=60),                                       # < 80 px
        "truncated": face_row(x=-30),                                     # cut off at the frame edge
        "score": face_row(score=0.7),
        "pose": face_row(layout=side_view(60)),                           # profile
    }
    for reason, row in cases.items():
        q = assess(FRAME.shape, row, [0, 0, 640, 480], face_crop())
        assert q["tier"] == "reject" and reason in q["reasons"], (reason, q)
    outside = assess(FRAME.shape, face_row(x=260), box, face_crop())      # landmarks past the person box
    assert outside["tier"] == "reject" and "outside_person" in outside["reasons"]
    blurred = assess(FRAME.shape, face_row(), box, np.full((112, 112, 3), 128, np.uint8))
    assert blurred["tier"] == "reject" and "blur" in blurred["reasons"]
    dark = assess(FRAME.shape, face_row(), box, (face_crop() // 8))
    assert "exposure" in dark["reasons"]
    assert QualityConfig.from_dict({"min_face_size": 50}).min_face_size == 50


def test_rejected_faces_are_never_embedded(registry):
    faces = FakeFaces()
    faces.embed = lambda image, face: pytest.fail("a rejected face must not be embedded")
    engine = IdentityEngine(faces, registry, IdentityConfig())
    for t, row in enumerate([face_row(size=60), face_row(x=-30), face_row(layout=side_view(60))] * 3):
        faces.rows = [row]
        engine.update(t * 2.0, FRAME, [person(1, [0, 0, 640, 480])])
    assert registry.list_persons() == [] and engine.stats["faces_rejected"] >= 6


# --- Part 2: track-level decisions ----------------------------------------------------------------------------------

def engine_for(registry, **cfg):
    faces = FakeFaces()
    faces.aligned_crop = lambda image, face: face_crop()
    return IdentityEngine(faces, registry, IdentityConfig(**cfg)), faces


def test_late_fragments_inherit_identity_and_never_spawn_unknowns(registry):
    oleh = person_with(registry, "Oleh", OLEH, 3)
    engine, faces = engine_for(registry)
    faces.rows, faces.embedding = [face_row()], photo(90, OLEH)
    assert engine.update(0.0, FRAME, [person(7)])[0][0]["identity"] == "Oleh"
    rng = np.random.default_rng(3)
    for t in range(1, 120):                     # side views / hats: usable faces that match nobody
        faces.rows, faces.embedding = [face_row(layout=side_view(42))], normalize(rng.normal(size=128))
        assert engine.update(t * 0.5, FRAME, [person(7)]) == []
    assert [p["person_id"] for p in registry.list_persons()] == [oleh]
    assert engine.tracks[7].identity == "Oleh"


def test_unknown_needs_several_good_faces(registry):
    engine, faces = engine_for(registry)
    stranger = base(5)
    faces.rows = [face_row(layout=side_view(42))]                 # usable only: never creates anyone
    for t in range(20):
        faces.embedding = photo(t, stranger)
        events = engine.update(t * 0.5, FRAME, [person(1)])
        if t == 0:
            assert events[0][0]["identity"] == UNIDENTIFIED and events[0][0]["person_id"] is None
    assert registry.list_persons() == []
    faces.rows = [face_row()]
    labels = []
    for t in range(4):                                             # 3 good faces over ≥ 1 s
        faces.embedding = photo(100 + t, stranger)
        labels += [p["identity"] for p, _, _ in engine.update(20 + t * 0.5, FRAME, [person(1)])]
    assert labels == ["unknown_1"]
    (unknown,) = registry.list_persons()
    assert unknown["face_samples"] >= 3


def test_vote_assigns_on_majority(registry):
    person_with(registry, "Oleh", OLEH, 3)
    engine, faces = engine_for(registry, strong_threshold=0.99)   # no single strong match: voting decides
    faces.rows = [face_row()]
    faces.embedding = photo(1, OLEH, noise=1.1)                   # ≈ 0.55: matches Oleh, not strongly
    events = engine.update(0.0, FRAME, [person(2)])
    assert events[0][0]["identity"] == "Oleh"


# --- Part 3: gallery ------------------------------------------------------------------------------------------------

def test_admission_keeps_diversity_quotas_and_limit():
    gallery = []
    quotas = Quotas(min_frontal=2, max_per_yaw=2)
    vectors = [photo(i, OLEH, noise=2.0) for i in range(20)]
    for i, v in enumerate(vectors):
        bucket = "frontal" if i < 3 else "left"
        decision = admit(gallery, v, "good", bucket, limit=5, quotas=quotas)
        if decision.accepted:
            gallery = [p for p in gallery if p.id not in decision.evict] + [Prototype(i, v, bucket)]
    buckets = [p.yaw_bucket for p in gallery]
    assert len(gallery) <= 5 and buckets.count("left") <= 2 and buckets.count("frontal") >= 2
    assert not admit(gallery, gallery[0].vector, "good", "frontal").accepted          # redundant (≥ 0.9)
    assert admit(gallery, vectors[0], "usable", "frontal").reason == "tier"


def test_full_gallery_drops_the_most_redundant(registry):
    registry.max_faces = 4
    pid = person_with(registry, "Oleh", OLEH, 3)
    close = photo(777, OLEH, noise=0.3)                            # near the person's average face: adds little
    assert registry.add_embedding(pid, "face", close, meta=GOOD)
    assert registry.add_embedding(pid, "face", photo(99, OLEH, noise=1.5), meta=GOOD)   # a new variation
    vectors = registry.gallery(pid)
    assert len(vectors) == 4                                       # still the limit
    assert not any(np.allclose(v, close) for v in vectors)         # the most redundant one was dropped


def test_top3_matching_covers_variations(registry):
    glasses = base(7)                                              # nearly orthogonal to OLEH
    pid = registry.create_person("Oleh")
    for i in range(3):
        registry.add_embedding(pid, "face", photo(i, OLEH, 0.5), meta=GOOD)
        registry.add_embedding(pid, "face", photo(10 + i, glasses, 0.5), meta=GOOD)
    assert registry.counts[pid]["face"] == 6
    query = photo(50, glasses, 0.3)
    centroid_score = float(registry.centroid(pid, "face") @ query)
    person_id, score = registry.match_face(query)
    assert person_id == pid and score > 0.8 and score - centroid_score > 0.15


# --- Part 4: dedup --------------------------------------------------------------------------------------------------

def test_duplicate_of_oleh_is_auto_merged_with_good_embeddings_only(registry):
    registry.max_faces = 8
    oleh = person_with(registry, "Oleh", OLEH, 5)
    duplicate = person_with(registry, None, OLEH, 6, seed0=200)
    stranger = person_with(registry, None, base(9), 4, seed0=300)
    report = DedupJob(registry).run()
    assert [(s, t) for s, t, *_ in report["merged"]] == [(duplicate, oleh)]
    assert registry.resolve(duplicate) == oleh and registry.resolve(stranger) == stranger
    assert registry.counts[oleh]["face"] == 8                       # grew, but never past the limit
    assert registry.history(oleh, 1)[0]["action"] == "auto_merge"


def test_cannot_link_blocks_merge_even_when_similar(registry):
    person_with(registry, "Oleh", OLEH, 4)
    twin = person_with(registry, None, OLEH, 4, seed0=200)         # e.g. a twin brother
    engine, faces = engine_for(registry)
    # Both face-confirmed in one frame (track 1 = Oleh, track 2 = the twin)
    engine.tracks[1] = TrackIdentity(person_id=registry.find("Oleh"), confirmed_at=0.0)
    engine.tracks[2] = TrackIdentity(person_id=twin, confirmed_at=0.0)
    engine._seen_together(1.0, [person(1), person(2)])
    assert registry.blocked(twin, registry.find("Oleh"))
    report = DedupJob(registry).run()
    assert report["merged"] == [] and report["queued"] == []
    assert IdentityRegistry(registry.path).blocked(twin, registry.find("Oleh"))   # persisted


def test_two_identified_people_in_one_frame_get_a_cannot_link(registry):
    oleh, anna = person_with(registry, "Oleh", OLEH, 2), person_with(registry, "Anna", base(2), 2, seed0=40)
    engine, faces = engine_for(registry)
    left, right = [40, 60, 240, 460], [380, 60, 580, 460]
    rows = {95: photo(70, OLEH), 435: photo(71, base(2))}
    faces.rows = [face_row(x=95, y=80, size=90), face_row(x=435, y=80, size=90)]
    faces.embed = lambda image, face: rows[int(face[0])]
    engine.update(0.0, FRAME, [person(1, left), person(2, right)])
    assert registry.blocked(oleh, anna)


def test_merge_undo_round_trip(registry):
    registry.max_faces = 6
    oleh = person_with(registry, "Oleh", OLEH, 5)
    duplicate = person_with(registry, None, OLEH, 5, seed0=200)
    before = {pid: registry.gallery(pid).copy() for pid in (oleh, duplicate)}
    event = registry.merge(duplicate, oleh, action="auto_merge")
    assert registry.counts[oleh]["face"] == 6 and registry.resolve(duplicate) == oleh
    registry.unmerge(event.payload["event_id"])
    for pid in (oleh, duplicate):
        assert registry.resolve(pid) == pid
        assert sorted(map(bytes, registry.gallery(pid))) == sorted(map(bytes, before[pid]))
    assert registry.name(duplicate) == "unknown_1" and registry.blocked(oleh, duplicate)
    with pytest.raises(ValueError):
        registry.unmerge(event.payload["event_id"])                 # only once
    assert DedupJob(registry).run()["merged"] == []                 # not merged again after an undo


def test_undo_restores_a_name_given_by_the_merge(registry):
    oleh = person_with(registry, "Oleh", OLEH, 2)
    unknown = person_with(registry, None, OLEH, 2, seed0=200)
    event = registry.merge(oleh, unknown)                           # a known source names an unknown target
    assert registry.name(unknown) == "Oleh"
    registry.unmerge(event.payload["event_id"])
    assert registry.name(unknown) == "unknown_1" and not registry.is_known(unknown) and registry.name(oleh) == "Oleh"


def test_undo_window(registry, clock):
    oleh = person_with(registry, "Oleh", OLEH, 2)
    event = registry.merge(person_with(registry, None, OLEH, 2, seed0=200), oleh)
    clock.now += 8 * 86400
    assert not registry.merges()[0]["undoable"]
    with pytest.raises(ValueError):
        registry.unmerge(event.payload["event_id"])
    assert registry.purge_archived() >= 0


def test_review_queue_and_not_same_person(registry):
    known = person_with(registry, "Oleh", OLEH, 4)
    cfg = DedupConfig(auto_merge_threshold=0.99, review_threshold=0.42)  # medium confidence → review
    maybe = person_with(registry, None, OLEH, 4, seed0=200)
    published = []
    report = DedupJob(registry, cfg, publish=published.append).run()
    assert report["merged"] == [] and len(report["queued"]) == 1
    assert published[0].type == "duplicate_candidate"
    (candidate,) = registry.candidates()
    assert {candidate["person_a"], candidate["person_b"]} == {known, maybe}
    registry.reject_candidate(candidate["id"])
    assert registry.candidates() == [] and registry.blocked(known, maybe)
    assert DedupJob(registry, cfg).run()["queued"] == []            # never proposed again


def test_review_merge_into_known_person(registry):
    known = person_with(registry, "Oleh", OLEH, 4)
    maybe = person_with(registry, None, OLEH, 4, seed0=200)
    DedupJob(registry, DedupConfig(auto_merge_threshold=0.99)).run()
    event = registry.accept_candidate(registry.candidates()[0]["id"])
    assert event.payload["person_id"] == known and registry.resolve(maybe) == known
    assert registry.merges()[0]["action"] == "review_merge"


def test_chain_merge_guard(registry):
    a, c = base(11), base(12)
    b = normalize(a + c)                                            # A ~ B, B ~ C, but A ≁ C
    ids = [person_with(registry, None, v, 4, seed0=100 * i) for i, v in enumerate((a, b, c), start=1)]
    report = DedupJob(registry, DedupConfig(auto_merge_threshold=0.55)).run()
    live = {registry.resolve(pid) for pid in ids}
    assert len(report["merged"]) == 1 and len(live) == 2            # never all three into one person
    assert len(report["queued"]) == 1                               # the other link goes to a human


def test_continuity_and_voice_are_bonuses(registry):
    oleh = person_with(registry, "Oleh", OLEH, 3)
    unknown = registry.create_unknown("face", photo(1, OLEH, 1.5), meta=GOOD, details={"continuity": [oleh]})
    plain, evidence = DedupJob(registry).score(unknown, oleh)
    assert evidence["continuity"] is True and not evidence["voice"]
    assert plain == pytest.approx(evidence["similarity"] + 0.05, abs=1e-3)


def test_junk_unknowns_are_cleaned_up(registry, clock):
    junk = registry.create_unknown("face", base(20), meta=GOOD)
    kept = person_with(registry, None, base(21), 3, seed0=500)
    voice = registry.create_unknown("voice", normalize(np.ones(192)))
    clock.now += 4 * 86400
    report = DedupJob(registry).run()
    assert report["deleted"] == [junk]
    assert registry.resolve(kept) == kept and registry.resolve(voice) == voice


def test_usable_face_cannot_create_a_person(registry):
    with pytest.raises(ValueError):
        registry.create_unknown("face", base(3), meta={"tier": "usable"})


# --- metrics + dashboard ----------------------------------------------------------------------------------------

def test_metrics_report_accumulates(tmp_path, registry):
    person_with(registry, None, base(3), 2, seed0=10)
    report = MetricsReport(str(tmp_path / "identity_metrics.json"))
    report.update(registry, {"rejected:small": 2, "faces_good": 3})
    data = report.update(registry, {"rejected:small": 1, "rejected:pose": 1})
    (day,) = data["days"].values()
    assert day["rejected_faces"] == {"small": 3, "pose": 1} and day["faces"] == {"good": 3}
    assert day["unknowns_created"] == 1 and sum(day["unknowns_created_per_hour"].values()) == 1
    assert day["gallery_sizes"]["unknown_people"] == 1
    assert "unknown_1" not in json.dumps(data)                      # no names in the report


def test_dashboard_duplicates_api(tmp_path):
    from tests.test_dashboard import AUTH, client

    http, state = client(tmp_path)
    reg = state.registry = IdentityRegistry(str(tmp_path / "identity.db"), faces_dir=str(tmp_path / "persons"))
    known = person_with(reg, "Oleh", OLEH, 4)
    maybe = person_with(reg, None, OLEH, 4, seed0=200)
    other = person_with(reg, None, OLEH, 4, seed0=400)
    DedupJob(reg, DedupConfig(auto_merge_threshold=0.99)).run()
    items = http.get("/api/persons/duplicates", auth=AUTH).json()["duplicates"]
    assert len(items) >= 2 and {items[0]["a"]["person_id"], items[0]["b"]["person_id"]} & {maybe, other}
    first = next(d for d in items if {d["a"]["person_id"], d["b"]["person_id"]} == {known, maybe})
    merged = http.post(f"/api/persons/duplicates/{first['id']}/merge", auth=AUTH).json()
    assert merged["merged_into"] == known
    while rest := http.get("/api/persons/duplicates", auth=AUTH).json()["duplicates"]:
        assert http.post(f"/api/persons/duplicates/{rest[0]['id']}/reject", auth=AUTH).status_code == 200
    assert http.get("/api/persons/duplicates", auth=AUTH).json()["count"] == 0
    (merge,) = [m for m in http.get("/api/identity/merges", auth=AUTH).json()["merges"]]
    assert merge["undoable"] and merge["source_name"] == "unknown_1"
    assert http.post(f"/api/identity/merges/{merge['event_id']}/undo", auth=AUTH).json()["restored"] == maybe
    assert http.post(f"/api/identity/merges/{merge['event_id']}/undo", auth=AUTH).status_code == 409
    assert http.post("/api/persons/duplicates/999/merge", auth=AUTH).status_code == 404


def test_queue_shows_one_card_per_live_pair(registry):
    oleh = person_with(registry, "Oleh", OLEH, 4)
    first = person_with(registry, None, OLEH, 4, seed0=200)
    second = person_with(registry, None, OLEH, 4, seed0=400)
    for unknown in (first, second):
        registry.propose(unknown, oleh, 0.5, {})
    registry.merge(second, first)                                   # both rows now mean first ~ Oleh
    (candidate,) = registry.candidates()
    assert {candidate["person_a"], candidate["person_b"]} == {first, oleh}
    assert registry.propose(first, oleh, 0.55, {}) == candidate["id"]   # updated, not added again
