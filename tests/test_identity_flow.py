"""F9 end-to-end pieces: speaker → linker in the audio module, names at snapshot time, overlay, migration, CLI."""
import json
import queue
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

from core.events import Event, make_event
from core.identity import IdentityRegistry
from modules.aggregator.aggregator import Aggregator
from modules.aggregator.digest import snapshot_digest
from modules.audio.mic_controller import MicController
from modules.audio.speaker import SpeakerAnalyzer
from tests.test_identity_registry import OLEH_FACE, OLEH_VOICE, face
from tests.test_speaker import fake_embed, fake_mask, tone

REPO = Path(__file__).resolve().parent.parent
SEGMENT = (tone(220, 2.0) * 32767).astype(np.int16)    # 2 s of "voice A" (see tests/test_speaker.py)


class Wall:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def audio_module(tmp_path):
    registry = IdentityRegistry(str(tmp_path / "identity.db"))
    face_id = registry.create_unknown("face", OLEH_FACE)
    registry.rename(face_id, "Oleh")
    wall = Wall()
    speaker = SpeakerAnalyzer(registry, fake_embed, fake_mask)
    config = {"audio": {"mic": {"device_id": "mic-test"}}, "identity": {"link_min_cooccurrences": 3,
                                                                        "location_id": "office"},
              "logging": {"dir": str(tmp_path / "logs")}}
    module = MicController(queue.Queue(), queue.Queue(), config, frame_source=object(), vad=lambda f: False,
                           transcriber=lambda audio, sr: {"text": "привіт", "language": "uk"}, speaker=speaker,
                           wall_clock=wall)
    return module, registry, face_id, wall


def control(module, type, payload):
    module.handle_control(make_event("camera", type, payload).to_json())


def speech_events(module):
    events = []
    while not module.event_queue.empty():
        event = Event.from_json(module.event_queue.get())
        if event.type == "speech":
            events.append(event)
    return events


def say(module, wall, seconds=2.0):
    wall.now += 10
    module._transcribe(SEGMENT, seconds, wall.now)


def test_voice_of_the_person_alone_in_view_gets_their_name(tmp_path):
    module, registry, oleh, wall = audio_module(tmp_path)
    control(module, "object_detected", {"track_id": 4, "class": "person", "state": "appeared"})
    control(module, "person_identified", {"track_id": 4, "person_id": oleh, "identity": "Oleh"})
    for _ in range(3):
        say(module, wall)
    events = speech_events(module)
    assert [e.payload["speaker_name"] for e in events] == ["unknown_2", "unknown_2", "Oleh"]   # linked on the 3rd
    assert events[0].payload["speaker_status"] == "unknown" and events[2].payload["speaker_status"] == "known"
    assert (events[0].payload["location_id"], events[0].payload["device_id"]) == ("office", "mic-test")
    assert registry.counts[oleh]["face"] == 1 and registry.counts[oleh]["voice"] >= 1
    control(module, "object_detected", {"track_id": 4, "class": "person", "state": "disappeared"})
    say(module, wall)                                                           # off camera
    (event,) = speech_events(module)
    assert event.payload["speaker_name"] == "Oleh" and event.payload["person_id"] == oleh
    assert "link_voice" in [h["action"] for h in registry.history()]


def test_two_people_in_view_keep_the_voice_separate(tmp_path):
    module, registry, oleh, wall = audio_module(tmp_path)
    anna = registry.create_unknown("face", face(2))
    for track, person in ((1, oleh), (2, anna)):
        control(module, "object_detected", {"track_id": track, "class": "person", "state": "appeared"})
        control(module, "person_identified", {"track_id": track, "person_id": person})
    for _ in range(5):
        say(module, wall)
    assert {e.payload["speaker_name"] for e in speech_events(module)} == {"unknown_3"}


def test_short_segments_and_disabled_speaker(tmp_path):
    module, _, _, wall = audio_module(tmp_path)
    module._transcribe(SEGMENT[:8000], 0.5, wall.now)                           # 0.5 s: too short
    (event,) = speech_events(module)
    assert event.payload["speaker_status"] == "uncertain" and event.payload["person_id"] is None
    off = MicController(queue.Queue(), queue.Queue(), {"audio": {"mic": {"transcript_log": False}}},
                        frame_source=object(), transcriber=lambda audio, sr: {"text": "x"})
    assert off.speaker is False and off._diarize(SEGMENT, 2.0) is None
    off._transcribe(SEGMENT, 2.0, 100.0)
    (plain,) = speech_events(off)
    assert "speaker_status" not in plain.payload and plain.payload["start"].endswith("Z")


def test_transcript_files_follow_the_registry(tmp_path):
    module, registry, _, wall = audio_module(tmp_path)
    say(module, wall)
    (event,) = speech_events(module)
    registry.rename(event.payload["person_id"], "Ivan")
    say(module, wall)
    module.teardown()
    (day,) = (tmp_path / "logs").iterdir()
    lines = (day / "transcripts.txt").read_text(encoding="utf-8").splitlines()
    assert "[office · mic-test] unknown_2: \"привіт\"" in lines[0] and "] Ivan: " in lines[1]
    assert len((day / "transcripts.jsonl").read_text().splitlines()) == 2


# --- aggregator: names resolved at snapshot time ----------------------------------------------------------------

def test_snapshot_uses_the_current_name(tmp_path):
    registry = IdentityRegistry(str(tmp_path / "identity.db"))
    pid = registry.create_unknown("face", OLEH_FACE)
    voice_id = registry.create_unknown("voice", OLEH_VOICE)
    config = {"aggregator": {"export_dir": str(tmp_path / "exports"), "incidents": {"enabled": False}}}
    aggregator = Aggregator(config, local_now=lambda: datetime(2026, 10, 17, 12, 0), registry=registry)
    aggregator.on_event(make_event("camera", "object_detected", {"track_id": 3, "class": "person",
                                                                 "state": "appeared"}))
    aggregator.on_event(make_event("camera", "person_identified", {"track_id": 3, "person_id": pid,
                                                                   "identity": "unknown_1", "known": False}))
    aggregator.on_event(make_event("audio", "speech", {"text": "добрий день", "language": "uk",
                                                       "person_id": voice_id, "speaker_name": "unknown_2",
                                                       "speaker_status": "unknown"}))
    other = IdentityRegistry(str(tmp_path / "identity.db"))                    # the dashboard renames
    other.rename(pid, "Oleh")
    other.merge(voice_id, pid)
    snapshot = aggregator.snapshot("interval", aggregator.clock())
    assert snapshot["people"][0]["identity"] == "Oleh" and snapshot["people"][0]["known"] is True
    assert snapshot["people"][0]["person_id"] == pid
    assert snapshot["speech"][0]["speaker"] == "Oleh"
    assert 'uk Oleh "добрий день"' in snapshot_digest(snapshot)
    aggregator.close()


# --- overlay ----------------------------------------------------------------------------------------------------

def test_overlay_follows_person_identified_events():
    from modules.vision.detector import VisionDetector, annotate
    from modules.vision.motion import MotionResult
    from modules.vision.state import Track

    module = VisionDetector(queue.Queue(), queue.Queue(), {"vision": {}}, frame_source=object(),
                            object_detector=object())
    module.handle_control(make_event("camera", "person_identified", {"track_id": 5, "identity": "Oleh"}).to_json())
    assert module.person_names == {5: "Oleh"}
    module.handle_control(make_event("camera", "person_identified",
                                     {"track_id": 5, "identity": "Petro", "renamed_from": "Oleh"}).to_json())
    assert module.person_names == {5: "Petro"}
    frame = np.zeros((120, 160, 3), np.uint8)
    track = Track(track_id=5, cls="person", bbox=[10, 30, 100, 110], confidence=0.9, first_seen=0, last_seen=0)
    motion = MotionResult(False)
    assert not np.array_equal(annotate(frame, motion, [track], {5: "Petro"}), annotate(frame, motion, [track]))


# --- migration and CLI ------------------------------------------------------------------------------------------

def legacy_faces(root: Path):
    root.mkdir(parents=True)
    np.savez(root / "faces.npz", names=np.array(["Oleh"]), embeddings=OLEH_FACE.reshape(1, -1),
             counts=np.array([3]))
    (root / "Oleh").mkdir()
    (root / "Oleh" / "a.jpg").write_bytes(b"jpeg")
    folder = root / "unknown" / "unknown_7"
    folder.mkdir(parents=True)
    np.save(folder / "embeddings.npy", np.stack([face(70), face(71, face(70))]).astype(np.float32))
    (folder / "meta.json").write_text("{}")
    (root / "unknown" / "_counter.json").write_text(json.dumps({"last": 9}))


def test_migration_imports_known_and_unknown_people(tmp_path):
    from modules.vision.legacy_faces import migrate

    legacy_faces(tmp_path / "faces")
    registry = IdentityRegistry(str(tmp_path / "identity.db"), faces_dir=str(tmp_path / "persons"))
    report = migrate(registry, str(tmp_path / "faces" / "faces.npz"))
    assert report == {"known": 1, "unknown": 1, "images": 1, "skipped": 0}
    oleh, stranger = registry.find("Oleh"), registry.find("unknown_7")              # numbers are kept
    assert registry.is_known(oleh) and registry.counts[stranger]["face"] == 2
    assert registry.match_face(face(5, OLEH_FACE))[0] == oleh and registry.thumbnail(oleh)
    assert registry.name(registry.create_unknown("face", face(99))) == "unknown_10"   # after _counter.json
    assert migrate(registry, str(tmp_path / "faces" / "faces.npz"))["skipped"] == 2   # safe to run twice


@pytest.fixture
def cli_config(tmp_path, monkeypatch):
    config = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    config["identity"].update(registry_path=str(tmp_path / "identity.db"), faces_dir=str(tmp_path / "persons"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    monkeypatch.chdir(tmp_path)                                                   # no .env here
    return str(path)


def test_identity_cli(cli_config, tmp_path, capsys):
    from tools import identity

    registry = IdentityRegistry(str(tmp_path / "identity.db"))
    pid = registry.create_unknown("face", OLEH_FACE)
    voice_id = registry.create_unknown("voice", OLEH_VOICE)
    identity.main(["--config", cli_config, "rename", "unknown_1", "Oleh"])
    identity.main(["--config", cli_config, "merge", "unknown_2", "Oleh"])
    identity.main(["--config", cli_config, "list"])
    out = capsys.readouterr().out
    assert "unknown_1 → Oleh" in out and "faces 1, voices 1" in out and "1 people" in out
    registry.poll()
    assert registry.resolve(voice_id) == pid and registry.name(pid) == "Oleh"
    with pytest.raises(SystemExit, match="already exists"):
        identity.main(["--config", cli_config, "rename", registry.create_unknown("face", face(3)), "oleh"])
    identity.main(["--config", cli_config, "delete", "Oleh", "--yes"])
    identity.main(["--config", cli_config, "history"])
    assert "delete" in capsys.readouterr().out
