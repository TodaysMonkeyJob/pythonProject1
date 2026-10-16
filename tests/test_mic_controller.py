import queue

import numpy as np
import pytest

from core.events import Event, make_event
from modules.audio.mic_controller import (ACTIVE, STANDBY, MicController, MicStateMachine, SpeechSegmenter,
                                          rms_dbfs)

FRAME = 480                       # 30 ms at 16 kHz
SILENCE = np.zeros(FRAME, np.int16)
SPEECH = (np.sin(np.linspace(0, 60 * np.pi, FRAME)) * 8000).astype(np.int16)   # ≈ -15 dBFS
NOISE = (np.random.default_rng(0).normal(size=FRAME) * 200).astype(np.int16)    # ≈ -44 dBFS, quiet room


class ScriptedMic:
    def __init__(self):
        self.pending = []
        self.opened = False

    def open(self):
        self.opened = True

    def read(self):
        frames, self.pending = self.pending, []
        return frames

    def close(self):
        pass


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make(**mic_cfg):
    clock, mic, calls = Clock(), ScriptedMic(), []

    def transcriber(audio, sample_rate):
        calls.append(len(audio) / sample_rate)
        return {"text": "hello there", "language": "en"}

    cfg = {"audio": {"mic": {"idle_timeout_sec": 15, "sound_threshold_db": -35, **mic_cfg}}}
    controller = MicController(queue.Queue(), queue.Queue(), cfg, frame_source=mic,
                               vad=lambda frame: rms_dbfs(frame) > -30, transcriber=transcriber, clock=clock)
    controller.setup()
    return controller, mic, clock, calls


def feed(controller, mic, clock, frames):
    for frame in frames:
        mic.pending = [frame]
        controller.step()
        clock.now += 0.03


def drain(controller):
    events = []
    while True:
        try:
            events.append(Event.from_json(controller.event_queue.get_nowait()))
        except queue.Empty:
            return events


# --- building blocks --------------------------------------------------------------------------------------------

def test_rms_dbfs():
    assert rms_dbfs(SILENCE) == -90.0
    assert -17 < rms_dbfs(SPEECH) < -13
    assert -48 < rms_dbfs(NOISE) < -40
    full = np.full(FRAME, 32767, np.int16)
    assert rms_dbfs(full) == pytest.approx(0, abs=0.01)


def test_state_machine():
    sm = MicStateMachine(idle_timeout_sec=15)
    assert sm.update(0, False, False, False) is None and sm.state == STANDBY
    assert sm.update(1, True, False, False) == (ACTIVE, "scene_active")
    assert sm.update(2, False, False, False) is None                       # scene gone, timer runs
    assert sm.update(10, False, True, False) is None                       # loud sound restarts the timer
    assert sm.update(24.9, False, False, False) is None
    assert sm.update(25, False, False, False) == (STANDBY, "idle")
    assert sm.update(30, False, False, True) == (ACTIVE, "speech")
    sm = MicStateMachine(15)
    sm.update(0, True, False, False)
    assert sm.update(100, True, False, False) is None                     # stays active while scene_active


def test_segmenter_preroll_end_and_limits():
    seg = SpeechSegmenter(frame_ms=30, preroll_sec=0.09, silence_end_sec=0.09, max_segment_sec=0.3, min_speech_sec=0.09)
    for _ in range(5):
        assert seg.push(NOISE, False, allowed=True) is None
    for _ in range(4):
        assert seg.push(SPEECH, True, allowed=True) is None
    seg.push(SILENCE, False, True)
    seg.push(SILENCE, False, True)
    segment = seg.push(SILENCE, False, True)
    assert segment is not None and len(segment) == (3 + 4 + 3) * FRAME     # pre-roll + speech + trailing silence
    assert np.array_equal(segment[:FRAME], NOISE)

    out = [seg.push(SPEECH, True, True) for _ in range(10)]               # continuous speech: cut at 0.3 s
    assert len([s for s in out if s is not None]) == 1 and len(next(s for s in out if s is not None)) == 10 * FRAME


def test_segmenter_ignores_blips_and_standby():
    seg = SpeechSegmenter(frame_ms=30, preroll_sec=0.09, silence_end_sec=0.09, min_speech_sec=0.3)
    assert seg.push(SPEECH, True, allowed=False) is None and not seg.recording   # standby: nothing recorded
    seg.push(SPEECH, True, True)
    assert all(seg.push(SILENCE, False, True) is None for _ in range(3))        # 1 speech frame < 0.3 s
    assert not seg.recording


# --- controller -------------------------------------------------------------------------------------------------

def test_quiet_empty_room_sends_nothing_to_whisper():
    controller, mic, clock, calls = make()
    feed(controller, mic, clock, [NOISE] * 2000)                           # 60 s of a quiet room
    assert calls == [] and controller.transcriptions == 0
    assert controller.state.state == STANDBY
    assert drain(controller) == []


def test_speech_in_standby_is_not_recorded_until_active():
    controller, mic, clock, calls = make()
    feed(controller, mic, clock, [SPEECH] * 30 + [SILENCE] * 40)            # wakes up on speech, then records
    events = drain(controller)
    assert [e.payload["state"] for e in events if e.type == "mic_state"] == ["active"]
    controller.executor.shutdown(wait=True)
    assert len(calls) == 1 and calls[0] < 30 * 0.03 + 0.5 + 0.9           # the waking frame started the segment
    speech = [e for e in drain(controller) + events if e.type == "speech"]
    assert speech and speech[0].payload == {"text": "hello there", "language": "en", "backend": None,
                                            "duration_sec": pytest.approx(calls[0], abs=0.01)}
    assert speech[0].source == "audio"


def test_scene_active_wakes_mic_immediately_and_idle_returns_to_standby():
    controller, mic, clock, calls = make(idle_timeout_sec=15)
    controller.handle_control(make_event("camera", "scene_active", {"active": True}).to_json())
    (event,) = drain(controller)
    assert (event.source, event.type, event.payload["state"], event.payload["reason"]) == \
        ("system", "mic_state", "active", "scene_active")
    controller.handle_control(make_event("camera", "scene_active", {"active": False}).to_json())
    feed(controller, mic, clock, [NOISE] * 499)                            # < 15 s
    assert controller.state.state == ACTIVE
    feed(controller, mic, clock, [NOISE] * 10)
    events = drain(controller)
    assert [e.payload for e in events if e.type == "mic_state"] == [{"state": "standby", "reason": "idle",
                                                                    "module": "audio"}]
    assert calls == []                                                     # nobody spoke


def test_loud_sound_event_with_cooldown():
    controller, mic, clock, _ = make(sound_cooldown_sec=5)
    loud = (np.ones(FRAME) * 3000).astype(np.int16)                        # loud but VAD says no speech here
    controller.vad = lambda frame: False
    feed(controller, mic, clock, [loud, SILENCE, loud, SILENCE])           # second bang within the cooldown
    sounds = [e for e in drain(controller) if e.type == "sound"]
    assert len(sounds) == 1 and sounds[0].payload["rms_db"] > -35


def test_microphone_errors_are_retried():
    controller, mic, clock, _ = make()

    def broken():
        raise OSError("device unplugged")

    mic.read = broken
    controller.step()
    assert not controller._source_ok
    mic.read = lambda: []
    clock.now += 11
    controller.step()
    assert controller._source_ok


def test_whisper_errors_do_not_stop_the_module():
    controller, mic, clock, _ = make()

    def failing(audio, sample_rate):
        raise ConnectionError("offline")

    controller.transcriber = failing
    feed(controller, mic, clock, [SPEECH] * 20 + [SILENCE] * 40)
    controller.executor.shutdown(wait=True)
    assert controller.transcriptions == 1
    assert not [e for e in drain(controller) if e.type == "speech"]


def test_create_transcriber_backends():
    from modules.llm.whisper_client import AzureWithLocalFallback, create_transcriber

    assert create_transcriber({"audio": {"mic": {"whisper_backend": "none"}}})(SPEECH, 16000)["text"] == ""
    fallback = create_transcriber({"audio": {"mic": {"whisper_backend": "azure"}}, "llm": {}})   # no keys
    assert isinstance(fallback, AzureWithLocalFallback) and fallback.azure is None and not fallback.using_azure
    local = create_transcriber({"audio": {"mic": {"whisper_backend": "local"}}})
    assert callable(local) and hasattr(local, "warm_up")
    with pytest.raises(ValueError):
        create_transcriber({"audio": {"mic": {"whisper_backend": "magic"}}})


class Recorder:
    def __init__(self, name, fail=False):
        self.name, self.fail, self.calls, self.warmed = name, fail, 0, False

    def __call__(self, audio, sample_rate):
        self.calls += 1
        if self.fail:
            raise ConnectionError("network is unreachable")
        return {"text": f"from {self.name}", "language": "en"}

    def warm_up(self):
        self.warmed = True


def fallback(azure, local, clock, retry_sec=300):
    from modules.llm.whisper_client import AzureWithLocalFallback

    config = {"audio": {"mic": {"azure_retry_sec": retry_sec}}}
    return AzureWithLocalFallback(config, azure_factory=lambda c: azure, local_factory=lambda c: local, clock=clock)


def test_azure_used_while_it_works():
    azure, local, clock = Recorder("azure"), Recorder("local"), Clock()
    transcriber = fallback(azure, local, clock)
    assert transcriber(SPEECH, 16000) == {"text": "from azure", "language": "en", "backend": "azure"}
    transcriber.warm_up()
    assert (azure.calls, local.calls, local.warmed) == (1, 0, False)          # local model not loaded needlessly


def test_azure_failure_switches_to_local_and_retries_later():
    azure, local, clock = Recorder("azure", fail=True), Recorder("local"), Clock()
    transcriber = fallback(azure, local, clock, retry_sec=300)
    first = transcriber(SPEECH, 16000)
    assert first["backend"] == "local" and first["text"] == "from local"     # the same segment is not lost
    clock.now = 100
    assert transcriber(SPEECH, 16000)["backend"] == "local"
    assert azure.calls == 1                                                 # no Azure attempt inside the window
    transcriber.warm_up()
    assert local.warmed
    azure.fail = False
    clock.now = 301
    assert transcriber(SPEECH, 16000)["backend"] == "azure"                 # Azure is back
    assert azure.calls == 2 and transcriber.azure_failed_at is None


def test_missing_azure_keys_use_local_from_the_start():
    from modules.llm.whisper_client import AzureWithLocalFallback

    local = Recorder("local")

    def no_keys(config):
        raise RuntimeError("Azure Whisper needs AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT (.env)")

    transcriber = AzureWithLocalFallback({}, azure_factory=no_keys, local_factory=lambda c: local)
    transcriber.warm_up()
    assert local.warmed
    assert transcriber(SPEECH, 16000)["backend"] == "local"


def test_local_failure_is_reported_not_hidden():
    azure, local, clock = Recorder("azure", fail=True), Recorder("local", fail=True), Clock()
    with pytest.raises(ConnectionError):
        fallback(azure, local, clock)(SPEECH, 16000)


def test_speech_event_shows_backend():
    controller, mic, clock, _ = make()
    controller.transcriber = lambda audio, sr: {"text": "hi", "language": "en", "backend": "local"}
    feed(controller, mic, clock, [SPEECH] * 20 + [SILENCE] * 40)
    controller.executor.shutdown(wait=True)
    (speech,) = [e for e in drain(controller) if e.type == "speech"]
    assert speech.payload["backend"] == "local"
