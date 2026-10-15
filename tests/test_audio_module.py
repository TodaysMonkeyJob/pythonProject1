import queue
import threading

from core.bus import CONTROL_STOP, create_queue, receive
from modules.audio.adapter import AudioModule, parse_transcription


def test_parse_transcription():
    assert parse_transcription("hello there; oleh") == ("hello there", "oleh")
    assert parse_transcription("a; b; unknown") == ("a; b", "unknown")
    assert parse_transcription("no speaker") == ("no speaker", "")


class FakeRecorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.is_shut_down = False
        self._results = queue.Queue()
        self._results.put("turn on the light; oleh")

    def text(self):
        try:
            return self._results.get(timeout=0.05)
        except queue.Empty:
            return ""

    def shutdown(self):
        self.is_shut_down = True


def test_audio_publishes_speech_events():
    event_queue, control_queue = create_queue(), create_queue()
    config = {"audio": {"recorder": {"language": "en"}}}
    module = AudioModule(event_queue, control_queue, config, recorder_factory=FakeRecorder)
    thread = threading.Thread(target=module.run)
    thread.start()

    received = [receive(event_queue, timeout=5) for _ in range(2)]
    control_queue.put(CONTROL_STOP)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert module.recorder.kwargs == {"language": "en"}
    assert module.recorder.is_shut_down
    speech = next(e for e in received if e.type == "speech")
    assert speech.source == "audio"
    assert speech.payload == {"text": "turn on the light", "speaker": "oleh"}
