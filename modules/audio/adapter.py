"""
Audio module adapter around the existing ``speech_to_text_whisper.AudioToTextRecorder``.

``recorder.text()`` blocks until an utterance is transcribed, so it runs in a background thread
while the module loop keeps sending heartbeats and reacting to control messages.
Each transcription becomes a ``speech`` event: ``{"text": ..., "speaker": ...}``.

The recorder spawns its own child processes, so the orchestrator must start this module in a
non-daemon process.
"""
import threading
import time
from typing import Any, Dict, Tuple

from core.module import BaseModule
from system_setup.logger_setup import init_logger

logger = init_logger()


def parse_transcription(result: str) -> Tuple[str, str]:
    """
    Splits the recorder's ``"<text>; <speaker>"`` output.

    :param result: String returned by AudioToTextRecorder.text()
    :return: (text, speaker); speaker is "" when absent
    """
    text, sep, speaker = result.rpartition("; ")
    if not sep:
        return result.strip(), ""
    return text.strip(), speaker.strip()


class AudioModule(BaseModule):
    name = "audio"
    source = "audio"

    def __init__(self, event_queue, control_queue, config: Dict[str, Any], recorder_factory=None):
        """
        :param recorder_factory: Callable(**kwargs) returning a recorder; defaults to AudioToTextRecorder
        """
        super().__init__(event_queue, control_queue, config)
        self.audio_config = config["audio"]
        self.recorder_factory = recorder_factory
        self.recorder = None
        self._listener = None

    def setup(self) -> None:
        if self.recorder_factory is None:
            # Heavy import (torch, whisper) only inside the audio process
            from speech_to_text_whisper import AudioToTextRecorder
            self.recorder_factory = AudioToTextRecorder
        self.recorder = self.recorder_factory(**self.audio_config.get("recorder", {}))
        self._listener = threading.Thread(target=self._listen, name="audio-listener", daemon=True)
        self._listener.start()

    def step(self) -> None:
        time.sleep(0.1)
        if self._listener is not None and not self._listener.is_alive():
            # Listener died (recorder error): stop so the orchestrator can restart the module
            self.running = False

    def teardown(self) -> None:
        if self.recorder is not None:
            self.recorder.shutdown()
        if self._listener is not None:
            self._listener.join(timeout=5)

    def _listen(self) -> None:
        try:
            while self.running and not getattr(self.recorder, "is_shut_down", False):
                result = self.recorder.text()
                if not result:
                    continue
                text, speaker = parse_transcription(result)
                if text:
                    self.emit("speech", {"text": text, "speaker": speaker})
        except Exception as e:
            logger.error(f"Audio listener failed: {e}", exc_info=True)
