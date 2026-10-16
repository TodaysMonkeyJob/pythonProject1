"""
Microphone controller (F4): STANDBY / ACTIVE state machine in front of Whisper.

* STANDBY — the microphone is open, but only loudness (RMS, dBFS) and VAD are computed on 30 ms frames; nothing
  is recorded or sent anywhere (only a 0.5 s in-memory pre-roll ring is kept, overwritten continuously).
* → ACTIVE when ``scene_active`` is true (forwarded from the camera by the orchestrator), or a frame is louder than
  ``sound_threshold_db``, or VAD hears speech.
* ACTIVE — speech is cut into segments (pre-roll 0.5 s, end after ``silence_end_sec`` of silence, at most
  ``max_segment_sec``), each segment is transcribed in a background thread and published as a ``speech`` event.
* → STANDBY after ``idle_timeout_sec`` without scene, loud sound or speech.

Every switch is a ``mic_state`` event (source ``system``); a loud sound in standby is a ``sound`` event.
"""
import collections
import math
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from core.events import make_event
from core.module import BaseModule
from system_setup.logger_setup import init_logger

logger = init_logger()

STANDBY, ACTIVE = "standby", "active"


def rms_dbfs(frame: np.ndarray) -> float:
    """
    :param frame: int16 samples
    :return: Loudness in dBFS (0 = full scale, silence ≈ -90)
    """
    if frame.size == 0:
        return -90.0
    rms = math.sqrt(float(np.mean(frame.astype(np.float64) ** 2))) / 32768.0
    return 20 * math.log10(rms) if rms > 0 else -90.0


class MicStateMachine:
    def __init__(self, idle_timeout_sec: float = 15.0):
        self.idle_timeout_sec = idle_timeout_sec
        self.state = STANDBY
        self.last_trigger: Optional[float] = None

    def update(self, now: float, scene_active: bool, loud: bool, speech: bool) -> Optional[Tuple[str, str]]:
        """
        :return: (new state, reason) when the state changes, otherwise None
        """
        reason = "scene_active" if scene_active else "sound" if loud else "speech" if speech else None
        if reason:
            self.last_trigger = now
            if self.state == STANDBY:
                self.state = ACTIVE
                return ACTIVE, reason
        elif self.state == ACTIVE and now - self.last_trigger >= self.idle_timeout_sec:
            self.state = STANDBY
            return STANDBY, "idle"
        return None


class SpeechSegmenter:
    """Cuts a stream of 30 ms frames into speech segments (only while recording is allowed)."""

    def __init__(self, frame_ms: int = 30, preroll_sec: float = 0.5, silence_end_sec: float = 0.8,
                 max_segment_sec: float = 30.0, min_speech_sec: float = 0.3):
        self.preroll: Deque[np.ndarray] = collections.deque(maxlen=max(1, round(preroll_sec * 1000 / frame_ms)))
        self.end_frames = max(1, round(silence_end_sec * 1000 / frame_ms))
        self.max_frames = max(1, round(max_segment_sec * 1000 / frame_ms))
        self.min_speech_frames = max(1, round(min_speech_sec * 1000 / frame_ms))
        self.frames: Optional[List[np.ndarray]] = None
        self.silence = 0
        self.speech_frames = 0

    @property
    def recording(self) -> bool:
        return self.frames is not None

    def push(self, frame: np.ndarray, is_speech: bool, allowed: bool) -> Optional[np.ndarray]:
        """
        :param frame: int16 frame
        :param is_speech: VAD result for the frame
        :param allowed: ACTIVE state; in STANDBY only the pre-roll ring is updated
        :return: A finished segment (int16), or None
        """
        if self.frames is None:
            if allowed and is_speech:
                self.frames = list(self.preroll) + [frame]
                self.silence, self.speech_frames = 0, 1
                self.preroll.clear()
            else:
                self.preroll.append(frame)
            return None
        self.frames.append(frame)
        if is_speech:
            self.silence = 0
            self.speech_frames += 1
        else:
            self.silence += 1
        if self.silence >= self.end_frames or len(self.frames) >= self.max_frames:
            return self.finish()
        return None

    def finish(self) -> Optional[np.ndarray]:
        frames, speech = self.frames, self.speech_frames
        self.frames, self.silence, self.speech_frames = None, 0, 0
        if not frames or speech < self.min_speech_frames:
            return None
        return np.concatenate(frames)

    def reset(self) -> None:
        self.frames, self.silence, self.speech_frames = None, 0, 0
        self.preroll.clear()


class SoundDeviceSource:
    """Microphone frames via sounddevice (callback → queue); read() returns what arrived since the last call."""

    def __init__(self, sample_rate: int = 16000, frame_samples: int = 480, device=None):
        self.sample_rate, self.frame_samples, self.device = sample_rate, frame_samples, device
        self.frames: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1000)
        self.stream = None

    def open(self) -> None:
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            try:
                self.frames.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass  # the controller fell behind; dropping is better than growing memory

        self.stream = sd.InputStream(samplerate=self.sample_rate, blocksize=self.frame_samples, channels=1,
                                     dtype="int16", device=self.device, callback=callback)
        self.stream.start()

    def read(self) -> List[np.ndarray]:
        frames = []
        while True:
            try:
                frames.append(self.frames.get_nowait())
            except queue.Empty:
                return frames

    def close(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


def webrtc_vad(aggressiveness: int, sample_rate: int) -> Callable[[np.ndarray], bool]:
    import webrtcvad

    vad = webrtcvad.Vad(aggressiveness)
    return lambda frame: vad.is_speech(frame.tobytes(), sample_rate)


class MicController(BaseModule):
    name = "audio"
    source = "audio"

    def __init__(self, event_queue, control_queue, config: Dict[str, Any], frame_source=None,
                 vad: Optional[Callable[[np.ndarray], bool]] = None, transcriber=None,
                 clock: Callable[[], float] = time.monotonic):
        """
        :param frame_source: Object with open() / read() -> [int16 frame] / close(); SoundDeviceSource by default
        :param vad: frame -> is speech; webrtcvad by default
        :param transcriber: (audio, sample_rate) -> {"text", "language"}; from modules.llm.whisper_client by default
        :param clock: Monotonic time source (tests)
        """
        super().__init__(event_queue, control_queue, config)
        cfg = config.get("audio", {}).get("mic", {})
        self.cfg = cfg
        self.sample_rate = int(cfg.get("sample_rate", 16000))
        self.frame_ms = int(cfg.get("frame_ms", 30))
        self.sound_threshold_db = float(cfg.get("sound_threshold_db", -35))
        self.sound_cooldown_sec = float(cfg.get("sound_cooldown_sec", 5))
        self.state = MicStateMachine(float(cfg.get("idle_timeout_sec", 15)))
        self.segmenter = SpeechSegmenter(self.frame_ms, float(cfg.get("preroll_sec", 0.5)),
                                         float(cfg.get("silence_end_sec", 0.8)), float(cfg.get("max_segment_sec", 30)),
                                         float(cfg.get("min_speech_sec", 0.3)))
        self.frame_source = frame_source
        self.vad = vad
        self.transcriber = transcriber
        self.clock = clock
        self.scene_active = False
        self.transcriptions = 0
        self.executor: Optional[ThreadPoolExecutor] = None
        self._source_ok = False
        self._next_open = 0.0
        self._was_loud = False
        self._last_sound_event = float("-inf")
        self._lock = threading.Lock()

    def setup(self) -> None:
        if self.frame_source is None:
            self.frame_source = SoundDeviceSource(self.sample_rate, self.sample_rate * self.frame_ms // 1000,
                                                  self.cfg.get("device"))
        if self.vad is None:
            self.vad = webrtc_vad(int(self.cfg.get("vad_aggressiveness", 2)), self.sample_rate)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self._open_source()

    def teardown(self) -> None:
        if self.frame_source is not None:
            try:
                self.frame_source.close()
            except Exception as e:
                logger.debug(f"Error closing microphone: {e}")
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)

    def handle_control(self, message: Any) -> None:
        """Forwarded ``scene_active`` events switch the microphone on immediately."""
        if isinstance(message, str) and '"scene_active"' in message:
            from core.events import Event

            self.scene_active = bool(Event.from_json(message).payload.get("active"))
            self._update_state(False, False)
        else:
            super().handle_control(message)

    def step(self) -> None:
        if not self._source_ok:
            if self.clock() >= self._next_open:
                self._open_source()
            self._update_state(False, False)
            time.sleep(0.05)
            return
        try:
            frames = self.frame_source.read()
        except Exception as e:
            logger.error(f"Microphone read failed: {e}")
            self._source_ok = False
            self._next_open = self.clock() + 10
            return
        if not frames:
            self._update_state(False, False)
            time.sleep(0.01)
            return
        for frame in frames:
            self.process_frame(np.asarray(frame, dtype=np.int16))

    def process_frame(self, frame: np.ndarray) -> None:
        """Loudness + VAD always; recording only in ACTIVE."""
        db = rms_dbfs(frame)
        loud = db >= self.sound_threshold_db
        speech = bool(self.vad(frame))
        self._update_state(loud, speech, db)
        segment = self.segmenter.push(frame, speech, allowed=self.state.state == ACTIVE)
        if segment is not None:
            self._submit(segment)

    def _update_state(self, loud: bool, speech: bool, db: Optional[float] = None) -> None:
        now = self.clock()
        if loud and not self._was_loud and now - self._last_sound_event >= self.sound_cooldown_sec:
            self._last_sound_event = now
            self.emit("sound", {"rms_db": round(db, 1), "state": self.state.state})
        self._was_loud = loud
        change = self.state.update(now, self.scene_active, loud, speech)
        if change:
            state, reason = change
            if state == STANDBY:
                self.segmenter.reset()
            self.publish(make_event("system", "mic_state", {"state": state, "reason": reason, "module": self.name}))
            logger.info(f"Microphone {state} ({reason})")

    def _submit(self, segment: np.ndarray) -> None:
        duration = round(len(segment) / self.sample_rate, 2)
        with self._lock:
            self.transcriptions += 1
        self.executor.submit(self._transcribe, segment, duration)

    def _get_transcriber(self):
        """Created on the first segment (in the Whisper thread), so start-up stays fast."""
        if self.transcriber is None:
            from modules.llm.whisper_client import create_transcriber

            try:
                self.transcriber = create_transcriber(self.config)
            except Exception as e:  # e.g. no Azure keys: keep standby/active + VAD running, just no transcripts
                logger.error(f"Whisper unavailable ({e}); speech segments will not be transcribed")
                self.transcriber = lambda audio, sample_rate: {"text": "", "language": None}
        return self.transcriber

    def _transcribe(self, segment: np.ndarray, duration: float) -> None:
        try:
            result = self._get_transcriber()(segment, self.sample_rate)
        except Exception as e:
            logger.error(f"Whisper failed for a {duration} s segment: {e}")
            return
        text = (result.get("text") or "").strip()
        if text:
            self.emit("speech", {"text": text, "language": result.get("language"), "duration_sec": duration})

    def _open_source(self) -> None:
        try:
            self.frame_source.open()
            self._source_ok = True
            logger.info("Microphone opened")
        except Exception as e:
            self._source_ok = False
            self._next_open = self.clock() + 10
            logger.error(f"Cannot open microphone: {e}; retrying in 10 s")


def run(event_queue, control_queue, config):
    """
    Module entry point for the orchestrator (own process).

    :param control_queue: "stop" and forwarded scene_active events
    """
    MicController(event_queue, control_queue, config).run()
