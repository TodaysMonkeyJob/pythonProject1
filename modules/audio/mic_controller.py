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

Speaker-attributed transcripts (F9 / F10, ``speaker_id``): each chunk is diarized (``speaker.SpeakerAnalyzer``:
windows → ECAPA → clusters → registry), Whisper segments are aligned to the speaker turns (``attribution.align``)
and published as one ``speech`` event per segment — who (``person_id``, ``speaker_name``, ``speaker_score``,
``speaker_status`` known / unknown / uncertain), what (``text``), when (``start`` / ``end``), where (``location_id``,
``device_id``) — and written to ``logs/<date>/transcripts.jsonl`` + ``transcripts.txt``. Forwarded
``object_detected`` / ``person_identified`` events tell who is in view, so a new voice is linked to the face of the
person who is alone in view while speaking (``core.cooccurrence``).
"""
import collections
import math
import queue
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from core.cooccurrence import PresenceTracker, VoiceFaceLinker
from core.events import Event, make_event
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


class SocketSource:
    """
    Microphone of a remote node (a phone's browser, F11): the dashboard forwards its 16 kHz int16 PCM as datagrams to
    a unix socket this source binds; read() re-cuts them into ``frame_samples`` frames.
    """

    def __init__(self, path: str, frame_samples: int = 480):
        self.path, self.frame_samples = path, frame_samples
        self.sock = None
        self.pending = np.zeros(0, np.int16)

    def open(self) -> None:
        import os
        import socket as sockets

        if os.path.exists(self.path):
            os.remove(self.path)  # stale socket of a previous run
        self.sock = sockets.socket(sockets.AF_UNIX, sockets.SOCK_DGRAM)
        self.sock.bind(self.path)
        self.sock.setblocking(False)

    def read(self) -> List[np.ndarray]:
        chunks = []
        while True:
            try:
                chunks.append(np.frombuffer(self.sock.recv(65536), np.int16))
            except BlockingIOError:
                break
        if chunks:
            self.pending = np.concatenate([self.pending, *chunks])
        frames = []
        while len(self.pending) >= self.frame_samples:
            frames.append(self.pending[:self.frame_samples].copy())
            self.pending = self.pending[self.frame_samples:]
        return frames

    def close(self) -> None:
        import os

        if self.sock is not None:
            self.sock.close()
            self.sock = None
            if os.path.exists(self.path):
                os.remove(self.path)


def audio_socket_path(device_id: str) -> str:
    """Where the dashboard sends a remote node's microphone audio (short: unix socket paths are limited)."""
    import tempfile

    return f"{tempfile.gettempdir().rstrip('/')}/iotguard_{device_id.split('-')[-1][:12]}.sock"


def webrtc_vad(aggressiveness: int, sample_rate: int) -> Callable[[np.ndarray], bool]:
    import webrtcvad

    vad = webrtcvad.Vad(aggressiveness)
    return lambda frame: vad.is_speech(frame.tobytes(), sample_rate)


class MicController(BaseModule):
    name = "audio"
    source = "audio"

    def __init__(self, event_queue, control_queue, config: Dict[str, Any], frame_source=None,
                 vad: Optional[Callable[[np.ndarray], bool]] = None, transcriber=None,
                 clock: Callable[[], float] = time.monotonic, speaker=None,
                 wall_clock: Callable[[], float] = time.time):
        """
        :param frame_source: Object with open() / read() -> [int16 frame] / close(); SoundDeviceSource by default
        :param vad: frame -> is speech; webrtcvad by default
        :param transcriber: (audio, sample_rate) -> {"text", "language"}; from modules.llm.whisper_client by default
        :param clock: Monotonic time source (tests)
        :param speaker: ``speaker.SpeakerIdentifier``; created from the config on the first segment by default,
                        False = no speaker identification
        :param wall_clock: Wall-clock time source, shared with the presence events (tests)
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
        self.wall_clock = wall_clock
        identity = config.get("identity", {})
        if not config.get("_device"):  # single microphone without discovery: the configured location
            self.location_id = identity.get("location_id") or "home"
        self.device_id = cfg.get("device_id") or (config.get("_device") or {}).get("device_id") \
            or f"mic-{socket.gethostname().split('.')[0]}"
        self.presence = PresenceTracker(self.location_id, clock=wall_clock)
        self.transcript_log = None
        self.speaker = speaker if speaker is not None or cfg.get("speaker_id", False) else False
        self.linker: Optional[VoiceFaceLinker] = None
        if self.speaker:
            self._attach_linker(self.speaker.registry)
        self.scene_by_camera: Dict[str, bool] = {}   # cameras in this microphone's location → scene active
        self.transcriptions = 0
        self.executor: Optional[ThreadPoolExecutor] = None
        self._source_ok = False
        self._next_open = 0.0
        self._was_loud = False
        self._last_sound_event = float("-inf")
        self._lock = threading.Lock()
        from modules.trust.health import HealthPublisher, registry_config

        health_cfg = registry_config(config)    # F23 (TOD-34): microphone self-diagnostics → device_health
        self.health_cfg = (health_cfg or {}).get("mic") if health_cfg is not None else None
        self.health = None                      # MicHealth, created when the source is open (actual sample rate)
        self.health_publisher = HealthPublisher(self.emit, "mic", (health_cfg or {}).get("health", {}).get(
            "refresh_sec", 300)) if health_cfg is not None else None

    @property
    def scene_active(self) -> bool:
        """A camera in the same location sees something (F11: scene_active is per location)."""
        return any(self.scene_by_camera.values())

    @scene_active.setter
    def scene_active(self, value: bool) -> None:  # tests / single-camera callers
        self.scene_by_camera = {"*": bool(value)}

    def _same_location(self, location_id: str) -> bool:
        """Only cameras in this microphone's location — unless one of them has no location yet (then: everywhere)."""
        return location_id in ("unassigned", "unknown") or self.location_id in ("unassigned", location_id)

    def on_device_updated(self, event: Event) -> None:
        super().on_device_updated(event)
        if event.payload.get("device_id") == self.device_id:
            self.presence.location_id = self.location_id
            self.scene_by_camera.clear()  # re-learnt from the next scene_active of the new location

    def heartbeat_payload(self) -> Dict[str, Any]:
        return {"mic_state": self.state.state, "microphone_ok": self._source_ok}

    def setup(self) -> None:
        device = self.cfg.get("device")
        if self.frame_source is None and isinstance(device, str) and device.startswith("socket:"):
            self.frame_source = SocketSource(device[len("socket:"):], self.sample_rate * self.frame_ms // 1000)
        if self.frame_source is None:
            self.frame_source = SoundDeviceSource(self.sample_rate, self.sample_rate * self.frame_ms // 1000,
                                                  self.cfg.get("device"))
        if self.vad is None:
            self.vad = webrtc_vad(int(self.cfg.get("vad_aggressiveness", 2)), self.sample_rate)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self.executor.submit(self._warm_up)  # in the Whisper thread: start-up and scene handling stay fast
        self._open_source()

    def teardown(self) -> None:
        if self.frame_source is not None:
            try:
                self.frame_source.close()
            except Exception as e:
                logger.debug(f"Error closing microphone: {e}")
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        if self.transcript_log is not None:
            self.transcript_log.close()

    def handle_control(self, message: Any) -> None:
        """
        Forwarded ``scene_active`` events switch the microphone on immediately; ``object_detected`` /
        ``person_identified`` update who is in view (for linking voices to faces).
        """
        if not (isinstance(message, str) and message.startswith("{")):
            super().handle_control(message)
            return
        event = Event.from_json(message)
        if event.type == "scene_active":
            if self._same_location(event.location_id):
                self.scene_by_camera[event.device_id] = bool(event.payload.get("active"))
            else:
                self.scene_by_camera.pop(event.device_id, None)   # that camera is elsewhere (or was moved)
            self._update_state(False, False)
        elif event.type in ("object_detected", "person_identified"):
            with self._lock:
                self.presence.on_event(event)
        # identity_* events: the registry is polled before each speaker identification

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
        if self.health_cfg is not None:
            if self.health is None:
                from modules.trust.health import MicHealth

                actual = getattr(self.frame_source, "actual_rate", None)
                self.health = MicHealth(self.health_cfg, self.sample_rate, actual)
            if (failing := self.health.observe(frame, self.clock())) is not None:
                self.health_publisher.publish(failing, self.clock())
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
            self.publish(make_event("system", "mic_state", {"state": state, "reason": reason, "module": self.name},
                                    device_id=self.device_id, location_id=self.location_id))
            logger.info(f"Microphone {state} ({reason})")

    def _submit(self, segment: np.ndarray) -> None:
        duration = round(len(segment) / self.sample_rate, 2)
        with self._lock:
            self.transcriptions += 1
        self.executor.submit(self._transcribe, segment, duration, self.wall_clock(), self.transcriptions)

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

    def _get_speaker(self):
        """SpeakerAnalyzer on the shared identity registry, created in the Whisper thread; None when off."""
        if self.speaker is None:
            from core.identity import registry_from_config
            from modules.audio.speaker import analyzer_from_config

            try:
                registry = registry_from_config(self.config.get("identity", {}), publish=self.publish)
                self.speaker = analyzer_from_config(registry, self.cfg)
                self._attach_linker(registry)
            except Exception as e:  # e.g. speechbrain missing: transcripts keep working, just without a speaker
                logger.error(f"Speaker identification unavailable ({e}); transcripts will have no speaker")
                self.speaker = False
        return self.speaker or None

    def _attach_linker(self, registry) -> None:
        identity = self.config.get("identity", {})
        self.linker = VoiceFaceLinker(registry, self.presence, int(identity.get("link_min_cooccurrences", 3)),
                                      float(identity.get("link_presence_ratio", 0.8)))

    def _warm_up(self) -> None:
        try:
            warm_up = getattr(self._get_transcriber(), "warm_up", None)
            if warm_up:
                warm_up()
        except Exception as e:
            logger.error(f"Whisper warm-up failed: {e}")
        if self.speaker is not False:
            self._get_speaker()

    def _transcribe(self, segment: np.ndarray, duration: float, ended_at: Optional[float] = None,
                    number: Optional[int] = None) -> None:
        """
        :param number: Submission number; a segment with ``whisper_max_backlog`` newer ones already waiting is skipped,
                       so a slow device (local Whisper on the Pi) stays near real time instead of falling behind
        """
        from modules.audio.attribution import Segment, align, iso, whisper_segments

        backlog = int(self.cfg.get("whisper_max_backlog", 2))
        if number is not None and backlog > 0 and self.transcriptions - number >= backlog:
            logger.warning(f"Whisper is {self.transcriptions - number} segments behind: "
                           f"skipping a {duration} s segment")
            return
        try:
            result = self._get_transcriber()(segment, self.sample_rate)
        except Exception as e:
            logger.error(f"Whisper failed for a {duration} s segment: {e}")
            return
        segments = whisper_segments(result, duration)
        if not segments:
            if self.health_publisher is not None:      # F23: the VAD sent a segment Whisper found no speech in
                self.emit("trust_evidence", {"channel": "vad", "kind": "mic", "verdict": "refute",
                                             "reason": f"Whisper не почув мови у {duration:.1f} с сегменті"})
            return
        chunk_start = (ended_at or self.wall_clock()) - duration
        diarization = self._diarize(segment, duration)
        if diarization is None:
            attributed = [Segment(s.start, s.end, s.text) for s in segments]
        else:
            attributed = align(segments, diarization.turns)
            self._link_voices(diarization, chunk_start)
        base = {"language": result.get("language"), "backend": result.get("backend"),
                "device_id": self.device_id, "location_id": self.location_id}
        for piece in attributed:
            payload = {"text": piece.text, **base, "duration_sec": round(piece.end - piece.start, 2),
                       "start": iso(chunk_start + piece.start), "end": iso(chunk_start + piece.end)}
            if piece.confidence is not None:
                payload["asr_confidence"] = piece.confidence
            if piece.overlap:
                payload["speaker_overlap"] = True
            if diarization is not None:
                who = diarization.speaker(piece.cluster)
                payload.update(person_id=who.person_id, speaker_name=who.name, speaker_score=who.score,
                               speaker_status=who.status)
            event = self.emit("speech", payload)
            self._log_transcript(event)

    def _diarize(self, segment: np.ndarray, duration: float):
        """:return: ``speaker.Diarization`` or None (speaker identification off / failed)"""
        analyzer = self._get_speaker() if self.speaker is not False else None
        if analyzer is None:
            return None
        try:
            started = time.monotonic()
            diarization = analyzer.analyze(segment, self.sample_rate)
            logger.debug(f"Diarized {duration:.1f} s in {time.monotonic() - started:.2f} s: "
                         f"{len(diarization.speakers)} speaker(s), {len(diarization.turns)} turn(s)")
            return diarization
        except Exception as e:
            logger.error(f"Speaker identification failed: {e}")
            return None

    def _link_voices(self, diarization, chunk_start: float) -> None:
        """Feeds the voice ↔ face linker once per speaker of the chunk (its longest turn)."""
        if self.linker is None:
            return
        for cluster, who in diarization.speakers.items():
            turns = [t for t in diarization.turns if t.cluster == cluster]
            if who.person_id is None or not turns:
                continue
            longest = max(turns, key=lambda t: t.end - t.start)
            with self._lock:
                linked = self.linker.on_segment(who.person_id, self.location_id, chunk_start + longest.start,
                                                chunk_start + longest.end)
            if linked is not None:
                who.person_id, who.name = linked.payload["person_id"], linked.payload["display_name"]
                who.status = "known" if self.speaker.registry.is_known(who.person_id) else "unknown"
                logger.info(f"Voice linked to {who.name} (alone in view while speaking)")

    def _log_transcript(self, event) -> None:
        if not self.cfg.get("transcript_log", True):
            return
        if self.transcript_log is None:
            from modules.audio.attribution import TranscriptLog

            log_cfg = self.config.get("logging", {})
            registry = getattr(self.speaker, "registry", None) if self.speaker else None
            self.transcript_log = TranscriptLog(log_cfg.get("dir", "logs"), log_cfg.get("max_total_mb", 2048),
                                                resolve_name=(lambda pid, name: registry.name(pid, name))
                                                if registry is not None else None)
        try:
            self.transcript_log.write(event)
        except OSError as e:
            logger.error(f"Transcript log not written: {e}")

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
