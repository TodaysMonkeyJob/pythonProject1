"""
Vision module (F1): camera → motion on every frame → YOLO + ByteTrack only on motion or every
``yolo_interval_sec`` → change events on the bus + latest frames in shared memory.

Events (source "camera"), published only on changes:
  * ``motion``           {"state": "start" | "end", "area_ratio", "boxes"}
  * ``object_detected``  {"track_id", "class", "bbox", "state": "appeared" | "disappeared"}
  * ``scene_active``     {"active": bool} — motion or at least one object; false after ``scene_timeout_sec``
  * ``alert``            {"reason": "camera_lost" | "camera_restored"}

Forwarded ``person_identified`` events put the person's current display name on the annotated frame (a rename on
the dashboard re-sends them, so the overlay follows within a frame).
"""
import time
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from core.logger import JsonlLogger
from core.module import BaseModule
from modules.vision.capture import ReconnectingSource, source_factory
from modules.vision.motion import MotionDetector, MotionResult
from modules.vision.shared_frame import SharedFrameBuffer
from modules.vision.state import MotionState, SceneState, Track, TrackRegistry
from modules.vision.yolo import ObjectDetector
from system_setup.logger_setup import init_logger

logger = init_logger()


class FpsCounter:
    """Counts events and reports the average rate over an interval."""

    def __init__(self):
        self.count = 0
        self.busy_sec = 0.0

    def add(self, duration_sec: float = 0.0) -> None:
        self.count += 1
        self.busy_sec += duration_sec

    def reset(self) -> None:
        self.count, self.busy_sec = 0, 0.0


def annotate(frame: np.ndarray, motion: MotionResult, tracks: List[Track],
             names: Optional[Dict[int, str]] = None) -> np.ndarray:
    """
    :param names: track id → person name from F3 (OpenCV fonts are ASCII-only: other letters show as "?")
    :return: Copy of the frame with motion boxes (grey) and tracked objects (green, persons orange)
    """
    names = names or {}
    out = frame.copy()
    for x1, y1, x2, y2 in motion.boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (160, 160, 160), 1)
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        color = (0, 140, 255) if track.cls == "person" else (0, 200, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        name = f" {names[track.track_id]}" if track.track_id in names else ""
        cv2.putText(out, f"{track.cls} #{track.track_id}{name} {track.confidence:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


class VisionDetector(BaseModule):
    name = "vision"
    source = "camera"

    def __init__(self, event_queue, control_queue, config: Dict[str, Any], frame_source=None,
                 object_detector=None, frame_buffer: Optional[SharedFrameBuffer] = None,
                 clock: Callable[[], float] = time.monotonic):
        """
        :param frame_source: Object with read() → BGR frame | None; built from config when None
        :param object_detector: Object with track(frame) → [Detection]; ObjectDetector from config when None
        :param frame_buffer: Shared buffer; created/attached by config name when None
        :param clock: Monotonic time source (injectable for tests)
        """
        super().__init__(event_queue, control_queue, config)
        cfg = config["vision"]
        self.cfg = cfg
        self.clock = clock
        self.frame_source = frame_source
        self.object_detector = object_detector
        self.frame_buffer = frame_buffer
        self.motion = MotionDetector(cfg.get("motion_min_area", 800), cfg.get("motion_history", 300),
                                     cfg.get("motion_var_threshold", 25), cfg.get("motion_warmup_frames", 30))
        self.motion_state = MotionState(cfg.get("motion_end_sec", 2.0))
        self.registry = TrackRegistry(cfg.get("track_lost_sec", 10.0), cfg.get("reid_iou", 0.5))
        self.scene = SceneState(cfg.get("scene_timeout_sec", 10))
        self.yolo_interval_sec = float(cfg.get("yolo_interval_sec", 3.0))
        self.shared_interval_sec = 1.0 / float(cfg.get("shared_fps", 10))
        self.jpeg_quality = int(cfg.get("jpeg_quality", 80))
        self.stats_interval_sec = float(cfg.get("stats_interval_sec", 30))
        self.event_log: Optional[JsonlLogger] = None
        self.frame_fps, self.yolo_fps = FpsCounter(), FpsCounter()
        self._last_yolo = float("-inf")
        self._last_shared = float("-inf")
        self._last_stats = None
        self._camera_ok: Optional[bool] = None
        self.last_stats: Dict[str, float] = {}
        self.person_names: Dict[int, str] = {}  # track id → display name (forwarded person_identified)

    def heartbeat_payload(self) -> Dict[str, Any]:
        return dict(self.last_stats)

    def handle_control(self, message: Any) -> None:
        if isinstance(message, str) and message.startswith("{"):
            from core.events import Event

            event = Event.from_json(message)
            p = event.payload
            if event.type == "person_identified" and p.get("track_id") is not None \
                    and event.device_id in (self.device_id, "unknown"):  # track ids are per camera
                if p.get("identity") in (None, "not_visible", "unidentified"):
                    self.person_names.pop(p["track_id"], None)
                else:
                    self.person_names[p["track_id"]] = str(p["identity"])
        else:
            super().handle_control(message)

    def setup(self) -> None:
        cfg = self.cfg
        width, height = cfg.get("resolution", [640, 480])
        if self.frame_source is None:
            self.frame_source = ReconnectingSource(source_factory(cfg), cfg.get("reconnect_max_backoff_sec", 30))
        if self.object_detector is None:
            self.object_detector = ObjectDetector(cfg["model"], cfg.get("imgsz", 320),
                                                  cfg.get("confidence_threshold", 0.4), cfg.get("classes"),
                                                  cfg.get("tracker", "bytetrack.yaml"))
            self.object_detector.load()
            from modules.vision.yolo import limit_threads

            limit_threads(cfg.get("threads", 2))  # after loading: importing ultralytics resets torch's threads
        if self.frame_buffer is None and cfg.get("shared_buffer"):
            self.frame_buffer = SharedFrameBuffer.create(cfg["shared_buffer"], width, height)
        if cfg.get("log_events", True):
            log_cfg = self.config.get("logging", {})
            self.event_log = JsonlLogger("camera", log_cfg.get("dir", "logs"), log_cfg.get("max_total_mb", 2048))
        self._last_stats = self.clock()

    def teardown(self) -> None:
        if self.frame_source is not None:
            self.frame_source.close()
        if self.frame_buffer is not None:
            self.frame_buffer.close()
        if self.event_log is not None:
            self.event_log.close()

    def emit(self, type: str, payload=None, severity: str = "info", confidence=None):
        event = super().emit(type, payload, severity=severity, confidence=confidence)
        if self.event_log is not None:
            self.event_log.write(event)
        return event

    def step(self) -> None:
        frame = self.frame_source.read()
        now = self.clock()
        self._update_camera_status(frame is not None)
        if frame is None:
            time.sleep(0.05)
        else:
            self.process_frame(frame, now)
        self._maybe_log_stats(now)

    def process_frame(self, frame: np.ndarray, now: float) -> None:
        """
        Runs motion, (maybe) YOLO, the state machines and the shared buffer update for one frame.
        """
        started = time.perf_counter()
        motion = self.motion.detect(frame)
        self.frame_fps.add(time.perf_counter() - started)

        change = self.motion_state.update(motion.motion, now)
        if change:
            self.emit("motion", {"state": change, "area_ratio": motion.area_ratio, "boxes": motion.boxes})

        if motion.motion or now - self._last_yolo >= self.yolo_interval_sec:
            self._last_yolo = now
            started = time.perf_counter()
            detections = self.object_detector.track(frame)
            self.yolo_fps.add(time.perf_counter() - started)
            appeared, disappeared = self.registry.update(detections, now)
        else:
            appeared, disappeared = [], self.registry.expire(now)

        for track in appeared:
            self._emit_object(track, "appeared")
        for track in disappeared:
            self._emit_object(track, "disappeared")

        scene_change = self.scene.update(self.motion_state.active or self.registry.has_objects, now)
        if scene_change is not None:
            self.emit("scene_active", {"active": scene_change})

        if self.frame_buffer is not None and now - self._last_shared >= self.shared_interval_sec:
            self._last_shared = now
            self._share(frame, motion, now)

    def _emit_object(self, track: Track, state: str) -> None:
        self.emit("object_detected", {"track_id": track.track_id, "class": track.cls, "bbox": track.bbox,
                                      "state": state}, confidence=track.confidence)

    def _share(self, frame: np.ndarray, motion: MotionResult, now: float) -> None:
        visible = self.registry.visible(now, max_age_sec=self.yolo_interval_sec)
        self.person_names = {k: v for k, v in self.person_names.items() if k in self.registry.tracks}
        ok, jpeg = cv2.imencode(".jpg", annotate(frame, motion, visible, self.person_names),
                                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        meta = {
            "persons": [{"track_id": t.track_id, "bbox": t.bbox, "confidence": t.confidence}
                        for t in visible if t.cls == "person"],
            "objects": [{"track_id": t.track_id, "class": t.cls, "bbox": t.bbox} for t in visible],
            "motion": motion.motion,
            "scene_active": self.scene.active,
        }
        try:
            self.frame_buffer.write(jpeg.tobytes(), frame, meta)
        except ValueError as e:
            logger.warning(f"Shared frame not written: {e}")

    def _update_camera_status(self, ok: bool) -> None:
        if ok == self._camera_ok:
            return
        if not ok and self._camera_ok is None:
            return  # still connecting at start-up
        previous, self._camera_ok = self._camera_ok, ok
        if not ok:
            reason = getattr(self.frame_source, "last_error", None)
            self.emit("alert", {"reason": "camera_lost", "error": reason}, severity="warning")
        elif previous is False:
            self.emit("alert", {"reason": "camera_restored"})

    def _maybe_log_stats(self, now: float) -> None:
        elapsed = now - self._last_stats
        if elapsed < self.stats_interval_sec:
            return
        self.last_stats = {
            "frame_fps": round(self.frame_fps.count / elapsed, 2),
            "yolo_fps": round(self.yolo_fps.count / elapsed, 2),
            "motion_ms": round(1000 * self.frame_fps.busy_sec / max(1, self.frame_fps.count), 2),
            "yolo_ms": round(1000 * self.yolo_fps.busy_sec / max(1, self.yolo_fps.count), 2),
            "tracks": len(self.registry.tracks),
        }
        logger.info(f"Vision stats over {elapsed:.0f} s: {self.last_stats}")
        self.frame_fps.reset()
        self.yolo_fps.reset()
        self._last_stats = now


def run(event_queue, control_queue, config):
    """
    Module entry point for the orchestrator (run it in its own process).

    :param event_queue: Bus queue to publish events to
    :param control_queue: Queue with control messages ("stop", ...)
    :param config: Full merged config
    """
    VisionDetector(event_queue, control_queue, config).run()
