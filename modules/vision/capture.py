"""
Frame sources: Pi Camera (picamera2), OpenCV camera or video file, plus a reconnecting wrapper.

Every source returns BGR ``uint8`` frames of shape (height, width, 3), or None when no frame is available.
"""
import time
from typing import Any, Callable, Dict, Optional, Union

import cv2
import numpy as np

from system_setup.logger_setup import init_logger

logger = init_logger()


class FrameSource:
    """Interface of a frame source."""

    def open(self) -> None:
        raise NotImplementedError

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class OpenCVSource(FrameSource):
    """
    USB camera (int index), video file (path) or network stream (rtsp:// / http:// URL, F11) through
    cv2.VideoCapture (OpenCV's bundled FFmpeg — nothing to install).
    """

    def __init__(self, device: Union[int, str] = 0, width: int = 640, height: int = 480, fps: float = 15,
                 realtime: bool = True, loop: bool = True):
        """
        :param device: Camera index or video file path
        :param width: Requested / output frame width
        :param height: Requested / output frame height
        :param fps: Requested camera FPS; playback rate for files when realtime is set
        :param realtime: Video file: sleep between frames to play at `fps`
        :param loop: Video file: rewind at the end
        """
        self.device = int(device) if isinstance(device, str) and device.isdigit() else device
        self.is_stream = isinstance(self.device, str) and "://" in self.device
        self.is_file = isinstance(self.device, str) and not self.is_stream
        self.width, self.height, self.fps = width, height, fps
        self.realtime, self.loop = realtime, loop
        self.cap = None
        self._next_frame_at = 0.0

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source {self.device!r}")
        if self.is_stream:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # latest frame, not a growing backlog
        elif not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self) -> Optional[np.ndarray]:
        if self.is_file and self.realtime:
            delay = self._next_frame_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next_frame_at = time.monotonic() + 1.0 / self.fps
        ok, frame = self.cap.read()
        if not ok and self.is_file and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            return None
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class Picamera2Source(FrameSource):
    """
    Raspberry Pi camera via picamera2 (install with apt: python3-picamera2).
    """

    def __init__(self, width: int = 640, height: int = 480, fps: float = 15):
        self.width, self.height, self.fps = width, height, fps
        self.camera = None

    def open(self) -> None:
        from picamera2 import Picamera2

        self.camera = Picamera2()
        # libcamera "RGB888" is BGR byte order in memory — what OpenCV expects
        config = self.camera.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": float(self.fps)})
        self.camera.configure(config)
        self.camera.start()

    def read(self) -> Optional[np.ndarray]:
        return self.camera.capture_array("main")

    def close(self) -> None:
        if self.camera is not None:
            self.camera.stop()
            self.camera.close()
            self.camera = None


class IngestSource(FrameSource):
    """
    Frames pushed by a remote node (a phone's browser, F11): the dashboard decodes them into a shared memory buffer,
    this source reads each new one. ``read()`` waits up to ``timeout_sec`` for the next frame.
    """

    def __init__(self, buffer_name: str, width: int = 640, height: int = 480, timeout_sec: float = 1.0):
        self.buffer_name, self.width, self.height, self.timeout_sec = buffer_name, width, height, timeout_sec
        self.buffer = None
        self.last_id = 0

    def open(self) -> None:
        from modules.vision.shared_frame import SharedFrameBuffer

        try:
            self.buffer = SharedFrameBuffer.attach(self.buffer_name, self.width, self.height)
        except FileNotFoundError as e:
            raise RuntimeError(f"Remote node not streaming yet ({self.buffer_name})") from e

    def read(self) -> Optional[np.ndarray]:
        deadline = time.monotonic() + self.timeout_sec
        while time.monotonic() < deadline:
            frame = self.buffer.read_raw()
            if frame is not None and frame[0] != self.last_id:
                self.last_id = frame[0]
                return frame[2]
            time.sleep(0.01)
        return None

    def close(self) -> None:
        if self.buffer is not None:
            self.buffer.close()
            self.buffer = None


def picamera2_available() -> bool:
    try:
        import picamera2  # noqa: F401
    except ImportError:
        return False
    return True


def source_factory(vision_config: Dict[str, Any]) -> Callable[[], FrameSource]:
    """
    :param vision_config: vision section of the config
    :return: Callable creating a fresh (unopened) source
    """
    camera = vision_config.get("camera", "auto")
    width, height = vision_config.get("resolution", [640, 480])
    fps = vision_config.get("fps", 15)
    device = vision_config.get("device", 0)
    if camera == "ingest":  # frames from a remote node (browser phone)
        return lambda: IngestSource(str(device), width, height)
    use_picamera = camera == "picamera2" or (camera == "auto" and isinstance(device, int) and picamera2_available())
    if use_picamera:
        return lambda: Picamera2Source(width, height, fps)
    if camera not in ("auto", "opencv"):
        raise ValueError(f"Unknown vision.camera {camera!r}")
    return lambda: OpenCVSource(device, width, height, fps,
                                realtime=vision_config.get("replay_realtime", True),
                                loop=vision_config.get("replay_loop", True))


class ReconnectingSource(FrameSource):
    """
    Opens the source lazily and reopens it with exponential backoff after errors or missing frames.
    ``read()`` never raises; it returns None while the camera is unavailable.
    """

    def __init__(self, factory: Callable[[], FrameSource], max_backoff_sec: float = 30.0,
                 max_missed_frames: int = 30, clock: Callable[[], float] = time.monotonic):
        """
        :param factory: Creates a new source
        :param max_backoff_sec: Upper bound for the delay between reconnect attempts
        :param max_missed_frames: Consecutive empty reads before the source is considered lost
        :param clock: Monotonic time source
        """
        self.factory = factory
        self.max_backoff_sec = max_backoff_sec
        self.max_missed_frames = max_missed_frames
        self.clock = clock
        self.source: Optional[FrameSource] = None
        self.connected = False
        self.last_error: Optional[str] = None
        self._backoff = 1.0
        self._next_attempt = 0.0
        self._missed = 0

    def open(self) -> None:
        self._try_open()

    def read(self) -> Optional[np.ndarray]:
        if not self.connected:
            if self.clock() < self._next_attempt or not self._try_open():
                return None
        try:
            frame = self.source.read()
        except Exception as e:
            self._fail(f"read failed: {e}")
            return None
        if frame is None:
            self._missed += 1
            if self._missed >= self.max_missed_frames:
                self._fail("no frames")
            return None
        self._missed = 0
        return frame

    def close(self) -> None:
        if self.source is not None:
            try:
                self.source.close()
            except Exception as e:
                logger.debug(f"Error closing camera: {e}")
        self.source = None
        self.connected = False

    def _try_open(self) -> bool:
        try:
            self.source = self.factory()
            self.source.open()
        except Exception as e:
            self._fail(f"open failed: {e}")
            return False
        self.connected = True
        self.last_error = None
        self._backoff = 1.0
        self._missed = 0
        logger.info("Camera connected")
        return True

    def _fail(self, reason: str) -> None:
        if self.connected or self.last_error is None:
            logger.warning(f"Camera {reason}; retrying in {self._backoff:.0f} s")
        self.close()
        self.last_error = reason
        self._next_attempt = self.clock() + self._backoff
        self._backoff = min(self._backoff * 2, self.max_backoff_sec)
