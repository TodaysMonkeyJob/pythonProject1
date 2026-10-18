"""
One WebSocket for all camera tiles of a page (F13 / TOD-24): browsers allow only ~6 parallel HTTP/1.1 connections per
host, so one MJPEG stream per tile stalls beyond ~5 cameras.

The page sends ``{"subscribe": [device_id, …], "focus": device_id | null}``; the server sends binary messages
``[4 bytes header length][header JSON {device_id, ts, frame_id, size}][JPEG]`` — the subscribed (visible) cameras as
small 320 px frames at ``dashboard.grid_fps``, the enlarged one at full size at ``dashboard.focus_fps``. Only new frames
are sent. Small frames are made once per camera frame and shared by every connection (``FrameCache``), so ten open
tabs cost one resize. Sources: the per-camera shared buffers (local, browser, RTSP cameras) and the previews of edge
(Raspberry Pi) cameras.
"""
import asyncio
import json
import time
from typing import Any, Callable, Dict, Optional, Tuple

SMALL_WIDTH = 320


def pack(header: Dict[str, Any], jpeg: bytes) -> bytes:
    raw = json.dumps(header, separators=(",", ":")).encode()
    return len(raw).to_bytes(4, "big") + raw + jpeg


def unpack(message: bytes) -> Tuple[Dict[str, Any], bytes]:
    size = int.from_bytes(message[:4], "big")
    return json.loads(message[4:4 + size]), message[4 + size:]


def shrink(jpeg: bytes, width: int = SMALL_WIDTH, quality: int = 70) -> bytes:
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[1] <= width:
        return jpeg
    height = round(image.shape[0] * width / image.shape[1])
    small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


class FrameCache:
    """Latest frame of every camera, read from its shared buffer (or edge preview), small version made once."""

    def __init__(self, resolution=(640, 480), previews: Optional[Dict[str, bytes]] = None,
                 buffer_name: Optional[Callable[[str], str]] = None, clock: Callable[[], float] = time.monotonic):
        from core.device_manager import buffer_name as default_name

        self.width, self.height = resolution
        self.previews = previews if previews is not None else {}
        self.buffer_name = buffer_name or default_name
        self.clock = clock
        self.buffers: Dict[str, Any] = {}
        self.small: Dict[str, Tuple[int, bytes]] = {}       # device → (frame id, small jpeg)
        self.preview_ids: Dict[str, Tuple[int, bytes]] = {}

    def latest(self, device_id: str) -> Optional[Tuple[int, float, bytes]]:
        """:return: (frame_id, ts, full jpeg) or None (camera not streaming)"""
        if device_id in self.previews:  # edge camera: a preview every ~2 s
            jpeg = self.previews[device_id]
            last = self.preview_ids.get(device_id)
            if last is None or last[1] is not jpeg:
                self.preview_ids[device_id] = ((last[0] + 1) if last else 1, jpeg)
            return self.preview_ids[device_id][0], time.time(), jpeg
        buffer = self.buffers.get(device_id)
        if buffer is None:
            from modules.vision.shared_frame import SharedFrameBuffer

            try:
                buffer = SharedFrameBuffer.attach(self.buffer_name(device_id), self.width, self.height)
            except FileNotFoundError:
                return None
            self.buffers[device_id] = buffer
        return buffer.read_annotated()

    def frame(self, device_id: str, size: str = "small") -> Optional[Tuple[int, float, bytes]]:
        latest = self.latest(device_id)
        if latest is None or size == "full":
            return latest
        frame_id, ts, jpeg = latest
        cached = self.small.get(device_id)
        if cached is None or cached[0] != frame_id:
            cached = (frame_id, shrink(jpeg))
            self.small[device_id] = cached
        return frame_id, ts, cached[1]

    def drop(self, device_id: str) -> None:
        buffer = self.buffers.pop(device_id, None)
        if buffer is not None:
            buffer.close()
        self.small.pop(device_id, None)

    def close(self) -> None:
        for device_id in list(self.buffers):
            self.drop(device_id)


async def stream(websocket, cache: FrameCache, grid_fps: float = 4.0, focus_fps: float = 10.0,
                 max_messages: Optional[int] = None) -> None:
    """Serves one page: reads subscription messages and sends new frames of the subscribed cameras."""
    subscribed: list = []
    focus: Optional[str] = None
    sent: Dict[Tuple[str, str], int] = {}          # (device, size) → last frame id sent
    due: Dict[Tuple[str, str], float] = {}
    count = 0

    async def receive() -> None:
        nonlocal subscribed, focus
        while True:
            message = json.loads(await websocket.receive_text())
            subscribed = [str(d) for d in message.get("subscribe", [])][:64]
            focus = message.get("focus")
            sent.clear()  # a re-subscribed tile gets its current frame at once

    reader = asyncio.create_task(receive())
    try:
        while max_messages is None or count < max_messages:
            if reader.done():
                reader.result()  # re-raises the disconnect
                return
            now = time.monotonic()
            wanted = [(d, "small") for d in subscribed if d != focus] + ([(focus, "full")] if focus else [])
            for device_id, size in wanted:
                key = (device_id, size)
                if now < due.get(key, 0.0):
                    continue
                due[key] = now + 1.0 / (focus_fps if size == "full" else grid_fps)
                frame = cache.frame(device_id, size)
                if frame is None or sent.get(key) == frame[0]:
                    continue
                sent[key] = frame[0]
                await websocket.send_bytes(pack({"device_id": device_id, "ts": frame[1], "frame_id": frame[0],
                                                 "size": size}, frame[2]))
                count += 1
            await asyncio.sleep(0.02)
    finally:
        reader.cancel()
