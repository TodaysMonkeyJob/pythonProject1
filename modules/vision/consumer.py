"""
Base class for modules that consume the vision shared buffer (F2 actions, F3 identity).

Polls the latest raw frame + metadata at ``fps``, skips frames already processed, and turns forwarded bus
events (JSON strings on the control queue) into ``on_event()`` calls.
"""
import time
from typing import Any, Dict, Optional

import numpy as np

from core.events import Event
from core.logger import JsonlLogger
from core.module import BaseModule
from modules.vision.shared_frame import SharedFrameBuffer
from system_setup.logger_setup import init_logger

logger = init_logger()


class SharedFrameConsumer(BaseModule):
    def __init__(self, event_queue, control_queue, config: Dict[str, Any], fps: float = 5.0,
                 frame_buffer: Optional[SharedFrameBuffer] = None):
        """
        :param fps: Maximum processing rate
        :param frame_buffer: Shared buffer; attached by the vision config name when None
        """
        super().__init__(event_queue, control_queue, config)
        self.vision_cfg = config.get("vision", {})
        self.frame_buffer = frame_buffer
        self.interval_sec = 1.0 / float(fps)
        self._last_frame_id = 0
        self._next_poll = 0.0
        self.event_log: Optional[JsonlLogger] = None
        module_cfg = config.get(self.name, {})
        if module_cfg.get("log_events", True) and "logging" in config:
            log_cfg = config["logging"]
            self.event_log = JsonlLogger(self.name, log_cfg.get("dir", "logs"), log_cfg.get("max_total_mb", 2048))

    def emit(self, type: str, payload=None, severity: str = "info", confidence=None):
        """Publishes on the bus and writes the event to logs/YYYY-MM-DD/<module>.jsonl."""
        event = super().emit(type, payload, severity=severity, confidence=confidence)
        if self.event_log is not None:
            self.event_log.write(event)
        return event

    def process(self, frame_id: int, timestamp: float, frame: np.ndarray, meta: Dict[str, Any]) -> None:
        """Handles one new shared frame; subclasses override."""
        raise NotImplementedError

    def on_event(self, event: Event, now: float) -> None:
        """Handles a bus event forwarded by the orchestrator; subclasses override."""

    def teardown(self) -> None:
        if self.frame_buffer is not None:
            self.frame_buffer.close()
        if self.event_log is not None:
            self.event_log.close()

    def handle_control(self, message: Any) -> None:
        if isinstance(message, str) and message.startswith("{"):
            try:
                self.on_event(Event.from_json(message), time.time())
            except ValueError as e:
                logger.debug(f"Ignoring malformed forwarded event: {e}")
        else:
            super().handle_control(message)

    def step(self) -> None:
        now = time.monotonic()
        if now < self._next_poll:
            time.sleep(min(0.05, self._next_poll - now))
            return
        self._next_poll = now + self.interval_sec
        if self.frame_buffer is None and not self._attach():
            self._next_poll = now + 1.0
            return
        frame = self.frame_buffer.read_raw()
        if frame is None or frame[0] == self._last_frame_id:
            return
        self._last_frame_id = frame[0]
        self.process(*frame)

    def _attach(self) -> bool:
        width, height = self.vision_cfg.get("resolution", [640, 480])
        try:
            self.frame_buffer = SharedFrameBuffer.attach(self.vision_cfg.get("shared_buffer", "iot_vision"),
                                                         width, height)
        except FileNotFoundError:
            logger.debug("Vision shared buffer not available yet")
            return False
        return True
