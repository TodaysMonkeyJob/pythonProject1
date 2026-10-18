"""
Base class for modules that consume vision shared buffers (F2 actions, F3 identity).

Single camera (no discovery): the buffer named ``vision.shared_buffer``. With ``network.discovery`` (F11) one process
serves every camera: buffers of the enabled, online cameras in the device registry, kept current from forwarded
``device_online`` / ``device_offline`` / ``device_updated`` events, served by a ``FairScheduler`` (active cameras at
``fps``, idle ones at ``idle_fps``, round-robin). ``process()`` runs with ``self.current`` = the camera's slot, and
events default to that camera's device and location.

Forwarded bus events (JSON strings on the control queue) become ``on_event()`` calls.
"""
import time
from typing import Any, Dict, Optional

import numpy as np

from core.events import Event
from core.logger import JsonlLogger
from core.module import BaseModule
from modules.vision.scheduler import CameraSlot, FairScheduler, is_active, online_cameras
from modules.vision.shared_frame import SharedFrameBuffer
from system_setup.logger_setup import init_logger

logger = init_logger()


class SharedFrameConsumer(BaseModule):
    def __init__(self, event_queue, control_queue, config: Dict[str, Any], fps: float = 5.0,
                 frame_buffer: Optional[SharedFrameBuffer] = None):
        """
        :param fps: Maximum processing rate per active camera
        :param frame_buffer: Shared buffer (tests); attached by the vision config name when None
        """
        super().__init__(event_queue, control_queue, config)
        self.vision_cfg = config.get("vision", {})
        self.discovery = bool(config.get("network", {}).get("discovery", False)) and frame_buffer is None
        module_cfg = config.get(self.name, {})
        self.scheduler = FairScheduler(fps, module_cfg.get("idle_fps", 1.0))
        self.current: Optional[CameraSlot] = None
        self.event_log: Optional[JsonlLogger] = None
        self._next_attach = 0.0
        if not self.discovery:
            node = self.device_id.split(":")[0]
            slot = CameraSlot(f"{node}:vision", self.location_id, self.vision_cfg.get("shared_buffer", "iot_vision"),
                              buffer=frame_buffer)
            self.scheduler.add(slot)
        if module_cfg.get("log_events", True) and "logging" in config:
            log_cfg = config["logging"]
            self.event_log = JsonlLogger(self.name, log_cfg.get("dir", "logs"), log_cfg.get("max_total_mb", 2048))

    # Backwards compatible single-camera accessors (tests, demo)
    @property
    def frame_buffer(self) -> Optional[SharedFrameBuffer]:
        slot = next(iter(self.scheduler), None)
        return slot.buffer if slot else None

    @property
    def _next_poll(self) -> float:
        return min((s.next_due for s in self.scheduler), default=0.0)

    @_next_poll.setter
    def _next_poll(self, value: float) -> None:
        for slot in self.scheduler:
            slot.next_due = value

    def emit(self, type: str, payload=None, severity: str = "info", confidence=None, device_id=None,
             location_id=None):
        """Publishes on the bus (as the current camera) and writes logs/YYYY-MM-DD/<module>.jsonl."""
        if self.current is not None:
            device_id = device_id or self.current.device_id
            location_id = location_id or self.current.location_id
        event = super().emit(type, payload, severity=severity, confidence=confidence, device_id=device_id,
                             location_id=location_id)
        if self.event_log is not None:
            self.event_log.write(event)
        return event

    def process(self, frame_id: int, timestamp: float, frame: np.ndarray, meta: Dict[str, Any]) -> None:
        """Handles one new shared frame of ``self.current``; subclasses override."""
        raise NotImplementedError

    def on_event(self, event: Event, now: float) -> None:
        """Handles a bus event forwarded by the orchestrator; subclasses override."""

    def on_camera_removed(self, device_id: str) -> None:
        """A camera went offline / was disabled; subclasses drop its per-camera state."""

    def setup(self) -> None:
        if self.discovery:
            self._load_cameras()

    def teardown(self) -> None:
        for slot in self.scheduler:
            if slot.buffer is not None:
                slot.buffer.close()
        if self.event_log is not None:
            self.event_log.close()

    # --- cameras ------------------------------------------------------------------------------------------------

    def _load_cameras(self) -> None:
        from core.devices import DeviceRegistry
        from core.device_manager import buffer_name

        network = self.config.get("network", {})
        try:
            registry = DeviceRegistry(network.get("devices_path", "data/devices.db"), network.get("node_id"))
            for device in online_cameras(registry, registry.node_id):
                self.scheduler.add(CameraSlot(device["device_id"], device["location_id"],
                                              buffer_name(device["device_id"])))
            registry.close()
        except Exception as e:
            logger.warning(f"{self.name}: cannot read the device registry ({e}); waiting for device events")

    def _on_device_event(self, event: Event) -> None:
        from core.device_manager import buffer_name

        p = event.payload
        if p.get("kind") != "camera" or not self.discovery:
            return
        device_id = p["device_id"]
        if event.type == "device_offline" or (event.type == "device_updated" and not p.get("enabled", True)):
            slot = self.scheduler.remove(device_id)
            if slot is not None:
                if slot.buffer is not None:
                    slot.buffer.close()
                self.on_camera_removed(device_id)
        elif p.get("status") == "online" and p.get("enabled", True):
            if device_id in self.scheduler.slots:
                self.scheduler.slots[device_id].location_id = p.get("location_id", self.location_id)
            else:
                self.scheduler.add(CameraSlot(device_id, p.get("location_id", self.location_id),
                                              buffer_name(device_id)))

    def handle_control(self, message: Any) -> None:
        if isinstance(message, str) and message.startswith("{"):
            try:
                event = Event.from_json(message)
            except ValueError as e:
                logger.debug(f"Ignoring malformed forwarded event: {e}")
                return
            if event.type.startswith("device_"):
                self._on_device_event(event)
            self.on_event(event, time.time())
        else:
            super().handle_control(message)

    # --- loop ---------------------------------------------------------------------------------------------------

    def step(self) -> None:
        now = time.monotonic()
        slot = self.scheduler.pick(now)
        if slot is None:
            due = self.scheduler.next_due()
            time.sleep(0.05 if due is None else max(0.001, min(0.05, due - now)))
            return
        if slot.buffer is None and not self._attach(slot):
            slot.next_due = now + 1.0
            return
        frame = slot.buffer.read_raw()
        if frame is None or frame[0] == slot.last_frame_id:
            self.scheduler.served(slot, now)
            return
        slot.last_frame_id = frame[0]
        self.scheduler.served(slot, now, active=is_active(frame[3]))
        self.current = slot
        try:
            self.process(*frame)
        finally:
            self.current = None

    def _attach(self, slot: CameraSlot) -> bool:
        width, height = self.vision_cfg.get("resolution", [640, 480])
        try:
            slot.buffer = SharedFrameBuffer.attach(slot.buffer_name, width, height)
        except FileNotFoundError:
            logger.debug(f"Shared buffer {slot.buffer_name} not available yet")
            return False
        return True
