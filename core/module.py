"""
Base module interface.

Every module exposes ``run(event_queue, control_queue, config)``. ``BaseModule`` implements that loop:
it calls ``step()`` repeatedly, publishes a heartbeat every ``HEARTBEAT_INTERVAL`` seconds and exits
on a ``"stop"`` message from the control queue.

Device identity (F11): a module started for one device (a camera / microphone found by discovery, or the gas sensor)
gets ``config["_device"]`` = {"device_id", "location_id", …} and ``config["_module"]`` (e.g. ``vision@cam-3f2a…``)
from the orchestrator; its events carry that device and location. Software modules use ``<node>:<module>`` and the
hub location. A forwarded ``device_updated`` event for its device moves the module to the new location at once.
"""
import queue
import time
from typing import Any, Dict, Optional

from core.bus import CONTROL_STOP, publish
from core.events import UNASSIGNED, Event, context, make_event
from system_setup.logger_setup import init_logger

logger = init_logger()

HEARTBEAT_INTERVAL = 5.0


class BaseModule:
    """
    Subclasses set ``name`` and ``source`` and override ``setup()``, ``step()`` and ``teardown()``.
    ``step()`` should return within a fraction of a second so heartbeats and stop requests stay timely.
    """
    name = "base"
    source = "system"

    def __init__(self, event_queue, control_queue, config: Dict[str, Any]):
        """
        :param event_queue: Bus queue the module publishes to
        :param control_queue: Queue the orchestrator sends commands to
        :param config: Full merged config
        """
        self.event_queue = event_queue
        self.control_queue = control_queue
        self.config = config
        self.running = False
        self._last_heartbeat = 0.0
        self.name = config.get("_module") or self.name
        device = config.get("_device") or {}
        network = config.get("network", {})
        node = context()["node_id"]
        self.device_id = device.get("device_id") or f"{node}:{type(self).name}"
        self.location_id = device.get("location_id") or network.get("hub_location") or UNASSIGNED

    def setup(self) -> None:
        """Acquire resources (sensors, models). Called once before the loop."""

    def step(self) -> None:
        """One iteration of work."""
        time.sleep(0.1)

    def teardown(self) -> None:
        """Release resources. Always called, also after an exception."""

    def handle_control(self, message: Any) -> None:
        """
        Handles a control message other than "stop".

        :param message: Message from the control queue
        """
        logger.debug(f"{self.name}: ignoring control message {message!r}")

    def publish(self, event: Event) -> None:
        """
        :param event: Event to put on the bus
        """
        publish(self.event_queue, event)

    def emit(self, type: str, payload: Optional[Dict[str, Any]] = None, severity: str = "info",
             confidence: Optional[float] = None, device_id: Optional[str] = None,
             location_id: Optional[str] = None) -> Event:
        """
        Creates an event with this module's source (and device / location, unless given) and publishes it.

        :return: The published event
        """
        event = make_event(self.source, type, payload, severity=severity, confidence=confidence,
                           device_id=device_id or self.device_id, location_id=location_id or self.location_id)
        self.publish(event)
        return event

    def heartbeat_payload(self) -> Dict[str, Any]:
        """Extra status shown on the dashboard (e.g. FPS, microphone state); override in modules."""
        return {}

    def heartbeat(self) -> None:
        """Publishes a heartbeat event (source='system') if HEARTBEAT_INTERVAL has passed."""
        now = time.monotonic()
        if now - self._last_heartbeat >= HEARTBEAT_INTERVAL:
            self._last_heartbeat = now
            self.publish(make_event("system", "heartbeat", {"module": self.name, **self.heartbeat_payload()},
                                    device_id=self.device_id, location_id=self.location_id))

    def poll_control(self) -> None:
        """Drains the control queue without blocking."""
        while True:
            try:
                message = self.control_queue.get_nowait()
            except queue.Empty:
                return
            if message == CONTROL_STOP:
                self.running = False
            else:
                if isinstance(message, str) and '"device_updated"' in message:
                    self.on_device_updated(Event.from_json(message))
                self.handle_control(message)

    def on_device_updated(self, event: Event) -> None:
        """A device was moved to another location / renamed on the dashboard; follow it if it is ours."""
        if event.payload.get("device_id") == self.device_id and event.payload.get("location_id"):
            if event.payload["location_id"] != self.location_id:
                logger.info(f"{self.name}: location {self.location_id} → {event.payload['location_id']}")
            self.location_id = event.payload["location_id"]

    def run(self) -> None:
        """Main loop: setup → (heartbeat, control, step)* → teardown."""
        self.running = True
        self.setup()
        try:
            while self.running:
                self.heartbeat()
                self.poll_control()
                if not self.running:
                    break
                self.step()
        finally:
            self.teardown()
