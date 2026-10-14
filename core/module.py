"""
Base module interface.

Every module exposes ``run(event_queue, control_queue, config)``. ``BaseModule`` implements that loop:
it calls ``step()`` repeatedly, publishes a heartbeat every ``HEARTBEAT_INTERVAL`` seconds and exits
on a ``"stop"`` message from the control queue.
"""
import queue
import time
from typing import Any, Dict, Optional

from core.bus import CONTROL_STOP, publish
from core.events import Event, make_event
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
             confidence: Optional[float] = None) -> Event:
        """
        Creates an event with this module's source and publishes it.

        :return: The published event
        """
        event = make_event(self.source, type, payload, severity=severity, confidence=confidence)
        self.publish(event)
        return event

    def heartbeat(self) -> None:
        """Publishes a heartbeat event (source='system') if HEARTBEAT_INTERVAL has passed."""
        now = time.monotonic()
        if now - self._last_heartbeat >= HEARTBEAT_INTERVAL:
            self._last_heartbeat = now
            self.publish(make_event("system", "heartbeat", {"module": self.name}))

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
                self.handle_control(message)

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
