"""
Agents module (B2 / TOD-49, own process): hosts every enabled agent (``agents.framework.orchestrator.AgentHost``,
one thread per agent) on the event bus. Publishes ``agent_claim``, ``agent_decision``, ``agent_operator_request`` and
every ``agents.status_interval_sec`` an ``agent_status`` (not logged; dashboard page «Агенти»). Switches arrive as
``agent_control`` events from the dashboard.
"""
import time
from typing import Any, Dict, Optional

from agents.framework.orchestrator import AgentHost
from agents.framework.registry import AgentRegistry
from core.events import Event, make_event
from core.module import BaseModule
from system_setup.logger_setup import init_logger

logger = init_logger()


class AgentsModule(BaseModule):
    name = "agents"
    source = "system"

    def __init__(self, event_queue, control_queue, config: Dict[str, Any], host: Optional[AgentHost] = None):
        super().__init__(event_queue, control_queue, config)
        self.cfg = config.get("agents") or {}
        self.host = host
        self.status_every = float(self.cfg.get("status_interval_sec", 2))
        self._status_at = 0.0

    def emit_event(self, type: str, payload: Dict[str, Any], severity: str, location_id: Optional[str]) -> None:
        self.publish(make_event("system", type, payload, severity=severity, device_id=self.device_id,
                                location_id=location_id or self.location_id))

    def setup(self) -> None:
        if self.host is None:
            self.host = AgentHost(AgentRegistry(self.config), self.emit_event, threaded=True,
                                  board_max_age_s=float(self.cfg.get("board_max_age_sec", 120)),
                                  decides=bool(self.cfg.get("decides", False)))
        self.host.start()
        on = self.host.registry.enabled()
        logger.info(f"agents on: {', '.join(on) or 'none'}; found {len(self.host.registry.manifests)}"
                    + (f"; invalid: {', '.join(self.host.registry.errors)}" if self.host.registry.errors else ""))

    def teardown(self) -> None:
        if self.host is not None:
            self.host.stop()

    def handle_control(self, message: Any) -> None:
        if not (isinstance(message, str) and message.startswith("{")):
            super().handle_control(message)
            return
        self.host.observe(Event.from_json(message))

    def heartbeat_payload(self) -> Dict[str, Any]:
        return {"agents_on": self.host.registry.enabled() if self.host else []}

    def step(self) -> None:
        self.host.tick()
        now = time.monotonic()
        if now - self._status_at >= self.status_every:
            self._status_at = now
            self.emit_event("agent_status", self.host.status(), "info", None)
        time.sleep(0.05)
