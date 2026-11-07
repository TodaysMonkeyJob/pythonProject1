"""
Agent interface (B2 / TOD-49). An agent implements ``on_message`` (and optionally ``load`` / ``unload`` / ``tick`` /
``health``); the host calls ``start()`` when it is enabled and ``stop()`` when it is disabled, so heavy models live
only while the agent is on (the same rule as the camera actions mode). ``stop()`` drops the references ``load()``
made; the registry then drops the agent object itself, so nothing keeps the model alive.

Messages the host delivers (``Message.kind``):

* ``event`` — an evidence event inside the agent's data scope (``message.event``);
* ``claim`` — another agent's claim it subscribed to (``message.claim``);
* ``decision_request`` — a decision round is open for ``message.location_id``: answer with the claims that should
  count, within the manifest's ``timeout_s`` (``message.corridor`` = the fuzzy agent's risk claim,
  ``message.board`` = the situation board within the agent's scope).

``on_message`` / ``tick`` return the claims to publish (``self.claim(...)`` builds them) or None.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.framework.bus_adapter import Claim
from agents.framework.manifest import Manifest
from core.events import Event


@dataclass
class Message:
    kind: str                                       # event | claim | decision_request
    event: Optional[Event] = None
    claim: Optional[Claim] = None
    round_id: Optional[str] = None
    location_id: Optional[str] = None
    corridor: Optional[Claim] = None
    board: List[Claim] = field(default_factory=list)

    @property
    def location(self) -> Optional[str]:
        if self.event is not None:
            return self.event.location_id
        if self.claim is not None:
            return self.claim.location_id
        return self.location_id


class Agent:
    """Base class of every agent; the manifest says who it is, ``agents.settings.<name>`` its settings."""

    def __init__(self, manifest: Manifest, config: Optional[Dict[str, Any]] = None,
                 full_config: Optional[Dict[str, Any]] = None):
        """
        :param manifest: the validated ``agent.yaml``
        :param config: manifest ``config`` defaults overridden by ``agents.settings.<name>`` of config.yaml
        :param full_config: the whole merged config (read only — agents never change configuration)
        """
        self.manifest = manifest
        self.config = config or {}
        self.full_config = full_config or {}
        self.loaded = False

    @property
    def name(self) -> str:
        return self.manifest.name

    # --- lifecycle ----------------------------------------------------------------------------------------------
    def load(self) -> None:
        """Load models / open resources (called by ``start``)."""

    def unload(self) -> None:
        """Drop every reference ``load`` made (called by ``stop``)."""

    def start(self) -> None:
        self.load()
        self.loaded = True

    def stop(self) -> None:
        self.unload()
        self.loaded = False

    # --- work ---------------------------------------------------------------------------------------------------
    def on_message(self, message: Message) -> Optional[List[Claim]]:
        """:return: claims to publish (or None)"""
        return None

    def tick(self, now: float) -> Optional[List[Claim]]:
        """Called about every 0.1 s in the host loop (agents with periodic work)."""
        return None

    def health(self) -> Dict[str, Any]:
        """:return: {status: ok | degraded | error, detail}"""
        return {"status": "ok" if self.loaded else "stopped"}

    # --- helpers ------------------------------------------------------------------------------------------------
    def claim(self, claim: str, location_id: str, value: Optional[Dict[str, Any]] = None, confidence: float = 1.0,
              interval: Optional[List[float]] = None, evidence_ids: Optional[List[str]] = None,
              round_id: Optional[str] = None) -> Claim:
        return Claim(agent=self.name, role=self.manifest.role, version=self.manifest.version, location_id=location_id,
                     claim=claim, value=value or {}, confidence=confidence, interval=interval,
                     evidence_ids=list(evidence_ids or []), round_id=round_id)


class ValidationAgent(Agent):
    """The only role that issues the final decision (``decide``); the host calls it when a round closes."""

    def decide(self, corridor: Claim, claims: List[Claim], weights: Dict[str, float],
               confirmations: List[Claim]) -> Dict[str, Any]:
        """
        :param corridor: the fuzzy agent's ``risk`` claim (risk, level, interval [y_l, y_r])
        :param claims: the answers of this round (late / missing agents are not in it)
        :param weights: agent → measured accuracy (unmeasured = 0)
        :param confirmations: live claims of the location's physical channels (video / gas) on the board
        :return: {risk, level, interval, branch, reasons, operator_request (optional)}
        """
        raise NotImplementedError
