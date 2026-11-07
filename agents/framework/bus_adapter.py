"""
Typed claims, data-scope filtering and the per-location situation board (B2 / TOD-49).

Agents never call each other: everything they say is a ``Claim`` on the event bus (event ``agent_claim``)::

    {"claim_id": "…", "agent": "fuzzy", "role": "fuzzy", "version": "1.0.0", "location_id": "kitchen",
     "ts": "2026-11-07T10:00:00.000Z", "claim": "risk", "value": {"risk": 0.82, "level": "critical"},
     "confidence": 1.0, "interval": [0.78, 1.0], "evidence_ids": ["…"], "round_id": null}

``claim`` is the claim type (``risk``, ``action``, ``echo`` …), ``value`` its content. ``ScopeFilter`` decides who
receives an event or a claim: an agent gets an event type only when it is both subscribed and inside its manifest's
``data_scope.events``, a claim only when it subscribed to ``claim:<type>`` and the claim's role is inside
``data_scope.claims`` (the audio agent never sees face data, whatever its code asks for). ``SituationBoard`` keeps
the latest claim of every (agent, claim type) per location — the blackboard — and shows each agent only its scope.
"""
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.framework.manifest import HOST_EVENTS, Manifest
from core.events import Event, utc_timestamp


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    agent: str
    role: str
    version: str
    location_id: str
    ts: str = Field(default_factory=utc_timestamp)
    claim: str
    value: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    interval: Optional[List[float]] = None
    evidence_ids: List[str] = Field(default_factory=list)
    round_id: Optional[str] = None
    at: float = Field(default_factory=time.time, exclude=True)    # local receive time (board expiry)

    @field_validator("interval")
    @classmethod
    def ordered(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is not None and (len(value) != 2 or value[0] > value[1] + 1e-9):
            raise ValueError(f"interval must be [low, high], got {value}")
        return value

    def payload(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_event(cls, event: Event) -> "Claim":
        return cls(**{k: v for k, v in event.payload.items() if k in cls.model_fields and k != "at"})


class ScopeFilter:
    """Who may receive what: subscriptions ∩ data scope (the scope always wins)."""

    def __init__(self, manifests: Iterable[Manifest]):
        self.manifests = {m.name: m for m in manifests}

    def event_receivers(self, event_type: str, names: Iterable[str]) -> List[str]:
        if event_type in HOST_EVENTS:
            return []
        return [n for n in names if event_type in self.manifests[n].event_subscriptions
                and self.manifests[n].data_scope.allows_event(event_type)]

    def claim_receivers(self, claim: Claim, names: Iterable[str]) -> List[str]:
        return [n for n in names if n != claim.agent and claim.claim in self.manifests[n].claim_subscriptions
                and self.manifests[n].data_scope.allows_claims_of(claim.role)]

    def may_publish(self, claim: Claim) -> bool:
        manifest = self.manifests.get(claim.agent)
        return manifest is not None and claim.role == manifest.role and claim.claim in manifest.publishes


class SituationBoard:
    """Per location: the latest claim of every (agent, claim type), older than ``max_age_s`` forgotten."""

    def __init__(self, max_age_s: float = 120.0, clock=time.time):
        self.max_age, self.clock = max_age_s, clock
        self.claims: Dict[str, Dict[tuple, Claim]] = {}

    def update(self, claim: Claim) -> None:
        self.claims.setdefault(claim.location_id, {})[(claim.agent, claim.claim)] = claim

    def forget_agent(self, agent: str) -> None:
        """A disabled agent's claims stop counting at once."""
        for claims in self.claims.values():
            for key in [k for k in claims if k[0] == agent]:
                del claims[key]

    def view(self, location_id: str, manifest: Optional[Manifest] = None) -> List[Claim]:
        """:return: live claims of the location, only the roles inside ``manifest``'s scope (None = all)"""
        now = self.clock()
        claims = self.claims.get(location_id, {})
        for key in [k for k, c in claims.items() if now - c.at > self.max_age]:
            del claims[key]
        return [c for c in claims.values() if manifest is None or c.agent == manifest.name
                or manifest.data_scope.allows_claims_of(c.role)]

    def locations(self) -> List[str]:
        return sorted(self.claims)
