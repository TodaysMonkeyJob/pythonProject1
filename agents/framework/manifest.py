"""
Agent manifest (B2 / TOD-49): ``agents/<name>/agent.yaml`` describes one agent — what it is, what it may read, what it
may say and how long the system waits for it. The core never changes when an agent is added: the registry finds the
folder, validates the manifest and imports the code only when the agent is enabled.

```yaml
name: echo                      # = folder name, unique
role: other                     # fuzzy | video | audio | gas | validation | explanation | rule_suggestion | other
version: 1.0.0                  # logged with every claim (reproducibility)
description: Repeats the evidence it receives as claims (template)
entry: agent:EchoAgent          # module (inside the folder, or a dotted path) : class
subscribes: [speech, "claim:risk"]   # event types and claim types ("claim:<type>") delivered to on_message
publishes: [echo]               # claim types it may publish; anything else is dropped
backend: {kind: rules}          # rules | local_model | cloud_model (+ models: [...]); cloud agents never start in
                                # mode autonomous
resources: {ram_mb: 5, gpu: false}
data_scope:                     # what the bus lets through — never wider than this, whatever ``subscribes`` says
  events: [speech]              # event types
  claims: [fuzzy]               # roles whose claims it may read (situation board + delivery), "*" = all
timeout_s: 1.0                  # answer to a decision request within this, or it is ignored for that decision
enabled_default: false
contributes: true               # asked for its claims at every decision round (default: every role except fuzzy /
                                # validation)
required: false                 # cannot be switched off (the fuzzy agent: the corridor is the safety baseline)
eval_file: eval.yaml            # measured accuracy → its weight in validation (no file = unmeasured = weight 0)
```
"""
import os
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.events import EVENT_TYPES

ROLES = ("fuzzy", "video", "audio", "gas", "validation", "explanation", "rule_suggestion", "other")
NON_CONTRIBUTING = ("fuzzy", "validation")      # they open / close decision rounds instead of answering them
CLAIM_PREFIX = "claim:"
# Events the agents host consumes itself; agents never receive them, whatever their manifest says
HOST_EVENTS = ("agent_claim", "agent_decision", "agent_status", "agent_control", "agent_operator_request",
               "heartbeat")


class ManifestError(ValueError):
    """An agent.yaml that cannot be used (the agent is listed with the error and never loaded)."""


class Backend(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["rules", "local_model", "cloud_model"] = "rules"
    models: List[str] = Field(default_factory=list)


class Resources(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ram_mb: int = 0
    gpu: bool = False


class DataScope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: List[str] = Field(default_factory=list)
    claims: List[str] = Field(default_factory=list)       # roles, or "*"

    @field_validator("events")
    @classmethod
    def known_events(cls, value: List[str]) -> List[str]:
        unknown = [t for t in value if t not in EVENT_TYPES]
        if unknown:
            raise ValueError(f"unknown event types {unknown}")
        hidden = [t for t in value if t in HOST_EVENTS]
        if hidden:
            raise ValueError(f"{hidden} are the agents host's own events, never delivered to agents")
        return value

    @field_validator("claims")
    @classmethod
    def known_roles(cls, value: List[str]) -> List[str]:
        unknown = [r for r in value if r != "*" and r not in ROLES]
        if unknown:
            raise ValueError(f"unknown roles {unknown} (one of {ROLES} or '*')")
        return value

    def allows_event(self, event_type: str) -> bool:
        return event_type in self.events

    def allows_claims_of(self, role: str) -> bool:
        return "*" in self.claims or role in self.claims


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    role: Literal[ROLES]  # type: ignore[valid-type]
    version: str
    description: str = ""
    entry: str = Field(pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
    subscribes: List[str] = Field(default_factory=list)
    publishes: List[str] = Field(default_factory=list)
    backend: Backend = Field(default_factory=Backend)
    resources: Resources = Field(default_factory=Resources)
    data_scope: DataScope = Field(default_factory=DataScope)
    timeout_s: float = Field(default=1.0, gt=0, le=60)
    enabled_default: bool = False
    contributes: Optional[bool] = None
    required: bool = False
    eval_file: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)  # defaults of its settings (agents.settings.<name>)
    path: Optional[str] = None                             # the folder (set by the loader, not in the file)

    @model_validator(mode="after")
    def consistent(self) -> "Manifest":
        events = [t for t in self.subscribes if not t.startswith(CLAIM_PREFIX)]
        outside = [t for t in events if not self.data_scope.allows_event(t)]
        if outside:
            raise ValueError(f"subscribes to {outside} outside its data_scope.events")
        if self.contributes is None:
            self.contributes = self.role not in NON_CONTRIBUTING
        return self

    @property
    def event_subscriptions(self) -> List[str]:
        return [t for t in self.subscribes if not t.startswith(CLAIM_PREFIX)]

    @property
    def claim_subscriptions(self) -> List[str]:
        return [t[len(CLAIM_PREFIX):] for t in self.subscribes if t.startswith(CLAIM_PREFIX)]

    def evaluation(self) -> Dict[str, Any]:
        """
        :return: the measured accuracy of ``eval_file`` ({accuracy, n, source, date, status}) — the agent's weight in
                 validation; no file → ``{"status": "unmeasured", "accuracy": None}`` (weight 0, never a typed value)
        """
        if not self.eval_file:
            return {"status": "unmeasured", "accuracy": None}
        path = self.eval_file if os.path.isabs(self.eval_file) else os.path.join(self.path or ".", self.eval_file)
        if not os.path.isfile(path):
            return {"status": "unmeasured", "accuracy": None, "missing_file": self.eval_file}
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        accuracy = data.get("accuracy")
        if accuracy is None or not 0.0 <= float(accuracy) <= 1.0 or not data.get("n"):
            return {"status": "unmeasured", "accuracy": None, "invalid_file": self.eval_file}
        return {"status": "measured", "accuracy": float(accuracy), "n": int(data["n"]),
                **{k: data[k] for k in ("source", "date", "scenarios") if k in data}}


def load_manifest(path: str) -> Manifest:
    """
    :param path: an ``agent.yaml``
    :return: the validated manifest; its ``name`` must equal the folder name
    :raises ManifestError: unreadable or invalid file
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        raise ManifestError(f"{path}: {e}") from e
    if not isinstance(data, dict):
        raise ManifestError(f"{path}: not a mapping")
    folder = os.path.dirname(os.path.abspath(path))
    try:
        manifest = Manifest(**data, path=folder)
    except ValueError as e:
        raise ManifestError(f"{path}: {e}") from e
    if manifest.name != os.path.basename(folder):
        raise ManifestError(f"{path}: name {manifest.name!r} differs from its folder {os.path.basename(folder)!r}")
    return manifest
