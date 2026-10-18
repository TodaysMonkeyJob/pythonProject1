"""
Unified event schema shared by every module.

Each event travels over the bus as a JSON string so the transport (``multiprocessing.Queue`` today,
MQTT later) can be swapped without changing the format.

Schema v2 (F11 / TOD-22): every event says where it comes from — ``device_id`` (camera / mic / sensor, or
``<node>:<module>`` for software modules), ``node_id`` (the machine) and ``location_id`` (room / territory,
``unassigned`` until the user picks one); ``received_at`` = when the hub got an event from another node.
``make_event`` fills them from the process context (``set_context``) unless
given; v1 log lines (without them) are still readable.
"""
import json
import socket
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

Source = Literal["camera", "audio", "gas", "system", "llm"]
Severity = Literal["info", "warning", "critical"]

EVENT_TYPES = (
    "object_detected",
    "motion",
    "action",
    "person_identified",
    "speech",
    "scene_text",               # F16: text seen in the frame (sign / screen; OCR or captioning) — weak evidence
    "sound",
    "gas_reading",
    "alert",
    "llm_decision",
    "deescalation_rejected",    # F17: the risk corridor raised the LLM's level (attack success rate metrics)
    "heartbeat",
    "scene_active",
    "mic_state",
    "identity_created",     # F9 identity registry (source system): person_id, display_name
    "identity_renamed",
    "identity_merged",
    "identity_linked",
    "identity_deleted",
    "identity_unmerged",    # F14: a merge undone (person_id = target, source_id, event_id)
    "duplicate_candidate",  # F14 dedup job: a pair queued for review (candidate_id, person_a, person_b, score)
    "device_online",        # F11 device registry (source system): device_id, kind, name, location_id
    "device_offline",
    "device_updated",
    "location_state_changed",   # F12 (source system): location_id, from, to, reason, event_ids
    "location_summary",         # F12 live summary per location (dashboard only, not logged)
    "notification",             # F12: id, location_id, level, title, body
    "notification_acked",
)
SCHEMA_VERSION = 2
UNASSIGNED = "unassigned"
_CONTEXT: Dict[str, str] = {}


def default_node_id() -> str:
    return socket.gethostname().split(".")[0] or "node"


def set_context(node_id: Optional[str] = None, device_id: Optional[str] = None,
                location_id: Optional[str] = None) -> None:
    """Process-wide defaults for ``make_event`` (set by the orchestrator in every module process)."""
    for key, value in (("node_id", node_id), ("device_id", device_id), ("location_id", location_id)):
        if value:
            _CONTEXT[key] = value


def context() -> Dict[str, str]:
    node = _CONTEXT.get("node_id") or default_node_id()
    return {"node_id": node, "device_id": _CONTEXT.get("device_id") or f"{node}:system",
            "location_id": _CONTEXT.get("location_id") or UNASSIGNED}
EventType = Literal[EVENT_TYPES]


def utc_timestamp(moment: Optional[datetime] = None) -> str:
    """
    Formats a moment as ISO 8601 UTC with millisecond precision, e.g. ``2026-10-14T12:30:05.123Z``.

    :param moment: Timezone-aware datetime; current time when None
    :return: Timestamp string
    """
    moment = (moment or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Event(BaseModel):
    """
    A single event in the unified schema.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SCHEMA_VERSION
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=utc_timestamp)
    device_id: str = Field(min_length=1, max_length=128)
    node_id: str = Field(min_length=1, max_length=64)
    location_id: str = Field(min_length=1, max_length=64)
    source: Source
    type: EventType
    severity: Severity = "info"
    payload: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    received_at: Optional[str] = None   # set by the hub for events from other nodes (their clock may differ)

    @field_validator("event_id")
    @classmethod
    def _check_event_id(cls, value: str) -> str:
        if uuid.UUID(value).version != 4:
            raise ValueError("event_id must be a UUID4")
        return value

    @field_validator("timestamp")
    @classmethod
    def _check_timestamp(cls, value: str) -> str:
        if not value.endswith("Z"):
            raise ValueError("timestamp must be UTC and end with 'Z'")
        datetime.fromisoformat(value[:-1] + "+00:00")
        return value

    def to_json(self) -> str:
        """
        :return: Compact JSON string of the event
        """
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> "Event":
        """
        :param data: JSON string produced by to_json() — schema v2, or a v1 log line (no device / node / location:
                     read as ``unknown`` / ``unassigned``)
        :return: Validated Event
        """
        if '"schema_version"' in data:
            return cls.model_validate_json(data)
        raw = json.loads(data)
        if not isinstance(raw, dict):
            raise ValueError("event JSON must be an object")
        raw.update(schema_version=1, device_id=raw.get("device_id") or "unknown",
                   node_id=raw.get("node_id") or "unknown", location_id=raw.get("location_id") or UNASSIGNED)
        return cls.model_validate(raw)


def make_event(source: str,
               type: str,
               payload: Optional[Dict[str, Any]] = None,
               severity: str = "info",
               confidence: Optional[float] = None,
               device_id: Optional[str] = None,
               node_id: Optional[str] = None,
               location_id: Optional[str] = None) -> Event:
    """
    Creates a validated event with a fresh id and the current UTC timestamp; device / node / location default to the
    process context.

    :param source: camera | audio | gas | system | llm
    :param type: One of EVENT_TYPES
    :param payload: Event-specific data (must be JSON-serialisable)
    :param severity: info | warning | critical
    :param confidence: 0..1 or None
    :return: Event
    """
    ctx = context()
    return Event(source=source, type=type, payload=payload or {}, severity=severity, confidence=confidence,
                 device_id=device_id or ctx["device_id"], node_id=node_id or ctx["node_id"],
                 location_id=location_id or ctx["location_id"])
