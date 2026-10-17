"""
Unified event schema shared by every module.

Each event travels over the bus as a JSON string so the transport (``multiprocessing.Queue`` today,
MQTT later) can be swapped without changing the format.
"""
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
    "sound",
    "gas_reading",
    "alert",
    "llm_decision",
    "heartbeat",
    "scene_active",
    "mic_state",
    "identity_created",     # F9 identity registry (source system): person_id, display_name
    "identity_renamed",
    "identity_merged",
    "identity_linked",
    "identity_deleted",
)
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

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=utc_timestamp)
    source: Source
    type: EventType
    severity: Severity = "info"
    payload: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

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
        :param data: JSON string produced by to_json()
        :return: Validated Event
        """
        return cls.model_validate_json(data)


def make_event(source: str,
               type: str,
               payload: Optional[Dict[str, Any]] = None,
               severity: str = "info",
               confidence: Optional[float] = None) -> Event:
    """
    Creates a validated event with a fresh id and the current UTC timestamp.

    :param source: camera | audio | gas | system | llm
    :param type: One of EVENT_TYPES
    :param payload: Event-specific data (must be JSON-serialisable)
    :param severity: info | warning | critical
    :param confidence: 0..1 or None
    :return: Event
    """
    return Event(source=source, type=type, payload=payload or {}, severity=severity, confidence=confidence)
