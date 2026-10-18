import json
import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from core.bus import create_queue, publish, receive
from core.events import EVENT_TYPES, Event, make_event, utc_timestamp

SCHEMA_KEYS = {"schema_version", "event_id", "timestamp", "source", "type", "severity", "payload", "confidence",
               "received_at", "device_id", "node_id", "location_id"}


def test_make_event_matches_schema():
    event = make_event("gas", "gas_reading", {"readings": {"mq2": 12.5}}, confidence=0.9)
    data = json.loads(event.to_json())

    assert set(data) == SCHEMA_KEYS
    assert uuid.UUID(data["event_id"]).version == 4
    assert data["timestamp"].endswith("Z")
    assert len(data["timestamp"].split(".")[1]) == 4  # "123Z"
    assert data["source"] == "gas"
    assert data["type"] == "gas_reading"
    assert data["severity"] == "info"
    assert data["payload"] == {"readings": {"mq2": 12.5}}
    assert data["confidence"] == 0.9


def test_defaults():
    event = make_event("system", "heartbeat")
    assert event.payload == {}
    assert event.confidence is None
    assert make_event("system", "heartbeat").event_id != event.event_id


def test_json_round_trip():
    event = make_event("camera", "object_detected", {"label": "person", "bbox": [1, 2, 3, 4]},
                       severity="warning", confidence=0.75)
    assert Event.from_json(event.to_json()) == event


def test_utc_timestamp_format():
    moment = datetime(2026, 10, 14, 12, 30, 5, 123456, tzinfo=timezone.utc)
    assert utc_timestamp(moment) == "2026-10-14T12:30:05.123Z"


@pytest.mark.parametrize("kwargs", [
    {"source": "toaster", "type": "alert"},
    {"source": "gas", "type": "unknown_type"},
    {"source": "gas", "type": "alert", "severity": "fatal"},
    {"source": "gas", "type": "alert", "confidence": 1.5},
])
def test_invalid_values_rejected(kwargs):
    with pytest.raises(ValidationError):
        make_event(**kwargs)


def test_invalid_json_rejected():
    data = json.loads(make_event("gas", "alert").to_json())
    for bad in ({"timestamp": "2026-10-14T12:30:05"},
                {"event_id": "not-a-uuid"},
                {"extra_field": 1}):
        with pytest.raises(ValidationError):
            Event.from_json(json.dumps({**data, **bad}))


def test_all_event_types_accepted():
    for event_type in EVENT_TYPES:
        assert make_event("system", event_type).type == event_type


def test_bus_round_trip():
    q = create_queue()
    event = make_event("audio", "speech", {"text": "hello", "speaker": "oleh"})
    publish(q, event)
    assert receive(q, timeout=2) == event
    assert receive(q, timeout=0.05) is None


def test_v2_fields_from_context_and_overrides():
    from core import events

    saved = dict(events._CONTEXT)
    try:
        events._CONTEXT.clear()
        event = make_event("system", "heartbeat")
        assert event.schema_version == 2 and event.location_id == "unassigned"
        assert event.device_id == f"{event.node_id}:system"
        events.set_context(node_id="hub", device_id="hub:gas", location_id="kitchen")
        event = make_event("gas", "gas_reading")
        assert (event.node_id, event.device_id, event.location_id) == ("hub", "hub:gas", "kitchen")
        override = make_event("camera", "motion", device_id="cam-1", location_id="room_b")
        assert (override.node_id, override.device_id, override.location_id) == ("hub", "cam-1", "room_b")
    finally:
        events._CONTEXT.clear()
        events._CONTEXT.update(saved)


def test_v2_fields_are_required_but_v1_logs_are_readable():
    data = json.loads(make_event("gas", "alert").to_json())
    for key in ("device_id", "node_id", "location_id"):
        with pytest.raises(ValidationError):
            Event.from_json(json.dumps({**{k: v for k, v in data.items() if k != key}}))
    v1 = {k: v for k, v in data.items() if k not in ("schema_version", "device_id", "node_id", "location_id")}
    old = Event.from_json(json.dumps(v1))
    assert (old.schema_version, old.device_id, old.node_id, old.location_id) == (1, "unknown", "unknown",
                                                                                 "unassigned")
