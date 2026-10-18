"""F12 locations: registry, per-location state machine, notifications, per-location snapshots and GPT, dashboard API."""
import json
import queue
import sqlite3
import time
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from core.events import Event, make_event
from core.locations import LocationRegistry
from modules.locations.state import ATTENTION, DANGER, OFFLINE, SAFE, LocationStateEngine
from modules.notify.service import NotificationService

AUTH = ("admin", "s3cret")


@pytest.fixture
def registry(tmp_path):
    bus = []
    reg = LocationRegistry(str(tmp_path / "devices.db"), publish=bus.append)
    reg.bus = bus
    yield reg
    reg.close()


# --- registry -----------------------------------------------------------------------------------------------------

def test_location_crud_order_and_delete_moves_devices(registry):
    warehouse = registry.create("Warehouse", icon="📦", rules={"gas_thresholds": {"acetone": "3"}, "junk": 1})
    office = registry.create("Office")
    assert (warehouse["location_id"], warehouse["order"], office["order"]) == ("warehouse", 1, 2)
    assert warehouse["rules"] == {"gas_thresholds": {"acetone": 3.0}}            # unknown keys dropped
    assert [loc["location_id"] for loc in registry.list()] == ["warehouse", "office", "unassigned"]
    registry.reorder(["office", "warehouse"])
    assert [loc["location_id"] for loc in registry.list()][:2] == ["office", "warehouse"]
    with pytest.raises(ValueError):
        registry.create("warehouse")                                             # same id
    camera, _ = registry.devices.seen("camera", "local", "cam", "hw:1", location_id="warehouse")
    moved = registry.delete("warehouse")
    assert moved == [camera["device_id"]] and registry.devices.get(camera["device_id"])["location_id"] == "unassigned"
    assert registry.bus[-1].type == "device_updated"                             # modules follow the move
    with pytest.raises(ValueError):
        registry.delete("unassigned")
    assert registry.update("office", name="Main office", rules={"unknown_person_after": "20:00"})["rules"] == \
        {"unknown_person_after": "20:00"}


def test_seed_file_and_older_database(tmp_path):
    path = str(tmp_path / "devices.db")
    old = sqlite3.connect(path)          # a devices.db made by F11, before order / icon / rules existed
    old.executescript("CREATE TABLE locations (location_id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at REAL);"
                      "INSERT INTO locations VALUES ('unassigned', 'Unassigned', 0);")
    old.close()
    seed = tmp_path / "locations.yaml"
    seed.write_text("locations:\n  - id: warehouse\n    name: Warehouse\n    order: 1\n    icon: 📦\n"
                    "    rules: {unknown_person_after: '20:00'}\n  - id: office\n    name: Office\n    order: 2\n",
                    encoding="utf-8")
    reg = LocationRegistry(path, seed_file=str(seed))
    assert [(loc["location_id"], loc["icon"]) for loc in reg.list()] == [("warehouse", "📦"), ("office", ""),
                                                                         ("unassigned", "")]
    assert reg.rules("warehouse") == {"unknown_person_after": "20:00"} and reg.name("office") == "Office"
    reg.close()


# --- state machine ------------------------------------------------------------------------------------------------

class World:
    """A fake clock, device list and rules for the engine."""

    def __init__(self, rules=None, hour=12):
        self.now = 1_000_000.0
        self.devices = [{"location_id": "warehouse", "status": "online", "enabled": True},
                        {"location_id": "office", "status": "online", "enabled": True}]
        self.rules = rules or {}
        self.engine = LocationStateEngine(lambda loc: loc.title(), lambda loc: self.rules.get(loc, {}),
                                          lambda: self.devices, {"cooldown_min": 2, "attention_window_min": 5},
                                          local_now=lambda: datetime(2026, 10, 18, hour, 0))

    def event(self, location, type="alert", severity="info", source="gas", **payload):
        event = make_event(source, type, payload, severity=severity, device_id=f"dev-{location}",
                           location_id=location)
        return self.engine.on_event(event, self.now)

    def state(self, location):
        return self.engine.live[location].state


def test_gas_alert_turns_only_its_location_red_and_cools_down():
    world = World()
    changes = world.event("warehouse", severity="critical", reason="gas_threshold", gas="acetone")
    assert [(c["location_id"], c["from"], c["to"]) for c in changes] == [("warehouse", SAFE, DANGER)]
    assert "gas_threshold acetone" in changes[0]["reason"] and changes[0]["event_ids"]
    assert world.engine.evaluate(world.now) == [] and world.state("office") == SAFE
    world.now += 60
    world.event("warehouse", severity="critical", reason="gas_threshold", gas="acetone")   # still going on
    assert world.engine.live["warehouse"].alerts.__len__() == 1                 # one alert, refreshed
    world.now += 119
    assert world.engine.evaluate(world.now) == []                                # < cooldown after the last one
    world.now += 2
    (change,) = world.engine.evaluate(world.now)
    assert (change["from"], change["to"]) == (DANGER, ATTENTION)                 # not acknowledged yet
    (change,) = world.engine.acknowledge("warehouse", world.now)
    assert change["to"] == SAFE and world.engine.summary("warehouse", world.now)["active_alerts"] == 0


def test_acknowledge_clears_danger_at_once():
    world = World()
    world.event("warehouse", type="action", source="camera", severity="critical", action="lying", track_id=3)
    assert world.state("warehouse") == DANGER
    (change,) = world.engine.acknowledge("warehouse", world.now)
    assert (change["from"], change["to"]) == (DANGER, SAFE)


def test_warnings_gpt_risk_and_offline():
    world = World()
    world.event("office", type="person_identified", source="camera", severity="warning", identity="unknown_4",
                track_id=1)
    assert world.state("office") == ATTENTION
    world.now += 301
    assert world.engine.evaluate(world.now)[0]["to"] == SAFE                     # warning window over
    world.event("office", type="llm_decision", source="llm", severity="warning", risk_level="medium",
                summary="незнайомець")
    assert world.state("office") == ATTENTION
    world.event("office", type="llm_decision", source="llm", severity="critical", risk_level="high",
                summary="ймовірне падіння")
    assert world.state("office") == DANGER and "GPT" in world.engine.live["office"].reason
    world.now += 601
    assert world.engine.evaluate(world.now)[0]["to"] == SAFE                     # GPT risk expired
    world.devices = [d if d["location_id"] != "warehouse" else {**d, "status": "offline"} for d in world.devices]
    changes = world.engine.evaluate(world.now)
    assert [(c["location_id"], c["to"]) for c in changes] == [("warehouse", OFFLINE)]


def test_location_rules():
    world = World(rules={"warehouse": {"gas_thresholds": {"acetone": 3.0}, "unknown_person_after": "20:00"}})
    world.event("warehouse", type="gas_reading", concentration_pct=2.5, gas="acetone", trend="rising")
    assert world.state("warehouse") == SAFE
    world.event("warehouse", type="gas_reading", concentration_pct=3.4, gas="acetone", trend="rising")
    assert world.state("warehouse") == DANGER and "≥ 3.0%" in world.engine.live["warehouse"].reason
    world.engine.acknowledge("warehouse", world.now)
    world.event("warehouse", type="person_identified", source="camera", severity="warning", identity="unknown_4",
                track_id=1)
    assert world.state("warehouse") == SAFE                                      # noon: a visitor
    night = World(rules=world.rules, hour=23)
    night.event("warehouse", type="person_identified", source="camera", severity="warning", identity="unknown_4",
                track_id=1)
    assert night.state("warehouse") == ATTENTION


def test_summary_and_hub_events_ignored():
    world = World()
    world.event("warehouse", type="object_detected", source="camera", track_id=4, state="appeared",
                **{"class": "person"})
    world.event("warehouse", type="person_identified", source="camera", track_id=4, identity="unknown_4",
                person_id="p4")
    world.event("warehouse", type="action", source="camera", track_id=4, action="walking")
    world.event("warehouse", type="speech", source="audio", text="Close the door", speaker_name="Oleh")
    world.event("warehouse", type="gas_reading", gas="clean_air", concentration_pct=0, trend="stable")
    world.event("hub", type="alert", source="system", severity="critical", reason="module_restarted")
    summary = world.engine.summary("warehouse", world.now)
    (person,) = summary["people"]
    assert (person["person_id"], person["name"], person["action"], person["device_id"]) == \
        ("p4", "unknown_4", "walking", "dev-warehouse") and person["since"].endswith("Z")
    assert summary["last_speech"]["name"] == "Oleh" and summary["gas"]["main"] == "clean_air"
    assert summary["devices"] == {"online": 1, "total": 1} and summary["state"] == SAFE
    assert "hub" not in world.engine.live


# --- notifications ------------------------------------------------------------------------------------------------

def test_notifications_rate_limit_telegram_and_ack(tmp_path):
    clock = [1000.0]
    sent, bus = [], []
    service = NotificationService(str(tmp_path / "n.db"), bus.append, repeat_min=5, telegram=sent.append,
                                  clock=lambda: clock[0])
    note = service.notify("warehouse", "danger", "Небезпека: Warehouse", "gas", "danger:gas", ["e1"])
    assert note and note["channels"] == ["dashboard", "browser", "telegram"]
    assert service.notify("warehouse", "danger", "x", "gas", "danger:gas") is None          # rate limited
    assert service.notify("office", "danger", "x", "gas", "danger:gas") is not None          # other location
    assert service.notify("warehouse", "attention", "x", "y", "attention:y")["channels"] == ["dashboard", "browser"]
    clock[0] += 301
    assert service.notify("warehouse", "danger", "x", "gas", "danger:gas") is not None       # after repeat_min
    time.sleep(0.1)
    assert len(sent) == 3 and sent[0].startswith("🔴 Небезпека: Warehouse")                # danger only
    assert bus[0].type == "notification" and bus[0].location_id == "warehouse"
    acked = service.ack(note["id"], "oleh")
    assert acked["acknowledged_by"] == "oleh" and acked["acknowledged_at"]
    assert len(service.list(unacked=True)) == 3 and service.ack_location("warehouse") == 2
    service.close()


def test_telegram_sender_uses_the_bot_api(monkeypatch):
    from modules.notify import service as svc

    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": true}'

    monkeypatch.setattr(svc.urllib.request, "urlopen", lambda url, data, timeout: calls.append((url, data))
                        or Response())
    assert svc.telegram_sender("TOKEN", "42")("🔴 Небезпека") is True
    assert calls[0][0] == "https://api.telegram.org/botTOKEN/sendMessage" and b"chat_id=42" in calls[0][1]


# --- module -------------------------------------------------------------------------------------------------------

def test_module_publishes_transitions_notifications_and_summaries(tmp_path):
    from modules.locations.module import LocationsModule

    world = World()
    clock = [world.now]
    service = NotificationService(str(tmp_path / "n.db"), repeat_min=5, clock=lambda: clock[0])
    module = LocationsModule(queue.Queue(), queue.Queue(), {"locations": {"summary_interval_sec": 0}},
                             engine=world.engine, notifications=service, clock=lambda: clock[0])
    service.publish = module.publish
    gas = make_event("gas", "alert", {"reason": "gas_threshold", "gas": "acetone"}, severity="critical",
                     device_id="gas-1", location_id="warehouse")
    module.handle_control(gas.to_json())
    module.step()
    events = [Event.from_json(module.event_queue.get_nowait()) for _ in range(module.event_queue.qsize())]
    kinds = [(e.type, e.location_id) for e in events]
    assert ("location_state_changed", "warehouse") in kinds and kinds.count(("notification", "warehouse")) == 1
    changed = next(e for e in events if e.type == "location_state_changed")
    assert changed.payload["to"] == DANGER and changed.severity == "critical"
    summaries = {e.location_id: e.payload for e in events if e.type == "location_summary"}
    assert summaries["warehouse"]["state"] == DANGER and summaries["office"]["state"] == SAFE
    module.handle_control(make_event("system", "notification_acked", {"location_id": "warehouse"},
                                     location_id="warehouse").to_json())
    back = [Event.from_json(module.event_queue.get_nowait()) for _ in range(module.event_queue.qsize())]
    assert any(e.type == "location_state_changed" and e.payload["to"] == SAFE for e in back)
    service.close()


# --- per-location snapshots and GPT -------------------------------------------------------------------------------

def test_critical_in_one_location_snapshots_only_that_location(tmp_path):
    from modules.aggregator.aggregator import Aggregator

    clock = [1000.0]
    locations = LocationRegistry(str(tmp_path / "devices.db"))
    locations.create("Warehouse", rules={"gas_thresholds": {"acetone": 3.0}})
    locations.create("Office")
    config = {"aggregator": {"export_dir": str(tmp_path / "exports"), "incidents": {"enabled": False},
                             "snapshot_interval_sec": 60, "global_snapshot_min": 10}}
    aggregator = Aggregator(config, clock=lambda: clock[0], local_now=lambda: datetime(2026, 10, 18, 12, 0),
                            locations=locations)
    for location in ("warehouse", "office"):
        aggregator.on_event(make_event("camera", "object_detected", {"track_id": 1, "class": "person",
                                                                     "state": "appeared"},
                                       device_id=f"cam-{location}", location_id=location))
    aggregator.on_event(make_event("system", "alert", {"reason": "module_restarted"}, severity="critical",
                                   location_id="hub"))                         # a hub event is no place
    snapshot = aggregator.on_event(make_event("gas", "alert", {"reason": "gas_threshold", "gas": "acetone"},
                                              severity="critical", device_id="gas-1", location_id="warehouse"))
    assert snapshot["location_id"] == "warehouse" and snapshot["location_name"] == "Warehouse"
    assert snapshot["trigger"] == "critical_event" and len(snapshot["people"]) == 1
    row = aggregator.outbox.pending()[0]
    assert row["digest"].startswith("Warehouse | ") and "rules: acetone ≥ 3.0% is dangerous here" in row["digest"]
    assert aggregator.tick() is None                                            # office: nothing critical
    clock[0] += 61
    assert aggregator.tick()["location_id"] == "office"                         # its regular snapshot
    assert aggregator.tick() is None                    # warehouse reported (critical) and has nothing new
    clock[0] += 600
    aggregator.tick(), aggregator.tick()
    site = next(r for r in aggregator.outbox.pending(limit=20) if r["trigger"] == "global")
    assert site["digest"].startswith("site overview") and "Warehouse: 1 people" in site["digest"]
    assert "Office: 1 people" in site["digest"]
    aggregator.close()


def test_gpt_throttle_is_per_location_and_decisions_carry_it(tmp_path):
    from modules.aggregator.outbox import Outbox
    from modules.llm.worker import LLMWorker
    from tests.test_llm import FakeGPT

    box = Outbox(str(tmp_path / "outbox.db"))
    for n, location in enumerate(("warehouse", "office", "warehouse", "office")):
        box.add({"snapshot_id": f"s{n}", "trigger": "interval", "location_id": location}, "interval",
                created_at=n + 1, digest=f"{location} {n}")
    box.add({"snapshot_id": "g", "trigger": "global"}, "global", created_at=9, digest="site overview")
    gpt = FakeGPT()
    worker = LLMWorker(queue.Queue(), queue.Queue(), {"llm": {"min_llm_interval_sec": 300, "poll_sec": 0}},
                       gpt=gpt, outbox=box, clock=lambda: 1000.0)
    worker.step()
    assert sorted(gpt.digests) == ["office 3", "site overview", "warehouse 2"]   # newest per location + site
    decisions = [Event.from_json(worker.event_queue.get_nowait()) for _ in range(3)]
    assert {d.location_id for d in decisions if d.payload.get("location_id")} == {"warehouse", "office"}
    assert all(d.payload.get("location_id", d.location_id) == d.location_id for d in decisions)


# --- dashboard API ------------------------------------------------------------------------------------------------

def test_dashboard_locations_and_notifications_api(tmp_path):
    from dashboard.server import DashboardState, create_app

    config = {"dashboard": {"user": AUTH[0], "password": AUTH[1]}, "logging": {"dir": str(tmp_path / "logs")},
              "aggregator": {"export_dir": str(tmp_path / "exports")}, "vision": {},
              "network": {"devices_path": str(tmp_path / "devices.db"), "node_port": 0}}
    state = DashboardState(config)
    bus = []
    state.publish = bus.append
    state.locations = LocationRegistry(str(tmp_path / "devices.db"), publish=bus.append)
    state.notifications = NotificationService(str(tmp_path / "n.db"))
    http = TestClient(create_app(state, config))
    created = http.post("/api/locations", json={"name": "Warehouse", "icon": "📦"}, auth=AUTH).json()
    assert created["location_id"] == "warehouse" and created["state"] == "safe"
    assert http.post("/api/locations", json={"name": "Warehouse"}, auth=AUTH).status_code == 409
    http.patch("/api/locations/warehouse", json={"rules": {"unknown_person_after": "20:00"}}, auth=AUTH)
    summary = {"location_id": "warehouse", "name": "Warehouse", "state": "danger", "reason": "gas", "people": [],
               "people_count": 0, "devices": {"online": 1, "total": 1}, "active_alerts": 1}
    state.on_event(make_event("system", "location_summary", summary, location_id="warehouse").to_json())
    listing = http.get("/api/locations", auth=AUTH).json()["locations"]
    assert listing[0]["state"] == "danger" and listing[0]["rules"] == {"unknown_person_after": "20:00"}
    assert http.get("/api/locations/warehouse/summary", auth=AUTH).json()["reason"] == "gas"
    assert not state.recent                                                      # summaries are not history
    note = state.notifications.notify("warehouse", "danger", "Небезпека", "gas", "danger:gas")
    assert [n["id"] for n in http.get("/api/notifications?unacked=true", auth=AUTH).json()["notifications"]] == \
        [note["id"]]
    acked = http.post(f"/api/notifications/{note['id']}/ack", auth=AUTH).json()
    assert acked["acknowledged_by"] == "admin" and bus[-1].type == "notification_acked"
    assert bus[-1].payload["location_id"] == "warehouse"
    assert http.post("/api/locations/warehouse/ack", auth=AUTH).json()["by"] == "admin"
    assert http.delete("/api/locations/unassigned", auth=AUTH).status_code == 422
    assert http.delete("/api/locations/warehouse", auth=AUTH).json()["deleted"] == "warehouse"
    state.locations.close()
    state.notifications.close()
    json.dumps(summary)
