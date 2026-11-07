"""M1 (TOD-52) mode switch: precedence, persistence + audit, switching mid-run, fallback, admin only."""
import base64
import json

import pytest

from agents.framework.orchestrator import AgentHost
from agents.framework.registry import AGENTS_DIR, AgentRegistry
from core.events import Event, make_event
from core.mode import AUTONOMOUS, MULTIAGENT, ModeStore, location_mode, normalise, start_mode
from core.network import NetworkProbe


def store_cfg(tmp_path, **config):
    return {"mode_store": {"state_path": str(tmp_path / "mode.json"), "audit_path": str(tmp_path / "audit.jsonl")},
            **config}


# --- precedence, persistence, audit ------------------------------------------------------------------------------

def test_start_up_precedence(tmp_path):
    config = store_cfg(tmp_path, mode="autonomous")
    assert start_mode(config, env={})[::2] == ("autonomous", "config")
    ModeStore(str(tmp_path / "mode.json"), None).save("multiagent", {"store": "autonomous"}, "admin")
    assert start_mode(config, env={}) == ("multiagent", {"store": "autonomous"}, "saved")
    assert start_mode(config, env={"IOT_MODE": "autonomous"})[::2] == ("autonomous", "env")
    assert start_mode(config, cli="multiagent", env={"IOT_MODE": "autonomous"})[::2] == ("multiagent", "cli")
    assert start_mode(store_cfg(tmp_path / "none"), env={})[::2] == ("multiagent", "default")
    assert normalise("hybrid") == MULTIAGENT                         # the TOD-41 name
    with pytest.raises(ValueError):
        normalise("gpt")


def orchestrator(tmp_path, mode="autonomous"):
    from core.orchestrator import Orchestrator

    config = store_cfg(tmp_path, mode=mode, orchestrator={"log_events": False}, trust={"enabled": False},
                       llm={"enabled": False})
    return Orchestrator(config, modules=[])


def emitted(orch, wait=0.5):
    """Everything the orchestrator published (the multiprocessing queue flushes in a background thread)."""
    import queue

    out = []
    while True:
        try:
            out.append(Event.from_json(orch.events.get(timeout=wait)))
        except queue.Empty:
            return out


def test_switch_is_saved_audited_and_broadcast(tmp_path):
    orch = orchestrator(tmp_path)
    orch._route(make_event("system", "mode_control", {"mode": "multiagent", "by": "admin"}).to_json())
    orch._route(make_event("system", "mode_control", {"mode": "autonomous", "location_id": "store",
                                                      "by": "admin"}).to_json())
    changed = [e.payload for e in emitted(orch) if e.type == "mode_changed"]
    assert [(c["scope"], c["from"], c["to"]) for c in changed] == [("global", "autonomous", "multiagent"),
                                                                     ("location", "multiagent", "autonomous")]
    assert changed[-1]["locations"] == {"store": "autonomous"} and orch.child_config["mode"] == "multiagent"
    saved = json.load(open(tmp_path / "mode.json", encoding="utf-8"))
    assert saved["mode"] == "multiagent" and saved["locations"] == {"store": "autonomous"} and saved["by"] == "admin"
    history = ModeStore(None, str(tmp_path / "audit.jsonl")).history()
    assert [(h["by"], h["scope"], h["from"], h["to"]) for h in history] == [
        ("admin", "global", "autonomous", "multiagent"), ("admin", "location", "multiagent", "autonomous")]
    assert start_mode(store_cfg(tmp_path, mode="autonomous"), env={})[:2] == ("multiagent",
                                                                               {"store": "autonomous"})   # restart
    orch._route(make_event("system", "mode_control", {"mode": "nonsense", "by": "x"}).to_json())   # refused
    assert not [e for e in emitted(orch, 0.2) if e.type == "mode_changed"]


# --- switching during a running scenario: one decision per cycle -------------------------------------------------

def test_switch_mid_run_no_lost_or_duplicated_decisions():
    from modules.hfs.worker import HierarchicalFuzzySystem
    from modules.locations.state import LocationStateEngine

    config = {"mode": AUTONOMOUS, "mode_locations": {}}
    clock = [1000.0]
    hfs = HierarchicalFuzzySystem({}, devices=lambda: [], clock=lambda: clock[0])
    bus = []
    host = AgentHost(AgentRegistry({"agents": {"enabled_agents": {"video": False, "audio": False}}},
                                   roots=[AGENTS_DIR], entry_points=False, state_path=None),
                     lambda t, p, s, loc: bus.append((t, p)), clock=lambda: clock[0],
                     mode=lambda loc: location_mode(config, loc))
    host.start()
    engine = LocationStateEngine(lambda i: i, lambda i: {}, lambda: [],
                                 {"mode": lambda loc: location_mode(config, loc)})
    deciding = []
    for cycle in range(12):
        if cycle == 4:
            config["mode"] = MULTIAGENT                      # mode_changed between two cycles
        if cycle == 8:
            config["mode"] = AUTONOMOUS
        clock[0] += 31                                       # past keepalive: one hfs_risk per cycle
        hfs.observe(make_event("gas", "gas_reading", {"gas": "ammonia", "concentration": 30 + 20 * cycle,
                                                      "unit": "ppm"}, location_id="lab", device_id="nh3"))
        [payload] = hfs.tick()
        payload.pop("location_id")
        payload["mode_requested"] = location_mode(config, "lab")
        event = make_event("system", "hfs_risk", payload, location_id="lab", device_id="hub:hfs")
        before = len(bus)
        engine.on_event(event, clock[0])
        host.observe(event)
        decisions = [p for t, p in bus[before:] if t == "agent_decision" and event.event_id in p["evidence_ids"]]
        for d in decisions:
            engine.on_event(make_event("system", "agent_decision", d, location_id="lab", device_id="a"), clock[0])
        record, from_agents = engine.decision(engine.live["lab"], clock[0])
        deciding.append(config["mode"])
        expected = 0 if config["mode"] == AUTONOMOUS else 1
        assert len(decisions) == expected, cycle                  # never two, never none in multiagent
        assert from_agents == (config["mode"] == MULTIAGENT), cycle
        if not from_agents:
            assert record["event_id"] == event.event_id, cycle      # autonomous: this cycle's hfs_risk decides
        else:
            assert decisions[0]["mode_effective"] == MULTIAGENT and decisions[0]["rules_version"] is None
    assert deciding == [AUTONOMOUS] * 4 + [MULTIAGENT] * 4 + [AUTONOMOUS] * 4
    host.stop()


# --- automatic fallback ---------------------------------------------------------------------------------------------

def hfs_event(location="lab"):
    return make_event("system", "hfs_risk", {"overall": 0.5, "level": "medium", "interval": [0.4, 0.6]},
                      location_id=location, device_id="hub:hfs")


def test_network_off_falls_back_and_comes_back():
    network = {"up": True}
    bus = []
    host = AgentHost(AgentRegistry({"agents": {"enabled_agents": {"video": False, "audio": False}}},
                                   roots=[AGENTS_DIR], entry_points=False, state_path=None),
                     lambda t, p, s, loc: bus.append((t, p)), network_ok=lambda: network["up"])
    host.start()
    for up in (True, False, True):
        network["up"] = up
        host.observe(hfs_event())
    decisions = [p for t, p in bus if t == "agent_decision"]
    assert [d["mode_effective"] for d in decisions] == [MULTIAGENT, AUTONOMOUS, MULTIAGENT]
    assert decisions[1]["fallback_reason"] == "немає мережі" and decisions[1]["branch"] == "fallback"
    assert decisions[1]["level"] == "medium" and decisions[1]["risk"] == 0.5          # the fuzzy corridor decides
    host.stop()


def test_required_agent_late_and_validator_off_fall_back():
    bus = []
    host = AgentHost(AgentRegistry({"agents": {"enabled_agents": {"validation": False, "video": False,
                                                                  "audio": False}}},
                                   roots=[AGENTS_DIR], entry_points=False, state_path=None),
                     lambda t, p, s, loc: bus.append((t, p)), required=["video"])
    host.start()
    host.observe(hfs_event())
    [decision] = [p for t, p in bus if t == "agent_decision"]
    assert decision["fallback_reason"] == "агент валідації вимкнений; не відповіли вчасно: video"
    host.stop()


def test_autonomous_location_gets_no_round_and_agents_are_suspended():
    bus = []
    modes = {"lab": AUTONOMOUS, "office": MULTIAGENT}
    registry = AgentRegistry({"agents": {"enabled_agents": {"echo": True}}}, roots=[AGENTS_DIR], entry_points=False,
                             state_path=None)
    host = AgentHost(registry, lambda t, p, s, loc: bus.append((t, p)), mode=lambda loc: modes[loc])
    host.start()
    host.observe(make_event("audio", "speech", {"text": "hi"}, location_id="lab", device_id="m"))
    assert host.stats["echo"].messages == 0                         # nothing for an autonomous location
    host.observe(hfs_event("lab"))
    host.observe(hfs_event("office"))
    assert [p["location_id"] for t, p in bus if t == "agent_decision"] == ["office"]
    host.suspend(True)
    assert "echo" not in registry.instances and "fuzzy" in registry.instances      # models unloaded
    host.suspend(False)
    assert "echo" in registry.instances
    host.stop()


def test_locations_fall_back_when_the_agents_are_silent():
    from modules.locations.state import LocationStateEngine

    engine = LocationStateEngine(lambda i: i, lambda i: {}, lambda: [], {"mode": lambda loc: MULTIAGENT})
    decision = make_event("system", "agent_decision", {"round_id": "r", "level": "high", "risk": 0.65,
                                                       "mode_effective": "multiagent"}, location_id="lab",
                          device_id="a", severity="warning")
    engine.on_event(decision, 1000.0)
    engine.on_event(hfs_event(), 1001.0)
    assert engine.decision(engine.live["lab"], 1001.0)[1] is True
    engine.on_event(hfs_event(), 1030.0)                          # 29 s later and no agent decision
    record, from_agents = engine.decision(engine.live["lab"], 1030.0)
    assert from_agents is False and engine.live["lab"].mode == {"requested": "multiagent", "effective": "autonomous",
                                                                "reason": "агенти не відповідають"}
    assert engine.summary("lab", 1030.0)["mode"]["effective"] == "autonomous"


def test_network_probe_caches_and_recovers():
    state = {"up": False}

    def connect():
        if not state["up"]:
            raise OSError("unreachable")

    clock = [0.0]
    probe = NetworkProbe(every_sec=30, clock=lambda: clock[0], connect=connect)
    assert probe.ok() is False
    state["up"] = True
    assert probe.ok() is False                                    # cached until every_sec
    clock[0] = 31
    assert probe.ok() is True


def test_aggregator_queues_for_gpt_only_in_multiagent(tmp_path):
    from modules.aggregator.aggregator import Aggregator

    agg = Aggregator({"aggregator": {"export_dir": str(tmp_path), "queue_for_llm": False, "export_csv": False,
                                     "incidents": {"enabled": False}}})
    assert agg.outbox is None
    agg.on_event(make_event("system", "mode_changed", {"mode": "multiagent", "locations": {}}))
    assert agg.outbox is not None
    agg.on_event(make_event("system", "mode_changed", {"mode": "autonomous", "locations": {"office": "multiagent"}}))
    assert agg.outbox is not None                                 # one location still multiagent
    agg.on_event(make_event("system", "mode_changed", {"mode": "autonomous", "locations": {}}))
    assert agg.outbox is None


# --- dashboard: admin only --------------------------------------------------------------------------------------------

def test_only_the_admin_switches(tmp_path):
    from fastapi.testclient import TestClient

    from dashboard.server import DashboardState, create_app
    from tests.test_dashboard import auth_header, config

    cfg = {**config(tmp_path), **store_cfg(tmp_path), "mode": "autonomous"}
    cfg["dashboard"]["viewers"] = "anna:pw1"
    state = DashboardState(cfg)
    sent = []
    state.publish = sent.append
    app = TestClient(create_app(state, cfg))
    viewer = {"Authorization": "Basic " + base64.b64encode(b"anna:pw1").decode()}
    assert app.get("/api/mode", headers=viewer).json()["role"] == "viewer"
    assert app.post("/api/mode", json={"mode": "multiagent", "confirm": True}, headers=viewer).status_code == 403
    assert app.post("/api/mode", json={"mode": "multiagent"}, headers=auth_header()).status_code == 400   # no confirm
    assert app.post("/api/mode", json={"mode": "gpt", "confirm": True}, headers=auth_header()).status_code == 400
    r = app.post("/api/mode", json={"mode": "multiagent", "confirm": True}, headers=auth_header())
    assert r.status_code == 200 and sent[0].type == "mode_control" and sent[0].payload["by"] == "admin"
    data = app.get("/api/mode", headers=auth_header()).json()
    assert data["role"] == "admin" and data["mode"] == "autonomous" and data["labels"]["multiagent"].startswith(
        "Мультиагентний")
    state.on_event(make_event("system", "mode_changed", {"mode": "multiagent", "locations": {}}).to_json())
    state.on_event(make_event("system", "location_summary", {"location_id": "lab", "name": "Лабораторія", "mode": {
        "requested": "multiagent", "effective": "autonomous", "reason": "немає мережі"}}).to_json())
    data = app.get("/api/mode", headers=auth_header()).json()
    assert data["mode"] == "multiagent" and data["fallback"] == [{"location_id": "lab", "name": "Лабораторія",
                                                                  "reason": "немає мережі"}]
    assert app.get("/api/status", headers=viewer).status_code == 200                 # viewers can read
