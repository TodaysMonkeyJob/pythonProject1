"""B2 (TOD-49) agent plug-in framework: discovery, switches, data scope, timeouts, validation, Article 1 equality."""
import functools
import gc
import os
import sys
import textwrap
import time
import weakref

import pytest
import yaml

from agents.framework.bus_adapter import Claim, ScopeFilter, SituationBoard
from agents.framework.manifest import ManifestError, load_manifest
from agents.framework.orchestrator import AgentHost
from agents.framework.registry import AGENTS_DIR, AgentRegistry
from core.events import make_event

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_agent(root, name, manifest, code):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "agent.yaml").write_text(textwrap.dedent(manifest).format(name=name), encoding="utf-8")
    (folder / "agent.py").write_text(textwrap.dedent(code), encoding="utf-8")
    return folder


SLEEPY = """
    name: {name}
    role: audio
    version: 0.1.0
    entry: agent:Sleepy
    subscribes: [speech]
    publishes: [risk_opinion]
    data_scope: {{events: [speech], claims: [fuzzy]}}
    timeout_s: 0.05
    enabled_default: true
    config: {{sleep: 0.0, risk: null}}
"""
SLEEPY_CODE = """
    import time
    from agents.framework.base import Agent

    class Model:                          # stands for a loaded network
        weights = bytearray(1024)

    class Sleepy(Agent):
        def load(self):
            self.model = Model()
            self.seen = []

        def unload(self):
            self.model = None

        def on_message(self, message):
            self.seen.append(message.kind if message.event is None else message.event.type)
            if message.kind != "decision_request":
                return None
            time.sleep(self.config["sleep"])
            if self.config["risk"] is None:
                return None
            return [self.claim("risk_opinion", message.location_id, {"risk": self.config["risk"]},
                               round_id=message.round_id)]
"""


def registry(tmp_path=None, roots=None, enabled=None, settings=None, state_path=None, mode="hybrid"):
    enabled = {"video": False, "audio": False, **(enabled or {})}      # TOD-47 agents: tests/test_agents_article2.py
    cfg = {"mode": mode, "agents": {"enabled_agents": enabled, "settings": settings or {}}}
    return AgentRegistry(cfg, roots=roots or [AGENTS_DIR], entry_points=False, state_path=state_path)


class Bus:
    def __init__(self):
        self.events = []

    def __call__(self, type, payload, severity, location_id):
        self.events.append((type, payload, severity, location_id))

    def of(self, type):
        return [p for t, p, _, _ in self.events if t == type]


def hfs_event(location="lab", overall=0.5, level="medium", interval=(0.4, 0.6)):
    return make_event("system", "hfs_risk", {"overall": overall, "level": level, "interval": list(interval),
                                             "risk1": overall, "risk2": None, "risk3": None, "partial": True,
                                             "missing": ["camera", "audio"]},
                      location_id=location, device_id="hub:hfs")


# --- manifests / discovery --------------------------------------------------------------------------------------------

def test_shipped_agents_are_valid():
    reg = registry()
    assert {"fuzzy", "validation", "echo", "video", "audio"} <= set(reg.manifests) and not reg.errors
    assert reg.manifests["fuzzy"].required and not reg.manifests["fuzzy"].contributes
    assert reg.manifests["echo"].contributes and not reg.manifests["echo"].enabled_default


def test_drop_in_agent_is_discovered_without_core_changes(tmp_path):
    write_agent(tmp_path, "dropin", SLEEPY, SLEEPY_CODE)
    reg = registry(roots=[AGENTS_DIR, str(tmp_path)])
    loaded = lambda: [k for k in sys.modules if k.startswith("_iot_agent_dropin_")]   # noqa: E731
    assert "dropin" in reg.manifests and not loaded()                    # code imported only when enabled
    reg.start_enabled()
    assert "dropin" in reg.instances and len(loaded()) == 1
    reg.stop_all()


def test_invalid_manifests_are_listed_and_the_rest_loads(tmp_path):
    write_agent(tmp_path, "nosy", SLEEPY.replace("subscribes: [speech]", "subscribes: [speech, person_identified]"),
                SLEEPY_CODE)
    write_agent(tmp_path, "wrongname", SLEEPY.replace("name: {name}", "name: other"), SLEEPY_CODE)
    write_agent(tmp_path, "badevent", SLEEPY.replace("events: [speech]", "events: [speech, no_such_event]"),
                SLEEPY_CODE)
    reg = registry(roots=[AGENTS_DIR, str(tmp_path)])
    assert set(reg.errors) == {"nosy", "wrongname", "badevent"}
    assert "outside its data_scope" in reg.errors["nosy"]
    assert {"fuzzy", "validation"} <= set(reg.manifests)
    with pytest.raises(ManifestError):
        load_manifest(str(tmp_path / "wrongname" / "agent.yaml"))


# --- switches ---------------------------------------------------------------------------------------------------------

def test_disable_frees_the_model_and_switches_persist(tmp_path):
    write_agent(tmp_path, "dropin", SLEEPY, SLEEPY_CODE)
    state = tmp_path / "state" / "agents.json"
    reg = registry(roots=[AGENTS_DIR, str(tmp_path)], state_path=str(state))
    host = AgentHost(reg, Bus())
    host.start()
    model, agent = weakref.ref(reg.instances["dropin"].model), weakref.ref(reg.instances["dropin"])
    assert host.set_enabled("dropin", False) is None
    gc.collect()
    assert model() is None and agent() is None and "dropin" not in reg.instances and "dropin" not in host.runners
    assert yaml.safe_load(state.read_text())["dropin"] is False
    again = registry(roots=[AGENTS_DIR, str(tmp_path)], state_path=str(state))
    assert again.wanted["dropin"] is False                                 # survives a restart
    assert host.set_enabled("dropin", True) is None and "dropin" in reg.instances
    host.stop()


def test_required_single_role_and_cloud_rules(tmp_path):
    cloud = SLEEPY.replace("role: audio", "role: explanation").replace(
        "version: 0.1.0", "version: 0.1.0\n    backend: {{kind: cloud_model}}")
    write_agent(tmp_path, "cloudy", cloud, SLEEPY_CODE)
    write_agent(tmp_path, "judge2", SLEEPY.replace("role: audio", "role: validation"), SLEEPY_CODE)
    reg = registry(roots=[AGENTS_DIR, str(tmp_path)], mode="autonomous")
    problems = reg.start_enabled()
    assert "cloud model" in problems["cloudy"] and "cloudy" not in reg.instances
    judges = [n for n in ("judge2", "validation") if n in reg.instances]
    assert len(judges) == 1 and "already has role validation" in problems[({"judge2", "validation"} - {*judges}).pop()]
    assert "cannot be switched off" in reg.set_enabled("fuzzy", False)
    reg.stop_all()


# --- data scope -----------------------------------------------------------------------------------------------------

def test_scope_wins_over_subscriptions(tmp_path):
    write_agent(tmp_path, "dropin", SLEEPY, SLEEPY_CODE)
    reg = registry(roots=[AGENTS_DIR, str(tmp_path)])
    host = AgentHost(reg, Bus())
    host.start()
    agent = reg.instances["dropin"]
    for type_, payload in (("person_identified", {"track_id": 1, "identity": "Oleh"}),
                           ("object_detected", {"class": "person"}), ("speech", {"text": "hi"}),
                           ("agent_decision", {}), ("gas_reading", {})):
        host.observe(make_event("system", type_, payload, location_id="lab", device_id="d"))
    assert agent.seen == ["speech"]                                        # no face / gas data
    board = SituationBoard()
    for role in ("video", "fuzzy"):
        board.update(Claim(agent=role, role=role, version="1", location_id="lab", claim="x"))
    assert [c.role for c in board.view("lab", reg.manifests["dropin"])] == ["fuzzy"]
    video = Claim(agent="v", role="video", version="1", location_id="lab", claim="risk")
    assert ScopeFilter(reg.manifests.values()).claim_receivers(video, ["dropin", "echo"]) == []
    host.stop()


def test_claims_outside_publishes_are_dropped(tmp_path):
    code = SLEEPY_CODE.replace('self.claim("risk_opinion"', 'self.claim("set_threshold"')
    write_agent(tmp_path, "dropin", SLEEPY.replace("risk: null", "risk: 0.9"), code)
    reg = registry(roots=[AGENTS_DIR, str(tmp_path)])
    bus = Bus()
    host = AgentHost(reg, bus)
    host.start()
    host.observe(hfs_event())
    assert [c["claim"] for c in bus.of("agent_claim")] == ["risk"]          # only the fuzzy corridor
    assert host.stats["dropin"].errors == 1 and bus.of("agent_decision")[0]["branch"] == "fuzzy_only"
    host.stop()


# --- rounds, timeouts -------------------------------------------------------------------------------------------------

def test_late_agent_is_ignored_and_noted(tmp_path):
    write_agent(tmp_path, "dropin", SLEEPY, SLEEPY_CODE)
    reg = registry(roots=[AGENTS_DIR, str(tmp_path)], settings={"dropin": {"sleep": 0.12, "risk": 0.95}})
    reg.manifests["dropin"].eval_file = None
    bus = Bus()
    host = AgentHost(reg, bus)
    host.start()
    host.observe(hfs_event())
    decision = bus.of("agent_decision")[0]
    assert decision["agents"] == {"dropin": "late"} and decision["missing"] == ["dropin"]
    assert decision["risk"] == 0.5 and decision["level"] == "medium" and host.stats["dropin"].late == 1
    host.stop()


def test_threaded_host_decides_at_the_deadline(tmp_path):
    write_agent(tmp_path, "dropin", SLEEPY, SLEEPY_CODE)
    reg = registry(roots=[AGENTS_DIR, str(tmp_path)], settings={"dropin": {"sleep": 1.0, "risk": 0.95}})
    bus = Bus()
    host = AgentHost(reg, bus, threaded=True)
    host.start()
    host.observe(hfs_event())
    began = time.monotonic()
    while not bus.of("agent_decision") and time.monotonic() - began < 2:
        host.tick()
        time.sleep(0.01)
    decision = bus.of("agent_decision")[0]
    assert time.monotonic() - began < 0.5                                  # did not wait for the slow agent
    assert decision["agents"]["dropin"] == "missing" and decision["branch"] == "fuzzy_only"
    host.stop()


def test_new_corridor_supersedes_an_open_round(tmp_path):
    write_agent(tmp_path, "dropin", SLEEPY.replace("timeout_s: 0.05", "timeout_s: 5"), SLEEPY_CODE)
    reg = registry(roots=[AGENTS_DIR, str(tmp_path)], settings={"dropin": {"sleep": 1.0}})
    bus = Bus()
    host = AgentHost(reg, bus, threaded=True)
    host.start()
    host.observe(hfs_event(overall=0.3, level="low"))
    host.observe(hfs_event(overall=0.6, level="medium"))
    began = time.monotonic()
    while not bus.of("agent_decision") and time.monotonic() - began < 2:
        host.drain()
        time.sleep(0.005)
    first = bus.of("agent_decision")[0]
    assert first["agents"]["dropin"] == "superseded" and first["risk"] == 0.3
    host.stop()


# --- validation policy ------------------------------------------------------------------------------------------------

@pytest.fixture()
def validator():
    reg = registry()
    reg.start_enabled()
    yield reg.instances["validation"]
    reg.stop_all()


def opinion(role, risk, agent=None, confidence=1.0):
    return Claim(agent=agent or role, role=role, version="1", location_id="lab", claim="risk_opinion",
                 value={"risk": risk}, confidence=confidence)


def corridor(risk=0.6, level="medium", interval=(0.5, 0.7)):
    return Claim(agent="fuzzy", role="fuzzy", version="1", location_id="lab", claim="risk",
                 value={"risk": risk, "level": level}, interval=list(interval))


def test_validation_policy(validator):
    weights = {"audio": 0.9, "video": 0.8, "gas": 0.7}
    d = validator.decide(corridor(), [opinion("audio", 0.1)], weights, [])
    assert d["branch"] == "deescalation_rejected" and d["risk"] == 0.6           # speech never lowers
    d = validator.decide(corridor(), [opinion("video", 0.1)], weights, [])
    assert d["branch"] == "deescalation_confirmed" and d["risk"] == 0.1 and d["level"] == "low"
    d = validator.decide(corridor(), [opinion("audio", 0.95)], weights, [])
    assert d["branch"] == "escalation_review" and d["risk"] == 0.7
    assert d["operator_request"]["level_agents"] == "critical"
    d = validator.decide(corridor(), [opinion("audio", 0.62)], weights, [])
    assert d["branch"] == "agree" and 0.6 < d["risk"] < 0.62
    d = validator.decide(corridor(), [opinion("audio", 0.95, agent="x")], weights, [])
    assert d["branch"] == "fuzzy_only" and "unmeasured" in d["reasons"][0]        # weight 0: logged, not counted


def test_operator_request_event(tmp_path):
    folder = write_agent(tmp_path, "dropin", SLEEPY.replace("timeout_s: 0.05", "timeout_s: 1")
                         .replace("enabled_default: true", "enabled_default: true\n    eval_file: eval.yaml"),
                         SLEEPY_CODE)
    (folder / "eval.yaml").write_text("accuracy: 0.8\nn: 40\nsource: test\n", encoding="utf-8")
    reg = registry(roots=[AGENTS_DIR, str(tmp_path)], settings={"dropin": {"risk": 0.99}})
    bus = Bus()
    host = AgentHost(reg, bus, decides=True)
    host.start()
    host.observe(hfs_event())
    decision = bus.of("agent_decision")[0]
    assert decision["branch"] == "escalation_review" and decision["decided_by"] == "validation"
    assert [s for t, _, s, _ in bus.events if t in ("agent_decision", "agent_operator_request")] == \
        ["warning", "warning"]                                       # risk = y_r 0.6 → high
    assert decision["versions"] == {"dropin": "0.1.0", "fuzzy": "1.0.0",
                                    "validation": reg.manifests["validation"].version}
    request = bus.of("agent_operator_request")[0]
    assert request["round_id"] == decision["round_id"] and request["agents"] == ["dropin"]
    host.stop()


# --- with every agent but the fuzzy one off: Article 1 ----------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def article1_payloads():
    from experiments.agents.run import LOCATION, article1_payload, scenario_events

    base = yaml.safe_load(open(os.path.join(REPO, "config.yaml"), encoding="utf-8"))
    config = {"hfs": base["hfs"], "identity": base.get("identity", {}), "gas": base.get("gas", {})}
    spec = yaml.safe_load(open(os.path.join(REPO, "experiments", "hfs", "scenarios.yaml"), encoding="utf-8"))
    out = []
    for scenario in spec["scenarios"]:
        payload = article1_payload(scenario, config, *scenario_events(scenario, 42))
        out.append(make_event("system", "hfs_risk", payload, location_id=LOCATION, device_id="hub:hfs"))
    return tuple(out)


@pytest.mark.parametrize("enabled", [{"validation": False, "echo": False}, {"validation": True, "echo": False},
                                     {"validation": True, "echo": True}])
def test_only_the_fuzzy_agent_reproduces_article1(enabled):
    events = article1_payloads()
    scenarios = yaml.safe_load(open(os.path.join(REPO, "experiments", "hfs", "scenarios.yaml"), encoding="utf-8"))
    assert len(events) == len(scenarios["scenarios"]) >= 50
    bus = Bus()
    host = AgentHost(registry(enabled=enabled), bus)
    host.start()
    for event in events:
        host.observe(event)
    decisions = bus.of("agent_decision")
    assert len(decisions) == len(events)
    severities = {p["level"]: s for t, p, s, _ in bus.events if t == "agent_decision"}   # TOD-52: the decision
    assert severities.get("critical", "critical") == "critical" and severities.get("low", "info") == "info"
    for event, decision in zip(events, decisions, strict=True):
        p = event.payload
        assert (decision["level"], decision["risk"], decision["interval"]) == (p["level"], p["overall"],
                                                                               p["interval"])
        assert decision["evidence_ids"] == [event.event_id] and decision["mode_requested"] == "multiagent"
        if enabled["validation"]:
            assert decision["branch"] == "fuzzy_only" and decision["mode_effective"] == "multiagent"
        else:                                    # TOD-52: no validation agent → automatic fallback to autonomous
            assert decision["branch"] == "fallback" and decision["mode_effective"] == "autonomous"
            assert decision["fallback_reason"] == "агент валідації вимкнений"
    host.stop()


# --- module, dashboard, location state -------------------------------------------------------------------------------

def test_module_switch_from_a_control_event():
    import queue

    from agents.module import AgentsModule

    events = queue.Queue()
    module = AgentsModule(events, queue.Queue(), {"agents": {}}, host=AgentHost(registry(), lambda *a: None))
    module.host.emit = module.emit_event
    module.setup()
    module.handle_control(make_event("system", "agent_control", {"agent": "echo", "enabled": True, "by": "op"})
                          .to_json())
    assert "echo" in module.host.registry.instances
    module.step()
    module.teardown()


def test_dashboard_agents_api(tmp_path):
    from fastapi.testclient import TestClient

    from dashboard.server import DashboardState, create_app
    from tests.test_dashboard import auth_header, config

    cfg = config(tmp_path)
    state = DashboardState(cfg)
    sent = []
    state.publish = sent.append
    app = TestClient(create_app(state, cfg))
    assert app.get("/api/agents", headers=auth_header()).status_code == 503
    host = AgentHost(registry(), lambda *a: None)
    host.start()
    state.on_event(make_event("system", "agent_status", host.status()).to_json())
    data = app.get("/api/agents", headers=auth_header()).json()
    assert {a["name"] for a in data["agents"]} >= {"fuzzy", "validation", "echo"}
    assert app.post("/api/agents/fuzzy", json={"enabled": False}, headers=auth_header()).status_code == 409
    assert app.post("/api/agents/nope", json={"enabled": True}, headers=auth_header()).status_code == 404
    assert app.post("/api/agents/echo", json={"enabled": True}, headers=auth_header()).status_code == 200
    assert sent[0].type == "agent_control" and sent[0].payload["agent"] == "echo"
    host.stop()


def test_location_state_follows_agent_decision_only_when_agents_decide():
    from modules.locations.state import DANGER, SAFE, LocationStateEngine

    decision = make_event("system", "agent_decision", {"round_id": "r", "risk": 0.9, "level": "critical",
                                                       "interval": [0.8, 1], "branch": "agree",
                                                       "reasons": ["x"], "missing": []},
                          severity="critical", location_id="lab", device_id="hub:agents")
    for decide, expected in ((False, SAFE), (True, DANGER)):
        engine = LocationStateEngine(lambda i: i, lambda i: {}, lambda: [], {"agents_decide": decide})
        engine.on_event(decision, time.time())
        assert engine.live["lab"].state == expected
