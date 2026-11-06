"""A1 (TOD-41) autonomous hierarchical fuzzy system: rule bases, top level, missing subsystems, mode switch."""
import json
import os
import queue
import subprocess
import sys
from datetime import datetime

import numpy as np
import pytest

from core.events import make_event
from core.fuzzy import FuzzyEngine, load_yaml
from core.mode import LLM_PACKAGES, apply_mode, is_autonomous, mode_of
from modules.hfs.__main__ import check
from modules.hfs.audio import AudioSubsystem, keyword_grade, keyword_grades
from modules.hfs.camera import CameraSubsystem, hours_outside
from modules.hfs.gas import GasSubsystem
from modules.hfs.policy import LEVELS, POLICIES, policy_level
from modules.hfs.top import TopLevel
from modules.hfs.worker import HFSModule, HierarchicalFuzzySystem
from modules.locations.state import DANGER, SAFE, LocationStateEngine
from tools.plot_surfaces import monotonicity

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = {"gas": {"thresholds": {"ethanol": 5.0, "ammonia": 3.0}}}


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def sub(risk, low=None, high=None):
    """A subsystem result with the given crisp risk (interval = the risk itself unless given)."""
    return {"risk": risk, "level": "?", "interval": [risk if low is None else low, risk if high is None else high],
            "fired_rules": [{"id": "X", "term": "?", "strength": 1.0, "explanation": "x"}], "inputs": {},
            "partial": False}


def ev(type_, payload, location="lab", device="dev-1", **kw):
    return make_event("system", type_, payload, location_id=location, device_id=device, **kw)


# --- rule bases -------------------------------------------------------------------------------------------------------

def test_rule_bases_valid_triangular_and_documented(capsys):
    assert check() == 0
    out = capsys.readouterr().out
    for name in ("gas", "camera", "audio", "top"):
        assert f"ok  {name}" in out
    assert "gauss" not in out                       # triangular / trapezoidal only (supervisor note)


# --- top level --------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def top():
    return TopLevel()


@pytest.mark.parametrize("risks, level", [
    ({"gas": 0.9, "camera": 0.1, "audio": 0.1}, "critical"),        # any subsystem critical
    ({"gas": 0.1, "camera": 0.1, "audio": 0.95}, "critical"),
    ({"gas": 0.68, "camera": 0.68, "audio": 0.1}, "critical"),      # two high
    ({"gas": 0.1, "camera": 0.68, "audio": 0.68}, "critical"),
    ({"gas": 0.68, "camera": 0.1, "audio": 0.1}, "high"),           # one high, others low
    ({"gas": 0.1, "camera": 0.1, "audio": 0.68}, "high"),
    ({"gas": 0.4, "camera": 0.4, "audio": 0.05}, "high"),           # two medium
    ({"gas": 0.05, "camera": 0.4, "audio": 0.05}, "medium"),        # one medium
    ({"gas": 0.05, "camera": 0.1, "audio": 0.05}, "low"),
])
def test_top_level_rules(top, risks, level):
    result = top.combine({name: sub(value) for name, value in risks.items()})
    assert result["level"] == level, result
    assert result["partial"] is False and result["missing"] == []
    assert result["fired_rules_by_level"]["top"] and set(result["fired_rules_by_level"]) == {"gas", "camera", "audio",
                                                                                             "top"}
    assert [result["risk1"], result["risk2"], result["risk3"]] == [risks["gas"], risks["camera"], risks["audio"]]
    assert result["interval"][0] <= result["overall"] <= result["interval"][1]


def test_top_level_explains_with_rule_ids(top):
    result = top.combine({"gas": sub(0.9), "camera": sub(0.1), "audio": sub(0.1)})
    fired = result["fired_rules_by_level"]["top"][0]
    assert fired["id"] == "T300" and "risk_gas" in fired["explanation"] and "critical" in fired["explanation"]


def test_missing_subsystem_is_partial_and_uses_the_rest(top):
    full = top.combine({"gas": sub(0.9), "camera": sub(0.1), "audio": sub(0.1)})
    partial = top.combine({"gas": sub(0.9), "camera": sub(0.1), "audio": None})
    assert partial["partial"] is True and partial["missing"] == ["audio"] and partial["risk3"] is None
    assert partial["level"] == "critical" and abs(partial["overall"] - full["overall"]) < 0.02
    # one high + the other present one low, third missing → high (missing neither escalates nor blocks)
    assert top.combine({"gas": None, "camera": sub(0.68), "audio": sub(0.1)})["level"] == "high"
    assert top.combine({"gas": None, "camera": None, "audio": sub(0.05)})["level"] == "low"
    assert top.combine({"gas": None, "camera": None, "audio": None}) is None


def test_partial_subsystem_marks_overall_partial(top):
    result = top.combine({"gas": {**sub(0.1), "partial": True}, "camera": sub(0.1), "audio": sub(0.1)})
    assert result["partial"] is True and result["missing"] == []


def test_top_interval_follows_subsystem_intervals(top):
    narrow = top.combine({"gas": sub(0.5), "camera": sub(0.1), "audio": sub(0.1)})
    wide = top.combine({"gas": sub(0.5, 0.3, 0.8), "camera": sub(0.1), "audio": sub(0.1)})
    assert wide["interval"][0] <= narrow["interval"][0] and wide["interval"][1] >= narrow["interval"][1]
    assert wide["interval"][1] - wide["interval"][0] > narrow["interval"][1] - narrow["interval"][0]


@pytest.mark.parametrize("name", ["gas", "camera", "audio", "top"])
def test_rule_table_is_complete_smooth_and_follows_the_policy(name):
    engine = FuzzyEngine.from_dict(load_yaml(os.path.join(REPO, "config", "fuzzy", f"{name}.yaml")))
    inputs = [v for v in engine.variables if v != engine.output]
    terms = {v: list(engine.variables[v].terms) for v in inputs}
    cells = {}
    for rule in engine.rules:
        literals = dict(rule.clauses())
        cells[tuple(terms[v].index(literals[v]) for v in inputs)] = LEVELS.index(rule.term)
    assert len(cells) == len(engine.rules) == int(np.prod([len(t) for t in terms.values()]))     # complete
    for cell, level in cells.items():
        assert level == POLICIES[name](*cell), (name, cell)
        for axis in range(len(inputs)):                       # monotone and smooth along every input
            if cell[axis] + 1 < len(terms[inputs[axis]]):
                neighbour = cells[cell[:axis] + (cell[axis] + 1,) + cell[axis + 1:]]
                assert 0 <= neighbour - level <= 1, (name, cell, inputs[axis])
    for v in inputs:                                          # Ruspini partition: memberships sum to 1
        _, curves = engine.variables[v].curves()
        assert np.allclose(np.sum(list(curves.values()), axis=0), 1.0), (name, v)


@pytest.mark.parametrize("name", ["gas", "camera", "audio", "top"])
def test_surface_is_exactly_monotonic(name):
    """A riskier input never lowers the risk (complete smooth table, Ruspini, product / sum inference)."""
    engine = FuzzyEngine.from_dict(load_yaml(os.path.join(REPO, "config", "fuzzy", f"{name}.yaml")))
    report = monotonicity(engine, 7 if name == "audio" else 11)
    assert all(stats["falls"] == 0 for stats in report.values()), report


def test_top_policy_matches_the_requirements():
    assert policy_level(["critical", "low", "low"]) == "critical"
    assert policy_level(["high", "high", "low"]) == "critical"
    assert policy_level(["high", "low", "medium"]) == "high"
    assert policy_level(["medium", "medium", "low"]) == "high"
    assert policy_level(["low", "medium", "low"]) == "medium"
    assert policy_level(["low", "low", "low"]) == "low"


# --- subsystems -------------------------------------------------------------------------------------------------------

def test_gas_subsystem_normalises_to_limits_and_goes_stale():
    clock = Clock()
    gas = GasSubsystem(CONFIG, clock=clock)
    gas.observe(ev("gas_reading", {"gas": "ammonia", "concentration_pct": 3.2, "rise_rate_pct_per_min": 0.0}),
                clock())
    result = gas.evaluate("lab", clock(), True)
    assert result["inputs"]["gas_level"] == pytest.approx(3.2 / 3.0, abs=1e-3) and result["level"] == "critical"
    assert result["fired_rules"][0]["id"] == "G312"          # critical level, stable, mixture high (one sensor)
    gas.observe(ev("gas_reading", {"gas": "ethanol", "concentration_pct": 3.2, "rise_rate_pct_per_min": 0.0}),
                clock())
    assert gas.evaluate("lab", clock(), True)["level"] == "high"            # same %, 5 % limit → elevated
    gas.observe(ev("gas_reading", {"gas": "clean_air", "concentration_pct": 0}), clock())
    assert gas.evaluate("lab", clock(), True)["level"] == "low"
    assert gas.evaluate("lab", clock() + 31, True) is None                  # stale → no data
    gas.forget_device("dev-1")
    assert gas.evaluate("lab", clock(), True) is None


def test_gas_rising_fast_escalates():
    gas = GasSubsystem(CONFIG)
    slow = gas.infer({"gas_level": 0.6, "gas_dynamics": 0.0})
    fast = gas.infer({"gas_level": 0.6, "gas_dynamics": 0.3})
    assert slow["level"] == "high" and fast["level"] == "critical"


def test_hours_outside_activity_window():
    at = lambda hh, mm=0: datetime(2026, 11, 6, hh, mm)  # noqa: E731
    assert hours_outside(at(12), "08:00", "19:00") == 0
    assert hours_outside(at(20), "08:00", "19:00") == 1
    assert hours_outside(at(2), "08:00", "19:00") == 6
    assert hours_outside(at(7, 30), "08:00", "19:00") == 0.5
    assert hours_outside(at(23), "22:00", "06:00") == 0                      # night shift wraps midnight
    assert hours_outside(at(12), "22:00", "06:00") == 6


def test_camera_subsystem_strangers_at_night_vs_known_by_day():
    clock = Clock()
    night = CameraSubsystem({}, clock=clock, local_now=lambda: datetime(2026, 11, 6, 2, 0))
    assert night.evaluate("lab", clock(), available=False) is None           # no camera, no events
    night.observe(ev("object_detected", {"class": "person", "track_id": 1, "state": "appeared"}), clock())
    night.observe(ev("person_identified", {"track_id": 1, "identity": "unknown_3", "person_id": "p3"}), clock())
    result = night.evaluate("lab", clock(), available=True)
    assert result["inputs"]["unknown_share"] == 1 and result["level"] == "high"      # a stranger standing still
    night.observe(ev("motion", {"state": "start"}), clock() - 30)
    assert night.evaluate("lab", clock(), available=True)["level"] == "critical"   # … moving a lot
    day = CameraSubsystem({}, clock=clock, local_now=lambda: datetime(2026, 11, 6, 12, 0),
                          rules=lambda loc: {"activity_hours": ["08:00", "19:00"]})
    day.observe(ev("object_detected", {"class": "person", "track_id": 1, "state": "appeared"}), clock())
    day.observe(ev("person_identified", {"track_id": 1, "identity": "Oleh", "person_id": "p1"}), clock())
    assert day.evaluate("lab", clock(), available=True)["level"] == "low"
    day.observe(ev("object_detected", {"class": "person", "track_id": 2, "state": "appeared"}), clock())
    assert day.unknown_share("lab", clock()) == 0                            # not decided yet …
    assert day.unknown_share("lab", clock() + 11) == 0.5                     # … unidentified after 10 s


def test_camera_motion_intensity_is_share_of_window():
    clock = Clock()
    camera = CameraSubsystem({}, clock=clock)
    camera.observe(ev("motion", {"state": "start"}), 1000.0)
    camera.observe(ev("motion", {"state": "end"}), 1015.0)
    assert camera.motion_share("lab", 1030.0) == pytest.approx(0.5)
    assert camera.motion_share("lab", 1060.0) == 0


def test_audio_keywords_and_unknown_voice():
    grades = keyword_grades({1.0: ["допоможіть", "fire"], 0.4: ["стоп"]})
    assert keyword_grade("Допоможіть, будь ласка!", grades) == (1.0, "допоможіть")
    assert keyword_grade("campfire stories", grades) == (0.0, None)          # whole-word start only
    clock = Clock()
    audio = AudioSubsystem({}, clock=clock)
    audio.observe(ev("sound", {"rms_db": -6}), clock())
    audio.observe(ev("speech", {"text": "Допоможіть!", "speaker_status": "unknown"}), clock())
    result = audio.evaluate("lab", clock(), available=True)
    assert result["inputs"]["danger_keywords"] == 1 and result["level"] == "critical"
    assert audio.evaluate("lab", clock() + 31, available=True)["level"] == "low"   # window passed
    quiet = AudioSubsystem({}, clock=clock)
    quiet.observe(ev("speech", {"text": "добрий день", "speaker_status": "known"}), clock())
    assert quiet.evaluate("lab", clock(), available=True)["level"] == "low"


# --- the whole system -------------------------------------------------------------------------------------------------

def devices(*kinds, location="lab", status="online"):
    return lambda: [{"device_id": f"{k}-1", "kind": k, "location_id": location, "status": status, "enabled": True}
                    for k in kinds]


def test_hfs_publishes_per_location_on_change_and_marks_partial():
    clock = Clock()
    system = HierarchicalFuzzySystem({**CONFIG, "hfs": {"keepalive_sec": 30}}, devices=devices("camera", "mic"),
                                     clock=clock, local_now=lambda: datetime(2026, 11, 6, 12, 0))
    [first] = system.tick()                       # no gas sensor in the lab → partial
    assert first["location_id"] == "lab" and first["missing"] == ["gas"] and first["partial"] is True
    assert first["level"] == "low" and first["risk1"] is None
    clock.t += 3
    assert system.tick() == []                    # nothing changed
    system.observe(ev("gas_reading", {"gas": "ammonia", "concentration_pct": 3.5, "rise_rate_pct_per_min": 0}))
    clock.t += 3
    [second] = system.tick()
    assert second["level"] == "critical" and second["missing"] == [] and second["partial"] is False
    assert second["fired_rules_by_level"]["gas"][0]["id"] == "G312"
    assert second["fired_rules_by_level"]["top"][0]["id"] == "T300"
    clock.t += 31
    [third] = system.tick()                       # gas reading stale → partial again
    assert third["missing"] == ["gas"]
    clock.t += 31
    assert len(system.tick()) == 1                # keepalive


def test_hfs_ignores_hub_and_device_offline_drops_evidence():
    clock = Clock()
    system = HierarchicalFuzzySystem(CONFIG, clock=clock)
    system.observe(ev("gas_reading", {"gas": "ammonia", "concentration_pct": 3.5}, location="hub"))
    system.observe(ev("gas_reading", {"gas": "ammonia", "concentration_pct": 3.5}, device="gas-9"))
    assert [p["location_id"] for p in system.tick()] == ["lab"]
    system.observe(ev("device_offline", {"device_id": "gas-9"}))
    clock.t += 3
    assert system.tick() == []                    # no subsystem has data any more → nothing to publish


def test_hfs_module_publishes_hfs_risk_with_mode_severity():
    clock = Clock()
    for mode, severity in (("autonomous", "critical"), ("hybrid", "info")):
        system = HierarchicalFuzzySystem(CONFIG, clock=clock)
        events = queue.Queue()
        module = HFSModule(events, queue.Queue(), {"mode": mode}, system=system, clock=clock)
        module.handle_control(ev("gas_reading", {"gas": "ammonia", "concentration_pct": 3.5}).to_json())
        module.step()
        data = json.loads(events.get_nowait())
        assert data["type"] == "hfs_risk" and data["location_id"] == "lab" and data["severity"] == severity
        assert data["payload"]["mode"] == mode and data["payload"]["level"] == "critical"


# --- location state ---------------------------------------------------------------------------------------------------

def hfs_event(level, overall):
    return ev("hfs_risk", {"risk1": overall, "risk2": 0.1, "risk3": None, "overall": overall, "level": level,
                           "interval": [overall - 0.05, overall], "partial": True, "missing": ["audio"],
                           "fired_rules_by_level": {"gas": [{"id": "G1", "explanation": "gas_level = 1.1 → critical"}],
                                                    "top": []}},
              severity="critical" if level == "critical" else "info", device="hub:hfs")


@pytest.mark.parametrize("decides, expected", [(True, DANGER), (False, SAFE)])
def test_location_state_follows_hfs_only_in_autonomous(decides, expected):
    engine = LocationStateEngine(lambda loc: loc, lambda loc: {}, lambda: [], {"hfs_decides": decides})
    engine.on_event(hfs_event("critical", 0.85), 100.0)
    assert engine.live["lab"].state == expected
    summary = engine.summary("lab", 100.0)["hfs"]
    assert summary["risk1"] == 0.85 and summary["missing"] == ["audio"]
    if decides:
        assert engine.live["lab"].reason.startswith("HFS critical 0.85 (gas 0.85, camera 0.10; no audio): G1")
        engine.on_event(hfs_event("low", 0.1), 101.0)
        assert engine.live["lab"].state == SAFE


# --- mode switch ------------------------------------------------------------------------------------------------------

def test_apply_mode_switches_cloud_paths_off():
    config = {"mode": "autonomous", "llm": {"enabled": True}, "audio": {"mic": {"whisper_backend": "azure"}},
              "rules": {"suggestions": {"enabled": True}}, "aggregator": {"global_snapshot_min": 10}}
    out = apply_mode(config)
    assert out["llm"]["enabled"] is False and out["rules"]["suggestions"]["enabled"] is False
    assert out["audio"]["mic"]["whisper_backend"] == "local" and out["aggregator"]["queue_for_llm"] is False
    assert out["hfs"]["enabled"] is True and config["llm"]["enabled"] is True        # input untouched
    assert apply_mode({"llm": {"enabled": True}}) == {"llm": {"enabled": True}}       # hybrid by default
    assert mode_of({}) == "hybrid" and is_autonomous({"mode": "Autonomous"})
    with pytest.raises(ValueError):
        mode_of({"mode": "cloud"})


AUTONOMOUS_RUN = r"""
import socket, sys
blocked = []
def deny(self, address, *a, **kw):
    host = address[0] if isinstance(address, tuple) else address
    if isinstance(host, str) and (host.startswith("127.") or host in ("::1", "localhost") or "/" in host):
        return real_connect(self, address, *a, **kw)
    blocked.append(host)
    raise OSError(f"network disabled in this test: {host}")
real_connect = socket.socket.connect
socket.socket.connect = deny
socket.socket.connect_ex = deny
socket.create_connection = lambda address, *a, **kw: deny(None, address)

import json, time
from core.config import load_config
from core.events import make_event
from core.fuzzy import FuzzyEngine, load_yaml
from core.mode import LLM_PACKAGES
from core.orchestrator import Orchestrator
from modules.hfs.worker import HierarchicalFuzzySystem

config = load_config("config.yaml", env_file="/nonexistent/.env")
config["mode"] = "autonomous"
config["trust"]["registry_db"] = ":memory:"
config["orchestrator"]["log_events"] = False
config["network"]["discovery"] = False
specs = Orchestrator.module_specs(config)
orch = Orchestrator(config, [])
assert orch.config["audio"]["mic"]["whisper_backend"] == "local"
system = HierarchicalFuzzySystem(orch.config)
for event in (make_event("gas", "gas_reading", {"gas": "ammonia", "concentration_pct": 3.5}, location_id="lab"),
              make_event("audio", "speech", {"text": "Допоможіть!", "speaker_status": "unknown"}, location_id="lab"),
              make_event("audio", "sound", {"rms_db": -5}, location_id="lab")):
    orch._route(event.to_json())                     # trust scoring in the router
    system.observe(event)
payloads = system.tick()
loaded = sorted(m for m in sys.modules if any(m == p or m.startswith(p + ".") for p in LLM_PACKAGES))
print(json.dumps({"specs": [s.name for s in specs], "loaded": loaded, "blocked": blocked,
                  "level": payloads[0]["level"], "missing": payloads[0]["missing"]}))
"""


def test_autonomous_mode_runs_without_network_or_llm_imports():
    out = subprocess.run([sys.executable, "-c", AUTONOMOUS_RUN], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert "llm" not in result["specs"] and "hfs" in result["specs"]
    assert result["loaded"] == [], f"LLM modules imported in autonomous mode: {result['loaded']}"
    assert result["blocked"] == []
    assert result["level"] == "critical" and result["missing"] == ["camera"]
    assert set(LLM_PACKAGES) >= {"modules.llm", "openai"}
