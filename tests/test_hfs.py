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
from modules.hfs.audio import AudioSubsystem
from modules.hfs.camera import CameraSubsystem
from modules.hfs.gas import GasSubsystem
from modules.hfs.policy import LEVELS, POLICIES, policy_level
from modules.hfs.top import TopLevel
from modules.hfs.worker import HFSModule, HierarchicalFuzzySystem
from modules.locations.state import DANGER, SAFE, LocationStateEngine
from tools.plot_surfaces import monotonicity

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = {}


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
    assert fired["id"] == "T300" and "max_risk" in fired["explanation"] and "critical" in fired["explanation"]
    assert result["explanation"].startswith("Газ: критичний")


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
    report = monotonicity(engine, 7 if name == "audio" else 11, nodes=name == "gas")      # gas: 5 inputs
    assert all(stats["falls"] == 0 for stats in report.values()), report


def test_top_policy_matches_the_requirements():
    """TOD-41 rules with quiet (calm) other subsystems, plus TOD-51: active subsystems escalate one step."""
    assert policy_level(["critical", "calm", "calm"]) == "critical"
    assert policy_level(["high", "high", "calm"]) == "critical"          # two high
    assert policy_level(["high", "calm", "calm"]) == "high"
    assert policy_level(["medium", "medium", "calm"]) == "high"          # two medium
    assert policy_level(["calm", "medium", "calm"]) == "medium"
    assert policy_level(["calm", "calm", "calm"]) == "low"
    assert policy_level(["low", "low", "calm"]) == "medium"              # TOD-51: two active lows
    assert policy_level(["medium", "low", "calm"]) == "high"
    assert policy_level(["high", "low", "calm"]) == "high"               # capped without evidence
    assert policy_level(["high", "low", "medium"]) == "critical"         # all three active, one high


# --- subsystems -------------------------------------------------------------------------------------------------------

def test_gas_subsystem_normalises_to_critical_concentration_and_goes_stale():
    clock = Clock()
    gas = GasSubsystem(CONFIG, clock=clock)
    gas.observe(ev("gas_reading", {"gas": "ammonia", "concentration": 180, "unit": "ppm"}), clock())
    result = gas.evaluate("lab", clock(), True)
    assert result["inputs"]["max_norm"] == pytest.approx(0.6) and result["level"] == "high"
    assert result["substances"] == {"ammonia": 0.6}
    gas.observe(ev("gas_reading", {"gas": "toluol", "concentration_pct": 0.25}), clock())     # graphene, vol %
    result = gas.evaluate("lab", clock(), True)                 # 2500 ppm vs IDLH 500 ppm → clipped to 1.2
    assert result["inputs"]["max_norm"] == 1.2 and result["level"] == "critical"
    assert result["substances"] == {"toluene": 1.2}             # the sensor's new state replaced ammonia
    gas.observe(ev("gas_reading", {"gas": "clean_air", "concentration_pct": 0}), clock())
    assert gas.evaluate("lab", clock(), True)["level"] == "low"
    assert gas.evaluate("lab", clock() + 31, True) is None                  # stale → no data
    gas.forget_device("dev-1")
    assert gas.evaluate("lab", clock(), True) is None


def test_any_single_substance_at_its_critical_level_is_critical():
    for gas_name, value, unit in (("ammonia", 300, "ppm"), ("carbon_monoxide", 1380, "mg_m3"),
                                  ("pm2_5", 225.5, "ug_m3"), ("ammonia", 0.03, "vol_pct"), ("CO", 5000, "ppm")):
        gas = GasSubsystem(CONFIG)
        gas.observe(ev("gas_reading", {"gas": gas_name, "concentration": value, "unit": unit}), 1000.0)
        result = gas.evaluate("lab", 1000.0, True)
        assert result["inputs"]["max_norm"] >= 0.999 and result["level"] == "critical", (gas_name, result)


def test_two_gases_of_one_effect_group_add_up_others_only_raise_one_step():
    """TOD-50 (replaces the TOD-42 Σ over every gas): the additive index only inside an effect group."""
    one, two = GasSubsystem(CONFIG), GasSubsystem(CONFIG)
    one.observe(ev("gas_reading", {"gas": "ammonia", "concentration": 180, "unit": "ppm"}), 1000.0)
    two.observe(ev("gas_reading", {"readings": [{"gas": "ammonia", "concentration": 180, "unit": "ppm"},
                                                {"gas": "CO", "concentration": 720, "unit": "ppm"}]}), 1000.0)
    single, mixed = one.evaluate("lab", 1000.0, True), two.evaluate("lab", 1000.0, True)
    assert mixed["inputs"]["second_norm"] == pytest.approx(0.6) and mixed["inputs"]["group_index"] == 0.6
    assert (single["level"], mixed["level"]) == ("high", "high")      # irritant + asphyxiant: capped at high
    same_group = GasSubsystem(CONFIG)                                  # NH₃ + Cl₂: both irritants, Σ = 1.2
    same_group.observe(ev("gas_reading", {"gas": "ammonia", "concentration": 180, "unit": "ppm"}), 1000.0)
    same_group.observe(ev("gas_reading", {"gas": "chlorine", "concentration": 6, "unit": "ppm"}, device="cl-1"),
                       1000.0)
    result = same_group.evaluate("lab", 1000.0, True)
    assert result["inputs"]["group_index"] == pytest.approx(1.2) and result["level"] == "critical"
    assert result["escalation"]["groups"] == {"irritant": 1.2}


def test_dynamics_is_the_slope_of_max_norm_over_the_window():
    gas = GasSubsystem(CONFIG)
    for second in range(0, 61, 2):                              # 30 → 180 ppm NH₃ in 60 s: +0.5 c_crit / min
        gas.observe(ev("gas_reading", {"gas": "ammonia", "concentration": 30 + 2.5 * second, "unit": "ppm"}),
                    1000.0 + second)
    result = gas.evaluate("lab", 1060.0, True)
    assert result["inputs"]["dynamics"] == pytest.approx(0.5, abs=1e-3)
    assert result["level"] == "critical"                        # dangerous (0.6) + sharp rise
    steady = GasSubsystem(CONFIG)
    for second in range(0, 61, 2):
        steady.observe(ev("gas_reading", {"gas": "ammonia", "concentration": 180, "unit": "ppm"}), 1000.0 + second)
    assert steady.evaluate("lab", 1060.0, True)["inputs"]["dynamics"] == 0
    early = GasSubsystem(CONFIG)                                # one sample: no trend yet
    early.observe(ev("gas_reading", {"gas": "ammonia", "concentration": 180, "unit": "ppm"}), 1000.0)
    assert early.evaluate("lab", 1000.0, True)["inputs"]["dynamics"] == 0


def test_gas_rules_from_the_issue():
    gas = GasSubsystem(CONFIG)
    level = lambda **inputs: gas.infer(inputs)["level"]  # noqa: E731
    assert level(max_norm=1.0) == "critical"
    assert level(max_norm=0.3, second_norm=0.15) == "high"        # a second gas raises one step below c_crit
    assert level(max_norm=0.6, group_index=0.6, dynamics=0.1) == "high"
    assert level(max_norm=0.3, dynamics=0.3) == "high"            # moderate + sharp rise
    assert level(max_norm=0.05) == "low"


def test_camera_subsystem_strangers_at_night_vs_known_by_day():
    clock = Clock()
    night = CameraSubsystem({}, clock=clock, local_now=lambda: datetime(2026, 11, 6, 2, 0))
    assert night.evaluate("lab", clock(), available=False) is None           # no camera, no events
    night.observe(ev("object_detected", {"class": "person", "track_id": 1, "state": "appeared"}), clock())
    night.observe(ev("person_identified", {"track_id": 1, "identity": "unknown_3", "person_id": "p3"}), clock())
    result = night.evaluate("lab", clock(), available=True)
    assert result["inputs"]["unknown_share"] == 1 and result["inputs"]["activity_time"] == 6
    assert result["level"] == "high"                                         # only a stranger, after hours
    night.observe(ev("motion_level", {"intensity": 0.2}), clock())
    assert night.evaluate("lab", clock(), available=True)["level"] == "critical"   # … moving a lot
    day = CameraSubsystem({}, clock=clock, local_now=lambda: datetime(2026, 11, 6, 12, 0),
                          rules=lambda loc: {"activity_hours": [["08:00", "19:00"]]})
    day.observe(ev("object_detected", {"class": "person", "track_id": 1, "state": "appeared"}), clock())
    day.observe(ev("person_identified", {"track_id": 1, "identity": "Oleh", "person_id": "p1"}), clock())
    assert day.evaluate("lab", clock(), available=True)["level"] == "low"   # nobody unknown, working hours


def test_unknown_share_counts_known_and_unknown_only():
    clock = Clock()
    camera = CameraSubsystem({}, clock=clock)
    assert camera.unknown_share("lab", clock()) == 0                         # nobody
    camera.observe(ev("person_identified", {"track_id": 1, "identity": "Oleh", "person_id": "p1"}), clock())
    camera.observe(ev("person_identified", {"track_id": 2, "identity": "unknown_4", "person_id": "p4"}), clock())
    assert camera.unknown_share("lab", clock()) == 0.5                       # 1 known + 1 unknown
    camera.observe(ev("object_detected", {"class": "person", "track_id": 3, "state": "appeared"}), clock())
    assert camera.unknown_share("lab", clock()) == 0.5                       # not decided yet: not counted …
    assert camera.unknown_share("lab", clock() + 11) == pytest.approx(2 / 3, abs=1e-3)   # … hidden face → unknown
    camera.observe(ev("object_detected", {"class": "person", "track_id": 2, "state": "disappeared"}), clock())
    camera.observe(ev("object_detected", {"class": "person", "track_id": 3, "state": "disappeared"}), clock())
    assert camera.unknown_share("lab", clock()) == 0


def test_unknown_share_interval_from_identification_accuracy():
    from modules.hfs.camera import binomial_quantile, identification_accuracy, share_interval

    p, source = identification_accuracy(os.path.join(REPO, "reports", "identity_eval_lfw.json"), 0.45)
    assert 0.99 < p < 1.0 and "650" in source                                # Wilson bound of 650 / 650 correct
    assert binomial_quantile(2, 0.006, 0.95) == 0 and binomial_quantile(20, 0.1, 0.95) == 4
    assert share_interval(1, 2, 0.006) == (0.5, 0.5)                          # two people, 99.4 %: no doubt
    assert share_interval(5, 20, 0.1) == (0.05, 0.45)                        # a crowd at 90 %: ±4 identities
    assert share_interval(0, 0, 0.1) == (0.0, 0.0)
    camera = CameraSubsystem({"hfs": {"camera": {"identification_accuracy": 0.9}}})
    for track in range(20):
        identity = "unknown_1" if track < 5 else "Oleh"
        camera.observe(ev("person_identified", {"track_id": track, "identity": identity, "person_id": f"p{track}"}),
                       1000.0)
    result = camera.evaluate("lab", 1000.0, True)
    assert result["input_intervals"] == {"unknown_share": [0.05, 0.45]} and result["persons"]["total"] == 20


def test_camera_motion_intensity_from_motion_level():
    camera = CameraSubsystem({})
    camera.observe(ev("motion", {"state": "start", "area_ratio": 0.03}), 1000.0)
    assert camera.motion_intensity("lab", 1000.0) == 0.03                    # no motion_level yet: the motion now
    camera.observe(ev("motion_level", {"intensity": 0.12}), 1001.0)
    camera.observe(ev("motion_level", {"intensity": 0.02}, device="cam-2"), 1001.0)
    assert camera.motion_intensity("lab", 1002.0) == 0.12                    # the busiest camera
    camera.observe(ev("motion", {"state": "end"}), 1003.0)
    assert camera.motion_intensity("lab", 1020.0) == 0                       # motion_level stale, motion ended


def test_camera_activity_time_uses_the_location_schedule():
    rules = {"office": {"object_type": "office"}, "plant": {"object_type": "factory"}}
    presets = {"office": {"hours": [["08:00", "19:00"]], "days": [0, 1, 2, 3, 4]},
               "factory": {"hours": [["06:00", "14:00"], ["14:00", "22:00"], ["22:00", "06:00"]]}}
    saturday_noon = datetime(2026, 11, 7, 12, 0)
    camera = CameraSubsystem({"hfs": {"camera": {"schedules": presets}}}, rules=lambda loc: rules.get(loc, {}),
                             local_now=lambda: saturday_noon)
    assert camera.activity_time("office") == 12                              # weekend: off hours
    assert camera.activity_time("plant") == -12                              # three shifts: always working
    assert camera.activity_time("lab") == -4                                 # default 08–19: 4 h before closing


def test_audio_critical_word_unknown_voice_and_window():
    clock = Clock()
    audio = AudioSubsystem({}, clock=clock)
    audio.observe(ev("speech", {"text": "Пожежа на складі!", "speaker_status": "known"}), clock())
    result = audio.evaluate("lab", clock(), available=True)
    assert result["inputs"]["keyword_danger"] == 1 and result["level"] == "critical"
    assert result["danger_words"][0]["entry"] == "пожежа"
    assert audio.evaluate("lab", clock() + 31, available=True)["level"] == "low"   # window passed
    voice = AudioSubsystem({}, clock=clock)
    voice.observe(ev("audio_level", {"intensity_db": 12}), clock())
    voice.observe(ev("speech", {"text": "Допоможіть!", "speaker_status": "unknown"}), clock())
    assert voice.evaluate("lab", clock(), available=True)["level"] == "high"        # dangerous words, unknown voice
    quiet = AudioSubsystem({}, clock=clock)
    quiet.observe(ev("speech", {"text": "добрий день", "speaker_status": "known"}), clock())
    assert quiet.evaluate("lab", clock(), available=True)["level"] == "low"


def test_audio_sound_type_and_intensity():
    clock = Clock()
    audio = AudioSubsystem({}, clock=clock)
    audio.observe(ev("sound_type", {"type": "unknown", "confidence": 0, "danger": None}), clock())   # no model
    assert audio.evaluate("lab", clock(), available=True)["inputs"]["sound_type_danger"] == 0
    audio.observe(ev("sound_type", {"type": "scream", "confidence": 0.9, "danger": 0.8}), clock())
    audio.observe(ev("sound", {"rms_db": -8, "relative_db": 26}), clock())
    result = audio.evaluate("lab", clock(), available=True)
    assert result["inputs"]["intensity"] == 26 and result["sound_type"] == "scream" and result["level"] == "high"
    audio.observe(ev("audio_level", {"intensity_db": 4}), clock())                   # the level wins over sounds
    assert audio.evaluate("lab", clock(), available=True)["level"] == "medium"       # a quiet scream: medium


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
    assert second["fired_rules_by_level"]["gas"][0]["id"] == "G30101"       # critical, alone, steady
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
loaded += sorted(m for m in ("modules.trust.registry", "modules.trust.scorer") if m in sys.modules)   # A5: no Beta
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
