"""A9 (TOD-51) cross-subsystem escalation: the matrix, evidence for critical, window, location, exceptions."""
import itertools
import os
from datetime import datetime

from core.events import make_event
from core.fuzzy import FuzzyEngine, load_yaml
from modules.hfs.policy import LEVELS, top
from modules.hfs.top import TopLevel
from modules.hfs.top_escalation import TopEscalation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOW, MEDIUM, HIGH, CRITICAL = range(4)
PEAKS = {"max_risk": [0.1, 0.35, 0.65, 0.9], "second_risk": [0.0, 0.17, 0.35, 0.8], "third_risk": [0.0, 0.5]}
RISK = {"calm": 0.1308, "low": 0.2, "medium": 0.35, "high": 0.65, "critical": 0.87}


def sub(level, width=0.0):
    risk = RISK[level]
    return {"risk": risk, "level": level, "interval": [risk - width, risk + width], "fired_rules": [],
            "inputs": {}}


def combine(gas="calm", camera="calm", audio="calm", **kw):
    results = {name: (None if level is None else sub(level)) for name, level in
               (("gas", gas), ("camera", camera), ("audio", audio))}
    return TopLevel().combine(results, **kw)


# --- the matrix ----------------------------------------------------------------------------------------------------

def test_matrix_of_the_issue():
    assert [[top(m, s, 0) for m in range(4)] for s in range(4)] == [
        [LOW, MEDIUM, HIGH, CRITICAL],          # спокійно
        [MEDIUM, HIGH, HIGH, CRITICAL],         # низький: +1, high stays high
        [MEDIUM, HIGH, HIGH, CRITICAL],         # середній
        [MEDIUM, HIGH, CRITICAL, CRITICAL]]     # високий: two at high → critical
    assert top(HIGH, 1, 1) == CRITICAL and top(MEDIUM, 2, 1) == HIGH      # all three active + one high


def test_every_cell_through_the_engine():
    engine = FuzzyEngine.from_dict(load_yaml(os.path.join(REPO, "config", "fuzzy", "top.yaml")))
    names = [v for v in engine.variables if v != engine.output]
    assert names == list(PEAKS)
    for cell in itertools.product(*(range(len(PEAKS[n])) for n in names)):
        result = engine.infer({n: PEAKS[n][i] for n, i in zip(names, cell, strict=True)})
        terms = engine.fuzzify(engine.output, result["crisp"])
        assert max(terms, key=terms.get) == LEVELS[top(*cell)], cell


# --- the issue's cases ---------------------------------------------------------------------------------------------

def test_one_active_subsystem_keeps_its_level():
    for level in ("low", "medium", "high", "critical"):
        result = combine(gas=level)
        assert result["level"] == level and result["escalation"]["counted"] == (["gas"] if level != "calm" else [])
        assert "без підвищення" in result["explanation"] or level == "critical"


def test_two_active_subsystems_raise_one_step_with_explanations():
    low_low = combine(gas="low", audio="low")
    assert low_low["level"] == "medium" and low_low["inputs"]["second_risk"] == 0.2
    assert "Дві підсистеми одночасно сигналізують" in low_low["explanation"]
    assert "рівень підвищено на один щабель: середній" in low_low["explanation"]
    medium_low = combine(camera="medium", gas="low")
    assert medium_low["level"] == "high"
    assert medium_low["explanation"].startswith("Газ: низький, камера: середній, звук: спокійно.")
    assert "Критичний не присвоєно: жодна підсистема не критична, жодна не на рівні «високий», активні не всі " \
           "три." in medium_low["explanation"]


def test_critical_only_with_confirmation():
    assert combine(gas="high", audio="high")["level"] == "critical"
    assert "дві підсистеми на рівні «високий» і вище" in combine(gas="high", audio="high")["explanation"]
    assert combine(gas="high", audio="medium")["level"] == "high"
    assert combine(gas="high", audio="medium", camera="low")["level"] == "critical"      # all three, one high
    for a, b, c in itertools.product(("calm", "low", "medium"), repeat=3):              # nothing at high
        assert combine(gas=a, camera=b, audio=c)["level"] != "critical", (a, b, c)
    for a, b in itertools.product(("low", "medium"), ("calm", "low", "medium")):        # one high at most, two active
        assert combine(gas="high", camera=a, audio="calm")["level"] == "high", (a, b)


def test_uncertainty_alone_is_not_a_signal():
    noisy = {"gas": sub("calm", width=0.0), "camera": {**sub("low"), "interval": [0.12, 0.28]}, "audio": sub("low")}
    result = TopLevel().combine(noisy)
    assert result["escalation"]["counted"] == ["audio"] and result["level"] == "low"     # camera's lower bound calm


# --- same place, same time ------------------------------------------------------------------------------------------

def test_co_occurrence_window():
    esc = TopEscalation({"co_window_sec": 60})
    calm, active = sub("calm"), sub("medium")
    esc.analyse({"gas": calm, "camera": calm, "audio": active}, "store", now=0.0)          # a scream at t = 0
    within = esc.analyse({"gas": sub("low"), "camera": calm, "audio": calm}, "store", now=40.0)
    assert within.counted == ["audio", "gas"] and within.payload()["held"] == {"audio": 40.0}
    result = TopLevel().combine({"gas": sub("low"), "camera": calm, "audio": calm}, within)
    assert result["level"] == "high" and "протягом 40 с" in result["explanation"]
    later = esc.analyse({"gas": sub("low"), "camera": calm, "audio": calm}, "store", now=100.0)
    assert later.counted == ["gas"]                                                          # 100 s apart: unrelated
    other_place = esc.analyse({"gas": sub("low"), "camera": calm, "audio": calm}, "office", now=41.0)
    assert other_place.counted == ["gas"]                                                    # another location


def test_signals_of_different_locations_are_not_combined():
    from modules.hfs.worker import HierarchicalFuzzySystem

    clock = [1000.0]
    system = HierarchicalFuzzySystem({}, devices=lambda: [], clock=lambda: clock[0],
                                     local_now=lambda: datetime(2026, 11, 4, 11, 0))
    system.observe(make_event("gas", "gas_reading", {"gas": "ammonia", "concentration": 60, "unit": "ppm"},
                              location_id="store", device_id="nh3"))
    system.observe(make_event("audio", "speech", {"text": "допоможіть", "speaker_status": "known"},
                              location_id="office", device_id="mic"))
    payloads = {p["location_id"]: p for p in system.tick()}
    assert payloads["store"]["escalation"]["counted"] == ["gas"] and payloads["store"]["missing"] == ["camera",
                                                                                                        "audio"]
    assert "gas" not in payloads["office"]["escalation"]["counted"] and payloads["office"]["missing"] == ["gas",
                                                                                                          "camera"]
    # the call for help is high but uncertain (danger-word δ 0.465, TOD-45): its lower bound is calm → not counted
    assert payloads["office"]["level"] == "high" and payloads["office"]["escalation"]["counted"] == []


# --- exceptions -------------------------------------------------------------------------------------------------------

def test_benign_combination_inside_vs_outside_activity_hours():
    esc = TopEscalation({"benign": {"default": [{"subsystems": ["camera", "audio"], "max_level": "low"}]}})
    results = {"gas": sub("calm"), "camera": sub("low"), "audio": sub("low")}
    inside = esc.analyse(results, "office", 0.0, inside_hours=True)
    outside = esc.analyse(results, "office", 0.0, inside_hours=False)
    assert inside.counted == ["camera"] and inside.benign and outside.counted == ["camera", "audio"]
    level = lambda a: TopLevel().combine(results, a)["level"]                     # noqa: E731
    assert (level(inside), level(outside)) == ("low", "medium")
    assert "у робочі години — звичайний збіг — без підвищення: низький" in TopLevel().combine(results, inside)[
        "explanation"]
    medium = {**results, "audio": sub("medium")}                                  # above max_level → escalates
    assert esc.analyse(medium, "office", 0.0, inside_hours=True).counted == ["audio", "camera"]
    gas = {**results, "gas": sub("low")}                                          # gas is not in the combination
    assert len(esc.analyse(gas, "office", 0.0, inside_hours=True).counted) == 3


def test_benign_with_the_real_schedule():
    from modules.hfs.worker import HierarchicalFuzzySystem

    def system_at(hour):
        clock = [1000.0]
        system = HierarchicalFuzzySystem({"hfs": {"top": {"benign": {"default": [
            {"subsystems": ["camera", "audio"], "max_level": "low"}]}}}}, devices=lambda: [], clock=lambda: clock[0],
            local_now=lambda: datetime(2026, 11, 4, hour, 0))
        return system

    results = {"gas": None, "camera": sub("low"), "audio": sub("low")}
    day = system_at(11).combine("lab", results, 1000.0)
    night = system_at(23).combine("lab", results, 1000.0)
    assert (day["level"], night["level"]) == ("low", "medium")


def test_correlated_channels_of_one_device_count_once():
    esc = TopEscalation({"correlated": [{"node": "phone-7"}]})
    results = {"gas": sub("calm"), "camera": sub("low"), "audio": sub("medium")}
    devices = {"camera": {"phone-7-cam"}, "audio": {"phone-7-mic"}, "gas": set()}
    nodes = {"phone-7-cam": "phone-7", "phone-7-mic": "phone-7"}
    once = esc.analyse(results, "lab", 0.0, devices=devices, nodes=nodes)
    assert once.counted == ["audio"] and once.merged == {"camera": "audio"}
    result = TopLevel().combine(results, once)
    assert result["level"] == "medium" and "від одного пристрою: враховано один раз" in result["explanation"]
    two_devices = esc.analyse(results, "lab", 0.0, devices={**devices, "camera": {"cam-1"}}, nodes=nodes)
    assert two_devices.counted == ["audio", "camera"]


def test_offline_subsystem_escalates_over_the_rest_and_is_partial():
    result = combine(gas="low", camera=None, audio="low")
    assert result["partial"] is True and result["missing"] == ["camera"] and result["level"] == "medium"
    assert "камера: немає даних" in result["explanation"] and "результат частковий" in result["explanation"]
    assert combine(gas=None, camera=None, audio=None) is None
