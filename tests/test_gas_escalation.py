"""A8 (TOD-50) multi-gas escalation: the matrix, evidence for critical, exceptions, the Ukrainian explanation."""
import itertools
import os

import pytest

from core.events import make_event
from core.fuzzy import FuzzyEngine, load_yaml
from modules.hfs.gas import GasSubsystem
from modules.hfs.gas_normalize import GasTable
from modules.hfs.policy import LEVELS, gas

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOW, MEDIUM, HIGH, CRITICAL = range(4)
# a crisp value at the peak of every term (μ = 1 there): the engine must give exactly the table level
PEAKS = {"max_norm": [0.05, 0.3, 0.6, 1.1], "second_norm": [0.0, 0.075, 0.3, 0.9], "group_index": [0.5, 2.0],
         "third_norm": [0.0, 0.9], "dynamics": [-0.3, 0.0, 0.1, 0.4]}


# --- the matrix ---------------------------------------------------------------------------------------------------

def test_matrix_of_the_issue():
    """rows: second gas none / low / medium / high, columns: highest gas low … critical (no evidence, steady)"""
    matrix = [[gas(m, s, 0, 0, 1) for m in range(4)] for s in range(4)]
    assert matrix == [[LOW, MEDIUM, HIGH, CRITICAL],      # спокійно: the highest gas alone
                      [MEDIUM, HIGH, HIGH, CRITICAL],     # низький: one step up, high stops at high
                      [MEDIUM, HIGH, HIGH, CRITICAL],
                      [MEDIUM, HIGH, HIGH, CRITICAL]]
    assert gas(HIGH, 1, 1, 0, 1) == CRITICAL               # group index ≥ 1
    assert gas(HIGH, 3, 0, 1, 1) == CRITICAL               # three gases at high
    assert gas(MEDIUM, 1, 0, 0, 1) == HIGH and gas(MEDIUM, 1, 1, 0, 1) == HIGH     # evidence only lifts the cap
    assert gas(HIGH, 0, 0, 0, 3) == CRITICAL               # a sharp rise is one step as before (TOD-42)


def test_every_cell_through_the_engine():
    engine = FuzzyEngine.from_dict(load_yaml(os.path.join(REPO, "config", "fuzzy", "gas.yaml")))
    names = [v for v in engine.variables if v != engine.output]
    assert names == list(PEAKS)
    for cell in itertools.product(*(range(len(PEAKS[n])) for n in names)):
        result = engine.infer({n: PEAKS[n][i] for n, i in zip(names, cell, strict=True)})
        terms = engine.fuzzify(engine.output, result["crisp"])
        assert max(terms, key=terms.get) == LEVELS[gas(*cell)], cell


def test_second_norm_calm_band_matches_gases_yaml():
    engine = FuzzyEngine.from_dict(load_yaml(os.path.join(REPO, "config", "fuzzy", "gas.yaml")))
    calm = GasTable.load().calm_norm
    assert engine.fuzzify("second_norm", calm)["low"] == 1.0 and engine.fuzzify("second_norm", 0.0)["none"] == 1.0


# --- the subsystem -------------------------------------------------------------------------------------------------

def reading(gas_name, value, unit="ppm", device="dev-1", selectivity=None, location="lab"):
    payload = {"gas": gas_name, "concentration": value, "unit": unit}
    if selectivity:
        payload["readings"] = [{**payload, "selectivity": selectivity}]
    return make_event("gas", "gas_reading", payload, location_id=location, device_id=device)


def run(*events, config=None, table=None, rules=None):
    subsystem = GasSubsystem(config or {}, table=table, rules=rules or (lambda location_id: {}))
    for event in events:
        subsystem.observe(event, 1000.0)
    subsystem.clock = lambda: 1000.0
    return subsystem.evaluate("lab", 1000.0, True)


def test_one_active_gas_keeps_its_level():
    for value in (45, 90, 180, 300):
        alone = run(reading("ammonia", value))
        with_calm = run(reading("ammonia", value), reading("CO", 30, device="co"))     # CO 0.025: calm
        assert with_calm["level"] == alone["level"] and with_calm["inputs"]["second_norm"] == 0, value
        assert with_calm["escalation"]["counted"] == ["ammonia"]
        assert with_calm["escalation"]["levels"]["carbon_monoxide"] == "спокійно"


def test_ammonia_low_and_co_medium_give_high_with_the_explanation():
    result = run(reading("ammonia", 45), reading("CO", 504, device="co"))
    assert result["level"] == "high"
    text = result["explanation"]
    assert text.startswith("CO: середній (42 % від критичного рівня), аміак: низький (15 %)")
    assert "Два гази одночасно вище норми — рівень підвищено на один щабель: високий." in text
    assert "Критичний не присвоєно: жоден газ не досягає критичного рівня" in text
    assert "«задушливі» 0,42 < 1 (гази різних груп не додаються)" in text


def test_group_index_below_one_never_critical_at_one_critical():
    below = run(reading("ammonia", 180), reading("chlorine", 3.4, device="cl"))      # 0.6 + 0.34 = 0.94
    above = run(reading("ammonia", 180), reading("chlorine", 4.2, device="cl"))      # 0.6 + 0.42 = 1.02
    assert below["inputs"]["group_index"] == pytest.approx(0.94) and below["level"] == "high"
    assert above["inputs"]["group_index"] == pytest.approx(1.02) and above["level"] == "critical"
    assert "індекс суміші групи «подразники дихальних шляхів» 1,02 ≥ 1" in above["explanation"]
    for a, b in itertools.product((60, 120, 170), (1.0, 2.0, 3.0)):                    # every pair below 1
        if a / 300 + b / 10 < 0.99:
            assert run(reading("ammonia", a), reading("chlorine", b, device="cl"))["level"] != "critical", (a, b)


def test_three_gases_at_high_are_critical():
    result = run(reading("ammonia", 180), reading("CO", 720, device="co"), reading("pm2_5", 140, "ug_m3",
                                                                                    device="pm"))
    assert result["inputs"]["third_norm"] == pytest.approx(0.6, abs=0.03) and result["level"] == "critical"
    assert "три гази на високому рівні" in result["explanation"]
    two = run(reading("ammonia", 180), reading("CO", 720, device="co"))
    assert two["level"] == "high"


def test_cross_sensitive_readings_without_confirmation_do_not_escalate():
    mos = run(reading("ammonia", 45, selectivity="cross_sensitive"),
              reading("CO", 504, device="mq7", selectivity="cross_sensitive"))
    alone = run(reading("CO", 504, device="mq7", selectivity="cross_sensitive"))
    assert mos["level"] == alone["level"] == "medium" and mos["escalation"]["suppressed"] == "cross_sensitive"
    assert "крос-чутливих сенсорів" in mos["explanation"]
    confirmed = run(reading("ammonia", 45, selectivity="cross_sensitive"),
                    reading("CO", 504, device="mq7", selectivity="cross_sensitive"),
                    reading("CO", 500, device="ec-co", selectivity="selective"))
    assert confirmed["level"] == "high" and confirmed["escalation"]["suppressed"] is None


def test_benign_co_occurrence_is_shown_but_not_counted():
    table = GasTable.load()
    table.benign = [{"gases": ["co2"], "when": "people_present"}]
    events = (reading("ammonia", 45), reading("co2", 16000, device="co2"))             # CO₂ 0.4 c_crit
    nobody = run(*events, table=table)
    person = make_event("camera", "object_detected", {"class": "person", "track_id": 1, "state": "appeared"},
                        location_id="lab", device_id="cam")
    people = run(person, *events, table=table)
    assert nobody["level"] == "high" and nobody["escalation"]["excluded"] == {}
    assert people["level"] == "medium" and set(people["escalation"]["excluded"]) == {"carbon_dioxide"}
    assert "CO₂ не враховано в підвищенні: відомий нешкідливий збіг — у приміщенні є люди." in people["explanation"]
    rule = run(*events, rules=lambda location_id: {"gas_benign": [{"gases": ["carbon_dioxide"], "when": "always",
                                                                   "reason": "вентиляція вимкнена вночі"}]})
    assert rule["level"] == "medium" and rule["escalation"]["excluded"] == {"carbon_dioxide":
                                                                         "вентиляція вимкнена вночі"}


def test_sensor_noise_in_the_calm_band_is_not_a_second_gas():
    noisy = make_event("gas", "gas_reading", {"readings": [{"gas": "ammonia", "concentration": 20, "unit": "ppm",
                                                            "accuracy": {"absolute": 5.0}}]},
                       location_id="lab", device_id="nh3")             # 0.067 − 0.017 = 0.05: calm
    result = run(reading("CO", 504, device="co"), noisy)
    assert result["escalation"]["levels"]["ammonia"] == "спокійно" and result["level"] == "medium"
    clear = make_event("gas", "gas_reading", {"readings": [{"gas": "ammonia", "concentration": 30, "unit": "ppm",
                                                            "accuracy": {"absolute": 5.0}}]},
                       location_id="lab", device_id="nh3")             # 0.1 − 0.017 > 0.05: active
    assert run(reading("CO", 504, device="co"), clear)["level"] == "high"


def test_saturated_reading_still_counts_as_active():
    full_scale = make_event("gas", "gas_reading", {"readings": [{"gas": "ammonia", "concentration": 100,
                                                                 "unit": "ppm", "range": [0, 100], "resolution": 1,
                                                                 "accuracy": {"relative": 0.1, "absolute": 5.0}}]},
                            location_id="lab", device_id="nh3")
    result = run(full_scale, reading("CO", 504, device="co"))
    assert result["saturated"] == ["ammonia"] and result["escalation"]["counted"] == ["carbon_monoxide", "ammonia"]


def test_config_effect_groups_and_co2():
    table = GasTable.load()
    co2 = table.get("CO2")
    assert (co2.c_crit, co2.unit, co2.group) == (40000, "ppm", "asphyxiant")
    assert {s.group for s in table.substances.values()} <= set(table.groups)
    assert table.get("NH3").group == table.get("Cl2").group == "irritant"


def test_explanation_reaches_the_hfs_risk_event():
    from modules.hfs.worker import HierarchicalFuzzySystem

    clock = [1000.0]
    system = HierarchicalFuzzySystem({}, devices=lambda: [], clock=lambda: clock[0])
    system.observe(reading("ammonia", 45))
    system.observe(reading("CO", 504, device="co"))
    [payload] = system.tick()
    gas_part = payload["subsystems"]["gas"]
    assert gas_part["level"] == "high" and "підвищено на один щабель" in gas_part["explanation"]
    assert gas_part["escalation"]["counted"] == ["carbon_monoxide", "ammonia"]
