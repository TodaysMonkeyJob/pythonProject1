"""G1 (TOD-53): one gas scale (C_crit of config/gases.yaml), the explosion flag and the rule-independent safety
invariant — any gas whose lower bound is ≥ C_crit gives critical, in autonomous and multiagent mode."""
import copy
import os
import random
import re

import pytest
import yaml

from agents.framework.orchestrator import AgentHost
from core.events import make_event
from modules.hfs import gas_safety
from modules.hfs.gas_normalize import GasTable
from modules.hfs.gas_safety import VIOLATIONS, assess, summary
from modules.hfs.worker import HierarchicalFuzzySystem
from modules.locations.state import DANGER, SAFE, LocationStateEngine
from tests.test_agents import Bus, registry

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = yaml.safe_load(open(os.path.join(REPO, "config.yaml"), encoding="utf-8"))
GASES = os.path.join(REPO, "config", "gases.yaml")
TABLE = GasTable.load(GASES)


def hfs_payload(readings, config=None, location="lab"):
    """:param readings: one gas_reading payload per sensor → the location's hfs_risk payload"""
    now = [1000.0]
    hfs = HierarchicalFuzzySystem(config or CONFIG, clock=lambda: now[0])
    for i, payload in enumerate(readings):
        hfs.observe(make_event("gas", "gas_reading", payload, device_id=f"gas-{i}", location_id=location))
    (payload,) = hfs.tick()
    return payload


# --- config/gases.yaml ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,c_crit,unit,lel", [
    ("ammonia", 300, "ppm", 15.0), ("acetone", 2500, "ppm", 2.5), ("ethanol", 3300, "ppm", 3.3),
    ("toluene", 500, "ppm", 1.1), ("carbon_monoxide", 1200, "ppm", 12.5), ("carbon_dioxide", 40000, "ppm", None),
    ("pm2_5", 225.5, "ug_m3", None)])
def test_gases_have_critical_concentration_lel_and_sources(name, c_crit, unit, lel):
    raw = yaml.safe_load(open(GASES, encoding="utf-8"))["gases"][name]
    substance = TABLE.substances[name]
    assert (substance.c_crit, substance.unit, substance.lel_pct) == (c_crit, unit, lel)
    assert raw["source"].strip()
    assert (lel is None) == ("lel_source" not in raw)
    if lel is not None:
        assert "niosh" in raw["lel_source"].lower()


def test_every_graphene_model_output_maps_to_a_substance():
    for gas in ("acetone", "ammonia", "ethanol", "toluol"):         # TOD-19 classifier labels (vol %)
        assert TABLE.get(gas) is not None, gas
    assert TABLE.get("toluol").name == "toluene" and TABLE.get("clean_air") is None


# --- the issue's cases ------------------------------------------------------------------------------------------------

def test_5_vol_pct_acetone_is_critical_and_explosive(monkeypatch):
    logged = []
    monkeypatch.setattr(gas_safety.logger, "error", logged.append)
    p = hfs_payload([{"gas": "acetone", "concentration_pct": 5.0}])
    assert p["level"] == "critical" and p["explosive"] is True
    assert p["safety"]["critical"] and p["safety"]["gases"][0]["ratio"] == 20.0
    assert "вибухонебезпечно" in p["explanation"] and "НКМПР 2,5 об.%" in p["explanation"]
    assert p["subsystems"]["gas"]["explosive"] and p["subsystems"]["gas"]["level"] == "critical"
    assert any("вибухонебезпечно" in line for line in logged)        # the Ukrainian log line
    assert "safety_overruled" not in p                               # the rules already said critical


def test_5_vol_pct_ammonia_is_critical_with_a_critical_corridor():
    p = hfs_payload([{"gas": "ammonia", "concentration_pct": 5.0}])
    assert p["level"] == "critical" and p["explosive"] is False      # LEL of NH₃ is 15 %
    assert p["safety"]["gases"][0]["ratio"] == pytest.approx(166.6667, abs=1e-3)
    # the reading error is subtracted before clipping at 1.2 C_crit: the corridor stays critical (was [0.65, 0.87])
    assert p["subsystems"]["gas"]["interval"][0] == p["subsystems"]["gas"]["risk"]


def test_far_exceedance_with_a_relative_sensor_error_keeps_its_lower_bound():
    reading = {"readings": [{"gas": "ammonia", "concentration": 50_000, "unit": "ppm", "range": [0, 100_000],
                             "accuracy": {"relative": 0.1}}]}
    (hazard,) = assess(TABLE, reading, lambda s, v, u, item: 0.1 * v)
    assert hazard.low_ratio == pytest.approx(150.0) and hazard.critical
    p = hfs_payload([reading])
    assert p["level"] == "critical" and p["subsystems"]["gas"]["interval"][0] >= 0.85


def test_below_c_crit_and_clean_air_raise_no_flag():
    assert summary(assess(TABLE, {"gas": "acetone", "concentration_pct": 0.2})) is None      # 0.8 × C_crit
    assert summary(assess(TABLE, {"gas": "clean_air", "concentration_pct": 3.0})) is None
    assert "safety" not in hfs_payload([{"gas": "acetone", "concentration_pct": 0.02}])


# --- invariant: property test ---------------------------------------------------------------------------------------

def test_any_gas_at_or_above_c_crit_is_critical():
    rng = random.Random(53)
    names = sorted(TABLE.substances)
    for _ in range(150):
        substance = TABLE.substances[rng.choice(names)]
        ratio = rng.uniform(1.4, 300.0)                  # well above the sensor errors of config/uncertainty.yaml
        unit = substance.unit if substance.unit == "ug_m3" else rng.choice(["ppm", "vol_pct", substance.unit])
        value = ratio * substance.c_crit_in(unit) if unit != "ug_m3" else ratio * substance.c_crit
        others = [{"gas": "clean_air", "concentration_pct": 0.0}] * rng.randint(0, 2)
        p = hfs_payload([*others, {"gas": substance.name, "concentration": value, "unit": unit}])
        assert p["safety"]["critical"], (substance.name, value, unit)
        assert p["level"] == "critical", (substance.name, value, unit, p["level"])


def all_low(src, dst):
    text = open(src, encoding="utf-8").read()
    dst.write_text(re.sub(r"THEN risk IS \w+", "THEN risk IS low", text), encoding="utf-8")
    return str(dst)


@pytest.mark.safety_violation_expected
def test_a_rule_base_saying_low_is_overruled_and_logged_as_an_error(tmp_path, monkeypatch):
    errors = []
    monkeypatch.setattr(gas_safety.logger, "error", errors.append)
    config = copy.deepcopy(CONFIG)
    config["hfs"]["gas"]["rules"] = all_low(os.path.join(REPO, "config", "fuzzy", "gas.yaml"), tmp_path / "gas.yaml")
    config["hfs"].setdefault("top", {})["rules"] = all_low(os.path.join(REPO, "config", "fuzzy", "top.yaml"),
                                                           tmp_path / "top.yaml")
    before = len(VIOLATIONS)
    p = hfs_payload([{"gas": "acetone", "concentration_pct": 1.0}], config)
    assert p["level"] == "critical" and p["safety_overruled"]["by"] == "top rule base"
    assert p["overall"] >= 0.85 and min(p["interval"]) >= 0.85
    assert p["subsystems"]["gas"]["level"] == "critical"
    assert [v["by"] for v in VIOLATIONS[before:]] == ["gas rule base (lab)", "top rule base"]
    assert sum("safety invariant violated" in e for e in errors) == 2


# --- multiagent mode ----------------------------------------------------------------------------------------------

@pytest.mark.safety_violation_expected
def test_multiagent_decision_cannot_go_below_critical():
    payload = hfs_payload([{"gas": "acetone", "concentration_pct": 5.0}])
    bus = Bus()
    host = AgentHost(registry(enabled={"validation": True, "echo": False}), bus, decides=True)
    host.start()
    validator = host.registry.first("validation")
    validator.decide = lambda corridor, *a: {"risk": 0.1, "level": "low", "interval": [0.1, 0.1],
                                             "branch": "deescalation_confirmed", "reasons": ["video: all calm"]}
    host.observe(make_event("system", "hfs_risk", payload, location_id="lab", device_id="hub:hfs"))
    (decision,) = bus.of("agent_decision")
    assert decision["level"] == "critical" and decision["explosive"] is True
    assert decision["risk"] == payload["overall"] and decision["interval"][0] >= payload["overall"]
    assert decision["safety_overruled"] == {"by": "agent validation", "level": "low"}
    assert [s for t, _, s, _ in bus.events if t == "agent_decision"] == ["critical"]
    host.stop()


def test_multiagent_fuzzy_only_decision_carries_the_flag():
    payload = hfs_payload([{"gas": "ammonia", "concentration_pct": 5.0}])
    bus = Bus()
    host = AgentHost(registry(enabled={"validation": True, "echo": False}), bus)
    host.start()
    host.observe(make_event("system", "hfs_risk", payload, location_id="lab", device_id="hub:hfs"))
    (decision,) = bus.of("agent_decision")
    assert decision["level"] == "critical" and decision["safety"]["critical"] and not decision["explosive"]
    host.stop()


# --- location state: both modes, acknowledge -------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["autonomous", "multiagent"])
def test_location_state_stays_danger_in_both_modes(mode):
    devices = [{"location_id": "lab", "status": "online", "enabled": True}]
    engine = LocationStateEngine(lambda loc: loc, lambda loc: {}, lambda: devices,
                                 {"mode": lambda loc: mode, "gases": GASES})
    now = 1000.0
    low = {"overall": 0.13, "level": "low", "interval": [0.13, 0.13]}         # a decider that says low
    engine.on_event(make_event("system", "hfs_risk", low, location_id="lab", device_id="hub:hfs"), now)
    engine.on_event(make_event("system", "agent_decision", {**low, "risk": 0.13, "branch": "agree"},
                               location_id="lab", device_id="hub:agents"), now)
    assert engine.live["lab"].state == SAFE
    engine.on_event(make_event("gas", "gas_reading", {"gas": "acetone", "concentration_pct": 5.0},
                               location_id="lab", device_id="gas-1"), now + 1)
    assert engine.live["lab"].state == DANGER and "вибухонебезпечно" in engine.live["lab"].reason
    engine.acknowledge("lab", now + 2)
    assert engine.live["lab"].state == DANGER
    assert engine.summary("lab", now + 2)["gas_safety"]["explosive"]


# --- no percentage thresholds left -----------------------------------------------------------------------------------

def test_no_percentage_gas_thresholds_remain():
    gas = CONFIG["gas"]
    assert not {"thresholds", "rise_rate_threshold", "trend_stable_band"} & set(gas)
    fuzzy = yaml.safe_load(open(os.path.join(REPO, "config", "fuzzy.yaml"), encoding="utf-8"))
    concentration = fuzzy["variables"]["gas_concentration"]
    assert concentration["unit"] == "C_crit" and "variants" not in concentration and "variants_by" not in concentration
    assert fuzzy["variables"]["gas_trend"]["unit"] == "C_crit/min"
    example = open(os.path.join(REPO, "config", "locations.example.yaml"), encoding="utf-8").read()
    assert "gas_thresholds" not in example
    from core.locations import RULE_KEYS, clean_rules

    assert "gas_thresholds" not in RULE_KEYS and clean_rules({"gas_thresholds": {"acetone": 3.0}}) == {}
    pattern = re.compile(r"GAS_LIMITS|gas\.thresholds|\"thresholds\"\)\.get\(|disagreement_pct|rise_rate_threshold|"
                         r"trend_stable_band|_ppm\(text\) >= 50000")
    found = []
    for folder in ("core", "modules", "agents", "dashboard", "experiments", "drivers"):
        for root, _, files in os.walk(os.path.join(REPO, folder)):
            for name in files:
                if name.endswith((".py", ".js", ".yaml")):
                    path = os.path.join(root, name)
                    if pattern.search(open(path, encoding="utf-8").read()):
                        found.append(os.path.relpath(path, REPO))
    assert not found, found


# --- TOD-53 clarification: % of C_crit, the graphene model's range, low-range scenarios ------------------------

def test_model_range_is_the_training_data_and_starts_at_or_above_c_crit():
    from modules.gas.data import GAS_SHEETS, MODEL_RANGE_VOL_PCT, load_dataset

    data = load_dataset(os.path.join(REPO, "data", "gas", "data.xls"))
    for index, gas in enumerate(GAS_SHEETS):
        values = data.y_conc[(data.source == index) & (data.y_conc > 0)]
        assert MODEL_RANGE_VOL_PCT[gas] == pytest.approx((values.min(), values.max())), gas
        assert MODEL_RANGE_VOL_PCT[gas][0] >= TABLE.get(gas).c_crit_in("vol_pct"), gas      # never sub-critical
    from drivers.gas.graphene_fet import model_range

    assert model_range("toluene") == [0.25, 9.0] and model_range("ammonia") == [0.3, 10.8]


def test_any_nonzero_graphene_reading_is_at_least_critical():
    for gas, floor in (("ethanol", 0.4), ("acetone", 0.4), ("toluol", 0.25), ("ammonia", 0.3)):
        reading = {"gas": gas, "concentration_pct": 0.001, "detection_floor": floor}
        block = summary(assess(TABLE, reading))
        assert block["critical"] and block["gases"][0]["below_range"], gas
        assert "нижче діапазону сенсора" in block["text"], gas
        p = hfs_payload([reading])
        assert p["level"] == "critical", gas
    driver_item = {"readings": [{"gas": "acetone", "concentration": 0.05, "unit": "vol_pct", "range": [0.4, 13.2]}]}
    assert summary(assess(TABLE, driver_item))["critical"]                    # the driver's range does the same


def test_values_are_shown_as_raw_plus_percent_of_the_critical_level():
    from modules.hfs.gas_normalize import flat_reading, pct_of_crit, reading_text

    assert (pct_of_crit(20.0), pct_of_crit(0.423), pct_of_crit(166.6667)) == (2000, 42.3, 16667)
    assert reading_text({"gas": "acetone", "concentration_pct": 5.0, "trend": "rising"}) == \
        "acetone 5 vol % (2000 % від критичного рівня) rising"
    ppm = {"readings": [{"gas": "ammonia", "concentration": 12, "unit": "ppm"},
                        {"gas": "carbon_monoxide", "concentration": 600, "unit": "ppm"}]}
    flat = flat_reading(ppm)                                  # the strongest substance, in the TOD-19 fields
    assert (flat["gas"], flat["concentration_pct"], flat["pct_of_crit"]) == ("carbon_monoxide", 0.06, 50.0)
    assert reading_text(ppm).startswith("carbon_monoxide 600 ppm (50 % від критичного рівня)")
    p = hfs_payload([{"gas": "acetone", "concentration_pct": 5.0}])
    assert "2000 % від критичного рівня" in p["explanation"] and "2000 %" in p["subsystems"]["gas"]["headline"]


def test_scripted_scenarios_are_fractions_of_c_crit_from_a_low_range_sensor(tmp_path):
    import json

    from experiments.scripted import build
    from modules.hfs.gas_normalize import flat_reading

    manifest = json.load(open(build(str(tmp_path), 42), encoding="utf-8"))
    assert manifest["labels_version"] == 3 and "C_crit" in manifest["gas_scale"]
    for scenario in manifest["scenarios"]:
        top = 0.0
        for line in open(tmp_path / scenario["event_log"], encoding="utf-8"):
            event = json.loads(line)
            if event["type"] != "gas_reading":
                continue
            p = event["payload"]
            assert "concentration_pct" not in p and p["sensor_model"] == "low_range_ppm"     # never the graphene model
            (item,) = p["readings"]
            assert item["unit"] == "ppm" and item["range"][0] <= item["concentration"] <= item["range"][1]
            top = max(top, flat_reading(p)["ratio"])
        assert top <= 2.0                                         # 0–2 × C_crit, not the 0–14 vol % training range
        if top >= 1.0:
            assert scenario["labels"]["annotator_a"]["level"] == "critical", scenario["scenario_id"]
