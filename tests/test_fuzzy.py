"""F15 (TOD-26): fuzzy inference core — membership functions, rules, engine, validation, overrides, golden cases."""
import socket
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from core.fuzzy import FuzzyConfigError, FuzzyEngine, RuleError, parse_rule
from core.fuzzy.membership import gaussmf, parse_term, shift, trapmf, trimf
from core.fuzzy.variables import LinguisticVariable

REPO = Path(__file__).resolve().parent.parent
CONFIG = str(REPO / "config" / "fuzzy.yaml")


def base_config():
    return yaml.safe_load(Path(CONFIG).read_text())


def engine(location=None, locations_path=None):
    return FuzzyEngine.from_config(CONFIG, location, locations_path)


LO_HI = {"lo": ["trap", 0, 0, 0.2, 0.8], "hi": ["trap", 0.2, 0.8, 1, 1]}


def tiny(rules, **extra):
    """Two inputs + one output, easy to compute by hand."""
    return {"output": "out",
            "variables": {
                "a": {"universe": [0, 1, 0.01], "terms": dict(LO_HI)},
                "b": {"universe": [0, 1, 0.01], "terms": dict(LO_HI)},
                "out": {"universe": [0, 1, 0.001], "terms": {"small": ["tri", 0, 0.2, 0.4],
                                                             "big": ["tri", 0.6, 0.8, 1.0],
                                                             "all": ["trap", 0, 0, 1, 1]}}},
            "rules": rules, **extra}


# --- membership functions -------------------------------------------------------------------------------------------

def test_membership_shapes():
    x = np.array([0, 1, 2, 3, 4, 5, 6])
    assert trimf(x, 1, 3, 5).tolist() == [0, 0, 0.5, 1, 0.5, 0, 0]
    assert trapmf(x, 1, 2, 4, 5).tolist() == [0, 0, 1, 1, 1, 0, 0]
    assert trapmf(x, 0, 0, 2, 4).tolist() == [1, 1, 1, 0.5, 0, 0, 0]          # left shoulder
    assert trapmf(x, 2, 4, 6, 6).tolist() == [0, 0, 0, 0.5, 1, 1, 1]          # right shoulder
    g = gaussmf(x, 3, 1)
    assert g[3] == 1 and g[2] == pytest.approx(np.exp(-0.5)) and g[2] == pytest.approx(g[4])
    night = parse_term([["trap", 0, 0, 5, 6], ["trap", 21, 22.5, 24, 24]])      # wraps around midnight
    var = LinguisticVariable("t", 0, 24, 0.5, {"night": night})
    assert var.fuzzify(2)["night"] == 1 and var.fuzzify(12)["night"] == 0 and var.fuzzify(23)["night"] == 1
    assert var.fuzzify(99)["night"] == 1                                       # clamped into the universe


def test_bad_terms_are_rejected():
    for spec, message in ((["blob", 1, 2], "unknown shape"), (["tri", 1, 2], "needs 3"),
                          (["tri", 3, 2, 1], "non-decreasing"), (["gauss", 0, 0], "sigma")):
        with pytest.raises(ValueError, match=message):
            parse_term(spec)


def test_shift_keeps_universe_edges():
    moved = shift(parse_term(["trap", 21, 22.5, 24, 24]), -1, 0, 24)
    assert moved == [("trap", (20, 21.5, 24, 24))]
    assert shift(parse_term(["gauss", 5, 1]), 2, 0, 24) == [("gauss", (7, 1))]


# --- rules ------------------------------------------------------------------------------------------------------

def test_rule_parsing_precedence_not_and_weight():
    rule = parse_rule("if a is hi and not (b is lo or b is hi) or a is lo then out is big with 0.5", 3)
    assert rule.id == "R3" and rule.weight == 0.5 and (rule.output, rule.term) == ("out", "big")
    assert rule.text == "IF a IS hi AND NOT (b IS lo OR b IS hi) OR a IS lo THEN out IS big"
    assert parse_rule("IF a IS NOT lo THEN out IS small", 1).literals() == [("a", "lo", True)]
    assert parse_rule({"id": "G1", "rule": "IF a IS hi THEN out IS big", "weight": 0.25}, 1).weight == 0.25
    for bad, message in (("a IS hi THEN out IS big", "expected IF"), ("IF a IS THEN out IS big", "term name"),
                         ("IF (a IS hi THEN out IS big", r"missing '\)'"),
                         ("IF a IS hi THEN out IS big extra", "unexpected"),
                         ("IF a IS hi THEN out IS big WITH 2", "within 0..1")):
        with pytest.raises(RuleError, match=message):
            parse_rule(bad, 1)


def test_rule_strengths_and_operators():
    mu = {"a": {"lo": 0.3, "hi": 0.7}, "b": {"lo": 0.6, "hi": 0.4}}
    rule = parse_rule("IF a IS hi AND b IS lo THEN out IS big", 1)
    assert rule.strength(mu) == 0.6                                            # min
    assert rule.strength(mu, "prod") == pytest.approx(0.42)
    either = parse_rule("IF a IS hi OR b IS hi THEN out IS big WITH 0.5", 1)
    assert either.strength(mu) == pytest.approx(0.35)                          # max × weight
    assert either.strength(mu, s_norm="probor") == pytest.approx((0.7 + 0.4 - 0.28) * 0.5)
    assert parse_rule("IF NOT a IS hi THEN out IS big", 1).strength(mu) == pytest.approx(0.3)


# --- engine -----------------------------------------------------------------------------------------------------

def test_centroid_and_other_defuzzifiers_on_known_cases():
    one = FuzzyEngine.from_dict(tiny(["IF a IS hi THEN out IS big"]))
    assert one.infer({"a": 1, "b": 0})["crisp"] == pytest.approx(0.8, abs=1e-3)     # symmetric triangle → its peak
    clipped = one.infer({"a": 0.5, "b": 0})                                         # μ(hi) = 0.5 → clipped, still 0.8
    assert clipped["crisp"] == pytest.approx(0.8, abs=1e-3) and clipped["memberships"]["big"] == 0.5
    two = FuzzyEngine.from_dict(tiny(["IF a IS hi THEN out IS big", "IF b IS hi THEN out IS small"]))
    assert two.infer({"a": 1, "b": 1})["crisp"] == pytest.approx(0.5, abs=1e-3)     # equal halves → the middle
    for method, expected in (("mom", 0.5), ("som", 0.0), ("lom", 1.0), ("bisector", 0.5)):
        e = FuzzyEngine.from_dict(tiny(["IF a IS hi THEN out IS all"], defuzzification=method))
        assert e.infer({"a": 1, "b": 0})["crisp"] == pytest.approx(expected, abs=2e-3), method
    nothing = FuzzyEngine.from_dict(tiny(["IF a IS hi THEN out IS big"], default_output=0.05))
    result = nothing.infer({"a": 0, "b": 0})
    assert result["crisp"] == 0.05 and result["fired_rules"] == []


def test_missing_inputs_use_defaults_or_skip_rules():
    cfg = tiny(["IF a IS hi THEN out IS big", "IF b IS hi THEN out IS small"])
    cfg["variables"]["b"]["default"] = 1.0
    result = FuzzyEngine.from_dict(cfg).infer({})
    assert result["skipped_rules"] == ["R1"] and result["inputs"]["b"]["defaulted"] is True
    assert [r["id"] for r in result["fired_rules"]] == ["R2"]
    assert "(default)" in result["fired_rules"][0]["explanation"]


def test_variants_select_per_gas_terms():
    e = engine()
    assert e.fuzzify("gas_concentration", 4.0)["dangerous"] == 1.0                   # base: 3 % limit
    assert e.fuzzify("gas_concentration", 4.0, "acetone")["dangerous"] == 0.0         # acetone: 5 % limit
    acetone = e.infer({"gas_concentration": 4.0, "gas": "acetone", "time_of_day_risk": 12})
    ammonia = e.infer({"gas_concentration": 4.0, "gas": "ammonia", "time_of_day_risk": 12})
    assert acetone["crisp"] < 0.85 <= ammonia["crisp"]
    assert acetone["inputs"]["gas_concentration"]["variant"] == "acetone"


def test_fired_rules_are_readable():
    result = engine().infer({"gas_concentration": 4.0, "gas": "acetone", "time_of_day_risk": 12})
    top = result["fired_rules"][0]
    assert top["id"] == "R3" and top["term"] == "high" and top["source"] == "base"
    assert top["explanation"] == ("gas_concentration = 4 % → elevated 1.00; gas_trend = 0 %/min (default) → "
                                  "NOT sharply_rising 1.00 ⇒ risk IS high (1.00)")


# --- golden scenarios (acceptance) ------------------------------------------------------------------------------

GOLDEN = [
    ("dangerous gas", {"gas_concentration": 6.0, "time_of_day_risk": 12}, 0.85, 1.0),
    ("dangerous acetone, people around", {"gas_concentration": 7.5, "gas": "acetone", "people_count": 2,
                                          "motion_level": 0.5, "time_of_day_risk": 14}, 0.85, 1.0),
    ("empty quiet room at noon", {"time_of_day_risk": 12}, 0.0, 0.15),
    ("known person working at noon", {"people_count": 1, "motion_level": 0.3, "action_risk": 0.1,
                                      "time_of_day_risk": 12}, 0.0, 0.15),
    ("person lying motionless", {"people_count": 1, "action_risk": 1.0, "motion_level": 0.0,
                                 "time_of_day_risk": 12}, 0.85, 1.0),
    ("stranger at night", {"people_count": 1, "unknown_share": 1.0, "motion_level": 0.3,
                           "time_of_day_risk": 23}, 0.55, 0.85),
    ("gas rising sharply from slight", {"gas_concentration": 1.0, "gas_trend": 3.0, "time_of_day_risk": 12},
     0.55, 0.85),
]


@pytest.mark.parametrize("name,inputs,low,high", GOLDEN, ids=[g[0] for g in GOLDEN])
def test_golden_scenarios(name, inputs, low, high):
    assert low <= engine().infer(inputs)["crisp"] <= high


def test_action_risk_map():
    e = engine()
    assert e.map_value("action_risk", ["walking", "lying"]) == 1.0
    assert e.map_value("action_risk", ["dancing"]) == 0.0


# --- per-location overrides -------------------------------------------------------------------------------------

def test_location_override_changes_only_that_location():
    case = {"people_count": 2, "unknown_share": 0.5, "time_of_day_risk": 19.75}
    base, office, warehouse = engine(), engine("office"), engine("warehouse")
    assert base.infer(case)["crisp"] == office.infer(case)["crisp"]                 # no overrides for the office
    assert warehouse.infer(case)["crisp"] > base.infer(case)["crisp"] + 0.1          # stricter warehouse
    sources = {r["source"] for r in warehouse.infer(case)["fired_rules"]}
    assert "location:warehouse" in sources
    assert "R21" not in {r.id for r in warehouse.rules} and "R21" in {r.id for r in base.rules}
    assert base.infer(case)["crisp"] == engine().infer(case)["crisp"]                # building one did not leak


def test_locations_yaml_fuzzy_block(tmp_path):
    locations = tmp_path / "locations.yaml"
    locations.write_text(yaml.safe_dump({"locations": [{"id": "lab", "name": "Lab", "fuzzy": {
        "shift": {"gas_concentration": {"dangerous": -1.0, "elevated": -1.0}},
        "weights": {"R14": 1.0}}}]}))
    lab, base = engine("lab", str(locations)), engine()
    gas = {"gas_concentration": 2.6, "time_of_day_risk": 12}
    assert lab.fuzzify("gas_concentration", 2.6)["dangerous"] == 1.0
    assert base.fuzzify("gas_concentration", 2.6)["dangerous"] == 0
    assert lab.infer(gas)["crisp"] > base.infer(gas)["crisp"]
    assert next(r for r in lab.rules if r.id == "R14").weight == 1.0


# --- validation -------------------------------------------------------------------------------------------------

def errors_of(cfg, overrides=None):
    with pytest.raises(FuzzyConfigError) as info:
        FuzzyEngine.from_dict(cfg, "test", overrides)
    return "\n".join(info.value.errors)


def test_validation_errors_are_clear():
    assert "unknown variable 'c'" in errors_of(tiny(["IF c IS hi THEN out IS big"]))
    assert "'a' has no term 'mid'" in errors_of(tiny(["IF a IS mid THEN out IS big"]))
    assert "output term 'huge' does not exist" in errors_of(tiny(["IF a IS hi THEN out IS huge"]))
    assert "R2 duplicates R1" in errors_of(tiny(["IF a IS hi THEN out IS big", "IF a IS hi THEN out IS big"]))
    assert "R2 conflicts with R1" in errors_of(tiny(["IF a IS hi THEN out IS big", "IF a IS hi THEN out IS small"]))
    gap = tiny(["IF a IS hi THEN out IS big"])
    gap["variables"]["a"]["terms"] = {"lo": ["trap", 0, 0, 0.2, 0.3], "hi": ["trap", 0.5, 0.6, 1, 1]}
    assert "a: no term covers 0.3–0.5" in errors_of(gap)
    same = tiny(["IF a IS hi THEN out IS big"])
    same["variables"]["b"]["terms"]["also_lo"] = ["trap", 0, 0, 0.2, 0.8]
    assert "terms 'lo' and 'also_lo' have the same definition" in errors_of(same)
    assert "unknown rule id 'R9'" in errors_of(tiny(["IF a IS hi THEN out IS big"]), [{"remove_rules": ["R9"]}])
    assert "operators.and" in errors_of(tiny(["IF a IS hi THEN out IS big"], operators={"and": "avg"}))
    assert "rule 1: expected THEN" in errors_of(tiny(["IF a IS hi out IS big"]))


def test_shipped_config_and_every_location_are_valid():
    for location in [None, *base_config()["locations"]]:
        assert engine(location).rules


# --- non-functional ---------------------------------------------------------------------------------------------

def test_no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the fuzzy core must not open sockets")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    engine("warehouse").infer({"gas_concentration": 4.0, "people_count": 3, "time_of_day_risk": 2})


def test_infer_is_fast():
    e = engine("warehouse")
    case = {"gas_concentration": 2.0, "gas_trend": 1.0, "people_count": 3, "unknown_share": 0.5, "motion_level": 0.5,
            "action_risk": 0.6, "time_of_day_risk": 20}
    e.infer(case)
    start = time.perf_counter()
    for _ in range(200):
        e.infer(case)
    per_call_ms = (time.perf_counter() - start) / 200 * 1000
    assert per_call_ms < 5.0                                                    # Pi target; ≈ 0.2–0.3 ms on the dev Mac


def test_cli_check_and_explain(capsys):
    from core.fuzzy.__main__ import main

    assert main(["--config", CONFIG, "--check"]) == 0
    assert main(["--config", CONFIG, "gas_concentration=6", "time_of_day_risk=12"]) == 0
    out = capsys.readouterr().out
    assert "ok  warehouse" in out and "R1" in out and "risk = 0.9" in out
