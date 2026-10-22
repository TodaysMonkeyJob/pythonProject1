"""F24 (TOD-35): interval type-2 engine — FOU, Karnik–Mendel, type-1 equivalence, monotone width, speed."""
import random
import time
from pathlib import Path

import numpy as np
import pytest

from core.fuzzy import FuzzyEngine, load_yaml
from core.fuzzy.engine2 import FuzzyEngine2
from core.fuzzy.it2 import brute_force_cos, cos_interval, term_interval
from core.fuzzy.membership import evaluate, parse_term

REPO = Path(__file__).resolve().parents[1]
CONFIG = str(REPO / "config" / "fuzzy.yaml")
K = {"gas_concentration": 1.0, "gas_trend": 1.0, "unknown_share": 0.3, "time_of_day_risk": 1.0,
     "people_count": 1.0, "motion_level": 0.2, "action_risk": 0.3}
DOC_RULES = {"R2", "R3", "R4", "R5", "R14"}
DOC_INPUTS = {"gas_concentration": 1.4, "gas_trend": 2.0, "unknown_share": 1.0, "time_of_day_risk": 14.0}


def doc_engine():
    data = load_yaml(CONFIG)
    return FuzzyEngine2.from_dict({**data, "rules": [r for r in data["rules"] if r["id"] in DOC_RULES]}, K)


def random_inputs(rng):
    return {"gas_concentration": rng.uniform(0, 8), "gas_trend": rng.uniform(-3, 6), "motion_level": rng.uniform(0, 1),
            "people_count": rng.uniform(0, 7), "unknown_share": rng.uniform(0, 1), "action_risk": rng.uniform(0, 1),
            "time_of_day_risk": rng.uniform(0, 24), "gas": rng.choice(["acetone", "ammonia", None])}


# --- membership ---------------------------------------------------------------------------------------------------

def test_fou_brackets_the_type1_function_and_collapses_at_zero():
    for spec in ([["trap", 0.2, 0.5, 1.0, 1.8]], [["tri", 0.25, 0.5, 0.75]], [["gauss", 0.5, 0.1]],
                 [["trap", 0, 0, 0.2, 0.5]], [[["trap", 0, 0, 5, 6], ["trap", 21, 22.5, 24, 24]]]):
        parts = parse_term(spec[0])
        for x in np.linspace(-0.5, 24.5, 301):
            t1 = float(evaluate(parts, x))
            assert term_interval(parts, float(x), 0.0, 0.0, 24.0) == pytest.approx((t1, t1), abs=1e-12)
            lo, hi = term_interval(parts, float(x), 0.3, 0.0, 24.0)
            assert lo - 1e-12 <= t1 <= hi + 1e-12
    shoulder = parse_term(["trap", 0, 0, 0.2, 0.5])                 # the left shoulder on the universe edge stays
    assert term_interval(shoulder, 0.0, 0.05, 0.0, 10.0) == (1.0, 1.0)      # no inward move at the edge
    assert term_interval(shoulder, 0.1, 0.05, 0.0, 10.0) == (1.0, 1.0)
    narrow = parse_term(["trap", 1.0, 1.2, 1.3, 1.5])                # inward edges cross → LMF below 1
    assert term_interval(narrow, 1.25, 0.2, 0.0, 10.0)[0] < 1.0


# --- type reduction -----------------------------------------------------------------------------------------------

def test_karnik_mendel_matches_brute_force_switch_point_search():
    rng = random.Random(1)
    for _ in range(500):
        n = rng.randint(1, 12)
        ys = [rng.uniform(0, 1) for _ in range(n)]
        lower = [rng.uniform(0, 1) for _ in range(n)]
        upper = [min(1.0, lo + rng.uniform(0, 1)) for lo in lower]
        if rng.random() < 0.2:
            lower = [0.0] * n                                         # only upper firing: the hard edge case
        km = cos_interval(ys, lower, upper)
        order = sorted(range(n), key=lambda i: ys[i])
        brute = brute_force_cos([ys[i] for i in order], [lower[i] for i in order], [upper[i] for i in order])
        assert km == pytest.approx(brute, abs=1e-9)
    assert cos_interval([], [], []) is None and cos_interval([0.5], [0.0], [0.0]) is None


# --- engine -------------------------------------------------------------------------------------------------------

def test_zero_uncertainty_reproduces_type1_centre_of_sets():
    rng = random.Random(2)
    engine2 = FuzzyEngine2.from_config(CONFIG)
    type1 = FuzzyEngine.from_config(CONFIG)
    centroids = engine2.centroids
    for _ in range(300):
        inputs = random_inputs(rng)
        fired = type1.infer(inputs)["fired_rules"]               # independent: the type-1 engine's firing strengths
        total = sum(r["strength"] for r in fired)
        expected = sum(r["strength"] * centroids[r["term"]] for r in fired) / total if total else 0.0
        result = engine2.infer(inputs, {v: 0.0 for v in K})
        assert result["interval"][0] == pytest.approx(result["interval"][1], abs=1e-6)
        assert result["interval"][0] == pytest.approx(expected, abs=2e-4)   # type-1 strengths are rounded to 1e-4
        assert result["type1"]["crisp"] == pytest.approx(result["interval"][0], abs=1e-6)


def test_interval_width_grows_monotonically_with_uncertainty():
    rng = random.Random(3)
    engine2 = FuzzyEngine2.from_config(CONFIG)
    engine2.k = dict(K)
    for _ in range(150):
        inputs = random_inputs(rng)
        chosen = rng.sample(sorted(K), rng.randint(1, len(K)))
        widths, intervals = [], []
        for u in np.linspace(0, 1, 11):
            r = engine2.infer(inputs, {v: float(u) for v in chosen}, explain=False)
            widths.append(r["width"])
            intervals.append(r["interval"])
        assert all(b >= a - 1e-9 for a, b in zip(widths, widths[1:], strict=False)), (inputs, chosen, widths)
        for (lo_a, hi_a), (lo_b, hi_b) in zip(intervals, intervals[1:], strict=False):   # nested intervals
            assert lo_b <= lo_a + 1e-9 and hi_b >= hi_a - 1e-9


def test_docs_example_and_explanation():
    engine = doc_engine()
    exact = engine.infer(DOC_INPUTS, {v: 0.0 for v in DOC_INPUTS})
    assert exact["interval"] == pytest.approx([0.602, 0.602], abs=1e-3)          # the issue's prototype number
    wider = engine.infer(DOC_INPUTS, {v: 0.3 for v in DOC_INPUTS})
    widest = engine.infer(DOC_INPUTS, {v: 0.5 for v in DOC_INPUTS})
    assert widest["interval"][0] < wider["interval"][0] < 0.602 < wider["interval"][1] < widest["interval"][1]
    assert wider["widened_by"] and wider["widened_by"][0]["variable"] in DOC_INPUTS
    assert {"gas_concentration", "gas_trend"} <= {w["variable"] for w in wider["widened_by"]}
    assert set(wider["fou"]["gas_concentration"]) == {"clean", "slight", "elevated", "dangerous"}
    lo, hi = wider["fou"]["gas_concentration"]["slight"]
    assert lo < 0.5 < hi                                                         # μ = 0.5 at δ = 0


def test_missing_inputs_use_type1_defaults_without_fou():
    engine = FuzzyEngine2.from_config(CONFIG)
    engine.k = dict(K)
    r = engine.infer({"people_count": 0, "time_of_day_risk": 12}, {"gas_concentration": 1.0})
    assert r["inputs"]["gas_concentration"]["defaulted"] and r["inputs"]["gas_concentration"]["delta"] == 0.0


def test_inference_is_fast_enough_for_the_pi():
    engine = FuzzyEngine2.from_config(CONFIG)
    engine.k = dict(K)
    inputs = {**DOC_INPUTS, "people_count": 2, "motion_level": 0.3, "action_risk": 0.6}
    uncertainty = {v: 0.5 for v in K}
    engine.infer(inputs, uncertainty)
    started = time.perf_counter()
    for _ in range(100):
        engine.infer(inputs, uncertainty)                    # full rule base + the width explanation
    ms = (time.perf_counter() - started) * 10
    assert ms < 2.5, f"{ms:.2f} ms on this machine (Pi 5 ≈ 4× slower; budget 10 ms)"
