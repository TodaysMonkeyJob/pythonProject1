"""F22 (TOD-33): measured trust inputs — calibration maps, priors, cost-based thresholds, fuzzy.yaml references."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from core.calibration import (IsotonicMap, LogisticMap, bayes_bins, beta_prior, brier, cllr, ece, equal_error_rate,
                              fit_cost_threshold, fit_isotonic, fit_logistic, load_map, map_from_dict, prior_entry,
                              reliability, wer, wilson_lower)
from core.fuzzy import load_yaml
from core.fuzzy.engine import MeasuredRefError

REPO = Path(__file__).resolve().parents[1]


# --- isotonic / logistic ----------------------------------------------------------------------------------------------

def test_isotonic_fit_is_monotone_and_clipped():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < x ** 2).astype(float)            # true P(y | x) = x²
    mapping = fit_isotonic(x, y)
    grid = np.linspace(-0.5, 1.5, 81)
    values = mapping.many(grid)
    assert np.all(np.diff(values) >= -1e-12) and values.min() >= 0 and values.max() <= 1
    assert abs(mapping(0.5) - 0.25) < 0.12 and mapping(-3) == mapping.y[0] and mapping(9) == mapping.y[-1]
    again = map_from_dict(mapping.to_dict())
    assert isinstance(again, IsotonicMap) and abs(again(0.7) - mapping(0.7)) < 1e-3


def test_logistic_fit_recovers_the_synthetic_model():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 4000)
    y = (rng.uniform(0, 1, 4000) < 1 / (1 + np.exp(-(3 * x - 1)))).astype(int)
    mapping = fit_logistic(x, y)
    assert abs(mapping.a - 3) < 0.4 and abs(mapping.b + 1) < 0.25
    p = mapping.many(x)
    assert ece(p, y) < 0.03 and brier(p, y) < brier(np.full(len(y), y.mean()), y)
    assert isinstance(map_from_dict(mapping.to_dict()), LogisticMap)


def test_reliability_bins_and_ece_of_a_perfect_forecast():
    p = np.repeat([0.1, 0.5, 0.9], 1000)
    y = np.concatenate([np.r_[np.ones(100), np.zeros(900)], np.r_[np.ones(500), np.zeros(500)],
                        np.r_[np.ones(900), np.zeros(100)]])
    rows = reliability(p, y)
    assert [r["n"] for r in rows] == [1000, 1000, 1000] and ece(p, y) < 1e-9


def test_eer_and_cllr():
    genuine, impostor = np.array([0.6, 0.7, 0.8, 0.9]), np.array([0.1, 0.2, 0.3, 0.65])
    eer, threshold = equal_error_rate(genuine, impostor)
    assert eer == pytest.approx(0.25) and 0.6 <= threshold <= 0.65
    assert cllr(np.full(10, 20.0), np.full(10, -20.0)) < 1e-6         # confident and right → 0 bits
    assert cllr(np.zeros(10), np.zeros(10)) == pytest.approx(1.0)      # LR = 1 always → uninformative
    assert cllr(np.full(10, -5.0), np.full(10, 5.0)) > 1.0             # confidently wrong → worse than useless


def test_bayes_bins_posterior_grows_with_similarity_and_respects_the_prior():
    thresholds, tar, far = [0.3, 0.4, 0.5], [1.0, 1.0, 0.9], [0.1, 0.01, 0.0]
    mapping, bins = bayes_bins(thresholds, tar, far, 100, 500, prior=0.5)
    values = [mapping(x) for x in np.linspace(0, 1, 21)]
    assert values == sorted(values) and values[0] < 0.1 and values[-1] > 0.95
    rare, _ = bayes_bins(thresholds, tar, far, 100, 500, prior=0.05)
    assert rare(0.45) < mapping(0.45)
    assert sum(b["known"] for b in bins) == pytest.approx(100)
    assert sum(b["strangers"] for b in bins) == pytest.approx(500)


# --- priors -------------------------------------------------------------------------------------------------------

def test_beta_prior_and_wilson_bound():
    entry = prior_entry(29, 0)
    assert entry["tau0"] == pytest.approx(30 / 31, abs=1e-4) and entry["u0"] == pytest.approx(2 / 31, abs=1e-4)
    assert entry["wilson_lb"] == pytest.approx(0.883, abs=1e-3)       # the issue's gas example
    assert wilson_lower(50, 100) == pytest.approx(0.4038, abs=1e-3)
    assert wilson_lower(0, 10) == 0.0 and wilson_lower(0, 0) == 0.0
    assert beta_prior(0, 0) == {"r0": 0, "s0": 0, "tau0": 0.5, "u0": 1.0}
    assert wilson_lower(10, 10) < wilson_lower(100, 100) < 1


# --- cost-based thresholds ----------------------------------------------------------------------------------------

def test_cost_minimisation_picks_the_expected_threshold_on_a_toy_set():
    scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    positive = [False, False, False, False, True, True, True, True]
    fit = fit_cost_threshold(scores, positive, cost_ratio=1, grid=np.round(np.arange(0, 1.01, 0.05), 2),
                             bandwidth=0)
    assert 0.4 < fit["threshold"] <= 0.6 and fit["expected_cost"] == 0
    # overlap: one positive at 0.35 — a costly miss moves the threshold below it, a cheap one does not
    scores2, positive2 = scores + [0.35], positive + [True]
    costly = fit_cost_threshold(scores2, positive2, 10, np.round(np.arange(0, 1.01, 0.05), 2), bandwidth=0)
    cheap = fit_cost_threshold(scores2, positive2, 0.1, np.round(np.arange(0, 1.01, 0.05), 2), bandwidth=0)
    assert costly["threshold"] <= 0.35 < cheap["threshold"] and costly["empirical_p_miss"] == 0


def test_cost_ratio_moves_a_smoothed_threshold_deterministically():
    rng = np.random.default_rng(3)
    neg, pos = rng.normal(0.3, 0.1, 300), rng.normal(0.7, 0.1, 300)
    scores, positive = np.r_[neg, pos], np.r_[np.zeros(300, bool), np.ones(300, bool)]
    t = [fit_cost_threshold(scores, positive, r)["threshold"] for r in (0.1, 1, 10, 100)]
    assert t == sorted(t, reverse=True) and t[0] > t[-1]               # misses dearer → flag earlier
    assert fit_cost_threshold(scores, positive, 10) == fit_cost_threshold(scores, positive, 10)


def test_trust_threshold_direction():
    trust = [0.05, 0.1, 0.2, 0.9, 0.95, 0.97]
    attack = [True, True, True, False, False, False]
    fit = fit_cost_threshold(trust, attack, 10, np.linspace(0.05, 0.95, 91), positive_above=False)
    assert 0.2 < fit["threshold"] < 0.9 and fit["empirical_p_miss"] == 0 and fit["empirical_p_fa"] == 0


# --- ASR ----------------------------------------------------------------------------------------------------------

def test_wer():
    assert wer("the gas sensor is broken", "the gas sensor is broken.") == 0
    assert wer("the gas sensor is broken", "the sensor is broken now") == pytest.approx(2 / 5)
    assert wer("Дякую, я повернуся за п'ять хвилин", "дякую я повернуся за п’ять хвилин") == 0
    assert wer("", "") == 0 and wer("", "x") == 1


# --- fuzzy.yaml references ----------------------------------------------------------------------------------------

def test_refs_resolve_from_neighbour_files_and_fail_loudly(tmp_path):
    (tmp_path / "thresholds.yaml").write_text(yaml.safe_dump({"values": {"theta": 0.61, "levels": {"high": 0.5}}}))
    (tmp_path / "fuzzy.yaml").write_text(yaml.safe_dump({"envelope": {"theta": "@thresholds:values.theta",
                                                                      "thresholds": "@thresholds:values.levels",
                                                                      "note": "plain @ text stays"}}))
    data = load_yaml(str(tmp_path / "fuzzy.yaml"))
    assert data["envelope"] == {"theta": 0.61, "thresholds": {"high": 0.5}, "note": "plain @ text stays"}
    (tmp_path / "fuzzy.yaml").write_text(yaml.safe_dump({"x": "@thresholds:values.missing"}))
    with pytest.raises(MeasuredRefError, match="missing"):
        load_yaml(str(tmp_path / "fuzzy.yaml"))
    (tmp_path / "fuzzy.yaml").write_text(yaml.safe_dump({"x": "@nofile:a"}))
    with pytest.raises(MeasuredRefError, match="not found"):
        load_yaml(str(tmp_path / "fuzzy.yaml"))


def test_no_unreferenced_trust_constant_or_threshold_in_fuzzy_yaml():
    raw = yaml.safe_load((REPO / "config" / "fuzzy.yaml").read_text())
    trust, envelope, gating = raw["trust"], raw["envelope"], raw["gating"]
    assert "base" not in trust and trust["registry"] == "trust_registry.yaml"          # F23: per device × channel
    registry = yaml.safe_load((REPO / "config" / "trust_registry.yaml").read_text())
    for kind, prior in registry["priors"].items():
        assert prior is None or str(prior).startswith("@trust_priors:"), kind
    for key in ("deescalation_floor", "instruction"):
        assert str(trust["consistency"][key]).startswith("@trust_priors:"), key
    assert str(registry["gas"]["disagreement_pct"]).startswith("@trust_priors:") and "gas_confidence_min" not in trust
    assert set(trust["calibration"]) == {"asr", "speaker", "face", "gas"}
    assert "theta" not in envelope and "locations" not in envelope                   # F24: no θ cut any more
    assert trust["claim_threshold"] == "@thresholds:values.claim_theta"
    assert raw["it2"]["k"] == "@thresholds:values.k_var"
    assert envelope["thresholds"] == "@thresholds:values.levels"
    assert gating["threshold"].startswith("@thresholds:") and gating["critical_floor"].startswith("@thresholds:")
    resolved = load_yaml(str(REPO / "config" / "fuzzy.yaml"))
    assert 0 < resolved["trust"]["claim_threshold"] < 1
    assert set(resolved["envelope"]["thresholds"]) == {"medium", "high", "critical"}


def test_committed_calibration_files_carry_provenance():
    for channel in ("asr", "speaker", "face", "gas"):
        data = yaml.safe_load((REPO / "config" / "calibration" / f"{channel}.yaml").read_text())
        assert data["channel"] == channel and data["n"] > 0 and len(data["dataset"]["sha256"]) == 64
        assert data["fitted_at"] and load_map(str(REPO / "config" / "calibration" / f"{channel}.yaml")) is not None
    speaker = yaml.safe_load((REPO / "config" / "calibration" / "speaker.yaml").read_text())
    assert {"eer", "cllr"} <= set(speaker["metrics"]["heldout"])
    asr = yaml.safe_load((REPO / "config" / "calibration" / "asr.yaml").read_text())
    assert asr["wer_max"] == 0.1 and asr["reliability"]
    priors = yaml.safe_load((REPO / "config" / "trust_priors.yaml").read_text())
    expected = {"camera": {"objects", "motion", "pose_action", "smoking", "scene_text"},
                "mic": {"vad", "asr", "asr_local", "speaker"}, "gas": {"gas_class", "gas_concentration", "gas_trend"},
                "identity": {"face_known", "face_unknown"}}
    assert {d: set(e) for d, e in priors["channels"].items()} == expected          # every TOD-34 channel
    for device, entries in priors["channels"].items():
        for name, entry in entries.items():
            assert {"prior_status", "r0", "s0", "tau0", "u0", "wilson_lb", "source", "explanation"} <= set(entry)
            if entry["prior_status"] == "unmeasured":                              # never a typed constant
                assert entry["r0"] == entry["s0"] == 0 and entry["tau0"] == 0.5 and entry["source"] is None
            else:
                assert entry["prior_status"] == "measured" and entry["r0"] + entry["s0"] > 0
                assert len(entry["sha256"]) == 64, f"{device}.{name}"
    assert {n for n, e in priors["channels"]["camera"].items() if e["prior_status"] == "unmeasured"} == {
        "objects", "motion", "scene_text"}
    assert all(p["measured"] is False for p in priors["policy"].values())
    thresholds = yaml.safe_load((REPO / "config" / "thresholds.yaml").read_text())
    assert thresholds["cost_ratio"] > 0 and len(thresholds["dataset"]["hash"]) == 64 and thresholds["expected_cost"]


def test_fit_thresholds_is_deterministic_and_follows_the_cost_ratio():
    from tools.fit_thresholds import fit_all

    rng = np.random.default_rng(5)
    evidence = [{"trust": float(t), "attack": a} for t, a in
                [(v, True) for v in rng.uniform(0.0, 0.5, 30)] + [(v, False) for v in rng.uniform(0.4, 1.0, 200)]]
    levels = ["low", "medium", "high", "critical"]
    outcomes = [{"expert_level": levels[i % 4], "risk_final": float(np.clip(0.1 + 0.27 * (i % 4) + rng.normal(0, .08),
                                                                          0, 1)), "recorded_llm": True}
                for i in range(80)]
    gate = [{"score": float(s), "override": None, "floor": float(f), "truth": "low"} for s, f in
            zip(rng.uniform(0, 1, 60), rng.uniform(0, 0.6, 60), strict=True)]
    data = {"evidence": evidence, "outcomes": outcomes, "gate": gate}
    a, b = fit_all(data, 10, 0.05, {}), fit_all(data, 10, 0.05, {})
    assert a == b
    cheap = fit_all(data, 1, 0.05, {})
    assert a["values"]["claim_theta"] >= cheap["values"]["claim_theta"]
    assert a["values"]["levels"]["high"] <= cheap["values"]["levels"]["high"]
    lv = a["values"]["levels"]
    assert lv["medium"] < lv["high"] < lv["critical"] == a["values"]["critical_floor"]
    # too few scenarios with recorded GPT answers → the previous level boundaries are kept, θ / gating still refit
    previous = {"fitted_at": "x", "values": {"levels": {"medium": 0.2, "high": 0.5, "critical": 0.8}}, "fits": {}}
    few = {**data, "outcomes": [{**o, "recorded_llm": False} for o in outcomes]}
    kept = fit_all(few, 10, 0.05, {}, previous)
    assert kept["values"]["levels"] == previous["values"]["levels"] and "kept_from" in kept["fits"]["level_high"]
