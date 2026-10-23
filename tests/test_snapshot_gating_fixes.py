"""F28 (TOD-39): snapshot v2.1 (term ranges), the medium / high rubric, robust calibration, corridor-driven gating."""
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from core.calibration import fit_cost_threshold
from core.fuzzy import FuzzyEngine
from modules.aggregator.aggregator import Aggregator
from modules.aggregator.llm_view import llm_view, term_ranges
from modules.llm.calibration import BOUNDS, check_constraints, fit, loo_cv, synthetic
from modules.llm.gating import LLMGate
from modules.llm.gpt_client import load_prompt, risk_scale
from tests.test_gating import ROUTINE, row, view
from tests.test_llm_v2 import scene

REPO = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((REPO / "schemas" / "snapshot.schema.json").read_text())
FUZZY = str(REPO / "config" / "fuzzy.yaml")
T0 = 1.7e9


def aggregator(tmp_path, clock, version):
    config = {"aggregator": {"export_dir": str(tmp_path / "exports"), "incidents": {"enabled": False},
                             "snapshot_interval_sec": 5}, "trust": {"fuzzy_config": FUZZY},
              "llm": {"snapshot_version": version}}
    return Aggregator(config, clock=lambda: clock[0])


# --- C: snapshot v2.1 -----------------------------------------------------------------------------------------

def test_snapshot_v21_carries_term_ranges_but_no_readings(tmp_path):
    clock = [T0]
    snapshot = scene(aggregator(tmp_path, clock, "v2.1"), clock)
    jsonschema.validate(snapshot, SCHEMA)
    view_ = snapshot["llm_view"]
    gas = next(item for item in view_["trusted_evidence"] if item["kind"] == "gas")
    assert view_["v"] == 2.1 and gas["level"] == {"slight": 1.0}
    assert gas["range_pct"] == [0.5, 1.5]                              # acetone's own slight term (5 % limit)
    assert gas["trend_range_pct_per_min"] == [2.5, 10.0]               # sharply_rising
    assert view_["gas_scale_pct"]["acetone"]["elevated"] == [2.5, 4.0]
    text = json.dumps(view_)
    for reading in ("1.23", "2.61", "12.7", "12300"):                  # concentration, rise rate, dB, ppm
        assert reading not in text


def test_snapshot_version_option(tmp_path):
    clock = [T0]
    v1 = aggregator(tmp_path / "a", clock, "v1")
    scene(v1, clock)
    (row_v1,) = [r for r in v1.outbox.pending(limit=10) if r["trigger"] != "global"][-1:]
    assert not row_v1["digest"].startswith("{") and "llm_view" not in row_v1["payload"]
    clock = [T0]
    v2 = aggregator(tmp_path / "b", clock, "v2")
    assert scene(v2, clock)["llm_view"]["v"] == 2 and "gas_scale_pct" not in scene(v2, clock)["llm_view"]
    with pytest.raises(ValueError, match="snapshot version"):
        llm_view({"window": {"from": "2026-10-23T10:00:00Z", "to": "2026-10-23T10:01:00Z"}, "trigger": "interval",
                  "people": [], "objects": [], "alerts": [], "sounds": [], "speech": []}, version="v3")


def test_term_ranges_from_the_location_engine():
    engine = FuzzyEngine.from_config(FUZZY)
    base = term_ranges(engine, "gas_concentration")
    assert base["elevated"] == [1.8, 2.6] and base["dangerous"] == [3.4, 10.0]
    assert term_ranges(engine, "gas_concentration", "ethanol")["elevated"] == [2.5, 4.0]


def test_prompt_rubric_uses_the_rule_base_risk_scale():
    prompt = load_prompt("security_analyst")
    scale = risk_scale(FUZZY)
    assert "{risk_scale}" not in prompt and scale in prompt
    assert scale.startswith("low 0–0.3") and "medium 0.15–0.6 (peak 0.4)" in scale and "high 0.45–0.85" in scale
    assert "choose high only if the trusted evidence itself shows likely harm" in prompt
    assert '"elevated" is a gas level, not a risk level' in prompt


# --- C: calibration -------------------------------------------------------------------------------------------

def test_calibration_constraints_hold_even_with_talked_down_answers():
    rows = synthetic(36, seed=3)
    distributions, targets = [r[1] for r in rows], [r[2] for r in rows]
    outliers = [{"low": 0.85, "medium": 0.15}] * 8                     # "low" answers in critical scenarios
    calibrator, metrics = fit(distributions + outliers, targets + [0.95] * 8, restarts=2)
    check = check_constraints(calibrator.params)
    assert check["ok"] and check["monotone_centres"] and check["centres_in_bounds"]
    assert check["centres"]["low"] <= BOUNDS["low"][1] and check["min_support"] >= 0.2 - 1e-6
    assert calibrator.risk({"low": 1.0}) < 0.3                        # "low" still means low
    bad = {**calibrator.params, "medium": [0.0, 0.05, 0.05, 0.1]}      # centre below low's, too narrow
    assert not check_constraints(bad)["ok"]


def test_leave_one_out_reports_held_out_metrics():
    rows = synthetic(14, seed=5)
    cv = loo_cv([r[1] for r in rows], [r[2] for r in rows])
    assert cv["n"] == 14 and len(cv["predictions"]) == 14
    assert {"brier", "ece"} <= set(cv) and {"brier", "ece"} <= set(cv["fixed"])
    in_sample = fit([r[1] for r in rows], [r[2] for r in rows])[1]
    assert cv["brier"] >= in_sample["brier"] - 1e-9                    # held-out error is never better than the fit


def test_harness_calibration_needs_recorded_answers(tmp_path, capsys):
    from types import SimpleNamespace

    from tools.calibrate_llm_risk import fit_harness

    args = SimpleNamespace(harness="data/eval/scripted", seed=42, calibration_share=0.3, min_width=0.2, report=None,
                           dry_run=False, out=str(tmp_path / "calibration.yaml"))
    assert fit_harness(args, [], 36) == 1 and not (tmp_path / "calibration.yaml").exists()
    assert "Fewer than 10 recorded answers" in capsys.readouterr().err


# --- E: gating ------------------------------------------------------------------------------------------------

def test_call_cost_raises_the_gating_threshold():
    scores = np.r_[np.full(20, 0.5), np.full(10, 0.9), np.full(3, 0.9)]
    needed = [False] * 20 + [False] * 10 + [True] * 3
    free = fit_cost_threshold(scores, needed, 10, np.linspace(0.05, 0.95, 91), 0.05)
    paid = fit_cost_threshold(scores, needed, 10, np.linspace(0.05, 0.95, 91), 0.05, flag_cost=0.5)
    assert paid["flagged_share"] < 1.0 and paid["threshold"] >= free["threshold"]
    assert paid["empirical_p_miss"] == 0.0                             # the needed calls are still made


NARROW = {"floor": 0.106, "ceil": 0.106, "width": 0.0, "fired_rules": [{"term": "low"}]}
WIDE = {"floor": 0.2, "ceil": 0.75, "width": 0.55, "fired_rules": [{"term": "low"}, {"term": "high"}]}


def test_gating_skips_a_stable_agreed_scene_and_calls_on_width():
    gate = LLMGate.from_config(FUZZY)
    assert {"width", "agreement"} <= set(gate.engine.variables)
    first = gate.evaluate(row(ROUTINE, sid="a"), NARROW, 0, None, T0)
    gate.record("office", 0.85, NARROW["floor"], T0)                  # GPT answered and agreed
    stable = gate.evaluate(row(ROUTINE, sid="b"), NARROW, 0, T0, T0 + 120)
    assert stable.inputs["agreement"] == 0.85 and stable.inputs["width"] == 0.0
    assert not stable.call and stable.reasons == []
    wide = gate.evaluate(row(ROUTINE, sid="c"), WIDE, 0, T0, T0 + 180)
    assert wide.call and wide.override is None and "G2" in wide.reasons
    assert first.inputs["novelty"] == 1.0 and "agreement" not in first.inputs     # never asked ≠ disagreed


def test_gating_calls_again_when_gpt_disagreed_and_something_changed():
    gate = LLMGate.from_config(FUZZY)
    gate.remember("office", {"person:Oleh"}, T0)
    gate.record("office", 0.0, NARROW["floor"], T0)                   # GPT's answer outside the corridor
    visitor = view(people=[("Oleh", ["sitting"]), ("unknown_7", ["walking"])])
    decision = gate.evaluate(row(visitor, sid="d"), NARROW, 0, T0, T0 + 120)
    assert decision.call and "G5" in decision.reasons
    agreed = LLMGate.from_config(FUZZY)
    agreed.remember("office", {"person:Oleh"}, T0)
    agreed.record("office", 0.85, NARROW["floor"], T0)
    assert agreed.evaluate(row(view(people=[("Oleh", ["sitting"])]), sid="e"), NARROW, 0, T0, T0 + 120).call is False


def test_replay_records_agreement_for_the_gate():
    from experiments.configs import PROPOSED
    from experiments.dataset import Dataset
    from experiments.llm import SimAnalyst
    from experiments.pipeline import run_scenario
    from experiments.scripted import build

    class Sim:
        def analyze(self, payload, version):
            from modules.llm.gpt_client import LLMDecision

            decision, usage = SimAnalyst().analyze(payload, version)
            return LLMDecision(**decision), usage

    import tempfile

    import modules.llm.gating as gating

    recorded = []
    original = gating.LLMGate.record

    def spy(self, location, h, floor, now=None):
        recorded.append(h)
        return original(self, location, h, floor, now)

    with tempfile.TemporaryDirectory() as tmp:
        build(tmp, 42)
        scenario = next(s for s in Dataset.load(tmp).scenarios if s.category == "dangerous")
        gating.LLMGate.record = spy
        try:
            result = run_scenario(scenario, PROPOSED, Sim(), FUZZY)
        finally:
            gating.LLMGate.record = original
    assert result.llm_calls == len(recorded) >= 1 and all(h is not None for h in recorded)
