"""TOD-53: offline means GPT takes no part — no call, no cache, no simulator (experiments, fit, rules gate)."""
import json

import pytest

from experiments.configs import LLM_ONLY, OFFLINE, PROPOSED, needs_gpt
from experiments.dataset import Dataset
from experiments.pipeline import run_scenario


@pytest.fixture
def no_gpt(monkeypatch):
    """Any GPT path (simulator, cached / Azure client, recorded answers) fails the test."""
    def refuse(*args, **kwargs):
        raise AssertionError("GPT was asked in an offline run")

    import experiments.llm as llm
    import tools.fit_thresholds as fit

    monkeypatch.setattr(llm.SimAnalyst, "analyze", refuse)
    monkeypatch.setattr(llm.CachedLLM, "__init__", refuse)
    monkeypatch.setattr(fit.RecordedGPT, "__init__", refuse)


@pytest.fixture
def toy(tmp_path):
    """Six scenarios of the scripted set (normal, gas attacks, fall) — enough for a split, fast to replay."""
    from experiments.scripted import build

    root = tmp_path / "set"
    manifest = json.load(open(build(str(root), 42), encoding="utf-8"))
    keep = [s for s in manifest["scenarios"] if s["category"] == "normal"][:2] + \
        [s for s in manifest["scenarios"] if s["category"] == "adversarial" and "gas" in s["description"]][:3] + \
        [s for s in manifest["scenarios"] if s["description"] == "fall_lying"][:1]
    (root / "manifest.json").write_text(json.dumps({**manifest, "scenarios": keep}), encoding="utf-8")
    return str(root)


def test_without_gpt_the_corridor_decides_every_snapshot(toy, no_gpt):
    scenario = next(s for s in Dataset.load(toy).scenarios if s.category == "adversarial")
    result = run_scenario(scenario, PROPOSED, None)
    assert result.decisions and {d.source for d in result.decisions} == {"fuzzy_only"}
    assert result.llm_calls == 0 and result.no_gpt > 0
    with pytest.raises(ValueError, match="needs GPT"):
        run_scenario(scenario, LLM_ONLY, None)              # nothing can decide without GPT there


@pytest.mark.parametrize("suite", ["v1", "v2"])
def test_offline_experiment_runs_only_gpt_free_configurations(toy, tmp_path, no_gpt, suite):
    from experiments.run import main

    out = tmp_path / f"out_{suite}"
    assert main(["--dataset", toy, "--seed", "42", "--offline", "--suite", suite, "--out", str(out),
                 *(["--all"] if suite == "v1" else [])]) == 0
    manifest = json.load(open(out / "manifest.json", encoding="utf-8"))
    assert manifest["llm"]["offline"] and manifest["llm"]["gpt"].startswith("none")
    assert set(manifest["configs"]) == {c.name for c in OFFLINE[suite]}
    assert not any(needs_gpt(c) for c in OFFLINE[suite])
    assert {"llm_only", "llm_pattern"} <= set(manifest["llm"]["not_run"])
    assert manifest["calibration"]["fitted"] is False
    text = (out / "RESULTS.md").read_text(encoding="utf-8")
    assert "offline run: GPT took no part" in text and "llm_only" in text
    if suite == "v2":
        assert set(manifest["failure_block"]["stats"]) == {"proposed_v2_local", "proposed_v2_local_static_trust"}


def test_threshold_fit_without_gpt_keeps_the_gpt_dependent_values():
    from tools.fit_thresholds import fit_all

    previous = {"fitted_at": "2026-10-23", "values": {"levels": {"medium": 0.24, "high": 0.5, "critical": 0.76},
                                                       "h_min": 0.85, "gating": 0.68},
                "fits": {"h_min": {"threshold": 0.85}, "gating": {"threshold": 0.68}}}
    data = {"outcomes": [{"scenario_id": "a", "category": "normal", "expert_level": "low", "risk_final": 0.1,
                          "recorded_llm": False}],
            "evidence": [{"trust": 0.9, "attack": False}, {"trust": 0.1, "attack": True}],
            "gate": [{"score": 0.4, "override": None, "floor": 0.2, "truth": "low"}],
            "decision": []}
    result = fit_all(data, 10, 0.05, {}, previous)
    values, fits = result["values"], result["fits"]
    assert values["levels"] == previous["values"]["levels"] and values["critical_floor"] == 0.76
    assert (values["h_min"], values["gating"]) == (0.85, 0.68)
    assert "GPT takes no part" in fits["gating"]["kept_reason"] and "kept_from" in fits["h_min"]
    assert "threshold" in fits["claim_theta"]                     # θ needs no GPT: fitted


def test_rules_gate_runs_without_gpt(toy, no_gpt):
    from tools.rules_gate import gate, markdown

    report = gate("config/fuzzy.yaml", "config/fuzzy.yaml", toy)
    assert report["passed"] and report["llm"] == {"gpt": "none", "calls": 0}
    assert "GPT: none — offline" in markdown(report)
