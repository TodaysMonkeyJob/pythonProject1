"""A6 (TOD-46): parameter provenance, evaluation harness, monotonicity measure."""
import os
import shutil

import numpy as np
import pytest
import yaml

from core.fuzzy import FuzzyEngine, load_yaml
from experiments.hfs import metrics as M
from experiments.hfs.run import CONFIGS, run_scenario, system_config, write_variants
from experiments.hfs.variants import gauss_of, gaussian, without_escalation
from tools.plot_surfaces import monotonicity

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- parameter provenance --------------------------------------------------------------------------------------------

def test_every_parameter_has_a_source(capsys):
    from tools.params_doc import main

    assert main(["--check"]) == 0
    assert "every parameter has a source" in capsys.readouterr().out


def test_params_doc_fails_on_a_missing_source(tmp_path, capsys):
    from tools.params_doc import main

    fuzzy = tmp_path / "fuzzy"
    shutil.copytree(os.path.join(REPO, "config", "fuzzy"), fuzzy)
    data = yaml.safe_load(open(fuzzy / "camera.yaml", encoding="utf-8"))
    del data["variables"]["motion_intensity"]["source"]["terms"]
    yaml.safe_dump(data, open(fuzzy / "camera.yaml", "w", encoding="utf-8"), allow_unicode=True)
    sources = yaml.safe_load(open(os.path.join(REPO, "config", "parameter_sources.yaml"), encoding="utf-8"))
    del sources["parameters"]["hfs.gas.window_sec"]
    yaml.safe_dump(sources, open(tmp_path / "sources.yaml", "w", encoding="utf-8"), allow_unicode=True)
    assert main(["--check", "--fuzzy-dir", str(fuzzy), "--sources", str(tmp_path / "sources.yaml")]) == 1
    err = capsys.readouterr().err
    assert "camera.motion_intensity.terms: no source" in err and "hfs.gas.window_sec: no source" in err


def test_params_doc_writes_the_answers(tmp_path):
    from tools.params_doc import main

    assert main(["--out", str(tmp_path / "p.md")]) == 0
    text = (tmp_path / "p.md").read_text(encoding="utf-8")
    assert "Slide 25" in text and "Slide 26" in text and "claim_theta" in text and "removed from Article 1" in text
    assert "OSHA 29 CFR 1910.1000" in text and "ANSI S3.5" in text


# --- monotonicity measure ---------------------------------------------------------------------------------------------

def tiny_engine(shape):
    terms = {"tri": {"low": ["trap", 0, 0, 0.2, 0.8], "high": ["trap", 0.2, 0.8, 1, 1]},
             # "far" maps a larger x back to low risk: the risk falls again after 0.5 — not monotonic
             "bump": {"low": ["trap", 0, 0, 0.2, 0.5], "high": ["tri", 0.2, 0.5, 0.8],
                      "far": ["trap", 0.5, 0.8, 1, 1]}}[shape]
    data = {"output": "risk", "operators": {"and": "prod", "or": "probor", "implication": "prod",
                                            "aggregation": "sum"},
            "variables": {"x": {"universe": [0, 1, 0.01], "terms": terms},
                          "risk": {"universe": [0, 1, 0.01], "terms": {"low": ["trap", 0, 0, 0.1, 0.6],
                                                                       "high": ["trap", 0.4, 0.9, 1, 1]}}},
            "rules": ["IF x IS low THEN risk IS low", "IF x IS high THEN risk IS high",
                      *(["IF x IS far THEN risk IS low"] if shape == "bump" else [])]}
    return FuzzyEngine.from_dict(data)


def test_monotonicity_measure_on_synthetic_surfaces():
    rising = monotonicity(tiny_engine("tri"), 21)["x"]
    assert rising["falls"] == 0 and rising["share"] == 0
    bump = monotonicity(tiny_engine("bump"), 21)["x"]
    assert bump["falls"] > 0 and bump["max_fall"] > 0.05


# --- variants and metrics ---------------------------------------------------------------------------------------------

def test_gaussian_variant_keeps_centres_and_half_widths():
    assert gauss_of(["tri", 0.1, 0.35, 0.6], 0, 1)[:2] == ["gauss", 0.35]
    mean, sigma = gauss_of(["tri", 0.1, 0.35, 0.6], 0, 1)[1:]
    assert np.exp(-0.5 * ((0.225 - mean) / sigma) ** 2) == pytest.approx(0.5)     # μ = 0.5 where the triangle had
    shoulder = gauss_of(["trap", 0, 0, 0.125, 0.375], 0, 1)
    assert shoulder[1] == 0 and np.exp(-0.5 * (0.25 / shoulder[2]) ** 2) == pytest.approx(0.5)
    data = load_yaml(os.path.join(REPO, "config", "fuzzy", "gas.yaml"))
    inputs_only = gaussian(data, outputs=False)
    assert inputs_only["variables"]["risk"]["terms"] == data["variables"]["risk"]["terms"]
    assert all(t[0] == "gauss" for t in inputs_only["variables"]["max_norm"]["terms"].values())
    FuzzyEngine.from_dict(gaussian(data))                                            # still a valid rule base


def test_without_escalation_keeps_the_single_gas_cells():
    data = without_escalation(load_yaml(os.path.join(REPO, "config", "fuzzy", "gas.yaml")))
    engine = FuzzyEngine.from_dict(data)
    assert not {"second_norm", "group_index", "third_norm"} & set(engine.variables) and len(engine.rules) == 16
    assert engine.infer({"max_norm": 0.42, "second_norm": 0.15})["crisp"] == engine.infer({"max_norm": 0.42})["crisp"]
    assert engine.infer({"max_norm": 1.0})["crisp"] == pytest.approx(0.8659, abs=1e-3)


def test_metrics():
    truth = ["low", "medium", "high", "critical"]
    assert M.macro_f1(truth, truth) == 1.0 and M.macro_f1(truth, ["low"] * 4) < 0.5
    one_hot = [[1.0 if level == t else 0.0 for level in M.LEVELS] for t in truth]
    assert M.brier(truth, one_hot) == 0 and M.ece(truth, one_hot) == 0
    assert M.covers([0.3, 0.5], (0.45, 0.6)) and not M.covers([0.1, 0.2], (0.45, 0.6))


# --- the harness ------------------------------------------------------------------------------------------------------

def test_scenarios_run_through_the_real_system(tmp_path):
    base = yaml.safe_load(open(os.path.join(REPO, "config.yaml"), encoding="utf-8"))
    spec = yaml.safe_load(open(os.path.join(REPO, "experiments", "hfs", "scenarios.yaml"), encoding="utf-8"))
    by_id = {s["id"]: s for s in spec["scenarios"]}
    assert len(by_id) == len(spec["scenarios"]) and all(s["label"] in M.LEVELS for s in spec["scenarios"])
    paths = write_variants(str(tmp_path), "tri_it2", CONFIGS["tri_it2"])
    t1_paths = write_variants(str(tmp_path), "tri_type1", CONFIGS["tri_type1"])
    it2, type1 = system_config(base, paths, True), system_config(base, t1_paths, False)
    for sid, level in (("g04", "critical"), ("c05", "high"), ("a04", "critical"), ("x07", "medium"),
                       ("x09", "critical"), ("g08", "high"), ("g14", "high"), ("g15", "critical")):
        result = run_scenario(by_id[sid], it2, 42)
        assert result["level"] == level, (sid, result)
    assert run_scenario(by_id["x07"], it2, 42)["partial"] is True                  # microphone offline
    a, b = run_scenario(by_id["a04"], it2, 42), run_scenario(by_id["a04"], type1, 42)
    assert a["risk"] == b["risk"]                                                    # same crisp risk
    assert b["interval"][0] == b["interval"][1]                                      # type-1: no corridor
    assert a["interval"][1] - a["interval"][0] > 0.1                                 # IT2: the danger word's error
