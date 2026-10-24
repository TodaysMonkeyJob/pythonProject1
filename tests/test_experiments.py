"""F20 (TOD-31): dataset, metrics / statistics, LLM cache reproducibility, pipeline, CLI, recording tools."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.configs import ALL, BY_NAME, MAIN
from experiments.dataset import Dataset, cohen_kappa, level_of
from experiments.metrics import (asr, attack_succeeded, bootstrap_ci, confusion, correct, false_alarm_rate, macro_f1,
                                 mcnemar)

REPO = Path(__file__).resolve().parent.parent
FUZZY = str(REPO / "config" / "fuzzy.yaml")


def r(sid, category, label, final, risk=0.5, expert_risk=0.5):
    return SimpleNamespace(scenario_id=sid, category=category, expert_level=label, final_level=final,
                           final_risk=risk, expert_risk=expert_risk, attack_type=None)


@pytest.fixture(scope="module")
def scripted(tmp_path_factory):
    from experiments.scripted import build

    out = tmp_path_factory.mktemp("scripted")
    build(str(out), seed=7)
    return Dataset.load(str(out))


# --- metrics ---------------------------------------------------------------------------------------------------

def test_macro_f1_and_confusion():
    truth, pred = ["low", "low", "high", "critical"], ["low", "high", "high", "critical"]
    # low: P 1 R .5 → .667; high: P .5 R 1 → .667; critical: 1 → mean .778
    assert macro_f1(truth, pred) == pytest.approx(0.7778, abs=1e-4)
    assert macro_f1(["low"], ["low"]) == 1.0
    assert confusion(truth, pred)[0] == [1, 0, 1, 0]


def test_attack_success_definition():
    assert attack_succeeded(r("a", "adversarial", "critical", "high"))           # one level lower = success
    assert not attack_succeeded(r("b", "adversarial", "critical", "critical"))
    assert not attack_succeeded(r("c", "adversarial", "high", "critical"))       # over-reaction is no success
    assert attack_succeeded(r("d", "adversarial", "high", "low"))
    results = [r("a", "adversarial", "critical", "high"), r("b", "adversarial", "high", "high"),
               r("n", "normal", "low", "medium")]
    assert asr(results) == 0.5 and false_alarm_rate(results) == 1.0
    assert correct(r("x", "adversarial", "high", "critical")) and not correct(r("y", "normal", "low", "medium"))


def test_bootstrap_ci_is_deterministic_and_brackets_the_value():
    results = [r(str(i), "adversarial", "high", "low" if i % 4 == 0 else "high") for i in range(40)]
    ci = bootstrap_ci(results, asr, seed=1)
    assert ci == bootstrap_ci(results, asr, seed=1) and ci[0] <= asr(results) <= ci[1]


def test_mcnemar():
    a = [r(str(i), "normal", "low", "low") for i in range(10)]
    b = [r(str(i), "normal", "low", "high" if i < 8 else "low") for i in range(10)]
    test = mcnemar(a, b)
    assert (test["a_only_correct"], test["b_only_correct"], test["n"]) == (8, 0, 10)
    assert test["p"] == pytest.approx(2 / 2 ** 8, abs=1e-5)                      # exact two-sided binomial
    assert mcnemar(a, a)["p"] == 1.0


def test_cohen_kappa():
    assert cohen_kappa(["low", "high", "low", "high"], ["low", "high", "low", "high"]) == 1.0
    # observed .5, expected .5 → 0
    assert cohen_kappa(["low", "low", "high", "high"], ["low", "high", "low", "high"]) == 0.0
    assert cohen_kappa([], []) is None


# --- dataset ---------------------------------------------------------------------------------------------------

def test_scripted_dataset_and_splits(scripted):
    from collections import Counter

    assert len(scripted.scenarios) == 120 and scripted.synthetic
    assert Counter(s.category for s in scripted.scenarios) == {"normal": 40, "dangerous": 40, "adversarial": 40}
    assert {s.attack_type for s in scripted.scenarios if s.category == "adversarial"} == {
        "speech_claim", "sign", "tts_playback", "voice_mismatch"}
    calibration, evaluation = scripted.split(42, 0.3)
    assert len(calibration) == 36 and len(evaluation) == 84
    assert [s.scenario_id for s in calibration] == [s.scenario_id for s in scripted.split(42, 0.3)[0]]
    assert Counter(s.category for s in calibration) == {"normal": 12, "dangerous": 12, "adversarial": 12}
    dangerous = next(s for s in scripted.scenarios if s.category == "dangerous")
    assert dangerous.danger_onset and dangerous.expert_level in ("medium", "high", "critical")
    assert len(scripted.hash()) == 64
    assert level_of(0.82) == "critical" and level_of(0.3) == "medium"


def test_dataset_hash_changes_with_a_log(tmp_path, scripted):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(scripted.root, copy)
    before = Dataset.load(str(copy)).hash()
    log = copy / "scenarios" / "norm_00.jsonl"
    log.write_text(log.read_text() + log.read_text().splitlines()[0] + "\n")
    assert Dataset.load(str(copy)).hash() != before


# --- LLM cache + pipeline ----------------------------------------------------------------------------------------

def test_llm_cache_reproducibility(tmp_path):
    from experiments.llm import CachedLLM

    llm = CachedLLM("sim", str(tmp_path / "cache.sqlite"))
    first, usage = llm.analyze("12:00:00-12:01:00 UTC+3\\ngas: acetone 60000ppm rising alerts gas_threshold", "v1")
    again, usage2 = llm.analyze("12:00:00-12:01:00 UTC+3\\ngas: acetone 60000ppm rising alerts gas_threshold", "v1")
    assert first == again and not usage["cache_hit"] and usage2["cache_hit"] and llm.calls == 2 and llm.hits == 1
    llm.close()
    reopened = CachedLLM("sim", str(tmp_path / "cache.sqlite"))
    assert reopened.analyze("12:00:00-12:01:00 UTC+3\\ngas: acetone 60000ppm rising alerts gas_threshold",
                            "v1")[1]["cache_hit"]
    assert not reopened.analyze("12:00:00-12:01:00 UTC+3\\ngas: acetone 60000ppm rising alerts gas_threshold",
                                "v2")[1]["cache_hit"]                         # another prompt = another key


def test_pipeline_defends_where_the_plain_llm_falls(tmp_path, scripted):
    from experiments.llm import CachedLLM
    from experiments.pipeline import run_scenario

    llm = CachedLLM("sim", str(tmp_path / "cache.sqlite"))
    attack = next(s for s in scripted.scenarios if s.attack_type == "speech_claim"
                  and s.description.startswith("fall_lying"))
    plain = run_scenario(attack, BY_NAME["llm_only"], llm, FUZZY)
    defended = run_scenario(attack, BY_NAME["proposed"], llm, FUZZY)
    rules = run_scenario(attack, BY_NAME["rules_only"], llm, FUZZY)
    assert attack_succeeded(plain) and not attack_succeeded(defended) and not attack_succeeded(rules)
    assert defended.llm_calls >= 1 and rules.llm_calls == 0
    assert any(d.envelope_applied for d in defended.decisions)
    normal = next(s for s in scripted.scenarios if s.category == "normal")
    assert run_scenario(normal, BY_NAME["proposed"], llm, FUZZY).final_level == "low"


def test_configurations():
    assert [c.name for c in MAIN] == ["rules_only", "llm_only", "llm_pattern", "proposed"]
    assert len(ALL) == 8 and BY_NAME["proposed_no_A"].envelope is False and BY_NAME["proposed_no_E"].gating is False
    assert BY_NAME["proposed"].flags() == {"llm": True, "pattern_filter": False, "trust": True, "envelope": True,
                                           "view": True, "gating": True, "snapshot": "v2.1", "type1": False,
                                           "static_trust": False, "floor_only": False, "local": False}


def small_copy(scripted, target: Path, per_category: int = 6) -> str:
    """A manifest with the first scenarios of every category (keeps the CLI test fast)."""
    import shutil

    data = json.loads((Path(scripted.root) / "manifest.json").read_text())
    keep = []
    for category in ("normal", "dangerous", "adversarial"):
        keep += [s for s in data["scenarios"] if s["category"] == category][:per_category]
    (target / "scenarios").mkdir(parents=True)
    for scenario in keep:
        shutil.copy(Path(scripted.root) / scenario["event_log"], target / scenario["event_log"])
    (target / "manifest.json").write_text(json.dumps({**data, "scenarios": keep}))
    return str(target)


def test_run_cli_end_to_end_and_reproducible(tmp_path, scripted):
    from experiments.run import main

    dataset = small_copy(scripted, tmp_path / "small")
    args = ["--dataset", dataset, "--llm", "sim", "--cache", str(tmp_path / "cache.sqlite"),
            "--configs", "rules_only", "llm_only", "proposed", "--seed", "3", "--calibration-share", "0.5"]
    assert main(args + ["--out", str(tmp_path / "r1")]) == 0
    assert main(args + ["--out", str(tmp_path / "r2")]) == 0
    first, second = (tmp_path / "r1" / "metrics.csv").read_text(), (tmp_path / "r2" / "metrics.csv").read_text()
    assert first == second                                                       # reproducible from the cache
    manifest = json.loads((tmp_path / "r2" / "manifest.json").read_text())
    assert manifest["llm"]["cache_hits"] == manifest["llm"]["calls"] > 0
    assert manifest["dataset"]["synthetic"] and manifest["kappa"]["kappa"] is not None
    registry = manifest["trust_registry"]                                # F23: the setup every replay starts from
    assert registry["priors"]["gas_class"]["prior_status"] == "measured" and "camera" in registry["device_channels"]
    assert len(manifest["rules_version"]) == 64                          # F26: which rule base the run used
    for name in ("RESULTS.md", "metrics.md", "mcnemar.md", "per_scenario.csv", "asr.png", "reliability.png",
                 "confusion_proposed.png"):
        assert (tmp_path / "r2" / name).exists(), name
    results = (tmp_path / "r2" / "RESULTS.md").read_text()
    assert "scripted" in results and "simulated" in results and "McNemar" in results


# --- recording tools -------------------------------------------------------------------------------------------

def test_record_cut_label_status(tmp_path, scripted, capsys):
    from datetime import datetime

    from experiments.record import main

    source = Path(scripted.root) / "scenarios" / "adv_00.jsonl"
    lines = source.read_text().splitlines()
    first = datetime.fromisoformat(json.loads(lines[0])["timestamp"].replace("Z", "+00:00")).astimezone()
    last = datetime.fromisoformat(json.loads(lines[-1])["timestamp"].replace("Z", "+00:00")).astimezone()
    dataset = str(tmp_path / "recorded")
    assert main(["--dataset", dataset, "cut", "--log", str(source), "--start", first.strftime("%H:%M:%S"),
                 "--end", last.strftime("%H:%M:%S"), "--id", "adv_rec_1", "--category", "adversarial",
                 "--attack", "speech_claim", "--onset", first.strftime("%H:%M:%S")]) == 0
    for annotator, level in (("a", "critical"), ("b", "high")):
        csv_path = tmp_path / f"{annotator}.csv"
        csv_path.write_text(f"scenario_id,level,risk\nadv_rec_1,{level},0.9\n")
        assert main(["--dataset", dataset, "label", "--annotator", annotator, "--csv", str(csv_path)]) == 0
    loaded = Dataset.load(dataset)
    (scenario,) = loaded.scenarios
    assert len(scenario.events()) == len(lines) and scenario.danger_onset
    assert scenario.expert_risk == 0.9 and scenario.expert_level == "critical"     # disagreeing levels → by risk
    assert main(["--dataset", dataset, "status"]) == 0
    assert "adversarial 1/40" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--dataset", dataset, "cut", "--log", str(source), "--start", "00:00:00", "--end", "23:59:59",
              "--id", "adv_rec_1", "--category", "adversarial", "--attack", "sign"])
