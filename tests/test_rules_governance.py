"""F26 (TOD-37): rule metadata, validation additions, version hash, regression gate, approval, rollback, stats."""
import json
import shutil
from pathlib import Path

import pytest
import yaml

from core.fuzzy import FuzzyConfigError, FuzzyEngine, load_yaml
from core.fuzzy.governance import metadata_errors, normalized, version_hash
from core.fuzzy.rules import parse_rule
from core.fuzzy.validate import subsumed

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "fuzzy.yaml"


def config_copy(tmp_path) -> Path:
    """A private config folder (fuzzy.yaml + the files its @refs need)"""
    target = tmp_path / "config"
    shutil.copytree(REPO / "config", target)
    return target / "fuzzy.yaml"


def edit(path: Path, change) -> None:
    data = yaml.safe_load(path.read_text())
    change(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))


# --- metadata and validation ------------------------------------------------------------------------------------

def test_every_rule_has_metadata_and_validation_fails_without_it(tmp_path):
    data = load_yaml(str(CONFIG))
    assert metadata_errors(data) == [] and len(normalized(data)) == 58
    path = config_copy(tmp_path)
    edit(path, lambda d: d["rules"][0].pop("meta"))
    assert metadata_errors(load_yaml(str(path))) == ["risk/R1: metadata missing source, author, rationale, added"]
    from core.fuzzy.__main__ import main

    assert main(["--config", str(path), "--check"]) == 1


def test_duplicates_in_any_clause_order_and_subsumed_rules():
    base = load_yaml(str(CONFIG))
    extra = {"id": "X1", "rule": "IF gas_trend IS sharply_rising AND gas_concentration IS elevated THEN risk IS high"}
    swapped = {**base, "rules": base["rules"] + [extra]}
    with pytest.raises(FuzzyConfigError, match="X1 conflicts with R2"):
        FuzzyEngine.from_dict(swapped)
    rules = [parse_rule("IF gas_concentration IS dangerous THEN risk IS critical", 1),
             parse_rule("IF gas_concentration IS dangerous AND motion_level IS high THEN risk IS critical", 2)]
    assert subsumed(rules) == ["R2 is subsumed by R1 (same conclusion risk IS critical, R1 needs fewer conditions)"]
    assert subsumed(FuzzyEngine.from_dict(base).rules) == []


def test_version_ignores_metadata_but_not_rules(tmp_path):
    path = config_copy(tmp_path)
    original = version_hash(load_yaml(str(path)))
    assert original == version_hash(load_yaml(str(CONFIG)))
    edit(path, lambda d: d["rules"][0]["meta"].update(rationale="reworded"))
    assert version_hash(load_yaml(str(path))) == original
    edit(path, lambda d: d["rules"][13].update(weight=0.4))
    assert version_hash(load_yaml(str(path))) != original


# --- gate, approval, rollback -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def toy_dataset(tmp_path_factory):
    """A few scripted scenarios: normal + adversarial gas attacks (the gate must stay fast in tests)"""
    from experiments.scripted import build

    root = tmp_path_factory.mktemp("toy")
    build(str(root), 42)
    manifest = json.loads((root / "manifest.json").read_text())
    adversarial = [s for s in manifest["scenarios"] if s["category"] == "adversarial"
                   and "gas" in s.get("description", "").lower()][:3]
    normal = [s for s in manifest["scenarios"] if s["category"] == "normal"][:2]
    (root / "manifest.json").write_text(json.dumps({**manifest, "scenarios": adversarial + normal}))
    return str(root)


def sim_llm():
    from tools.fit_thresholds import OfflineLLM

    return OfflineLLM("sim", "/nonexistent/cache.sqlite")             # no cache: the local simulator answers


def weaken_gas(data):
    for rule in data["rules"]:
        if rule["id"] in ("R1", "R2", "R3", "R4"):
            rule["rule"] = rule["rule"].rsplit("THEN", 1)[0] + "THEN risk IS low"


def test_gate_blocks_a_rule_change_that_lets_an_attack_succeed(tmp_path, toy_dataset):
    from tools.rules_gate import gate, markdown

    path = config_copy(tmp_path)
    same = gate(str(path), str(path), toy_dataset, llm=sim_llm(), config_dir=str(path.parent))
    assert same["passed"] and same["diff"] == [] and same["baseline"] == same["candidate"]
    weak = tmp_path / "weak.yaml"
    shutil.copy(path, weak)
    edit(weak, weaken_gas)
    report = gate(str(weak), str(path), toy_dataset, llm=sim_llm(), config_dir=str(path.parent))
    assert not report["passed"] and report["failing_scenarios"]
    assert any(r.startswith("ASR") and report["failing_scenarios"][0] in r for r in report["reasons"])
    assert report["diff"] and "BLOCKED" in markdown(report)


def test_propose_approve_and_rollback_keep_a_changelog(tmp_path, toy_dataset):
    from tools.rules import RuleStore

    path = config_copy(tmp_path)
    store = RuleStore(str(tmp_path / "rules"), str(path))
    base = store.current()
    assert store.register_baseline("tester", "author", "baseline")["to"] == base
    assert store.register_baseline("tester", "author", "baseline") is None                # once
    candidate = tmp_path / "candidate.yaml"
    shutil.copy(path, candidate)
    edit(candidate, lambda d: d["rules"][13].update(weight=0.4))                       # R14 a bit weaker
    report = store.propose(str(candidate), toy_dataset, llm=sim_llm())
    assert report["passed"], report.get("gate", {}).get("reasons")
    with pytest.raises(SystemExit, match="--by"):
        store.approve(report["proposal"], "", "supervisor", "ok")
    entry = store.approve(report["proposal"], "Supervisor", "supervisor", "weaker daytime strangers")
    new = store.current()
    assert entry["from"] == base and entry["to"] == new != base and entry["role"] == "supervisor"
    assert (tmp_path / "rules" / "versions" / f"{new}.yaml").is_file()
    back = store.rollback(base[:12], "Owner", "security_owner", "back to the baseline")
    assert store.current() == base and back["from"] == new
    assert [e["action"] for e in store.history()] == ["baseline", "approve", "rollback"]

    broken = tmp_path / "broken.yaml"
    shutil.copy(path, broken)
    edit(broken, lambda d: d["rules"][0].pop("meta"))
    rejected = store.propose(str(broken), toy_dataset, llm=sim_llm())
    assert not rejected["passed"] and rejected["checks"]["errors"] and "gate" not in rejected
    with pytest.raises(SystemExit, match="cannot be approved"):
        store.approve(rejected["proposal"], "Supervisor", "supervisor", "no")


# --- monitoring and version in decisions --------------------------------------------------------------------------

def test_rule_stats_and_version_in_decisions():
    from modules.envelope import RiskEnvelope
    from modules.llm.rule_stats import RuleStats

    stats = RuleStats()
    stats.record("v1", ["R1", "R2", "R1"])
    stats.record("v1", ["R1"])
    stats.outcome("v1", ["R1"], bound_kept=True)
    stats.outcome("v1", ["R2"], bound_kept=False)
    rows = {r["rule_id"]: r for r in stats.table("v1")}
    assert (rows["R1"]["fired"], rows["R1"]["confirmed"], rows["R2"]["rejected"]) == (2, 1, 1)
    envelope = RiskEnvelope.from_config(str(CONFIG), None)
    decision = envelope.decide("lab", "low", now=1.7e9, local_hour=12)
    assert decision["rules_version"] == version_hash(load_yaml(str(CONFIG)))[:12]
    assert isinstance(decision["fired_ids"], list)
