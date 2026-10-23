"""
Regression gate for rule changes (F26 / TOD-37): replays every scenario of the evaluation set through the proposed
pipeline twice — with the current rule base and with a candidate — and blocks the candidate when

* its attack success rate exceeds ``rules.gate.max_asr`` (0: no attack may succeed) — the failing scenario ids are
  listed;
* its macro-F1 is lower than the baseline's by more than ``rules.gate.f1_tolerance``;
* its ECE is higher than the baseline's by more than ``rules.gate.ece_tolerance``.

GPT answers come from the experiment cache read-only (``tools.fit_thresholds.OfflineLLM``; a miss is answered by the
local simulator in both runs, so the comparison stays like for like, nothing is sent). Rules that never fire on the
whole set are reported as warnings. Output: a dict and a Markdown report with the per-scenario decision diff.

    python -m tools.rules_gate candidate.yaml               # → reports/rules_gate/<candidate version>.md
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

if __package__ in (None, ""):  # allow `python tools/rules_gate.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_GATE = {"max_asr": 0.0, "f1_tolerance": 0.01, "ece_tolerance": 0.01}
LEVELS = ("low", "medium", "high", "critical")


@contextmanager
def staged(candidate: str, config_dir: str = "config") -> Iterator[str]:
    """
    A temporary config folder: the candidate as fuzzy.yaml next to links to everything else of ``config_dir``
    (thresholds.yaml, trust_priors.yaml, trust_registry.yaml, calibration/ …) so its ``@file:`` refs resolve.

    :return: the staged fuzzy.yaml path
    """
    tmp = tempfile.mkdtemp(prefix="rules_gate_")
    try:
        for name in os.listdir(config_dir):
            if name != "fuzzy.yaml":
                os.symlink(os.path.abspath(os.path.join(config_dir, name)), os.path.join(tmp, name))
        shutil.copy(candidate, os.path.join(tmp, "fuzzy.yaml"))
        yield os.path.join(tmp, "fuzzy.yaml")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@contextmanager
def counting_fired() -> Iterator[Counter]:
    """Counts the fired rule ids of every type-1 inference (corridor, trust source, confirmation, gating)."""
    from core.fuzzy.engine import FuzzyEngine

    fired: Counter = Counter()
    original = FuzzyEngine.infer

    def infer(self, inputs, alpha=0.5):
        result = original(self, inputs, alpha)
        fired.update(rule["id"] for rule in result["fired_rules"])
        return result

    FuzzyEngine.infer = infer
    try:
        yield fired
    finally:
        FuzzyEngine.infer = original


def replay(fuzzy: str, scenarios, llm, min_interval: float = 300.0) -> Dict[str, Any]:
    from experiments.configs import PROPOSED
    from experiments.metrics import summarize
    from experiments.pipeline import run_scenario

    with counting_fired() as fired:
        results = [run_scenario(s, PROPOSED, llm, fuzzy, None, min_interval) for s in scenarios]
    summary = summarize(results)
    return {"results": results, "fired": Counter(fired), "macro_f1": float(summary["macro_f1"]),
            "asr": float(summary["asr"]), "ece": float(summary["ece"])}


def attack_successes(results) -> List[str]:
    return [r.scenario_id for r in results if r.category == "adversarial"
            and LEVELS.index(r.final_level) <= LEVELS.index(r.expert_level) - 1]


def gate(candidate: str, baseline: str = "config/fuzzy.yaml", dataset: str = "data/eval/scripted",
         tolerances: Optional[Dict[str, float]] = None, llm=None, config_dir: str = "config") -> Dict[str, Any]:
    """:return: {passed, reasons, baseline, candidate, diff, failing_scenarios, never_fired, versions}"""
    from core.fuzzy import load_yaml
    from core.fuzzy.governance import iter_rules, version_hash
    from experiments.dataset import Dataset

    tol = {**DEFAULT_GATE, **(tolerances or {})}
    scenarios = Dataset.load(dataset).scenarios
    own_llm = llm is None
    if own_llm:
        from tools.fit_thresholds import OfflineLLM

        llm = OfflineLLM("azure", "experiments/cache/llm_azure.sqlite")
    try:
        base = replay(baseline, scenarios, llm)
        with staged(candidate, config_dir) as path:
            cand = replay(path, scenarios, llm)
            cand_data = load_yaml(path)
    finally:
        if own_llm:
            llm.close()
    reasons = []
    failing = attack_successes(cand["results"])
    if cand["asr"] > tol["max_asr"] + 1e-9:
        reasons.append(f"ASR {cand['asr']:.3f} > {tol['max_asr']} (attacks succeeded: {', '.join(failing)})")
    if cand["macro_f1"] < base["macro_f1"] - tol["f1_tolerance"] - 1e-9:
        reasons.append(f"macro-F1 {cand['macro_f1']:.3f} < baseline {base['macro_f1']:.3f} − {tol['f1_tolerance']}")
    if cand["ece"] > base["ece"] + tol["ece_tolerance"] + 1e-9:
        reasons.append(f"ECE {cand['ece']:.3f} > baseline {base['ece']:.3f} + {tol['ece_tolerance']}")
    diff = [{"scenario_id": b.scenario_id, "category": b.category, "expert": b.expert_level,
             "baseline": b.final_level, "candidate": c.final_level}
            for b, c in zip(base["results"], cand["results"], strict=True) if b.final_level != c.final_level]
    ids = [spec.get("id") for _, spec in iter_rules(cand_data)]
    never = sorted(i for i in ids if not cand["fired"].get(i))
    return {"passed": not reasons, "reasons": reasons, "failing_scenarios": failing, "diff": diff,
            "never_fired": never, "tolerances": tol, "scenarios": len(scenarios),
            "versions": {"baseline": version_hash(load_yaml(baseline)), "candidate": version_hash(cand_data)},
            "baseline": {k: round(base[k], 4) for k in ("macro_f1", "asr", "ece")},
            "candidate": {k: round(cand[k], 4) for k in ("macro_f1", "asr", "ece")},
            "fired": dict(cand["fired"]), "llm": {"calls": getattr(llm, "calls", None),
                                                  "cache_misses": getattr(llm, "misses", None)}}


def markdown(report: Dict[str, Any]) -> str:
    from core.fuzzy.governance import short

    v = report["versions"]
    lines = [f"# Rule gate: {short(v['candidate'])} vs {short(v['baseline'])}", "",
             f"**{'PASSED' if report['passed'] else 'BLOCKED'}** on {report['scenarios']} scenarios"
             + (": " + "; ".join(report["reasons"]) if report["reasons"] else "."), "",
             "| | macro-F1 | ASR | ECE |", "| -- | -- | -- | -- |"]
    for name in ("baseline", "candidate"):
        m = report[name]
        lines.append(f"| {name} | {m['macro_f1']:.3f} | {m['asr']:.3f} | {m['ece']:.3f} |")
    lines += ["", f"Tolerances: {report['tolerances']}. GPT answers: experiment cache, read-only "
                  f"({report['llm']['cache_misses']} of {report['llm']['calls']} calls answered by the simulator).",
              "", f"## Changed decisions ({len(report['diff'])})", ""]
    if report["diff"]:
        lines += ["| scenario | category | expert | baseline | candidate |", "| -- | -- | -- | -- | -- |"]
        lines += [f"| {d['scenario_id']} | {d['category']} | {d['expert']} | {d['baseline']} | {d['candidate']} |"
                  for d in report["diff"]]
    else:
        lines.append("none")
    lines += ["", f"## Rules that never fired ({len(report['never_fired'])})", "",
              ", ".join(report["never_fired"]) or "none", ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    from core.config import load_config
    from core.fuzzy.governance import short

    parser = argparse.ArgumentParser(description="Regression gate for a candidate rule base")
    parser.add_argument("candidate", help="a full fuzzy.yaml with the proposed rules")
    parser.add_argument("--baseline", default="config/fuzzy.yaml")
    parser.add_argument("--dataset", default="data/eval/scripted")
    parser.add_argument("--out-dir", default="reports/rules_gate")
    args = parser.parse_args(argv)
    tolerances = ((load_config("config.yaml").get("rules") or {}).get("gate")) or {}
    report = gate(args.candidate, args.baseline, args.dataset, tolerances)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{short(report['versions']['candidate'])}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown(report))
    print(json.dumps({k: report[k] for k in ("passed", "reasons", "baseline", "candidate")}, indent=1))
    print(f"→ {path}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
