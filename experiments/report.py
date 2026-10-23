"""
Experiment report (F20 / TOD-31): ``reports/experiment_<date>/`` with

* ``metrics.csv`` / ``metrics.md`` — every configuration: macro-F1 [95 % CI], ASR [CI], false alarms, ECE, Brier,
  LLM calls / h, cost / h, latency; ``ablation.md`` — the proposed method vs each ablation; ``mcnemar.md``;
* ``per_scenario.csv`` — final / max level, risk, calls, latency per scenario and configuration;
* ``confusion_<config>.png``, ``reliability.png``, ``asr.png`` (bar chart with CIs);
* ``manifest.json`` — git commit, config / prompt hashes, LLM backend + deployment, dataset hash, seed, splits, κ;
* ``RESULTS.md`` — a short summary for the thesis chapter (with the caveats of the run).
"""
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List

from experiments.dataset import LEVELS

FILES_TO_HASH = ["config/fuzzy.yaml", "config.yaml", "prompts/security_analyst.md",
                 "experiments/prompts/security_analyst_v1.md"]


def _sha(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return "missing"


def _git() -> Dict[str, Any]:
    def run(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def _ci(value, ci) -> str:
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def _fmt(v) -> str:
    return "—" if v is None else f"{v}"


def write(out_dir: str, summaries: Dict[str, Dict[str, Any]], results: Dict[str, List[Any]],
          mcnemar: Dict[str, Dict[str, Any]], manifest: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    names = list(summaries)
    columns = ["config", "macro_f1", "macro_f1_ci", "asr", "asr_ci", "false_alarm_rate", "ece", "brier",
               "llm_calls", "llm_calls_per_hour", "cost_usd_per_hour", "latency_median_s", "latency_p95_s",
               "critical_within_5s", "missed_alarms"]
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for name in names:
            s = summaries[name]
            writer.writerow([name, s["macro_f1"], s["macro_f1_ci"], s["asr"], s["asr_ci"], s["false_alarm_rate"],
                             s["ece"], s["brier"], s["llm_calls"], s["llm_calls_per_hour"], s["cost_usd_per_hour"],
                             s["latency"]["median_s"], s["latency"]["p95_s"], s["latency"]["critical_within_5s"],
                             s["latency"]["missed"]])
    table = ["| Configuration | Macro-F1 [95 % CI] | ASR [95 % CI] | False alarms | ECE | Brier | LLM calls / h | "
             "Cost / h, $ | Latency median / p95, s | Critical ≤ 5 s |", "|" + " -- |" * 10]
    for name in names:
        s = summaries[name]
        lat = s["latency"]
        table.append(f"| {name} | {_ci(s['macro_f1'], s['macro_f1_ci'])} | {_ci(s['asr'], s['asr_ci'])} | "
                     f"{s['false_alarm_rate']:.3f} | {s['ece']:.3f} | {s['brier']:.3f} | {s['llm_calls_per_hour']} | "
                     f"{s['cost_usd_per_hour']:.4f} | {_fmt(lat['median_s'])} / {_fmt(lat['p95_s'])} | "
                     f"{_fmt(lat['critical_within_5s'])} |")
    with open(os.path.join(out_dir, "metrics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(table) + "\n")
    if "proposed" in summaries:
        base = summaries["proposed"]
        rows = ["| Variant | ΔMacro-F1 | ΔASR | ΔECE | ΔLLM calls / h |", "| -- | -- | -- | -- | -- |",
                f"| proposed (A+B+C+E) | {base['macro_f1']:.3f} | {base['asr']:.3f} | {base['ece']:.3f} | "
                f"{base['llm_calls_per_hour']} |"]
        for name in [n for n in names if n.startswith("proposed_no_")]:
            s = summaries[name]
            calls = s["llm_calls_per_hour"] - base["llm_calls_per_hour"]
            rows.append(f"| {name} | {s['macro_f1'] - base['macro_f1']:+.3f} | {s['asr'] - base['asr']:+.3f} | "
                        f"{s['ece'] - base['ece']:+.3f} | {calls:+.1f} |")
        with open(os.path.join(out_dir, "ablation.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
    rows = ["| proposed vs | only proposed correct | only other correct | n | exact McNemar p |",
            "| -- | -- | -- | -- | -- |"]
    for name, test in mcnemar.items():
        rows.append(f"| {name} | {test['a_only_correct']} | {test['b_only_correct']} | {test['n']} | {test['p']:.3g} |")
    with open(os.path.join(out_dir, "mcnemar.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    with open(os.path.join(out_dir, "per_scenario.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "scenario_id", "category", "attack_type", "expert_level", "expert_risk",
                         "final_level", "max_level", "final_risk", "llm_calls", "cache_hits", "latency_s",
                         "decisions"])
        for name in names:
            for r in results[name]:
                writer.writerow([name, r.scenario_id, r.category, r.attack_type or "", r.expert_level, r.expert_risk,
                                 r.final_level, r.max_level, r.final_risk, r.llm_calls, r.cache_hits,
                                 _fmt(r.latency_s), "; ".join(f"{d.source}:{d.level_final}" for d in r.decisions)])
    _plots(out_dir, summaries)
    manifest = {**manifest, "git": _git(), "hashes": {p: _sha(p) for p in FILES_TO_HASH},
                "written_at": datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    _results_md(out_dir, summaries, mcnemar, manifest)
    return out_dir


def _plots(out_dir: str, summaries: Dict[str, Dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summaries)
    for name, s in summaries.items():
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        ax.imshow(s["confusion"], cmap="Blues")
        for i, row in enumerate(s["confusion"]):
            for j, value in enumerate(row):
                ax.text(j, i, value, ha="center", va="center", fontsize=9)
        ax.set_xticks(range(4), LEVELS, rotation=30, fontsize=8)
        ax.set_yticks(range(4), LEVELS, fontsize=8)
        ax.set_xlabel("final level")
        ax.set_ylabel("expert level")
        ax.set_title(name, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"confusion_{name}.png"), dpi=120)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    for name in [n for n in ("rules_only", "llm_only", "proposed") if n in summaries]:
        bins = summaries[name]["reliability"]
        ax.plot([b["mean_pred"] for b in bins], [b["mean_target"] for b in bins], "o-", label=name)
    ax.set_xlabel("final risk (mean per bin)")
    ax.set_ylabel("expert risk (mean per bin)")
    ax.legend(fontsize=8)
    ax.set_title("Reliability", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reliability.png"), dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(5, len(names) * 0.9), 3.4))
    values = [summaries[n]["asr"] for n in names]
    errors = [[v - summaries[n]["asr_ci"][0] for n, v in zip(names, values, strict=True)],
              [summaries[n]["asr_ci"][1] - v for n, v in zip(names, values, strict=True)]]
    ax.bar(range(len(names)), values, yerr=errors, capsize=4, color=["#c44" if n == "proposed" else "#888"
                                                                      for n in names])
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("attack success rate")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "asr.png"), dpi=120)
    plt.close(fig)


def _results_md(out_dir: str, summaries, mcnemar, manifest) -> None:
    lines = [f"# Results — {manifest['dataset']['name']} ({manifest['written_at'][:10]})", ""]
    caveats = []
    if manifest["dataset"].get("synthetic"):
        caveats.append("the dataset is **scripted** (generated from templates by `experiments/scripted.py`) — it "
                       "validates the harness, it is not the recorded evaluation set")
    if manifest["llm"]["backend"] == "sim":
        caveats.append("the LLM is the deterministic **simulated** analyst (`experiments/llm.py`), not GPT — numbers "
                       "of the LLM configurations say nothing about GPT yet")
    if caveats:
        lines += ["> **Caveats:** " + "; ".join(caveats) + ".", ""]
    split = manifest["split"]
    lines += [f"Evaluation split: {split['evaluation']} scenarios (calibration {split['calibration']}, seed "
              f"{manifest['seed']}); annotator agreement κ = {_fmt((manifest.get('kappa') or {}).get('kappa'))}; "
              f"LLM: {manifest['llm']['backend']} / {manifest['llm']['model']}, temperature "
              f"{manifest['llm']['temperature']}.", ""]
    with open(os.path.join(out_dir, "metrics.md"), encoding="utf-8") as f:
        lines += ["## Metrics", "", f.read()]
    if "proposed" in summaries:
        p = summaries["proposed"]
        best_baseline = min((n for n in summaries if not n.startswith("proposed")), key=lambda n: summaries[n]["asr"])
        lines += ["## Summary", "",
                  f"* Attack success rate: proposed {p['asr']:.3f} {p['asr_ci']} vs best baseline "
                  f"{best_baseline} {summaries[best_baseline]['asr']:.3f} {summaries[best_baseline]['asr_ci']}; "
                  f"LLM only {summaries.get('llm_only', p)['asr']:.3f}.",
                  f"* Macro-F1 (normal + dangerous): proposed {p['macro_f1']:.3f} {p['macro_f1_ci']}.",
                  f"* Calibration: ECE {p['ece']:.3f}, Brier {p['brier']:.3f}; false alarm rate "
                  f"{p['false_alarm_rate']:.3f}.",
                  f"* Efficiency: {p['llm_calls_per_hour']} LLM calls / h (LLM only: "
                  f"{summaries.get('llm_only', p)['llm_calls_per_hour']}).", ""]
    calibration = manifest.get("calibration") or {}
    if calibration.get("fitted"):                      # F28: in-sample vs leave-one-out, constraints, used or not
        fitted, cv = calibration.get("fit") or {}, calibration.get("cv") or {}
        check = calibration.get("constraints") or {}
        lines += ["## Calibration of the LLM levels (C)", "",
                  f"Fitted on {calibration['n']} of {calibration.get('split', calibration['n'])} calibration scenarios "
                  f"({calibration.get('rejected_deescalations', 0)} answers rejected by the corridor as manipulated "
                  f"left out). In-sample Brier {_fmt(fitted.get('brier'))}, "
                  f"ECE {_fmt(fitted.get('ece'))}; leave-one-out Brier {_fmt(cv.get('brier'))}, ECE "
                  f"{_fmt(cv.get('ece'))} (fixed mapping: Brier {_fmt((cv.get('fixed') or {}).get('brier'))}, ECE "
                  f"{_fmt((cv.get('fixed') or {}).get('ece'))}). Constraints "
                  f"{'hold' if check.get('ok') else 'VIOLATED'}: monotone centres {check.get('centres')}, "
                  f"min support {check.get('min_support')} ≥ {check.get('min_width')}. "
                  + ("The fitted sets are used." if calibration.get("used", True) else
                     "Not used: " + calibration.get("note", "") + "."), ""]
    if os.path.isfile(os.path.join(out_dir, "ablation.md")):
        with open(os.path.join(out_dir, "ablation.md"), encoding="utf-8") as f:
            lines += ["## Ablations", "", f.read()]
    with open(os.path.join(out_dir, "mcnemar.md"), encoding="utf-8") as f:
        lines += ["## McNemar (per-scenario correctness, same scenarios)", "", f.read()]
    lines += ["Figures: `asr.png`, `reliability.png`, `confusion_<config>.png`; per scenario: `per_scenario.csv`; "
              "reproducibility: `manifest.json` (git commit, config / prompt hashes, dataset hash) and the LLM cache."]
    with open(os.path.join(out_dir, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
