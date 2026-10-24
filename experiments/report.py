"""
Experiment report (F20 / TOD-31): ``reports/experiment_<date>/`` with

* ``metrics.csv`` / ``metrics.md`` — every configuration: macro-F1 [95 % CI], ASR [CI], false alarms, ECE, Brier,
  LLM calls / h, cost / h, latency; ``ablation.md`` — the proposed method vs each ablation; ``mcnemar.md``;
* ``per_scenario.csv`` — final / max level, risk, calls, latency per scenario and configuration;
* ``confusion_<config>.png``, ``reliability.png``, ``asr.png`` (bar chart with CIs);
* ``manifest.json`` — git commit, config / prompt hashes, LLM backend + deployment, dataset hash, seed, splits, κ;
* ``RESULTS.md`` — a short summary for the thesis chapter (with the caveats of the run).

Experiment v2 (F29 / TOD-40, ``write_v2``) → ``reports/experiment_v2_<date>/`` adds ``corridor.md`` (width vs trust
uncertainty, agreement h, policy branch shares, escalation reviews / h and their correctness), ``failure.md`` (the
sensor-failure block, registry vs static trust), ``questions.md`` (one row per department question, answered by
numbers), figures ``width_vs_u.png``, ``h_hist.png``, ``branches.png``, ``failure.png``; McNemar vs ``proposed_v2``;
the manifest also hashes the thresholds / trust files.
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
                 "experiments/prompts/security_analyst_v1.md", "config/thresholds.yaml", "config/trust_registry.yaml",
                 "config/trust_priors.yaml"]


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
    reference = manifest.get("reference", "proposed")
    rows = [f"| {reference} vs | only {reference} correct | only other correct | n | exact McNemar p |",
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
    for name in [n for n in ("rules_only", "llm_only", "proposed", "proposed_v2") if n in summaries]:
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
    ax.bar(range(len(names)), values, yerr=errors, capsize=4,
           color=["#c44" if n in ("proposed", "proposed_v2") else "#888" for n in names])
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


# --- experiment v2 (F29 / TOD-40) -----------------------------------------------------------------------------------

QUESTIONS = [("Does type 2 add value? (IT2 vs type-1 corridor)", "proposed_v2_type1"),
             ("Does measured trust add value? (registry vs TOD-27 constants)", "proposed_v2_static_trust"),
             ("Does the two-sided corridor add value? (vs floor-only max-projection)", "proposed_v2_floor_only"),
             ("Why not fully local? (corridor + local claim detector, no GPT)", "proposed_v2_local")]


def _delta(a, b) -> str:
    return "—" if a is None or b is None else f"{b - a:+.3f}"


def _verdict(test: Dict[str, Any]) -> str:
    """Paired correctness: which side wins scenarios, and whether the difference is significant (p < 0.05)"""
    a, b, p = test["a_only_correct"], test["b_only_correct"], test["p"]
    if a == b == 0:
        return "same decisions on every scenario"
    side = "proposed_v2" if a > b else "the variant"
    significance = "significant" if p < 0.05 else "not significant"
    return f"{side} better on {max(a, b)} vs {min(a, b)} scenarios ({significance}, p = {p:.3g})"


def questions_table(summaries, tests, corridor, failure) -> List[str]:
    ref = summaries["proposed_v2"]
    rows = ["| Question | Variant | Macro-F1 (v2 → variant) | ASR | ECE | LLM calls / h | Extra | Answer |",
            "| -- | -- | -- | -- | -- | -- | -- | -- |"]
    for question, name in QUESTIONS:
        if name not in summaries:
            rows.append(f"| {question} | {name} | not run | | | | | |")
            continue
        s = summaries[name]
        extra = ""
        if name == "proposed_v2_type1":
            extra = (f"mean width {_fmt(corridor['proposed_v2']['mean_width'])} vs "
                     f"{_fmt(corridor[name]['mean_width'])}; corr(w, u) "
                     f"{_fmt(corridor['proposed_v2']['width_u_corr'])}")
        elif name == "proposed_v2_static_trust" and failure:
            fr, fs = failure.get("proposed_v2") or {}, failure.get(name) or {}
            extra = (f"sensor failure: ASR {_fmt(fr.get('asr'))} vs {_fmt(fs.get('asr'))}, false alarms "
                     f"{_fmt(fr.get('false_alarm_rate'))} vs {_fmt(fs.get('false_alarm_rate'))}")
        elif name == "proposed_v2_floor_only":
            extra = (f"escalation reviews / h {corridor['proposed_v2']['reviews_per_hour']} (right "
                     f"{_fmt(corridor['proposed_v2']['reviews_correct'])}) vs {corridor[name]['reviews_per_hour']}")
        elif name == "proposed_v2_local":
            extra = f"GPT calls / h {ref['llm_calls_per_hour']} vs {s['llm_calls_per_hour']}"
        rows.append(f"| {question} | {name} | {ref['macro_f1']:.3f} → {s['macro_f1']:.3f} | "
                    f"{ref['asr']:.3f} → {s['asr']:.3f} | {ref['ece']:.3f} → {s['ece']:.3f} | "
                    f"{ref['llm_calls_per_hour']} → {s['llm_calls_per_hour']} | {extra} | {_verdict(tests[name])} |")
    return rows


def corridor_table(corridor) -> List[str]:
    from experiments.metrics import BRANCHES

    rows = ["| Configuration | mean w | mean u | corr(w, u) | h median [q25, q75] | "
            + " | ".join(BRANCHES) + " | reviews / h | reviews right |",
            "|" + " -- |" * (6 + len(BRANCHES))]
    for name, c in corridor.items():
        if not c["decisions"] or c["mean_width"] is None:
            continue
        h = c["h"]
        shares = " | ".join(_fmt(c["branches"].get(b)) for b in BRANCHES)
        rows.append(f"| {name} | {_fmt(c['mean_width'])} | {_fmt(c['mean_u'])} | {_fmt(c['width_u_corr'])} | "
                    f"{_fmt(h['median'])} [{_fmt(h['q25'])}, {_fmt(h['q75'])}] | {shares} | "
                    f"{c['reviews_per_hour']} | {_fmt(c['reviews_correct'])} |")
    return rows


def failure_table(failure, summaries) -> List[str]:
    rows = ["| Configuration | ASR [95 % CI] (failure) | ASR (healthy) | False alarms (failure / healthy) | "
            "Dangerous right (failure) | Dangerous under-rated | mean w (failure) |", "|" + " -- |" * 7]
    for name, f in failure.items():
        healthy = summaries.get(name, {})
        rows.append(f"| {name} | {_ci(f['asr'], f['asr_ci'])} | {_fmt(healthy.get('asr'))} | "
                    f"{f['false_alarm_rate']:.3f} / {_fmt(healthy.get('false_alarm_rate'))} | "
                    f"{_fmt(f['dangerous_correct'])} | {_fmt(f['dangerous_under'])} | {_fmt(f['mean_width'])} |")
    return rows


def write_v2(out_dir: str, summaries, results, tests, manifest, corridor, failure, failure_results) -> str:
    manifest = {**manifest, "reference": "proposed_v2",
                "corridor": {n: {k: v for k, v in c.items() if k != "width_u_points"} for n, c in corridor.items()},
                "failure": failure}
    write(out_dir, summaries, results, tests, manifest)
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)                                  # with git / hashes / written_at
    ref = summaries["proposed_v2"]
    tables = {"questions.md": questions_table(summaries, tests, corridor, failure),
              "corridor.md": corridor_table(corridor), "failure.md": failure_table(failure, summaries)}
    ablation = ["| Variant | ΔMacro-F1 | ΔASR | ΔECE | ΔLLM calls / h |", "| -- | -- | -- | -- | -- |",
                f"| proposed_v2 | {ref['macro_f1']:.3f} | {ref['asr']:.3f} | {ref['ece']:.3f} | "
                f"{ref['llm_calls_per_hour']} |"]
    for name in [n for n in summaries if n.startswith("proposed_v2_")]:
        s = summaries[name]
        calls = s["llm_calls_per_hour"] - ref["llm_calls_per_hour"]
        ablation.append(f"| {name} | {_delta(ref['macro_f1'], s['macro_f1'])} | {_delta(ref['asr'], s['asr'])} | "
                        f"{_delta(ref['ece'], s['ece'])} | {calls:+.1f} |")
    tables["ablation.md"] = ablation
    if failure_results:
        with open(os.path.join(out_dir, "per_scenario_failure.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["config", "scenario_id", "category", "expert_level", "final_level", "final_risk",
                             "decisions"])
            for name, rs in failure_results.items():
                for r in rs:
                    writer.writerow([name, r.scenario_id, r.category, r.expert_level, r.final_level, r.final_risk,
                                     "; ".join(f"{d.source}:{d.level_final}:{d.branch}" for d in r.decisions)])
    for name, rows in tables.items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
    _plots_v2(out_dir, summaries, corridor, failure)
    lines = [f"# Experiment v2 — {manifest['dataset']['name']} ({manifest['written_at'][:10]})", ""]
    caveats = []
    if manifest["dataset"].get("synthetic"):
        caveats.append("the dataset is **scripted** (`experiments/scripted.py`) — it validates the method, it is not "
                       "the recorded evaluation set")
    llm = manifest["llm"]
    if llm["backend"] == "sim":
        caveats.append("the LLM is the deterministic **simulated** analyst, not GPT")
    if llm.get("simulated_misses"):
        caveats.append(f"{llm['simulated_misses']} of {llm['calls']} GPT calls were not in the cache and were answered "
                       "by the simulator (offline run)")
    if caveats:
        lines += ["> **Caveats:** " + "; ".join(caveats) + ".", ""]
    split = manifest["split"]
    lines += [f"Evaluation split: {split['evaluation']} scenarios (calibration {split['calibration']}, seed "
              f"{manifest['seed']}); LLM: {llm['backend']} / {llm['model']}, temperature {llm['temperature']}; rule "
              f"base `{str(manifest.get('rules_version', ''))[:12]}`; thresholds `"
              f"{manifest['hashes'].get('config/thresholds.yaml')}`; trust registry `"
              f"{manifest['hashes'].get('config/trust_registry.yaml')}`. Reference for McNemar: proposed_v2.", ""]
    with open(os.path.join(out_dir, "metrics.md"), encoding="utf-8") as f:
        metrics = f.read()
    lines += ["## Department questions", "", *tables["questions.md"], "",
              "## Metrics", "", metrics, "## Ablations (vs proposed_v2)", "", *ablation, "",
              "## Corridor", "",
              "w = y_r − y_l of the corridor behind each decision, u = mean trust uncertainty of its evidence; h = "
              "agreement of GPT's answer with the corridor; branch shares over all decisions; an escalation review "
              "is right when GPT's level does not exceed the expert level at that moment.", "",
              *tables["corridor.md"], "",
              "## Sensor-failure block", "",
              f"Every evaluation scenario replayed with its gas sensor broken: clean-air baseline drift "
              f"+{manifest['failure_block']['drift_pct']} % on every reading and a `baseline_drift` self-test report "
              f"with health {manifest['failure_block']['health']}. Registry trust discounts the sensor (τ·h, wider "
              "corridor); static trust keeps it at 0.95.", "", *tables["failure.md"], ""]
    with open(os.path.join(out_dir, "mcnemar.md"), encoding="utf-8") as f:
        lines += ["## McNemar (per-scenario correctness vs proposed_v2)", "", f.read()]
    calibration = manifest.get("calibration") or {}
    if calibration.get("fitted"):
        cv = calibration.get("cv") or {}
        lines += ["## Calibration of the LLM levels (C)", "",
                  f"{calibration['n']} of {calibration.get('split')} calibration scenarios; leave-one-out Brier "
                  f"{_fmt(cv.get('brier'))} vs fixed mapping {_fmt((cv.get('fixed') or {}).get('brier'))} → "
                  + ("fitted sets used." if calibration.get("used", True) else "fixed mapping used."), ""]
    lines += ["Figures: `asr.png`, `reliability.png`, `width_vs_u.png`, `h_hist.png`, `branches.png`, `failure.png`, "
              "`confusion_<config>.png`; per scenario: `per_scenario.csv`, `per_scenario_failure.csv`; "
              "reproducibility: `manifest.json` (git commit, rule base version, trust registry snapshot, config / "
              "prompt / thresholds hashes, dataset hash) and the LLM cache."]
    with open(os.path.join(out_dir, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_dir


def _plots_v2(out_dir: str, summaries, corridor, failure) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.metrics import BRANCHES

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    for name, colour in (("proposed_v2", "#c44"), ("proposed_v2_type1", "#48c"), ("proposed_v2_static_trust", "#888")):
        points = (corridor.get(name) or {}).get("width_u_points") or []
        if points:
            ax.scatter([p[1] for p in points], [p[0] for p in points], s=10, alpha=0.6, color=colour, label=name)
    ax.set_xlabel("mean trust uncertainty u")
    ax.set_ylabel("corridor width w")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "width_vs_u.png"), dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    histogram = (corridor.get("proposed_v2") or {}).get("h", {}).get("histogram") or [0] * 10
    ax.bar([i / 10 + 0.05 for i in range(10)], histogram, width=0.09, color="#c44")
    ax.set_xlabel("agreement h (proposed_v2)")
    ax.set_ylabel("GPT answers")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "h_hist.png"), dpi=120)
    plt.close(fig)
    names = [n for n, c in corridor.items() if c.get("branches")]
    fig, ax = plt.subplots(figsize=(max(5, len(names) * 0.9), 3.6))
    bottom = [0.0] * len(names)
    for branch in BRANCHES:
        values = [corridor[n]["branches"].get(branch, 0.0) for n in names]
        if any(values):
            ax.bar(range(len(names)), values, bottom=bottom, label=branch)
            bottom = [b + v for b, v in zip(bottom, values, strict=True)]
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("share of decisions")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "branches.png"), dpi=120)
    plt.close(fig)
    if failure:
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        names = list(failure)
        x = range(len(names))
        ax.bar([i - 0.3 for i in x], [summaries[n]["asr"] for n in names], 0.2, label="ASR healthy", color="#bbb")
        ax.bar([i - 0.1 for i in x], [failure[n]["asr"] for n in names], 0.2, label="ASR failure", color="#c44")
        ax.bar([i + 0.1 for i in x], [summaries[n]["false_alarm_rate"] for n in names], 0.2,
               label="false alarms healthy", color="#9bd")
        ax.bar([i + 0.3 for i in x], [failure[n]["false_alarm_rate"] for n in names], 0.2,
               label="false alarms failure", color="#36c")
        ax.set_xticks(list(x), names, fontsize=8)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "failure.png"), dpi=120)
        plt.close(fig)
