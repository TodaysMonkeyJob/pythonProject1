"""README.md of the A6 (TOD-46) Article 1 evaluation (``python -m experiments.hfs.run``)."""
import os
from typing import Any, Dict, List

LEVEL_ORDER = ("low", "medium", "high", "critical")


def pct(value: float) -> str:
    return f"{value:.1%}"


def reading(scenarios: List[Dict[str, Any]], results: Dict[str, Any], summary: Dict[str, Any],
            main: str) -> List[str]:
    """The interpretation, with the numbers of this run."""
    lows = [s for s in scenarios if s["label"] == "low"]
    gauss = "gauss_in_type1"
    gauss_low_ok = sum(results[gauss][s["id"]]["level"] == "low" for s in lows) if gauss in results else 0
    gauss_low_risk = max((results[gauss][s["id"]]["risk"] for s in lows), default=0.0) if gauss in results else 0
    crit = [s for s in scenarios if s["label"] == "critical"]
    gauss_crit = [results[gauss][s["id"]]["risk"] for s in crit] if gauss in results else []
    speech = [s for s in scenarios if (s.get("audio") or {}).get("speech")]
    quiet = [s for s in scenarios if not (s.get("audio") or {}).get("speech")]

    def width(items):
        return sum(results[main][s["id"]]["interval"][1] - results[main][s["id"]]["interval"][0]
                   for s in items) / max(1, len(items))

    lines = ["", "### Reading the results", "",
             "* **Agreement is high by construction.** The labels and the rule tables come from the same "
             "requirements (the supervisor's examples), so the deployed system agrees with "
             f"{summary[main]['accuracy']:.0%} of them; the disagreements are the informative part — e.g. a visitor "
             "with a known person (1 of 2 unknown = 0.5) lies between the terms part and most and reads medium. The "
             "labels need an independent expert pass."]
    if gauss_crit:
        lines.append(
            f"* **Gaussian membership functions break the low end.** A Gaussian never reaches 0, so in a quiet scene "
            f"every rule still fires a little and the centroid is pulled up: only {gauss_low_ok} of {len(lows)} "
            f"low scenarios stay low (risk up to {gauss_low_risk:.2f}), and the critical ones are compressed "
            f"({min(gauss_crit):.2f}–{max(gauss_crit):.2f} instead of 0.87). The triangular terms on a Ruspini "
            "partition are 0 outside their support — that is what keeps the table cells separate and the surface "
            "monotonic (table below).")
    lines.append(f"* **The corridor is widest when somebody speaks**: mean width {width(speech):.2f} with speech vs "
                 f"{width(quiet):.2f} without — the danger-word accuracy (δ = 0.465, 23 noisy synthetic samples) is "
                 "the least measured input; recording real danger words narrows it most.")
    return lines


def write_report(out: str, manifest: Dict[str, Any], scenarios: List[Dict[str, Any]], results: Dict[str, Any],
                 summary: Dict[str, Any], sweep: Dict[float, Any], latency: Dict[str, Any], mono: Dict[str, Any],
                 bench: Dict[str, Any], configs: Dict[str, Any], main: str) -> None:
    lines = [f"# Article 1 evaluation — autonomous hierarchical fuzzy system ({manifest['date'][:10]})", "",
             f"`python -m experiments.hfs.run --seed {manifest['seed']}` at commit `{manifest['git']}`: "
             f"{manifest['scenarios']} scenarios (`experiments/hfs/scenarios.yaml`) × {len(configs)} configurations, "
             "replayed through the real `HierarchicalFuzzySystem` (60 s of events per scenario, seeded noise).", "",
             f"**Labels: {manifest['labels']}.** They were written from the requirements before running; the "
             "numbers below measure agreement with these draft labels, not ground truth.", "",
             "## Overall level vs the expert labels", "",
             "| Configuration | Macro-F1 | Accuracy | Brier | ECE | Corridor width (mean) "
             "| Corridor reaches the label |",
             "| -- | -- | -- | -- | -- | -- | -- |"]
    for name, s in summary.items():
        mark = " (deployed)" if name == main else ""
        lines.append(f"| `{name}`{mark} | {s['macro_f1']:.3f} | {pct(s['accuracy'])} | {s['brier']:.3f} | "
                     f"{s['ece']:.3f} | {s['corridor_width_mean']:.3f} | {pct(s['corridor_coverage'])} |")
    lines += ["", "Macro-F1 by group (deployed configuration): " +
              ", ".join(f"{g} {v:.3f}" for g, v in summary[main]["macro_f1_by_group"].items()) + ".", ""]
    misses = summary[main]["misses"]
    lines += ["### Disagreements of the deployed configuration with the labels", ""]
    if misses:
        lines += ["| Scenario | Label | System | Why the label (scenarios.yaml) |", "| -- | -- | -- | -- |"]
        why = {s["id"]: s["why"] for s in scenarios}
        lines += [f"| {m['id']} | {m['label']} | {m['got']} | {why[m['id']]} |" for m in misses]
    else:
        lines.append("None.")
    lines += ["", "Confusion (rows = label, columns = system):", "",
              "| label \\ system | " + " | ".join(LEVEL_ORDER) + " |", "| -- " * 5 + "|"]
    for t in LEVEL_ORDER:
        lines.append(f"| {t} | " + " | ".join(str(summary[main]["confusion"][t][p]) for p in LEVEL_ORDER) + " |")
    lines += reading(scenarios, results, summary, main)
    lines += ["", "## Type-1 vs interval type-2", "",
              "The crisp risk is the same (δ = 0 reproduces type-1 exactly); type-2 adds the corridor [y_l, y_r] from "
              "the measured input errors. Corridor width vs the errors (δ × scale, deployed rule bases):", "",
              "| δ scale | Mean corridor width | Corridor reaches the label | Macro-F1 |", "| -- | -- | -- | -- |"]
    for scale, s in sweep.items():
        lines.append(f"| ×{scale:g} | {s['corridor_width_mean']:.3f} | {pct(s['corridor_coverage'])} | "
                     f"{s['macro_f1']:.3f} |")
    lines += ["", "## Triangular vs Gaussian membership functions (slide 8, item 1)", "",
              "Monotonicity measure = share of grid steps where the risk decreases while a risk-raising input "
              "increases (every input of these rule bases raises the risk), over a grid of all other inputs.", "",
              "| Level | Triangular | Gaussian inputs | Gaussian inputs + risk terms |", "| -- | -- | -- | -- |"]

    def cell(m):
        return f"{m['falls']} / {m['steps']} ({pct(m['share'])}), max fall {m['max_fall']:.4f}, " \
               f"{m['level_drops']} level drops"

    for level, m in mono.items():
        lines.append(f"| {level} | {cell(m['triangular'])} | {cell(m['gaussian_inputs'])} | "
                     f"{cell(m['gaussian_all'])} |")
    if mono:
        lines += ["", "Surfaces: `surfaces/<level>_<triangular|gaussian_inputs|gaussian_all>_<x>__<y>.png` (the "
                  "other inputs at the middle of their universe)."]
    lines += ["", "## With vs without the mixture index", "",
              f"Macro-F1 {summary['tri_it2']['macro_f1']:.3f} with, {summary['tri_it2_no_mixture']['macro_f1']:.3f} "
              "without. Scenarios that change: " +
              (", ".join(f"{s['id']} ({results['tri_it2'][s['id']]['level']} → "
                         f"{results['tri_it2_no_mixture'][s['id']]['level']}, label {s['label']})"
                         for s in scenarios
                         if results['tri_it2'][s['id']]['level'] != results['tri_it2_no_mixture'][s['id']]['level'])
               or "none") + ".", "",
              "## Latency (this machine — not the Raspberry Pi)", "",
              "| Level | Type-1 mean / p95, ms | IT2 mean / p95, ms |", "| -- | -- | -- |"]
    for level, data in latency.items():
        lines.append(f"| {level} | {data['tri_type1']['mean']} / {data['tri_type1']['p95']} | "
                     f"{data['tri_it2']['mean']} / {data['tri_it2']['p95']} |")
    lines += ["", "**The Raspberry Pi has not been measured** (no Pi was available): run "
              "`python -m experiments.hfs.run --no-surfaces`, `python -m tools.camera_bench` and "
              "`python -m tools.audio_bench` on it."]
    actions, motion = bench.get("actions"), bench.get("motion")
    if actions and "error" not in actions:
        lines += ["", "## Camera: light vs actions mode, false motion (reports/camera_bench/)", "",
                  f"Actions process: {actions['rss_light_mb']} MB in light mode vs {actions['rss_actions_mb']} MB with "
                  f"the pose / action models ({actions['pose_ms_mean']} ms pose per frame), back to "
                  f"{actions['rss_light_mb']} MB after the planned restart. The camera mode does not enter the fuzzy "
                  "rules (load only)."]
    if motion:
        rows = [f"{kind} {d['before']['false_motion_rate']:.1%} → {d['after']['false_motion_rate']:.1%}"
                for kind, d in motion.get("clips", {}).items()]
        lines.append(f"False motion rate, original → adaptive detector (synthetic clips): {'; '.join(rows)}.")
    lines += ["", "## Per scenario", "", "| Scenario | Label | " + " | ".join(configs) + " |",
              "| -- " * (len(configs) + 2) + "|"]
    for s in scenarios:
        cells = []
        for name in configs:
            r = results[name][s["id"]]
            low, high = r["interval"]
            corridor = f" [{low:.2f}–{high:.2f}]" if high > low else ""
            mark = "" if r["level"] == s["label"] else " ✗"
            cells.append(f"{r['level']} {r['risk']:.2f}{corridor}{mark}")
        lines.append(f"| {s['id']} | {s['label']} | " + " | ".join(cells) + " |")
    lines += ["", "Files: metrics.json (everything above), scenarios.csv, variants/ (the rule bases of every "
              "configuration), surfaces/."]
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
