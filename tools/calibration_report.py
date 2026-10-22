"""
Calibration report for the thesis (F22 / TOD-33): reliability diagrams + tables from the committed measured inputs
(``config/calibration/*.yaml``, ``config/trust_priors.yaml``, ``config/thresholds.yaml``)
→ ``reports/calibration_<date>/``.

    python -m tools.calibration_report          # → reports/calibration_<date>/calibration.md + *.png
"""
import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):  # allow `python tools/calibration_report.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration import map_from_dict, read_yaml  # noqa: E402

CHANNELS = ("asr", "speaker", "face", "gas")


def reliability_png(channel: str, data: Dict[str, Any], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.8))
    mapping = map_from_dict(data["map"])
    xs = np.linspace(0, 1, 201)
    right.plot(xs, mapping.many(xs), color="#1f6feb")
    right.set(xlabel=f"raw score: {data['raw_score']}"[:70], ylabel="calibrated probability",
              title="fitted map", xlim=(0, 1), ylim=(0, 1.02))
    rows = data.get("reliability") or []
    if rows:
        left.plot([0, 1], [0, 1], "--", color="#999")
        left.bar([(r["lo"] + r["hi"]) / 2 for r in rows], [r["observed"] for r in rows], width=0.09,
                 color="#9ecae1", label="observed frequency")
        left.plot([r["mean_p"] for r in rows], [r["observed"] for r in rows], "o-", color="#d62728",
                  label="mean predicted → observed")
        for r in rows:
            left.annotate(str(r["n"]), ((r["lo"] + r["hi"]) / 2, r["observed"]), ha="center", va="bottom", fontsize=7)
        left.legend(fontsize=7, loc="upper left")
        left.set(title="reliability (held-out / out-of-fold)")
    else:                                               # face: Bayes bins, no raw scores
        bins = data.get("bins") or []
        left.step([b["lo"] for b in bins] + [1.0], [b["posterior"] for b in bins] + [bins[-1]["posterior"]],
                  where="post", color="#d62728")
        left.set(title=f"P(known | bin), prior {data.get('prior')}")
    left.set(xlabel="predicted probability" if rows else "similarity", ylabel="observed / posterior",
             xlim=(0, 1), ylim=(0, 1.05))
    fig.suptitle(f"{channel}: {data['meaning']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def table(rows: List[List[Any]], head: List[str]) -> str:
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("--" for _ in head) + " |"]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in rows]
    return "\n".join(lines)


def build(cal_dir: str, priors_path: str, thresholds_path: str, out: str) -> str:
    os.makedirs(out, exist_ok=True)
    tools = ("`python -m tools.calibrate_channels …`, `python -m tools.trust_priors`, "
             "`python -m tools.fit_thresholds`, then `python -m tools.calibration_report`")
    head = [f"# Measured trust inputs — {datetime.now():%Y-%m-%d}", "",
            "F22 / TOD-33. Every trust degree and decision threshold of `config/fuzzy.yaml` refers to one of the "
            f"files below (`@trust_priors:…`, `@thresholds:…`, `trust.calibration`). Regenerate with {tools}.", "",
            "## 1. Channel score calibration", ""]
    parts: List[str] = []
    summary = []
    for channel in CHANNELS:
        data = read_yaml(os.path.join(cal_dir, f"{channel}.yaml"))
        if not data:
            continue
        png = f"{channel}_reliability.png"
        reliability_png(channel, data, os.path.join(out, png))
        metrics = data.get("metrics") or {}
        held = metrics.get("heldout") or metrics.get("out_of_fold") or {}
        summary.append([channel, data["raw_score"], data["meaning"], data["method"], data["n"],
                        held.get("brier", "—"), held.get("ece", "—"), held.get("eer", "—"), held.get("cllr", "—"),
                        data["dataset"]["sha256"][:12]])
        parts += [f"### {channel}", "", f"![{channel}]({png})", "",
                  "Raw score → calibrated probability: " + ", ".join(f"{k} → {v:.2f}" for k, v in data["table"].items())
                  + ".", ""]
        if channel == "asr":
            snr = metrics.get("mean_wer_by_snr_db", {})
            parts += [f"This confidence means: of the phrases at confidence 0.5, {data['table']['0.5']:.0%} had WER ≤ "
                      f"{data['wer_max']:.0%}; at 0.8 — {data['table']['0.8']:.0%}. Mean WER by SNR (dB): "
                      + ", ".join(f"{k}: {v:.2f}" for k, v in snr.items()) + ".", ""]
        if channel == "speaker":
            at = data["at_voice_threshold"]
            parts += [f"Held-out pairs: EER {held['eer']:.1%} at cosine {held['eer_threshold']}, "
                      f"Cllr {held['cllr']:.3f} bits. At `identity.voice_threshold` {at['threshold']}: "
                      f"known speakers right "
                      f"{at['known_correct']}/{at['known_correct'] + at['known_wrong']}, strangers rejected "
                      f"{at['stranger_correct']}/{at['stranger_correct'] + at['stranger_wrong']}.", ""]
        if data["dataset"].get("note"):
            parts += [f"Data: {data['dataset']['note']}.", ""]
    columns = ["channel", "raw score", "meaning", "method", "n", "Brier", "ECE", "EER", "Cllr", "data sha256"]
    parts = head + [table(summary, columns), ""] + parts
    priors = read_yaml(priors_path) or {}
    rows = [[f"{device}.{name}", e["prior_status"], e["r0"], e["s0"], e["tau0"], e["u0"], e["wilson_lb"],
             e["explanation"]]
            for device, entries in (priors.get("channels") or {}).items() for name, e in entries.items()]
    counts = {s: sum(r[1] == s for r in rows) for s in ("measured", "unmeasured")}
    parts += ["## 2. Channel priors per device × channel (TOD-34)", "",
              f"{counts['measured']} measured, {counts['unmeasured']} unmeasured (r0 = s0 = 0 until labelled data "
              "exists).", "",
              table(rows, ["channel", "prior status", "r0", "s0", "τ0", "u0", "Wilson 95 % LB", "source"]), "",
              "Rules and interim values (`policy`, `measured: false`):", "",
              table([[n, p["value"], p["explanation"]] for n, p in (priors.get("policy") or {}).items()],
                    ["name", "value", "why"]), ""]
    th = read_yaml(thresholds_path) or {}
    if th.get("values"):
        values, fits = th["values"], th.get("fits") or {}

        def row(name, key):
            fit = fits.get(key) or {}
            return [name, fit.get("threshold", "—"), fit.get("expected_cost", "—"), fit.get("empirical_p_miss", "—"),
                    fit.get("empirical_p_fa", "—"), f"{fit.get('n_pos', '—')} / {fit.get('n_neg', '—')}",
                    "kept: " + fit["kept_reason"] if "kept_reason" in fit else "fitted"]

        ds = th["dataset"]
        kind = "synthetic" if ds.get("synthetic") else "recorded"
        parts += ["## 3. Cost-based thresholds", "",
                  f"Cost ratio C_miss / C_fa = **{th['cost_ratio']}** (the only expert input, "
                  f"`calibration.cost_ratio`); dataset {ds['name']} ({kind}), calibration split, {ds['scenarios']} "
                  f"scenarios, hash `{ds['hash'][:12]}`.", "",
                  table([row("envelope.theta", "theta"), row("level medium", "level_medium"),
                         row("level high", "level_high"), row("level critical", "level_critical"),
                         row("gating.threshold", "gating")],
                        ["threshold", "value", "expected cost", "P_miss", "P_fa", "pos / neg", "status"]), "",
                  f"`gating.critical_floor` = level critical = {values['critical_floor']}. Not applicable yet: "
                  + "; ".join(f"{k} — {v}" for k, v in (th.get("not_applicable") or {}).items()) + ".", ""]
    path = os.path.join(out, "calibration.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Calibration report (reliability diagrams + tables)")
    parser.add_argument("--calibration-dir", default="config/calibration")
    parser.add_argument("--priors", default="config/trust_priors.yaml")
    parser.add_argument("--thresholds", default="config/thresholds.yaml")
    parser.add_argument("--out", help="default reports/calibration_<date>")
    args = parser.parse_args(argv)
    out = args.out or os.path.join("reports", f"calibration_{datetime.now():%Y-%m-%d}")
    print(f"→ {build(args.calibration_dir, args.priors, args.thresholds, out)}")


if __name__ == "__main__":
    main()
