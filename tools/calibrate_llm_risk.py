"""
Fit the output membership functions of the LLM risk levels (F18 / TOD-29) and report calibration before / after.

    # labelled scenarios (TOD-31): CSV snapshot_id,expert_risk + the decisions GPT gave for them
    python -m tools.calibrate_llm_risk --labels data/eval/labels.csv --decisions "logs/*/envelope.jsonl"
    # or one CSV with the distribution columns: snapshot_id,expert_risk,p_low,p_medium,p_high,p_critical[,risk_level]
    python -m tools.calibrate_llm_risk --table data/eval/scenarios.csv
    # demo / smoke test on a synthetic overconfident model (never written to config by default)
    python -m tools.calibrate_llm_risk --synthetic 400 --dry-run

Scenarios are split deterministically by a hash of snapshot_id (``--eval-share``, default 0.5): the trapezoids are
fitted on the calibration split, Brier / ECE / reliability bins of the fixed level → number mapping and of the
calibrated risk are reported on the evaluation split. The result goes to ``config/calibration.yaml`` (params,
dataset sha256, date, metrics) unless ``--dry-run``; the risk envelope uses it from then on.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

if __package__ in (None, ""):  # allow `python tools/calibrate_llm_risk.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm.calibration import LEVELS, dataset_hash, evaluate, fit, synthetic  # noqa: E402

Row = Tuple[str, Dict[str, float], float, Optional[str]]     # snapshot id, distribution, expert risk, chosen level


def from_table(path: str) -> List[Row]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dist = {level: float(r.get(f"p_{level}") or 0.0) for level in LEVELS}
            rows.append((r["snapshot_id"], dist, float(r["expert_risk"]), r.get("risk_level") or None))
    return rows


def from_logs(labels_path: str, pattern: str) -> List[Row]:
    with open(labels_path, encoding="utf-8") as f:
        labels = {r["snapshot_id"]: float(r["expert_risk"]) for r in csv.DictReader(f)}
    rows: Dict[str, Row] = {}
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                event = json.loads(line)
                p = event.get("payload", {})
                sid = str(p.get("snapshot_id"))
                if event.get("type") != "llm_decision" or sid not in labels:
                    continue
                level = p.get("risk_level")
                dist = p.get("risk_distribution") or {name: float(name == level) for name in LEVELS}
                rows[sid] = (sid, dist, labels[sid], level)           # the latest decision per snapshot wins
    return list(rows.values())


def split(rows: List[Row], eval_share: float) -> Tuple[List[Row], List[Row]]:
    def bucket(sid: str) -> float:
        return int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return [r for r in rows if bucket(r[0]) >= eval_share], [r for r in rows if bucket(r[0]) < eval_share]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the LLM's verbal risk with fuzzy output sets")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--table", help="CSV with snapshot_id, expert_risk, p_low … p_critical")
    source.add_argument("--labels", help="CSV snapshot_id, expert_risk (with --decisions)")
    source.add_argument("--synthetic", type=int, metavar="N", help="synthetic overconfident model (demo)")
    parser.add_argument("--decisions", default="logs/*/envelope.jsonl", help="glob of JSONL logs with llm_decision")
    parser.add_argument("--eval-share", type=float, default=0.5)
    parser.add_argument("--out", default="config/calibration.yaml")
    parser.add_argument("--report", help="also write the full report (incl. reliability bins) as JSON")
    parser.add_argument("--dry-run", action="store_true", help="report only, do not write the calibration")
    args = parser.parse_args(argv)

    if args.table:
        rows, name = from_table(args.table), args.table
    elif args.labels:
        rows, name = from_logs(args.labels, args.decisions), f"{args.labels} + {args.decisions}"
    else:
        rows, name = [(sid, d, y, None) for sid, d, y in synthetic(args.synthetic)], f"synthetic({args.synthetic})"
    calibration, evaluation = split(rows, args.eval_share)
    if len(calibration) < 10 or len(evaluation) < 10:
        print(f"Not enough labelled scenarios ({len(calibration)} calibration / {len(evaluation)} evaluation; need "
              f"≥ 10 each)", file=sys.stderr)
        return 1
    calibrator, fit_metrics = fit([r[1] for r in calibration], [r[2] for r in calibration])
    report = evaluate(calibrator, [r[1] for r in evaluation], [r[2] for r in evaluation], [r[3] for r in evaluation])
    print(f"dataset {name}: {len(rows)} scenarios ({len(calibration)} calibration / {len(evaluation)} evaluation)")
    print(f"{'':12}{'Brier':>9}{'ECE':>9}")
    for key in ("fixed", "calibrated"):
        print(f"{key:12}{report[key]['brier']:>9.4f}{report[key]['ece']:>9.4f}")
    print("reliability (evaluation split, calibrated): " + "; ".join(
        f"{b['bin'][0]:.1f}-{b['bin'][1]:.1f} n={b['n']} pred {b['mean_pred']:.2f} true {b['mean_target']:.2f}"
        for b in report["calibrated"]["reliability"]))
    print("fitted trapezoids: " + "; ".join(f"{level} [{', '.join(f'{v:.2f}' for v in p)}]"
                                            for level, p in calibrator.params.items()))
    extra = {"dataset": name, "dataset_sha256": dataset_hash([(r[0], r[1], r[2]) for r in rows]),
             "n_calibration": len(calibration), "n_evaluation": len(evaluation), "fit": fit_metrics,
             "evaluation": {k: {m: report[k][m] for m in ("brier", "ece")} for k in ("fixed", "calibrated")}}
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({**extra, "report": report, "params": calibrator.params}, f, ensure_ascii=False, indent=1)
    if args.dry_run:
        print("dry run: calibration not written")
    else:
        calibrator.to_yaml(args.out, extra)
        print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
