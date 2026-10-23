"""
Fit the output membership functions of the LLM risk levels (F18 / TOD-29) and report calibration before / after.

    # labelled scenarios (TOD-31): CSV snapshot_id,expert_risk + the decisions GPT gave for them
    python -m tools.calibrate_llm_risk --labels data/eval/labels.csv --decisions "logs/*/envelope.jsonl"
    # or one CSV with the distribution columns: snapshot_id,expert_risk,p_low,p_medium,p_high,p_critical[,risk_level]
    python -m tools.calibrate_llm_risk --table data/eval/scenarios.csv
    # demo / smoke test on a synthetic overconfident model (never written to config by default)
    python -m tools.calibrate_llm_risk --synthetic 400 --dry-run
    # F28 (TOD-39): the harness's calibration split (36 scripted scenarios), GPT answers from the experiment cache
    python -m tools.calibrate_llm_risk --harness data/eval/scripted [--seed 42]

Scenarios are split deterministically by a hash of snapshot_id (``--eval-share``, default 0.5): the trapezoids are
fitted on the calibration split, Brier / ECE / reliability bins of the fixed level → number mapping and of the
calibrated risk are reported on the evaluation split. The result goes to ``config/calibration.yaml`` (params,
dataset sha256, date, metrics) unless ``--dry-run``; the risk envelope uses it from then on.

F28 (TOD-39): the sets are constrained (monotone centres, supports ≥ ``--min-width``) and every fit also reports
leave-one-out Brier / ECE (``cv``) and the constraint check next to the in-sample metrics. ``--harness`` fits on the
whole calibration split of the evaluation set (``experiments.dataset.Dataset.split``, same seed / share as
``experiments.run``) with LOO as the evaluation; GPT answers are read from ``experiments/cache/llm_<backend>.sqlite``
(read-only, misses answered by the simulator and left out; answers the corridor rejected as manipulated —
``deescalation_rejected`` — left out too) — fewer than 10 recorded answers → nothing is written.
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

from modules.llm.calibration import (LEVELS, MIN_WIDTH, check_constraints, dataset_hash, evaluate, fit,  # noqa: E402
                                     loo_cv, synthetic)

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


def from_harness(dataset_path: str, seed: int, share: float, backend: str, fuzzy: str) -> Tuple[List[Row], int]:
    """:return: (rows with recorded GPT answers of the calibration split, number of scenarios answered by the sim)"""
    from experiments.configs import PROPOSED
    from experiments.dataset import Dataset
    from experiments.pipeline import run_scenario
    from tools.fit_thresholds import OfflineLLM

    calibration, _ = Dataset.load(dataset_path).split(seed, share)
    llm = OfflineLLM(backend, f"experiments/cache/llm_{backend}.sqlite")
    rows, simulated = [], 0
    try:
        for scenario in calibration:
            misses, trace = llm.misses, {}
            result = run_scenario(scenario, PROPOSED, llm, fuzzy, None, trace=trace)
            if llm.misses != misses:
                simulated += 1
            elif (trace.get("decision") or [{}])[-1].get("branch") == "deescalation_rejected":
                continue                                  # a talked-down answer is not GPT's honest verbal risk
            elif result.distributions:
                last = result.decisions[-1] if result.decisions else None
                rows.append((scenario.scenario_id, result.distributions[-1], scenario.expert_risk,
                             last.level_llm if last else None))
    finally:
        llm.close()
    return rows, simulated


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the LLM's verbal risk with fuzzy output sets")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--table", help="CSV with snapshot_id, expert_risk, p_low … p_critical")
    source.add_argument("--labels", help="CSV snapshot_id, expert_risk (with --decisions)")
    source.add_argument("--synthetic", type=int, metavar="N", help="synthetic overconfident model (demo)")
    source.add_argument("--harness", metavar="DATASET", help="calibration split of an evaluation set (F28)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-share", type=float, default=0.3)
    parser.add_argument("--llm", choices=("azure", "sim"), default="azure", help="whose cached answers (--harness)")
    parser.add_argument("--fuzzy", default="config/fuzzy.yaml")
    parser.add_argument("--min-width", type=float, default=MIN_WIDTH, help="minimum support of every level set")
    parser.add_argument("--decisions", default="logs/*/envelope.jsonl", help="glob of JSONL logs with llm_decision")
    parser.add_argument("--eval-share", type=float, default=0.5)
    parser.add_argument("--out", default="config/calibration.yaml")
    parser.add_argument("--report", help="also write the full report (incl. reliability bins) as JSON")
    parser.add_argument("--dry-run", action="store_true", help="report only, do not write the calibration")
    args = parser.parse_args(argv)

    if args.harness:
        rows, simulated = from_harness(args.harness, args.seed, args.calibration_share, args.llm, args.fuzzy)
        return fit_harness(args, rows, simulated)
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
    calibrator, fit_metrics = fit([r[1] for r in calibration], [r[2] for r in calibration], min_width=args.min_width)
    cv = loo_cv([r[1] for r in calibration], [r[2] for r in calibration], min_width=args.min_width)
    constraints = check_constraints(calibrator.params, args.min_width)
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
    print(f"leave-one-out on the calibration split: Brier {cv['brier']:.4f} ECE {cv['ece']:.4f}; constraints "
          f"{'hold' if constraints['ok'] else 'VIOLATED'}")
    extra = {"dataset": name, "dataset_sha256": dataset_hash([(r[0], r[1], r[2]) for r in rows]),
             "n_calibration": len(calibration), "n_evaluation": len(evaluation), "fit": fit_metrics,
             "cv": {k: v for k, v in cv.items() if k != "predictions"}, "constraints": constraints,
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


def fit_harness(args, rows: List[Row], simulated: int) -> int:
    """F28: fit on the whole calibration split, judged by leave-one-out (no further split of 36 scenarios)"""
    print(f"harness {args.harness}: {len(rows)} calibration scenarios with recorded GPT answers, {simulated} "
          f"answered by the simulator (left out)")
    if len(rows) < 10:
        print("Fewer than 10 recorded answers: refresh the cache with python -m experiments.run --llm azure first; "
              "nothing written", file=sys.stderr)
        return 1
    distributions, targets = [r[1] for r in rows], [r[2] for r in rows]
    calibrator, fit_metrics = fit(distributions, targets, min_width=args.min_width)
    cv = loo_cv(distributions, targets, min_width=args.min_width)
    constraints = check_constraints(calibrator.params, args.min_width)
    print(f"{'':16}{'Brier':>9}{'ECE':>9}")
    print(f"{'in-sample':16}{fit_metrics['brier']:>9.4f}{fit_metrics['ece']:>9.4f}")
    print(f"{'leave-one-out':16}{cv['brier']:>9.4f}{cv['ece']:>9.4f}")
    print(f"{'fixed mapping':16}{cv['fixed']['brier']:>9.4f}{cv['fixed']['ece']:>9.4f}")
    print(f"constraints {'hold' if constraints['ok'] else 'VIOLATED'}: centres {constraints['centres']}, "
          f"min support {constraints['min_support']}")
    extra = {"dataset": args.harness, "dataset_sha256": dataset_hash([(r[0], r[1], r[2]) for r in rows]),
             "split": {"seed": args.seed, "calibration_share": args.calibration_share}, "n_calibration": len(rows),
             "simulated_left_out": simulated, "fit": fit_metrics,
             "cv": {k: v for k, v in cv.items() if k != "predictions"}, "constraints": constraints}
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({**extra, "params": calibrator.params, "loo_predictions": cv["predictions"]}, f,
                      ensure_ascii=False, indent=1)
    worse = cv["brier"] > cv["fixed"]["brier"]
    if args.dry_run or not constraints["ok"] or worse:
        print("dry run: calibration not written" if args.dry_run else "constraints violated: not written"
              if not constraints["ok"] else "leave-one-out Brier worse than the fixed mapping: not written (the "
              "envelope keeps the fixed level mapping)")
        return 0 if args.dry_run else 1
    calibrator.to_yaml(args.out, extra)
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
