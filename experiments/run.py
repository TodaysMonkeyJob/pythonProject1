"""
Run the evaluation (F20 / TOD-31): all configurations + ablations on a dataset, metrics, statistics, report.

    python -m experiments.run --all --seed 42                                  # data/eval/recorded, GPT (cached)
    python -m experiments.run --all --seed 42 --scripted --llm sim             # generated set + simulated LLM
    python -m experiments.run --configs proposed llm_only --dataset data/eval/recorded

    python -m experiments.run --suite v2 --seed 42 --scripted --llm azure     # F29 (TOD-40): experiment v2
    python -m experiments.run --suite v2 --seed 42 --scripted --llm azure --offline   # cache read-only, no network

Steps: load the manifest → stratified calibration / evaluation split (seed) → fit the LLM risk calibration (C) on
the calibration split with the proposed pipeline → replay every evaluation scenario through every configuration
(``experiments.pipeline``; GPT answers cached in ``experiments/cache/llm_<backend>.sqlite``, so a re-run gives the
same numbers without network) → metrics + McNemar → ``reports/experiment_<date>/``.

``--suite v2`` (F29): the new design ``proposed_v2`` and its variants (``configs.V2``), McNemar vs ``proposed_v2``,
corridor metrics (width vs u, agreement h, branch shares, escalation reviews), the sensor-failure block (every
evaluation scenario replayed with a broken gas sensor, ``experiments.variants.with_gas_failure``, registry vs static
trust) → ``reports/experiment_v2_<date>/`` with the department-question table. ``--offline`` reads the GPT cache
read-only (misses answered by the simulator and counted in the manifest — refresh them with a normal run).
"""
import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List

from experiments.configs import ALL, BY_NAME, FAILURE, MAIN, PROPOSED, V2
from experiments.dataset import Dataset, annotator_kappa
from experiments.llm import CachedLLM
from experiments.metrics import mcnemar, summarize
from experiments.pipeline import run_scenario
from experiments.report import write


def calibration_rows(scenarios, llm, fuzzy: str, min_interval: float):
    """
    C: (last LLM distribution, expert risk, scenario id) per scenario of the calibration split — F28 (TOD-39): every
    scenario of the split (36), not only the 21 non-adversarial ones chosen by their label. Left out are only answers
    the system itself rejected as manipulated: the corridor's ``deescalation_rejected`` branch (GPT below the trusted
    lower bound L) — a talked-down "low" is not GPT's honest verbal risk and would teach "low" to mean "high".

    :return: (rows, number of answers left out as rejected de-escalations)
    """
    rows, rejected = [], 0
    for scenario in scenarios:
        trace: dict = {}
        result = run_scenario(scenario, PROPOSED, llm, fuzzy, None, min_interval, trace=trace)
        if not result.distributions:
            continue
        last = (trace.get("decision") or [{}])[-1]
        if last.get("branch") == "deescalation_rejected":
            rejected += 1
            continue
        rows.append((result.distributions[-1], scenario.expert_risk, scenario.scenario_id))
    return rows, rejected


def fit_calibration(scenarios, llm, fuzzy: str, min_interval: float, cv: bool = True):
    """:return: (calibrator or None, info: n, in-sample fit, LOO CV Brier / ECE, constraint check, params)"""
    from modules.llm.calibration import check_constraints, fit, loo_cv

    rows, rejected = calibration_rows(scenarios, llm, fuzzy, min_interval)
    if len(rows) < 10:
        return None, {"fitted": False, "n": len(rows), "rejected_deescalations": rejected,
                      "note": "fewer than 10 calibration scenarios with an LLM answer"}
    distributions, targets = [r[0] for r in rows], [r[1] for r in rows]
    calibrator, metrics = fit(distributions, targets)
    info = {"fitted": True, "n": len(rows), "rejected_deescalations": rejected, "split": len(scenarios), "fit": metrics,
            "constraints": check_constraints(calibrator.params),
            "params": calibrator.params}
    if cv:
        info["cv"] = {k: v for k, v in loo_cv(distributions, targets).items() if k != "predictions"}
        if info["cv"]["brier"] > info["cv"]["fixed"]["brier"]:      # decided by data: the fit must beat the fixed map
            info["used"] = False
            info["note"] = "leave-one-out Brier worse than the fixed level mapping: the fixed mapping is used"
            return None, info
    info["used"] = True
    return calibrator, info


def rules_version(fuzzy: str) -> str:
    """F26: sha256 of the normalised rule set the run used (core.fuzzy.governance)"""
    from core.fuzzy import load_yaml
    from core.fuzzy.governance import version_hash

    return version_hash(load_yaml(fuzzy))


def registry_snapshot(fuzzy: str) -> Dict:
    """F23: the trust registry setup every replay starts from (channels, measured priors, γ, health factors)"""
    from core.fuzzy import load_yaml
    from modules.trust.registry import TrustRegistry

    path = os.path.join(os.path.dirname(os.path.abspath(fuzzy)),
                        (load_yaml(fuzzy).get("trust") or {}).get("registry", "trust_registry.yaml"))
    registry = TrustRegistry(load_yaml(path))
    try:
        return {"config": os.path.relpath(path), **registry.config_snapshot()}
    finally:
        registry.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the fuzzy-bounded LLM pipeline against baselines")
    parser.add_argument("--dataset", default="data/eval/recorded", help="folder with manifest.json")
    parser.add_argument("--scripted", action="store_true", help="use (and generate if missing) data/eval/scripted")
    parser.add_argument("--all", action="store_true", help="4 configurations + 4 ablations")
    parser.add_argument("--suite", choices=("v1", "v2"), default="v1", help="v2 = experiment v2 (F29 / TOD-40)")
    parser.add_argument("--offline", action="store_true", help="GPT cache read-only, misses → simulator")
    parser.add_argument("--configs", nargs="*", help=f"subset of {list(BY_NAME)}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-share", type=float, default=0.3)
    parser.add_argument("--llm", choices=("azure", "sim"), default="azure")
    parser.add_argument("--cache", help="LLM cache (default experiments/cache/llm_<backend>.sqlite)")
    parser.add_argument("--fuzzy", default="config/fuzzy.yaml")
    parser.add_argument("--min-interval", type=float, default=300.0, help="llm.min_llm_interval_sec")
    parser.add_argument("--out", help="report folder (default reports/experiment_<date>)")
    args = parser.parse_args(argv)

    dataset_path = "data/eval/scripted" if args.scripted else args.dataset
    if args.scripted and not os.path.isfile(os.path.join(dataset_path, "manifest.json")):
        from experiments.scripted import build

        build(dataset_path, args.seed)
    if not os.path.isfile(os.path.join(dataset_path, "manifest.json")):
        print(f"No dataset at {dataset_path} (record scenarios with python -m experiments.record, or --scripted)",
              file=sys.stderr)
        return 1
    dataset = Dataset.load(dataset_path)
    calibration, evaluation = dataset.split(args.seed, args.calibration_share)
    v2 = args.suite == "v2"
    configs = V2 if v2 and not args.configs else ALL if args.all else \
        [BY_NAME[n] for n in args.configs] if args.configs else MAIN
    config = {}
    if args.llm == "azure":
        from core.config import load_config

        config = load_config("config.yaml")
    cache = args.cache or f"experiments/cache/llm_{args.llm}.sqlite"
    if args.offline:
        from tools.fit_thresholds import OfflineLLM

        llm = OfflineLLM(args.llm, cache)
    else:
        llm = CachedLLM(args.llm, cache, config)
    prices = {"in": float(config.get("llm", {}).get("price_per_1k_input", 0.15)),
              "out": float(config.get("llm", {}).get("price_per_1k_output", 0.6))}
    print(f"{dataset.name}: {len(dataset.scenarios)} scenarios ({len(calibration)} calibration / {len(evaluation)} "
          f"evaluation), {len(configs)} configurations, LLM {args.llm}")
    calibrator, calibration_info = fit_calibration(calibration, llm, args.fuzzy, args.min_interval) \
        if any(c.view for c in configs) else (None, {"fitted": False})
    results: Dict[str, List] = {}
    for cfg in configs:
        results[cfg.name] = [run_scenario(s, cfg, llm, args.fuzzy, calibrator, args.min_interval, prices)
                             for s in evaluation]
        print(f"  {cfg.name}: done")
    summaries = {name: summarize(rs, args.seed) for name, rs in results.items()}
    reference = "proposed_v2" if v2 else "proposed"
    tests = {name: mcnemar(results[reference], rs) for name, rs in results.items()
             if name != reference and reference in results}
    failure = {}
    if v2:
        from experiments.metrics import corridor_stats, failure_stats
        from experiments.variants import with_gas_failure

        for name in [n for n in FAILURE if n in BY_NAME]:
            failure[name] = [run_scenario(with_gas_failure(s), BY_NAME[name], llm, args.fuzzy, calibrator,
                                          args.min_interval, prices) for s in evaluation]
            print(f"  failure block {name}: done")
    out = args.out or os.path.join("reports", f"experiment{'_v2' if v2 else ''}_{datetime.now():%Y-%m-%d}")
    manifest = {"dataset": {"name": dataset.name, "path": dataset_path, "hash": dataset.hash(),
                            "synthetic": dataset.synthetic, "scenarios": len(dataset.scenarios)},
                "seed": args.seed, "split": {"calibration": len(calibration), "evaluation": len(evaluation),
                                             "calibration_share": args.calibration_share},
                "kappa": annotator_kappa(dataset.labelled()),
                "llm": {"backend": args.llm, "model": getattr(getattr(llm, "backend", llm), "model", None),
                        "temperature": getattr(getattr(llm, "backend", llm), "temperature", None),
                        "calls": llm.calls, "cache_hits": getattr(llm, "hits", llm.calls - getattr(llm, "misses", 0)),
                        "invalid_json": getattr(llm, "invalid", 0), "offline": bool(args.offline),
                        "simulated_misses": getattr(llm, "misses", 0)},
                "calibration": calibration_info, "configs": {c.name: c.flags() for c in configs},
                "min_interval_sec": args.min_interval, "trust_registry": registry_snapshot(args.fuzzy),
                "rules_version": rules_version(args.fuzzy)}
    if v2:
        from experiments.report import write_v2

        manifest["suite"] = "v2"
        manifest["failure_block"] = {"configs": list(failure), "drift_pct": 1.5, "health": 0.3,
                                     "scenarios": len(evaluation)}
        write_v2(out, summaries, results, tests, manifest, {n: corridor_stats(rs) for n, rs in results.items()},
                 {n: failure_stats(rs) for n, rs in failure.items()}, failure)
    else:
        write(out, summaries, results, tests, manifest)
    llm.close()
    width = max(len(n) for n in summaries)
    for name, s in summaries.items():
        print(f"{name:<{width}}  F1 {s['macro_f1']:.3f}  ASR {s['asr']:.3f} {s['asr_ci']}  "
              f"FAR {s['false_alarm_rate']:.3f}"
              f"  ECE {s['ece']:.3f}  calls/h {s['llm_calls_per_hour']}")
    hits = getattr(llm, "hits", llm.calls - getattr(llm, "misses", 0))
    print(f"→ {out}/RESULTS.md (LLM cache hits {hits}/{llm.calls})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
