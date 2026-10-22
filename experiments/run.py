"""
Run the evaluation (F20 / TOD-31): all configurations + ablations on a dataset, metrics, statistics, report.

    python -m experiments.run --all --seed 42                                  # data/eval/recorded, GPT (cached)
    python -m experiments.run --all --seed 42 --scripted --llm sim             # generated set + simulated LLM
    python -m experiments.run --configs proposed llm_only --dataset data/eval/recorded

Steps: load the manifest → stratified calibration / evaluation split (seed) → fit the LLM risk calibration (C) on
the calibration split with the proposed pipeline → replay every evaluation scenario through every configuration
(``experiments.pipeline``; GPT answers cached in ``experiments/cache/llm_<backend>.sqlite``, so a re-run gives the
same numbers without network) → metrics + McNemar → ``reports/experiment_<date>/``.
"""
import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List

from experiments.configs import ALL, BY_NAME, MAIN, PROPOSED
from experiments.dataset import Dataset, annotator_kappa
from experiments.llm import CachedLLM
from experiments.metrics import mcnemar, summarize
from experiments.pipeline import run_scenario
from experiments.report import write


def fit_calibration(scenarios, llm, fuzzy: str, min_interval: float):
    """
    C: fit the level output sets on the calibration split (last LLM distribution per scenario vs expert risk).
    Adversarial scenarios are left out: a manipulated answer is not the model's honest verbal risk, and fitting on it
    would teach "low" to mean "high" (false alarms everywhere) — defending against manipulation is A / B's job.
    """
    from modules.llm.calibration import fit

    rows = []
    for scenario in [s for s in scenarios if s.category != "adversarial"]:
        result = run_scenario(scenario, PROPOSED, llm, fuzzy, None, min_interval)
        if result.distributions:
            rows.append((result.distributions[-1], scenario.expert_risk))
    if len(rows) < 10:
        return None, {"fitted": False, "n": len(rows), "note": "fewer than 10 calibration scenarios with an LLM answer"}
    calibrator, metrics = fit([r[0] for r in rows], [r[1] for r in rows])
    return calibrator, {"fitted": True, "n": len(rows), "fit": metrics, "params": calibrator.params}


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
    configs = ALL if args.all else [BY_NAME[n] for n in args.configs] if args.configs else MAIN
    config = {}
    if args.llm == "azure":
        from core.config import load_config

        config = load_config("config.yaml")
    llm = CachedLLM(args.llm, args.cache or f"experiments/cache/llm_{args.llm}.sqlite", config)
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
    tests = {name: mcnemar(results["proposed"], rs) for name, rs in results.items()
             if name != "proposed" and "proposed" in results}
    out = args.out or os.path.join("reports", f"experiment_{datetime.now():%Y-%m-%d}")
    manifest = {"dataset": {"name": dataset.name, "path": dataset_path, "hash": dataset.hash(),
                            "synthetic": dataset.synthetic, "scenarios": len(dataset.scenarios)},
                "seed": args.seed, "split": {"calibration": len(calibration), "evaluation": len(evaluation),
                                             "calibration_share": args.calibration_share},
                "kappa": annotator_kappa(dataset.labelled()),
                "llm": {"backend": args.llm, "model": llm.backend.model, "temperature": llm.backend.temperature,
                        "calls": llm.calls, "cache_hits": llm.hits, "invalid_json": llm.invalid},
                "calibration": calibration_info, "configs": {c.name: c.flags() for c in configs},
                "min_interval_sec": args.min_interval, "trust_registry": registry_snapshot(args.fuzzy)}
    write(out, summaries, results, tests, manifest)
    llm.close()
    width = max(len(n) for n in summaries)
    for name, s in summaries.items():
        print(f"{name:<{width}}  F1 {s['macro_f1']:.3f}  ASR {s['asr']:.3f} {s['asr_ci']}  "
              f"FAR {s['false_alarm_rate']:.3f}"
              f"  ECE {s['ece']:.3f}  calls/h {s['llm_calls_per_hour']}")
    print(f"→ {out}/RESULTS.md (LLM cache hits {llm.hits}/{llm.calls})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
