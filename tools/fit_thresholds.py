"""
Cost-based decision thresholds (F22 / TOD-33) → ``config/thresholds.yaml``.

Every threshold minimises the expected cost C_miss · P_miss + C_fa · P_fa (C_fa = 1) on the calibration split of the
evaluation set (``experiments.dataset.Dataset.split``, same seed / share as ``experiments.run``). The only expert
input is the cost ratio C_miss / C_fa (``calibration.cost_ratio`` in config.yaml, default 10, or ``--cost-ratio``).

| threshold | score | positive (= a miss is costly) |
| -- | -- | -- |
| trust.claim_threshold | τ of a speech / scene-text event | lowering claim in an adversarial scenario |
| levels medium / high / critical | r_final of the scenario's last decision | expert level ≥ that level |
| gating.threshold | call_llm score of a non-override snapshot | fuzzy-only gives the wrong level (floor ≠ truth) |

F24 (TOD-35): the IT2 corridor's FOU scale k_var = κ · span_var (span = the variable's mean edge width, its natural
unit of vagueness) — κ minimises the mean interval score of the final corridor [r_floor, r_ceil] against the expert
risk y: (r_ceil − r_floor) + C·(r_floor − y)⁺ + C·(y − r_ceil)⁺ with C = the cost ratio (a corridor that misses the
expert risk costs C per unit, its width 1), replayed through trust scoring + the envelope only (no LLM).

Truth per snapshot for gating: the scenario's expert level from the danger onset on, ``low`` before it. Rates are
kernel smoothed (``--bandwidth``), so the result moves with the cost ratio even where the calibration split is
separable. One replay with the current config (no fixed-point iteration). GPT answers come from the experiment cache
(``--llm azure``, offline: a cache miss is answered by the local simulator, counted, and that scenario is left out of
the level fit — its final risk would not be GPT's; θ / gating do not depend on the answer text); nothing is sent. λ and
h_min of the corridor issue do not exist in the code yet and are listed as not applicable.

    python -m tools.fit_thresholds                      # cost ratio from config.yaml
    python -m tools.fit_thresholds --cost-ratio 5       # → config/thresholds.yaml + reports/thresholds_log.jsonl
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

if __package__ in (None, ""):  # allow `python tools/fit_thresholds.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration import fit_cost_threshold, now_iso, sha256_files, write_yaml  # noqa: E402

OUT = "config/thresholds.yaml"
LOG = "reports/thresholds_log.jsonl"
LEVELS = ("low", "medium", "high", "critical")
LOWERING = ("deescalation", "instruction_to_system")
# The hand-picked values before F22 — only written when no thresholds.yaml exists yet, so the first replay can load
# config/fuzzy.yaml (which refers to this file); the fit then replaces them.
BOOTSTRAP = {"claim_theta": 0.6, "levels": {"medium": 0.25, "high": 0.5, "critical": 0.8}, "gating": 0.5,
             "critical_floor": 0.8, "k_var": {}}
KAPPA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
FOU_VARIABLES = ("gas_concentration", "gas_trend", "motion_level", "people_count", "unknown_share", "action_risk",
                 "known_present")
HEADER = """F22 (TOD-33) cost-based thresholds — written by python -m tools.fit_thresholds; do not edit by hand.
Each value minimises C_miss * P_miss + C_fa * P_fa on the calibration split; cost_ratio = C_miss / C_fa is the only
expert input. config/fuzzy.yaml reads `values` through "@thresholds:values.<name>" references."""


class OfflineLLM:
    """
    Read-only view of the experiment cache (sqlite opened with mode=ro, so nothing is ever written to it): a hit
    returns the recorded GPT answer, a miss is answered by the local simulator and counted. No network.
    """

    def __init__(self, backend: str, cache_path: str):
        import sqlite3

        from experiments.llm import PROMPT_V1, PROMPT_V2, SimAnalyst, load_prompt

        self.sim = SimAnalyst()
        self.backend_name = backend
        if backend == "sim":
            self.model, self.temperature = self.sim.model, self.sim.temperature
        else:                                                   # same key fields as experiments.llm.AzureAnalyst
            from core.config import load_config

            llm = load_config("config.yaml").get("llm", {})
            self.model = llm.get("gpt_deployment") or llm.get("chat_deployment", "gpt-4o-mini")
            self.temperature = float(llm.get("temperature", 0.2))
        self.prompts = {"v1": load_prompt(*PROMPT_V1), "v2": load_prompt(*PROMPT_V2)}
        self.db = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True) if os.path.isfile(cache_path) else None
        self.calls = self.misses = 0

    def analyze(self, payload: str, prompt_version: str):
        import hashlib

        from modules.llm.gpt_client import LLMDecision

        key = hashlib.sha256(json.dumps([self.backend_name, self.model, self.temperature,
                                         self.prompts[prompt_version], payload]).encode()).hexdigest()
        self.calls += 1
        row = self.db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone() if self.db else None
        if row is None:
            self.misses += 1
            decision, usage = self.sim.analyze(payload, prompt_version)
            return LLMDecision(**decision), {**usage, "cache_hit": False, "offline_sim": True}
        cached = json.loads(row[0])
        return LLMDecision(**cached["decision"]), {**cached["usage"], "cache_hit": True}

    def close(self) -> None:
        if self.db is not None:
            self.db.close()


def collect(scenarios, llm, fuzzy: str, min_interval: float) -> Dict[str, List[Dict[str, Any]]]:
    """One replay of the proposed pipeline per scenario → per-scenario outcomes, evidence trust, gate scores."""
    from experiments.configs import PROPOSED
    from experiments.pipeline import run_scenario
    from modules.trust.channels import epoch

    outcomes, evidence, gates = [], [], []
    for scenario in scenarios:
        trace: Dict[str, List[Dict[str, Any]]] = {"evidence": [], "gate": []}
        misses = getattr(llm, "misses", 0)
        result = run_scenario(scenario, PROPOSED, llm, fuzzy, None, min_interval, trace=trace)
        outcomes.append({"scenario_id": scenario.scenario_id, "category": scenario.category,
                         "expert_level": scenario.expert_level, "risk_final": result.final_risk,
                         "recorded_llm": getattr(llm, "misses", 0) == misses})
        for item in trace["evidence"]:
            if item["type"] not in ("speech", "scene_text"):     # F24: the threshold only gates language claims
                continue
            attack = scenario.category == "adversarial" and item["claim_type"] in LOWERING
            evidence.append({**item, "attack": attack, "scenario_id": scenario.scenario_id})
        onset = epoch(scenario.danger_onset) if scenario.danger_onset else None
        for item in trace["gate"]:
            truth = scenario.expert_level if (scenario.category == "normal" or onset is None
                                              or item["t"] >= onset - 1) else "low"
            gates.append({**item, "truth": truth, "scenario_id": scenario.scenario_id})
    return {"outcomes": outcomes, "evidence": evidence, "gate": gates}


MIN_LEVEL_SCENARIOS = 10


def fit_all(data: Dict[str, List[Dict[str, Any]]], cost_ratio: float, bandwidth: float,
            current_levels: Dict[str, float], previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    :param previous: The current thresholds.yaml — its level boundaries are kept when fewer than
        ``MIN_LEVEL_SCENARIOS`` calibration scenarios have recorded GPT answers (a changed trust config changes the
        payloads, so the cache has to be refreshed by an ``experiments.run`` with Azure before the levels can be refit)
    :return: {values: {...}, fits: {...}} — deterministic for the same data and cost ratio
    """
    from modules.envelope.projection import level_of

    fits: Dict[str, Any] = {}
    evidence = data["evidence"]
    fits["claim_theta"] = fit_cost_threshold([e["trust"] for e in evidence], [e["attack"] for e in evidence],
                                             cost_ratio, np.linspace(0.05, 0.95, 91), bandwidth, positive_above=False)
    outcomes, levels, lo = [o for o in data["outcomes"] if o["recorded_llm"]], {}, 0.05   # real GPT answers only
    kept = previous if (previous and not previous.get("bootstrap")) else None
    if len(outcomes) < MIN_LEVEL_SCENARIOS and kept:
        levels = dict(kept["values"]["levels"])
        for level in LEVELS[1:]:
            fits[f"level_{level}"] = {**(kept.get("fits") or {}).get(f"level_{level}", {}), "kept_from": kept.get(
                "fitted_at"), "kept_reason": f"only {len(outcomes)} calibration scenarios with recorded GPT answers"}
    else:
        for level in LEVELS[1:]:                          # boundaries in order, each above the previous one
            positive = [LEVELS.index(o["expert_level"]) >= LEVELS.index(level) for o in outcomes]
            fit = fit_cost_threshold([o["risk_final"] for o in outcomes], positive, cost_ratio,
                                     np.linspace(0.05, 0.95, 91), bandwidth, positive_above=True, lo=lo)
            fits[f"level_{level}"] = fit
            levels[level] = fit["threshold"]
            lo = round(fit["threshold"] + 0.05, 4)
    gates = [g for g in data["gate"] if g["override"] is None]
    wrong = [level_of(g["floor"], levels) != g["truth"] for g in gates]
    fits["gating"] = fit_cost_threshold([g["score"] for g in gates], wrong, cost_ratio, np.linspace(0.05, 0.95, 91),
                                        bandwidth, positive_above=True)
    used = {"used": len(outcomes), "skipped_without_recorded_gpt": len(data["outcomes"]) - len(outcomes)}
    values = {"claim_theta": fits["claim_theta"]["threshold"], "levels": levels, "gating": fits["gating"]["threshold"],
              "critical_floor": levels["critical"]}
    return {"values": values, "fits": fits, "previous_levels": current_levels, "level_scenarios": used}


def spans(fuzzy: str) -> Dict[str, float]:
    """:return: {variable: mean width of its sloped term edges} (Gaussians: σ) — the unit of k_var"""
    from core.fuzzy import FuzzyEngine, load_yaml

    root = load_yaml(fuzzy)
    variables = dict(FuzzyEngine.from_dict(root).variables)
    confirmation = ((root.get("envelope") or {}).get("confirmation") or {}).get("variables") or {}
    if confirmation:
        variables.update(FuzzyEngine.from_dict({"output": "confirmation", "variables": confirmation,
                                                "rules": []}).variables)
    out = {}
    for name in FOU_VARIABLES:
        if name not in variables:
            continue
        widths = []
        for parts in variables[name].terms.values():
            for shape, p in parts:
                if shape == "gauss":
                    widths.append(p[1])
                else:
                    pts = (p[0], p[1], p[1], p[2]) if shape == "tri" else p
                    widths += [w for w in (pts[1] - pts[0], pts[3] - pts[2]) if w > 0]
        out[name] = round(float(np.mean(widths)), 4) if widths else 0.0
    return out


def corridor_scores(scenarios, fuzzy: str, k: Dict[str, float]) -> List[Dict[str, Any]]:
    """Replays trust scoring + the envelope (no aggregator / LLM) → the final corridor of every scenario."""
    from core.events import Event
    from modules.envelope import RiskEnvelope
    from modules.trust import TrustScorer
    from modules.trust.channels import epoch

    rows = []
    for scenario in scenarios:
        scorer, envelope = TrustScorer.from_config(fuzzy), RiskEnvelope.from_config(fuzzy, None)
        envelope.corridor.k_override = k
        envelope.confirmation.it2.k = dict(k)
        now, location = 0.0, None
        for line in scenario.events():
            event = Event.from_json(line)
            event.payload.pop("trust", None)
            now = epoch(event.timestamp) or now
            location = location or (event.location_id if event.location_id != "unassigned" else None)
            scorer.apply(event)
            envelope.observe(event)
        _, _, corridor = envelope.assess(location, now + 1)
        rows.append({"scenario_id": scenario.scenario_id, "floor": corridor["floor"], "ceil": corridor["ceil"],
                     "expert": scenario.expert_risk})
        scorer.registry.close()
    return rows


def interval_score(rows: List[Dict[str, Any]], cost_ratio: float) -> Dict[str, float]:
    width = [r["ceil"] - r["floor"] for r in rows]
    below = [max(0.0, r["floor"] - r["expert"]) for r in rows]
    above = [max(0.0, r["expert"] - r["ceil"]) for r in rows]
    score = [w + cost_ratio * (b + a) for w, b, a in zip(width, below, above, strict=True)]
    covered = [r["floor"] - 1e-9 <= r["expert"] <= r["ceil"] + 1e-9 for r in rows]
    return {"interval_score": round(float(np.mean(score)), 4), "mean_width": round(float(np.mean(width)), 4),
            "coverage": round(float(np.mean(covered)), 4)}


def fit_k(scenarios, fuzzy: str, cost_ratio: float) -> Dict[str, Any]:
    """:return: {"k_var": {...}, "kappa", "by_kappa": [...], "spans"} — the smallest κ with the lowest score"""
    unit = spans(fuzzy)
    tried = []
    for kappa in KAPPA_GRID:
        k = {name: round(kappa * span, 4) for name, span in unit.items()}
        tried.append({"kappa": kappa, **interval_score(corridor_scores(scenarios, fuzzy, k), cost_ratio)})
    best = min(tried, key=lambda t: (t["interval_score"], t["kappa"]))
    return {"kappa": best["kappa"], "k_var": {name: round(best["kappa"] * span, 4) for name, span in unit.items()},
            "spans": unit, "by_kappa": tried, "expected_cost": best["interval_score"]}


def main(argv: Optional[List[str]] = None) -> int:
    from core.config import load_config
    from experiments.dataset import Dataset

    parser = argparse.ArgumentParser(description="Fit decision thresholds by expected cost")
    parser.add_argument("--dataset", default="data/eval/scripted")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-share", type=float, default=0.3)
    parser.add_argument("--cost-ratio", type=float, help="C_miss / C_fa (default: calibration.cost_ratio or 10)")
    parser.add_argument("--bandwidth", type=float, default=0.05, help="kernel width of the smoothed rates")
    parser.add_argument("--llm", choices=("azure", "sim"), default="azure", help="whose cached answers (offline)")
    parser.add_argument("--fuzzy", default="config/fuzzy.yaml")
    parser.add_argument("--min-interval", type=float, default=300.0)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--log", default=LOG)
    args = parser.parse_args(argv)
    config = load_config("config.yaml")
    cost_ratio = args.cost_ratio if args.cost_ratio is not None else \
        float((config.get("calibration") or {}).get("cost_ratio", 10.0))
    if args.dataset == "data/eval/scripted" and not os.path.isfile(os.path.join(args.dataset, "manifest.json")):
        from experiments.scripted import build

        build(args.dataset, args.seed)
    if not os.path.isfile(args.out):
        write_yaml(args.out, {"version": 1, "bootstrap": True, "values": BOOTSTRAP}, HEADER)
    dataset = Dataset.load(args.dataset)
    calibration, _ = dataset.split(args.seed, args.calibration_share)
    llm = OfflineLLM(args.llm, f"experiments/cache/llm_{args.llm}.sqlite")
    try:
        from core.fuzzy import load_yaml

        current = (load_yaml(args.fuzzy).get("envelope") or {}).get("thresholds") or {}
        data = collect(calibration, llm, args.fuzzy, args.min_interval)
    finally:
        llm.close()
    from core.calibration import read_yaml

    result = fit_all(data, cost_ratio, args.bandwidth, current, read_yaml(args.out))
    k_fit = fit_k(calibration, args.fuzzy, cost_ratio)                  # F24: FOU scale of the IT2 corridor
    result["values"]["k_var"] = k_fit["k_var"]
    result["fits"]["k_var"] = {key: v for key, v in k_fit.items() if key != "k_var"}
    output = {"version": 1, "fitted_at": now_iso(), "cost_ratio": cost_ratio, "c_fa": 1.0, "c_miss": cost_ratio,
              "bandwidth": args.bandwidth,
              "dataset": {"path": args.dataset, "name": dataset.name, "hash": dataset.hash(),
                          "synthetic": dataset.synthetic, "split": "calibration", "seed": args.seed,
                          "scenarios": len(calibration), "evidence_events": len(data["evidence"]),
                          "gate_evaluations": len(data["gate"])},
              "fuzzy_config": {"path": os.path.relpath(args.fuzzy) if not os.path.relpath(args.fuzzy).startswith("..")
                               else os.path.basename(args.fuzzy), "sha256": sha256_files([args.fuzzy])},
              "llm": {"backend": args.llm, "offline": True, "calls": llm.calls, "cache_misses": llm.misses},
              "values": result["values"],
              "expected_cost": {name: fit.get("expected_cost") for name, fit in result["fits"].items()},
              "fits": result["fits"], "level_scenarios": result["level_scenarios"],
              "previous_levels": result["previous_levels"],
              "not_applicable": {"lambda": "corridor risk attitude — not in the code yet (corridor issue)",
                                 "h_min": "agreement threshold — not in the code yet (corridor issue)"}}
    write_yaml(args.out, output, HEADER)
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    with open(args.log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"fitted_at": output["fitted_at"], "cost_ratio": cost_ratio, "bandwidth": args.bandwidth,
                            "dataset_hash": output["dataset"]["hash"], "values": result["values"],
                            "expected_cost": output["expected_cost"], "cache_misses": llm.misses,
                            "levels_kept": any("kept_from" in f for f in result["fits"].values())},
                           ensure_ascii=False) + "\n")
    print(json.dumps({"cost_ratio": cost_ratio, "values": result["values"], "expected_cost": output["expected_cost"],
                      "cache_misses": llm.misses, "calls": llm.calls}, indent=1))
    print(f"→ {args.out} (+ {args.log})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
