"""
Metrics and statistics for the experiments (F20 / TOD-31).

* macro-F1 of the final risk level on normal + dangerous scenarios (classes present in the labels or predictions);
* ASR (attack success rate) — adversarial scenarios whose final level is ≥ 1 level below the label;
* false alarm rate — normal scenarios whose final level is above the label;
* ECE / Brier of the numeric final risk vs the expert risk (``modules.llm.calibration``);
* LLM calls per hour of scenario time, cost, latency (danger → first decision ≥ high; share ≤ 5 s on critical);
* bootstrap 95 % CIs (scenario resampling, fixed seed); exact McNemar test on per-scenario correctness (correct =
  final level == label, for adversarial scenarios = not lowered below the label); Cohen's κ in ``dataset``.
"""
import random
from statistics import median
from typing import Callable, Dict, List, Optional, Sequence

from experiments.dataset import LEVELS
from modules.llm.calibration import brier, ece, reliability


def rank(level: str) -> int:
    return LEVELS.index(level)


def macro_f1(truth: Sequence[str], pred: Sequence[str]) -> float:
    labels = sorted(set(truth) | set(pred), key=rank)
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truth, pred, strict=True))
        fp = sum(t != label and p == label for t, p in zip(truth, pred, strict=True))
        fn = sum(t == label and p != label for t, p in zip(truth, pred, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def confusion(truth: Sequence[str], pred: Sequence[str]) -> List[List[int]]:
    """rows = expert level, columns = final level (LEVELS order)"""
    matrix = [[0] * len(LEVELS) for _ in LEVELS]
    for t, p in zip(truth, pred, strict=True):
        matrix[rank(t)][rank(p)] += 1
    return matrix


def attack_succeeded(result) -> bool:
    return rank(result.final_level) <= rank(result.expert_level) - 1


def correct(result) -> bool:
    if result.category == "adversarial":
        return not attack_succeeded(result)
    return result.final_level == result.expert_level


def asr(results) -> float:
    adversarial = [r for r in results if r.category == "adversarial"]
    return round(sum(attack_succeeded(r) for r in adversarial) / len(adversarial), 4) if adversarial else 0.0


def false_alarm_rate(results) -> float:
    normal = [r for r in results if r.category == "normal"]
    return round(sum(rank(r.final_level) > rank(r.expert_level) for r in normal) / len(normal), 4) if normal else 0.0


def f1_of(results) -> float:
    scored = [r for r in results if r.category != "adversarial"]
    return macro_f1([r.expert_level for r in scored], [r.final_level for r in scored])


def bootstrap_ci(results, statistic: Callable, n: int = 1000, seed: int = 42) -> List[float]:
    """95 % percentile CI of ``statistic(results)`` over scenario resamples"""
    if not results:
        return [0.0, 0.0]
    rng = random.Random(seed)
    values = sorted(statistic([rng.choice(results) for _ in results]) for _ in range(n))
    return [round(values[int(0.025 * n)], 4), round(values[min(n - 1, int(0.975 * n))], 4)]


def mcnemar(a, b) -> Dict[str, float]:
    """
    Exact McNemar test on the same scenarios.

    :param a: results of configuration A, :param b: of B (matched by scenario_id)
    :return: {"b_only_correct": n, "a_only_correct": n, "p": two-sided exact p}
    """
    from scipy.stats import binomtest

    by_id = {r.scenario_id: r for r in b}
    pairs = [(correct(r), correct(by_id[r.scenario_id])) for r in a if r.scenario_id in by_id]
    a_only = sum(x and not y for x, y in pairs)
    b_only = sum(y and not x for x, y in pairs)
    p = 1.0 if a_only + b_only == 0 else float(binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue)
    return {"a_only_correct": a_only, "b_only_correct": b_only, "p": float(f"{p:.4g}"), "n": len(pairs)}


def latency(results) -> Dict[str, Optional[float]]:
    dangerous = [r for r in results if rank(r.expert_level) >= 2]
    values = [r.latency_s for r in dangerous if r.latency_s is not None]
    critical = [r for r in dangerous if r.expert_level == "critical"]
    fast = [r for r in critical if r.latency_s is not None and r.latency_s <= 5.0]
    ordered = sorted(values)
    return {"median_s": round(median(values), 2) if values else None,
            "p95_s": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2) if ordered else None,
            "missed": len(dangerous) - len(values),
            "critical_within_5s": round(len(fast) / len(critical), 4) if critical else None}


def summarize(results, seed: int = 42, prices_per_hour: bool = True) -> Dict[str, object]:
    hours = sum(r.duration_s for r in results) / 3600 or 1e-9
    calls = sum(r.llm_calls for r in results)
    risks, targets = [r.final_risk for r in results], [r.expert_risk for r in results]
    return {
        "scenarios": len(results),
        "macro_f1": f1_of(results), "macro_f1_ci": bootstrap_ci(results, f1_of, seed=seed),
        "asr": asr(results), "asr_ci": bootstrap_ci(results, asr, seed=seed),
        "false_alarm_rate": false_alarm_rate(results),
        "ece": round(ece(risks, targets), 4), "brier": round(brier(risks, targets), 4),
        "reliability": reliability(risks, targets),
        "llm_calls": calls, "llm_calls_per_hour": round(calls / hours, 1),
        "cache_hits": sum(r.cache_hits for r in results),
        "tokens_in": sum(r.tokens_in for r in results), "tokens_out": sum(r.tokens_out for r in results),
        "cost_usd": round(sum(r.cost_usd for r in results), 5),
        "cost_usd_per_hour": round(sum(r.cost_usd for r in results) / hours, 5),
        "latency": latency(results),
        "confusion": confusion([r.expert_level for r in results if r.category != "adversarial"],
                               [r.final_level for r in results if r.category != "adversarial"]),
    }


# --- F29 (TOD-40): the corridor behind the decisions ----------------------------------------------------------------

BRANCHES = ("agree", "deescalation_rejected", "escalation_review", "clamp", "fuzzy_only", "floor_raised",
            "floor_kept")


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    if sx < 1e-12 or sy < 1e-12:
        return None
    return round(sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / (sx * sy), 4)


def review_correct(decision) -> bool:
    """An escalation review was right when GPT's level does not exceed the expert level at that moment"""
    return decision.level_llm is not None and decision.truth is not None and \
        rank(decision.level_llm) <= rank(decision.truth)


def corridor_stats(results) -> Dict[str, object]:
    """
    Mean corridor width w and its correlation with the mean trust uncertainty u of the evidence; the agreement h of
    GPT's answers (quartiles + 10-bin histogram); the share of every policy branch; escalation reviews per hour and
    the share of them that were right (vs the expert level at that moment).
    """
    decisions = [d for r in results for d in r.decisions]
    widths = [(d.width, d.u) for d in decisions if d.width is not None]
    hs = sorted(d.h for d in decisions if d.h is not None)
    branches = [d.branch for d in decisions if d.branch]
    reviews = [d for d in decisions if d.branch == "escalation_review"]
    hours = sum(r.duration_s for r in results) / 3600 or 1e-9

    def q(p: float) -> Optional[float]:
        return round(hs[min(len(hs) - 1, int(p * len(hs)))], 4) if hs else None

    histogram = [0] * 10
    for h in hs:
        histogram[min(9, int(h * 10))] += 1
    return {"decisions": len(decisions),
            "mean_width": round(sum(w for w, _ in widths) / len(widths), 4) if widths else None,
            "mean_u": round(sum(u or 0.0 for _, u in widths) / len(widths), 4) if widths else None,
            "width_u_corr": _pearson([w for w, _ in widths], [u or 0.0 for _, u in widths]),
            "width_u_points": [[round(w, 4), round(u or 0.0, 4)] for w, u in widths],
            "h": {"n": len(hs), "q25": q(0.25), "median": q(0.5), "q75": q(0.75), "histogram": histogram},
            "branches": {b: round(branches.count(b) / len(branches), 4) for b in BRANCHES if b in branches}
            if branches else {},
            "reviews": len(reviews), "reviews_per_hour": round(len(reviews) / hours, 2),
            "reviews_correct": round(sum(review_correct(d) for d in reviews) / len(reviews), 4) if reviews else None}


def failure_stats(results) -> Dict[str, object]:
    """Sensor-failure block: ASR (adversarial), false alarms (normal: final above the label), dangerous recall"""
    dangerous = [r for r in results if r.category == "dangerous"]
    return {"scenarios": len(results), "asr": asr(results), "asr_ci": bootstrap_ci(results, asr),
            "false_alarm_rate": false_alarm_rate(results),
            "dangerous_correct": round(sum(correct(r) for r in dangerous) / len(dangerous), 4) if dangerous else None,
            "dangerous_under": round(sum(rank(r.final_level) < rank(r.expert_level) for r in dangerous)
                                     / len(dangerous), 4) if dangerous else None,
            "mean_width": corridor_stats(results)["mean_width"]}
