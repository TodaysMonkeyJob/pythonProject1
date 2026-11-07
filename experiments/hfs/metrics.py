"""Metrics of the A6 (TOD-46) evaluation: macro-F1 of the overall level, Brier / ECE, corridor coverage."""
from typing import Dict, List, Sequence, Tuple

import numpy as np

LEVELS = ("low", "medium", "high", "critical")


def macro_f1(truth: Sequence[str], pred: Sequence[str]) -> float:
    scores = []
    for level in LEVELS:
        tp = sum(t == level and p == level for t, p in zip(truth, pred, strict=True))
        fp = sum(t != level and p == level for t, p in zip(truth, pred, strict=True))
        fn = sum(t == level and p != level for t, p in zip(truth, pred, strict=True))
        if tp + fp + fn == 0:
            continue                                       # a level absent from both: not part of the average
        scores.append(2 * tp / (2 * tp + fp + fn))
    return round(float(np.mean(scores)), 4) if scores else 0.0


def confusion(truth: Sequence[str], pred: Sequence[str]) -> Dict[str, Dict[str, int]]:
    return {t: {p: sum(a == t and b == p for a, b in zip(truth, pred, strict=True)) for p in LEVELS} for t in LEVELS}


def probabilities(memberships: Dict[str, float]) -> List[float]:
    """:return: the output-term memberships at the crisp risk, normalised to sum 1 (the level's probabilities)"""
    values = np.array([max(0.0, float(memberships.get(level, 0.0))) for level in LEVELS])
    total = values.sum()
    return list(values / total) if total > 0 else [0.25] * 4


def brier(truth: Sequence[str], probs: Sequence[Sequence[float]]) -> float:
    errors = [sum((p - (level == t)) ** 2 for p, level in zip(row, LEVELS, strict=True))
              for t, row in zip(truth, probs, strict=True)]
    return round(float(np.mean(errors)), 4)


def ece(truth: Sequence[str], probs: Sequence[Sequence[float]], bins: int = 5) -> float:
    confidence = np.array([max(row) for row in probs])
    correct = np.array([LEVELS[int(np.argmax(row))] == t for t, row in zip(truth, probs, strict=True)], float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        inside = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        if inside.any():
            total += inside.mean() * abs(confidence[inside].mean() - correct[inside].mean())
    return round(float(total), 4)


def level_region(curves: Dict[str, np.ndarray], grid: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """:return: per output term the risk range where it is the strongest term (its decision region)"""
    stack = np.vstack([curves[level] for level in LEVELS])
    winner = stack.argmax(axis=0)
    out = {}
    for i, level in enumerate(LEVELS):
        xs = grid[winner == i]
        out[level] = (float(xs.min()), float(xs.max())) if len(xs) else (float("nan"), float("nan"))
    return out


def covers(interval: Sequence[float], region: Tuple[float, float]) -> bool:
    """:return: the corridor reaches into the expected level's decision region"""
    return interval[0] <= region[1] + 1e-9 and interval[1] >= region[0] - 1e-9
