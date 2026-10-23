"""
Measured trust inputs (F22 / TOD-33): calibration maps, validation priors and cost-based thresholds.

Every trust degree / decision threshold in ``config/fuzzy.yaml`` comes from one of three files written by the tools
(``tools/calibrate_channels.py``, ``tools/trust_priors.py``, ``tools/fit_thresholds.py``):

* ``config/calibration/<channel>.yaml`` — a map raw model score → calibrated probability (isotonic knots, logistic
  a·s + b, or Bayes bins), with the dataset sha256, n, date and metrics (Brier, ECE, EER, Cllr);
* ``config/trust_priors.yaml`` — per channel r0 (correct) / s0 (wrong) on held-out data → τ0 = (r0+1)/(r0+s0+2),
  u0 = 2/(r0+s0+2) and the Wilson 95 % lower bound;
* ``config/thresholds.yaml`` — thresholds minimising C_miss·P_miss + C_fa·P_fa for an explicit cost ratio.

``config/fuzzy.yaml`` refers to them with strings ``"@<file>:<dotted.path>"`` (``@thresholds:envelope.theta``,
``@trust_priors:gas.tau0``), resolved by ``core.fuzzy.engine.load_yaml``. Functions here are numpy only (sklearn only
inside the two fit functions) so the live system can apply a map on the Pi without sklearn.
"""
import hashlib
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

Z95 = 1.959963984540054


# --- provenance -----------------------------------------------------------------------------------------------------

def sha256_files(paths: Iterable[str]) -> str:
    """:return: sha256 over the files' bytes in the given order (dataset hash)"""
    digest = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as f:
            digest.update(hashlib.sha256(f.read()).digest())
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_yaml(path: str, data: Dict[str, Any], header: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in header.strip().splitlines():
            f.write(f"# {line}\n")
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, width=120)


def read_yaml(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --- metrics for binary outcomes ------------------------------------------------------------------------------------

def brier(p: Sequence[float], y: Sequence[float]) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def reliability(p: Sequence[float], y: Sequence[float], bins: int = 10) -> List[Dict[str, float]]:
    """:return: per non-empty equal-width bin of p: {lo, hi, n, mean_p, observed}"""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        if mask.any():
            rows.append({"lo": round(float(edges[b]), 2), "hi": round(float(edges[b + 1]), 2), "n": int(mask.sum()),
                         "mean_p": round(float(p[mask].mean()), 4), "observed": round(float(y[mask].mean()), 4)})
    return rows


def ece(p: Sequence[float], y: Sequence[float], bins: int = 10) -> float:
    """Expected calibration error: Σ |B|/n · |mean p_B − observed_B|"""
    n = len(p)
    return float(sum(r["n"] / n * abs(r["mean_p"] - r["observed"]) for r in reliability(p, y, bins))) if n else 0.0


def equal_error_rate(genuine: Sequence[float], impostor: Sequence[float]) -> Tuple[float, float]:
    """:return: (EER, threshold where the false accept rate ≈ the false reject rate)"""
    genuine, impostor = np.asarray(genuine, float), np.asarray(impostor, float)
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    far = np.array([np.mean(impostor >= t) for t in thresholds])
    frr = np.array([np.mean(genuine < t) for t in thresholds])
    best = int(np.argmin(np.abs(far - frr)))
    return float((far[best] + frr[best]) / 2), float(thresholds[best])


def cllr(genuine_llr: Sequence[float], impostor_llr: Sequence[float]) -> float:
    """
    Log-likelihood-ratio cost (bits; 0 = perfect, 1 = uninformative): 0.5·(mean log2(1 + e^−llr) over genuine
    + mean log2(1 + e^llr) over impostor pairs), llr in natural log.
    """
    g, i = np.asarray(genuine_llr, float), np.asarray(impostor_llr, float)
    return float(0.5 * (np.mean(np.logaddexp(0, -g)) + np.mean(np.logaddexp(0, i))) / math.log(2))


def logit(p: float) -> float:
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


# --- calibration maps -----------------------------------------------------------------------------------------------

class IsotonicMap:
    """Monotone piecewise-linear map through knots (x ascending), clipped outside — np.interp, no sklearn."""

    kind = "isotonic"

    def __init__(self, x: Sequence[float], y: Sequence[float]):
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        if len(self.x) == 0 or len(self.x) != len(self.y):
            raise ValueError("isotonic map needs matching, non-empty knots")

    def __call__(self, score: float) -> float:
        return float(np.interp(float(score), self.x, self.y))

    def many(self, scores: Sequence[float]) -> np.ndarray:
        return np.interp(np.asarray(scores, float), self.x, self.y)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "x": [round(float(v), 4) for v in self.x], "y": [round(float(v), 4) for v in self.y]}


class LogisticMap:
    """P = 1 / (1 + e^−(a·s + b)); ``llr`` subtracts the training prior log-odds (for Cllr / other priors)."""

    kind = "logistic"

    def __init__(self, a: float, b: float, train_prior: float = 0.5):
        self.a, self.b, self.train_prior = float(a), float(b), float(train_prior)

    def logit(self, scores) -> np.ndarray:
        return self.a * np.asarray(scores, float) + self.b

    def llr(self, scores) -> np.ndarray:
        return self.logit(scores) - logit(self.train_prior)

    def many(self, scores) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(scores)))

    def __call__(self, score: float) -> float:
        return float(self.many([score])[0])

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "a": round(self.a, 5), "b": round(self.b, 5),
                "train_prior": round(self.train_prior, 4)}


def map_from_dict(data: Dict[str, Any]):
    """:param data: the ``map`` block of a calibration file"""
    if data["kind"] in ("isotonic", "bayes_bins"):
        return IsotonicMap(data["x"], data["y"])
    if data["kind"] == "logistic":
        return LogisticMap(data["a"], data["b"], data.get("train_prior", 0.5))
    raise ValueError(f"unknown calibration map kind {data['kind']!r}")


def load_map(path: str):
    """:return: The calibrated map of a ``config/calibration/<channel>.yaml``, or None when not fitted yet"""
    data = read_yaml(path)
    return map_from_dict(data["map"]) if data and data.get("map") else None


def fit_isotonic(scores: Sequence[float], outcomes: Sequence[float]) -> IsotonicMap:
    """Non-decreasing least-squares fit of outcome (0/1) on score; knots = the fitted step corners."""
    from sklearn.isotonic import IsotonicRegression

    x, y = np.asarray(scores, float), np.asarray(outcomes, float)
    model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(x, y)
    knots_x, knots_y = np.asarray(model.X_thresholds_, float), np.asarray(model.y_thresholds_, float)
    keep = np.ones(len(knots_x), bool)                   # drop interior knots on a flat stretch (same y both sides)
    keep[1:-1] = ~((knots_y[1:-1] == knots_y[:-2]) & (knots_y[1:-1] == knots_y[2:]))
    return IsotonicMap(knots_x[keep], knots_y[keep])


def fit_logistic(scores: Sequence[float], outcomes: Sequence[int]) -> LogisticMap:
    """1-D logistic regression (weak L2, C = 1e4 → practically unregularised Platt scaling)."""
    from sklearn.linear_model import LogisticRegression

    x, y = np.asarray(scores, float).reshape(-1, 1), np.asarray(outcomes, int)
    model = LogisticRegression(C=1e4, max_iter=1000).fit(x, y)
    return LogisticMap(float(model.coef_[0, 0]), float(model.intercept_[0]), float(y.mean()))


def bayes_bins(thresholds: Sequence[float], tar: Sequence[float], far: Sequence[float], n_known: int, n_strangers: int,
               prior: float) -> Tuple[IsotonicMap, List[Dict[str, float]]]:
    """
    P(known | similarity) from measured TAR / FAR at ascending thresholds (an identification sweep that kept only the
    rates, e.g. ``reports/identity_eval_lfw.json``). The share of each class in the bin [t_i, t_i+1) is the rate
    difference; counts get +0.5 (Jeffreys) so a bin with no stranger never claims certainty. Posterior per bin by
    Bayes with ``prior`` = P(a face in view belongs to a known person); knots at the bin centres, made monotone.
    TAR here = known faces matched to the right person at ≥ t (the rest of the known faces count below t_0).

    :return: (map, per-bin table)
    """
    t = list(thresholds)
    tar_ext, far_ext = [1.0] + list(tar) + [0.0], [1.0] + list(far) + [0.0]
    edges = [0.0] + t + [1.0]
    rows, xs, ys, bins = [], [], [], len(edges) - 1
    for i in range(bins):
        known = max(0.0, tar_ext[i] - tar_ext[i + 1]) * n_known + 0.5
        strangers = max(0.0, far_ext[i] - far_ext[i + 1]) * n_strangers + 0.5
        l_known, l_strangers = known / (n_known + 0.5 * bins), strangers / (n_strangers + 0.5 * bins)
        posterior = prior * l_known / (prior * l_known + (1 - prior) * l_strangers)
        lo, hi = edges[i], edges[i + 1]
        rows.append({"lo": lo, "hi": hi, "known": round(known - 0.5, 1), "strangers": round(strangers - 0.5, 1),
                     "posterior": round(posterior, 4)})
        xs.append((lo + hi) / 2)
        ys.append(posterior)
    ys = list(np.maximum.accumulate(ys))                 # P(known) must not fall as similarity grows
    return IsotonicMap(xs, ys), rows


# --- priors from validation counts ----------------------------------------------------------------------------------

def beta_prior(correct: int, wrong: int) -> Dict[str, float]:
    """Beta(r0 + 1, s0 + 1) evidence: τ0 = expected trust, u0 = uncertainty mass (subjective logic, W = 2)."""
    n = correct + wrong
    return {"r0": int(correct), "s0": int(wrong), "tau0": round((correct + 1) / (n + 2), 4),
            "u0": round(2 / (n + 2), 4)}


def wilson_lower(correct: int, n: int, z: float = Z95) -> float:
    """Wilson score interval lower bound of a proportion (conservative, works for 0 errors)."""
    if n <= 0:
        return 0.0
    p = correct / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / (1 + z * z / n))


def prior_entry(correct: int, wrong: int, **meta) -> Dict[str, Any]:
    entry = {**beta_prior(correct, wrong), "wilson_lb": round(wilson_lower(correct, correct + wrong), 4),
             "n": int(correct + wrong), "measured": True}
    entry.update(meta)
    return entry


# --- cost-based thresholds ------------------------------------------------------------------------------------------

def _rates(scores: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    """P(score ≥ t) for every t in the grid; Gaussian-kernel smoothed when bandwidth > 0 (empirical otherwise)."""
    if len(scores) == 0:
        return np.zeros(len(grid))
    if bandwidth <= 0:
        return (scores[None, :] >= grid[:, None] - 1e-12).mean(axis=1)
    from math import erf

    z = (scores[None, :] - grid[:, None]) / (bandwidth * math.sqrt(2))
    return (0.5 * (1 + np.vectorize(erf)(z))).mean(axis=1)


def fit_cost_threshold(scores: Sequence[float], positive: Sequence[bool], cost_ratio: float,
                       grid: Optional[Sequence[float]] = None, bandwidth: float = 0.05,
                       positive_above: bool = True, lo: Optional[float] = None,
                       hi: Optional[float] = None, flag_cost: float = 0.0) -> Dict[str, Any]:
    """
    Threshold t minimising C_miss · P_miss + C_fa · P_fa (C_fa = 1, C_miss = cost_ratio) [+ flag_cost · P_flag].

    ``flag_cost`` (F28 / TOD-39, gating): a cost of every flagged case, needed or not (a GPT call costs tokens even
    when it was worth making), times the share of flagged cases P_flag.

    ``positive_above``: a case is flagged when score ≥ t (risk boundaries, gating) — miss = positive below t; otherwise
    flagged when score < t (trust θ: "distrust below θ") — miss = positive at / above t. Rates are kernel-smoothed
    (``bandwidth`` on the score scale) so separable data still gives a cost-ratio dependent margin instead of an
    arbitrary point of the zero-error gap; ties → the smallest t (with positive_above) / largest t (otherwise) —
    the side that misses less. Deterministic.

    :return: {threshold, expected_cost, p_miss, p_fa (smoothed), empirical_p_miss, empirical_p_fa, n_pos, n_neg}
    """
    s, pos = np.asarray(scores, float), np.asarray(positive, bool)
    grid = np.round(np.asarray(grid if grid is not None else np.linspace(0, 1, 101), float), 6)
    if lo is not None or hi is not None:
        grid = grid[(grid >= (lo if lo is not None else -np.inf)) & (grid <= (hi if hi is not None else np.inf))]
    if not len(grid):
        raise ValueError("empty threshold grid")
    above_pos, above_neg = _rates(s[pos], grid, bandwidth), _rates(s[~pos], grid, bandwidth)
    if positive_above:
        p_miss, p_fa = 1 - above_pos, above_neg
    else:
        p_miss, p_fa = above_pos, 1 - above_neg
    if not pos.any():
        p_miss = np.zeros(len(grid))
    if pos.all():
        p_fa = np.zeros(len(grid))
    cost = cost_ratio * p_miss + p_fa
    if flag_cost:
        flagged_share = _rates(s, grid, bandwidth) if positive_above else 1 - _rates(s, grid, bandwidth)
        cost = cost + flag_cost * flagged_share
    best_cost = cost.min()
    candidates = np.flatnonzero(cost <= best_cost + 1e-9)
    i = int(candidates[0] if positive_above else candidates[-1])
    t = float(grid[i])
    flagged = s >= t if positive_above else s < t
    return {"threshold": round(t, 4), "expected_cost": round(float(cost[i]), 4),
            "p_miss": round(float(p_miss[i]), 4), "p_fa": round(float(p_fa[i]), 4),
            "empirical_p_miss": round(float(np.mean(~flagged[pos])) if pos.any() else 0.0, 4),
            "empirical_p_fa": round(float(np.mean(flagged[~pos])) if (~pos).any() else 0.0, 4),
            "n_pos": int(pos.sum()), "n_neg": int((~pos).sum()),
            **({"flag_cost": flag_cost, "flagged_share": round(float(np.mean(flagged)) if len(s) else 0.0, 4)}
               if flag_cost else {})}


# --- ASR ------------------------------------------------------------------------------------------------------------

def _words(text: str) -> List[str]:
    import re

    return re.sub(r"[^\w\s'’]", " ", (text or "").lower()).replace("’", "'").split()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate = word-level Levenshtein distance / reference words (no jiwer needed)."""
    ref, hyp = _words(reference), _words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h))
        previous = current
    return previous[-1] / len(ref)


def asr_confidence(avg_logprob: float, no_speech_prob: float) -> float:
    """The live raw score (``modules.llm.whisper_client``): exp(avg_logprob) · (1 − no_speech_prob)"""
    return math.exp(min(0.0, float(avg_logprob))) * (1.0 - float(no_speech_prob or 0.0))
