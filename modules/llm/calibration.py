"""
Calibrated LLM risk (F18 / TOD-29): defuzzification of the model's verbal risk distribution.

GPT answers ``risk_distribution`` {low, medium, high, critical} (probabilities). Each level has an output membership
function (trapezoid on [0, 1]); the calibrated risk is the centroid of their probability-weighted sum::

    r_LLM = centroid( Σ_t p_t · μ_t(x) )

The trapezoids are fitted on labelled scenarios (expert risk in [0, 1]) by minimising the Brier score (+ a share
of the ECE) — ``python -m tools.calibrate_llm_risk`` — and stored in ``config/calibration.yaml`` with the dataset
hash and date. Without that file the envelope keeps the fixed level → number mapping (TOD-28), so nothing changes
until a calibration was actually fitted.

Metrics for a continuous target y ∈ [0, 1]: Brier = mean (r − y)²; ECE = Σ_bins |B|/n · |mean r_B − mean y_B|
(10 equal-width bins of r); ``reliability()`` returns the per-bin data for a reliability diagram.

F28 (TOD-39): fitted on 21 scenarios the sets overfit (a "high" answer landed just above the high boundary and medium
scenarios became high). The fit is now constrained and robust:

* every level's centre stays inside the support of its own risk output term (``BOUNDS`` = the risk terms of
  config/fuzzy.yaml: GPT's "low" can never mean 0.7) and the centres strictly increase (≥ 0.05 apart);
* every support is at least ``min_width`` wide, slopes at most 0.3;
* the loss is a Huber loss (δ = 0.1) instead of the squared error — a talked-down answer in an attack scenario ("low"
  while the expert says critical) is an outlier of the honest verbal risk, its pull grows only linearly;
* ``loo_cv`` reports leave-one-out Brier / ECE next to the in-sample ones (a fit judged on scenarios it did not see).
"""
import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from core.fuzzy.membership import trapmf

LEVELS = ("low", "medium", "high", "critical")
# Start values = the risk output terms of config/fuzzy.yaml
DEFAULT_PARAMS: Dict[str, List[float]] = {"low": [0.0, 0.0, 0.1, 0.3], "medium": [0.15, 0.4, 0.4, 0.6],
                                          "high": [0.45, 0.7, 0.7, 0.85], "critical": [0.75, 0.9, 1.0, 1.0]}
FIXED = {"low": 0.1, "medium": 0.4, "high": 0.7, "critical": 0.95}     # TOD-28 fixed mapping (baseline)
GRID = np.linspace(0.0, 1.0, 201)
MIN_WIDTH = 0.2          # default minimum support width of a fitted level set (risk units)
MIN_GAP = 0.05           # minimum distance of consecutive centres
HUBER_DELTA = 0.1
BOUNDS = {level: (p[0], p[3]) for level, p in DEFAULT_PARAMS.items()}   # a centre stays in its risk term's support


class RiskCalibrator:
    def __init__(self, params: Optional[Dict[str, Sequence[float]]] = None, meta: Optional[Dict[str, Any]] = None):
        self.params = {level: [float(v) for v in (params or DEFAULT_PARAMS)[level]] for level in LEVELS}
        self.meta = meta or {}
        self._curves = {level: trapmf(GRID, *p) for level, p in self.params.items()}

    @classmethod
    def from_file(cls, path: str = "config/calibration.yaml") -> Optional["RiskCalibrator"]:
        """:return: The fitted calibrator, or None when no calibration was fitted yet"""
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data["params"], {k: v for k, v in data.items() if k != "params"})

    def risk(self, distribution: Dict[str, float]) -> float:
        """:return: r_LLM ∈ [0, 1] — centroid of Σ p_t · μ_t"""
        aggregate = sum(float(distribution.get(level, 0.0)) * self._curves[level] for level in LEVELS)
        total = float(np.sum(aggregate))
        if total <= 1e-12:
            return float(sum(float(distribution.get(level, 0.0)) * FIXED[level] for level in LEVELS))
        return round(float(np.sum(GRID * aggregate) / total), 4)

    def risks(self, matrix: np.ndarray) -> np.ndarray:
        """Vectorised ``risk`` for an (n, 4) matrix of distributions (columns in LEVELS order) — used by the fit."""
        curves = np.vstack([self._curves[level] for level in LEVELS])           # (4, grid)
        aggregate = matrix @ curves                                                # (n, grid)
        total = aggregate.sum(axis=1)
        fallback = matrix @ np.array([FIXED[level] for level in LEVELS])
        with np.errstate(invalid="ignore", divide="ignore"):
            centroid = (aggregate @ GRID) / total
        return np.where(total > 1e-12, centroid, fallback)

    def to_yaml(self, path: str, extra: Dict[str, Any]) -> None:
        data = {"version": 1, "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **extra,
                "params": {level: [round(v, 4) for v in p] for level, p in self.params.items()}}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("# F18 (TOD-29): fitted output membership functions of the LLM risk levels "
                    "(python -m tools.calibrate_llm_risk)\n")
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def fixed_risk(distribution: Dict[str, float], level: Optional[str] = None) -> float:
    """Baseline: the fixed number of the chosen level (argmax of the distribution when no level is given)."""
    level = level or max(LEVELS, key=lambda name: distribution.get(name, 0.0))
    return FIXED[level]


# --- metrics ----------------------------------------------------------------------------------------------------

def huber(pred: Sequence[float], target: Sequence[float], delta: float = HUBER_DELTA) -> float:
    """Mean Huber loss (×2, so it equals the squared error below ``delta``)"""
    r = np.abs(np.asarray(pred, float) - np.asarray(target, float))
    return float(np.mean(np.where(r <= delta, r ** 2, 2 * delta * r - delta ** 2))) if len(r) else 0.0


def brier(pred: Sequence[float], target: Sequence[float]) -> float:
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    return float(np.mean((pred - target) ** 2)) if len(pred) else 0.0


def reliability(pred: Sequence[float], target: Sequence[float], bins: int = 10) -> List[Dict[str, Any]]:
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    index = np.minimum((pred * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        mask = index == b
        if mask.any():
            out.append({"bin": [round(b / bins, 2), round((b + 1) / bins, 2)], "n": int(mask.sum()),
                        "mean_pred": round(float(pred[mask].mean()), 4),
                        "mean_target": round(float(target[mask].mean()), 4)})
    return out


def ece(pred: Sequence[float], target: Sequence[float], bins: int = 10) -> float:
    n = len(pred)
    return float(sum(r["n"] / n * abs(r["mean_pred"] - r["mean_target"]) for r in reliability(pred, target, bins))) \
        if n else 0.0


# --- fitting ----------------------------------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _unpack(theta: np.ndarray, min_width: float = MIN_WIDTH) -> Dict[str, List[float]]:
    """
    Unconstrained vector → ordered trapezoids: each centre inside its ``BOUNDS`` and ≥ ``MIN_GAP`` above the previous
    one, plateau / slopes ≥ 0 (slope ≤ 0.3), clipped to [0, 1], every support (d − a) ≥ ``min_width`` (widened away
    from the universe edge after clipping).
    """
    s = _sigmoid(np.asarray(theta, float))
    centres = []
    for i, level in enumerate(LEVELS):
        lo, hi = BOUNDS[level]
        c = lo + s[i] * (hi - lo)
        centres.append(max(c, centres[-1] + MIN_GAP) if centres else c)
    params = {}
    for i, level in enumerate(LEVELS):
        half, slope = 0.15 * s[4 + i], 0.02 + 0.28 * s[8 + i]
        slope = max(slope, min_width / 2 - half)
        c = min(1.0, centres[i])
        a, b, cc, d = (float(np.clip(v, 0, 1)) for v in (c - half - slope, c - half, c + half, c + half + slope))
        if d - a < min_width:
            if a <= 0.0:
                d = min(1.0, max(d, a + min_width))
            else:
                a = max(0.0, min(a, d - min_width))
        params[level] = [a, b, cc, d]
    return params


def check_constraints(params: Dict[str, Sequence[float]], min_width: float = MIN_WIDTH) -> Dict[str, Any]:
    """:return: {"ok", "monotone_centres", "centres", "min_support", "min_width"} of a set of level trapezoids"""
    centres = [round((float(params[level][1]) + float(params[level][2])) / 2, 4) for level in LEVELS]
    supports = [float(params[level][3]) - float(params[level][0]) for level in LEVELS]
    monotone = all(a < b for a, b in zip(centres, centres[1:], strict=False))
    bounded = all(BOUNDS[level][0] - 1e-6 <= c <= BOUNDS[level][1] + 1e-6
                  for level, c in zip(LEVELS, centres, strict=True))
    return {"ok": bool(monotone and bounded and min(supports) >= min_width - 1e-6), "monotone_centres": monotone,
            "centres_in_bounds": bounded, "centres": dict(zip(LEVELS, centres, strict=True)),
            "min_support": round(min(supports), 4), "min_width": min_width}


def fit(distributions: Sequence[Dict[str, float]], targets: Sequence[float], ece_weight: float = 0.25,
        seed: int = 0, min_width: float = MIN_WIDTH, restarts: int = 4) -> Tuple[RiskCalibrator, Dict[str, float]]:
    """
    :param distributions: The LLM's risk distributions on the calibration split
    :param targets: Expert risk per scenario, 0..1
    :param min_width: Minimum support width of every level set (constraint, with monotone centres)
    :param restarts: Random Nelder-Mead restarts besides the zero start
    :return: (fitted calibrator, {"loss", "brier", "ece"} on the calibration split)
    """
    from scipy.optimize import minimize

    targets = np.asarray(targets, float)
    matrix = np.array([[float(d.get(level, 0.0)) for level in LEVELS] for d in distributions])

    def loss(theta):
        pred = RiskCalibrator(_unpack(theta, min_width)).risks(matrix)
        return huber(pred, targets) + ece_weight * ece(pred, targets)

    rng = np.random.default_rng(seed)
    best = None
    for start in [np.zeros(12)] + [rng.normal(0, 1, 12) for _ in range(restarts)]:
        result = minimize(loss, start, method="Nelder-Mead", options={"maxiter": 1500, "xatol": 1e-4, "fatol": 1e-6})
        if best is None or result.fun < best.fun:
            best = result
    calibrator = RiskCalibrator(_unpack(best.x, min_width), {"min_width": min_width})
    pred = [calibrator.risk(d) for d in distributions]
    return calibrator, {"loss": round(float(best.fun), 5), "brier": round(brier(pred, targets), 5),
                        "ece": round(ece(pred, targets), 5)}


def loo_cv(distributions: Sequence[Dict[str, float]], targets: Sequence[float], ece_weight: float = 0.25,
           min_width: float = MIN_WIDTH, restarts: int = 1) -> Dict[str, Any]:
    """
    Leave-one-out: every scenario is predicted by a fit on all the others (same constraints).

    :return: {"n", "brier", "ece", "fixed": {"brier", "ece"} of the fixed mapping, "predictions"}
    """
    targets = list(map(float, targets))
    predictions = []
    for i in range(len(distributions)):
        rest = [d for j, d in enumerate(distributions) if j != i]
        calibrator, _ = fit(rest, [y for j, y in enumerate(targets) if j != i], ece_weight, i, min_width, restarts)
        predictions.append(calibrator.risk(distributions[i]))
    fixed = [fixed_risk(d) for d in distributions]
    return {"n": len(targets), "brier": round(brier(predictions, targets), 5),
            "ece": round(ece(predictions, targets), 5),
            "fixed": {"brier": round(brier(fixed, targets), 5), "ece": round(ece(fixed, targets), 5)},
            "predictions": [round(p, 4) for p in predictions]}


def evaluate(calibrator: RiskCalibrator, distributions: Sequence[Dict[str, float]], targets: Sequence[float],
             levels: Optional[Sequence[Optional[str]]] = None) -> Dict[str, Any]:
    """:return: Brier / ECE / reliability of the fixed mapping vs the calibrated risk"""
    levels = levels or [None] * len(distributions)
    fixed = [fixed_risk(d, level) for d, level in zip(distributions, levels, strict=True)]
    calibrated = [calibrator.risk(d) for d in distributions]
    return {"n": len(targets),
            "fixed": {"brier": round(brier(fixed, targets), 5), "ece": round(ece(fixed, targets), 5),
                      "reliability": reliability(fixed, targets)},
            "calibrated": {"brier": round(brier(calibrated, targets), 5), "ece": round(ece(calibrated, targets), 5),
                           "reliability": reliability(calibrated, targets)}}


def dataset_hash(rows: Iterable[Tuple[str, Dict[str, float], float]]) -> str:
    digest = hashlib.sha256()
    for snapshot_id, distribution, target in sorted(rows, key=lambda r: r[0]):
        digest.update(f"{snapshot_id}|{[round(distribution.get(k, 0), 4) for k in LEVELS]}|{target:.4f}\n".encode())
    return digest.hexdigest()


def synthetic(n: int = 400, seed: int = 0, overconfidence: float = 2.5) -> List[Tuple[str, Dict[str, float], float]]:
    """
    A labelled set imitating an overconfident LLM (tests / demo — not thesis evidence): the true risk y is drawn,
    the model's distribution peaks too sharply around a level whose fixed number is biased upwards for mid risks.
    """
    rng = np.random.default_rng(seed)
    centres = np.array([0.1, 0.4, 0.7, 0.95])
    rows = []
    for i in range(n):
        y = float(np.clip(rng.beta(1.4, 2.2), 0, 1))
        noisy = np.clip(y + rng.normal(0, 0.08), 0, 1)
        logits = -overconfidence * 12 * (centres - (0.15 + 0.85 * noisy)) ** 2
        p = np.exp(logits - logits.max())
        p /= p.sum()
        rows.append((f"syn-{i:04d}", {level: round(float(v), 4) for level, v in zip(LEVELS, p, strict=True)}, y))
    return rows
