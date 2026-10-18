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

def _unpack(theta: np.ndarray) -> Dict[str, List[float]]:
    """Unconstrained vector → ordered trapezoids: centres increase, plateau / slopes ≥ 0, clipped to [0, 1]."""
    softplus = np.log1p(np.exp(theta))
    centres = np.cumsum(softplus[:4]) / (1.0 + np.cumsum(softplus[:4])[-1])        # strictly increasing in (0, 1)
    params = {}
    for i, level in enumerate(LEVELS):
        half, slope = 0.15 * softplus[4 + i] / (1 + softplus[4 + i]), 0.02 + softplus[8 + i] * 0.1
        c = centres[i]
        params[level] = [float(np.clip(v, 0, 1)) for v in (c - half - slope, c - half, c + half, c + half + slope)]
    return params


def fit(distributions: Sequence[Dict[str, float]], targets: Sequence[float], ece_weight: float = 0.25,
        seed: int = 0) -> Tuple[RiskCalibrator, Dict[str, float]]:
    """
    :param distributions: The LLM's risk distributions on the calibration split
    :param targets: Expert risk per scenario, 0..1
    :return: (fitted calibrator, {"loss", "brier", "ece"} on the calibration split)
    """
    from scipy.optimize import minimize

    targets = np.asarray(targets, float)
    matrix = np.array([[float(d.get(level, 0.0)) for level in LEVELS] for d in distributions])

    def loss(theta):
        pred = RiskCalibrator(_unpack(theta)).risks(matrix)
        return brier(pred, targets) + ece_weight * ece(pred, targets)

    rng = np.random.default_rng(seed)
    best = None
    for start in [np.zeros(12)] + [rng.normal(0, 1, 12) for _ in range(4)]:
        result = minimize(loss, start, method="Nelder-Mead", options={"maxiter": 1500, "xatol": 1e-4, "fatol": 1e-6})
        if best is None or result.fun < best.fun:
            best = result
    calibrator = RiskCalibrator(_unpack(best.x))
    pred = [calibrator.risk(d) for d in distributions]
    return calibrator, {"loss": round(float(best.fun), 5), "brier": round(brier(pred, targets), 5),
                        "ece": round(ece(pred, targets), 5)}


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
