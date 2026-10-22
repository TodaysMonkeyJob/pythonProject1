"""
Two-sided fuzzy corridor (F25 / TOD-36): GPT's fuzzy answer is intersected with the corridor; an agreement degree h
decides whether GPT refines the risk inside the corridor or a bound holds.

* corridor set K over risk x ∈ [0, 1]: plateau 1 on [L, U], linear slopes of width ``slope`` outside (the mean edge
  width of the risk output terms — the same granularity the rule base speaks in), with the bounds
  ``L = (1 − c)·(y_l + λ·(y_r − y_l))`` (risk attitude λ, confirmation c) and ``U = y_r`` from the IT2 corridor
  (TOD-35);
* GPT set: ``μ_LLM(x) = max_t p_t · μ_t(x)`` — its risk_distribution weighting the calibrated level sets
  (``modules.llm.calibration``; without a distribution the stated level gets p = 1); ``r_llm`` = its centroid;
* decision set ``μ_D = min(μ_LLM, μ_K)``, ``r_D`` = its centroid kept inside [L, U] (the slopes of K may pull the raw
  centroid ``r_D_raw`` slightly outside), agreement ``h = sup μ_D`` ∈ [0, 1];
* policy: h ≥ h_min → ``agree`` (r_final = r_D, kept inside [L, U]); else GPT below the corridor →
  ``deescalation_rejected`` (r_final = L); GPT above → ``escalation_review`` (r_final = U, the operator may accept
  GPT's level); a disagreement neither below nor above (a split distribution) → ``clamp`` (r_llm into [L, U]);
  no GPT answer → ``fuzzy_only`` (r_final = L).

Language / scene-text claims never feed y_l, y_r or c, so no claim can push the result below L. λ and h_min come
from ``config/thresholds.yaml`` (``tools/fit_thresholds.py``). numpy only, ≈ 0.1 ms per decision.
"""
from typing import Any, Dict, Optional, Sequence

import numpy as np

from core.fuzzy.membership import trapmf

LEVELS = ("low", "medium", "high", "critical")
GRID = np.linspace(0.0, 1.0, 1001)
BRANCHES = ("agree", "deescalation_rejected", "escalation_review", "clamp", "fuzzy_only")


def centroid(mu: np.ndarray) -> Optional[float]:
    total = float(mu.sum())
    return None if total <= 1e-12 else float((GRID * mu).sum() / total)


def corridor_set(lower: float, upper: float, slope: float) -> np.ndarray:
    """μ_K: 1 on [lower, upper], falling to 0 over ``slope`` on both sides (clipped to the risk universe)."""
    lower, upper = min(lower, upper), max(lower, upper)
    return trapmf(GRID, lower - slope, lower, upper, upper + slope)


def llm_set(distribution: Optional[Dict[str, float]], level: Optional[str],
            params: Dict[str, Sequence[float]]) -> np.ndarray:
    """μ_LLM(x) = max_t p_t · μ_t(x) over the calibrated level sets (p renormalised; none → the level, p = 1)."""
    p = {t: max(0.0, float((distribution or {}).get(t, 0.0))) for t in LEVELS}
    total = sum(p.values())
    if total <= 1e-12:
        if level not in LEVELS:
            raise ValueError(f"no risk_distribution and unknown level {level!r}")
        p, total = {t: float(t == level) for t in LEVELS}, 1.0
    curves = [p[t] / total * trapmf(GRID, *params[t]) for t in LEVELS]
    return np.maximum.reduce(curves)


def bounds(y_l: float, y_r: float, confirmation: float, lam: float) -> Dict[str, float]:
    lower = (1.0 - confirmation) * (y_l + lam * (y_r - y_l))
    return {"L": round(float(lower), 4), "U": round(float(max(y_r, lower)), 4)}


def decide(y_l: float, y_r: float, confirmation: float, distribution: Optional[Dict[str, float]],
           level: Optional[str], params: Dict[str, Sequence[float]], lam: float, h_min: float,
           slope: float) -> Dict[str, Any]:
    """
    :param y_l: / y_r: the IT2 corridor interval (TOD-35)
    :param confirmation: c ∈ [0, 1] (lower end of the IT2 confirmation; operator acknowledgement = 1)
    :param distribution: GPT's risk_distribution, or None (then ``level`` alone)
    :param level: GPT's risk_level; None = no GPT answer (offline / gated) → fuzzy_only
    :return: {L, U, r_llm, r_D, h, policy_branch, r_final}
    """
    for name, value in (("y_l", y_l), ("y_r", y_r), ("confirmation", confirmation), ("lambda", lam)):
        if not 0.0 <= value <= 1.0 + 1e-9:
            raise ValueError(f"{name} must be within 0..1, got {value}")
    b = bounds(y_l, y_r, confirmation, lam)
    lower, upper = b["L"], b["U"]
    if level is None and not distribution:
        return {**b, "r_llm": None, "r_D": None, "h": None, "policy_branch": "fuzzy_only", "r_final": lower}
    mu_llm = llm_set(distribution, level, params)
    mu_d = np.minimum(mu_llm, corridor_set(lower, upper, slope))
    r_llm = centroid(mu_llm)
    h = float(mu_d.max())
    r_raw = centroid(mu_d)
    r_d = None if r_raw is None else min(upper, max(lower, r_raw))
    if h >= h_min and r_d is not None:
        branch, final = "agree", r_d
    elif r_llm < lower:
        branch, final = "deescalation_rejected", lower
    elif r_llm > upper:
        branch, final = "escalation_review", upper
    else:
        branch, final = "clamp", min(upper, max(lower, r_llm))
    return {**b, "r_llm": round(r_llm, 4), "r_D": None if r_d is None else round(r_d, 4),
            "r_D_raw": None if r_raw is None else round(r_raw, 4), "h": round(h, 4), "policy_branch": branch,
            "r_final": round(float(final), 4)}


def output_slope(terms: Dict[str, Sequence[tuple]]) -> float:
    """Mean width of the sloped edges of the risk output terms (parsed ``membership.parse_term`` parts)."""
    widths = []
    for parts in terms.values():
        for shape, p in parts:
            if shape == "gauss":
                widths.append(p[1])
                continue
            pts = (p[0], p[1], p[1], p[2]) if shape == "tri" else p
            widths += [w for w in (pts[1] - pts[0], pts[3] - pts[2]) if w > 0]
    return float(np.mean(widths)) if widths else 0.1
