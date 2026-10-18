"""
Asymmetric projection of the LLM's risk into the corridor (F17 / TOD-28)::

    r_final = max(r_LLM, (1 − c) · r_floor)

Raising is always allowed (the LLM can escalate freely, ``r_ceil`` never caps it); lowering below the floor needs
trusted confirmation ``c``. Levels ↔ numbers come from ``envelope.llm_levels`` / ``envelope.thresholds`` (TOD-29 will
calibrate the mapping).
"""
from typing import Any, Dict, Optional

LEVELS = ("low", "medium", "high", "critical")
DEFAULT_LLM_LEVELS = {"low": 0.1, "medium": 0.4, "high": 0.7, "critical": 0.95}
DEFAULT_THRESHOLDS = {"medium": 0.25, "high": 0.5, "critical": 0.8}


def level_value(level: Optional[str], mapping: Optional[Dict[str, float]] = None) -> float:
    mapping = mapping or DEFAULT_LLM_LEVELS
    if level not in mapping:
        raise ValueError(f"unknown risk level {level!r}")
    return float(mapping[level])


def level_of(risk: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    """:return: The highest level whose threshold ``risk`` reaches"""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    level = "low"
    for name in LEVELS[1:]:
        if risk >= float(thresholds[name]) - 1e-9:
            level = name
    return level


def project(r_llm: float, floor: float, ceil: float, confirmation: float,
            thresholds: Optional[Dict[str, float]] = None, over_reaction_margin: float = 0.3) -> Dict[str, Any]:
    """
    :return: {"risk_final", "risk_level_final", "envelope_applied" (the floor raised the LLM's level),
              "bound" ((1 − c) · r_floor), "over_reaction" (LLM far above r_ceil — informational)}
    """
    for name, value in (("r_llm", r_llm), ("floor", floor), ("ceil", ceil), ("confirmation", confirmation)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be within 0..1, got {value}")
    bound = (1.0 - confirmation) * floor
    final = max(r_llm, bound)
    applied = LEVELS.index(level_of(final, thresholds)) > LEVELS.index(level_of(r_llm, thresholds))
    return {"risk_final": round(final, 3), "risk_level_final": level_of(final, thresholds),
            "envelope_applied": applied, "bound": round(bound, 3),
            "over_reaction": r_llm > ceil + over_reaction_margin}
