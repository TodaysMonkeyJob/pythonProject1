"""Linguistic variables (F15 / TOD-26): a universe (min, max, step) and named terms with membership functions."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.fuzzy.membership import evaluate, parse_term


@dataclass
class LinguisticVariable:
    name: str
    low: float
    high: float
    step: float
    terms: Dict[str, List[tuple]]                 # term → parsed parts (membership.parse_term)
    unit: str = ""
    default: Optional[float] = None               # used when an input is missing (None = rules using it are skipped)
    variants: Dict[str, Dict[str, List[tuple]]] = field(default_factory=dict)   # e.g. per gas
    variants_by: Optional[str] = None             # the input key that selects a variant ("gas")

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "LinguisticVariable":
        low, high, *rest = (float(v) for v in data["universe"])
        step = rest[0] if rest else (high - low) / 200
        variants = {key: {term: parse_term(spec) for term, spec in (value.get("terms") or value).items()}
                    for key, value in (data.get("variants") or {}).items()}
        return cls(name, low, high, step, {term: parse_term(spec) for term, spec in data["terms"].items()},
                   data.get("unit", ""), data.get("default"), variants, data.get("variants_by"))

    @property
    def grid(self) -> np.ndarray:
        return np.linspace(self.low, self.high, int(round((self.high - self.low) / self.step)) + 1)

    def term_set(self, variant: Optional[str] = None) -> Dict[str, List[tuple]]:
        """:return: The terms of a variant (its own definitions on top of the base ones)"""
        if variant and variant in self.variants:
            return {**self.terms, **self.variants[variant]}
        return self.terms

    def clamp(self, value: float) -> float:
        return float(min(self.high, max(self.low, value)))

    def fuzzify(self, value: float, variant: Optional[str] = None) -> Dict[str, float]:
        """:return: {term: μ} of a crisp value (clamped into the universe)"""
        x = self.clamp(float(value))
        return {term: round(float(evaluate(parts, x)), 6) for term, parts in self.term_set(variant).items()}

    def curves(self, variant: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """:return: (grid, {term: μ over the grid}) — output aggregation, coverage checks, plots"""
        grid = self.grid
        return grid, {term: evaluate(parts, grid) for term, parts in self.term_set(variant).items()}
