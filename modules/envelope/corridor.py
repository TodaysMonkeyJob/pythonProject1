"""
Fuzzy risk corridor (F17 / TOD-28, interval type-2 since F24 / TOD-35): ``[r_floor, r_ceil] = [y_l, y_r]`` of the
risk rule base evaluated by ``core.fuzzy.engine2.FuzzyEngine2`` on ALL physical evidence of the location.

Each input's footprint of uncertainty is widened by δ = k_var · u (u = the trust registry's health-discounted
uncertainty of the evidence behind it, ``TrustedState.features`` → ``facts["uncertainty"]``): uncertain evidence no
longer disappears behind a θ cut, it makes the corridor wider. Karnik–Mendel centre-of-sets gives [y_l, y_r];
``envelope.floor``:

* ``it2`` (default) — r_floor = y_l, r_ceil = y_r;
* ``it2_mom`` — r_floor = max(y_l, type-1 MOM of the lower firing): a fully fired severe rule is not diluted by
  co-firing milder ones (the TOD-31 finding: gas 6.5 % + a stranger → critical, not high);
* ``mom`` — the type-1 F17 corridor (MOM floor, α-cut ceiling; uncertainty ignored) — ablation.
"""
from typing import Any, Dict, Optional

from core.fuzzy import FuzzyEngine
from core.fuzzy.engine2 import FuzzyEngine2

FLOORS = ("it2", "it2_mom", "mom")


class Corridor:
    def __init__(self, path: str = "config/fuzzy.yaml", locations_path: Optional[str] = "config/locations.yaml",
                 alpha: float = 0.5, floor: str = "it2"):
        if floor not in FLOORS:
            raise ValueError(f"envelope.floor must be one of {FLOORS}, got {floor!r}")
        self.path, self.locations_path, self.alpha, self.floor = path, locations_path, alpha, floor
        self._engines: Dict[str, FuzzyEngine2] = {}
        self.k_override: Optional[Dict[str, float]] = None      # tools/fit_thresholds.py tries candidate k_var

    def it2(self, location_id: Optional[str]) -> FuzzyEngine2:
        key = location_id or ""
        if key not in self._engines:
            self._engines[key] = FuzzyEngine2.from_config(self.path, location_id or None, self.locations_path)
            if self.k_override is not None:
                self._engines[key].k = dict(self.k_override)
        return self._engines[key]

    def engine(self, location_id: Optional[str]) -> FuzzyEngine:
        """The type-1 engine of the location (fuzzified terms for the LLM view, readable fired rules)."""
        return self.it2(location_id).base

    def compute(self, location_id: Optional[str], inputs: Dict[str, Any],
                uncertainty: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        :param uncertainty: {FIS variable: u ∈ [0, 1]} (None = all certain → the type-1 centre of sets)
        :return: {"floor", "ceil", "width", "type1", "widened_by", "fired_rules", "inputs", "uncertainty"}
        """
        result = self.engine(location_id).infer(inputs, alpha=self.alpha)
        it2 = self.it2(location_id).infer(inputs, uncertainty or {})
        y_l, y_r = it2["interval"]
        if self.floor == "mom":
            floor = float(result["mom"])
            ceil = max(floor, float(result["cut"][1])) if result["cut"] else floor
        else:
            floor, ceil = float(y_l), float(y_r)
            if self.floor == "it2_mom":
                floor = max(floor, self._lower_mom(it2))
                ceil = max(ceil, floor)
        return {"floor": round(floor, 3), "ceil": round(ceil, 3), "width": round(ceil - floor, 3),
                "interval": [round(y_l, 3), round(y_r, 3)], "type1": it2["type1"]["crisp"],
                "widened_by": it2["widened_by"], "fired_rules": result["fired_rules"],
                "inputs": {k: v["value"] for k, v in result["inputs"].items()},
                "uncertainty": {k: round(float(v), 3) for k, v in (uncertainty or {}).items()}}

    @staticmethod
    def _lower_mom(it2: Dict[str, Any]) -> float:
        """Centroid of the consequent with the strongest LOWER firing — the most certain conclusion."""
        best = max(it2["firing"], key=lambda f: f["lo"], default=None)
        return float(best["centroid"]) if best and best["lo"] > 0 else 0.0
