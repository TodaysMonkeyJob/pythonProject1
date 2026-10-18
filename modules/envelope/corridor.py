"""
Fuzzy risk corridor (F17 / TOD-28): ``[r_floor, r_ceil] = FIS({e_i : τ(e_i) ≥ θ})``.

The risk rule base of ``core.fuzzy`` (TOD-26, with the location's overrides) is evaluated on trusted evidence only:
``r_floor`` = the mean of maxima (MOM) of the aggregated output set — the most supported conclusion — and ``r_ceil``
= the upper end of its α-cut (informational: an LLM risk far above it is flagged as a possible over-reaction, but
escalation is never capped). MOM instead of the centroid because the centroid dilutes a fully fired severe rule
("gas dangerous → critical") with co-firing milder ones ("stranger in daytime → medium") — found by the TOD-31
harness (gas 6.5 % + a stranger: centroid 0.65 = high, MOM 0.95 = critical). ``envelope.floor`` = centroid keeps
the old behaviour (ablation).
"""
from typing import Any, Dict, Optional

from core.fuzzy import FuzzyEngine


class Corridor:
    def __init__(self, path: str = "config/fuzzy.yaml", locations_path: Optional[str] = "config/locations.yaml",
                 alpha: float = 0.5, floor: str = "mom"):
        if floor not in ("mom", "centroid"):
            raise ValueError(f"envelope.floor must be mom or centroid, got {floor!r}")
        self.path, self.locations_path, self.alpha, self.floor = path, locations_path, alpha, floor
        self._engines: Dict[str, FuzzyEngine] = {}

    def engine(self, location_id: Optional[str]) -> FuzzyEngine:
        key = location_id or ""
        if key not in self._engines:
            self._engines[key] = FuzzyEngine.from_config(self.path, location_id or None, self.locations_path)
        return self._engines[key]

    def compute(self, location_id: Optional[str], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """:return: {"floor", "ceil", "fired_rules", "inputs"} for trusted FIS inputs of one location"""
        result = self.engine(location_id).infer(inputs, alpha=self.alpha)
        floor = float(result["mom"] if self.floor == "mom" else result["crisp"])
        ceil = max(floor, float(result["cut"][1])) if result["cut"] else floor
        return {"floor": round(floor, 3), "ceil": round(ceil, 3), "fired_rules": result["fired_rules"],
                "inputs": {k: v["value"] for k, v in result["inputs"].items()}}
