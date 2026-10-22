"""
Interval type-2 Mamdani engine (F24 / TOD-35): the type-1 rule base of ``FuzzyEngine`` with trust-driven
footprints of uncertainty (FOU) on the inputs and Karnik–Mendel centre-of-sets type reduction.

* every input term gets UMF / LMF (``it2.term_interval``) with δ_var = k_var · u_var — u_var ∈ [0, 1] the uncertainty
  of the evidence behind that input (the trust registry, TOD-34: u = 1 − h·(1 − u_counts)), k_var (variable units per
  unit of u) from ``it2.k`` in ``config/fuzzy.yaml`` (fitted by ``tools/fit_thresholds.py``); u = 0 → type-1;
* firing intervals [f̲, f̄]: AND / OR = the t-norm / s-norm on both bounds, NOT [1 − f̄, 1 − f̲], weight × both;
* centre-of-sets type reduction on the consequent term centroids (one entry per fired rule) → [y_l, y_r] (KM);
  ``type1`` = the same reduction with type-1 firing (Σ f·c / Σ f);
* explanation: the inputs ranked by how much the interval shrinks when their δ is set to 0.

``FuzzyEngine2.from_config(path, location_id)`` takes the same overrides as ``FuzzyEngine``. ≈ 0.1 ms per inference
on the dev Mac without the explanation (target ≤ 10 ms on the Pi with it).
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.fuzzy.engine import FuzzyEngine, load_yaml, location_overrides
from core.fuzzy.it2 import Interval, cos_interval, term_interval
from core.fuzzy.membership import evaluate as membership
from core.fuzzy.rules import S_NORMS, T_NORMS, Node

MIN_UPPER = 1e-6


def interval_strength(node: Node, fuzzified: Dict[str, Dict[str, Interval]], t_norm, s_norm) -> Interval:
    kind = node[0]
    if kind == "is":
        return fuzzified[node[1]][node[2]]
    if kind == "not":
        lo, hi = interval_strength(node[1], fuzzified, t_norm, s_norm)
        return 1.0 - hi, 1.0 - lo
    values = [interval_strength(child, fuzzified, t_norm, s_norm) for child in node[1]]
    combine = t_norm if kind == "and" else s_norm
    lo, hi = values[0]
    for v_lo, v_hi in values[1:]:
        lo, hi = combine(lo, v_lo), combine(hi, v_hi)
    return lo, hi


class FuzzyEngine2:
    def __init__(self, engine: FuzzyEngine, k: Optional[Dict[str, float]] = None):
        """
        :param engine: A type-1 engine (variables, rules, operators, location overrides already applied)
        :param k: δ per unit of uncertainty for each input variable (missing = 0: that input stays type-1)
        """
        self.base = engine
        self.k = {name: float(v) for name, v in (k or {}).items()}
        self.t_norm = T_NORMS[engine.operators["and"]]
        self.s_norm = S_NORMS[engine.operators["or"]]
        grid, curves = engine.variables[engine.output].curves()
        self.centroids = {term: float((grid * mu).sum() / mu.sum()) if mu.sum() > 0 else float(grid.mean())
                          for term, mu in curves.items()}

    @classmethod
    def from_config(cls, path: str = "config/fuzzy.yaml", location_id: Optional[str] = None,
                    locations_path: Optional[str] = "config/locations.yaml") -> "FuzzyEngine2":
        data = load_yaml(path)
        engine = FuzzyEngine.from_dict(data, location_id, location_overrides(data, location_id, locations_path))
        return cls(engine, (data.get("it2") or {}).get("k"))

    @classmethod
    def from_dict(cls, data: Dict[str, Any], k: Optional[Dict[str, float]] = None) -> "FuzzyEngine2":
        return cls(FuzzyEngine.from_dict(data), k if k is not None else (data.get("it2") or {}).get("k"))

    @property
    def variables(self):
        return self.base.variables

    def deltas(self, uncertainty: Optional[Dict[str, float]]) -> Dict[str, float]:
        """δ_var = k_var · clip(u_var, 0, 1)"""
        return {var: self.k.get(var, 0.0) * min(1.0, max(0.0, float(u)))
                for var, u in (uncertainty or {}).items() if var in self.base.variables}

    def _fuzzify(self, inputs: Dict[str, Any], deltas: Dict[str, float]):
        fuzzified: Dict[str, Dict[str, Interval]] = {}
        described: Dict[str, Dict[str, Any]] = {}
        for name, variable in self.base.variables.items():
            if name == self.base.output:
                continue
            raw, defaulted = inputs.get(name), False
            if raw is None:
                if variable.default is None:
                    continue
                raw, defaulted = variable.default, True
            variant = inputs.get(variable.variants_by) if variable.variants_by else None
            value = variable.clamp(float(raw))
            delta = 0.0 if defaulted else deltas.get(name, 0.0)
            fuzzified[name] = {term: term_interval(parts, value, delta, variable.low, variable.high)
                               for term, parts in variable.term_set(variant).items()}
            described[name] = {"value": value, "defaulted": defaulted, "delta": round(delta, 4),
                               **({"variant": variant} if variant else {})}
        return fuzzified, described

    def _reduce(self, fuzzified: Dict[str, Dict[str, Interval]]):
        firing: List[Dict[str, Any]] = []
        for rule in self.base.rules:
            if not rule.variables() <= fuzzified.keys():
                continue
            lo, hi = interval_strength(rule.antecedent, fuzzified, self.t_norm, self.s_norm)
            lo, hi = lo * rule.weight, hi * rule.weight
            if hi <= MIN_UPPER:
                continue
            firing.append({"id": rule.id, "term": rule.term, "lo": lo, "hi": hi, "centroid": self.centroids[rule.term],
                           "rule": rule})
        interval = cos_interval([f["centroid"] for f in firing], [f["lo"] for f in firing],
                                [f["hi"] for f in firing])
        return firing, interval

    def infer(self, inputs: Dict[str, Any], uncertainty: Optional[Dict[str, float]] = None,
              explain: bool = True) -> Dict[str, Any]:
        """
        :param inputs: {variable: crisp value} (+ selectors such as ``"gas"``), as for ``FuzzyEngine.infer``
        :param uncertainty: {variable: u ∈ [0, 1]} — widens that input's FOU by k_var · u
        :param explain: rank the inputs by their contribution to the interval width (one extra pass per input)
        :return: {"interval": [y_l, y_r], "width", "type1" {crisp, firing}, "firing" [{id, term, lo, hi}],
                  "fou" {var: {term: [lmf, umf]}}, "inputs", "widened_by" [{variable, width}], "location_id"}
        """
        deltas = self.deltas(uncertainty)
        fuzzified, described = self._fuzzify(inputs, deltas)
        firing, interval = self._reduce(fuzzified)
        default = self.base.default_output
        y_l, y_r = interval if interval is not None else (default, default)
        type1 = self.type1(inputs)
        widened: List[Dict[str, Any]] = []
        width = y_r - y_l
        if explain and width > 1e-9:
            for var, delta in deltas.items():
                if delta <= 0 or var not in fuzzified:
                    continue
                reduced = self._reduce(self._fuzzify(inputs, {**deltas, var: 0.0})[0])[1]
                narrower = width - ((reduced[1] - reduced[0]) if reduced else 0.0)
                if narrower > 1e-6:
                    widened.append({"variable": var, "width": round(narrower, 4), "delta": round(delta, 4)})
            widened.sort(key=lambda w: -w["width"])
        firing.sort(key=lambda f: -f["hi"])
        return {"interval": [round(y_l, 6), round(y_r, 6)], "width": round(width, 6),
                "type1": type1, "location_id": self.base.location_id,
                "firing": [{"id": f["id"], "term": f["term"], "lo": round(f["lo"], 4), "hi": round(f["hi"], 4),
                            "centroid": round(f["centroid"], 4), "rule": f["rule"].text} for f in firing],
                "fou": {var: {t: [round(lo, 4), round(hi, 4)] for t, (lo, hi) in terms.items()}
                        for var, terms in fuzzified.items()},
                "inputs": described, "widened_by": widened}

    def type1(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Type-1 centre-of-sets: Σ f_i · c_i / Σ f_i over the rules' type-1 firing (no FOU)."""
        fuzzified, _ = self._fuzzify(inputs, {})
        firing, _ = self._reduce(fuzzified)
        total = sum(f["hi"] for f in firing)
        crisp = sum(f["hi"] * f["centroid"] for f in firing) / total if total > 0 else self.base.default_output
        return {"crisp": round(crisp, 6), "firing": {f["id"]: round(f["hi"], 6) for f in firing}}

    def fou_curves(self, variable: str, delta: float, variant: Optional[str] = None
                   ) -> Tuple[np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """:return: (grid, {term: (LMF, UMF) over the grid}) — plots / dashboard"""
        var = self.base.variables[variable]
        grid = var.grid
        out = {}
        for term, parts in var.term_set(variant).items():
            pairs = [term_interval(parts, float(x), delta, var.low, var.high) for x in grid]
            out[term] = (np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))
        return grid, out


def type1_membership(parts, x: float) -> float:
    return float(membership(parts, x))
