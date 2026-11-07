"""
Rule-base variants of the A6 (TOD-46) evaluation, derived from ``config/fuzzy/*.yaml`` — never edited by hand:

* ``gaussian(data, outputs)`` — the same rule table with every triangle / trapezoid replaced by a Gaussian of the
  same centre and the same width at μ = 0.5 (FWHM); a shoulder on the universe edge becomes a Gaussian centred on
  that edge with μ = 0.5 where the shoulder had 0.5. ``outputs=False`` converts the input terms only (the question
  of slide 8 — the shape of the input membership functions — with the output terms unchanged); ``True`` converts
  the risk terms too (their equal areas are lost as well);
* ``without_mixture(data)`` — the gas table without the mixture index (the cells of a low mixture kept, the
  variable removed): the "with vs without mixture" configuration.
"""
import copy
import math
from typing import Any, Dict, List

HALF = math.sqrt(2 * math.log(2))          # μ = 0.5 at mean ± HALF · σ


def gauss_of(spec: List[Any], low: float, high: float) -> List[Any]:
    """:return: ``[gauss, mean, sigma]`` for a ``[tri, …]`` / ``[trap, …]`` term"""
    shape, *p = spec
    a, b, c, d = (p[0], p[1], p[1], p[2]) if shape == "tri" else p
    left_half, right_half = (a + b) / 2, (c + d) / 2           # where the term crosses 0.5
    if a == b == low and c == d == high:
        return ["gauss", (low + high) / 2, (high - low)]        # the whole universe: flat-ish
    if a == b == low:                                          # left shoulder → centred on the edge
        return ["gauss", low, max(1e-6, (right_half - low) / HALF)]
    if c == d == high:
        return ["gauss", high, max(1e-6, (high - left_half) / HALF)]
    return ["gauss", (b + c) / 2, max(1e-6, (right_half - left_half) / 2 / HALF)]


def gaussian(data: Dict[str, Any], outputs: bool = True) -> Dict[str, Any]:
    out = copy.deepcopy(data)
    for name, var in out["variables"].items():
        if name == out.get("output", "risk") and not outputs:
            continue
        low, high = float(var["universe"][0]), float(var["universe"][1])
        var["terms"] = {term: gauss_of(spec, low, high) for term, spec in var["terms"].items()}
        var.setdefault("source", {})["terms"] = {"kind": "design", "text": "A6 Gaussian variant of the triangular "
                                                 "terms (same centre, same width at μ = 0.5)"}
    return out


def without_mixture(data: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(data)
    out["variables"].pop("mixture_index", None)
    rules = []
    for entry in out["rules"]:
        text = entry["rule"]
        if " AND mixture_index IS low" in text:
            rules.append({**entry, "id": entry["id"][:-1], "rule": text.replace(" AND mixture_index IS low", "")})
    out["rules"] = rules
    return out
