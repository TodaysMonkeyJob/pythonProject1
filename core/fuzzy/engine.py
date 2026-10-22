"""
Mamdani fuzzy inference engine (F15 / TOD-26) — the shared core of the fuzzy layers around the LLM (trust in evidence
TOD-27, risk corridor TOD-28). Deterministic, numpy only, no network, runs on the hub / Pi in well under 5 ms.

Configuration: ``config/fuzzy.yaml`` (variables, terms, rules, operators, defuzzification) with per-location
overrides from its ``locations:`` section and from ``fuzzy:`` blocks in ``config/locations.yaml``::

    locations:
      warehouse:
        terms: {time_of_day_risk: {night: [[trap, 0, 0, 6, 7], [trap, 19, 20.5, 24, 24]]}}   # replace / add terms
        shift: {gas_concentration: {dangerous: -1.0}}     # move a term (edges of the universe stay put)
        rules: ["IF unknown_share IS some AND time_of_day_risk IS off_hours THEN risk IS high"]
        remove_rules: [R14]
        weights: {R16: 0.8}

Usage::

    engine = FuzzyEngine.from_config("config/fuzzy.yaml", location_id="warehouse")
    result = engine.infer({"gas_concentration": 4.2, "gas": "acetone", "time_of_day_risk": 23.5})
    result["crisp"], result["fired_rules"][0]["explanation"]

Measured values (F22 / TOD-33): a string ``"@<name>:<dotted.path>"`` anywhere in the file is replaced on load by that
value of ``<name>.yaml`` next to it (``@thresholds:envelope.theta`` → ``config/thresholds.yaml``), so every trust
degree / threshold names the file that measured or fitted it. A missing file / key fails the load.
"""
import copy
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml

from core.fuzzy.membership import shift as shift_term
from core.fuzzy.rules import Rule, parse_rule
from core.fuzzy.validate import FuzzyConfigError, check
from core.fuzzy.variables import LinguisticVariable

DEFUZZIFIERS = ("centroid", "bisector", "mom", "som", "lom")
MIN_STRENGTH = 1e-6


REF_PREFIX = "@"


class MeasuredRefError(ValueError):
    pass


def resolve_refs(data: Any, base_dir: str, _cache: Optional[Dict[str, Any]] = None) -> Any:
    """:return: ``data`` with every ``"@<name>:<dotted.path>"`` string replaced by the referenced value"""
    cache = {} if _cache is None else _cache
    if isinstance(data, dict):
        return {k: resolve_refs(v, base_dir, cache) for k, v in data.items()}
    if isinstance(data, list):
        return [resolve_refs(v, base_dir, cache) for v in data]
    if not (isinstance(data, str) and data.startswith(REF_PREFIX) and ":" in data):
        return data
    name, dotted = data[1:].split(":", 1)
    path = os.path.join(base_dir, f"{name}.yaml")
    if path not in cache:
        if not os.path.isfile(path):
            raise MeasuredRefError(f"{data}: {path} not found (run the tool that writes it, see core/calibration.py)")
        with open(path, encoding="utf-8") as f:
            cache[path] = yaml.safe_load(f) or {}
    value: Any = cache[path]
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise MeasuredRefError(f"{data}: no {key!r} in {path}")
        value = value[key]
    return value


def load_yaml(path: str) -> Dict[str, Any]:
    """Reads a fuzzy config (or locations.yaml) and resolves its measured-value references."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return resolve_refs(data, os.path.dirname(os.path.abspath(path)))


def location_overrides(data: Dict[str, Any], location_id: Optional[str],
                       locations_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """:return: The override blocks for a location: ``fuzzy.yaml`` → ``locations:`` first, then ``locations.yaml``"""
    if not location_id:
        return []
    blocks = []
    own = (data.get("locations") or {}).get(location_id)
    if own:
        blocks.append(own)
    if locations_path and os.path.isfile(locations_path):
        for entry in load_yaml(locations_path).get("locations") or []:
            if str(entry.get("id")) == location_id and entry.get("fuzzy"):
                blocks.append(entry["fuzzy"])
    return blocks


class FuzzyEngine:
    def __init__(self, variables: Dict[str, LinguisticVariable], output: str, rules: List[Rule],
                 defuzzification: str = "centroid", operators: Optional[Dict[str, str]] = None,
                 default_output: float = 0.0, location_id: Optional[str] = None,
                 maps: Optional[Dict[str, Dict[str, float]]] = None):
        ops = {"and": "min", "or": "max", "implication": "min", "aggregation": "max", **(operators or {})}
        errors = check(variables, output, rules)
        if defuzzification not in DEFUZZIFIERS:
            errors.append(f"defuzzification must be one of {DEFUZZIFIERS}, got {defuzzification!r}")
        for key, allowed in (("and", ("min", "prod")), ("or", ("max", "probor")), ("implication", ("min", "prod")),
                             ("aggregation", ("max", "sum"))):
            if ops[key] not in allowed:
                errors.append(f"operators.{key} must be one of {allowed}, got {ops[key]!r}")
        if errors:
            raise FuzzyConfigError(errors, location_id or "base")
        self.variables, self.output, self.rules = variables, output, rules
        self.defuzzification, self.operators = defuzzification, ops
        self.default_output = float(default_output)
        self.location_id = location_id
        self.maps = maps or {}
        self._grid, self._curves = variables[output].curves()

    # --- construction ----------------------------------------------------------------------------------------------

    @classmethod
    def from_config(cls, path: str = "config/fuzzy.yaml", location_id: Optional[str] = None,
                    locations_path: Optional[str] = "config/locations.yaml") -> "FuzzyEngine":
        """
        :param location_id: Apply this location's overrides (None = the base rule base)
        :param locations_path: ``locations.yaml`` with optional ``fuzzy:`` blocks per location (None = ignore)
        """
        data = load_yaml(path)
        return cls.from_dict(data, location_id, location_overrides(data, location_id, locations_path))

    @classmethod
    def from_dict(cls, data: Dict[str, Any], location_id: Optional[str] = None,
                  overrides: Optional[List[Dict[str, Any]]] = None) -> "FuzzyEngine":
        data = copy.deepcopy(data)
        variables_cfg = data.get("variables") or {}
        shifts: List[tuple] = []
        extra_rules: List[tuple] = []
        removed, weights = set(), {}
        for block in overrides or []:
            for var, terms in (block.get("terms") or {}).items():
                if var not in variables_cfg:
                    raise FuzzyConfigError([f"override for unknown variable {var!r}"], location_id or "")
                variables_cfg[var]["terms"].update(terms)
            shifts += [(var, term, float(delta)) for var, terms in (block.get("shift") or {}).items()
                       for term, delta in terms.items()]
            extra_rules += [(spec, f"location:{location_id}") for spec in block.get("rules") or []]
            removed |= set(block.get("remove_rules") or [])
            weights.update(block.get("weights") or {})
        try:
            variables = {name: LinguisticVariable.from_dict(name, cfg) for name, cfg in variables_cfg.items()}
        except (KeyError, TypeError, ValueError) as e:
            raise FuzzyConfigError([f"variables: {e}"], location_id or "base") from e
        errors = []
        for var, term, delta in shifts:
            variable = variables.get(var)
            if variable is None or term not in variable.terms:
                errors.append(f"shift: unknown term {var}.{term}")
                continue
            variable.terms[term] = shift_term(variable.terms[term], delta, variable.low, variable.high)
        parsed: List[Rule] = []
        specs = [(spec, "base") for spec in data.get("rules") or []] + extra_rules
        for index, (spec, source) in enumerate(specs, start=1):
            try:
                parsed.append(parse_rule(spec, index, source))
            except ValueError as e:
                errors.append(f"rule {index}: {e}")
        known_ids = {rule.id for rule in parsed}
        unknown_ids = (removed | set(weights)) - known_ids
        errors += [f"override refers to unknown rule id {rid!r}" for rid in sorted(unknown_ids)]
        rules = [rule for rule in parsed if rule.id not in removed]
        for rule in rules:
            if rule.id in weights:
                rule.weight = float(weights[rule.id])
        if errors:
            raise FuzzyConfigError(errors, location_id or "base")
        return cls(variables, data.get("output", "risk"), rules, data.get("defuzzification", "centroid"),
                   data.get("operators"), data.get("default_output", 0.0), location_id, data.get("maps"))

    # --- inference -------------------------------------------------------------------------------------------------

    def fuzzify(self, variable: str, value: float, variant: Optional[str] = None) -> Dict[str, float]:
        """:return: {term: μ} — also used for the linguistic terms in prompts (TOD-29)"""
        return self.variables[variable].fuzzify(value, variant)

    def infer(self, inputs: Dict[str, Any], alpha: float = 0.5) -> Dict[str, Any]:
        """
        :param inputs: {variable: crisp value} plus selectors (e.g. ``"gas": "acetone"`` picks that gas's terms).
                       A missing variable uses its ``default``; without one, rules that need it are skipped.
        :return: {"crisp", "memberships" (output term → activation), "fired_rules" [{id, rule, strength, weight,
                 term, source, explanation}] strongest first, "inputs" {variable: {value, terms, defaulted}},
                 "skipped_rules", "location_id", "cut" [lo, hi] of the aggregated output set at ``alpha`` × its height
                 (None when nothing fired) — the F17 corridor uses its upper end as ``r_ceil``, "mom" the mean of
                 maxima of that set (F17 floor)}
        """
        fuzzified: Dict[str, Dict[str, float]] = {}
        described: Dict[str, Dict[str, Any]] = {}
        for name, variable in self.variables.items():
            if name == self.output:
                continue
            raw, defaulted = inputs.get(name), False
            if raw is None:
                if variable.default is None:
                    continue
                raw, defaulted = variable.default, True
            variant = inputs.get(variable.variants_by) if variable.variants_by else None
            value = variable.clamp(float(raw))
            fuzzified[name] = variable.fuzzify(value, variant)
            described[name] = {"value": value, "terms": fuzzified[name], "defaulted": defaulted,
                               **({"variant": variant} if variant else {})}
        activation = {term: 0.0 for term in self._curves}
        fired, skipped = [], []
        aggregate = np.zeros_like(self._grid)
        for rule in self.rules:
            if not rule.variables() <= fuzzified.keys():
                skipped.append(rule.id)
                continue
            strength = rule.strength(fuzzified, self.operators["and"], self.operators["or"])
            if strength <= MIN_STRENGTH:
                continue
            activation[rule.term] = max(activation[rule.term], strength)
            if self.operators["aggregation"] == "sum":
                aggregate += self._implicate(rule.term, strength)
            fired.append({"id": rule.id, "rule": rule.text, "strength": round(strength, 4),
                          "weight": rule.weight, "term": rule.term, "source": rule.source,
                          "explanation": self._explain(rule, described, strength)})
        if self.operators["aggregation"] == "max":
            for term, strength in activation.items():
                if strength > MIN_STRENGTH:
                    aggregate = np.maximum(aggregate, self._implicate(term, strength))
        else:
            aggregate = np.minimum(aggregate, 1.0)
        crisp = self._defuzzify(aggregate)
        fired.sort(key=lambda r: -r["strength"])
        cut, mom = None, self.default_output
        if aggregate.max() > MIN_STRENGTH:
            inside = np.flatnonzero(aggregate >= alpha * aggregate.max() - 1e-9)
            cut = [round(float(self._grid[inside[0]]), 4), round(float(self._grid[inside[-1]]), 4)]
            mom = float(self._grid[aggregate >= aggregate.max() - 1e-9].mean())
        return {"crisp": round(crisp, 4), "memberships": {t: round(v, 4) for t, v in activation.items()},
                "fired_rules": fired, "inputs": described, "skipped_rules": skipped, "location_id": self.location_id,
                "cut": cut, "mom": round(mom, 4)}

    def _implicate(self, term: str, strength: float) -> np.ndarray:
        curve = self._curves[term]
        return np.minimum(curve, strength) if self.operators["implication"] == "min" else curve * strength

    def _defuzzify(self, mu: np.ndarray) -> float:
        x = self._grid
        total = float(mu.sum())
        if total <= MIN_STRENGTH:
            return self.default_output
        if self.defuzzification == "centroid":
            return float((x * mu).sum() / total)
        if self.defuzzification == "bisector":
            cumulative = np.cumsum(mu)
            return float(x[int(np.searchsorted(cumulative, total / 2))])
        peak = np.flatnonzero(mu >= mu.max() - 1e-9)
        return float({"mom": x[peak].mean(), "som": x[peak[0]], "lom": x[peak[-1]]}[self.defuzzification])

    def _explain(self, rule: Rule, described: Dict[str, Dict[str, Any]], strength: float) -> str:
        """``gas_concentration = 4.2 % → dangerous 1.00; … ⇒ risk IS critical (0.82)``"""
        parts = []
        for var, term, negated in dict.fromkeys(rule.literals()):
            info = described[var]
            unit = f" {self.variables[var].unit}" if self.variables[var].unit else ""
            default = " (default)" if info["defaulted"] else ""
            mu = info["terms"][term]
            label = f"NOT {term} {1 - mu:.2f}" if negated else f"{term} {mu:.2f}"
            parts.append(f"{var} = {info['value']:g}{unit}{default} → {label}")
        weight = f", weight {rule.weight:g}" if rule.weight != 1.0 else ""
        return f"{'; '.join(parts)} ⇒ {self.output} IS {rule.term} ({strength:.2f}{weight})"

    # --- helpers for callers (TOD-27 / TOD-28) ---------------------------------------------------------------------

    def map_value(self, name: str, keys: Iterable[str], default: float = 0.0) -> float:
        """:return: The largest mapped value of the keys, e.g. ``action_risk`` of the current actions"""
        table = self.maps.get(name) or {}
        values = [float(table[k]) for k in keys if k in table]
        return max(values) if values else default
