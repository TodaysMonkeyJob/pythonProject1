"""
Configuration checks (F15 / TOD-26), run on every load: a broken rule base must fail loudly at start-up, not produce
a silently wrong risk bound later.

* rules use only declared variables / terms, and conclude on the output variable;
* every universe is covered: no x where all terms are 0 (that input would be "invisible" to the rules);
* no duplicate term definitions inside a variable, no duplicate rule ids, no duplicate or conflicting rules (the
  same antecedent twice, with the same or a different conclusion — for conjunctions in any clause order).

F26 (TOD-37): ``subsumed()`` lists rules made redundant by a more general rule with the same conclusion (warnings;
weights may make them intentional); metadata and never-firing rules are checked by ``core.fuzzy.governance`` /
``tools/rules_gate.py``.
"""
from typing import Dict, FrozenSet, List, Optional

import numpy as np

from core.fuzzy.rules import Rule, render
from core.fuzzy.variables import LinguisticVariable

COVERAGE_EPS = 1e-6


class FuzzyConfigError(ValueError):
    def __init__(self, errors: List[str], where: str = ""):
        self.errors = errors
        super().__init__(f"Invalid fuzzy configuration{f' ({where})' if where else ''}:\n  - " + "\n  - ".join(errors))


def coverage_gaps(variable: LinguisticVariable, variant: Optional[str] = None) -> List[str]:
    """:return: Readable intervals of the universe where every term is 0"""
    grid, curves = variable.curves(variant)
    covered = np.max(np.vstack(list(curves.values())), axis=0) > COVERAGE_EPS if curves else np.zeros_like(grid, bool)
    gaps, start = [], None
    for i, ok in enumerate(covered):
        if not ok and start is None:
            start = i
        elif ok and start is not None:
            gaps.append((grid[start], grid[i - 1]))
            start = None
    if start is not None:
        gaps.append((grid[start], grid[-1]))
    unit = f" {variable.unit}" if variable.unit else ""
    return [f"{a:g}{unit}" if a == b else f"{a:g}–{b:g}{unit}" for a, b in gaps]


def conjunction(node) -> Optional[FrozenSet[tuple]]:
    """:return: the literals (var, term, negated) of a pure conjunction, None for anything with OR"""
    kind = node[0]
    if kind == "is":
        return frozenset({(node[1], node[2], False)})
    if kind == "not" and node[1][0] == "is":
        return frozenset({(node[1][1], node[1][2], True)})
    if kind == "and":
        parts = [conjunction(child) for child in node[1]]
        return None if any(p is None for p in parts) else frozenset().union(*parts)
    return None


def subsumed(rules: List[Rule]) -> List[str]:
    """:return: warnings "B is subsumed by A": A's conditions are a strict subset of B's, same conclusion"""
    warnings = []
    literals = [(rule, conjunction(rule.antecedent)) for rule in rules]
    for a, la in literals:
        for b, lb in literals:
            if a is b or la is None or lb is None or a.term != b.term or a.output != b.output:
                continue
            if la < lb and a.weight >= b.weight:
                warnings.append(f"{b.id} is subsumed by {a.id} (same conclusion {a.output} IS {a.term}, "
                                f"{a.id} needs fewer conditions)")
    return warnings


def check(variables: Dict[str, LinguisticVariable], output: str, rules: List[Rule]) -> List[str]:
    """:return: Error messages (empty = valid)"""
    errors: List[str] = []
    if output not in variables:
        errors.append(f"output variable {output!r} is not declared under variables")
    for name, variable in variables.items():
        if variable.high <= variable.low or variable.step <= 0:
            errors.append(f"{name}: universe must be [min < max, step > 0]")
            continue
        if variable.default is not None and not variable.low <= variable.default <= variable.high:
            errors.append(f"{name}: default {variable.default} is outside the universe")
        for variant in [None, *variable.variants]:
            label = f"{name}[{variant}]" if variant else name
            for gap in coverage_gaps(variable, variant):
                errors.append(f"{label}: no term covers {gap} (every μ = 0 there)")
            seen: Dict[tuple, str] = {}
            for term, parts in variable.term_set(variant).items():
                key = tuple(parts)
                if key in seen:
                    errors.append(f"{label}: terms {seen[key]!r} and {term!r} have the same definition")
                seen[key] = term
    ids: Dict[str, str] = {}
    antecedents: Dict[object, Rule] = {}
    for rule in rules:
        if rule.id in ids:
            errors.append(f"rule id {rule.id!r} is used twice")
        ids[rule.id] = rule.text
        if rule.output != output:
            errors.append(f"{rule.id}: concludes on {rule.output!r}, the output variable is {output!r}")
        elif output in variables and rule.term not in variables[output].terms:
            have = sorted(variables[output].terms)
            errors.append(f"{rule.id}: output term {rule.term!r} does not exist (have {have})")
        for var, term in rule.clauses():
            if var not in variables:
                errors.append(f"{rule.id}: unknown variable {var!r} in {rule.text!r}")
            elif var == output:
                errors.append(f"{rule.id}: the output variable {output!r} cannot be an input")
            elif term not in variables[var].terms:
                errors.append(f"{rule.id}: {var!r} has no term {term!r} (have {sorted(variables[var].terms)})")
        text = render(rule.antecedent)
        key = conjunction(rule.antecedent) or text                 # clause order does not matter for AND
        if key in antecedents:
            other = antecedents[key]
            kind = "duplicates" if other.term == rule.term else "conflicts with"
            errors.append(f"{rule.id} {kind} {other.id}: same condition '{text}' "
                          f"(→ {other.term} / {rule.term}); merge them or use a weight")
        antecedents[key] = rule
    return errors
