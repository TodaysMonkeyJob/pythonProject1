"""
Membership functions (F15 / TOD-26), vectorised over numpy arrays (or a scalar).

A term is defined in YAML as ``[shape, *params]`` — or a list of those, combined with max (a term that wraps around,
e.g. ``night`` = 21:00–24:00 and 00:00–06:00):

* ``[tri, a, b, c]``       — triangle: 0 at a, 1 at b, 0 at c;
* ``[trap, a, b, c, d]``   — trapezoid: rises a→b, 1 on b..c, falls c→d (a == b / c == d = shoulder);
* ``[gauss, mean, sigma]`` — Gaussian bell.
"""
from typing import Any, Callable, Dict, List, Sequence

import numpy as np


def trimf(x, a: float, b: float, c: float) -> np.ndarray:
    return trapmf(x, a, b, b, c)


def trapmf(x, a: float, b: float, c: float, d: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)
    y[(x >= b) & (x <= c)] = 1.0
    if b > a:
        rising = (x > a) & (x < b)
        y[rising] = (x[rising] - a) / (b - a)
    if d > c:
        falling = (x > c) & (x < d)
        y[falling] = (d - x[falling]) / (d - c)
    return y


def gaussmf(x, mean: float, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)


SHAPES: Dict[str, Callable[..., np.ndarray]] = {"tri": trimf, "trap": trapmf, "gauss": gaussmf}
ARITY = {"tri": 3, "trap": 4, "gauss": 2}


class TermError(ValueError):
    pass


def parse_term(spec: Any) -> List[tuple]:
    """
    :param spec: ``[shape, *params]`` or a list of them
    :return: [(shape, params)] — validated
    :raise TermError: unknown shape, wrong number of parameters, unordered points, sigma ≤ 0
    """
    parts = spec if spec and isinstance(spec[0], (list, tuple)) else [spec]
    out = []
    for part in parts:
        if not isinstance(part, (list, tuple)) or not part or part[0] not in SHAPES:
            raise TermError(f"unknown shape in {part!r} (use tri / trap / gauss)")
        shape, params = part[0], [float(p) for p in part[1:]]
        if len(params) != ARITY[shape]:
            raise TermError(f"{shape} needs {ARITY[shape]} numbers, got {params}")
        if shape == "gauss" and params[1] <= 0:
            raise TermError(f"gauss sigma must be > 0, got {params[1]}")
        if shape != "gauss" and any(params[i] > params[i + 1] for i in range(len(params) - 1)):
            raise TermError(f"{shape} points must be non-decreasing, got {params}")
        out.append((shape, tuple(params)))
    return out


def evaluate(parts: Sequence[tuple], x) -> np.ndarray:
    """:return: μ(x) of a parsed term (max over its parts)"""
    values = [SHAPES[shape](x, *params) for shape, params in parts]
    return values[0] if len(values) == 1 else np.maximum.reduce(values)


def shift(parts: Sequence[tuple], delta: float, low: float, high: float) -> List[tuple]:
    """
    Moves a term by ``delta`` (per-location overrides: "stricter at night"). Points on the universe edges stay there,
    so shoulders keep covering the ends; Gaussians move their mean.
    """
    out = []
    for shape, params in parts:
        if shape == "gauss":
            out.append((shape, (params[0] + delta, params[1])))
        else:
            moved = [p if p in (low, high) else min(high, max(low, p + delta)) for p in params]
            out.append((shape, tuple(moved)))
    return out
