"""
Interval type-2 building blocks (F24 / TOD-35), numpy-free scalars for speed on the Pi.

* **IT2 membership** of a type-1 term with an uncertainty δ (in the variable's units): the upper function (UMF)
  moves every sloped edge outwards by δ, the lower one (LMF) inwards; shoulders on the universe edges stay where they
  are. When the inward edges cross, the LMF is the min of its two ramps — a triangle lower than 1 (the plateau is
  gone). Gaussians get an uncertain mean m ± δ. δ = 0 gives LMF = UMF = the type-1 function exactly.
* **Karnik–Mendel** centre-of-sets type reduction of rule centroids y_i with firing intervals [f̲_i, f̄_i] →
  [y_l, y_r]; ``brute_force_cos`` checks every switch point (tests).
"""
import math
from typing import Iterable, List, Optional, Sequence, Tuple

Interval = Tuple[float, float]
EPS = 1e-12


def _rise(x: float, a: float, b: float) -> float:
    """Left ramp: 0 at / below a, 1 at / above b (a step when a == b)."""
    if x >= b:
        return 1.0
    if x <= a or b <= a:
        return 0.0
    return (x - a) / (b - a)


def _fall(x: float, c: float, d: float) -> float:
    if x <= c:
        return 1.0
    if x >= d or d <= c:
        return 0.0
    return (d - x) / (d - c)


def trap_interval(x: float, a: float, b: float, c: float, d: float, delta: float, low: float,
                  high: float) -> Interval:
    """:return: (LMF, UMF) of trap(a, b, c, d) at x with the edges moved by ±delta"""
    left_fixed = a == b and a <= low          # shoulder on the universe edge
    right_fixed = c == d and d >= high
    if left_fixed:
        up_l = lo_l = 1.0 if x >= low else 0.0
    else:
        up_l = _rise(x, a - delta, b - delta)
        lo_l = _rise(x, a + delta, b + delta)
    if right_fixed:
        up_r = lo_r = 1.0 if x <= high else 0.0
    else:
        up_r = _fall(x, c + delta, d + delta)
        lo_r = _fall(x, c - delta, d - delta)
    return min(lo_l, lo_r), min(up_l, up_r)


def gauss_interval(x: float, mean: float, sigma: float, delta: float) -> Interval:
    distance = abs(x - mean)
    upper = 1.0 if distance <= delta else math.exp(-0.5 * ((distance - delta) / sigma) ** 2)
    lower = math.exp(-0.5 * ((distance + delta) / sigma) ** 2)
    return lower, upper


def term_interval(parts: Sequence[tuple], x: float, delta: float, low: float, high: float) -> Interval:
    """:param parts: a parsed type-1 term (``membership.parse_term``; several parts = their max)"""
    lo = hi = 0.0
    for shape, params in parts:
        if shape == "gauss":
            part = gauss_interval(x, params[0], params[1], delta)
        elif shape == "tri":
            part = trap_interval(x, params[0], params[1], params[1], params[2], delta, low, high)
        else:
            part = trap_interval(x, *params, delta, low, high)
        lo, hi = max(lo, part[0]), max(hi, part[1])
    return lo, hi


# --- type reduction -----------------------------------------------------------------------------------------------

def _weighted(ys: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    total = sum(weights)
    return None if total <= EPS else sum(y * w for y, w in zip(ys, weights, strict=True)) / total


def km_endpoint(ys: Sequence[float], lower: Sequence[float], upper: Sequence[float], left: bool,
                max_iter: int = 100) -> Optional[float]:
    """
    Karnik–Mendel iterative procedure for one end of the centre-of-sets interval (``ys`` ascending).

    y_l: upper firing for the centroids left of the switch point, lower firing right of it; y_r the reverse.
    :return: The endpoint, or None when no rule fires
    """
    n = len(ys)
    if n == 0 or sum(upper) <= EPS:
        return None
    weights = [(lo + hi) / 2 for lo, hi in zip(lower, upper, strict=True)]
    y = _weighted(ys, weights)
    if y is None:
        y = _weighted(ys, upper)
    for _ in range(max_iter):
        k = max(0, min(n - 2, sum(1 for v in ys if v <= y) - 1)) if n > 1 else 0
        if left:
            weights = [upper[i] if i <= k else lower[i] for i in range(n)]
        else:
            weights = [lower[i] if i <= k else upper[i] for i in range(n)]
        new = _weighted(ys, weights)
        if new is None:                        # only lower bounds of 0 on one side: fall back to the upper set
            new = _weighted(ys, upper)
        if abs(new - y) <= 1e-12:
            return new
        y = new
    return y


def brute_force_cos(ys: Sequence[float], lower: Sequence[float], upper: Sequence[float]) -> Optional[Interval]:
    """Reference: the min / max over every switch point (and the all-upper set) — what KM must find."""
    n = len(ys)
    candidates_l, candidates_r = [], []
    for k in range(-1, n):
        left = [upper[i] if i <= k else lower[i] for i in range(n)]
        right = [lower[i] if i <= k else upper[i] for i in range(n)]
        for weights, bucket in ((left, candidates_l), (right, candidates_r)):
            value = _weighted(ys, weights)
            if value is not None:
                bucket.append(value)
    if not candidates_l:
        return None
    return min(candidates_l), max(candidates_r)


def cos_interval(ys: Iterable[float], lower: Iterable[float], upper: Iterable[float]) -> Optional[Interval]:
    """:return: [y_l, y_r] of centre-of-sets type reduction (inputs in any order), None when nothing fires"""
    order = sorted(zip(ys, lower, upper, strict=True), key=lambda t: t[0])
    if not order:
        return None
    ys_s: List[float] = [t[0] for t in order]
    lo_s: List[float] = [t[1] for t in order]
    hi_s: List[float] = [t[2] for t in order]
    y_l = km_endpoint(ys_s, lo_s, hi_s, left=True)
    y_r = km_endpoint(ys_s, lo_s, hi_s, left=False)
    if y_l is None or y_r is None:
        return None
    return min(y_l, y_r), max(y_l, y_r)
