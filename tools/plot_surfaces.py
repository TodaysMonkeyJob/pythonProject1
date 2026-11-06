"""
Input-output surfaces of the hierarchical fuzzy system (A1 / TOD-41) → ``reports/surfaces/``::

    python -m tools.plot_surfaces [--out reports/surfaces] [--points 41]

For every subsystem (gas, camera, audio) and the top level: one 3-D surface per pair of inputs, the other inputs
fixed at the middle of their universe (the top level also with the third risk at 0 = missing / low), plus
``monotonicity.json`` / ``README.md``: for every input, over a grid of all the others, the share of steps where the
output falls although the input grows, the largest such fall and the number of level drops. Every input of these
rule bases is oriented "more = riskier", so a monotonic system has no falls.
"""
import argparse
import itertools
import json
import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core.fuzzy import FuzzyEngine, load_yaml  # noqa: E402
from modules.hfs.base import CONFIG_DIR, level_of  # noqa: E402
from modules.hfs.top import LEVELS, SUBSYSTEMS  # noqa: E402

EPS = 1e-6


def inputs_of(engine: FuzzyEngine) -> List[str]:
    return [name for name in engine.variables if name != engine.output]


def middle(engine: FuzzyEngine, name: str) -> float:
    variable = engine.variables[name]
    return (variable.low + variable.high) / 2


def surface(engine: FuzzyEngine, x_name: str, y_name: str, fixed: Dict[str, float], points: int):
    xs = np.linspace(engine.variables[x_name].low, engine.variables[x_name].high, points)
    ys = np.linspace(engine.variables[y_name].low, engine.variables[y_name].high, points)
    z = np.array([[engine.infer({**fixed, x_name: x, y_name: y})["crisp"] for x in xs] for y in ys])
    return xs, ys, z


def plot(engine: FuzzyEngine, level: str, x_name: str, y_name: str, fixed: Dict[str, float], points: int,
         path: str) -> None:
    xs, ys, z = surface(engine, x_name, y_name, fixed, points)
    gx, gy = np.meshgrid(xs, ys)
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(gx, gy, z, cmap="viridis", vmin=0, vmax=1, linewidth=0, antialiased=True)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel("risk")
    ax.set_zlim(0, 1)
    rest = ", ".join(f"{k} = {v:g}" for k, v in fixed.items())
    ax.set_title(f"{level}: risk({x_name}, {y_name})" + (f"\n{rest}" if rest else ""), fontsize=9)
    ax.view_init(elev=28, azim=-130)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def monotonicity(engine: FuzzyEngine, points: int) -> Dict[str, Dict[str, Any]]:
    """:return: {input: {steps, falls, share, max_fall, level_drops}} along that input over a grid of the others"""
    names = inputs_of(engine)
    grids = {n: np.linspace(engine.variables[n].low, engine.variables[n].high, points) for n in names}
    out = {}
    for moving in names:
        others = [n for n in names if n != moving]
        steps = falls = drops = 0
        worst = 0.0
        for combo in itertools.product(*(grids[n] for n in others)):
            fixed = dict(zip(others, combo, strict=True))
            values = [engine.infer({**fixed, moving: v})["crisp"] for v in grids[moving]]
            for earlier, later in zip(values[:-1], values[1:], strict=True):
                steps += 1
                if earlier - later > EPS:
                    falls += 1
                    worst = max(worst, earlier - later)
                    drops += LEVELS.index(level_of(engine, later)) < LEVELS.index(level_of(engine, earlier))
        out[moving] = {"steps": steps, "falls": falls, "share": round(falls / max(steps, 1), 4),
                       "max_fall": round(worst, 4), "level_drops": drops}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="HFS input-output surfaces and monotonicity")
    parser.add_argument("--dir", default=CONFIG_DIR, help="rule bases (gas / camera / audio / top .yaml)")
    parser.add_argument("--out", default=os.path.join("reports", "surfaces"))
    parser.add_argument("--points", type=int, default=41, help="grid points per axis of a surface")
    parser.add_argument("--check-points", type=int, default=11, help="grid points per input of the monotonicity check")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    report: Dict[str, Any] = {}
    figures: List[str] = []
    for level in (*SUBSYSTEMS, "top"):
        engine = FuzzyEngine.from_dict(load_yaml(os.path.join(args.dir, f"{level}.yaml")))
        names = inputs_of(engine)
        fixings = [{}]
        if len(names) > 2:
            fixings = [None]                                     # the others at the middle of their universe
            if level == "top":
                fixings.append("zero")                           # … and the third risk missing (0)
        for x_name, y_name in itertools.combinations(names, 2):
            for fixing in fixings:
                rest = [n for n in names if n not in (x_name, y_name)]
                fixed = {n: 0.0 if fixing == "zero" else middle(engine, n) for n in rest}
                suffix = "_others0" if fixing == "zero" else ""
                name = f"{level}_{x_name}__{y_name}{suffix}.png"
                plot(engine, level, x_name, y_name, fixed, args.points, os.path.join(args.out, name))
                figures.append(name)
        report[level] = {"operators": engine.operators, "rules": len(engine.rules),
                         "inputs": monotonicity(engine, args.check_points)}
        worst = max(v["max_fall"] for v in report[level]["inputs"].values())
        print(f"{level:<7} {len(figures):>3} figures so far; largest fall {worst:.4f}")
    with open(os.path.join(args.out, "monotonicity.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    lines = ["# HFS input-output surfaces (A1 / TOD-41)", "",
             "Generated by `python -m tools.plot_surfaces`. Monotonicity: for every input, over a grid of all the "
             f"others ({args.check_points} points each), the share of steps where the risk falls although the input "
             "grows. Every input is oriented \"more = riskier\".", "",
             "| Level | Input | Falls / steps | Largest fall | Level drops |", "| -- | -- | -- | -- | -- |"]
    for level, data in report.items():
        for name, stats in data["inputs"].items():
            lines.append(f"| {level} | `{name}` | {stats['falls']} / {stats['steps']} ({stats['share']:.1%}) | "
                         f"{stats['max_fall']:.4f} | {stats['level_drops']} |")
    lines += ["", "## Figures", ""] + [f"- [{name}]({name})" for name in figures]
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"→ {args.out}/ ({len(figures)} figures, monotonicity.json, README.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
