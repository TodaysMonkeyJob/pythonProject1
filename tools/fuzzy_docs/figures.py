"""Figures of the fuzzy-logic documentation, computed from the live rule base (config/fuzzy.yaml)."""
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from core.fuzzy import FuzzyEngine, load_yaml  # noqa: E402
from core.fuzzy.engine2 import FuzzyEngine2  # noqa: E402
from core.fuzzy.membership import evaluate, parse_term, trapmf  # noqa: E402
from tools.fuzzy_docs.labels import ACTION_UA, term_ua, var_ua  # noqa: E402

COLORS = ["#2b8a3e", "#e8a317", "#d9480f", "#a61e4d", "#1c7ed6", "#7048e8", "#0b7285", "#5c940d"]
ZOOM = {"gas_concentration": (0, 6), "gas_trend": (-3, 4), "people_count": (0, 7)}   # readable ranges in examples
LEVEL_COLORS = {"low": "#2b8a3e", "medium": "#e8a317", "high": "#d9480f", "critical": "#a61e4d"}
plt.rcParams.update({"font.family": ["Arial", "Arial Unicode MS"], "font.size": 9, "axes.titlesize": 10,
                     "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 100})


class Figures:
    def __init__(self, out_dir: str, fuzzy: str = "config/fuzzy.yaml"):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.fuzzy = fuzzy
        self.root = load_yaml(fuzzy)
        self.base = FuzzyEngine.from_config(fuzzy, None, None)
        self.warehouse = FuzzyEngine.from_config(fuzzy, "warehouse", None) if "warehouse" in (
            self.root.get("locations") or {}) else None
        self.it2 = FuzzyEngine2.from_config(fuzzy, None, None)
        self.levels = (self.root.get("envelope") or {}).get("thresholds") or {"medium": 0.25, "high": 0.5,
                                                                              "critical": 0.8}

    def save(self, fig, name: str) -> str:
        path = os.path.join(self.dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # --- general -------------------------------------------------------------------------------------------------

    def pipeline(self) -> str:
        fig, ax = plt.subplots(figsize=(11, 5.4))
        ax.set_xlim(0, 110)
        ax.set_ylim(0, 56)
        ax.axis("off")

        def box(x, y, w, h, text, color):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=color,
                                        ec="#333", lw=0.8))
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

        def arrow(x1, y1, x2, y2, text="", dx=0.0, dy=1.6):
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1, color="#333"))
            if text:
                ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, text, ha="center", fontsize=7, color="#444")

        ax.text(1, 53, "Нечітка логіка працює в B (правила довіри), A (правила ризику та підтвердження) і E (правила "
                       "виклику GPT).\nGPT ніколи не опускає ризик нижче межі L.", fontsize=8.5, color="#1f3b73",
                va="top")
        top, h = 30, 13
        box(1, top, 15, h, "Сенсори\nкамера\nмікрофон\nгаз", "#e7f5ff")
        box(23, top, 17, h, "B. Довіра\nτ — довіра\nu — невизначеність\nh — здоров'я", "#fff4e6")
        box(47, top, 17, h, "C. Знімок сцени\nнечіткі терми,\nдовірені /\nнедовірені дані", "#f3f0ff")
        box(71, top, 16, h, "E. Гейт GPT\nчи потрібен\nвиклик", "#e6fcf5")
        box(94, top, 15, h, "GPT\nрівень +\nрозподіл", "#f8f9fa")
        box(16, 3, 26, 16, "A. Коридор ризику (тип 2)\n[y_l, y_r] з правил ризику;\nпідтвердження безпеки c"
                           "\n→ межі L і U", "#ffe3e3")
        box(52, 3, 24, 16, "Перетин відповіді GPT\nз коридором\n→ згода h → гілка", "#fff0f6")
        box(85, 3, 24, 16, "Рішення\nr_final від 0 до 1\n→ рівень ризику\n(low … critical)", "#ebfbee")
        arrow(16.8, 36.5, 22.2, 36.5, "події")
        arrow(40.8, 36.5, 46.2, 36.5, "τ, u")
        arrow(64.8, 36.5, 70.2, 36.5, "знімок")
        arrow(87.8, 36.5, 93.2, 36.5, "так")
        arrow(31, 29.2, 29, 19.8, "фізичні докази + u", dx=-9, dy=0)
        arrow(100, 29.2, 70, 19.8, "відповідь GPT", dx=9, dy=0)
        arrow(77, 29.2, 64, 19.8, "ні → рішення = L", dx=-3, dy=2.5)
        arrow(42.8, 11, 51.2, 11)
        arrow(76.8, 11, 84.2, 11)
        return self.save(fig, "01_pipeline")

    def shapes(self) -> str:
        fig, axes = plt.subplots(1, 4, figsize=(11, 2.6))
        x = np.linspace(0, 10, 1001)
        specs = [("Трикутник tri(a, b, c)", [["tri", 2, 5, 8]], {"a": 2, "b": 5, "c": 8}),
                 ("Трапеція trap(a, b, c, d)", [["trap", 1, 3, 6, 8]], {"a": 1, "b": 3, "c": 6, "d": 8}),
                 ("Гаусіана gauss(m, σ)", [["gauss", 5, 1.5]], {"m": 5}),
                 ("Два шматки (через північ)", [["trap", 0, 0, 2, 3], ["trap", 7, 8, 10, 10]], {})]
        for ax, (title, spec, marks) in zip(axes, specs, strict=True):
            mu = evaluate(parse_term(spec), x)
            ax.fill_between(x, mu, alpha=0.25, color="#1c7ed6")
            ax.plot(x, mu, color="#1c7ed6")
            for k, v in marks.items():
                ax.axvline(v, color="#999", lw=0.6, ls=":")
                ax.text(v, -0.16, k, ha="center", fontsize=8)
            ax.set_ylim(-0.2, 1.1)
            ax.set_title(title)
            ax.set_yticks([0, 0.5, 1])
        axes[0].set_ylabel("μ(x)")
        fig.tight_layout()
        return self.save(fig, "02_shapes")

    def operators(self) -> str:
        x = np.linspace(0, 10, 1001)
        a, b = trapmf(x, 1, 3, 5, 7), trapmf(x, 4, 6, 8, 10)
        fig, axes = plt.subplots(1, 3, figsize=(11, 2.6))
        for ax, (title, y) in zip(axes, [("І (AND) = мінімум", np.minimum(a, b)), ("АБО (OR) = максимум",
                                                                                  np.maximum(a, b)),
                                         ("НЕ (NOT) = 1 − μ(A)", 1 - a)], strict=True):
            ax.plot(x, a, "--", color="#1c7ed6", lw=1, label="A")
            if "НЕ" not in title:
                ax.plot(x, b, "--", color="#d9480f", lw=1, label="B")
            ax.fill_between(x, y, alpha=0.35, color="#7048e8")
            ax.plot(x, y, color="#7048e8", lw=1.6, label="результат")
            ax.set_title(title)
            ax.set_ylim(0, 1.1)
            ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        return self.save(fig, "03_operators")

    # --- variables -----------------------------------------------------------------------------------------------

    def _variable(self, ax, engine: FuzzyEngine, name: str, variant: Optional[str] = None, dashed=None,
                  title: Optional[str] = None, value: Optional[float] = None, peaks: bool = False,
                  xlim: Optional[tuple] = None) -> None:
        var = engine.variables[name]
        grid, curves = var.curves(variant)
        for i, (term, mu) in enumerate(curves.items()):
            color = LEVEL_COLORS.get(term) if name in ("risk", "severity") else COLORS[i % len(COLORS)]
            ax.fill_between(grid, mu, alpha=0.18, color=color)
            ax.plot(grid, mu, color=color, lw=1.5, label=term_ua(name, term))
        if dashed is not None:
            grid2, curves2 = dashed.variables[name].curves(variant)
            for i, (term, mu) in enumerate(curves2.items()):
                if not np.allclose(mu, curves[term]):
                    ax.plot(grid2, mu, color=COLORS[i % len(COLORS)], lw=1.2, ls="--")
        if value is not None:
            ax.axvline(value, color="#222", lw=1.1)
            for mu_t in var.fuzzify(value, variant).values():
                if mu_t > 0.005:
                    ax.plot([value], [mu_t], "o", color="#222", ms=4)
                    ax.text(value, mu_t + 0.04, f"{mu_t:.2f}", fontsize=7, ha="left")
        unit = f", {var.unit}" if var.unit else ""
        ax.set_xlabel(f"{var_ua(name)}{unit}")
        ax.set_ylabel("μ")
        ax.set_ylim(0, 1.27 if peaks else 1.12)
        ax.set_title(title or var_ua(name))
        lo, hi = xlim or (var.low, var.high)
        ax.set_xlim(lo, hi)
        if peaks:                                         # compact: term names over their cores, no legend
            for i, (term, parts) in enumerate(var.term_set(variant).items()):
                shape, p = parts[-1] if term == "night" else parts[0]
                x = (p[1] + p[2]) / 2 if shape == "trap" else p[1] if shape == "tri" else p[0]
                x = min(hi - 0.06 * (hi - lo), max(lo + 0.06 * (hi - lo), x))
                ax.text(x, 1.06 if i % 2 == 0 else 1.16, term_ua(name, term), fontsize=6.3, ha="center",
                        color=COLORS[i % len(COLORS)]
                        if name not in ("risk", "severity") else LEVEL_COLORS.get(term, "#222"))
        else:
            ax.legend(fontsize=7, loc="upper right", ncol=2 if len(curves) > 3 else 1, framealpha=0.85)

    def variable(self, name: str, engine: Optional[FuzzyEngine] = None, prefix: str = "var",
                 variant: Optional[str] = None, value: Optional[float] = None, width: float = 7.0) -> str:
        fig, ax = plt.subplots(figsize=(width, 2.6))
        self._variable(ax, engine or self.base, name, variant, value=value)
        fig.tight_layout()
        return self.save(fig, f"{prefix}_{name}{'_' + variant if variant else ''}")

    def gas_variants(self) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(11, 2.8), sharey=True)
        self._variable(axes[0], self.base, "gas_concentration",
                       title="Базові терми (ліміт 3 %: аміак, толуол, невідомі гази)")
        self._variable(axes[1], self.base, "gas_concentration", "ethanol",
                       title="Етанол і ацетон (ліміт 5 %) — терми зсунуті праворуч")
        for ax in axes:
            ax.set_xlim(0, 7)
        fig.tight_layout()
        return self.save(fig, "var_gas_concentration_variants")

    def action_map(self) -> str:
        fig, ax = plt.subplots(figsize=(11, 3.2))
        self._variable(ax, self.base, "action_risk")
        mapping = (self.root.get("maps") or {}).get("action_risk") or {}
        groups: Dict[float, List[str]] = {}
        for action, v in mapping.items():
            groups.setdefault(float(v), []).append(action)
        for v, names in sorted(groups.items()):
            names = [ACTION_UA.get(n, n) for n in names]
            label = ", ".join(names) if len(names) <= 3 else ", ".join(names[:2]) + f" і ще {len(names) - 2}"
            ax.plot([v], [0.02], "v", color="#222", ms=5)
            ax.text(v, 0.06, label, rotation=90, fontsize=7, ha="center", va="bottom", color="#222")
        ax.set_title("Ризик дії: кожна дія отримує число (maps.action_risk), число → терми")
        fig.tight_layout()
        return self.save(fig, "var_action_risk_map")

    def time_override(self) -> Optional[str]:
        if self.warehouse is None:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(11, 2.8), sharey=True)
        self._variable(axes[0], self.base, "time_of_day_risk", title="Базовий розклад (усі локації)")
        self._variable(axes[1], self.warehouse, "time_of_day_risk", dashed=self.base,
                       title="Склад: ніч починається раніше (пунктир — базовий)")
        fig.tight_layout()
        return self.save(fig, "var_time_of_day_override")

    def risk_output(self) -> str:
        fig, ax = plt.subplots(figsize=(9, 2.9))
        self._variable(ax, self.base, "risk", title="Вихідна змінна «ризик» і межі рівнів для рішення")
        bounds = [0.0, self.levels["medium"], self.levels["high"], self.levels["critical"], 1.0]
        for i, level in enumerate(("low", "medium", "high", "critical")):
            ax.axvspan(bounds[i], bounds[i + 1], ymin=0, ymax=0.06, color=LEVEL_COLORS[level], alpha=0.85)
            ax.text((bounds[i] + bounds[i + 1]) / 2, 0.08, term_ua("risk", level), ha="center", fontsize=7)
        for v in bounds[1:-1]:
            ax.axvline(v, color="#222", lw=0.8, ls="--")
            ax.text(v, 1.05, f"{v:.2f}", ha="center", fontsize=7)
        fig.tight_layout()
        return self.save(fig, "var_risk_output")

    def overlap(self) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(11, 2.8))
        self._variable(axes[0], self.base, "gas_concentration", value=1.4,
                       title="Накладання термів: 1,4 % — одночасно «незначна» і «підвищена»")
        axes[0].set_xlim(0, 5)
        var = self.base.variables["gas_concentration"]
        xs = np.linspace(0, 5, 501)
        grid, curves = var.curves()
        for i, mu in enumerate(curves.values()):
            axes[1].plot(grid, mu, color=COLORS[i], lw=1, alpha=0.6)
        total = sum(np.interp(xs, grid, mu) for mu in curves.values())
        axes[1].plot(xs, total, color="#222", lw=1.6, label="сума μ усіх термів")
        axes[1].axhline(1, color="#999", ls=":")
        axes[1].set_xlim(0, 5)
        axes[1].set_ylim(0, 1.3)
        axes[1].set_title("Покриття: у кожній точці є хоча б один терм (перевіряється при завантаженні)")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        return self.save(fig, "concept_overlap")

    def small_multiples(self, engine_dict: Dict[str, Any], names: Sequence[str], fname: str, cols: int = 3,
                        title: Optional[str] = None) -> str:
        """Plots variables of a rule base given as a dict (trust.source, gating, …)."""
        engine = FuzzyEngine.from_dict(engine_dict)
        rows = math.ceil(len(names) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(11, 2.5 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, name in zip(axes, names, strict=False):
            self._variable(ax, engine, name)
        for ax in axes[len(names):]:
            ax.axis("off")
        if title:
            fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        return self.save(fig, fname)

    # --- inference -----------------------------------------------------------------------------------------------

    def inference(self, inputs: Dict[str, Any], name: str, title: str) -> Dict[str, Any]:
        """Step-by-step Mamdani for one case: fuzzified inputs, fired rules, clipped / aggregated output, crisp"""
        engine = self.base
        result = engine.infer(inputs)
        shown = [v for v in ("gas_concentration", "gas_trend", "motion_level", "people_count", "unknown_share",
                             "action_risk", "time_of_day_risk") if v in inputs]
        fig = plt.figure(figsize=(11, 6.8))
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.25])
        for i, var in enumerate(shown[:8]):
            ax = fig.add_subplot(gs[i // 4, i % 4])
            self._variable(ax, engine, var, inputs.get("gas") if var == "gas_concentration" else None,
                           value=float(inputs[var]), title=f"{var_ua(var)} = {inputs[var]:g}", peaks=True,
                           xlim=ZOOM.get(var))
            ax.set_xlabel("")
        ax_rules = fig.add_subplot(gs[2, :2])
        fired = result["fired_rules"][:8]
        if fired:
            labels = [f"{r['id']} → {term_ua('risk', r['term'])}" for r in fired][::-1]
            values = [r["strength"] * r.get("weight", 1.0) for r in fired][::-1]
            ax_rules.barh(labels, values, color=[LEVEL_COLORS[r["term"]] for r in fired][::-1])
            for y, v in enumerate(values):
                ax_rules.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=7)
        ax_rules.set_xlim(0, 1.1)
        ax_rules.set_title("Крок 2: сила спрацювання правил (min для «і», max для «або»)")
        ax_out = fig.add_subplot(gs[2, 2:])
        grid, curves = engine._grid, engine._curves
        aggregate = np.zeros_like(grid)
        for term, mu in curves.items():
            act = result["memberships"].get(term, 0.0)
            ax_out.plot(grid, mu, color=LEVEL_COLORS[term], lw=0.8, ls=":")
            if act > 0:
                clipped = np.minimum(mu, act)
                ax_out.fill_between(grid, clipped, alpha=0.35, color=LEVEL_COLORS[term],
                                    label=f"{term_ua('risk', term)} ≤ {act:.2f}")
                aggregate = np.maximum(aggregate, clipped)
        ax_out.plot(grid, aggregate, color="#222", lw=1.6)
        ax_out.axvline(result["crisp"], color="#1c7ed6", lw=1.5)
        ax_out.text(result["crisp"], 1.04, f"центроїд {result['crisp']:.2f}", color="#1c7ed6", fontsize=7,
                    ha="center")
        if result.get("mom") is not None:
            ax_out.axvline(result["mom"], color="#7048e8", lw=1.2, ls="--")
            ax_out.text(result["mom"], 0.92, f"MOM {result['mom']:.2f}", color="#7048e8", fontsize=7, ha="center")
        ax_out.set_ylim(0, 1.12)
        ax_out.set_xlabel("ризик")
        ax_out.set_title("Крок 3–4: обрізання, об'єднання (max) і дефазифікація")
        ax_out.legend(fontsize=7, loc="upper left")
        fig.suptitle(title + " — крок 1: фазифікація входів (чорна лінія = значення, точки = μ термів)",
                     fontsize=10)
        fig.tight_layout()
        result["figure"] = self.save(fig, name)
        return result

    def gas_matrix(self) -> str:
        """Rule matrix: concentration term × trend term → crisp risk of the base rule base (other inputs calm)"""
        conc = self.base.variables["gas_concentration"]
        trend = self.base.variables["gas_trend"]
        c_terms, t_terms = list(conc.terms), list(trend.terms)

        def centre(parts):
            shape, p = parts[0]
            return (p[1] + p[2]) / 2 if shape == "trap" else p[1] if shape == "tri" else p[0]

        matrix = np.zeros((len(c_terms), len(t_terms)))
        labels = [["" for _ in t_terms] for _ in c_terms]
        for i, c in enumerate(c_terms):
            for j, t in enumerate(t_terms):
                out = self.base.infer({"gas_concentration": min(9.5, centre(conc.terms[c])),
                                       "gas_trend": max(-9.5, min(9.5, centre(trend.terms[t]))),
                                       "time_of_day_risk": 12})
                matrix[i, j] = out["crisp"]
                best = max(out["memberships"], key=out["memberships"].get) if any(out["memberships"].values()) \
                    else "low"
                labels[i][j] = f"{term_ua('risk', best)}\n{out['crisp']:.2f}"
        fig, ax = plt.subplots(figsize=(7, 3.6))
        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1)
        ax.set_xticks(range(len(t_terms)), [term_ua("gas_trend", t) for t in t_terms])
        ax.set_yticks(range(len(c_terms)), [term_ua("gas_concentration", c) for c in c_terms])
        for i in range(len(c_terms)):
            for j in range(len(t_terms)):
                ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=7.5)
        ax.set_xlabel("динаміка газу")
        ax.set_ylabel("концентрація газу")
        ax.set_title("Таблиця газових правил: результат рушія в центрі кожного терму")
        fig.colorbar(im, ax=ax, label="ризик")
        fig.tight_layout()
        return self.save(fig, "rules_gas_matrix")

    def gas_curve(self) -> str:
        fig, ax = plt.subplots(figsize=(9, 2.9))
        xs = np.linspace(0, 8, 161)
        for gas, label, color in ((None, "аміак / толуол / невідомий (ліміт 3 %)", "#d9480f"),
                                  ("acetone", "ацетон / етанол (ліміт 5 %)", "#1c7ed6")):
            for trend, ls in ((0.0, "-"), (3.0, "--")):
                ys = [self.base.infer({"gas_concentration": x, "gas_trend": trend, "time_of_day_risk": 12,
                                       **({"gas": gas} if gas else {})})["crisp"] for x in xs]
                ax.plot(xs, ys, ls=ls, color=color, label=f"{label}, {'стабільна' if trend == 0 else 'різко зростає'}")
        for level, v in self.levels.items():
            ax.axhline(v, color="#999", lw=0.6, ls=":")
            ax.text(8.05, v, term_ua("risk", level), fontsize=7, va="center")
        ax.set_xlabel("концентрація газу, %")
        ax.set_ylabel("ризик (центроїд)")
        ax.set_title("Як ризик залежить від концентрації: однакові правила, різні діапазони термів для газів")
        ax.legend(fontsize=7)
        fig.tight_layout()
        return self.save(fig, "rules_gas_curve")

    def warehouse_effect(self) -> Optional[str]:
        if self.warehouse is None:
            return None
        hours = np.linspace(0, 24, 241)
        fig, ax = plt.subplots(figsize=(9, 2.9))
        for engine, label, color in ((self.base, "базові правила", "#1c7ed6"), (self.warehouse, "склад", "#d9480f")):
            ys = [engine.infer({"people_count": 1, "unknown_share": 0.5, "time_of_day_risk": h})["crisp"]
                  for h in hours]
            ax.plot(hours, ys, color=color, label=label)
        ax.set_xticks(range(0, 25, 2))
        ax.set_xlabel("година доби")
        ax.set_ylabel("ризик")
        ax.set_title("Незнайомець у кадрі протягом доби: локальні правила складу (W1, W2) піднімають ризик раніше")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return self.save(fig, "rules_warehouse_effect")

    def rule_counts(self) -> str:
        sections = {"ризик": self.root.get("rules") or [],
                    "довіра: джерело мови": ((self.root.get("trust") or {}).get("source") or {}).get("rules") or [],
                    "підтвердження безпеки": (((self.root.get("envelope") or {}).get("confirmation") or {})
                                              .get("rules") or []),
                    "виклик GPT": (self.root.get("gating") or {}).get("rules") or []}
        fig, ax = plt.subplots(figsize=(8, 2.6))
        names = list(sections)
        counts = [len(v) for v in sections.values()]
        ax.barh(names, counts, color=COLORS[:len(names)])
        for y, c in enumerate(counts):
            ax.text(c + 0.3, y, str(c), va="center")
        ax.set_xlabel("кількість правил")
        ax.set_title("Скільки правил у кожній базі")
        fig.tight_layout()
        return self.save(fig, "rules_counts")

    # --- trust ---------------------------------------------------------------------------------------------------

    def trust_beta(self) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(11, 2.9))
        r = np.arange(0, 41)
        for s, color in ((0, "#2b8a3e"), (2, "#e8a317"), (8, "#d9480f"), (20, "#a61e4d")):
            axes[0].plot(r, (r + 1) / (r + s + 2), color=color, label=f"спростувань s = {s}")
        axes[0].set_xlabel("підтверджень r")
        axes[0].set_ylabel("τ (при h = 1)")
        axes[0].set_ylim(0, 1)
        axes[0].set_title("τ = (r + 1) / (r + s + 2): довіра росте з підтвердженнями")
        axes[0].legend(fontsize=7)
        n = np.arange(0, 41)
        for h, color in ((1.0, "#1c7ed6"), (0.6, "#e8a317"), (0.3, "#d9480f")):
            axes[1].plot(n, 1 - h * (1 - 2 / (n + 2)), color=color, label=f"здоров'я h = {h}")
        axes[1].set_xlabel("усього спостережень r + s")
        axes[1].set_ylabel("u_eff")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Невизначеність u_eff = 1 − h·(1 − 2/(r + s + 2))")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        return self.save(fig, "trust_beta")

    def trust_timeline(self) -> str:
        """Simulated channel: confirmations, a refutation burst, a health failure and its gradual recovery"""
        reg = load_yaml(os.path.join(os.path.dirname(self.fuzzy), "trust_registry.yaml"))
        recovery = float(((reg.get("health") or {}).get("recovery_per_min")) or 0.1)
        t = np.arange(0, 120.5, 0.5)                      # minutes
        r, s, h, taus, hs = 4.0, 1.0, 1.0, [], []
        for minute in t:
            if minute % 5 == 0 and minute < 40:
                r += 1
            if 40 <= minute < 45 and minute % 1 == 0:
                s += 1
            if minute == 60:
                h = 0.3
            if minute > 80:
                h = min(1.0, h + recovery * 0.5)
            taus.append((r + 1) / (r + s + 2) * h)
            hs.append(h)
        fig, ax = plt.subplots(figsize=(10, 2.9))
        ax.plot(t, taus, color="#1c7ed6", lw=1.6, label="τ (довіра)")
        ax.plot(t, hs, color="#d9480f", lw=1, ls="--", label="h (здоров'я)")
        for x, text in ((20, "підтвердження"), (42, "спростування"), (60, "збій: h = 0,3"),
                        (95, f"ремонт: h росте на {recovery:g}/хв")):
            ax.annotate(text, (x, 0.05), fontsize=7, ha="center", color="#444")
        ax.set_xlabel("хвилини")
        ax.set_ylim(0, 1.05)
        ax.set_title("Як змінюється довіра одного каналу (модель реєстру довіри)")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        return self.save(fig, "trust_timeline")

    def trust_terms(self, claim_theta: float) -> str:
        cfg = {"output": "trust", "variables": {"trust": {"universe": [0, 1, 0.01],
                                                          "terms": (self.root["trust"]["terms"])}}, "rules": []}
        engine = FuzzyEngine.from_dict(cfg)
        fig, ax = plt.subplots(figsize=(8, 2.6))
        self._variable(ax, engine, "trust", title="Терми довіри та поріг θ для заяв (мова / текст у кадрі)")
        ax.axvline(claim_theta, color="#222", ls="--")
        ax.text(claim_theta + 0.01, 1.02, f"θ = {claim_theta:.2f}: нижче — заява не довірена", fontsize=7)
        fig.tight_layout()
        return self.save(fig, "trust_terms")

    # --- type 2 ---------------------------------------------------------------------------------------------------

    def fou(self, variable: str = "gas_concentration", u: float = 0.5) -> str:
        delta = self.it2.deltas({variable: u}).get(variable, 0.0)
        grid, curves = self.it2.fou_curves(variable, delta)
        fig, ax = plt.subplots(figsize=(9, 2.9))
        for i, (term, (lmf, umf)) in enumerate(curves.items()):
            ax.fill_between(grid, lmf, umf, color=COLORS[i], alpha=0.3, label=f"{term_ua(variable, term)}: FOU")
            ax.plot(grid, umf, color=COLORS[i], lw=1)
            ax.plot(grid, lmf, color=COLORS[i], lw=1, ls="--")
        ax.set_xlim(self.base.variables[variable].low, min(self.base.variables[variable].high, 6))
        ax.set_ylim(0, 1.1)
        ax.set_xlabel(f"{var_ua(variable)}, {self.base.variables[variable].unit}")
        ax.set_ylabel("μ")
        ax.set_title(f"Тип 2: при u = {u:g} межі термів «розмиваються» на δ = k·u = {delta:.2f} "
                     "(суцільна — верхня UMF, пунктир — нижня LMF)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        return self.save(fig, f"it2_fou_{variable}")

    def interval_vs_u(self) -> str:
        inputs = {"gas_concentration": 2.0, "gas": "acetone", "gas_trend": 0.8, "people_count": 1,
                  "action_risk": 0.1, "time_of_day_risk": 14}
        us = np.linspace(0, 1, 21)
        lo, hi, t1 = [], [], []
        for u in us:
            out = self.it2.infer(inputs, {"gas_concentration": u, "gas_trend": u}, explain=False)
            lo.append(out["interval"][0])
            hi.append(out["interval"][1])
            t1.append(out["type1"]["crisp"])
        fig, ax = plt.subplots(figsize=(9, 2.9))
        ax.fill_between(us, lo, hi, color="#ffa8a8", alpha=0.5, label="коридор [y_l, y_r]")
        ax.plot(us, lo, color="#c92a2a", lw=1.2, label="y_l (нижня межа)")
        ax.plot(us, hi, color="#862e9c", lw=1.2, label="y_r (верхня межа)")
        ax.plot(us, t1, color="#222", ls=":", label="тип 1 (без невизначеності)")
        ax.set_xlabel("невизначеність сенсора газу u")
        ax.set_ylabel("ризик")
        ax.set_title("Чим менше ми певні в сенсорі, тим ширший коридор (ацетон 2 %, повільне зростання)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        return self.save(fig, "it2_interval_vs_u")

    # --- corridor ------------------------------------------------------------------------------------------------

    def corridor_cases(self, lam: float, h_min: float) -> List[Dict[str, Any]]:
        from modules.envelope import fuzzy_corridor
        from modules.llm.calibration import DEFAULT_PARAMS

        slope = fuzzy_corridor.output_slope(self.base.variables["risk"].terms)
        cases = [("GPT погоджується (agree)", 0.62, 0.8, 0.0, {"high": 0.9, "medium": 0.1}),
                 ("GPT занижує → залишається L", 0.62, 0.8, 0.0, {"low": 0.9, "medium": 0.1}),
                 ("GPT вище коридору → перевірка оператором", 0.15, 0.3, 0.0, {"critical": 0.85, "high": 0.15}),
                 ("Підтвердження безпеки c = 0,6 опускає L", 0.62, 0.8, 0.6, {"medium": 0.9, "low": 0.1})]
        fig, axes = plt.subplots(2, 2, figsize=(11, 5.4))
        out = []
        for ax, (title, y_l, y_r, c, dist) in zip(axes.ravel(), cases, strict=True):
            res = fuzzy_corridor.decide(y_l, y_r, c, dist, None, DEFAULT_PARAMS, lam, h_min, slope)
            grid = fuzzy_corridor.GRID
            k = fuzzy_corridor.corridor_set(res["L"], res["U"], slope)
            g = fuzzy_corridor.llm_set(dist, None, DEFAULT_PARAMS)
            d = np.minimum(k, g)
            ax.fill_between(grid, k, color="#ffc9c9", alpha=0.6, label="коридор K")
            ax.plot(grid, g, color="#1c7ed6", lw=1.4, label="відповідь GPT")
            ax.fill_between(grid, d, color="#7048e8", alpha=0.5, label="перетин μ_D")
            ax.axvline(res["L"], color="#c92a2a", ls="--", lw=1)
            ax.axvline(res["U"], color="#862e9c", ls="--", lw=1)
            ax.axvline(res["r_final"], color="#222", lw=2)
            ax.text(res["L"], 1.17, "L", color="#c92a2a", ha="center", fontsize=8)
            ax.text(res["U"], 1.17, "U", color="#862e9c", ha="center", fontsize=8)
            ax.text(0.01, 0.93, f"h = {res['h']:.2f} (поріг {h_min:.2f})\nгілка: {res['policy_branch']}\n"
                                f"r_final = {res['r_final']:.2f}", fontsize=7, transform=ax.transAxes, va="top")
            ax.set_ylim(0, 1.25)
            ax.set_title(title)
            ax.set_xlabel("ризик")
            out.append({"title": title, "y_l": y_l, "y_r": y_r, "c": c, "dist": dist, **res})
        axes[0, 0].legend(fontsize=7, loc="center right")
        fig.tight_layout()
        self.corridor_path = self.save(fig, "corridor_cases")
        return out

    def bounds_vs_c(self) -> str:
        y_l, y_r = 0.6, 0.85
        c = np.linspace(0, 1, 101)
        fig, ax = plt.subplots(figsize=(8, 2.8))
        for lam, color in ((0.0, "#c92a2a"), (0.5, "#e8a317"), (1.0, "#1c7ed6")):
            ax.plot(c, (1 - c) * (y_l + lam * (y_r - y_l)), color=color, label=f"L при λ = {lam:g}")
        ax.axhline(y_r, color="#862e9c", ls="--", label="U = y_r")
        ax.set_xlabel("підтвердження безпеки c")
        ax.set_ylabel("межа ризику")
        ax.set_title("Нижня межа L = (1 − c)·(y_l + λ·(y_r − y_l)), приклад y_l = 0,6, y_r = 0,85")
        ax.legend(fontsize=7)
        fig.tight_layout()
        return self.save(fig, "corridor_bounds_vs_c")

    def decision_tree(self) -> str:
        fig, ax = plt.subplots(figsize=(10, 4.4))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 48)
        ax.axis("off")

        def box(x, y, w, h, text, color):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color,
                                        ec="#333", lw=0.8))
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

        def arrow(x1, y1, x2, y2, text=""):
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1))
            if text:
                ax.text((x1 + x2) / 2 + 0.5, (y1 + y2) / 2, text, fontsize=7, color="#444")

        box(38, 39, 24, 7, "Є відповідь GPT?", "#f8f9fa")
        box(4, 26, 22, 8, "fuzzy_only\nr_final = L", "#e9ecef")
        box(38, 26, 24, 8, "Згода h ≥ h_min?", "#f8f9fa")
        box(70, 26, 26, 8, "agree\nr_final = r_D (у межах [L, U])", "#d3f9d8")
        box(4, 6, 26, 9, "deescalation_rejected\nGPT нижче L → r_final = L", "#ffe3e3")
        box(37, 6, 26, 9, "escalation_review\nGPT вище U → r_final = U,\nоператор вирішує", "#fff3bf")
        box(70, 6, 26, 9, "clamp\nr_llm всередині, але розподіл\nрозмитий → обрізати в [L, U]", "#e7f5ff")
        arrow(44, 39, 18, 34, "ні")
        arrow(50, 39, 50, 34, "так")
        arrow(62, 30, 70, 30, "так")
        arrow(44, 26, 17, 15, "ні, r_llm < L")
        arrow(50, 26, 50, 15, "ні, r_llm > U")
        arrow(57, 26, 82, 15, "ні, інакше")
        return self.save(fig, "corridor_decision_tree")

    def calibration_sets(self) -> str:
        from modules.llm.calibration import DEFAULT_PARAMS, LEVELS, RiskCalibrator

        cal = RiskCalibrator.from_file("config/calibration.yaml")
        params = cal.params if cal is not None else DEFAULT_PARAMS
        grid = np.linspace(0, 1, 501)
        fig, ax = plt.subplots(figsize=(9, 2.8))
        for level in LEVELS:
            ax.fill_between(grid, trapmf(grid, *params[level]), alpha=0.25, color=LEVEL_COLORS[level])
            ax.plot(grid, trapmf(grid, *params[level]), color=LEVEL_COLORS[level], label=term_ua("risk", level))
        ax.set_xlabel("ризик")
        ax.set_ylabel("μ")
        ax.set_title("Що означає слово GPT («high» тощо) як нечітка множина "
                     + ("(підібрано калібруванням)" if cal else "(фіксоване відображення: калібрування не краще)"))
        ax.legend(fontsize=7)
        fig.tight_layout()
        return self.save(fig, "calibration_sets")

    # --- gating / thresholds -------------------------------------------------------------------------------------

    def gating_examples(self, threshold: float) -> List[Dict[str, Any]]:
        engine = FuzzyEngine.from_dict(self.root["gating"])
        cases = [("Спокійна знайома сцена, GPT щойно погодився",
                  {"novelty": 0.0, "width": 0.0, "agreement": 0.9, "untrusted_claims": 0.0, "time_since_last_llm": 2,
                   "severity": 0.1}),
                 ("Нова людина, вузький коридор, перший знімок",
                  {"novelty": 0.9, "width": 0.0, "untrusted_claims": 0.0, "time_since_last_llm": 60, "severity": 0.1}),
                 ("Широкий коридор (сенсор непевний)",
                  {"novelty": 0.2, "width": 0.5, "agreement": 0.9, "untrusted_claims": 0.0, "time_since_last_llm": 3,
                   "severity": 0.3}),
                 ("Недовірена заява «все добре»",
                  {"novelty": 0.0, "width": 0.0, "agreement": 0.9, "untrusted_claims": 1.0, "time_since_last_llm": 3,
                   "severity": 0.4}),
                 ("GPT не погодився, з'явилося нове",
                  {"novelty": 0.6, "width": 0.0, "agreement": 0.0, "untrusted_claims": 0.0, "time_since_last_llm": 4,
                   "severity": 0.3})]
        rows = []
        for title, inputs in cases:
            out = engine.infer(inputs)
            rows.append({"title": title, "inputs": inputs, "score": out["crisp"],
                         "fired": [r["id"] for r in out["fired_rules"]]})
        fig, ax = plt.subplots(figsize=(9, 2.9))
        ax.barh([r["title"] for r in rows][::-1], [r["score"] for r in rows][::-1],
                color=["#d9480f" if r["score"] >= threshold else "#2b8a3e" for r in rows][::-1])
        ax.axvline(threshold, color="#222", ls="--")
        ax.text(threshold + 0.01, len(rows) - 0.5, f"поріг {threshold:.2f}", fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("потреба викликати GPT (call_llm)")
        ax.set_title("Гейт: червоне — виклик GPT, зелене — рішення без GPT (за коридором)")
        fig.tight_layout()
        self.gating_path = self.save(fig, "gating_examples")
        return rows

    def cost_threshold(self, cost_ratio: float, call_cost: float) -> str:
        from core.calibration import _rates

        rng = np.random.default_rng(3)
        pos = np.clip(rng.normal(0.72, 0.12, 60), 0, 1)
        neg = np.clip(rng.normal(0.35, 0.15, 240), 0, 1)
        grid = np.linspace(0.05, 0.95, 91)
        p_miss = 1 - _rates(pos, grid, 0.05)
        p_fa = _rates(neg, grid, 0.05)
        flagged = _rates(np.r_[pos, neg], grid, 0.05)
        cost = cost_ratio * p_miss + p_fa
        cost_call = cost + call_cost * flagged
        fig, ax = plt.subplots(figsize=(9, 2.9))
        ax.plot(grid, cost_ratio * p_miss, color="#c92a2a", ls="--",
                label=f"пропуск: C_miss·P_miss (C_miss = {cost_ratio:g})")
        ax.plot(grid, p_fa, color="#e8a317", ls="--", label="хибна тривога: C_fa·P_fa (C_fa = 1)")
        ax.plot(grid, cost, color="#222", lw=1.6, label="разом")
        ax.plot(grid, cost_call, color="#1c7ed6", lw=1.2, label=f"+ вартість виклику {call_cost:g}·P_виклику")
        for curve, color in ((cost, "#222"), (cost_call, "#1c7ed6")):
            i = int(np.argmin(curve))
            ax.plot(grid[i], curve[i], "o", color=color)
            ax.annotate(f"мінімум t = {grid[i]:.2f}", (grid[i], curve[i]), textcoords="offset points",
                        xytext=(5, 8), fontsize=7, color=color)
        ax.set_ylim(0, 3)
        ax.set_xlabel("поріг t")
        ax.set_ylabel("очікувана вартість")
        ax.set_title("Як підбирається поріг: мінімум очікуваної вартості помилок (ілюстрація на синтетичних даних)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        return self.save(fig, "thresholds_cost")

    def governance_flow(self) -> str:
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.set_xlim(0, 110)
        ax.set_ylim(0, 40)
        ax.axis("off")
        steps = [("Ідея правила\n(експерт або\nпропозиція GPT)", "#e7f5ff"),
                 ("Кандидат\nfuzzy.yaml\n+ метадані", "#f8f9fa"),
                 ("Перевірки\nсинтаксис, покриття,\nконфлікти, дублікати", "#fff4e6"),
                 ("Регресійний гейт\n120 сценаріїв:\nASR, F1, ECE", "#ffe3e3"),
                 ("Схвалення\nхто, роль,\nкоментар", "#d3f9d8"),
                 ("Нова версія\nsha256, журнал\nзмін", "#f3f0ff")]
        for i, (text, color) in enumerate(steps):
            x = 2 + i * 18
            ax.add_patch(FancyBboxPatch((x, 18), 15, 14, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color,
                                        ec="#333", lw=0.8))
            ax.text(x + 7.5, 25, text, ha="center", va="center", fontsize=9.5)
            if i:
                ax.add_patch(FancyArrowPatch((x - 3, 25), (x, 25), arrowstyle="-|>", mutation_scale=11, lw=1))
        ax.add_patch(FancyArrowPatch((99, 18), (60, 6), arrowstyle="-|>", mutation_scale=11, lw=1, ls="--",
                                     connectionstyle="arc3,rad=-0.2"))
        ax.text(80, 6, "відкат: rollback <версія>", fontsize=9.5)
        ax.text(38, 13, "не пройшло → правило не застосовується, звіт з причинами", fontsize=9.5, color="#c92a2a")
        ax.text(2, 37, "Жодне правило не змінює систему без перевірок і запису про схвалення", fontsize=10.5,
                color="#1f3b73")
        return self.save(fig, "governance_flow")
