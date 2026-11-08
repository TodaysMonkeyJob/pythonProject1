"""
Figures of the logic documentation (D1 / TOD-55), computed from the live rule bases and configuration. The style is
the one of the earlier fuzzy-logic document (tools/fuzzy_docs): the same palette, level colours and font.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from core.fuzzy import FuzzyEngine  # noqa: E402
from core.fuzzy.it2 import term_interval  # noqa: E402
from core.fuzzy.membership import evaluate  # noqa: E402
from modules.hfs import policy  # noqa: E402
from tools.logic_docs.data import LEVEL_UK, TERM_UK, VAR_UK, Facts  # noqa: E402

COLORS = ["#2b8a3e", "#e8a317", "#d9480f", "#a61e4d", "#1c7ed6", "#7048e8", "#0b7285", "#5c940d"]
LEVEL_COLORS = {"low": "#2b8a3e", "medium": "#e8a317", "high": "#d9480f", "critical": "#a61e4d"}
LEVELS = ("low", "medium", "high", "critical")
plt.rcParams.update(
    {
        "font.family": ["Arial", "Arial Unicode MS", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100,
    }
)


class Figures:
    def __init__(self, facts: Facts, out_dir: str):
        self.f, self.dir = facts, out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.engines = {n: FuzzyEngine.from_dict(b) for n, b in facts.bases.items()}
        self.made: List[str] = []

    def save(self, fig, name: str) -> str:
        path = os.path.join(self.dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.made.append(name)
        return name

    # --- schemes ------------------------------------------------------------------------------------------------------

    def _boxes(
        self,
        ax,
        boxes: Dict[str, Tuple[float, float, float, float, str, str]],
        arrows: Sequence[Tuple[str, str]],
        label_arrows: Optional[Dict[Tuple[str, str], str]] = None,
    ) -> None:
        for x, y, w, h, text, color in boxes.values():
            ax.add_patch(
                FancyBboxPatch(
                    (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=color, ec="#555", lw=0.8, alpha=0.9
                )
            )
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, wrap=True)
        for a, b in arrows:
            xa, ya, wa, ha_, *_ = boxes[a]
            xb, yb, wb, hb, *_ = boxes[b]
            start, end = (xa + wa, ya + ha_ / 2), (xb, yb + hb / 2)
            if abs(ya - yb) > 0.25 and abs(xa - xb) < 0.1:
                start, end = (xa + wa / 2, ya), (xb + wb / 2, yb + hb)
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color="#444", lw=0.9))
            if label_arrows and (a, b) in label_arrows:
                ax.text(
                    (start[0] + end[0]) / 2,
                    (start[1] + end[1]) / 2 + 0.02,
                    label_arrows[(a, b)],
                    fontsize=7,
                    ha="center",
                    color="#333",
                )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    def overview(self) -> str:
        fig, ax = plt.subplots(figsize=(11, 4.2))
        light = {"s": "#e7f5ff", "sub": "#fff4e6", "top": "#f3d9fa", "out": "#ebfbee"}
        boxes = {
            "gas_s": (0.0, 0.72, 0.17, 0.18, "Газові сенсори\n(графеновий, DFRobot,\nчастинки)", light["s"]),
            "cam_s": (0.0, 0.41, 0.17, 0.18, "Камери\n(люди, обличчя, рух)", light["s"]),
            "mic_s": (0.0, 0.10, 0.17, 0.18, "Мікрофони\n(гучність, звуки, мова)", light["s"]),
            "gas": (
                0.27,
                0.72,
                0.19,
                0.18,
                f"Підсистема «газ»\n{self.f.values['rules.gas']} правил → ризик 1",
                light["sub"],
            ),
            "cam": (
                0.27,
                0.41,
                0.19,
                0.18,
                f"Підсистема «камера»\n{self.f.values['rules.camera']} правил → ризик 2",
                light["sub"],
            ),
            "mic": (
                0.27,
                0.10,
                0.19,
                0.18,
                f"Підсистема «мікрофон»\n{self.f.values['rules.audio']} правил → ризик 3",
                light["sub"],
            ),
            "top": (
                0.56,
                0.38,
                0.19,
                0.24,
                f"Верхній рівень\n{self.f.values['rules.top']} правил\nризики 1–3 → загальний ризик",
                light["top"],
            ),
            "out": (0.84, 0.38, 0.16, 0.24, "Рішення\nрівень + коридор\n«безпечно / увага /\nнебезпека»", light["out"]),
        }
        self._boxes(
            ax,
            boxes,
            [
                ("gas_s", "gas"),
                ("cam_s", "cam"),
                ("mic_s", "mic"),
                ("gas", "top"),
                ("cam", "top"),
                ("mic", "top"),
                ("top", "out"),
            ],
        )
        return self.save(fig, "01_overview")

    def calibration_states(self) -> str:
        fig, ax = plt.subplots(figsize=(11, 3.6))
        c = {"a": "#e7f5ff", "b": "#fff4e6", "g": "#ebfbee", "r": "#ffe3e3"}
        boxes = {
            "start": (0.0, 0.40, 0.13, 0.2, "Запуск:\nпристрій\nрозпізнано", c["a"]),
            "warm": (0.2, 0.68, 0.15, 0.2, "Прогрів\n(збережені значення,\nширша похибка)", c["b"]),
            "check": (0.42, 0.40, 0.15, 0.2, "Коротка\nперевірка\n(секунди)", c["b"]),
            "ready": (0.66, 0.68, 0.15, 0.2, "Відкалібровано\n(збережено, дата)", c["g"]),
            "need": (0.66, 0.10, 0.15, 0.2, "Потребує\nперевірки\n(ширша похибка)", c["r"]),
            "cal": (0.86, 0.40, 0.14, 0.2, "Повне\nкалібрування", c["b"]),
        }
        self._boxes(
            ax,
            boxes,
            [
                ("start", "warm"),
                ("warm", "check"),
                ("start", "check"),
                ("check", "ready"),
                ("check", "need"),
                ("need", "cal"),
                ("cal", "ready"),
            ],
            {("check", "ready"): "так", ("check", "need"): "ні", ("need", "cal"): "N разів поспіль"},
        )
        ax.text(
            0.0,
            0.05,
            "Повне калібрування також: новий пристрій, застаріла калібровка, пошкоджений файл, кнопка на панелі.",
            fontsize=8,
            color="#333",
        )
        return self.save(fig, "calibration_states")

    def agents_flow(self) -> str:
        fig, ax = plt.subplots(figsize=(11, 4))
        c = {"s": "#e7f5ff", "a": "#fff4e6", "v": "#f3d9fa", "o": "#ebfbee", "x": "#ffe3e3"}
        boxes = {
            "ev": (0.0, 0.4, 0.13, 0.2, "Події\nсенсорів", c["s"]),
            "fz": (0.2, 0.72, 0.17, 0.18, "Нечіткий агент\nризик + коридор", c["a"]),
            "vid": (0.2, 0.41, 0.17, 0.18, "Відеоагент\nдії, люди", c["a"]),
            "aud": (0.2, 0.10, 0.17, 0.18, "Аудіоагент\nзміст мови, звуки", c["a"]),
            "board": (0.45, 0.4, 0.15, 0.2, "Дошка\nситуації\n(заяви)", c["s"]),
            "val": (0.67, 0.4, 0.15, 0.2, "Валідаційний\nагент", c["v"]),
            "dec": (0.88, 0.62, 0.12, 0.2, "Рішення", c["o"]),
            "op": (0.88, 0.18, 0.12, 0.2, "Запит\nоператору", c["x"]),
        }
        self._boxes(
            ax,
            boxes,
            [
                ("ev", "fz"),
                ("ev", "vid"),
                ("ev", "aud"),
                ("fz", "board"),
                ("vid", "board"),
                ("aud", "board"),
                ("board", "val"),
                ("val", "dec"),
                ("val", "op"),
            ],
        )
        return self.save(fig, "agents_flow")

    def modes(self) -> str:
        fig, ax = plt.subplots(figsize=(11, 3.2))
        c = {"a": "#ebfbee", "m": "#f3d9fa", "x": "#ffe3e3", "s": "#e7f5ff"}
        boxes = {
            "sw": (0.0, 0.38, 0.16, 0.24, "Перемикач режиму\n(адміністратор,\nз підтвердженням)", c["s"]),
            "auto": (0.26, 0.66, 0.24, 0.26, "Автономний (стаття 1)\nрішення — нечітка система", c["a"]),
            "multi": (
                0.26,
                0.08,
                0.24,
                0.26,
                "Мультиагентний (стаття 2)\nрішення — валідаційний агент\nу межах коридору",
                c["m"],
            ),
            "fb": (0.62, 0.08, 0.2, 0.26, "Автоматичний перехід\nв автономний режим", c["x"]),
            "why": (
                0.62,
                0.62,
                0.36,
                0.3,
                "Коли: агент валідації вимкнений або\nнедоступний, немає мережі, обов'язковий "
                "агент\nне відповів, агенти ще запускаються",
                "#fff",
            ),
        }
        self._boxes(ax, boxes, [("sw", "auto"), ("sw", "multi"), ("multi", "fb")])
        return self.save(fig, "modes")

    # --- gases --------------------------------------------------------------------------------------------------------

    def gas_ladder(self) -> str:
        from modules.gas.data import MODEL_RANGE_VOL_PCT

        fig, ax = plt.subplots(figsize=(10, 3.4))
        names, shares, starts = [], [], []
        for gas, (lo, _) in MODEL_RANGE_VOL_PCT.items():
            s = self.f.table.get(gas)
            crit = s.c_crit_in("vol_pct")
            names.append(s.name_uk)
            shares.append(5.0 / crit * 100)
            starts.append(lo / crit * 100)
        y = np.arange(len(names))
        ax.barh(y + 0.18, shares, 0.34, color=LEVEL_COLORS["critical"], label="5 об.% (як у сценарії)")
        ax.barh(y - 0.18, starts, 0.34, color="#868e96", label="найменша концентрація в даних моделі")
        ax.axvline(100, color="#000", lw=1.2, ls="--")
        ax.text(100, len(names) - 0.4, " 100 % = критичний рівень", fontsize=8)
        ax.set_xscale("log")
        ax.set_yticks(y, names)
        ax.set_xlabel("% від критичного рівня (логарифмічна шкала)")
        for yi, v in zip(y, shares, strict=True):
            ax.text(v * 1.05, yi + 0.18, f"{v:,.0f} %".replace(",", " "), va="center", fontsize=8)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        return self.save(fig, "gas_ladder")

    def sensor_ranges(self) -> str:
        """Measuring range of each sensor as a share of its gas's critical level."""
        from drivers.gas.electrochemical import GAS_TYPES
        import yaml

        rows = []
        for gas, model, rng, *_ in GAS_TYPES.values():
            if rng:
                s = self.f.table.get(gas)
                rows.append((model, rng[0] / s.c_crit_in("ppm"), rng[1] / s.c_crit_in("ppm")))
        catalog = yaml.safe_load(open(os.path.join(self.f.root, "config/sensor_catalog.yaml"), encoding="utf-8"))
        for c in catalog["sensors"]:
            s = self.f.table.get(c["measures"][0])
            if c.get("range") and s is not None:
                rows.append(
                    (c["model"], c["range"][0] / s.c_crit_in(c["unit"]), c["range"][1] / s.c_crit_in(c["unit"]))
                )
        from modules.gas.data import MODEL_RANGE_VOL_PCT

        for gas, (lo, hi) in MODEL_RANGE_VOL_PCT.items():
            s = self.f.table.get(gas)
            rows.append((f"графеновий: {s.name_uk}", lo / s.c_crit_in("vol_pct"), hi / s.c_crit_in("vol_pct")))
        fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1))
        for i, (name, lo, hi) in enumerate(rows):
            ax.plot(
                [max(lo, 0.01) * 100, hi * 100],
                [i, i],
                lw=7,
                solid_capstyle="butt",
                color=COLORS[4] if "графеновий" not in name else COLORS[3],
            )
        ax.axvline(100, color="#000", ls="--", lw=1)
        ax.text(100, len(rows) - 0.3, " критичний рівень", fontsize=8)
        ax.set_xscale("log")
        ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
        ax.set_xlabel("діапазон вимірювання, % від критичного рівня свого газу (логарифмічна шкала)")
        return self.save(fig, "sensor_ranges")

    def dynamics_example(self) -> str:
        fig, ax = plt.subplots(figsize=(10, 3))
        t = np.arange(0, 61, 2.0)
        c = 0.1 + 0.25 * t / 60
        slope = np.polyfit(t, c, 1)[0] * 60
        ax.plot(t, c * 100, "o", ms=3, color=COLORS[4], label="показ (частка критичного рівня), кожні 2 с")
        ax.plot(
            t,
            (np.polyval(np.polyfit(t, c, 1), t)) * 100,
            color=COLORS[2],
            label=f"пряма найменших квадратів: нахил {slope:.2f} критичного рівня за хвилину".replace(".", ","),
        )
        ax.set_xlabel("секунди (вікно 60 с)")
        ax.set_ylabel("% від критичного рівня")
        ax.legend(fontsize=8, frameon=False)
        return self.save(fig, "dynamics_example")

    # --- membership functions -----------------------------------------------------------------------------------------

    def membership(
        self,
        base: str,
        var: str,
        delta: float = 0.0,
        mark: Optional[float] = None,
        title: Optional[str] = None,
        name: Optional[str] = None,
        xlabel: str = "",
    ) -> str:
        engine = self.engines[base]
        v = engine.variables[var]
        x = np.linspace(v.low, v.high, 600)
        fig, ax = plt.subplots(figsize=(10, 2.6))
        for i, (term, parts) in enumerate(v.terms.items()):
            color = LEVEL_COLORS.get(term) or COLORS[i % len(COLORS)]
            mu = np.array([evaluate(parts, float(xx)) for xx in x])
            ax.plot(x, mu, color=color, lw=1.8, label=TERM_UK.get(var, {}).get(term, term))
            if delta > 0:
                band = np.array([term_interval(parts, float(xx), delta, v.low, v.high) for xx in x])
                ax.fill_between(x, band[:, 0], band[:, 1], color=color, alpha=0.18, lw=0)
        if mark is not None:
            ax.axvline(mark, color="#000", lw=1, ls=":")
            values = engine.fuzzify(var, mark)
            strongest = {t: m for t, m in values.items() if m > 0.001}
            ax.text(
                mark,
                1.06,
                "  "
                + ", ".join(
                    f"{TERM_UK.get(var, {}).get(t, t)} {m:.2f}".replace(".", ",") for t, m in strongest.items()
                ),
                fontsize=8,
            )
        ax.set_ylim(-0.02, 1.15)
        ax.set_ylabel("належність μ")
        ax.set_xlabel(xlabel or VAR_UK.get(var, var))
        ax.set_title(title or VAR_UK.get(var, var), loc="left")
        ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper right", bbox_to_anchor=(1, 1.32))
        return self.save(fig, name or f"mf_{base}_{var}")

    # --- surfaces / matrices ------------------------------------------------------------------------------------------

    def surface(
        self,
        base: str,
        xv: str,
        yv: str,
        fixed: Dict[str, float],
        name: str,
        title: str,
        xlabel: str = "",
        ylabel: str = "",
    ) -> str:
        engine = self.engines[base]
        vx, vy = engine.variables[xv], engine.variables[yv]
        xs, ys = np.linspace(vx.low, vx.high, 70), np.linspace(vy.low, vy.high, 70)
        z = np.array([[engine.infer({**fixed, xv: float(a), yv: float(b)})["crisp"] for a in xs] for b in ys])
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        from matplotlib.colors import ListedColormap, BoundaryNorm

        cmap = ListedColormap([LEVEL_COLORS[lv] for lv in LEVELS])
        grid = np.linspace(0, 1, 1001)  # level boundaries of the rule base's output terms
        levels = [max(engine.fuzzify("risk", float(g)).items(), key=lambda kv: kv[1])[0] for g in grid]
        bounds = [0.0] + [float(grid[i]) for i in range(1, len(grid)) if levels[i] != levels[i - 1]] + [1.0]
        mesh = ax.pcolormesh(xs, ys, z, cmap=cmap, norm=BoundaryNorm(bounds, cmap.N), shading="auto", alpha=0.85)
        contour = ax.contour(xs, ys, z, levels=[0.2, 0.35, 0.5, 0.65, 0.8], colors="#fff", linewidths=0.6)
        ax.clabel(contour, fmt=lambda v: f"{v:.2f}".replace(".", ","), fontsize=7)
        cbar = fig.colorbar(mesh, ax=ax, ticks=[0.12, 0.37, 0.63, 0.88])
        cbar.ax.set_yticklabels([LEVEL_UK[lv] for lv in LEVELS])
        ax.set_xlabel(xlabel or VAR_UK.get(xv, xv))
        ax.set_ylabel(ylabel or VAR_UK.get(yv, yv))
        ax.set_title(title, loc="left")
        return self.save(fig, name)

    def matrix(
        self, rows: List[str], cols: List[str], levels: List[List[int]], name: str, title: str, xlabel: str, ylabel: str
    ) -> str:
        fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(rows) + 1.4))
        for i, row in enumerate(levels):
            for j, level in enumerate(row):
                key = LEVELS[level]
                ax.add_patch(plt.Rectangle((j, len(rows) - 1 - i), 1, 1, color=LEVEL_COLORS[key], alpha=0.85))
                ax.text(
                    j + 0.5,
                    len(rows) - 0.5 - i,
                    LEVEL_UK[key],
                    ha="center",
                    va="center",
                    color="#fff",
                    fontsize=8.5,
                    fontweight="bold",
                )
        ax.set_xlim(0, len(cols))
        ax.set_ylim(0, len(rows))
        ax.set_xticks(np.arange(len(cols)) + 0.5, cols)
        ax.set_yticks(np.arange(len(rows)) + 0.5, list(reversed(rows)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        return self.save(fig, name)

    def gas_matrix(self) -> str:
        second = ["немає", "низький", "середній", "високий"]
        maxes = [TERM_UK["max_norm"][t] for t in ("safe", "moderate", "dangerous", "critical")]
        levels = [[policy.gas(m, s, 0, 0, 1) for m in range(4)] for s in range(4)]
        return self.matrix(
            second,
            maxes,
            levels,
            "gas_matrix",
            "Два гази одночасно (динаміка стабільна, індекс суміші < 1)",
            "найвищий газ",
            "другий активний газ",
        )

    def top_matrix(self) -> str:
        second = ["немає", "низький", "середній", "високий"]
        maxes = [LEVEL_UK[lv] for lv in LEVELS]
        levels = [[policy.top(m, s, 0) for m in range(4)] for s in range(4)]
        return self.matrix(
            second,
            maxes,
            levels,
            "top_matrix",
            "Кілька підсистем одночасно (третьої немає)",
            "найвищий ризик підсистеми",
            "друга активна підсистема",
        )

    # --- corridor -----------------------------------------------------------------------------------------------------

    def corridor_example(self, value: float, delta: float, result: dict, name: str = "corridor_example") -> str:
        engine = self.engines["gas"]
        v = engine.variables["max_norm"]
        x = np.linspace(v.low, v.high, 600)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3), gridspec_kw={"width_ratios": [1.2, 1]})
        for i, (term, parts) in enumerate(v.terms.items()):
            color = list(LEVEL_COLORS.values())[i]
            band = np.array([term_interval(parts, float(xx), delta, v.low, v.high) for xx in x])
            axes[0].fill_between(x, band[:, 0], band[:, 1], color=color, alpha=0.25, lw=0)
            axes[0].plot(
                x, [evaluate(parts, float(xx)) for xx in x], color=color, lw=1.4, label=TERM_UK["max_norm"][term]
            )
        axes[0].axvspan(value - delta, value + delta, color="#000", alpha=0.08)
        axes[0].axvline(value, color="#000", ls=":")
        axes[0].set_xlabel("найвищий газ, частка критичного рівня")
        axes[0].set_ylabel("належність μ")
        axes[0].set_title(f"Показ {value:.2f} ± {delta:.2f}: смуга навколо кожної кривої".replace(".", ","), loc="left")
        axes[0].legend(fontsize=7, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.08))
        r = engine.variables["risk"]
        xr = np.linspace(0, 1, 400)
        for term, parts in r.terms.items():
            axes[1].plot(
                xr, [evaluate(parts, float(xx)) for xx in xr], color=LEVEL_COLORS[term], lw=1.2, label=LEVEL_UK[term]
            )
        lo, hi = result["interval"]
        axes[1].axvspan(lo, hi, color="#1c7ed6", alpha=0.25, label="коридор [y_l, y_r]")
        axes[1].axvline(result["risk"], color="#1c7ed6", lw=1.6, label="ризик (тип-1)")
        axes[1].set_xlabel("ризик газу 0…1")
        axes[1].set_title(f"Коридор {lo:.2f}…{hi:.2f}".replace(".", ","), loc="left")
        axes[1].legend(fontsize=7, frameon=False, loc="upper left")
        return self.save(fig, name)

    # --- audio / camera illustrations ---------------------------------------------------------------------------------

    def background_example(self) -> str:
        from modules.audio.intensity import BackgroundLevel

        rng = np.random.default_rng(55)
        level = BackgroundLevel(warmup_sec=10)
        t = np.arange(0, 120, 0.03)
        db = -55 + rng.normal(0, 1.0, t.size)
        db[(t > 60) & (t < 63)] += 25  # someone speaks loudly
        rel = [level.update(float(d), float(tt)) for d, tt in zip(db, t, strict=True)]
        backgrounds = [level.background]
        fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
        ax[0].plot(t, db, lw=0.4, color="#868e96", label="рівень кадрів, дБ")
        ax[0].axhline(backgrounds[-1], color=COLORS[4], lw=1.4, label=f"фон ({backgrounds[-1]:.0f} дБ)")
        ax[0].legend(fontsize=8, frameon=False, loc="upper left")
        ax[0].set_ylabel("дБ (повна шкала)")
        ax[1].plot(t, rel, color=COLORS[2], lw=1)
        ax[1].set_ylabel("гучність над фоном, дБ")
        ax[1].set_xlabel("секунди")
        return self.save(fig, "audio_background")

    def activity_example(self) -> str:
        import datetime as dt

        from core.schedule import Schedule

        schedule = Schedule(self.f.config["hfs"]["camera"]["activity_hours"])
        xs = np.linspace(0, 23.95, 288)
        values = [schedule.activity_time(dt.datetime(2026, 11, 4, int(h), int(round((h % 1) * 60)) % 60)) for h in xs]
        fig, ax = plt.subplots(figsize=(10, 2.6))
        ax.plot(xs, values, color=COLORS[4])
        ax.axhline(0, color="#000", lw=0.6)
        ax.set_xticks(range(0, 25, 2))
        ax.set_xlabel("година доби (середа)")
        ax.set_ylabel("годин до межі розкладу")
        ax.set_title("Мінус — усередині робочих годин, плюс — поза ними", loc="left")
        return self.save(fig, "activity_time")


def build_all(facts: Facts, out_dir: str) -> List[str]:
    fig = Figures(facts, out_dir)
    fig.overview()
    fig.gas_ladder()
    fig.sensor_ranges()
    fig.dynamics_example()
    acetone = facts.table.get("ammonia")
    delta = 12 / acetone.c_crit  # 120 ppm NH₃ with ±10 % (DFRobot): δ 12 ppm
    fig.membership(
        "gas",
        "max_norm",
        delta,
        0.4,
        "Найвищий газ: частка критичного рівня (смуга — похибка сенсора)",
        xlabel="c / критичний рівень",
    )
    fig.membership("gas", "second_norm", 0, None, "Другий активний газ", xlabel="c / критичний рівень")
    fig.membership(
        "gas", "group_index", 0, None, "Індекс суміші в групі (сума часток)", xlabel="Σ c / критичний рівень"
    )
    fig.membership("gas", "dynamics", 0, 0.1, "Швидкість зміни", xlabel="частка критичного рівня за хвилину")
    fig.membership("gas", "risk", 0, None, "Вихід кожної підсистеми: ризик", xlabel="ризик 0…1")
    fig.membership(
        "camera", "unknown_share", 0.05, 0.5, "Частка незнайомих людей у кадрі", xlabel="незнайомі / усі люди в кадрі"
    )
    fig.membership(
        "camera",
        "motion_intensity",
        float(facts.uncertainty["camera"]["motion_intensity"]["delta"]),
        0.05,
        "Інтенсивність руху",
        xlabel="частка кадру, що рухається",
    )
    fig.membership("camera", "activity_time", 0, None, "Час відносно розкладу", xlabel="годин до межі розкладу")
    fig.membership("audio", "intensity", 0, 14, "Гучність над фоном мікрофона", xlabel="дБ над фоном")
    fig.membership("audio", "sound_type_danger", 0, None, "Небезпечність типу звуку", xlabel="ступінь 0…1")
    fig.membership(
        "audio",
        "keyword_danger",
        float(facts.uncertainty["audio"]["keyword_danger"]["delta"]),
        0.7,
        "Небезпечні слова (смуга — похибка розпізнавання мови)",
        xlabel="ступінь словника 0…1",
    )
    fig.membership(
        "audio",
        "speaker_unknown",
        float(facts.uncertainty["audio"]["speaker_unknown"]["delta"]),
        1.0,
        "Чий голос",
        xlabel="0 — відомий, 0,5 — невизначений, 1 — невідомий",
    )
    fig.membership("top", "max_risk", 0, None, "Верхній рівень: найвищий ризик підсистеми", xlabel="ризик 0…1")
    fig.membership("top", "second_risk", 0, None, "Верхній рівень: друга активна підсистема", xlabel="ризик 0…1")
    fig.surface(
        "gas",
        "max_norm",
        "dynamics",
        {"second_norm": 0, "group_index": 0, "third_norm": 0},
        "surface_gas_dynamics",
        "Газ: рівень і швидкість зміни (один газ)",
        "найвищий газ, частка критичного рівня",
        "частка критичного рівня за хвилину",
    )
    fig.surface(
        "gas",
        "max_norm",
        "second_norm",
        {"group_index": 0, "third_norm": 0, "dynamics": 0},
        "surface_gas_two",
        "Газ: два гази різних груп",
        "найвищий газ",
        "другий газ",
    )
    fig.surface(
        "camera",
        "unknown_share",
        "motion_intensity",
        {"activity_time": 6.0},
        "surface_camera_night",
        "Камера: поза робочими годинами",
        "частка незнайомих",
        "інтенсивність руху",
    )
    fig.surface(
        "camera",
        "unknown_share",
        "motion_intensity",
        {"activity_time": -6.0},
        "surface_camera_day",
        "Камера: у робочі години",
        "частка незнайомих",
        "інтенсивність руху",
    )
    fig.surface(
        "audio",
        "intensity",
        "keyword_danger",
        {"sound_type_danger": 0, "speaker_unknown": 1},
        "surface_audio",
        "Мікрофон: гучність і слова (голос невідомий)",
        "дБ над фоном",
        "ступінь слів",
    )
    fig.surface(
        "top",
        "max_risk",
        "second_risk",
        {"third_risk": 0},
        "surface_top",
        "Верхній рівень",
        "найвищий ризик",
        "друга активна підсистема",
    )
    fig.gas_matrix()
    fig.top_matrix()
    example = facts.examples["ammonia_120"]
    fig.corridor_example(
        120 / acetone.c_crit,
        delta,
        {"interval": example["subsystems"]["gas"][2], "risk": example["subsystems"]["gas"][1]},
    )
    fig.background_example()
    fig.activity_example()
    fig.calibration_states()
    fig.agents_flow()
    fig.modes()
    return fig.made
