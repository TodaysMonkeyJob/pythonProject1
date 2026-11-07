"""
Gas normalisation (A2 / TOD-42): every substance as a share of its critical concentration (``config/gases.yaml``).

* units: ``vol_pct`` (1 vol % = 10 000 ppm), ``ppm``, ``mg_m3`` (mg/m³ = ppm · M / 24.45 at 25 °C, 1 atm), ``ug_m3``;
* ``c_norm = clip(c / c_crit, 0, 1.2)`` — 1.0 = the critical (IDLH / EPA Hazardous) level, a little above 1 is kept
  to show an exceedance;
* ``max_norm`` = max over substances; ``mixture_index`` = Σ c_norm (the additive mixture rule of occupational
  exposure limits for substances with similar effects — since TOD-50 summed only inside an effect group,
  ``gas_escalation.py``);
* ``dynamics`` = least-squares slope of max_norm per minute over the last window.

The sensors and their models keep their native units (TOD-48); c_norm is only used to compare substances here.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

MOLAR_VOLUME = 24.45            # l/mol at 25 °C, 1 atm
PPM_PER_VOL_PCT = 10_000.0
NORM_MAX = 1.2
UNITS = ("ppm", "vol_pct", "mg_m3", "ug_m3")


class UnitError(ValueError):
    pass


def ppm_to_mg_m3(ppm: float, molar_mass: float) -> float:
    return ppm * molar_mass / MOLAR_VOLUME


def mg_m3_to_ppm(mg_m3: float, molar_mass: float) -> float:
    return mg_m3 * MOLAR_VOLUME / molar_mass


def convert(value: float, unit: str, to: str, molar_mass: Optional[float] = None) -> float:
    """
    :param value: concentration in ``unit``
    :return: the same concentration in ``to`` (gas units need ``molar_mass`` between volume and mass units)
    :raise UnitError: unknown unit, or a volume ↔ mass conversion without a molar mass
    """
    for u in (unit, to):
        if u not in UNITS:
            raise UnitError(f"unknown unit {u!r} (use {', '.join(UNITS)})")
    if unit == to:
        return float(value)
    mass = {"mg_m3": 1.0, "ug_m3": 1e-3}
    volume = {"ppm": 1.0, "vol_pct": PPM_PER_VOL_PCT}
    if unit in volume and to in volume:
        return float(value) * volume[unit] / volume[to]
    if unit in mass and to in mass:
        return float(value) * mass[unit] / mass[to]
    if not molar_mass:
        raise UnitError(f"{unit} → {to} needs a molar mass (particles are mass concentrations only)")
    if unit in volume:
        return ppm_to_mg_m3(float(value) * volume[unit], molar_mass) / mass[to]
    return mg_m3_to_ppm(float(value) * mass[unit], molar_mass) / volume[to]


@dataclass
class Substance:
    name: str
    c_crit: float
    unit: str
    molar_mass: Optional[float] = None
    source: str = ""
    formula: str = ""
    aliases: List[str] = field(default_factory=list)
    group: Optional[str] = None          # TOD-50 effect group (additive mixture inside a group only)
    name_uk: str = ""

    def c_crit_in(self, unit: str) -> float:
        """:return: the critical concentration in another unit (e.g. NH₃ 300 ppm → 0.03 vol %)"""
        return convert(self.c_crit, self.unit, unit, self.molar_mass)

    def norm(self, value: float, unit: str) -> float:
        """:return: clip(c / c_crit, 0, 1.2)"""
        return min(NORM_MAX, max(0.0, convert(value, unit, self.unit, self.molar_mass) / self.c_crit))


class GasTable:
    """The substances of ``config/gases.yaml`` by name and alias."""

    def __init__(self, substances: Sequence[Substance], calm_norm: float = 0.05,
                 groups: Optional[Dict[str, Dict[str, Any]]] = None, benign: Optional[List[Dict[str, Any]]] = None):
        self.substances = {s.name: s for s in substances}
        self.calm_norm, self.groups, self.benign = float(calm_norm), dict(groups or {}), list(benign or [])
        self._names = {alias.lower(): s.name for s in substances for alias in [s.name, s.formula, *s.aliases] if alias}

    @classmethod
    def load(cls, path: str = "config/gases.yaml") -> "GasTable":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        out = []
        for name, entry in (data.get("gases") or {}).items():
            if entry.get("unit") not in UNITS or not float(entry.get("c_crit") or 0) > 0:
                raise UnitError(f"gases.yaml {name}: needs c_crit > 0 and unit in {UNITS}")
            group = entry.get("group")
            if group is not None and group not in (data.get("groups") or {}):
                raise UnitError(f"gases.yaml {name}: group {group!r} not in groups")
            out.append(Substance(name, float(entry["c_crit"]), entry["unit"], entry.get("molar_mass"),
                                 str(entry.get("source", "")), str(entry.get("formula", "")),
                                 [str(a) for a in entry.get("aliases") or []], group,
                                 str(entry.get("name_uk") or entry.get("formula") or name)))
        return cls(out, (data.get("activity") or {}).get("calm_norm", 0.05), data.get("groups"), data.get("benign"))

    def get(self, name: Optional[str]) -> Optional[Substance]:
        return self.substances.get(self._names.get(str(name or "").lower(), ""))


def readings_of(payload: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    """
    :param payload: a ``gas_reading`` — ``{gas, concentration, unit}``, ``{readings: [{gas, concentration, unit}]}``
                    (one sensor measuring several substances) or the TOD-19 graphene worker's ``{gas,
                    concentration_pct}`` (vol %)
    :return: [(gas, value, unit)]
    """
    items = payload.get("readings") or [payload]
    out = []
    for item in items:
        if item.get("concentration") is not None:
            out.append((item.get("gas"), float(item["concentration"]), str(item.get("unit") or "ppm")))
        elif item.get("concentration_pct") is not None:
            out.append((item.get("gas"), float(item["concentration_pct"]), "vol_pct"))
    return out


def reading_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """:return: the readings of a ``gas_reading`` as dicts {gas, value, unit, range?, resolution?, accuracy?} — the
    sensor's own range / accuracy travel with each reading (TOD-48 drivers)"""
    items = payload.get("readings") or [payload]
    out = []
    for item in items:
        if item.get("concentration") is not None:
            value, unit = float(item["concentration"]), str(item.get("unit") or "ppm")
        elif item.get("concentration_pct") is not None:
            value, unit = float(item["concentration_pct"]), "vol_pct"
        else:
            continue
        out.append({"gas": item.get("gas"), "value": value, "unit": unit,
                    **{k: item[k] for k in ("range", "resolution", "accuracy") if item.get(k) is not None}})
    return out


def mixture_index(norms: Sequence[float]) -> float:
    return float(sum(norms))


def slope_per_minute(samples: Sequence[Tuple[float, float]], min_samples: int = 3, min_span_sec: float = 0.0) -> float:
    """
    :param samples: [(time s, value)] inside the window
    :return: least-squares slope in value per minute; 0 with too few samples / too short a span (no trend yet)
    """
    if len(samples) < min_samples:
        return 0.0
    ts = [t for t, _ in samples]
    if max(ts) - min(ts) < max(min_span_sec, 1e-9):
        return 0.0
    mean_t = sum(ts) / len(ts)
    mean_v = sum(v for _, v in samples) / len(samples)
    num = sum((t - mean_t) * (v - mean_v) for t, v in samples)
    den = sum((t - mean_t) ** 2 for t in ts)
    return 60.0 * num / den if den else 0.0
