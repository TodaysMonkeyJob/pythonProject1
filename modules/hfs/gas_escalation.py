"""
Multi-gas escalation (A8 / TOD-50): the inputs of the gas rule base beyond the highest gas, and the Ukrainian
explanation of every multi-gas decision.

* **active gas** — the lower bound of its c / c_crit interval (reading − sensor error, TOD-45 / TOD-48) is above
  ``activity.calm_norm`` of ``config/gases.yaml`` ("спокійно" otherwise): sensor noise around clean air never counts;
* ``second_norm`` / ``third_norm`` — the second / third highest c / c_crit among the active gases that count;
* ``group_index`` — max over effect groups of E_g = Σ c_i / c_crit,i of the counted gases of that group (the
  additive mixture rule; gases of different groups are not added — their co-occurrence still raises one step via
  ``second_norm``);
* **not counted**: every active reading comes from a cross-sensitive sensor (MOS, chemiresistor) and no selective
  sensor (electrochemical, optical particle) confirms → no escalation, the reason is logged; a known benign
  co-occurrence (``benign`` of ``config/gases.yaml``, or the location rule ``gas_benign``: e.g. CO₂ while people are
  in the room) → that gas is shown but not counted.

The levels themselves come from the rule table (``modules/hfs/policy.py`` ``gas``): this module only measures and
explains.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from modules.hfs.gas_normalize import pct_of_crit

LEVEL_UK = {"low": "низький", "medium": "середній", "high": "високий", "critical": "критичний"}
TERM_LEVEL = {"safe": "низький", "moderate": "середній", "dangerous": "високий", "critical": "критичний"}
CALM = "спокійно"
COUNT = {2: "Два гази", 3: "Три гази", 4: "Чотири гази"}


def num(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


@dataclass
class GasState:
    """One substance in a location (the strongest fresh reading of it)."""
    name: str
    norm: float
    low: float                          # lower bound of c / c_crit (reading − error)
    group: Optional[str] = None
    name_uk: str = ""
    selectivity: Optional[str] = None   # selective | cross_sensitive | None (unknown)
    level: str = ""                     # Ukrainian per-gas level
    delta: float = 0.0                  # δ of c / c_crit (FOU half-width, A5)
    ratio: Optional[float] = None       # c / c_crit not clipped (TOD-53: «% від критичного рівня»); None = norm

    @property
    def pct(self) -> str:
        """:return: «2000 %» / «42 %» of the critical level"""
        value = self.norm if self.ratio is None else self.ratio
        return f"{pct_of_crit(value):g} %".replace(".", ",")


@dataclass
class Analysis:
    max_norm: float = 0.0
    second_norm: float = 0.0
    third_norm: float = 0.0
    group_index: float = 0.0
    group: Optional[str] = None
    groups: Dict[str, float] = field(default_factory=dict)
    gases: List[GasState] = field(default_factory=list)     # every fresh substance, strongest first
    counted: List[str] = field(default_factory=list)
    excluded: Dict[str, str] = field(default_factory=dict)  # gas → why it is not counted (benign)
    suppressed: Optional[str] = None                        # cross_sensitive

    def payload(self) -> Dict[str, Any]:
        return {"active": [g.name for g in self.gases if g.level != CALM], "counted": self.counted,
                "excluded": self.excluded, "suppressed": self.suppressed,
                "groups": {g: round(v, 4) for g, v in self.groups.items()},
                "group": self.group, "levels": {g.name: g.level for g in self.gases}}


def analyse(gases: List[GasState], calm_norm: float, level_of: Callable[[float], str],
            benign: Optional[Dict[str, str]] = None) -> Analysis:
    """
    :param level_of: c / c_crit → the max_norm term (safe / moderate / dangerous / critical)
    :param benign: gas → reason, for the gases a benign exception excludes now
    """
    benign = benign or {}
    gases = sorted(gases, key=lambda g: g.norm, reverse=True)
    for g in gases:
        g.level = CALM if g.low <= calm_norm else TERM_LEVEL[level_of(g.norm)]
    active = [g for g in gases if g.level != CALM]
    out = Analysis(max_norm=max((g.norm for g in gases), default=0.0), gases=gases)
    out.excluded = {g.name: benign[g.name] for g in active if g.name in benign}
    counted = [g for g in active if g.name not in out.excluded]
    if len(counted) >= 2 and all(g.selectivity == "cross_sensitive" for g in counted):
        out.suppressed = "cross_sensitive"
        counted = counted[:1]                   # one real gas can look like several on cross-sensitive sensors
    out.counted = [g.name for g in counted]
    out.second_norm = counted[1].norm if len(counted) >= 2 else 0.0
    out.third_norm = counted[2].norm if len(counted) >= 3 else 0.0
    for g in counted:
        key = g.group or g.name
        out.groups[key] = out.groups.get(key, 0.0) + g.norm
    if out.groups:
        out.group = max(out.groups, key=out.groups.get)
        out.group_index = out.groups[out.group]
    return out


def explain(a: Analysis, level: str, groups_uk: Dict[str, str], dynamics_sharp: bool = False,
            third_high: Optional[bool] = None) -> Optional[str]:
    """
    :param level: the decided gas level (low / medium / high / critical, from the rule table)
    :return: the Ukrainian explanation (None without an active gas)
    """
    active = [g for g in a.gases if g.level != CALM]
    if not active:
        return None
    shown = [g for g in a.gases if g.level != CALM or g.norm > 0][:5]
    parts = [f"{g.name_uk or g.name}: {g.level} ({g.pct}" + (" від критичного рівня)" if i == 0 else ")")
             for i, g in enumerate(shown)]
    text = [", ".join(parts)[0].upper() + ", ".join(parts)[1:] + "."]
    final = LEVEL_UK[level]
    for name, why in a.excluded.items():
        g = next(x for x in a.gases if x.name == name)
        text.append(f"{g.name_uk or name} не враховано в підвищенні: {why}.")
    if a.suppressed == "cross_sensitive":
        text.append("Підвищення не застосовано: усі активні показання — від крос-чутливих сенсорів, і жоден "
                    f"селективний сенсор (електрохімічний, оптичний) їх не підтверджує. Рівень: {final}.")
    elif len(a.counted) >= 2:
        n = len(a.counted)
        text.append(f"{COUNT.get(n, f'{n} газів')} одночасно вище норми — рівень підвищено на один щабель: {final}.")
    else:
        text.append(f"Активний один газ — без підвищення: {final}.")
    if dynamics_sharp:
        text.append("Концентрація різко зростає — ще один щабель.")
    group_uk = groups_uk.get(a.group or "", a.group or "")
    critical_gas = next((g for g in active if g.level == "критичний"), None)
    if level == "critical":
        reasons = []
        if critical_gas is not None:
            name = critical_gas.name_uk or critical_gas.name
            reasons.append(f"{name} на критичному рівні ({critical_gas.pct} від критичного рівня)")
        if a.group_index >= 1.0 and len(a.counted) >= 2:
            reasons.append(f"індекс суміші групи «{group_uk}» {num(a.group_index)} ≥ 1")
        if third_high:
            reasons.append("три гази на високому рівні")
        if reasons:
            text.append("Критичний: " + "; ".join(reasons) + ".")
    elif len(a.counted) >= 2:
        why = ["жоден газ не досягає критичного рівня"]
        if a.group is not None:
            why.append(f"найбільший індекс суміші — у групі «{group_uk}» {num(a.group_index)} < 1"
                       + ("" if len({x for x in a.groups}) == 1 else " (гази різних груп не додаються)"))
        if not third_high:
            why.append("менше трьох газів на високому рівні")
        text.append("Критичний не присвоєно: " + ", ".join(why) + ".")
    return " ".join(text)
