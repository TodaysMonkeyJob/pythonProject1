"""
Cross-subsystem escalation (A9 / TOD-51): the inputs of the top-level rule base and the Ukrainian explanation of
every overall decision. Like the multi-gas escalation inside the gas subsystem (TOD-50), one level up.

* **active subsystem** — the lower bound of its risk interval is above ``hfs.top.calm_risk`` (0.15, just above the
  quiet floor 0.13): uncertainty alone never counts as a signal;
* **same place, same time** — the top level works per location; a subsystem that was active within the last
  ``co_window_sec`` (60 s) still counts while another one is active now (a scream 40 s ago + smoke now), with the risk
  it had; older signals do not combine;
* ``max_risk`` = the highest risk (present subsystems and held signals), ``second_risk`` / ``third_risk`` = the second
  / third counted active signal;
* **exceptions** — a benign combination (``hfs.top.benign``: per object type, else ``default``; e.g. camera low +
  audio low = people talking at work) inside the location's activity hours counts once (outside them it escalates);
  two subsystems whose evidence comes only from the devices of one correlated group (``hfs.top.correlated``: a node,
  e.g. a phone streaming camera + microphone of the same person, or a list of devices) count once; an offline
  subsystem is simply missing (partial).

The levels come from the rule table (``policy.top``); this module only forms the inputs and explains.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

NAME_UK = {"gas": "газ", "camera": "камера", "audio": "звук"}
LEVEL_UK = {"low": "низький", "medium": "середній", "high": "високий", "critical": "критичний"}
CALM = "спокійно"
COUNT = {2: "Дві підсистеми", 3: "Три підсистеми"}
LEVEL_INDEX = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass
class Signal:
    name: str
    risk: float
    low: float
    high: float
    level: str                       # low / medium / high / critical (by the risk), CALM when not active
    active: bool
    held_sec: float = 0.0            # > 0: active before, not now — held from that many seconds ago
    detail: str = ""

    @property
    def delta(self) -> float:
        return max(0.0, (self.high - self.low) / 2)

    @property
    def centre(self) -> float:
        return (self.high + self.low) / 2


@dataclass
class Analysis:
    signals: List[Signal] = field(default_factory=list)       # present subsystems (+ held), strongest first
    counted: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    merged: Dict[str, str] = field(default_factory=dict)      # subsystem → the one it was counted with (one device)
    benign: Optional[str] = None                              # why the combination counted once
    location: str = ""

    def signal(self, name: str) -> Optional[Signal]:
        return next((s for s in self.signals if s.name == name), None)

    def ranked(self) -> List[Signal]:
        return [self.signal(n) for n in self.counted]

    @property
    def max_signal(self) -> Optional[Signal]:
        return max(self.signals, key=lambda s: s.risk) if self.signals else None

    def inputs(self) -> Dict[str, float]:
        counted = self.ranked()
        top = self.max_signal
        return {"max_risk": round(top.risk, 4) if top else 0.0,
                "second_risk": round(counted[1].risk, 4) if len(counted) >= 2 else 0.0,
                "third_risk": round(counted[2].risk, 4) if len(counted) >= 3 else 0.0}

    def it2(self) -> Dict[str, Dict[str, float]]:
        """(centres, deltas) of the inputs: each input is its subsystem's corridor (TOD-45)"""
        counted, top = self.ranked(), self.max_signal
        chosen = {"max_risk": top, "second_risk": counted[1] if len(counted) >= 2 else None,
                  "third_risk": counted[2] if len(counted) >= 3 else None}
        return {"centres": {k: s.centre for k, s in chosen.items() if s is not None},
                "deltas": {k: s.delta for k, s in chosen.items() if s is not None}}

    def payload(self) -> Dict[str, Any]:
        return {"active": [s.name for s in self.signals if s.active], "counted": self.counted,
                "held": {s.name: round(s.held_sec, 1) for s in self.signals if s.held_sec > 0},
                "merged": self.merged, "benign": self.benign,
                "levels": {s.name: s.level for s in self.signals}}


class TopEscalation:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None, level_of=None):
        """
        :param cfg: ``hfs.top`` of config.yaml (calm_risk, co_window_sec, benign, correlated)
        :param level_of: risk → low / medium / high / critical (the max_risk terms)
        """
        cfg = cfg or {}
        self.calm = float(cfg.get("calm_risk", 0.15))
        self.window = float(cfg.get("co_window_sec", 60))
        self.benign = cfg.get("benign") or {}
        self.correlated = cfg.get("correlated") or []
        self.level_of = level_of
        self.last_active: Dict[tuple, Dict[str, Any]] = {}           # (location, subsystem) → last active signal

    def _level(self, risk: float) -> str:
        return self.level_of(risk) if self.level_of else ("low" if risk < 0.25 else "medium" if risk < 0.5
                                                          else "high" if risk < 0.75 else "critical")

    def analyse(self, results: Dict[str, Optional[Dict[str, Any]]], location_id: str = "", now: float = 0.0,
                inside_hours: Optional[bool] = None, object_type: Optional[str] = None,
                devices: Optional[Dict[str, Set[str]]] = None, nodes: Optional[Dict[str, str]] = None,
                location_name: str = "", stateless: bool = False) -> Analysis:
        """
        :param results: subsystem → its result (None = missing / offline)
        :param inside_hours: the location is inside its activity hours now (None = unknown → no benign exception)
        :param devices: subsystem → the devices its current evidence comes from
        :param nodes: device → node (``correlated: [{node: …}]``)
        :param stateless: no co-occurrence window (a single evaluation, e.g. ``TopLevel.combine`` alone)
        """
        a = Analysis(location=location_name or location_id)
        a.missing = [n for n, r in results.items() if r is None]
        current: List[Signal] = []
        for name, r in results.items():
            if r is None:
                continue
            low, high = (r.get("interval") or [r["risk"], r["risk"]])
            active = low > self.calm
            signal = Signal(name, float(r["risk"]), float(low), float(high),
                            self._level(r["risk"]) if active else CALM, active, detail=detail_of(name, r))
            current.append(signal)
            if active and not stateless:
                self.last_active[(location_id, name)] = {"at": now, "signal": signal}
        held = []
        if any(s.active for s in current) and not stateless:
            for s in current:
                if s.active:
                    continue
                last = self.last_active.get((location_id, s.name))
                if last is not None and now - last["at"] <= self.window:
                    old = last["signal"]
                    held.append(Signal(s.name, old.risk, old.low, old.high, old.level, True,
                                       held_sec=max(1e-3, now - last["at"]), detail=old.detail))
        signals = {s.name: s for s in current}
        signals.update({s.name: s for s in held})
        a.signals = sorted(signals.values(), key=lambda s: s.risk, reverse=True)
        counted = [s for s in a.signals if s.active]
        counted = self._correlated(counted, devices or {}, nodes or {}, a)
        if len(counted) >= 2 and inside_hours:
            for entry in self.benign.get(object_type or "", self.benign.get("default", [])) or []:
                names = set(entry.get("subsystems") or [])
                limit = LEVEL_INDEX.get(entry.get("max_level", "low"), 0)
                if {s.name for s in counted} <= names and all(LEVEL_INDEX[s.level] <= limit for s in counted):
                    a.benign = (f"{' + '.join(NAME_UK[s.name] for s in counted)} на рівні не вище "
                                f"«{LEVEL_UK[entry.get('max_level', 'low')]}» у робочі години — звичайний збіг")
                    counted = counted[:1]
                    break
        a.counted = [s.name for s in counted]
        return a

    def _correlated(self, counted: List[Signal], devices: Dict[str, Set[str]], nodes: Dict[str, str],
                    a: Analysis) -> List[Signal]:
        """Two subsystems fed only by the devices of one correlated group count once (the stronger one)."""
        out: List[Signal] = []
        for s in counted:
            twin = next((o for o in out if self._same_source(devices.get(o.name), devices.get(s.name), nodes)), None)
            if twin is None:
                out.append(s)
            else:
                a.merged[s.name] = twin.name
        return out

    def _same_source(self, first: Optional[Iterable[str]], second: Optional[Iterable[str]],
                     nodes: Dict[str, str]) -> bool:
        if not first or not second:
            return False
        both = set(first) | set(second)
        for group in self.correlated:
            if "node" in group and {nodes.get(d) for d in both} == {group["node"]}:
                return True
            if "devices" in group and both <= set(group["devices"]):
                return True
        return False


def detail_of(name: str, result: Dict[str, Any]) -> str:
    """A few words on why a subsystem signals (from its inputs)."""
    inputs = result.get("inputs") or {}
    words = []
    if name == "gas":
        if result.get("headline"):
            words.append(result["headline"])
        if float(inputs.get("dynamics") or 0) >= 0.1:
            words.append("концентрація зростає")
    elif name == "camera":
        if float(inputs.get("unknown_share") or 0) > 0:
            words.append("невідомі особи")
        if float(inputs.get("motion_intensity") or 0) >= 0.1:
            words.append("рух")
        if float(inputs.get("activity_time") or 0) > 0:
            words.append("поза робочими годинами")
    elif name == "audio":
        if float(inputs.get("keyword_danger") or 0) > 0:
            words.append("небезпечні слова")
        if float(inputs.get("sound_type_danger") or 0) >= 0.5:
            words.append("небезпечний звук")
        if float(inputs.get("speaker_unknown") or 0) >= 0.5:
            words.append("невідомий голос")
    return ", ".join(words)


def explain(a: Analysis, level: str, third_active: bool) -> str:
    """:param level: the decided overall level (low … critical)"""
    parts = []
    for s in sorted(a.signals, key=lambda x: ["gas", "camera", "audio"].index(x.name)):
        held = f", {s.held_sec:.0f} с тому" if s.held_sec > 0 else ""
        inner = ", ".join(x for x in (s.detail, held.lstrip(", ")) if x)
        parts.append(f"{NAME_UK[s.name]}: {LEVEL_UK.get(s.level, s.level)}" + (f" ({inner})" if inner else ""))
    parts += [f"{NAME_UK[n]}: немає даних" for n in a.missing]
    text = [(", ".join(parts)[0].upper() + ", ".join(parts)[1:]) + "."]
    final = LEVEL_UK[level]
    for name, twin in a.merged.items():
        text.append(f"{NAME_UK[name].capitalize()} і {NAME_UK[twin]} — від одного пристрою: враховано один раз.")
    counted = a.ranked()
    if a.benign:
        text.append(f"{a.benign.capitalize()} — без підвищення: {final}.")
    elif len(counted) >= 2:
        held = [s for s in counted if s.held_sec > 0]
        span = f" протягом {max(s.held_sec for s in held):.0f} с" if held else ""
        text.append(f"{COUNT.get(len(counted), f'{len(counted)} підсистеми')} одночасно сигналізують у локації "
                    f"«{a.location}»{span} — рівень підвищено на один щабель: {final}.")
    elif counted:
        text.append(f"Сигналізує одна підсистема — без підвищення: {final}.")
    else:
        text.append(f"Жодна підсистема не сигналізує: {final}.")
    if a.missing:
        text.append(f"Без {', '.join(NAME_UK[n] for n in a.missing)} (офлайн / немає даних): рішення за рештою "
                    "підсистем, результат частковий.")
    at_high = [s for s in counted if s.level in ("high", "critical")]
    if level == "critical":
        reasons = []
        critical = [s for s in a.signals if s.level == "critical"]
        if critical:
            reasons.append(f"{NAME_UK[critical[0].name]} на критичному рівні")
        if len(at_high) >= 2:
            reasons.append("дві підсистеми на рівні «високий» і вище")
        if third_active and at_high and len(counted) >= 3:
            reasons.append("усі три підсистеми активні, одна на рівні «високий»")
        if reasons:
            text.append("Критичний: " + "; ".join(reasons) + ".")
    elif len(counted) >= 2 and not a.benign:
        text.append("Критичний не присвоєно: жодна підсистема не критична, "
                    + ("лише одна на рівні «високий» і вище" if at_high else "жодна не на рівні «високий»")
                    + ("" if len(counted) >= 3 else ", активні не всі три") + ".")
    return " ".join(text)
