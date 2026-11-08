"""
Gas safety invariant and explosion flag (G1 / TOD-53), independent of every rule base and agent.

* **Invariant (hard floor):** a gas whose lower bound — reading − its sensor error, as c / C_crit of
  ``config/gases.yaml`` — is ≥ 1 makes the final level **critical**. It is applied where a final level is made:
  the HFS top level (``hfs_risk``, mode autonomous), the agents host (``agent_decision``, mode multiagent) and the
  location state (an acknowledge mutes the notification but cannot make such a location safe). A component that
  produced a lower level is overruled and the violation is logged as an error (tests fail on it).
* **Explosion flag:** a flammable gas (``lel_pct`` in config/gases.yaml) read at or above its lower explosive limit
  → ``explosive: true`` and «вибухонебезпечно» in the decision and the log, on top of the critical level.

``assess`` turns one ``gas_reading`` into hazards; ``SafetyTracker`` keeps the latest reading of every sensor per
location (a reading is its sensor's whole current state) and gives the location's ``safety`` block::

    {"critical": true, "explosive": true, "gases": [{"gas": "acetone", "value": 5.0, "unit": "vol_pct",
     "ratio": 20.0, "pct_of_crit": 2000, "low_ratio": 19.95, "lel_pct": 2.5, "vol_pct": 5.0, "critical": true,
     "explosive": true, "below_range": false}],
     "text": "Ацетон 5 об.% — 2000 % від критичного рівня; вибухонебезпечно: ≥ НКМПР 2,5 об.%"}

A reading below its sensor's detection floor (the graphene model: trained from 0.25–0.4 vol % up, ≥ C_crit of every
gas) counts as the floor and is shown as «≥ floor (нижче діапазону сенсора)».
"""
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from modules.hfs.gas_normalize import GasTable, UnitError, convert, pct_of_crit, reading_items
from system_setup.logger_setup import init_logger

logger = init_logger()

CRITICAL_RATIO = 1.0            # lower bound of c / C_crit at which the level is critical, whatever else is said
CLEAN = ("clean_air", "air", "none")
UNIT_UK = {"vol_pct": "об.%", "ppm": "ppm", "mg_m3": "мг/м³", "ug_m3": "мкг/м³"}
LEVELS = ("low", "medium", "high", "critical")
VIOLATIONS: List[Dict[str, Any]] = []      # every overruled record of this process (tests/conftest.py fails on them)


@dataclass
class Hazard:
    gas: str
    name_uk: str
    value: float
    unit: str
    ratio: float                 # c / C_crit, not clipped
    low_ratio: float             # (c − δ) / C_crit
    lel_pct: Optional[float]
    vol_pct: Optional[float]     # the reading in vol % (gases; None for particles)
    critical: bool
    explosive: bool
    pct_of_crit: float = 0.0     # «% від критичного рівня»
    below_range: bool = False    # a non-zero reading below the sensor's detection floor, counted as the floor


def number_uk(x: float) -> str:
    return (f"{x:.3g}" if x < 100 else f"{x:.0f}").replace(".", ",")


def assess(table: GasTable, payload: Dict[str, Any],
           delta: Callable[[Any, float, str, Dict[str, Any]], float] = lambda substance, value, unit, item: 0.0
           ) -> List[Hazard]:
    """
    :param payload: a ``gas_reading`` (``concentration`` + ``unit``, ``readings: [...]`` or ``concentration_pct``)
    :param delta: (substance, value, unit, reading item) → the reading's error in its own unit
    :return: one hazard per known substance of the reading (clean air / unknown substances: none)
    """
    out = []
    for item in reading_items(payload):
        if str(item["gas"] or "").lower() in CLEAN:
            continue
        substance = table.get(item["gas"])
        if substance is None:
            continue
        value, unit = item["value"], item["unit"]
        try:
            ratio = substance.ratio(value, unit)
            low = substance.ratio(max(0.0, value - abs(float(delta(substance, value, unit, item) or 0.0))), unit)
        except UnitError as e:
            logger.warning(f"gas safety: {item['gas']} {value} {unit}: {e}")
            continue
        try:
            vol = convert(value, unit, "vol_pct", substance.molar_mass)
        except UnitError:                       # particles: a mass concentration, no vol %
            vol = None
        explosive = substance.lel_pct is not None and vol is not None and vol >= substance.lel_pct
        out.append(Hazard(substance.name, substance.name_uk or substance.name, value, unit, round(ratio, 4),
                          round(low, 4), substance.lel_pct, None if vol is None else round(vol, 5),
                          low >= CRITICAL_RATIO, explosive, pct_of_crit(ratio), bool(item.get("below_range"))))
    return out


def describe(h: Hazard) -> str:
    name = h.name_uk[:1].upper() + h.name_uk[1:]
    shown = f"≥ {number_uk(h.value)} {UNIT_UK.get(h.unit, h.unit)} (нижче діапазону сенсора)" if h.below_range \
        else f"{number_uk(h.value)} {UNIT_UK.get(h.unit, h.unit)}"
    text = f"{name} {shown} — {h.pct_of_crit:g} % від критичного рівня"
    if h.explosive:
        text += f"; вибухонебезпечно: ≥ НКМПР {number_uk(h.lel_pct)} об.%"
    return text


def summary(hazards: List[Hazard]) -> Optional[Dict[str, Any]]:
    """:return: the ``safety`` block (None when no gas is critical or explosive)"""
    flagged = sorted((h for h in hazards if h.critical or h.explosive), key=lambda h: -h.ratio)
    if not flagged:
        return None
    return {"critical": any(h.critical for h in flagged), "explosive": any(h.explosive for h in flagged),
            "gases": [asdict(h) for h in flagged], "text": "; ".join(describe(h) for h in flagged)}


def enforce(record: Dict[str, Any], safety: Optional[Dict[str, Any]], who: str, floor: float,
            level_key: str = "level", risk_keys=("risk",), interval_key: Optional[str] = "interval") -> bool:
    """
    Applies the invariant to a decision record in place: ``safety`` + ``explosive`` attached; a record below critical
    is overruled — level critical, risk and corridor raised to ``floor`` (a record already critical keeps its values).

    :param floor: the lowest risk the deciding engine reads as critical
    :return: True when the record had a lower level (logged as an error)
    """
    if not safety:
        return False
    record["safety"] = safety
    record["explosive"] = bool(safety.get("explosive"))
    if not safety.get("critical"):
        return False
    if record.get(level_key) == "critical":
        return False
    logger.error(f"safety invariant violated: {who} gave {record.get(level_key)!r} while {safety['text']} — "
                 f"level set to critical")
    record["safety_overruled"] = {"by": who, "level": record.get(level_key)}
    VIOLATIONS.append({"by": who, "level": record.get(level_key), "text": safety["text"]})
    record[level_key] = "critical"
    for key in risk_keys:
        if isinstance(record.get(key), (int, float)):
            record[key] = round(max(float(record[key]), floor), 4)
    if interval_key and isinstance(record.get(interval_key), (list, tuple)) and len(record[interval_key]) == 2:
        low, high = record[interval_key]
        record[interval_key] = [round(max(float(low), floor), 4), round(max(float(high), floor), 4)]
    return True


def critical_floor(engine) -> float:
    """:return: the smallest output value whose strongest term is ``critical`` with membership 1"""
    variable = engine.variables[engine.output]
    for x in variable.grid:
        if engine.fuzzify(engine.output, float(x)).get("critical", 0.0) >= 0.999:
            return round(float(x), 4)
    return float(variable.high)


def reading_error(uncertainty, substance, value: float, unit: str, item: Dict[str, Any],
                  model: Optional[str] = None) -> float:
    """:return: δ of one reading in its own unit — the sensor's own accuracy (TOD-48 drivers) or the measured error
    of ``config/uncertainty.yaml`` (A5); 0 without either"""
    if "accuracy" in item or "range" in item:
        from modules.hfs.gas_model_builder import delta_at, delta_spec
        return delta_at(delta_spec(item), value)
    return uncertainty.reading_delta(substance.name, value, unit, model) if uncertainty is not None else 0.0


class SafetyTracker:
    """The latest hazards of every gas sensor per location (fresh for ``stale_sec``)."""

    def __init__(self, table: GasTable, uncertainty=None, stale_sec: float = 30.0):
        """:param uncertainty: ``modules.hfs.uncertainty.Uncertainty`` for the lower bounds (None: δ = 0)"""
        self.table, self.uncertainty, self.stale_sec = table, uncertainty, float(stale_sec)
        self.sensors: Dict[str, Dict[str, tuple]] = {}         # location → device → (ts, hazards)
        self.logged: Dict[str, str] = {}

    @classmethod
    def from_files(cls, gases: str = "config/gases.yaml", uncertainty: Optional[str] = "config/uncertainty.yaml",
                   stale_sec: float = 30.0) -> "SafetyTracker":
        from modules.hfs.uncertainty import Uncertainty
        table = GasTable.load(gases)
        return cls(table, Uncertainty.load(uncertainty, table), stale_sec)

    def observe(self, location_id: str, device_id: str, payload: Dict[str, Any], now: float) -> None:
        model = payload.get("sensor_model")
        hazards = assess(self.table, payload, lambda substance, value, unit, item:
                         reading_error(self.uncertainty, substance, value, unit, item, model))
        self.sensors.setdefault(location_id, {})[device_id] = (now, hazards)

    def forget(self, device_id: str) -> None:
        for per_location in self.sensors.values():
            per_location.pop(device_id, None)

    def state(self, location_id: str, now: float) -> Optional[Dict[str, Any]]:
        hazards = [h for ts, found in (self.sensors.get(location_id) or {}).values() if now - ts <= self.stale_sec
                   for h in found]
        block = summary(hazards)
        text = (block or {}).get("text", "")
        if block and self.logged.get(location_id) != text:
            (logger.error if block["explosive"] else logger.warning)(f"gas safety {location_id}: {text}")
        self.logged[location_id] = text
        return block
