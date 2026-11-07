"""
Measured input errors for the HFS (A5 / TOD-45): ``config/uncertainty.yaml`` (``python -m tools.fit_uncertainty``)
→ the half-width δ of every input, in the input's units, for one inference. δ moves the membership edges by ±δ
(interval type-2 FOU, ``core.fuzzy.engine2``); Karnik–Mendel turns it into the corridor [y_l, y_r].

Constant δ come straight from the file (motion noise, danger-word accuracy, speaker EER); live ones are computed from
the data behind the inference with the formula recorded in the file (gas reading error → c_norm, dynamics slope
error, binomial unknown share, background spread, a sound type's confidence). ``unmeasured`` entries give 0 —
the corridor then does not contain that error, and the file says so.
"""
import math
import os
from typing import Any, Dict, Optional, Sequence

import yaml

from modules.hfs.gas_normalize import GasTable, UnitError, convert

DEFAULT_PATH = os.path.join("config", "uncertainty.yaml")


class Uncertainty:
    def __init__(self, data: Optional[Dict[str, Any]] = None, table: Optional[GasTable] = None):
        """:param data: the parsed ``config/uncertainty.yaml`` (None / {} = every δ 0: the type-1 system)"""
        self.data = data or {}
        self.table = table
        gas = (self.data.get("gas") or {}).get("concentration") or {}
        self.sensors = gas.get("sensors") or {}
        self.by_unit = gas.get("default_by_unit") or {}
        self.by_gas = gas.get("default_by_gas") or {}

    @classmethod
    def load(cls, path: Optional[str] = DEFAULT_PATH, table: Optional[GasTable] = None) -> "Uncertainty":
        if not path or not os.path.isfile(path):
            return cls({}, table)
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f) or {}, table)

    def entry(self, section: str, name: str) -> Dict[str, Any]:
        return (self.data.get(section) or {}).get(name) or {}

    def constant(self, section: str, name: str) -> float:
        entry = self.entry(section, name)
        return 0.0 if entry.get("kind") in (None, "unmeasured", "live") else float(entry.get("delta") or 0.0)

    # --- gas -----------------------------------------------------------------------------------------------------

    def sensor_for(self, gas: str, unit: str, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """The reading's sensor: named in the reading (TOD-48), else the default of the gas, else of the unit
        (vol % = our graphene FET)."""
        if model is None:
            model = self.by_unit.get(unit) if unit == "vol_pct" else self.by_gas.get(gas) or self.by_unit.get(unit)
        return self.sensors.get(model or "")

    def reading_delta(self, gas: str, value: float, unit: str, model: Optional[str] = None) -> float:
        """:return: δ of one reading in its own unit: max(absolute, relative · value) of its sensor (0 unmeasured)"""
        sensor = self.sensor_for(gas, unit, model)
        if not sensor or sensor.get("kind") == "unmeasured":
            return 0.0
        if "per_gas" in sensor:
            per_gas = sensor["per_gas"]
            entry = per_gas.get(gas)
            if entry is None and self.table is not None:      # the classifier says "toluol", the table "toluene"
                entry = next((v for k, v in per_gas.items() if (s := self.table.get(k)) and s.name == gas), None)
            absolute = float((entry or {}).get("absolute") or 0.0)
            return convert(absolute, sensor.get("unit", unit), unit) if sensor.get("unit") != unit else absolute
        delta = max(float(sensor.get("absolute") or 0.0), float(sensor.get("relative") or 0.0) * abs(value))
        return delta if sensor.get("unit", unit) == unit else convert(delta, sensor["unit"], unit)

    def norm_delta(self, gas: str, value: float, unit: str, model: Optional[str] = None) -> float:
        """:return: δ of c_norm = c / c_crit for one reading"""
        if self.table is None:
            return 0.0
        substance = self.table.get(gas)
        if substance is None:
            return 0.0
        try:
            delta = self.reading_delta(substance.name, value, unit, model)
            return convert(delta, unit, substance.unit, substance.molar_mass) / substance.c_crit
        except UnitError:
            return 0.0

    @staticmethod
    def dynamics_delta(norm_delta: float, times: Sequence[float]) -> float:
        """:return: slope error per minute: δ_c_norm · 60 / sqrt(Σ (t − t̄)²) (0 without a usable window)"""
        if len(times) < 3 or norm_delta <= 0:
            return 0.0
        mean = sum(times) / len(times)
        spread = sum((t - mean) ** 2 for t in times)
        return 60.0 * norm_delta / math.sqrt(spread) if spread > 0 else 0.0

    # --- camera / audio ----------------------------------------------------------------------------------------------

    @staticmethod
    def interval_delta(value: float, interval: Optional[Sequence[float]]) -> float:
        """:return: the half-width that covers [lo, hi] around the value (asymmetric intervals: the larger side)"""
        if not interval:
            return 0.0
        return max(0.0, value - float(interval[0]), float(interval[1]) - value)

    def sound_type_delta(self, sound_type: Optional[str], confidence: Optional[float],
                         per_type_accuracy: Optional[Dict[str, float]] = None) -> float:
        """:return: 1 − the type's measured accuracy (Wilson lower bound) when known, else 1 − its confidence"""
        if sound_type is None:
            return 0.0
        if per_type_accuracy is None:
            per_type_accuracy = self.entry("audio", "sound_type_danger").get("per_type_accuracy")
        if per_type_accuracy and sound_type in per_type_accuracy:
            return max(0.0, 1.0 - float(per_type_accuracy[sound_type]))
        return max(0.0, 1.0 - float(confidence or 0.0))
