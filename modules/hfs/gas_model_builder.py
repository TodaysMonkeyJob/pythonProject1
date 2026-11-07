"""
Per-sensor gas model (A7 / TOD-48): every quantity a detected sensor reports becomes a fuzzy input **in that
sensor's native unit** — ``config/fuzzy/gas_generated.yaml`` (machine-specific, git-ignored; a provenance entry for
every value):

* **C_crit in the native unit** — ``config/gases.yaml`` converted (``gas_normalize.convert``: 1 vol % = 10 000 ppm,
  mg/m³ = ppm · M / 24.45): NH₃ 300 ppm → 0.03 vol % for the graphene model, CO 1200 ppm → 1374.4 mg/m³;
* **terms** at the C_crit fractions of the ``max_norm`` terms of ``config/fuzzy/gas.yaml`` (0.1 / 0.3 / 0.6 / 1.0 —
  one source, so the per-sensor input and the c / C_crit the rules use are the same partition), scaled by C_crit;
* **δ** (FOU half-width, TOD-45) from the descriptor: δ(c) = max(absolute, relative · c, resolution / 2); no
  accuracy published → ``null`` (the corridor gets no width from it, and the log says so);
* **range below C_crit** (e.g. NH₃ 0–100 ppm vs 300 ppm): flagged — a full-scale reading is "at least full scale",
  an open upper bound (``reading_interval``), and the sensor cannot confirm a critical concentration;
* **index-only** quantities (ENS160 AQI-UBA …): a separate input with its own terms, never mixed with
  concentrations.

``python -m modules.hfs.gas_model_builder [--state data/gas_sensors.json]`` rebuilds it from the last discovery; the
gas module rebuilds it after every discovery. The gas subsystem itself compares substances by c / C_crit and uses the
same δ / saturation rules on every reading (the readings carry their sensor's range and accuracy).
"""
import argparse
import datetime as dt
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from core.fuzzy import load_yaml
from modules.hfs.gas_normalize import NORM_MAX, GasTable, UnitError, convert
from system_setup.logger_setup import init_logger

logger = init_logger()

DEFAULT_OUT = os.path.join("config", "fuzzy", "gas_generated.yaml")
RULE_BASE = os.path.join("config", "fuzzy", "gas.yaml")
# AQI-UBA levels 1 excellent · 2 good · 3 moderate · 4 poor · 5 unhealthy (ENS160): a Ruspini partition on the levels
INDEX_TERMS = {"aqi_uba": {"terms": {"good": ["trap", 1, 1, 2, 3], "moderate": ["tri", 2, 3, 4],
                                     "poor": ["tri", 3, 4, 5], "unhealthy": ["trap", 4, 5, 5, 5]},
                           "universe": [1, 5, 0.05],
                           "source": "UBA levels 1–5 (ENS160 datasheet); terms: design (one level per term, Ruspini)"}}


def fractions(rule_base: str = RULE_BASE) -> Dict[str, List[Any]]:
    """:return: the ``max_norm`` terms of the gas rule base — breakpoints as fractions of C_crit"""
    return dict(load_yaml(rule_base)["variables"]["max_norm"]["terms"])


def scale_term(spec: List[Any], factor: float, top: float) -> List[Any]:
    shape, *points = spec
    scaled = [round(min(p * factor, top) if p < NORM_MAX else top, 6) for p in points]
    return [shape, *scaled]


def delta_spec(quantity: Dict[str, Any]) -> Optional[Dict[str, float]]:
    accuracy = quantity.get("accuracy") or {}
    resolution = quantity.get("resolution")
    if not accuracy and not resolution:
        return None
    return {"absolute": float(accuracy.get("absolute", 0.0)), "relative": float(accuracy.get("relative", 0.0)),
            "resolution": float(resolution or 0.0)}


def delta_at(spec: Optional[Dict[str, float]], value: float) -> float:
    """δ(c) = max(absolute, relative · c, resolution / 2) in the sensor's unit (0 when unknown)"""
    if not spec:
        return 0.0
    return max(spec.get("absolute", 0.0), spec.get("relative", 0.0) * abs(value), spec.get("resolution", 0.0) / 2)


def saturated(value: float, rng: Optional[Iterable[float]], resolution: Optional[float] = None) -> bool:
    """A reading at the top of the sensor's range: the true value may be anything above it."""
    if not rng:
        return False
    high = float(list(rng)[1])
    return value >= high - max(float(resolution or 0.0), 1e-9 * max(1.0, high))


def reading_interval(value: float, unit: str, quantity: Dict[str, Any], substance) -> Tuple[float, float, bool]:
    """
    :param quantity: {range, resolution, accuracy} of the sensor (descriptor / reading)
    :return: (low, high, open) of c / C_crit — ±δ, and an open upper bound (high = NORM_MAX) at full scale
    """
    norm = substance.norm(value, unit)
    delta = convert(delta_at(delta_spec(quantity), value), unit, substance.unit, substance.molar_mass) \
        / substance.c_crit
    is_open = saturated(value, quantity.get("range"), quantity.get("resolution"))
    return max(0.0, norm - delta), (NORM_MAX if is_open else min(NORM_MAX, norm + delta)), is_open


def build(sensors: List[Dict[str, Any]], table: GasTable, rule_base: str = RULE_BASE) -> Dict[str, Any]:
    """:param sensors: descriptors (``Descriptor.to_dict``) :return: the gas_generated.yaml content"""
    terms = fractions(rule_base)
    out: Dict[str, Any] = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                           "generator": "python -m modules.hfs.gas_model_builder",
                           "term_fractions": {"terms": terms, "source": f"{rule_base} max_norm (fractions of C_crit)"},
                           "sensors": {}, "warnings": []}
    for sensor in sensors:
        inputs = {}
        for q in sensor.get("quantities") or []:
            name = f"{sensor['device_id']}__{q['key']}"
            if q.get("kind") == "index":
                spec = INDEX_TERMS.get(q["gas"])
                inputs[name] = {"quantity": q["key"], "kind": "index", "unit": q["unit"], "range": q.get("range"),
                                "universe": spec["universe"] if spec else None,
                                "terms": spec["terms"] if spec else None,
                                "provenance": {"terms": spec["source"] if spec else
                                               "no agreed scale for this index: no terms (shown, not inferred)",
                                               "range": q.get("source", "")}}
                continue
            substance = table.get(q["gas"])
            if substance is None:
                out["warnings"].append(f"{name}: no C_crit for {q['gas']!r} in config/gases.yaml — no input")
                continue
            try:
                c_crit = substance.c_crit_in(q["unit"])
            except UnitError as e:
                out["warnings"].append(f"{name}: {e}")
                continue
            rng = q.get("range")
            top = max(float(rng[1]) if rng else 0.0, NORM_MAX * c_crit)
            entry = {"quantity": q["key"], "kind": "concentration", "gas": substance.name, "unit": q["unit"],
                     "c_crit": {"value": round(c_crit, 6), "unit": q["unit"],
                                "from": {"value": substance.c_crit, "unit": substance.unit}},
                     "universe": [0.0, round(top, 6), round(top / 240, 6)],
                     "terms": {t: scale_term(spec, c_crit, top) for t, spec in terms.items()},
                     "delta": delta_spec(q), "range": rng,
                     "provenance": {"c_crit": f"config/gases.yaml {substance.name}: {substance.source}",
                                    "terms": f"{rule_base} max_norm fractions × C_crit in {q['unit']}",
                                    "universe": "0 … max(sensor range, 1.2 · C_crit)",
                                    "delta": q.get("source") or "descriptor",
                                    "range": q.get("source") or "descriptor"}}
            if entry["delta"] is None:
                entry["provenance"]["delta"] = "no accuracy published: δ unknown (no corridor width)"
            if rng and float(rng[1]) < c_crit:
                share = float(rng[1]) / c_crit
                entry["saturation"] = {"range_high": float(rng[1]), "share_of_c_crit": round(share, 4),
                                       "open_upper_bound": True}
                out["warnings"].append(
                    f"{name}: range ends at {rng[1]} {q['unit']} = {share:.2f} C_crit — a full-scale reading is an "
                    f"open upper bound; this sensor cannot confirm a critical {substance.name} concentration")
            inputs[name] = entry
        out["sensors"][sensor["device_id"]] = {k: sensor.get(k) for k in ("driver", "model", "bus", "address",
                                                                          "serial")} | {"inputs": inputs}
    return out


def write(content: Dict[str, Any], path: str = DEFAULT_OUT) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("# Per-sensor gas inputs in native units (TOD-48) — generated, machine-specific; do not edit.\n")
        yaml.safe_dump(content, f, allow_unicode=True, sort_keys=False, width=120)
    os.replace(tmp, path)
    for warning in content["warnings"]:
        logger.warning(f"gas model: {warning}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build config/fuzzy/gas_generated.yaml from the detected sensors")
    parser.add_argument("--state", default=os.path.join("data", "gas_sensors.json"))
    parser.add_argument("--gases", default=os.path.join("config", "gases.yaml"))
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    with open(args.state, encoding="utf-8") as f:
        sensors = json.load(f)["sensors"]
    content = build(sensors, GasTable.load(args.gases))
    write(content, args.out)
    print(f"{args.out}: {sum(len(s['inputs']) for s in content['sensors'].values())} inputs from {len(sensors)} "
          f"sensors, {len(content['warnings'])} warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
