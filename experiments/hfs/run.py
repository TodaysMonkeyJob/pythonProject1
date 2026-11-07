"""
Article 1 evaluation (A6 / TOD-46) — one command::

    python -m experiments.hfs.run --seed 42 [--out reports/article1_<date>]

Every scenario of ``experiments/hfs/scenarios.yaml`` (60 s of events: gas readings every 2 s, persons in view,
motion level, microphone level, a sound type, a transcript — seeded measurement noise) is replayed through the real
``HierarchicalFuzzySystem`` in every configuration:

| configuration | membership functions | inference | multi-gas escalation (TOD-50) |
| -- | -- | -- | -- |
| tri_type1 | triangular / trapezoidal | type-1 (δ = 0) | yes |
| tri_it2 (the deployed system) | triangular / trapezoidal | interval type-2, δ from config/uncertainty.yaml | yes |
| gauss_in_type1 | Gaussian inputs (same centres / widths), triangular risk terms | type-1 | yes |
| gauss_in_it2 | Gaussian inputs | interval type-2 | yes |
| gauss_all_type1 | Gaussian inputs and risk terms | type-1 | yes |
| tri_it2_no_escalation | triangular | interval type-2 | no (the highest gas alone) |

Metrics against the expert labels: macro-F1 and accuracy of the overall level, Brier / ECE of the level
probabilities (output-term memberships at the risk), corridor width and coverage (the corridor reaches the expected
level), the corridor width vs δ (×0, ×0.5, ×1, ×2), latency per subsystem and top level on this machine, the
monotonicity measure and surfaces of every subsystem for triangular vs Gaussian terms, and — from the camera
benchmark — RAM / pose time in light vs actions mode and the false motion rate.
"""
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from core.events import make_event
from core.fuzzy import FuzzyEngine, load_yaml
from experiments.hfs import metrics as M
from experiments.hfs.variants import gaussian, without_escalation
from modules.hfs.worker import HierarchicalFuzzySystem

LEVELS = ("gas", "camera", "audio", "top")
CONFIGS = {
    "tri_type1": {"shape": "tri", "it2": False, "escalation": True},
    "tri_it2": {"shape": "tri", "it2": True, "escalation": True},
    "gauss_in_type1": {"shape": "gauss_in", "it2": False, "escalation": True},
    "gauss_in_it2": {"shape": "gauss_in", "it2": True, "escalation": True},
    "gauss_all_type1": {"shape": "gauss", "it2": False, "escalation": True},
    "tri_it2_no_escalation": {"shape": "tri", "it2": True, "escalation": False},
}
MAIN = "tri_it2"
SURFACE_PAIRS = {"gas": [("max_norm", "dynamics"), ("max_norm", "second_norm")],
                 "camera": [("unknown_share", "motion_intensity"), ("unknown_share", "activity_time")],
                 "audio": [("intensity", "sound_type_danger"), ("keyword_danger", "speaker_unknown")],
                 "top": [("risk_gas", "risk_camera")]}
WEDNESDAY = dt.date(2026, 11, 4)
LOCATION = "lab"


def write_variants(out_dir: str, name: str, spec: Dict[str, Any]) -> Dict[str, str]:
    paths = {}
    folder = os.path.join(out_dir, "variants", name)
    os.makedirs(folder, exist_ok=True)
    for level in LEVELS:
        data = load_yaml(os.path.join("config", "fuzzy", f"{level}.yaml"))
        if spec["shape"] in ("gauss", "gauss_in"):
            data = gaussian(data, outputs=spec["shape"] == "gauss")
        if level == "gas" and not spec["escalation"]:
            data = without_escalation(data)
        path = os.path.join(folder, f"{level}.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, width=120)
        paths[level] = path
    return paths


def system_config(base: Dict[str, Any], paths: Dict[str, str], it2: bool, scale: float = 1.0) -> Dict[str, Any]:
    hfs = copy.deepcopy(base.get("hfs") or {})
    hfs.update(it2=it2, uncertainty_scale=scale)
    for level in LEVELS:
        hfs.setdefault(level, {})
        hfs[level] = {**(hfs[level] or {}), "rules": paths[level]}
    return {"hfs": hfs, "identity": base.get("identity", {}), "gas": base.get("gas", {})}


def events_of(scenario: Dict[str, Any], rng: np.random.Generator) -> Tuple[List[Tuple[float, Any]], List[Dict]]:
    """:return: ([(t, event)], devices) of one scenario"""
    events: List[Tuple[float, Any]] = []
    offline = set(scenario.get("offline") or [])
    devices = []

    def ev(t, type_, payload, device, source="system"):
        events.append((t, make_event(source, type_, payload, location_id=LOCATION, device_id=device)))

    gases = scenario.get("gas") or [{"gas": "clean_air", "value": 0, "unit": "ppm"}]
    if "gas" not in offline:
        for i, g in enumerate(gases):
            device = f"gas-{i}"
            devices.append({"device_id": device, "kind": "gas", "location_id": LOCATION, "status": "online"})
            for t in range(0, 61, 2):
                value = g["value"] + (g.get("to", g["value"]) - g["value"]) * t / 60
                value *= 1 + rng.normal(0, 0.01)                        # 1 % reading noise
                ev(float(t), "gas_reading", {"gas": g["gas"], "concentration": max(0.0, value), "unit": g["unit"]},
                   device, "gas")
    cam = scenario.get("camera") or {}
    if "camera" not in offline:
        devices.append({"device_id": "cam-1", "kind": "camera", "location_id": LOCATION, "status": "online"})
        track = 0
        for kind in ("known", "unknown"):
            for i in range(int(cam.get(kind, 0))):
                track += 1
                identity, person = (f"Person{i}", f"p{i}") if kind == "known" else (f"unknown_{i + 1}", f"u{i}")
                ev(1.0, "object_detected", {"class": "person", "track_id": track, "state": "appeared"}, "cam-1",
                   "camera")
                ev(2.0, "person_identified", {"track_id": track, "identity": identity, "person_id": person},
                   "cam-1", "camera")
        motion = float(cam.get("motion", 0.0))
        for t in range(0, 61, 2):
            level = max(0.0, motion * (1 + rng.normal(0, 0.05))) if motion > 0 else 0.0
            ev(float(t), "motion_level", {"intensity": round(level, 4)}, "cam-1", "camera")
    audio = scenario.get("audio") or {}
    if "mic" not in offline:
        devices.append({"device_id": "mic-1", "kind": "mic", "location_id": LOCATION, "status": "online"})
        intensity = float(audio.get("intensity", 0.0))
        for t in range(0, 61, 2):
            level = max(0.0, intensity + rng.normal(0, 0.5)) if intensity > 0 else 0.0
            ev(float(t), "audio_level", {"intensity_db": round(level, 1), "background_spread_db": 1.0}, "mic-1",
               "audio")
        if audio.get("sound"):
            ev(50.0, "sound_type", {**audio["sound"], "source": "sound"}, "mic-1", "audio")
        if audio.get("speech"):
            ev(55.0, "speech", {"text": audio["speech"]["text"], "speaker_status": audio["speech"]["speaker"]},
               "mic-1", "audio")
    events.sort(key=lambda item: item[0])
    return events, devices


def run_scenario(scenario: Dict[str, Any], config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    events, devices = events_of(scenario, rng)
    hh, mm = (int(x) for x in str((scenario.get("camera") or {}).get("time", "11:00")).split(":"))
    moment = dt.datetime.combine(WEDNESDAY, dt.time(hh, mm))
    start = 1_000_000.0
    now = [start]
    system = HierarchicalFuzzySystem(config, devices=lambda: devices, clock=lambda: now[0],
                                     local_now=lambda: moment + dt.timedelta(seconds=now[0] - start))
    for t, event in events:
        now[0] = start + t
        system.observe(event)
    now[0] = start + 60.0
    timings = {}
    results = system.results.setdefault(LOCATION, {})
    for name, subsystem in system.subsystems.items():
        began = time.perf_counter()
        results[name] = subsystem.evaluate(LOCATION, now[0], system.available(subsystem.device_kind, LOCATION))
        timings[name] = (time.perf_counter() - began) * 1000
    began = time.perf_counter()
    payload = system.top.combine({name: results.get(name) for name in ("gas", "camera", "audio")})
    timings["top"] = (time.perf_counter() - began) * 1000
    memberships = system.top.engine.fuzzify(system.top.engine.output, payload["overall"])
    return {"level": payload["level"], "risk": payload["overall"], "interval": payload["interval"],
            "partial": payload["partial"], "probs": M.probabilities(memberships),
            "subsystems": {k: (v["level"], v["risk"], v["interval"]) for k, v in payload["subsystems"].items()},
            "ms": timings}


def evaluate(results: Dict[str, Dict[str, Any]], scenarios: List[Dict[str, Any]],
             regions: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    truth = [s["label"] for s in scenarios]
    pred = [results[s["id"]]["level"] for s in scenarios]
    probs = [results[s["id"]]["probs"] for s in scenarios]
    widths = [results[s["id"]]["interval"][1] - results[s["id"]]["interval"][0] for s in scenarios]
    covered = [M.covers(results[s["id"]]["interval"], regions[s["label"]]) for s in scenarios]
    by_group = {}
    for group in sorted({s["group"] for s in scenarios}):
        idx = [i for i, s in enumerate(scenarios) if s["group"] == group]
        by_group[group] = M.macro_f1([truth[i] for i in idx], [pred[i] for i in idx])
    accuracy = round(float(np.mean(np.array(truth) == np.array(pred))), 4)
    return {"macro_f1": M.macro_f1(truth, pred), "accuracy": accuracy,
            "brier": M.brier(truth, probs), "ece": M.ece(truth, probs),
            "corridor_width_mean": round(float(np.mean(widths)), 4),
            "corridor_coverage": round(float(np.mean(covered)), 4), "macro_f1_by_group": by_group,
            "misses": [{"id": s["id"], "label": s["label"], "got": results[s["id"]]["level"]}
                       for s in scenarios if results[s["id"]]["level"] != s["label"]],
            "confusion": M.confusion(truth, pred)}


def surfaces(out_dir: str, variant_paths: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    from tools.plot_surfaces import middle, monotonicity, plot

    folder = os.path.join(out_dir, "surfaces")
    os.makedirs(folder, exist_ok=True)
    table: Dict[str, Any] = {}
    for shape, config in (("triangular", "tri_type1"), ("gaussian_inputs", "gauss_in_type1"),
                          ("gaussian_all", "gauss_all_type1")):
        for level in LEVELS:
            engine = FuzzyEngine.from_dict(load_yaml(variant_paths[config][level]))
            stats = monotonicity(engine, {"audio": 7, "gas": 6}.get(level, 11))   # gas: 5 inputs, same grid for
                                                                                    # every shape
            steps = sum(s["steps"] for s in stats.values())
            falls = sum(s["falls"] for s in stats.values())
            table.setdefault(level, {})[shape] = {
                "falls": falls, "steps": steps, "share": round(falls / steps, 4),
                "max_fall": max(s["max_fall"] for s in stats.values()),
                "level_drops": sum(s["level_drops"] for s in stats.values())}
            for x, y in SURFACE_PAIRS[level]:
                rest = {n: 0.0 if level == "gas" else middle(engine, n)        # gas: no other gas, steady
                        for n in engine.variables if n not in (x, y, engine.output)}
                name = os.path.join(folder, f"{level}_{shape}_{x}__{y}.png")
                plot(engine, f"{level} ({shape})", x, y, rest, 31, name)
    return table


def sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Article 1 evaluation (A6)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=os.path.join("reports", f"article1_{dt.date.today().isoformat()}"))
    parser.add_argument("--scenarios", default=os.path.join("experiments", "hfs", "scenarios.yaml"))
    parser.add_argument("--no-surfaces", action="store_true")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    with open("config.yaml", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    spec = yaml.safe_load(open(args.scenarios, encoding="utf-8"))
    scenarios = spec["scenarios"]
    variant_paths = {name: write_variants(args.out, name, c) for name, c in CONFIGS.items()}
    top = FuzzyEngine.from_dict(load_yaml(os.path.join("config", "fuzzy", "top.yaml")))
    grid, curves = top.variables[top.output].curves()
    regions = M.level_region(curves, grid)

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name, c in CONFIGS.items():
        config = system_config(base, variant_paths[name], c["it2"])
        results[name] = {s["id"]: run_scenario(s, config, args.seed + i) for i, s in enumerate(scenarios)}
        print(f"{name}: done")
    summary = {name: evaluate(results[name], scenarios, regions) for name in CONFIGS}
    sweep = {}
    for scale in (0.0, 0.5, 1.0, 2.0):
        config = system_config(base, variant_paths[MAIN], True, scale)
        run = {s["id"]: run_scenario(s, config, args.seed + i) for i, s in enumerate(scenarios)}
        stats = evaluate(run, scenarios, regions)
        sweep[scale] = {"corridor_width_mean": stats["corridor_width_mean"],
                        "corridor_coverage": stats["corridor_coverage"], "macro_f1": stats["macro_f1"]}
    latency = {}
    for name in ("tri_type1", "tri_it2"):
        for level in LEVELS:
            values = sorted(r["ms"][level] for r in results[name].values())
            latency.setdefault(level, {})[name] = {"mean": round(float(np.mean(values)), 3),
                                                   "p95": round(values[int(0.95 * (len(values) - 1))], 3)}
    mono = {} if args.no_surfaces else surfaces(args.out, variant_paths)
    bench = {}
    for name in ("actions", "motion"):
        path = os.path.join("reports", "camera_bench", f"{name}.json")
        if os.path.isfile(path):
            bench[name] = json.load(open(path, encoding="utf-8"))
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = {"seed": args.seed, "git": git, "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "scenarios": len(scenarios), "labels": spec.get("status"),
                "files": {p: sha(p) for p in [args.scenarios, "config/uncertainty.yaml", "config/gases.yaml",
                                              *[os.path.join("config", "fuzzy", f"{lv}.yaml") for lv in LEVELS]]}}
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"manifest": manifest, "summary": summary, "delta_sweep": sweep, "latency_ms": latency,
                   "monotonicity": mono, "camera_bench": bench}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "scenarios.csv"), "w", encoding="utf-8") as f:
        f.write("id,group,label," + ",".join(f"{n}_level,{n}_risk,{n}_y_l,{n}_y_r" for n in CONFIGS) + "\n")
        for s in scenarios:
            row = [s["id"], s["group"], s["label"]]
            for name in CONFIGS:
                r = results[name][s["id"]]
                row += [r["level"], f"{r['risk']:.4f}", f"{r['interval'][0]:.4f}", f"{r['interval'][1]:.4f}"]
            f.write(",".join(row) + "\n")
    from experiments.hfs.report import write_report

    write_report(args.out, manifest, scenarios, results, summary, sweep, latency, mono, bench, CONFIGS, MAIN)
    print(f"→ {args.out}/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
