"""
Measurement errors of every HFS input → ``config/uncertainty.yaml`` (A5 / TOD-45)::

    python -m tools.fit_uncertainty [--out config/uncertainty.yaml]

Trust in Article 1 comes from what is known about each sensor and decision — its accuracy — not from Beta counters
(real sensors rarely fail and their readings can rarely be confirmed). The error of an input is a half-width δ in
the input's own units; it moves the edges of every membership function by ±δ (interval type-2 FOU) and the
Karnik–Mendel type reduction turns it into the corridor [y_l, y_r].

Every value names its source and date; nothing is typed by hand except datasheet figures, which name the datasheet.
Kinds:

* ``absolute`` / ``relative`` (+ ``absolute`` floor) — sensor accuracy, δ_c = max(abs, rel · c) in the reading's
  unit, propagated to c_norm = c / c_crit;
* ``live`` — computed for every inference from the data behind it (formula + parameters recorded here): the
  dynamics slope error from δ_c and the sample times, the unknown_share binomial interval from the identification
  accuracy, the background spread of a microphone, a sound type's (1 − confidence);
* ``decision`` — inputs that are the outcome of a decision with accuracy a: δ = 1 − a (a = Wilson 95 % lower bound,
  so a perfect small sample is not overconfident) — the largest change a wrong decision makes on a 0..1 input times
  its probability;
* ``unmeasured`` — no data yet: δ = 0 and the corridor does not include that error (listed, never invented).
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from typing import Any, Dict, Tuple

import yaml

Z = 1.96


def wilson_lower(correct: float, n: int, z: float = Z) -> float:
    if n <= 0:
        return 0.0
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(centre - half, 4)


def sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def mtime(path: str) -> str:
    return dt.datetime.fromtimestamp(os.path.getmtime(path), dt.timezone.utc).date().isoformat()


def gas_section() -> Dict[str, Any]:
    out: Dict[str, Any] = {"sensors": {}, "default_by_unit": {"vol_pct": "graphene_fet", "ug_m3": "sds011"},
                           "default_by_gas": {"ammonia": "dfrobot_sen0469"}}
    path = "reports/gas_metrics.json"
    if os.path.isfile(path):
        metrics = json.load(open(path, encoding="utf-8"))
        per_gas = {}
        for gas, entry in (metrics.get("concentration", {}).get("per_gas") or {}).items():
            mae = max(float(entry.get("cv_mae_pct") or 0), float(entry.get("test_mae_pct") or 0))
            per_gas[gas] = {"absolute": round(mae, 4), "n_test": entry.get("n_test")}
        out["sensors"]["graphene_fet"] = {
            "kind": "absolute", "unit": "vol_pct", "per_gas": per_gas,
            "source": f"{path}: concentration regressor MAE per gas, max(cross-validation, test) in vol %",
            "date": str(metrics.get("trained_at", mtime(path)))[:10], "status": "measured"}
    out["sensors"]["dfrobot_sen0469"] = {
        "kind": "relative", "unit": "ppm", "relative": 0.10, "absolute": 5.0,
        "source": "DFRobot SEN0469 (NH3 0–100 ppm): ±10 % of reading or ±5 % FS, the greater — DFRobot forum reply "
                  "(docs/sensors.md); confirm against the datasheet", "date": "2026-11-06", "status": "datasheet"}
    out["sensors"]["dfrobot_sen0466"] = {
        "kind": "unmeasured", "unit": "ppm",
        "source": "DFRobot SEN0466 (CO 0–1000 ppm): accuracy not published on the wiki (docs/sensors.md)",
        "date": "2026-11-06", "status": "unmeasured"}
    out["sensors"]["sds011"] = {
        "kind": "relative", "unit": "ug_m3", "relative": 0.15, "absolute": 10.0,
        "source": "Nova Fitness SDS011 datasheet copy: max of ±15 % and ±10 µg/m³ (docs/sensors.md)",
        "date": "2026-11-06", "status": "datasheet"}
    out["sensors"]["sps30"] = {
        "kind": "relative", "unit": "ug_m3", "relative": 0.10, "absolute": 10.0,
        "source": "Sensirion SPS30: ±10 µg/m³ (0–100), ±10 % (100–1000) — Sensirion datasheet copy / secondary "
                  "table (docs/sensors.md)", "date": "2026-11-06", "status": "datasheet"}
    return out


def unknown_share_section() -> Dict[str, Any]:
    from modules.hfs.camera import identification_accuracy

    path = "reports/identity_eval_lfw.json"
    threshold = float(yaml.safe_load(open("config.yaml", encoding="utf-8"))["identity"]["face_threshold"])
    p, source = identification_accuracy(path, threshold)
    return {"kind": "live", "formula": "binomial interval of the unknown count: each of n identities wrong with "
                                       "probability 1 − p, its 95 % quantile f → δ = f / n",
            "p": p, "source": source, "date": mtime(path), "status": "measured"}


def motion_section() -> Dict[str, Any]:
    path = "reports/camera_bench/motion.json"
    if not os.path.isfile(path):
        return {"kind": "unmeasured", "delta": 0.0, "status": "unmeasured",
                "source": "run python -m tools.camera_bench (static-scene noise of the adaptive detector)"}
    noise = (json.load(open(path, encoding="utf-8")).get("static_noise") or {})
    return {"kind": "absolute", "delta": noise.get("max_area_ratio", 0.0), "status": "measured",
            "source": f"{path}: the largest moving share of the frame the adaptive detector reported on scenes "
                      f"without anyone (lighting, foliage, noise clips; {noise.get('frames')} frames; its 95th "
                      f"percentile is {noise.get('p95_area_ratio')}: false motion is rare but not zero) — "
                      "synthetic clips", "date": mtime(path)}


def keyword_section() -> Dict[str, Any]:
    from modules.audio.keywords import DangerWords

    path = "data/calibration/asr_phrases.csv"
    words = DangerWords.load()
    rows = list(csv.DictReader(open(path, encoding="utf-8")))

    def accuracy(selection) -> Tuple[int, int]:
        right = wrong = 0
        for row in selection:
            said = {m.entry for m in words.find(row["reference"])}
            heard = {m.entry for m in words.find(row["hypothesis"])}
            right += len(said & heard)
            wrong += len(said - heard) + len(heard - said)
        return right, wrong

    right, wrong = accuracy(rows)
    quiet_right, quiet_wrong = accuracy([r for r in rows if not r["snr_db"] or float(r["snr_db"]) >= 10])
    lower = wilson_lower(right, right + wrong)
    return {"kind": "decision", "delta": round(1 - lower, 4), "accuracy": round(right / (right + wrong), 4),
            "accuracy_lower": lower, "n": right + wrong,
            "source": f"{path} (sha {sha(path)}): danger-dictionary words said vs heard by "
                      f"{sorted({r['model'] for r in rows})[0]} on {len(rows)} synthetic TTS phrases with noise "
                      f"(−5 … 30 dB SNR): {right} right, {wrong} missed / false",
            "alternative": f"SNR ≥ 10 dB / clean only: {quiet_right} right, {quiet_wrong} wrong "
                           f"(Wilson lower bound {wilson_lower(quiet_right, quiet_right + quiet_wrong)}) — not used: "
                           "no cherry-picking; few danger words in the set, record more",
            "date": mtime(path), "status": "measured"}


def speaker_section() -> Dict[str, Any]:
    path = "config/calibration/speaker.yaml"
    data = yaml.safe_load(open(path, encoding="utf-8"))
    eer, n = float(data["metrics"]["heldout"]["eer"]), int(data["n_heldout"])
    lower = wilson_lower((1 - eer) * n, n)
    deployed = data.get("at_voice_threshold") or {}
    return {"kind": "decision", "delta": round(1 - lower, 4), "eer": eer, "n": n,
            "source": f"{path}: held-out speaker verification EER {eer} on {n} pairs (synthetic TTS voices), "
                      "Wilson upper bound of the error",
            "note": f"at the deployed identity.voice_threshold {deployed.get('threshold')} the error is "
                    f"{deployed.get('wrong')}/{deployed.get('wrong', 0) + deployed.get('correct', 0)} — raise the "
                    "threshold toward the EER point or the corridor understates it",
            "date": str(data.get("fitted_at", ""))[:10], "status": "measured"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="HFS input errors → config/uncertainty.yaml")
    parser.add_argument("--out", default="config/uncertainty.yaml")
    args = parser.parse_args(argv)
    sound_metrics = "reports/sound_types/metrics.json"
    per_type = {}
    if os.path.isfile(sound_metrics):                      # per-type accuracy of the trained classifier (TOD-44)
        metrics = json.load(open(sound_metrics, encoding="utf-8"))
        for name, stats in (metrics.get("per_type") or {}).items():
            if stats.get("accuracy") is not None and stats.get("clips"):
                per_type[name] = wilson_lower(stats["accuracy"] * stats["clips"], int(stats["clips"]))
    data = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generator": "python -m tools.fit_uncertainty",
        "z": Z,
        "gas": {
            "concentration": gas_section(),
            "dynamics": {"kind": "live", "formula": "least-squares slope error from δ_c (≈ 95 % bound) and the "
                         "sample times: δ = δ_c_norm · 60 / sqrt(Σ (t − t̄)²) per minute", "status": "derived",
                         "source": "propagated from gas.concentration and the sampling period (docs/sensors.md)"},
        },
        "camera": {
            "unknown_share": unknown_share_section(),
            "motion_intensity": motion_section(),
            "activity_time": {"kind": "absolute", "delta": 0.0, "status": "exact",
                              "source": "the location's schedule and the clock: no measurement error"},
        },
        "audio": {
            "intensity": {"kind": "live", "status": "derived",
                          "formula": "half the spread (p30 − p10) / 2 of this microphone's per-second background "
                                     "levels over the background window, dB",
                          "calibration": "unmeasured — no reference microphone; the absolute mic calibration error "
                                         "is not in the corridor",
                          "source": "modules/audio/intensity.py (live statistics of the microphone itself)"},
            "sound_type_danger": {
                "kind": "live", "status": "measured" if os.path.isfile(sound_metrics) else "unmeasured",
                "formula": "1 − per-type accuracy (Wilson lower bound) when reports/sound_types/metrics.json "
                           "exists, else 1 − the classifier's confidence of that sound",
                "source": sound_metrics if os.path.isfile(sound_metrics) else
                "no trained model yet (TOD-44): sound types are not evidence",
                **({"per_type_accuracy": per_type} if per_type else {})},
            "keyword_danger": keyword_section(),
            "speaker_unknown": speaker_section(),
        },
        "top": {"risk": {"kind": "live", "status": "derived",
                         "formula": "the subsystem corridor [y_l, y_r] → input m = (y_l + y_r) / 2, "
                                    "δ = (y_r − y_l) / 2",
                         "source": "the subsystems' interval type-2 inference"}},
    }
    header = ("# A5 (TOD-45) measurement errors of the HFS inputs — written by python -m tools.fit_uncertainty; do "
              "not edit.\n# δ in the input's units moves every membership edge by ±δ (IT2 FOU) → corridor "
              "[y_l, y_r]. docs/uncertainty.md\n")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, width=120)
    print(f"→ {args.out}")
    for section in ("gas", "camera", "audio"):
        for name, entry in data[section].items():
            print(f"  {section}.{name}: {entry.get('status')} δ={entry.get('delta', entry.get('kind'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
