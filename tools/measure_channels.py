"""
Validation counts for channel priors (F22 / TOD-33, TOD-34 channels) → ``reports/channel_eval/<channel>.json``
(correct, wrong, how it was measured, source sha256). ``tools/trust_priors.py`` turns them into r0 / s0 → τ0.

    python -m tools.measure_channels all
    python -m tools.measure_channels gas_concentration --tolerance 0.5

* camera.pose_action — ``reports/actions_eval_human_behaviour_640.json`` (``--actions-report``): segments passed;
* camera.smoking — the same report, hand_to_mouth segments: episodes detected vs missed;
* gas.gas_concentration — held-out curves of data.xls (split as in training), deployed models: |error| ≤ tolerance %;
* gas.gas_trend — held-out curves replayed as ramps / holds with a known rate (curves interpolated between held-out
  rows, as the training augmentation does) through the worker's ``TrendTracker``: label right, every 10 s;
* mic.vad — ``data/calibration/speech`` clips (speech) + noise-only clips of the same lengths, webrtcvad + the live
  ``SpeechSegmenter``: a segment is produced iff there is speech.

camera.objects / camera.motion / camera.scene_text need labelled clips that do not exist yet → not measured here
(``trust_priors.yaml`` keeps them ``prior_status: unmeasured``).
"""
import argparse
import csv
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

import numpy as np

if __package__ in (None, ""):  # allow `python tools/measure_channels.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration import now_iso, sha256_files  # noqa: E402

OUT_DIR = "reports/channel_eval"


def save(name: str, data: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    data = {"channel": name, "measured_at": now_iso(), **data}
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    print(f"{name:<24} {data['correct']:>5} correct / {data['wrong']:>4} wrong  → {out_dir}/{name}.json")
    return data


# --- camera (from the action evaluation report) -----------------------------------------------------------------

ACTIONS_REPORT = "reports/actions_eval_human_behaviour_640.json"   # the source TOD-33 / TOD-34 specify


def actions_report(config: Dict[str, Any], override: Optional[str] = None) -> str:
    """The action evaluation the pose / smoking priors come from (``--actions-report``; default the 640 px run named
    in TOD-33 / TOD-34 — the entry records the deployed ``actions.imgsz`` and the other run as an alternative)"""
    return override or ACTIONS_REPORT


def pose_action(path: str, deployed_imgsz: Optional[int] = None) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    posture = report["posture"]
    alternatives = {}
    for other in ("320", "640"):
        alt = f"reports/actions_eval_human_behaviour_{other}.json"
        if alt != path and os.path.isfile(alt):
            with open(alt, encoding="utf-8") as f:
                p = json.load(f)["posture"]
            alternatives[alt] = {"correct": p["passed"], "wrong": p["segments"] - p["passed"]}
    return {"device": "camera", "correct": int(posture["passed"]),
            "wrong": int(posture["segments"] - posture["passed"]),
            "unit": "labelled video segment (standing / sitting / lying / walking)", "source": path,
            "sha256": sha256_files([path]), "alternatives": alternatives,
            "deployed_imgsz": deployed_imgsz,
            "note": "evaluation run named in TOD-33 / TOD-34; compare deployed_imgsz with the run's resolution"}


def smoking(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    episodes = report["hand_to_mouth"]
    false_alarms = (report.get("non_smoking_videos") or {}).get("possible_smoking_false_alarms")
    return {"device": "camera", "correct": int(episodes["detected"]),
            "wrong": int(episodes["segments"] - episodes["detected"]), "unit": "hand-to-mouth episode",
            "source": path, "sha256": sha256_files([path]), "false_alarms_in_non_smoking_videos": false_alarms}


# --- gas ----------------------------------------------------------------------------------------------------------

def _gas_setup(config: Dict[str, Any]):
    from modules.gas.data import CLASS_NAMES, load_dataset
    from modules.gas.inference import GasPredictor
    from modules.gas.train import split_indices

    gas = config.get("gas") or {}
    data_path = "data/gas/data.xls"
    dataset = load_dataset(data_path)
    _, test_idx = split_indices(dataset)                  # the deployed models' held-out rows
    predictor = GasPredictor(gas.get("classifier_path", "models/gas_classifier.joblib"),
                             gas.get("regressor_path", "models/gas_concentration.joblib"))
    sources = [data_path, gas.get("classifier_path", "models/gas_classifier.joblib"),
               gas.get("regressor_path", "models/gas_concentration.joblib")]
    return dataset, test_idx, predictor, CLASS_NAMES, sources


def gas_concentration(config: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
    dataset, test_idx, predictor, _, sources = _gas_setup(config)
    errors = [abs(predictor.predict(list(dataset.X[i]))["concentration_pct"] - float(dataset.y_conc[i]))
              for i in test_idx]
    correct = sum(e <= tolerance + 1e-9 for e in errors)
    return {"device": "gas", "correct": int(correct), "wrong": len(errors) - int(correct),
            "unit": "held-out curve (gas classified + regressed as live)", "tolerance_pct": tolerance,
            "mae_pct": round(float(np.mean(errors)), 4), "max_error_pct": round(float(np.max(errors)), 4),
            "source": sources[0], "sha256": sha256_files(sources), "models": sources[1:]}


def gas_trend(config: Dict[str, Any], rates=(0.5, 1.0, 3.0, 6.0), hold_sec: int = 120,
              sample_every: int = 10) -> Dict[str, Any]:
    from modules.gas.worker import TrendTracker

    gas_cfg = config.get("gas") or {}
    window, band = float(gas_cfg.get("trend_window_sec", 60)), float(gas_cfg.get("trend_stable_band", 0.2))
    poll = float(gas_cfg.get("gas_poll_sec", 1.0))
    dataset, test_idx, predictor, _, sources = _gas_setup(config)
    correct = wrong = 0
    by_truth: Dict[str, List[int]] = {"rising": [0, 0], "falling": [0, 0], "stable": [0, 0]}

    def run(curves: Callable[[float], np.ndarray], true_rate: float, duration: float) -> None:
        nonlocal correct, wrong
        tracker, truth = TrendTracker(window, band), ("rising" if true_rate > band else "falling"
                                                     if true_rate < -band else "stable")
        t = 0.0
        while t <= duration:
            _, label = tracker.add(t, predictor.predict(list(curves(t)))["concentration_pct"])
            if t >= window / 2 and int(t) % sample_every == 0:
                ok = label == truth
                correct, wrong = correct + ok, wrong + (not ok)
                by_truth[truth][0 if ok else 1] += 1
            t += poll

    for sheet in sorted(set(dataset.source[test_idx])):
        rows = sorted((i for i in test_idx if dataset.source[i] == sheet and dataset.y_conc[i] > 0),
                      key=lambda i: dataset.y_conc[i])
        for i in rows:                                   # holds: one held-out curve, constant concentration
            run(lambda t, i=i: dataset.X[i], 0.0, hold_sec)
        for a, b in zip(rows[:-1], rows[1:], strict=True):
            ca, cb = float(dataset.y_conc[a]), float(dataset.y_conc[b])
            if cb - ca < 1e-9:
                continue
            for rate in rates:
                duration = (cb - ca) / rate * 60.0
                if duration < window:                    # too short to fill the trend window
                    continue
                for sign, start, end in ((1, a, b), (-1, b, a)):
                    def curve(t, start=start, end=end, duration=duration):
                        w = min(1.0, t / duration)
                        return (1 - w) * dataset.X[start] + w * dataset.X[end]
                    run(curve, sign * rate, duration)
    return {"device": "gas", "correct": int(correct), "wrong": int(wrong),
            "unit": f"trend label every {sample_every} s after half a window (ramps {list(rates)} %/min, holds)",
            "by_truth": {k: {"correct": v[0], "wrong": v[1]} for k, v in by_truth.items()},
            "window_sec": window, "stable_band": band, "source": sources[0], "sha256": sha256_files(sources),
            "note": "ramps interpolate the resistance curve between two held-out rows (like the training "
                    "augmentation); real sensor dynamics may differ"}


# --- microphone -----------------------------------------------------------------------------------------------------

def vad(config: Dict[str, Any], folder: str, seed: int = 42) -> Dict[str, Any]:
    import soundfile as sf

    from modules.audio.mic_controller import SpeechSegmenter, webrtc_vad

    mic = (config.get("audio") or {}).get("mic") or {}
    rate, frame_ms = 16000, int(mic.get("frame_ms", 30))
    size = rate * frame_ms // 1000
    detector = webrtc_vad(int(mic.get("vad_aggressiveness", 2)), rate)
    rng = np.random.default_rng(seed)

    def segments(audio: np.ndarray) -> int:
        segmenter = SpeechSegmenter(frame_ms, float(mic.get("preroll_sec", 0.5)),
                                    float(mic.get("silence_end_sec", 0.8)), float(mic.get("max_segment_sec", 15)),
                                    float(mic.get("min_speech_sec", 0.3)))
        pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        count = 0
        for start in range(0, len(pcm) - size + 1, size):
            frame = pcm[start:start + size]
            count += segmenter.push(frame, bool(detector(frame)), True) is not None
        return count + (segmenter.finish() is not None if segmenter.recording else 0)

    manifest = os.path.join(folder, "manifest.csv")
    with open(manifest, encoding="utf-8") as f:
        clips = list(csv.DictReader(f))
    tp = fn = tn = fp = 0
    by_level: Dict[str, List[int]] = {}                   # noise dBFS bucket → [rejected, false]
    for clip in clips:
        audio, _ = sf.read(os.path.join(folder, clip["path"]), dtype="float32")
        quiet = rng.normal(0, 10 ** (-60 / 20), rate).astype(np.float32)
        padded = np.concatenate([quiet, audio, quiet, quiet])
        found = segments(padded) > 0
        tp, fn = tp + found, fn + (not found)
        level_db = rng.uniform(-60, -20)                   # noise only: room hum … loud fan, same length
        noise = rng.normal(0, 10 ** (level_db / 20), len(padded)).astype(np.float32)
        false = segments(noise) > 0
        fp, tn = fp + false, tn + (not false)
        bucket = f"{int(level_db // 10 * 10)}..{int(level_db // 10 * 10) + 10} dBFS"
        by_level.setdefault(bucket, [0, 0])[1 if false else 0] += 1
    return {"device": "mic", "correct": int(tp + tn), "wrong": int(fn + fp), "unit": "clip: segment iff speech",
            "speech": {"detected": int(tp), "missed": int(fn)}, "noise_only": {"rejected": int(tn), "false": int(fp)},
            "noise_false_by_level": {k: {"rejected": v[0], "false": v[1]} for k, v in sorted(by_level.items())},
            "sound_threshold_db": mic.get("sound_threshold_db"),
            "source": manifest, "sha256": sha256_files([manifest]),
            "note": "synthetic TTS speech (incl. low SNR) and white noise −60…−20 dBFS; no real room recordings yet"}


def main(argv: Optional[List[str]] = None) -> None:
    from core.config import load_config

    parser = argparse.ArgumentParser(description="Validation counts per channel (→ trust priors)")
    parser.add_argument("what", choices=("all", "pose_action", "smoking", "gas_concentration", "gas_trend", "vad"))
    parser.add_argument("--actions-report", help=f"pose / smoking source (default {ACTIONS_REPORT})")
    parser.add_argument("--tolerance", type=float, default=0.5, help="gas_concentration: allowed abs error, %%")
    parser.add_argument("--speech", default="data/calibration/speech", help="vad: tools/synth_speech.py output")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    report = actions_report(config, args.actions_report)
    imgsz = int((config.get("actions") or {}).get("imgsz", 320))
    jobs = {"pose_action": lambda: pose_action(report, imgsz), "smoking": lambda: smoking(report),
            "gas_concentration": lambda: gas_concentration(config, args.tolerance),
            "gas_trend": lambda: gas_trend(config), "vad": lambda: vad(config, args.speech)}
    for name in (jobs if args.what == "all" else [args.what]):
        save(name, jobs[name](), args.out_dir)


if __name__ == "__main__":
    main()
