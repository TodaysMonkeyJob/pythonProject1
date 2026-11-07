"""
Train the sound-type classifier of the noise stage (A4 / TOD-44)::

    python -m tools.train_sound_types --data <dir> [--out models/sound_types.joblib] [--report reports/sound_types]

``<dir>/<type>/*.wav`` (also .flac / .ogg): one folder per type of ``config/audio_types.yaml`` (keyboard, walking,
fan, scream, optionally impact / glass_breaking / alarm; ``unknown`` is not trained — it is a low-confidence
prediction). Every file is resampled to 16 kHz mono and cut into 2 s clips.

The split is by **file** (stratified, ``--test-share`` of the files of every type held out), so clips of one
recording never land on both sides. Reports per-type accuracy (recall), precision, the confusion matrix and the
share of held-out clips that fall below ``min_confidence`` (would be published as ``unknown``) →
``<report>/metrics.json`` + ``README.md``; the model (scaler + random forest) → ``--out`` (models/ is
git-ignored). Needs at least ``--min-files`` files per type.
"""
import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from modules.audio.sound_type import SAMPLE_RATE, UNKNOWN, clips, features, load_types

AUDIO_EXT = (".wav", ".flac", ".ogg")


def read_audio(path: str) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        g = np.gcd(int(rate), SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, int(rate) // g).astype(np.float32)
    return audio


def dataset(root: str, types: List[str]) -> Dict[str, List[str]]:
    """:return: {type: [files]} for the type folders that exist"""
    out = {}
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        if name not in types or name == UNKNOWN:
            print(f"skipping folder {name!r}: not a trainable type of config/audio_types.yaml", file=sys.stderr)
            continue
        files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(AUDIO_EXT))
        if files:
            out[name] = files
    return out


def split_files(files: Dict[str, List[str]], test_share: float, seed: int) -> Tuple[Dict, Dict]:
    rng = np.random.default_rng(seed)
    train, test = {}, {}
    for name, items in files.items():
        order = list(rng.permutation(len(items)))
        held = max(1, int(round(len(items) * test_share)))
        test[name] = [items[i] for i in order[:held]]
        train[name] = [items[i] for i in order[held:]]
    return train, test


def to_xy(files: Dict[str, List[str]], clip_sec: float) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for name, items in files.items():
        for path in items:
            for clip in clips(read_audio(path), SAMPLE_RATE, clip_sec):
                xs.append(features(clip))
                ys.append(name)
    return np.array(xs), np.array(ys)


def build_model(seed: int):
    """Scaler + random forest, single-threaded: ~3 ms per clip on a Mac core (gradient boosting took 50 ms and
    spun all cores — too much next to Whisper on the Pi); robust on the small per-type datasets we expect."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                                                  class_weight="balanced", n_jobs=1,
                                                                  random_state=seed))


def evaluate(model, x: np.ndarray, y: np.ndarray, min_confidence: float) -> Dict[str, object]:
    proba = model.predict_proba(x)
    classes = [str(c) for c in model.classes_]
    pred = np.array([classes[i] for i in proba.argmax(axis=1)])
    confident = proba.max(axis=1) >= min_confidence
    per_type = {}
    for name in classes:
        mask = y == name
        predicted = pred == name
        per_type[name] = {"clips": int(mask.sum()),
                          "accuracy": round(float((pred[mask] == name).mean()), 3) if mask.any() else None,
                          "precision": round(float((y[predicted] == name).mean()), 3) if predicted.any() else None,
                          "below_min_confidence": round(float((~confident[mask]).mean()), 3) if mask.any() else None}
    confusion = {t: dict(Counter(pred[y == t])) for t in classes}
    return {"accuracy": round(float((pred == y).mean()), 3), "per_type": per_type, "confusion": confusion,
            "unknown_share": round(float((~confident).mean()), 3)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the noise-stage sound-type classifier")
    parser.add_argument("--data", required=True, help="<dir>/<type>/*.wav")
    parser.add_argument("--types", default="config/audio_types.yaml")
    parser.add_argument("--out", default="models/sound_types.joblib")
    parser.add_argument("--report", default=os.path.join("reports", "sound_types"))
    parser.add_argument("--clip-sec", type=float, default=2.0)
    parser.add_argument("--test-share", type=float, default=0.25)
    parser.add_argument("--min-files", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    types = load_types(args.types)
    files = dataset(args.data, list(types["types"]))
    few = {k: len(v) for k, v in files.items() if len(v) < args.min_files}
    if len(files) < 2 or few:
        print(f"need ≥ 2 types with ≥ {args.min_files} files each; have {({k: len(v) for k, v in files.items()})}",
              file=sys.stderr)
        return 1
    train, test = split_files(files, args.test_share, args.seed)
    x_train, y_train = to_xy(train, args.clip_sec)
    x_test, y_test = to_xy(test, args.clip_sec)
    model = build_model(args.seed).fit(x_train, y_train)
    metrics = evaluate(model, x_test, y_test, float(types.get("min_confidence", 0.5)))
    metrics.update(data=os.path.abspath(args.data), split="by file", test_share=args.test_share, seed=args.seed,
                   files={k: len(v) for k, v in files.items()}, train_clips=int(len(y_train)),
                   test_clips=int(len(y_test)), clip_sec=args.clip_sec)
    import joblib

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    joblib.dump(model, args.out)
    os.makedirs(args.report, exist_ok=True)
    with open(os.path.join(args.report, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    lines = ["# Sound-type classifier (A4 / TOD-44)", "",
             f"`python -m tools.train_sound_types --data {args.data}` — split by file ({args.test_share:.0%} of every "
             f"type's files held out), {metrics['train_clips']} train / {metrics['test_clips']} test clips of "
             f"{args.clip_sec:g} s. Overall accuracy {metrics['accuracy']:.1%}; {metrics['unknown_share']:.1%} of the "
             "held-out clips fall below min_confidence (published as unknown).", "",
             "| Type | Files | Test clips | Accuracy | Precision | Below min confidence |",
             "| -- | -- | -- | -- | -- | -- |"]
    for name, stats in metrics["per_type"].items():
        lines.append(f"| {name} | {metrics['files'].get(name, 0)} | {stats['clips']} | {fmt(stats['accuracy'])} | "
                      f"{fmt(stats['precision'])} | {fmt(stats['below_min_confidence'])} |")
    with open(os.path.join(args.report, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"→ {args.out}, {args.report}/metrics.json")
    return 0


def fmt(value) -> str:
    return "—" if value is None else f"{value:.1%}"


if __name__ == "__main__":
    raise SystemExit(main())
