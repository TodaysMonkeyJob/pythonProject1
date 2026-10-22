"""
Channel score calibration (F22 / TOD-33): raw model score → calibrated probability, fitted on labelled data and
stored with the dataset hash, n, date and metrics in ``config/calibration/<channel>.yaml``.

    python -m tools.calibrate_channels asr --audio data/calibration/speech   # local faster-whisper → CSV → fit
    python -m tools.calibrate_channels asr --csv data/calibration/asr_phrases.csv
    python -m tools.calibrate_channels speaker --audio data/calibration/speech   # ECAPA pair scores → CSV → fit
    python -m tools.calibrate_channels speaker --pairs data/calibration/speaker_pairs.csv
    python -m tools.calibrate_channels face --prior 0.5        # Bayes from reports/identity_eval_lfw.json
    python -m tools.calibrate_channels gas                     # predict_proba on held-out curves

| channel | raw score | calibrated meaning | method |
| -- | -- | -- | -- |
| asr | exp(avg_logprob)·(1 − no_speech_prob) | P(WER ≤ wer_max) | isotonic regression |
| speaker | ECAPA cosine to the centroid | P(same speaker) | logistic regression; EER + Cllr on held-out pairs |
| face | SFace similarity of the best match | P(known \\| similarity) | Bayes over measured TAR / FAR bins, prior |
| gas | predict_proba of the predicted gas | P(label right) | isotonic regression over repeated held-out splits |

Metrics are out-of-fold (5-fold) / held-out, the map itself is fitted on all calibration rows. Everything local.
"""
import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

if __package__ in (None, ""):  # allow `python tools/calibrate_channels.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration import (LogisticMap, asr_confidence, bayes_bins, brier, cllr, ece,  # noqa: E402
                              equal_error_rate, fit_isotonic, fit_logistic, logit, now_iso, reliability,
                              sha256_files, wer, write_yaml)

OUT_DIR = "config/calibration"
HEADER = "F22 (TOD-33) measured channel calibration — written by python -m tools.calibrate_channels {cmd}; do not edit"
TABLE_AT = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def out_of_fold(scores: np.ndarray, outcomes: np.ndarray, fit, folds: int = 5, seed: int = 42) -> np.ndarray:
    """:return: out-of-fold calibrated probabilities (honest Brier / ECE for the map fitted on everything)"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(scores))
    predicted = np.zeros(len(scores))
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test)
        predicted[test] = fit(scores[train], outcomes[train]).many(scores[test])
    return predicted


def metrics(p: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    return {"brier": round(brier(p, y), 4), "ece": round(ece(p, y), 4), "base_rate": round(float(np.mean(y)), 4),
            "brier_base_rate": round(brier(np.full(len(y), np.mean(y)), y), 4)}


def save(channel: str, data: Dict[str, Any], out_dir: str) -> str:
    path = os.path.join(out_dir, f"{channel}.yaml")
    write_yaml(path, {"channel": channel, "version": 1, "fitted_at": now_iso(), **data}, HEADER.format(cmd=channel))
    print(f"→ {path}")
    return path


# --- ASR ------------------------------------------------------------------------------------------------------------

ASR_COLUMNS = ["audio_id", "reference", "hypothesis", "avg_logprob", "no_speech_prob", "language", "snr_db", "model"]


def transcribe(folder: str, csv_path: str, model_name: str, config_path: str) -> None:
    """Local faster-whisper with the live options (beam, VAD, uk/en detection, hallucination filter) → CSV."""
    import soundfile as sf
    from faster_whisper import WhisperModel

    from core.config import load_config
    from modules.llm.whisper_client import WhisperOptions, _segments

    mic = load_config(config_path).get("audio", {}).get("mic", {})
    opts = WhisperOptions.from_mic(mic)
    model = WhisperModel(model_name, device="cpu", compute_type=opts.compute_type, cpu_threads=opts.cpu_threads)
    with open(os.path.join(folder, "manifest.csv"), encoding="utf-8") as f:
        clips = list(csv.DictReader(f))
    rows = []
    for i, clip in enumerate(clips, 1):
        audio, _ = sf.read(os.path.join(folder, clip["path"]), dtype="float32")
        language = None
        if opts.language or len(opts.languages) == 1:
            language = opts.language or opts.languages[0]
        elif opts.languages:
            _, _, probabilities = model.detect_language(audio)
            allowed = [(p, code) for code, p in probabilities if code in opts.languages]
            language = max(allowed)[1] if allowed else opts.languages[0]
        raw, _ = model.transcribe(audio, language=language, beam_size=opts.beam_size, vad_filter=True,
                                  condition_on_previous_text=opts.condition_on_previous_text,
                                  no_speech_threshold=opts.no_speech_max, log_prob_threshold=opts.logprob_min,
                                  compression_ratio_threshold=opts.compression_max)
        raw = list(raw)
        kept = _segments(raw, opts)
        weights = np.array([max(0.01, s.end - s.start) for s in raw]) if raw else np.ones(1)
        logprob = float(np.average([s.avg_logprob for s in raw], weights=weights)) if raw else -5.0
        no_speech = float(np.average([s.no_speech_prob for s in raw], weights=weights)) if raw else 1.0
        rows.append({"audio_id": clip["audio_id"], "reference": clip["reference"],
                     "hypothesis": " ".join(s["text"] for s in kept).strip(), "avg_logprob": round(logprob, 4),
                     "no_speech_prob": round(no_speech, 4), "language": language or "", "snr_db": clip["snr_db"],
                     "model": f"faster-whisper-{model_name}"})
        if i % 25 == 0:
            print(f"  {i}/{len(clips)}")
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ASR_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} phrases → {csv_path}")


def calibrate_asr(csv_path: str, wer_max: float, out_dir: str) -> Dict[str, Any]:
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    scores = np.array([asr_confidence(float(r["avg_logprob"]), float(r["no_speech_prob"])) for r in rows])
    errors = np.array([wer(r["reference"], r["hypothesis"]) for r in rows])
    good = (errors <= wer_max + 1e-9).astype(float)
    mapping = fit_isotonic(scores, good)
    oof = out_of_fold(scores, good, fit_isotonic)
    by_snr: Dict[str, Dict[str, float]] = {}
    for r, e in zip(rows, errors, strict=True):
        key = r.get("snr_db") or "clean"
        by_snr.setdefault(key, []).append(e)
    return {"raw_score": "exp(avg_logprob) * (1 - no_speech_prob)",
            "meaning": f"P(WER <= {wer_max:.0%} for this phrase)", "method": "isotonic regression",
            "wer_max": wer_max, "n": len(rows), "positives": int(good.sum()),
            "dataset": {"path": csv_path, "sha256": sha256_files([csv_path]),
                        "models": sorted({r.get("model", "") for r in rows}),
                        "note": "synthetic TTS phrases (tools/synth_speech.py) + white noise; refit on recorded "
                                "phrases with the same columns"},
            "metrics": {"in_sample": metrics(mapping.many(scores), good), "out_of_fold": metrics(oof, good),
                        "mean_wer": round(float(errors.mean()), 4),
                        "mean_wer_by_snr_db": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_snr.items())}},
            "reliability": reliability(oof, good),
            "table": {f"{t:.1f}": round(mapping(t), 3) for t in TABLE_AT},
            "map": mapping.to_dict()}


# --- speaker --------------------------------------------------------------------------------------------------------

PAIR_COLUMNS = ["audio_id", "speaker", "against", "score", "genuine", "split"]


def speaker_pairs(folder: str, csv_path: str, config_path: str, window_sec: float = 1.5) -> None:
    """
    ECAPA embedding per clip (mean of 1.5 s / 0.75 s windows, like the live analyser). Per speaker the clips with an
    even index (SNR ≥ 20 dB or clean) are the enrollment → centroid; every odd clip is scored against every centroid.
    Split: odd index % 4 == 1 → fit, == 3 → held-out.
    """
    import soundfile as sf

    from core.config import load_config
    from core.identity import normalize
    from modules.audio.speaker import SAMPLE_RATE, ecapa_batch_embedder, preprocess

    mic = load_config(config_path).get("audio", {}).get("mic", {})
    embed = ecapa_batch_embedder(mic.get("speaker_model_dir", "model/speechbrain"), mic.get("speaker_device", "auto"))
    with open(os.path.join(folder, "manifest.csv"), encoding="utf-8") as f:
        clips = list(csv.DictReader(f))
    size, hop = int(window_sec * SAMPLE_RATE), int(window_sec * SAMPLE_RATE / 2)
    vectors: Dict[str, np.ndarray] = {}
    for clip in clips:
        audio, rate = sf.read(os.path.join(folder, clip["path"]), dtype="float32")
        audio = preprocess(audio, rate)
        if len(audio) < size:
            audio = np.pad(audio, (0, size - len(audio)))
        pieces = np.stack([audio[s:s + size] for s in range(0, len(audio) - size + 1, hop)])
        vectors[clip["audio_id"]] = normalize(np.mean([normalize(e) for e in embed(pieces)], axis=0))
    index = {c["audio_id"]: int(c["audio_id"].rsplit("_", 1)[1]) for c in clips}
    enroll: Dict[str, List[np.ndarray]] = {}
    for c in clips:
        snr = c["snr_db"]
        if index[c["audio_id"]] % 2 == 0 and (snr == "" or float(snr) >= 20):
            enroll.setdefault(c["speaker"], []).append(vectors[c["audio_id"]])
    centroids = {name: normalize(np.mean(v, axis=0)) for name, v in enroll.items()}
    rows = []
    for c in clips:
        i = index[c["audio_id"]]
        if i % 2 == 0:
            continue
        for name, centroid in centroids.items():
            rows.append({"audio_id": c["audio_id"], "speaker": c["speaker"], "against": name,
                         "score": round(float(vectors[c["audio_id"]] @ centroid), 5),
                         "genuine": int(name == c["speaker"]), "split": "fit" if i % 4 == 1 else "heldout"})
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} pairs ({len(centroids)} speakers) → {csv_path}")


def identification_counts(rows: List[Dict[str, Any]], threshold: float) -> Dict[str, int]:
    """
    Decisions the live matcher would make on the held-out clips at ``threshold``: an enrolled speaker → correct if
    the best centroid is theirs and ≥ threshold; the same clip with the own centroid removed (a stranger) → correct
    if nothing reaches the threshold (unknown). :return: {correct, wrong, known_*, stranger_*}
    """
    by_clip: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_clip.setdefault(r["audio_id"], []).append(r)
    known_ok = known_bad = stranger_ok = stranger_bad = 0
    for pairs in by_clip.values():
        best = max(pairs, key=lambda r: r["score"])
        if best["genuine"] and best["score"] >= threshold:
            known_ok += 1
        else:
            known_bad += 1
        others = [r for r in pairs if not r["genuine"]]
        if others and max(r["score"] for r in others) >= threshold:
            stranger_bad += 1
        else:
            stranger_ok += 1
    return {"correct": known_ok + stranger_ok, "wrong": known_bad + stranger_bad, "known_correct": known_ok,
            "known_wrong": known_bad, "stranger_correct": stranger_ok, "stranger_wrong": stranger_bad}


def calibrate_speaker(csv_path: str, threshold: float, prior: float, out_dir: str) -> Dict[str, Any]:
    """
    The logistic fit sees one genuine pair per speaker for every impostor centroid (pair prior ≈ 1 / speakers);
    metrics are measured at that prior on held-out pairs, the stored map is shifted to ``prior`` = P(the voice
    belongs to the matched person) before the match is known (same meaning as the face prior).
    """
    with open(csv_path, encoding="utf-8") as f:
        rows = [{**r, "score": float(r["score"]), "genuine": int(r["genuine"])} for r in csv.DictReader(f)]
    fit_rows = [r for r in rows if r["split"] == "fit"]
    held = [r for r in rows if r["split"] == "heldout"]
    xs, ys = np.array([r["score"] for r in fit_rows]), np.array([r["genuine"] for r in fit_rows])
    mapping = fit_logistic(xs, ys)
    hx, hy = np.array([r["score"] for r in held]), np.array([r["genuine"] for r in held])
    p = mapping.many(hx)
    llr = mapping.llr(hx)
    eer, eer_threshold = equal_error_rate(hx[hy == 1], hx[hy == 0])
    shifted = LogisticMap(mapping.a, mapping.b + logit(prior) - logit(mapping.train_prior), prior)
    return {"raw_score": "ECAPA cosine of a speech chunk to the person's voice centroid",
            "meaning": "P(same speaker | score) at the prior", "method": "logistic regression (Platt)",
            "prior": prior, "pair_prior": round(mapping.train_prior, 4),
            "prior_meaning": "P(a voice in view is the matched known person) — set per site (--prior)",
            "n": len(rows), "n_fit": len(fit_rows), "n_heldout": len(held),
            "speakers": len({r["speaker"] for r in rows}),
            "dataset": {"path": csv_path, "sha256": sha256_files([csv_path]),
                        "note": "synthetic TTS voices (tools/synth_speech.py, each voice = one speaker); real "
                                "voices differ less from each other than TTS voices — refit on recorded people"},
            "metrics": {"heldout": {**metrics(p, hy), "eer": round(eer, 4), "eer_threshold": round(eer_threshold, 4),
                                    "cllr": round(cllr(llr[hy == 1], llr[hy == 0]), 4)},
                        "genuine_mean": round(float(hx[hy == 1].mean()), 4),
                        "impostor_mean": round(float(hx[hy == 0].mean()), 4)},
            "at_voice_threshold": {"threshold": threshold, **identification_counts(held, threshold)},
            "reliability": reliability(p, hy),
            "table": {f"{t:.1f}": round(shifted(t), 3) for t in TABLE_AT},
            "map": shifted.to_dict(), "map_at_pair_prior": mapping.to_dict()}


# --- face -----------------------------------------------------------------------------------------------------------

def calibrate_face(report_path: str, prior: float, out_dir: str) -> Dict[str, Any]:
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    rows = sorted(report["by_threshold"], key=lambda r: r["threshold"])
    mapping, bins = bayes_bins([r["threshold"] for r in rows], [r["known_accuracy"] for r in rows],
                               [r["false_accept_rate"] for r in rows], int(report["known_test_images"]),
                               int(report["stranger_images"]), prior)
    return {"raw_score": "SFace similarity of the best gallery match",
            "meaning": "P(the face belongs to the matched known person | similarity)",
            "method": "Bayes over measured TAR / FAR bins (+0.5 Jeffreys counts), monotone", "prior": prior,
            "prior_meaning": "P(a face in view is a known person) — set per site (--prior)",
            "n": int(report["known_test_images"]) + int(report["stranger_images"]),
            "n_known": int(report["known_test_images"]), "n_strangers": int(report["stranger_images"]),
            "dataset": {"path": report_path, "sha256": sha256_files([report_path]), "source": report.get("root")},
            "bins": bins, "table": {f"{t:.2f}": round(mapping(t), 3) for t in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)},
            "map": {**mapping.to_dict(), "kind": "bayes_bins"}}


# --- gas ------------------------------------------------------------------------------------------------------------

def gas_predictions(data_path: str, classifier_path: str, n_splits: int) -> Dict[str, np.ndarray]:
    """Re-fit the deployed classifier (fixed hyper-parameters) on ``n_splits`` stratified splits → held-out probas."""
    import joblib
    from sklearn.base import clone

    from modules.gas.data import load_dataset
    from modules.gas.train import augment, split_indices, training_set

    estimator = joblib.load(classifier_path)["model"]
    dataset = load_dataset(data_path)
    proba, correct, split = [], [], []
    for seed in range(n_splits):
        train_idx, test_idx = split_indices(dataset, random_state=seed)
        X, y, _, _ = training_set(dataset, train_idx, augment(dataset, train_idx))
        model = clone(estimator).fit(X, y)
        predicted = model.predict(dataset.X[test_idx])
        probabilities = model.predict_proba(dataset.X[test_idx])
        classes = list(model.classes_)
        for row, label in zip(probabilities, predicted, strict=True):
            proba.append(float(row[classes.index(label)]))
        correct += list((predicted == dataset.y_gas[test_idx]).astype(float))
        split += [seed] * len(test_idx)
    return {"proba": np.array(proba), "correct": np.array(correct), "split": np.array(split)}


def calibrate_gas(data_path: str, classifier_path: str, metrics_path: str, n_splits: int,
                  out_dir: str) -> Dict[str, Any]:
    data = gas_predictions(data_path, classifier_path, n_splits)
    proba, correct = data["proba"], data["correct"]
    mapping = fit_isotonic(proba, correct)
    oof = out_of_fold(proba, correct, fit_isotonic)
    with open(metrics_path, encoding="utf-8") as f:
        deployed = json.load(f)
    test_rows = int(deployed["data"]["test_rows"])
    test_correct = int(round(float(deployed["classifier"]["test_accuracy"]) * test_rows))
    return {"raw_score": "predict_proba of the predicted gas (gas_reading.gas_confidence)",
            "meaning": "P(the gas label is right | proba)", "method": "isotonic regression over held-out splits",
            "n": len(proba), "splits": n_splits, "wrong": int((correct == 0).sum()),
            "dataset": {"path": data_path, "sha256": sha256_files([data_path]),
                        "model": classifier_path, "model_sha256": sha256_files([classifier_path])},
            "deployed_test": {"path": metrics_path, "correct": test_correct, "wrong": test_rows - test_correct,
                              "sha256": sha256_files([metrics_path])},
            "metrics": {"in_sample": metrics(mapping.many(proba), correct), "out_of_fold": metrics(oof, correct),
                        "mean_proba": round(float(proba.mean()), 4), "accuracy": round(float(correct.mean()), 4),
                        "min_proba": round(float(proba.min()), 4)},
            "reliability": reliability(proba, correct),
            "table": {f"{t:.1f}": round(mapping(t), 3) for t in TABLE_AT},
            "map": mapping.to_dict()}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate channel scores → probabilities (config/calibration/)")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    asr = sub.add_parser("asr")
    asr.add_argument("--audio", help="folder with manifest.csv (tools/synth_speech.py) → transcribe first")
    asr.add_argument("--csv", default="data/calibration/asr_phrases.csv")
    asr.add_argument("--model", default="base", help="faster-whisper model (the Pi fallback: base)")
    asr.add_argument("--wer-max", type=float, default=0.10)
    speaker = sub.add_parser("speaker")
    speaker.add_argument("--audio", help="folder with manifest.csv → ECAPA pair scores first")
    speaker.add_argument("--pairs", default="data/calibration/speaker_pairs.csv")
    speaker.add_argument("--prior", type=float, default=0.5)
    face = sub.add_parser("face")
    face.add_argument("--report", default="reports/identity_eval_lfw.json")
    face.add_argument("--prior", type=float, default=0.5)
    gas = sub.add_parser("gas")
    gas.add_argument("--data", default="data/gas/data.xls")
    gas.add_argument("--classifier", default="models/gas_classifier.joblib")
    gas.add_argument("--metrics", default="reports/gas_metrics.json")
    gas.add_argument("--splits", type=int, default=10)
    args = parser.parse_args(argv)

    if args.cmd == "asr":
        if args.audio:
            transcribe(args.audio, args.csv, args.model, args.config)
        result = calibrate_asr(args.csv, args.wer_max, args.out_dir)
    elif args.cmd == "speaker":
        from core.config import load_config

        if args.audio:
            speaker_pairs(args.audio, args.pairs, args.config)
        threshold = float(load_config(args.config).get("identity", {}).get("voice_threshold", 0.3))
        result = calibrate_speaker(args.pairs, threshold, args.prior, args.out_dir)
    elif args.cmd == "face":
        result = calibrate_face(args.report, args.prior, args.out_dir)
    else:
        result = calibrate_gas(args.data, args.classifier, args.metrics, args.splits, args.out_dir)
    save(args.cmd, result, args.out_dir)
    print(json.dumps({k: result[k] for k in ("n", "metrics", "table") if k in result}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
