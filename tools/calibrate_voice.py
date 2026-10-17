"""
Calibrates ``identity.voice_threshold`` (F10 / TOD-21): same-speaker vs different-speaker ECAPA scores, the equal
error rate (EER) and a suggested threshold.

    python -m tools.calibrate_voice --audio data/voices/      # <dir>/<name>/*.wav, ≥ 2 people (recommended)
    python -m tools.calibrate_voice                           # voice embeddings already in the identity registry

From audio, every file is cut into ``--window`` second pieces (like the live pipeline compares a chunk's speech with
a person's centroid). Each piece is scored against the centroid of the person's OTHER pieces (same speaker) and the
centroids of everyone else (different speaker). Recommended: 3 × 10 s of clean speech per person.
"""
import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

if __package__ in (None, ""):  # allow `python tools/calibrate_voice.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.identity import normalize  # noqa: E402


def scores(people: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param people: name → (n, d) normalised embeddings (n ≥ 2)
    :return: (same-speaker scores, different-speaker scores)
    """
    centroids = {name: normalize(e.mean(axis=0)) for name, e in people.items()}
    same, different = [], []
    for name, embeddings in people.items():
        for i, e in enumerate(embeddings):
            others = np.delete(embeddings, i, axis=0)
            same.append(float(normalize(others.mean(axis=0)) @ e))
            different += [float(c @ e) for other, c in centroids.items() if other != name]
    return np.array(same), np.array(different)


def equal_error_rate(same: np.ndarray, different: np.ndarray) -> Tuple[float, float]:
    """:return: (EER, threshold where false accepts ≈ false rejects)"""
    thresholds = np.linspace(-0.2, 1.0, 1201)
    far = np.array([np.mean(different >= t) for t in thresholds])
    frr = np.array([np.mean(same < t) for t in thresholds])
    best = int(np.argmin(np.abs(far - frr)))
    return float((far[best] + frr[best]) / 2), float(thresholds[best])


def from_audio(folder: str, window_sec: float, mic: Dict) -> Dict[str, np.ndarray]:
    import torchaudio

    from modules.audio.speaker import SAMPLE_RATE, ecapa_batch_embedder, preprocess, webrtc_speech_mask

    embed = ecapa_batch_embedder(mic.get("speaker_model_dir", "model/speechbrain"), mic.get("speaker_device", "auto"))
    mask = webrtc_speech_mask(int(mic.get("vad_aggressiveness", 2)))
    size, people = int(window_sec * SAMPLE_RATE), {}
    for person_dir in sorted(p for p in glob.glob(os.path.join(folder, "*")) if os.path.isdir(p)):
        pieces = []
        for path in sorted(glob.glob(os.path.join(person_dir, "*.wav"))):
            waveform, sample_rate = torchaudio.load(path)
            audio = preprocess(waveform.mean(dim=0).numpy(), sample_rate)
            speech = mask(audio)
            for start in range(0, len(audio) - size + 1, size // 2):
                frames = speech[int(start / SAMPLE_RATE / 0.03):int((start + size) / SAMPLE_RATE / 0.03)]
                if frames.mean() >= 0.5:                                  # mostly speech
                    pieces.append(audio[start:start + size])
        if len(pieces) >= 2:
            people[os.path.basename(person_dir)] = np.array([normalize(e) for e in embed(np.stack(pieces))])
        else:
            print(f"  skip {person_dir}: fewer than 2 usable {window_sec} s pieces")
    return people


def from_registry(identity_cfg: Dict) -> Dict[str, np.ndarray]:
    from core.identity import registry_from_config

    registry = registry_from_config(identity_cfg)
    people = {}
    try:
        for person in registry.list_persons():
            rows = registry.db.execute("SELECT vector FROM voice_embeddings WHERE person_id = ?",
                                       (person["person_id"],)).fetchall()
            if len(rows) >= 2:
                people[person["display_name"]] = np.array([normalize(np.frombuffer(r[0], np.float32)) for r in rows])
    finally:
        registry.close()
    return people


def report(people: Dict[str, np.ndarray]) -> Dict:
    same, different = scores(people)
    eer, threshold = equal_error_rate(same, different)
    pct = lambda a: {f"p{p}": round(float(np.percentile(a, p)), 3) for p in (5, 50, 95)}  # noqa: E731
    return {"people": {name: len(e) for name, e in people.items()}, "same_speaker": {
        "n": len(same), "mean": round(float(same.mean()), 3), **pct(same)}, "different_speaker": {
        "n": len(different), "mean": round(float(different.mean()), 3), **pct(different)},
        "eer": round(eer, 4), "suggested_voice_threshold": round(threshold, 3)}


def main(argv: Optional[List[str]] = None) -> None:
    from core.config import load_config

    parser = argparse.ArgumentParser(description="Voice threshold calibration (EER)")
    parser.add_argument("--audio", help="Folder <name>/*.wav (default: embeddings in the identity registry)")
    parser.add_argument("--window", type=float, default=3.0, help="Seconds per scored piece")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", help="Also write the report as JSON")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    people = from_audio(args.audio, args.window, config.get("audio", {}).get("mic", {})) if args.audio \
        else from_registry(config.get("identity", {}))
    if len(people) < 2:
        raise SystemExit("Need at least 2 people with ≥ 2 voice samples each (python -m tools.enroll --voice ...)")
    result = report(people)
    print(json.dumps(result, indent=1, ensure_ascii=False))
    current = config.get("identity", {}).get("voice_threshold")
    print(f"\nEER {result['eer']:.1%} at threshold {result['suggested_voice_threshold']} "
          f"(config identity.voice_threshold: {current})")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
