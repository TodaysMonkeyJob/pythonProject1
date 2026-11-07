"""
Audio subsystem latency (A4 / TOD-44) → ``reports/audio_bench/``::

    python -m tools.audio_bench [--out reports/audio_bench] [--whisper tiny] [--no-whisper]

Per-stage cost on this machine: one 30 ms frame of the microphone loop with the A4 additions (background level +
noise profile + burst detector), the sound-type features + prediction of a 2 s clip, danger-word matching of a
transcript, and local Whisper (faster-whisper ``--whisper``, CPU int8) on ``speech_to_text_whisper/warmup_audio.wav``
→ real-time factor. Real time means: a frame's work ≪ 30 ms and Whisper's real-time factor < 1 with the sound
types in their own thread.

The classifier is timed with a model trained on synthetic sounds (``synthetic_sounds``: hum, clicks, thumps, a
pitched scream) — for timing only; there is no trained model of real recordings yet. Run the same command on the
Raspberry Pi for the deployment numbers (not measured here).
"""
import argparse
import json
import os
import platform
import time
from typing import Any, Dict, Tuple

import numpy as np

from modules.audio.intensity import BackgroundLevel
from modules.audio.keywords import DangerWords
from modules.audio.noise_stage import NoiseProfile, SoundBurst
from modules.audio.sound_type import SAMPLE_RATE, SoundTypeClassifier, features, load_types

FRAME = 480     # 30 ms at 16 kHz


def synthetic_sounds(kind: str, seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    """Very simple stand-ins (tests, timing): fan = filtered hum, keyboard = click trains, walking = thumps, scream."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    noise = rng.normal(0, 0.005, n)
    if kind == "fan":
        base = rng.uniform(90, 140)
        return (0.05 * np.sin(2 * np.pi * base * t) + 0.03 * np.convolve(rng.normal(0, 1, n), np.ones(40) / 40,
                                                                           "same") + noise).astype(np.float32)
    if kind in ("keyboard", "walking"):
        rate = rng.uniform(5, 9) if kind == "keyboard" else rng.uniform(1.6, 2.4)
        out = noise.copy()
        for start in np.arange(0, seconds, 1 / rate) + rng.uniform(0, 0.05):
            i = int(start * SAMPLE_RATE)
            length = 160 if kind == "keyboard" else 1600
            if i + length >= n:
                break
            env = np.exp(-np.arange(length) / (length / 5))
            carrier = rng.normal(0, 1, length) if kind == "keyboard" else np.sin(2 * np.pi * 70 * np.arange(length)
                                                                                    / SAMPLE_RATE)
            out[i:i + length] += (0.3 if kind == "keyboard" else 0.5) * env * carrier
        return out.astype(np.float32)
    if kind == "scream":
        pitch = rng.uniform(700, 1100) * (1 + 0.03 * np.sin(2 * np.pi * 6 * t))
        phase = 2 * np.pi * np.cumsum(pitch) / SAMPLE_RATE
        voice = sum(np.sin(k * phase) / k for k in range(1, 6))
        return (0.4 * voice * np.minimum(1, t / 0.1) + noise).astype(np.float32)
    raise ValueError(kind)


def synthetic_model(seed: int = 0):
    from tools.train_sound_types import build_model

    x, y = [], []
    for kind in ("fan", "keyboard", "walking", "scream"):
        for i in range(20):
            x.append(features(synthetic_sounds(kind, seed=seed * 100 + i)))
            y.append(kind)
    return build_model(seed).fit(np.array(x), np.array(y))


def timed(fn, repeat: int) -> Tuple[float, float]:
    times = []
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        times.append((time.perf_counter() - started) * 1000)
    times.sort()
    return round(float(np.mean(times)), 3), round(times[int(0.95 * (len(times) - 1))], 3)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audio subsystem latency")
    parser.add_argument("--out", default=os.path.join("reports", "audio_bench"))
    parser.add_argument("--whisper", default="tiny")
    parser.add_argument("--no-whisper", action="store_true")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(1)
    report: Dict[str, Any] = {"machine": f"{platform.system()} {platform.machine()}",
                              "python": platform.python_version()}

    background, noise, burst = BackgroundLevel(), NoiseProfile(), SoundBurst()
    frames = (rng.normal(0, 800, (2000, FRAME))).astype(np.int16)
    state = {"i": 0, "t": 0.0}

    def frame_step():
        frame = frames[state["i"] % len(frames)]
        state["i"] += 1
        state["t"] += 0.03
        db = 20 * np.log10(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) / 32768 + 1e-12)
        background.update(db, state["t"])
        relative = background.relative(db)
        noise.update(frame, False)
        burst.push(frame, relative, False)

    report["frame_ms"] = dict(zip(("mean", "p95"), timed(frame_step, 2000), strict=True))
    classifier = SoundTypeClassifier(load_types(), synthetic_model(), "synthetic (timing only)")
    clip = synthetic_sounds("keyboard", 2.0, seed=99)
    report["sound_type_ms_per_2s_clip"] = dict(zip(("mean", "p95"), timed(lambda: classifier.classify(clip), 50),
                                                   strict=True))
    words = DangerWords.load()
    text = "Ой, тут щось горить, здається пожежа на кухні, допоможіть будь ласка"
    report["keywords_ms"] = dict(zip(("mean", "p95"), timed(lambda: words.grade(text), 200), strict=True))
    if not args.no_whisper:
        report["whisper"] = whisper_rtf(args.whisper)
    with open(os.path.join(args.out, "latency.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    lines = ["# Audio subsystem latency (A4 / TOD-44)", "",
             f"`python -m tools.audio_bench` on {report['machine']} (Python {report['python']}). **Not the Raspberry "
             "Pi** — run the same command there for the deployment numbers.", "",
             "| Stage | Mean, ms | p95, ms | Budget |", "| -- | -- | -- | -- |",
             f"| Microphone frame (background + noise profile + burst) | {report['frame_ms']['mean']} | "
             f"{report['frame_ms']['p95']} | 30 ms per frame |",
             f"| Sound type of a 2 s clip (features + prediction) | {report['sound_type_ms_per_2s_clip']['mean']} | "
             f"{report['sound_type_ms_per_2s_clip']['p95']} | own thread, per segment |",
             f"| Danger words of a transcript | {report['keywords_ms']['mean']} | {report['keywords_ms']['p95']} | "
             "per speech segment |"]
    if "whisper" in report:
        w = report["whisper"]
        lines.append(f"| Whisper {w['model']} on {w['audio_sec']} s of speech | {w['ms']} | — | real-time factor "
                     f"{w['rtf']} (< 1 = faster than real time) |")
    lines += ["", "The classifier was timed with a model trained on synthetic sounds (timing only)."]
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def whisper_rtf(model_name: str) -> Dict[str, Any]:
    import soundfile as sf
    from faster_whisper import WhisperModel

    audio, rate = sf.read("speech_to_text_whisper/warmup_audio.wav", dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        g = np.gcd(int(rate), SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, int(rate) // g).astype(np.float32)
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=4)
    list(model.transcribe(audio[:SAMPLE_RATE], beam_size=1)[0])         # warm-up
    started = time.perf_counter()
    segments, _ = model.transcribe(audio, beam_size=5, language="uk", condition_on_previous_text=False)
    text = " ".join(s.text for s in segments)
    elapsed = time.perf_counter() - started
    seconds = len(audio) / SAMPLE_RATE
    return {"model": model_name, "audio_sec": round(seconds, 1), "ms": round(elapsed * 1000),
            "rtf": round(elapsed / seconds, 3), "chars": len(text.strip())}


if __name__ == "__main__":
    raise SystemExit(main())
