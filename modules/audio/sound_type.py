"""
Sound-type classification (A4 / TOD-44): what the background / a non-speech sound is — keyboard, walking, fan /
machinery, scream, unknown (optional impact, glass, alarm) — and how dangerous that type is
(``config/audio_types.yaml``).

Features (numpy only, 16 kHz, 32 ms frames / 16 ms hop): 40 log-mel bands (mean, std), spectral centroid,
bandwidth, roll-off, flatness, zero-crossing rate (mean, std each), frame energy mean / std / max, crest and
kurtosis of the frame energies (impulsive sounds: keys, steps, impacts) and the energy modulation peak 0.5–8 Hz
(steps ~2 Hz, typing 4–8 Hz) — 96 numbers per clip.

The model (``models/sound_types.joblib``, ``python -m tools.train_sound_types``) is a scikit-learn pipeline (scaler +
single-threaded random forest, ~3 ms per clip); a
prediction below ``min_confidence`` is ``unknown``. Without a model every clip is ``unknown`` with ``danger: None``
(not evidence: the HFS ignores it) — no recordings of our own sound types are in the repo yet.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

SAMPLE_RATE = 16000
N_FFT, HOP, N_MELS = 512, 256, 40
UNKNOWN = "unknown"


def mel_filterbank(sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS,
                   fmin: float = 50.0, fmax: Optional[float] = None) -> np.ndarray:
    """:return: (n_mels, n_fft // 2 + 1) triangular mel filters (HTK mel scale)"""
    fmax = fmax or sample_rate / 2

    def mel(f):
        return 2595 * np.log10(1 + np.asarray(f) / 700)

    def hz(m):
        return 700 * (10 ** (np.asarray(m) / 2595) - 1)

    points = hz(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    bank = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        left, centre, right = points[i:i + 3]
        rising = (bins - left) / max(centre - left, 1e-9)
        falling = (right - bins) / max(right - centre, 1e-9)
        bank[i] = np.maximum(0, np.minimum(rising, falling))
    return bank


_MEL = mel_filterbank()
_WINDOW = np.hanning(N_FFT)


def frames_of(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < N_FFT:
        audio = np.pad(audio, (0, N_FFT - len(audio)))
    count = 1 + (len(audio) - N_FFT) // HOP
    index = np.arange(N_FFT)[None, :] + HOP * np.arange(count)[:, None]
    return audio[index]


def features(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """:return: the 96-number feature vector of a clip (float32 −1..1 or int16 audio at 16 kHz)"""
    audio = np.asarray(audio)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"features need {SAMPLE_RATE} Hz audio, got {sample_rate}")
    frames = frames_of(audio)
    spectrum = np.abs(np.fft.rfft(frames * _WINDOW, axis=1)) + 1e-10
    power = spectrum ** 2
    logmel = np.log(power @ _MEL.T + 1e-10)
    freqs = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
    total = spectrum.sum(axis=1)
    centroid = (spectrum * freqs).sum(axis=1) / total
    bandwidth = np.sqrt((spectrum * (freqs - centroid[:, None]) ** 2).sum(axis=1) / total)
    cumulative = np.cumsum(spectrum, axis=1)
    rolloff = freqs[np.argmax(cumulative >= 0.85 * cumulative[:, -1:], axis=1)]
    flatness = np.exp(np.log(spectrum).mean(axis=1)) / spectrum.mean(axis=1)
    zcr = (np.abs(np.diff(np.sign(frames), axis=1)) > 0).mean(axis=1)
    energy = np.log(power.sum(axis=1) + 1e-10)
    linear = np.exp(energy - energy.max())
    crest = float(linear.max() / (linear.mean() + 1e-10))
    centred = energy - energy.mean()
    kurtosis = float((centred ** 4).mean() / ((centred ** 2).mean() ** 2 + 1e-10))
    modulation = 0.0
    if len(energy) >= 16:
        rate = SAMPLE_RATE / HOP
        spectrum_e = np.abs(np.fft.rfft(centred))
        mod_freqs = np.fft.rfftfreq(len(centred), 1 / rate)
        band = (mod_freqs >= 0.5) & (mod_freqs <= 8.0)
        if band.any():
            modulation = float(mod_freqs[band][np.argmax(spectrum_e[band])])
    scalars = []
    for series in (centroid / 1000, bandwidth / 1000, rolloff / 1000, flatness, zcr):
        scalars += [float(series.mean()), float(series.std())]
    scalars += [float(energy.mean()), float(energy.std()), float(energy.max()), crest, kurtosis, modulation]
    return np.concatenate([logmel.mean(axis=0), logmel.std(axis=0), scalars]).astype(np.float32)


@dataclass
class SoundType:
    type: str
    confidence: float
    danger: Optional[float]                 # None: no model, not evidence
    model: Optional[str] = None
    scores: Optional[Dict[str, float]] = None

    def payload(self) -> Dict[str, Any]:
        out = {"type": self.type, "confidence": round(self.confidence, 3), "danger": self.danger,
               "model": self.model}
        if self.scores:
            out["scores"] = {k: round(v, 3) for k, v in sorted(self.scores.items(), key=lambda kv: -kv[1])[:3]}
        return out


def load_types(path: str = "config/audio_types.yaml") -> Dict[str, Any]:
    """:return: {"types": {name: {danger, …}}, "min_confidence", "unknown_danger"}"""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    types = data.get("types") or {}
    for name, entry in types.items():
        danger = float((entry or {}).get("danger", -1))
        if not 0 <= danger <= 1:
            raise ValueError(f"audio_types.yaml {name}: danger must be within 0..1")
    if UNKNOWN not in types:
        raise ValueError("audio_types.yaml needs an 'unknown' type")
    return data


class SoundTypeClassifier:
    def __init__(self, types: Dict[str, Any], model=None, model_path: Optional[str] = None):
        """
        :param types: ``load_types()``
        :param model: a fitted scikit-learn classifier with ``predict_proba`` / ``classes_`` (None = no model)
        """
        self.types = types.get("types") or {}
        self.min_confidence = float(types.get("min_confidence", 0.5))
        self.model, self.model_path = model, model_path

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]] = None) -> "SoundTypeClassifier":
        """``audio.mic.sound_types`` (YAML) and ``audio.mic.sound_type_model`` (absent file → no model)."""
        cfg = cfg or {}
        types = load_types(cfg.get("sound_types", "config/audio_types.yaml"))
        path = cfg.get("sound_type_model", "models/sound_types.joblib")
        model = None
        if path and os.path.isfile(path):
            import joblib

            model = joblib.load(path)
        return cls(types, model, path if model is not None else None)

    def danger(self, name: str) -> float:
        return float((self.types.get(name) or self.types[UNKNOWN])["danger"])

    def classify(self, audio: np.ndarray) -> SoundType:
        if self.model is None:
            return SoundType(UNKNOWN, 0.0, None, None)
        proba = self.model.predict_proba(features(audio)[None, :])[0]
        scores = {str(c): float(p) for c, p in zip(self.model.classes_, proba, strict=True)}
        best = max(scores, key=scores.get)
        if scores[best] < self.min_confidence:
            return SoundType(UNKNOWN, scores[best], self.danger(UNKNOWN), os.path.basename(self.model_path or ""),
                             scores)
        return SoundType(best, scores[best], self.danger(best), os.path.basename(self.model_path or ""), scores)


def clips(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, clip_sec: float = 2.0) -> List[np.ndarray]:
    """:return: the audio cut into ``clip_sec`` pieces (the last one if at least half long) — training / evaluation"""
    size = int(clip_sec * sample_rate)
    out = [audio[i:i + size] for i in range(0, max(1, len(audio) - size + 1), size)]
    tail = len(audio) % size
    if len(audio) > size and tail >= size // 2:
        out.append(audio[-size:])
    return [c for c in out if len(c) >= size // 2]
