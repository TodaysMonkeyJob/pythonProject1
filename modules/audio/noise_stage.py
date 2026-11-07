"""
Noise stage (A4 / TOD-44): the background the microphone hears between words is information, not just something to
remove.

* ``NoiseProfile`` keeps a recursive average of the power spectrum of the non-speech frames (the VAD says no speech)
  and the last ``keep_sec`` of non-speech audio — the "noise profile" the sound-type classifier labels (fan, keyboard,
  walking, …) every time a speech segment is processed;
* ``wiener_denoise`` — a soft Wiener mask from that noise spectrum (gain = SNR / (1 + SNR), floor ``min_gain``)
  for speech segments before Whisper; off by default (``audio.mic.denoise``): Whisper is trained on noisy speech and
  a mask can remove consonants;
* ``SoundBurst`` cuts non-speech sound events (louder than the background by ``min_db``, ``min_sec``–``max_sec``
  long) — a scream, a bang, glass — for the classifier too.
"""
import collections
from typing import Deque, Optional

import numpy as np

N_FFT = 512


class NoiseProfile:
    def __init__(self, sample_rate: int = 16000, keep_sec: float = 3.0, alpha: float = 0.95):
        """
        :param keep_sec: non-speech audio kept for classification
        :param alpha: recursive averaging of the noise power spectrum (per non-speech frame)
        """
        self.sample_rate = sample_rate
        self.alpha = alpha
        self.psd: Optional[np.ndarray] = None
        self.frames: Deque[np.ndarray] = collections.deque()
        self.keep = int(keep_sec * sample_rate)
        self._kept = 0
        self._window = np.hanning(N_FFT)

    def update(self, frame: np.ndarray, is_speech: bool) -> None:
        """:param frame: int16 or float samples of one VAD frame"""
        if is_speech:
            return
        audio = to_float(frame)
        self.frames.append(audio)
        self._kept += len(audio)
        while self._kept - len(self.frames[0]) >= self.keep:
            self._kept -= len(self.frames.popleft())
        padded = np.zeros(N_FFT)
        padded[:min(N_FFT, len(audio))] = audio[:N_FFT]
        power = np.abs(np.fft.rfft(padded * self._window)) ** 2
        self.psd = power if self.psd is None else self.alpha * self.psd + (1 - self.alpha) * power

    def audio(self) -> np.ndarray:
        """:return: the last ``keep_sec`` of non-speech audio (float32, −1..1)"""
        return np.concatenate(list(self.frames)).astype(np.float32) if self.frames else np.zeros(0, np.float32)

    @property
    def seconds(self) -> float:
        return self._kept / self.sample_rate


def to_float(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    return frame.astype(np.float32) / 32768.0 if frame.dtype == np.int16 else frame.astype(np.float32)


def wiener_denoise(segment: np.ndarray, noise_psd: Optional[np.ndarray], min_gain: float = 0.1,
                   over_subtraction: float = 1.0) -> np.ndarray:
    """
    :param segment: int16 or float audio
    :param noise_psd: the noise power spectrum of ``NoiseProfile`` (N_FFT // 2 + 1 bins); None → unchanged
    :return: denoised float32 audio of the same length (soft Wiener mask, 50 % overlap-add)
    """
    audio = to_float(segment)
    if noise_psd is None or len(audio) < N_FFT:
        return audio
    hop = N_FFT // 2
    window = np.sqrt(np.hanning(N_FFT + 1)[:N_FFT])          # √Hann analysis + synthesis: weights sum to 1 at 50 %
    padded = np.concatenate([np.zeros(hop), audio, np.zeros(N_FFT)])   # every sample is covered by two frames
    out = np.zeros(len(padded))
    norm = np.zeros(len(padded))
    for start in range(0, len(audio) + hop, hop):
        spectrum = np.fft.rfft(padded[start:start + N_FFT] * window)
        power = np.abs(spectrum) ** 2
        snr = np.maximum(power / (over_subtraction * noise_psd + 1e-12) - 1.0, 0.0)
        gain = np.maximum(snr / (1.0 + snr), min_gain)
        out[start:start + N_FFT] += np.fft.irfft(spectrum * gain, N_FFT) * window
        norm[start:start + N_FFT] += window ** 2
    return (out / np.maximum(norm, 1e-3))[hop:hop + len(audio)].astype(np.float32)


class SoundBurst:
    """Collects a non-speech sound event: frames louder than the background by ``min_db``."""

    def __init__(self, sample_rate: int = 16000, min_db: float = 10.0, min_sec: float = 0.3, max_sec: float = 3.0,
                 gap_sec: float = 0.3):
        self.sample_rate, self.min_db = sample_rate, min_db
        self.min_samples, self.max_samples = int(min_sec * sample_rate), int(max_sec * sample_rate)
        self.gap_samples = int(gap_sec * sample_rate)
        self.frames: list = []
        self.quiet = 0
        self.peak = 0.0

    def push(self, frame: np.ndarray, relative_db: float, is_speech: bool) -> Optional[np.ndarray]:
        """:return: a finished burst (float32) or None"""
        loud = relative_db >= self.min_db and not is_speech
        if loud:
            self.frames.append(to_float(frame))
            self.quiet = 0
            self.peak = max(self.peak, relative_db)
        elif self.frames:
            self.quiet += len(frame)
            self.frames.append(to_float(frame))
        length = sum(len(f) for f in self.frames)
        if self.frames and (self.quiet >= self.gap_samples or length >= self.max_samples or is_speech):
            audio, sound = np.concatenate(self.frames), length - self.quiet    # the trailing quiet does not count
            self.frames, self.quiet = [], 0
            return audio if sound >= self.min_samples else None
        return None

    def take_peak(self) -> float:
        peak, self.peak = self.peak, 0.0
        return peak
