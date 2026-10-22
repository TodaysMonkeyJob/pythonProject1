"""
Device self-diagnostics (F23 / TOD-34): graded health checks per device kind, run inside the device workers.

Each monitor returns the *failing* checks as ``{check: {"value": …, "detail": "…"}}`` (factor and reason are in
``config/trust_registry.yaml`` → ``health.checks``); the worker publishes them as a ``device_health`` event when the
set changes (and every ``refresh_sec``), the router hands them to ``TrustRegistry.set_health`` → h = product of
the factors. Thresholds come from the ``camera`` / ``mic`` / ``gas`` sections of the same file.

* camera — no frames, FPS below target, frozen frame (identical sample hash), too dark / overexposed (mean
  brightness), blurred (Laplacian variance), lens covered (near-uniform frame), camera moved (most pixels changed
  between two samples while the motion detector saw no motion);
* microphone — clipping ratio, noise floor (10th-percentile frame loudness), digital silence (the router turns it
  into ``stuck_silence`` when the location's cameras see people), wrong sample rate;
* gas — sensor errors, flat / stuck curves, values outside the training range, clean-air baseline drift
  (R_peak and the Dirac point = the gate voltage of R_peak, vs the training clean-air mean).

numpy (+ cv2 for the Laplacian when available) only; one camera check ≈ 1 ms on a 160 px sample.
"""
import hashlib
import math
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

Failing = Dict[str, Dict[str, Any]]


def registry_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """:return: The parsed trust registry config when self-diagnostics are on (``trust.health``), else None"""
    trust = config.get("trust") or {}
    if not trust.get("health", False):
        return None
    from core.fuzzy import load_yaml

    return load_yaml(trust.get("registry_config", "config/trust_registry.yaml"))


class HealthPublisher:
    """Emits ``device_health`` when the set of failing checks changes, or every ``refresh_sec`` otherwise."""

    def __init__(self, emit: Callable[..., Any], kind: str, refresh_sec: float = 300.0):
        self.emit, self.kind, self.refresh = emit, kind, float(refresh_sec)
        self.last_key: Optional[tuple] = None
        self.last_at = float("-inf")

    def publish(self, failing: Failing, now: float, force: bool = False, **extra) -> bool:
        key = tuple(sorted(failing))
        if not force and not extra and key == self.last_key and now - self.last_at < self.refresh:
            return False
        self.last_key, self.last_at = key, now
        severity = "warning" if any(v.get("factor", 0) != 1.0 for v in failing.values()) else "info"
        self.emit("device_health", {"kind": self.kind, "failing": failing, **extra}, severity=severity)
        return True


def _gray(frame: np.ndarray, width: int = 160) -> np.ndarray:
    """Small grayscale sample of a BGR / gray frame (nearest-neighbour, numpy only)."""
    gray = frame if frame.ndim == 2 else frame[..., :3].mean(axis=2)
    step = max(1, gray.shape[1] // width)
    return gray[::step, ::step].astype(np.float32)


def laplacian_variance(gray: np.ndarray) -> float:
    try:
        import cv2

        return float(cv2.Laplacian(gray, cv2.CV_32F).var())
    except ImportError:                                       # numpy fallback (same 4-neighbour kernel)
        lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
        return float(lap.var())


def frame_checks(gray: np.ndarray, cfg: Dict[str, Any]) -> Failing:
    """Checks of one frame on its own: dark / overexposed / uniform (covered) / blurred."""
    failing: Failing = {}
    mean, std = float(gray.mean()), float(gray.std())
    if std < float(cfg.get("uniform_std", 6)):
        failing["lens_covered"] = {"value": round(std, 2), "detail": f"σ яскравості {std:.1f}"}
    elif mean < float(cfg.get("dark_mean", 25)):
        failing["too_dark"] = {"value": round(mean, 1), "detail": f"середня яскравість {mean:.0f}"}
    elif mean > float(cfg.get("bright_mean", 235)):
        failing["overexposed"] = {"value": round(mean, 1), "detail": f"середня яскравість {mean:.0f}"}
    if "lens_covered" not in failing:
        sharpness = laplacian_variance(gray)
        if sharpness < float(cfg.get("blur_laplacian", 15)):
            failing["blurred"] = {"value": round(sharpness, 2), "detail": f"дисперсія Лапласіана {sharpness:.1f}"}
    return failing


class CameraHealth:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.interval = float(cfg.get("interval_sec", 5))
        self.last_check: Optional[float] = None
        self.last_hash: Optional[str] = None
        self.same_since: Optional[float] = None
        self.previous: Optional[np.ndarray] = None
        self.moved_at: Optional[float] = None
        self.frames = 0
        self.no_frames_since: Optional[float] = None
        self.failing: Failing = {}

    def no_frame(self, now: float) -> Optional[Failing]:
        """A read returned nothing. :return: new failing checks when due"""
        if self.no_frames_since is None:
            self.no_frames_since = now
        if now - self.no_frames_since >= self.interval and "no_frames" not in self.failing:
            self.failing = {"no_frames": {"value": round(now - self.no_frames_since, 1), "detail": None}}
            return self.failing
        return None

    def observe(self, frame: np.ndarray, now: float, motion: bool = False) -> Optional[Failing]:
        """:return: The failing checks (a dict, possibly empty) when a check ran, else None"""
        self.frames += 1
        self.no_frames_since = None
        if self.last_check is not None and now - self.last_check < self.interval:
            return None
        elapsed = None if self.last_check is None else now - self.last_check
        self.last_check = now
        gray = _gray(frame)
        failing = frame_checks(gray, self.cfg)
        digest = hashlib.blake2b(np.ascontiguousarray(frame).tobytes(), digest_size=8).hexdigest()
        if digest == self.last_hash:
            self.same_since = self.same_since if self.same_since is not None else now - (elapsed or 0)
            frozen = now - self.same_since
            if frozen >= float(self.cfg.get("frozen_sec", 30)):
                failing["frozen_frame"] = {"value": round(frozen, 1), "detail": None}
        else:
            self.same_since = None
        self.last_hash = digest
        if self.previous is not None and self.previous.shape == gray.shape and not motion:
            changed = float(np.mean(np.abs(gray - self.previous) > 40))
            if changed >= float(self.cfg.get("moved_change", 0.6)):
                self.moved_at = now
        if self.moved_at is not None and now - self.moved_at <= float(self.cfg.get("moved_hold_sec", 60)):
            failing["camera_moved"] = {"value": None, "detail": None}
        self.previous = gray
        if elapsed:
            fps = self.frames / elapsed
            if fps < float(self.cfg.get("target_fps", 5)):
                failing["low_fps"] = {"value": round(fps, 2), "detail": f"{fps:.1f} кадр/с"}
        self.frames = 0
        self.failing = failing
        return failing


class MicHealth:
    def __init__(self, cfg: Dict[str, Any], expected_rate: int = 16000, actual_rate: Optional[int] = None):
        self.cfg = cfg
        self.interval = float(cfg.get("interval_sec", 10))
        self.rate_ok = actual_rate is None or int(actual_rate) == int(expected_rate)
        self.expected, self.actual = expected_rate, actual_rate
        self.start: Optional[float] = None
        self.samples = self.clipped = 0
        self.levels: list = []

    def observe(self, frame: np.ndarray, now: float) -> Optional[Failing]:
        """:param frame: int16 PCM frame. :return: failing checks when a window closed, else None"""
        if self.start is None:
            self.start = now
        x = np.abs(frame.astype(np.int32))
        self.samples += len(x)
        self.clipped += int(np.sum(x >= int(self.cfg.get("clip_level", 32000))))
        rms = math.sqrt(float(np.mean(frame.astype(np.float64) ** 2))) / 32768.0 if len(frame) else 0.0
        self.levels.append(20 * math.log10(rms) if rms > 0 else -120.0)
        if now - self.start < self.interval:
            return None
        failing: Failing = {}
        ratio = self.clipped / max(1, self.samples)
        if ratio > float(self.cfg.get("clip_ratio", 0.01)):
            failing["clipping"] = {"value": round(ratio, 4), "detail": f"{ratio:.0%} відліків"}
        levels = np.array(self.levels)
        if levels.max() < float(self.cfg.get("silence_db", -85)):
            failing["digital_silence"] = {"value": round(float(levels.max()), 1), "detail": None, "factor": 1.0}
        else:
            floor = float(np.percentile(levels, 10))
            if floor > float(self.cfg.get("noise_floor_db", -35)):
                failing["noisy"] = {"value": round(floor, 1), "detail": f"шумова підкладка {floor:.0f} dBFS"}
        if not self.rate_ok:
            failing["wrong_sample_rate"] = {"value": self.actual, "detail": f"{self.actual} Гц замість {self.expected}"}
        self.start, self.samples, self.clipped, self.levels = now, 0, 0, []
        return failing


class GasHealth:
    def __init__(self, cfg: Dict[str, Any], train_min: Sequence[float], train_max: Sequence[float],
                 clean_mean: Optional[Sequence[float]], voltages: Sequence[float]):
        """
        :param train_min / train_max: per-voltage resistance range of the training curves
        :param clean_mean: mean clean-air curve of the training data (None = no baseline check)
        """
        self.cfg = cfg
        self.lo, self.hi = np.asarray(train_min, float), np.asarray(train_max, float)
        self.clean = None if clean_mean is None else np.asarray(clean_mean, float)
        self.voltages = list(voltages)
        self.last_curve: Optional[np.ndarray] = None
        self.same = 0
        self.errors = 0

    @classmethod
    def from_dataset(cls, cfg: Dict[str, Any], data_path: str = "data/gas/data.xls") -> "GasHealth":
        from modules.gas.data import CLEAN_AIR, load_dataset

        dataset = load_dataset(data_path)
        clean = dataset.X[dataset.y_gas == CLEAN_AIR]
        voltages = [round(0.1 * i, 1) for i in range(dataset.X.shape[1])]
        return cls(cfg, dataset.X.min(axis=0), dataset.X.max(axis=0), clean.mean(axis=0) if len(clean) else None,
                   voltages)

    def error(self) -> Failing:
        self.errors += 1
        return {"sensor_error": {"value": self.errors, "detail": f"{self.errors} помилок поспіль"}}

    def observe(self, curve: Sequence[float], gas: Optional[str]) -> Failing:
        """:return: failing checks of this reading (``baseline_ok`` True/False when a clean-air self-test ran)"""
        self.errors = 0
        x = np.asarray(curve, float)
        failing: Failing = {}
        if self.last_curve is not None and np.array_equal(x, self.last_curve):
            self.same += 1
        else:
            self.same = 0
        self.last_curve = x
        if self.same >= int(self.cfg.get("flat_polls", 30)):
            failing["flat_readings"] = {"value": self.same, "detail": f"{self.same} однакових показів"}
        margin = float(self.cfg.get("range_margin", 0.5))
        outside = (x < self.lo * (1 - margin)) | (x > self.hi * (1 + margin))
        if outside.any():
            failing["out_of_range"] = {"value": int(outside.sum()), "detail": f"{int(outside.sum())} з {len(x)} точок"}
        return failing

    def baseline(self, curve: Sequence[float]) -> Optional[Dict[str, Any]]:
        """
        Clean-air self-test: R_peak and the Dirac point (the gate voltage of R_peak) vs the training clean-air mean.

        :return: {"ok": bool, "detail": …} or None when no baseline is known
        """
        if self.clean is None:
            return None
        x = np.asarray(curve, float)
        peak, ref = float(x.max()), float(self.clean.max())
        dirac, ref_dirac = int(np.argmax(x)), int(np.argmax(self.clean))
        drift = abs(peak - ref) / max(abs(ref), 1e-9)
        ok = drift <= float(self.cfg.get("baseline_tolerance", 0.25)) and abs(dirac - ref_dirac) <= 2
        return {"ok": ok, "drift": round(drift, 3), "dirac_v": self.voltages[dirac] if dirac < len(self.voltages)
                else dirac, "detail": f"R_peak {drift:.0%} від еталону, точка Дірака "
                                      f"{self.voltages[dirac] if dirac < len(self.voltages) else dirac} В"}
