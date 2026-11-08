"""
Camera calibration (C1 / TOD-54): the motion background image and the per-cell noise levels of one camera
(``vision.motion.MotionDetector.export_state`` / ``restore_state``).

* stored → the background model is seeded with the stored image, the noise levels taken over, no warm-up; short
  check: the median difference of ``check_frames`` (15) live frames from the stored background (both 160 px grey)
  within ``tolerance`` grey levels (camera moved / scene changed → fails);
* full calibration: ``calibrate_sec`` (30 s) of frames without motion, then the detector's learned state is stored.
"""
from typing import List, Optional

import cv2
import numpy as np

from modules.calibration.base import Calibrator


def small_gray(frame: np.ndarray, width: int = 160) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape[:2]
    return cv2.resize(gray, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)


class CameraCalibrator(Calibrator):
    kind = "camera"

    def __init__(self, *args, motion=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion = motion
        self._diffs: List[float] = []
        self._still_since: Optional[float] = None
        self._reference: Optional[np.ndarray] = None

    def apply_stored(self) -> None:
        values = self.state.values
        if not values or self.motion is None:
            return
        self.motion.restore_state(values)
        reference = self.motion.background_of(values)
        self._reference = None if reference is None else reference.astype(np.float32)

    def observe(self, frame: np.ndarray, moving: bool, now: float) -> None:
        self.state.tick()
        if self.state.state == "checking" and self._reference is not None:
            live = small_gray(frame, self._reference.shape[1]).astype(np.float32)
            if live.shape != self._reference.shape:
                live = cv2.resize(live, self._reference.shape[::-1])
            self._diffs.append(float(np.mean(np.abs(live - self._reference))))
            if len(self._diffs) >= int(self.policy.get("check_frames", 15)):
                diff, tolerance = float(np.median(self._diffs)), float(self.policy.get("tolerance", 20))
                self.state.check_result(diff <= tolerance, f"кадр відрізняється від збереженого фону на "
                                                           f"{diff:.1f} рівня сірого (допуск {tolerance:g})",
                                        difference=round(diff, 1))
                self._diffs = []
        elif self.state.state == "checking":                # nothing to compare with (no stored image)
            self.state.check_result(False, "немає збереженого фону")
        elif self.state.state == "calibrating":
            if moving:
                self._still_since = None                    # people / objects moving: wait for a still scene
                return
            if self._still_since is None:
                self._still_since = now
            seconds = float(self.policy.get("calibrate_sec", 30))
            self.state.set_progress((now - self._still_since) / seconds)
            if now - self._still_since >= seconds and self.motion is not None:
                values = self.motion.export_state()
                if values is not None:
                    self.state.complete(values, f"{seconds:g} с кадрів без руху: фон MOG2 + рівні шуму комірок",
                                        validation={"grid": values.get("grid")})
                    reference = self.motion.background_of(values)
                    self._reference = None if reference is None else reference.astype(np.float32)
                self._still_since = None

    def control(self, payload) -> bool:
        done = super().control(payload)
        if done:
            self._diffs, self._still_since = [], None
        return done
