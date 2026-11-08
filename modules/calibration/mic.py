"""
Microphone calibration (C1 / TOD-54): the background level (dB) and the noise power spectrum of one microphone.

* stored → the background (``intensity.BackgroundLevel.seed``) and the noise profile start from the stored values,
  no warm-up; short check: the median level of ``check_sec`` (3 s) of non-speech audio within ``tolerance_db`` of the
  stored background;
* full calibration: ``calibrate_sec`` (60 s) of non-speech audio → the 20th percentile of the frame levels (as the
  live background) + the noise spectrum.
"""
from typing import Any, Dict, List

import numpy as np

from modules.calibration.base import Calibrator


class MicCalibrator(Calibrator):
    kind = "mic"

    def __init__(self, *args, background=None, noise=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.background, self.noise = background, noise
        self._levels: List[float] = []
        self._first = None

    def apply_stored(self) -> None:
        values = self.state.values
        if values.get("background_db") is not None and self.background is not None:
            self.background.seed(float(values["background_db"]))
        if values.get("noise_psd") and self.noise is not None:
            self.noise.psd = np.asarray(values["noise_psd"], dtype=float)

    def observe(self, db: float, speech: bool, now: float) -> None:
        self.state.tick()
        if speech or self.state.state not in ("checking", "calibrating"):
            return
        if self._first is None:
            self._first, self._levels = now, []
        self._levels.append(float(db))
        elapsed = now - self._first
        if self.state.state == "checking" and elapsed >= float(self.policy.get("check_sec", 3)):
            stored = float(self.state.values.get("background_db"))
            measured = float(np.median(self._levels))
            tolerance = float(self.policy.get("tolerance_db", 6))
            self.state.check_result(abs(measured - stored) <= tolerance,
                                    f"фон {measured:.1f} дБ, збережено {stored:.1f} дБ (допуск ±{tolerance:g})",
                                    measured_db=round(measured, 1))
            self._first = None
        elif self.state.state == "calibrating":
            seconds = float(self.policy.get("calibrate_sec", 60))
            self.state.set_progress(elapsed / seconds)
            if elapsed >= seconds:
                background = round(float(np.percentile(self._levels, 20)), 1)
                values: Dict[str, Any] = {"background_db": background}
                if self.noise is not None and self.noise.psd is not None:
                    values["noise_psd"] = np.round(np.asarray(self.noise.psd, float), 10).tolist()
                self.state.complete(values, f"{seconds:g} с фону без мовлення, 20-й перцентиль рівня кадрів",
                                    validation={"frames": len(self._levels)})
                if self.background is not None:
                    self.background.seed(background)
                self._first = None

    def control(self, payload: Dict[str, Any]) -> bool:
        done = super().control(payload)
        if done:
            self._first = None
        return done
