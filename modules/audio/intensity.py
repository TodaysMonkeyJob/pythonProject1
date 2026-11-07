"""
Sound intensity relative to the location's background (A4 / TOD-44).

The same 60 dB is silence next to a fan and a shout in a quiet bedroom, so the HFS audio input ``intensity`` is the
level above **this microphone's** background, in dB:

* background = the ``percentile`` (20th) of the per-second median frame levels over the last ``window_sec``
  (5 min) — follows a fan switched on, ignores short events; a fixed ``background_db`` (calibrated once per
  microphone, ``audio.mic.background_db`` or the device's config) overrides it;
* intensity = frame level − background, ≥ 0, smoothed exponentially over ``smooth_sec`` (2 s).

Until ``warmup_sec`` of audio has been seen the background is the quietest second so far.
"""
import collections
import math
from typing import Deque, List, Optional


class BackgroundLevel:
    def __init__(self, window_sec: float = 300.0, percentile: float = 20.0, smooth_sec: float = 2.0,
                 background_db: Optional[float] = None, warmup_sec: float = 10.0):
        self.percentile = float(percentile)
        self.smooth_sec = float(smooth_sec)
        self.fixed = None if background_db is None else float(background_db)
        self.warmup_sec = float(warmup_sec)
        self.seconds: Deque[float] = collections.deque(maxlen=max(1, int(window_sec)))   # per-second medians
        self._second: List[float] = []
        self._second_start: Optional[float] = None
        self.intensity = 0.0
        self._last: Optional[float] = None

    @property
    def background(self) -> Optional[float]:
        """:return: the background level, dBFS (None before the first full second)"""
        if self.fixed is not None:
            return self.fixed
        if not self.seconds:
            return None
        values = sorted(self.seconds)
        if len(values) < self.warmup_sec:
            return values[0]
        rank = (len(values) - 1) * self.percentile / 100
        low, high = math.floor(rank), math.ceil(rank)
        return values[low] + (values[high] - values[low]) * (rank - low)

    def update(self, db: float, now: float) -> float:
        """
        :param db: level of one frame, dBFS
        :param now: time, s
        :return: the smoothed intensity above the background, dB
        """
        if self._second_start is None or now - self._second_start >= 1.0:
            if self._second:
                self._second.sort()
                self.seconds.append(self._second[len(self._second) // 2])
            self._second, self._second_start = [], now
        self._second.append(float(db))
        background = self.background
        relative = 0.0 if background is None else max(0.0, db - background)
        dt = 0.0 if self._last is None else max(0.0, now - self._last)
        self._last = now
        alpha = 1.0 if self.smooth_sec <= 0 else 1.0 - math.exp(-dt / self.smooth_sec)
        self.intensity += alpha * (relative - self.intensity)
        return self.intensity

    def relative(self, db: float) -> float:
        """:return: dB above the background of one level (0 before calibration)"""
        background = self.background
        return 0.0 if background is None else round(max(0.0, db - background), 1)
