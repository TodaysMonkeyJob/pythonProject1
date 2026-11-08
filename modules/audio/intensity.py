"""
Sound intensity relative to the location's background (A4 / TOD-44).

The same 60 dB is silence next to a fan and a shout in a quiet bedroom, so the HFS audio input ``intensity`` is the
level above **this microphone's** background, in dB:

* background = the ``percentile`` (20th) of the per-second median frame levels over the last ``window_sec``
  (5 min) — follows a fan switched on, ignores short events; a fixed ``background_db`` (calibrated once per
  microphone, ``audio.mic.background_db`` or the device's config) overrides it;
* intensity = frame level − background, ≥ 0, smoothed exponentially over ``smooth_sec`` (2 s).

Until ``warmup_sec`` of audio has been seen the background is the quietest second so far — or, C1 (TOD-54), the
microphone's stored calibration (``seed``).
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

    def seed(self, background_db: float) -> None:
        """C1 (TOD-54): start from a stored background — the warm-up is skipped, the percentile keeps adapting"""
        self.seconds.clear()
        self.seconds.extend([float(background_db)] * max(1, int(self.warmup_sec)))

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

    def spread(self) -> float:
        """:return: half the spread (p30 − p10) / 2 of the per-second background levels, dB — how far the background
                    estimate can be off (A5 / TOD-45 intensity error); 0 with a calibrated ``background_db``"""
        if self.fixed is not None or len(self.seconds) < self.warmup_sec:
            return 0.0
        values = sorted(self.seconds)

        def at(q: float) -> float:
            rank = (len(values) - 1) * q / 100
            low, high = math.floor(rank), math.ceil(rank)
            return values[low] + (values[high] - values[low]) * (rank - low)

        return round(max(0.0, at(self.percentile + 10) - at(max(0.0, self.percentile - 10))) / 2, 2)

    def relative(self, db: float) -> float:
        """:return: dB above the background of one level (0 before calibration)"""
        background = self.background
        return 0.0 if background is None else round(max(0.0, db - background), 1)
