"""
Gas worker: reads an R(Ug) curve every ``gas_poll_sec``, classifies it and publishes events.

Curve sources (``gas.mode``):
  * ``replay`` — rows of ``gas.replay_csv`` (see modules.gas.data.write_replay_csv), looped if ``replay_loop``;
  * ``sensor`` — ``gas.reader: "package.module:function"`` returning 16 resistance values for Ug = 0.0..1.5 V.

Events:
  * ``gas_reading`` (info) every poll: gas, gas_confidence, concentration_pct, trend, sensor_id, curve summary;
  * ``alert`` (critical) when concentration ≥ ``thresholds[gas]`` (reason ``gas_threshold``) or the
    concentration rises faster than ``rise_rate_threshold`` %/min over ``trend_window_sec`` (reason ``gas_rising``).
    Each reason is repeated at most once per ``alert_cooldown_sec``.
"""
import collections
import importlib
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.module import BaseModule
from modules.gas.data import read_replay_csv
from system_setup.logger_setup import init_logger

logger = init_logger()

Reader = Callable[[], Sequence[float]]


class CsvReplayReader:
    """
    Returns the curves of a replay CSV one by one.
    """

    def __init__(self, path: str, loop: bool = True):
        """
        :param path: CSV written by modules.gas.data.write_replay_csv()
        :param loop: Start over after the last row; otherwise keep returning the last row
        """
        self.curves = read_replay_csv(path)
        self.loop = loop
        self.position = 0

    def __call__(self) -> List[float]:
        curve = self.curves[self.position]
        if self.position + 1 < len(self.curves):
            self.position += 1
        elif self.loop:
            self.position = 0
        return curve


def load_reader(gas_config: Dict[str, Any]) -> Reader:
    """
    :param gas_config: gas section of the config
    :return: Callable returning one curve
    """
    mode = gas_config.get("mode", "replay")
    if mode == "replay":
        return CsvReplayReader(gas_config["replay_csv"], loop=gas_config.get("replay_loop", True))
    if mode == "sensor":
        module_name, _, attr = str(gas_config.get("reader", "")).partition(":")
        if not attr:
            raise ValueError("gas.reader must be 'package.module:function' in sensor mode")
        return getattr(importlib.import_module(module_name), attr)
    raise ValueError(f"Unknown gas.mode {mode!r} (expected 'replay' or 'sensor')")


class TrendTracker:
    """
    Keeps (time, concentration) points of the last ``window_sec`` seconds and fits a line through them.
    """

    def __init__(self, window_sec: float = 60.0, stable_band: float = 0.2):
        """
        :param window_sec: History length, seconds
        :param stable_band: |slope| below this (%/min) counts as "stable"
        """
        self.window_sec = window_sec
        self.stable_band = stable_band
        self.points: Deque[Tuple[float, float]] = collections.deque()

    def add(self, now: float, concentration: float) -> Tuple[Optional[float], str]:
        """
        :param now: Monotonic time, seconds
        :param concentration: Concentration, %
        :return: (rise rate in %/min or None if not enough history, "rising" | "falling" | "stable")
        """
        self.points.append((now, concentration))
        while self.points and now - self.points[0][0] > self.window_sec:
            self.points.popleft()
        if len(self.points) < 3 or self.points[-1][0] - self.points[0][0] < 1e-6:
            return None, "stable"
        t, c = np.array(self.points).T
        rate = float(np.polyfit(t - t[0], c, 1)[0]) * 60.0
        if rate > self.stable_band:
            return rate, "rising"
        if rate < -self.stable_band:
            return rate, "falling"
        return rate, "stable"


def curve_summary(curve: Sequence[float], voltages: Sequence[float]) -> Dict[str, float]:
    """
    :return: Compact description of a curve for the event payload
    """
    values = np.asarray(curve, dtype=float)
    peak = int(np.argmax(values))
    return {"r_min": round(float(values.min()), 5), "r_max": round(float(values.max()), 5),
            "r_mean": round(float(values.mean()), 5), "peak_ug_v": float(voltages[peak])}


class GasWorker(BaseModule):
    name = "gas"
    source = "gas"

    def __init__(self, event_queue, control_queue, config: Dict[str, Any],
                 predictor=None, reader: Optional[Reader] = None, clock: Callable[[], float] = time.monotonic):
        """
        :param predictor: Object with predict(curve) and voltages; loads GasPredictor from config when None
        :param reader: Curve source; built from config when None
        :param clock: Monotonic time source (injectable for tests)
        """
        super().__init__(event_queue, control_queue, config)
        gas = config["gas"]
        self.gas_config = gas
        self.poll_sec = float(gas.get("gas_poll_sec", 1.0))
        self.sensor_id = gas.get("sensor_id", "gas-1")
        self.thresholds: Dict[str, float] = gas.get("thresholds", {})
        self.rise_rate_threshold = float(gas.get("rise_rate_threshold", 1.0))
        self.alert_cooldown_sec = float(gas.get("alert_cooldown_sec", 30.0))
        self.trend = TrendTracker(float(gas.get("trend_window_sec", 60.0)), float(gas.get("trend_stable_band", 0.2)))
        self.predictor = predictor
        self.reader = reader
        self.clock = clock
        self._next_poll = 0.0
        self._last_alert: Dict[str, float] = {}
        from modules.trust.health import HealthPublisher, registry_config

        health_cfg = registry_config(config)    # F23 (TOD-34): gas self-diagnostics + clean-air self-test
        self.health_cfg = (health_cfg or {}).get("gas") if health_cfg is not None else None
        self.health = None
        self.health_publisher = HealthPublisher(self.emit, "gas", (health_cfg or {}).get("health", {}).get(
            "refresh_sec", 300)) if health_cfg is not None else None
        self._last_selftest = float("-inf")
        self._drift: Optional[Dict[str, Any]] = None

    def setup(self) -> None:
        if self.predictor is None:
            from modules.gas.inference import GasPredictor
            self.predictor = GasPredictor(self.gas_config["classifier_path"], self.gas_config["regressor_path"])
        if self.reader is None:
            self.reader = load_reader(self.gas_config)
        if self.health_cfg is not None and self.health is None:
            from modules.trust.health import GasHealth

            try:
                self.health = GasHealth.from_dataset(self.health_cfg, self.gas_config.get("data_path",
                                                                                          "data/gas/data.xls"))
            except Exception as e:  # self-diagnostics must never stop the sensor
                logger.warning(f"Gas self-diagnostics off: {e}")

    def step(self) -> None:
        now = self.clock()
        if now < self._next_poll:
            time.sleep(min(0.1, self._next_poll - now))
            return
        self._next_poll = now + self.poll_sec
        self.poll_once()

    def poll_once(self) -> Optional[Dict[str, Any]]:
        """
        Reads one curve, predicts and publishes events.

        :return: The gas_reading payload, or None on a sensor/model error
        """
        now = self.clock()
        try:
            curve = list(self.reader())
            prediction = self.predictor.predict(curve)
        except Exception as e:
            logger.error(f"Gas poll failed: {e}", exc_info=True)
            self._alert(now, "sensor_error", {"error": str(e)}, severity="warning")
            if self.health is not None:
                self.health_publisher.publish(self.health.error(), now)
            return None

        gas, concentration = prediction["gas"], prediction["concentration_pct"]
        rate, trend = self.trend.add(now, concentration)
        payload = {
            "gas": gas,
            "gas_confidence": prediction["gas_confidence"],
            "concentration_pct": concentration,
            "trend": trend,
            "rise_rate_pct_per_min": None if rate is None else round(rate, 3),
            "sensor_id": self.sensor_id,
            "curve": curve_summary(curve, self.predictor.voltages),
        }
        self.emit("gas_reading", payload, confidence=prediction["gas_confidence"])
        if self.health is not None:
            self._check_health(curve, gas, now)

        threshold = self.thresholds.get(gas)
        if threshold is not None and concentration >= threshold:
            self._alert(now, "gas_threshold", {"gas": gas, "concentration_pct": concentration,
                                               "threshold_pct": threshold})
        if rate is not None and rate > self.rise_rate_threshold:
            self._alert(now, "gas_rising", {"gas": gas, "concentration_pct": concentration,
                                            "rise_rate_pct_per_min": round(rate, 3),
                                            "rise_rate_threshold": self.rise_rate_threshold})
        return payload

    def _check_health(self, curve, gas: str, now: float) -> None:
        """F23: flat / out-of-range readings every poll; a clean-air self-test when the air is classified clean."""
        failing = self.health.observe(curve, gas)
        extra: Dict[str, Any] = {}
        interval = float(self.health_cfg.get("selftest_interval_sec", 600))
        if gas == "clean_air" and now - self._last_selftest >= interval:
            self._last_selftest = now
            result = self.health.baseline(curve)
            if result is not None:
                self._drift = None if result["ok"] else {"value": result["drift"], "detail": result["detail"]}
                extra = {"baseline_ok": result["ok"], "baseline_detail": result["detail"]}
        if self._drift is not None:
            failing["baseline_drift"] = self._drift
        self.health_publisher.publish(failing, now, **extra)

    def _alert(self, now: float, reason: str, details: Dict[str, Any], severity: str = "critical") -> None:
        last = self._last_alert.get(reason)
        if last is not None and now - last < self.alert_cooldown_sec:
            return
        self._last_alert[reason] = now
        self.emit("alert", {"reason": reason, "sensor_id": self.sensor_id, **details}, severity=severity)


def run(event_queue, control_queue, config):
    """
    Module entry point for the orchestrator.

    :param event_queue: Bus queue to publish events to
    :param control_queue: Queue with control messages ("stop", ...)
    :param config: Full merged config
    """
    GasWorker(event_queue, control_queue, config).run()
