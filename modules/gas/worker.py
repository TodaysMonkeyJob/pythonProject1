"""
Gas worker: reads an R(Ug) curve every ``gas_poll_sec``, classifies it and publishes events.

Curve sources (``gas.mode``):
  * ``replay`` — rows of ``gas.replay_csv`` (see modules.gas.data.write_replay_csv), looped if ``replay_loop``;
  * ``sensor`` — ``gas.reader: "package.module:function"`` returning 16 resistance values for Ug = 0.0..1.5 V;
  * ``drivers`` (A7 / TOD-48) — every sensor on this Pi found automatically and read in its native unit
    (``modules/gas/drivers_mode.py``, ``drivers/gas``).

Events:
  * ``gas_reading`` (info) every poll: gas, gas_confidence, concentration_pct (the model's prediction, vol %), trend,
    sensor_id, curve summary, ``ratio`` = c / C_crit and ``pct_of_crit`` («% від критичного рівня»). G1 (TOD-53): the
    model was trained from ``MODEL_RANGE_VOL_PCT`` up (0.25–0.4 vol %, ≥ C_crit of every gas): a gas it reports
    counts as at least that floor (``detection_floor``, ``below_model_range`` when the prediction is lower) and the
    reading carries ``model_note``, which is also logged once per gas;
  * ``alert`` — G1 (TOD-53) on the scale of ``config/gases.yaml`` (``gas.gases``), never on percentages:
    ``gas_threshold`` (critical) when the reading is ≥ the gas's C_crit; ``gas_explosive`` (critical,
    «вибухонебезпечно») when it is ≥ the gas's lower explosive limit; ``gas_rising`` (warning) when c / C_crit rises
    faster than ``rise_alert_norm_per_min`` over ``trend_window_sec`` (how much a rise means is the HFS's
    ``dynamics`` input). Each reason is repeated at most once per ``alert_cooldown_sec``. The trend label uses
    ``trend_stable_norm`` (C_crit per minute); ``rise_rate_pct_per_min`` stays in vol % per minute.
"""
import collections
import importlib
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.module import BaseModule
from modules.gas.data import MODEL_RANGE_VOL_PCT, read_replay_csv
from modules.hfs.gas_normalize import GasTable, pct_of_crit
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


def model_note(substance, floor_pct: float) -> str:
    """G1 (TOD-53): why a graphene reading means "at or above the critical level" (Ukrainian, logs + payload)"""
    share = pct_of_crit(substance.ratio(floor_pct, "vol_pct"))
    return (f"модель графенового сенсора навчена для «{substance.name_uk}» від {floor_pct} об.% ({share:g} % від "
            "критичного рівня): нижчих концентрацій вона не розрізняє, тому будь-який ненульовий показ — не нижче "
            "критичного рівня; рівні нижче критичного вимірюють лише низькодіапазонні сенсори")


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
        self.table = GasTable.load(gas.get("gases", "config/gases.yaml"))     # G1 (TOD-53): C_crit + LEL
        self.rise_alert_norm = float(gas.get("rise_alert_norm_per_min", 0.25))
        self.alert_cooldown_sec = float(gas.get("alert_cooldown_sec", 30.0))
        window = float(gas.get("trend_window_sec", 60.0))
        self.trend = TrendTracker(window, 0.0)                                   # vol % / min (payload only)
        self.trend_norm = TrendTracker(window, float(gas.get("trend_stable_norm", 0.05)))   # C_crit / min
        self.predictor = predictor
        self.reader = reader
        self.clock = clock
        self._next_poll = 0.0
        self._last_alert: Dict[str, float] = {}
        self._noted: set = set()
        from modules.trust.health import HealthPublisher, registry_config

        health_cfg = registry_config(config)    # F23 (TOD-34): gas self-diagnostics + clean-air self-test
        self.health_cfg = (health_cfg or {}).get("gas") if health_cfg is not None else None
        self.health = None
        self.health_publisher = HealthPublisher(self.emit, "gas", (health_cfg or {}).get("health", {}).get(
            "refresh_sec", 300)) if health_cfg is not None else None
        self._last_selftest = float("-inf")
        self._drift: Optional[Dict[str, Any]] = None
        self.poller = None                      # TOD-48 gas.mode drivers
        self.calibration = None                 # C1 (TOD-54): the graphene board's stored clean-air baseline

    def setup(self) -> None:
        if self.gas_config.get("mode") == "drivers":
            from modules.gas.drivers_mode import DriverPoller

            self.poller = self.poller or DriverPoller(self.gas_config, self.emit, self.clock, config=self.config)
            self.poller.discover()
            return
        if self.predictor is None:
            from modules.gas.inference import GasPredictor
            self.predictor = GasPredictor(self.gas_config["classifier_path"], self.gas_config["regressor_path"])
        if self.reader is None:
            self.reader = load_reader(self.gas_config)
        from modules.calibration import GrapheneCalibrator, enabled

        if enabled(self.config) and self.calibration is None:
            from core.sensor_calibration import device_uid

            uid = device_uid("graphene_fet", self.sensor_id, str(self.gas_config.get("mode", "replay")))
            self.calibration = GrapheneCalibrator(self.config, uid, self.device_id,
                                                  lambda p: self.emit("calibration_status", p), self.clock,
                                                  voltages=self.predictor.voltages)
            self.calibration.start()
        if self.health_cfg is not None and self.health is None:
            from modules.trust.health import GasHealth

            try:
                self.health = GasHealth.from_dataset(self.health_cfg, self.gas_config.get("data_path",
                                                                                          "data/gas/data.xls"))
            except Exception as e:  # self-diagnostics must never stop the sensor
                logger.warning(f"Gas self-diagnostics off: {e}")

    def handle_control(self, message: Any) -> None:
        if isinstance(message, str) and message.startswith("{"):
            from core.events import Event

            event = Event.from_json(message)
            if self.poller is not None:
                self.poller.handle_event(event)
            elif event.type == "calibration_control" and self.calibration is not None:
                self.calibration.control(event.payload)
            return
        super().handle_control(message)

    def teardown(self) -> None:
        if self.poller is not None:
            self.poller.close()

    def step(self) -> None:
        if self.poller is not None:
            self.poller.step()
            time.sleep(0.05)
            return
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
        if self.calibration is not None:
            self.calibration.observe(curve, gas)
        substance = self.table.get(gas)
        floor = (MODEL_RANGE_VOL_PCT.get(gas) or (None,))[0] if substance is not None else None
        effective = max(concentration, floor) if floor is not None and concentration > 0 else concentration
        ratio = substance.ratio(effective, "vol_pct") if substance is not None else 0.0
        rate, _ = self.trend.add(now, concentration)
        rate_norm, trend = self.trend_norm.add(now, ratio)
        payload = {
            "gas": gas,
            "gas_confidence": prediction["gas_confidence"],
            "concentration_pct": concentration,
            "ratio": round(ratio, 4),
            "pct_of_crit": pct_of_crit(ratio),
            "trend": trend,
            "rise_rate_pct_per_min": None if rate is None else round(rate, 3),
            "sensor_id": self.sensor_id,
            "curve": curve_summary(curve, self.predictor.voltages),
        }
        if floor is not None:
            payload.update(detection_floor=floor, below_model_range=0 < concentration < floor,
                           model_note=model_note(substance, floor))
            if gas not in self._noted:
                self._noted.add(gas)
                logger.warning(f"gas {gas}: {payload['model_note']}")
        self.emit("gas_reading", payload, confidence=prediction["gas_confidence"])
        if self.health is not None:
            self._check_health(curve, gas, now)

        if substance is not None:
            limit = {"gas": gas, "concentration_pct": concentration, "ratio": round(ratio, 4),
                     "pct_of_crit": pct_of_crit(ratio), "c_crit_pct": round(substance.c_crit_in("vol_pct"), 6)}
            if ratio >= 1.0:
                self._alert(now, "gas_threshold", limit)
            if substance.lel_pct is not None and concentration >= substance.lel_pct:
                self._alert(now, "gas_explosive", {**limit, "lel_pct": substance.lel_pct,
                                                   "text": f"вибухонебезпечно: {substance.name_uk} {concentration} "
                                                           f"об.% ≥ НКМПР {substance.lel_pct} об.%"})
            if rate_norm is not None and rate_norm > self.rise_alert_norm:
                self._alert(now, "gas_rising", {**limit, "rise_rate_pct_per_min": round(rate or 0.0, 3),
                                                "rise_norm_per_min": round(rate_norm, 3),
                                                "rise_alert_norm_per_min": self.rise_alert_norm},
                            severity="warning")
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
