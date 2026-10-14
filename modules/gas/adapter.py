"""
Gas module adapter.

Polls a reader callable that returns ``{sensor_name: value}`` and publishes one ``gas_reading`` event
per poll, plus an ``alert`` event for every sensor above its warning/critical threshold.

The existing gas sensor code plugs in via ``gas.reader`` in config.yaml as ``"package.module:function"``;
``"simulated"`` uses the built-in random-walk reader for development without hardware.
"""
import importlib
import time
from typing import Any, Callable, Dict, Optional

from core.module import BaseModule
from modules.gas.simulated import SimulatedGasReader
from system_setup.logger_setup import init_logger

logger = init_logger()

Reader = Callable[[], Dict[str, float]]


def load_reader(spec: str, sensors: Dict[str, Any]) -> Reader:
    """
    :param spec: "simulated" or "package.module:function"
    :param sensors: gas.sensors config section
    :return: Callable returning {sensor_name: value}
    """
    if spec == "simulated":
        return SimulatedGasReader(sensors)
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"gas.reader must be 'simulated' or 'package.module:function', got {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


def classify(value: float, limits: Dict[str, Any]) -> Optional[str]:
    """
    :param value: Sensor reading
    :param limits: Sensor config with optional "warning" / "critical" thresholds
    :return: "critical", "warning" or None
    """
    critical = limits.get("critical")
    warning = limits.get("warning")
    if critical is not None and value >= critical:
        return "critical"
    if warning is not None and value >= warning:
        return "warning"
    return None


class GasModule(BaseModule):
    name = "gas"
    source = "gas"

    def __init__(self, event_queue, control_queue, config: Dict[str, Any], reader: Optional[Reader] = None):
        """
        :param reader: Overrides gas.reader from config (useful for tests)
        """
        super().__init__(event_queue, control_queue, config)
        self.gas_config = config["gas"]
        self.sensors: Dict[str, Any] = self.gas_config.get("sensors", {})
        self.poll_interval = float(self.gas_config.get("poll_interval", 1.0))
        self.reader = reader
        self._next_poll = 0.0

    def setup(self) -> None:
        if self.reader is None:
            self.reader = load_reader(self.gas_config.get("reader", "simulated"), self.sensors)

    def step(self) -> None:
        now = time.monotonic()
        if now < self._next_poll:
            time.sleep(min(0.1, self._next_poll - now))
            return
        self._next_poll = now + self.poll_interval
        self.poll_once()

    def poll_once(self) -> None:
        """Reads all sensors once and publishes the resulting events."""
        try:
            readings = self.reader()
        except Exception as e:
            logger.error(f"Gas reader failed: {e}", exc_info=True)
            self.emit("alert", {"reason": "sensor_error", "error": str(e)}, severity="warning")
            return

        worst = "info"
        alerts = []
        for sensor, value in readings.items():
            limits = self.sensors.get(sensor, {})
            level = classify(value, limits)
            if level:
                alerts.append((sensor, value, level, limits))
                if level == "critical" or worst == "info":
                    worst = level

        payload = {"readings": readings, "units": {s: self.sensors.get(s, {}).get("unit") for s in readings}}
        self.emit("gas_reading", payload, severity=worst)

        for sensor, value, level, limits in alerts:
            self.emit("alert", {
                "reason": "gas_threshold",
                "sensor": sensor,
                "value": value,
                "unit": limits.get("unit"),
                "threshold": limits.get(level),
                "level": level,
            }, severity=level)
