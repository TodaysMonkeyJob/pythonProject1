"""Common part of the per-sensor calibrators (C1 / TOD-54)."""
from typing import Any, Callable, Dict, List, Optional

from core.sensor_calibration import CalibrationStore, DeviceCalibration, policy_of


def control_targets(payload: Dict[str, Any]) -> List[str]:
    """:return: the device ids a ``calibration_control`` event is for"""
    ids = payload.get("device_ids") or ([payload["device_id"]] if payload.get("device_id") else [])
    return [str(i) for i in ids]


class Calibrator:
    """One device's calibration inside its worker: the policy / files of ``DeviceCalibration`` + a measurement."""

    kind = ""

    def __init__(self, config: Dict[str, Any], uid: str, device_id: Optional[str],
                 emit: Callable[[Dict[str, Any]], Any], clock: Callable[[], float],
                 store: Optional[CalibrationStore] = None, kind: Optional[str] = None):
        self.kind = kind or self.kind
        self.policy = policy_of(config, self.kind)
        self.state = DeviceCalibration(store or CalibrationStore.from_config(config), uid, self.kind, device_id,
                                       self.policy, emit, clock)
        self.clock = clock
        self._samples: List[Any] = []
        self._calibration_started: Optional[float] = None

    def start(self) -> str:
        state = self.state.start()
        self.apply_stored()
        return state

    def apply_stored(self) -> None:
        """Puts the stored values into the live model (worker-specific)."""

    def control(self, payload: Dict[str, Any]) -> bool:
        """:return: the event concerned this device (recalibrate / cancel / reload)"""
        if self.state.device_id not in control_targets(payload):
            return False
        action = payload.get("action")
        if action == "recalibrate":
            self.state.request(str(payload.get("by") or "dashboard"))
            self._samples, self._calibration_started = [], None
        elif action == "cancel":
            self.state.cancel()
            self._samples, self._calibration_started = [], None
        elif action == "reload":
            self.state.reload()
            self.apply_stored()
            self._samples = []
        return True

    def status(self) -> Dict[str, Any]:
        return self.state.status()

    @property
    def calibrating(self) -> bool:
        return self.state.state == "calibrating"


def enabled(config: Dict[str, Any]) -> bool:
    """Persistent calibration is on when config.yaml has a ``sensor_calibration`` section (tests / replays: off)."""
    cfg = config.get("sensor_calibration")
    return bool(cfg) and cfg.get("enabled", True) is not False
