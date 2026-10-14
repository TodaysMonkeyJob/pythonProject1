"""
Random-walk gas reader for development without sensor hardware.
"""
import random
from typing import Any, Dict


class SimulatedGasReader:
    """
    Produces readings that drift around each sensor's "baseline" (default 10 % of its warning threshold).
    """

    def __init__(self, sensors: Dict[str, Any], seed: int = None):
        """
        :param sensors: gas.sensors config section
        :param seed: Random seed for reproducible runs
        """
        self._rng = random.Random(seed)
        self._values = {}
        self._baselines = {}
        for name, limits in sensors.items():
            baseline = limits.get("baseline", 0.1 * limits.get("warning", 100))
            self._baselines[name] = baseline
            self._values[name] = baseline

    def __call__(self) -> Dict[str, float]:
        """
        :return: {sensor_name: value}
        """
        for name, baseline in self._baselines.items():
            drift = self._rng.gauss(0, 0.05 * baseline or 1.0)
            pull = 0.1 * (baseline - self._values[name])
            self._values[name] = max(0.0, self._values[name] + drift + pull)
        return {name: round(value, 2) for name, value in self._values.items()}
