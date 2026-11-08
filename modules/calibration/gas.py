"""
Graphene FET calibration (C1 / TOD-54): the clean-air baseline curve R(V_G), its Dirac point (gate voltage of the
peak) and the normalisation peak R_peak of this board.

* stored → used at once; short check: the first curve the classifier calls ``clean_air`` within ``tolerance``
  (25 %) of the stored R_peak and ``dirac_steps`` (2) gate steps of the stored Dirac point;
* full calibration: ``calibrate_samples`` (30) clean-air curves (the user provides clean air; other curves are
  waited out) → their mean is the new baseline.
"""
from typing import List, Optional, Sequence

import numpy as np

from modules.calibration.base import Calibrator


def describe(curve: Sequence[float], voltages: Sequence[float]):
    x = np.asarray(curve, float)
    index = int(np.argmax(x))
    return float(x.max()), index, (float(voltages[index]) if index < len(voltages) else float(index))


class GrapheneCalibrator(Calibrator):
    kind = "graphene_fet"

    def __init__(self, *args, voltages: Optional[Sequence[float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.voltages = list(voltages or [])
        self._curves: List[np.ndarray] = []

    def observe(self, curve: Sequence[float], gas: Optional[str]) -> None:
        self.state.tick()
        clean = gas == "clean_air"
        if self.state.state == "checking" and clean:
            values = self.state.values
            peak, index, dirac = describe(curve, self.voltages)
            ref_peak, ref_index = float(values["peak"]), int(values["dirac_index"])
            drift = abs(peak - ref_peak) / max(abs(ref_peak), 1e-9)
            tolerance, steps = float(self.policy.get("tolerance", 0.25)), int(self.policy.get("dirac_steps", 2))
            self.state.check_result(drift <= tolerance and abs(index - ref_index) <= steps,
                                    f"чисте повітря: R_peak {drift:.0%} від збереженого еталону (допуск "
                                    f"{tolerance:.0%}), точка Дірака {dirac:g} В (збережено {values['dirac_v']:g} В)",
                                    drift=round(drift, 3), dirac_v=dirac)
        elif self.state.state == "calibrating" and clean:
            self._curves.append(np.asarray(curve, float))
            needed = int(self.policy.get("calibrate_samples", 30))
            self.state.set_progress(len(self._curves) / needed)
            if len(self._curves) >= needed:
                mean = np.mean(self._curves, axis=0)
                peak, index, dirac = describe(mean, self.voltages)
                self.state.complete({"baseline_curve": np.round(mean, 6).tolist(), "peak": round(peak, 6),
                                     "dirac_index": index, "dirac_v": dirac},
                                    f"середнє {needed} кривих R(V_G), класифікованих як чисте повітря",
                                    validation={"curves": needed, "spread": round(float(np.std(self._curves,
                                                                                               axis=0).max()), 6)})
                self._curves = []

    def control(self, payload) -> bool:
        done = super().control(payload)
        if done:
            self._curves = []
        return done


class MosCalibrator(Calibrator):
    """
    MOS (MQ) sensor: R₀ = R_s in clean air ÷ the datasheet's clean-air ratio (``clean_air_ratio`` in the sensor's
    declaration, 1 when absent). Stored R₀ replaces the declared one; short check after the heater warm-up: R_s / R₀
    within ``tolerance`` (50 %) of the clean-air ratio; full calibration: ``calibrate_samples`` (60) readings in clean
    air → the mean R_s.
    """

    kind = "mos"

    def __init__(self, *args, driver=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver = driver
        self._rs: List[float] = []

    def ratio(self) -> float:
        return float((getattr(self.driver, "options", None) or {}).get("clean_air_ratio", 1.0))

    def apply_stored(self) -> None:
        r0 = self.state.values.get("r0_ohm")
        if r0 and self.driver is not None:
            self.driver.options["r0_ohm"] = float(r0)

    def observe(self, volts: Optional[float]) -> None:
        self.state.tick()
        if volts is None or volts <= 0 or self.driver is None:
            return
        options = self.driver.options
        rs = float(options.get("rl_ohm", 10_000.0)) * (float(options.get("vc", 5.0)) - volts) / volts
        if self.state.state == "checking":
            r0, expected = float(options["r0_ohm"]), self.ratio()
            deviation = abs(rs / r0 - expected) / max(expected, 1e-9)
            tolerance = float(self.policy.get("tolerance", 0.5))
            self.state.check_result(deviation <= tolerance, f"R_s/R₀ {rs / r0:.2f}, у чистому повітрі {expected:g} "
                                                            f"(допуск {tolerance:.0%})", rs_ohm=round(rs, 1))
        elif self.state.state == "calibrating":
            self._rs.append(rs)
            needed = int(self.policy.get("calibrate_samples", 60))
            self.state.set_progress(len(self._rs) / needed)
            if len(self._rs) >= needed:
                r0 = float(np.mean(self._rs)) / self.ratio()
                self.state.complete({"r0_ohm": round(r0, 1)}, f"середнє R_s за {needed} показів у чистому повітрі ÷ "
                                                              f"{self.ratio():g}", validation={"samples": needed})
                options["r0_ohm"] = round(r0, 1)
                self._rs = []

    def control(self, payload) -> bool:
        done = super().control(payload)
        if done:
            self._rs = []
        return done
