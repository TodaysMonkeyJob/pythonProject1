"""
Worker side of the persistent sensor calibration (C1 / TOD-54, ``core/sensor_calibration.py``): what a short check
and a full calibration measure for each sensor kind. Every calibrator is driven by its worker (``observe`` per frame
/ reading) and by ``calibration_control`` events (``control``)::

    MicCalibrator      background level (dB) + noise power spectrum of one microphone
    CameraCalibrator   motion background image + per-cell noise levels of one camera
    GrapheneCalibrator clean-air baseline curve R(V_G), Dirac point, normalisation peak of the graphene board
    (factory-calibrated gas modules and MOS R₀: modules/gas/drivers_mode.py)
"""
from modules.calibration.base import Calibrator, control_targets, enabled  # noqa: F401
from modules.calibration.camera import CameraCalibrator  # noqa: F401
from modules.calibration.gas import GrapheneCalibrator, MosCalibrator  # noqa: F401
from modules.calibration.mic import MicCalibrator  # noqa: F401
