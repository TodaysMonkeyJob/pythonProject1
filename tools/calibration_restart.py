"""
Ready time after a restart (C1 / TOD-54): the first start of every sensor kind (full calibration) and a restart one
minute later with the stored calibration (short check / warm-up only), driven through the real calibrators with
synthetic signals and a simulated clock — the time until each device is usable (ready / factory).

    python -m tools.calibration_restart [--out reports/calibration_restart]   → restart.json + README.md

The live system reports the same per device: ``ready_after_sec`` in ``calibration_status`` (dashboard «Калібрування»).
"""
import argparse
import datetime as dt
import json
import os
import sys
from typing import Any, Callable, Dict

import numpy as np

if __package__ in (None, ""):  # allow `python tools/calibration_restart.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sensor_calibration import CalibrationStore  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def run(make: Callable[[CalibrationStore, Clock], Any], feed: Callable[[Any, Clock], None], store: CalibrationStore,
        limit: float = 3600.0) -> Dict[str, Any]:
    """:return: {ready_after_sec, status, full_calibration} of one start (the feed advances the clock)"""
    clock = Clock()
    calibrator = make(store, clock)
    started = calibrator.start()
    full = started == "calibrating"
    while not calibrator.state.usable and clock.t < limit:
        feed(calibrator, clock)
        full = full or calibrator.state.state == "calibrating"
    return {"ready_after_sec": calibrator.state.ready_after, "status": calibrator.state.state,
            "text": calibrator.state.text(), "full_calibration": full}


def scenarios(config: Dict[str, Any]) -> Dict[str, tuple]:
    from modules.audio.intensity import BackgroundLevel
    from modules.calibration import CameraCalibrator, GrapheneCalibrator, MicCalibrator, MosCalibrator
    from modules.calibration.base import Calibrator
    from modules.vision.motion import MotionDetector

    rng = np.random.default_rng(54)
    frame = np.zeros((120, 160, 3), np.uint8)
    frame[:, 80:] = 160
    voltages = [round(0.1 * i, 1) for i in range(16)]
    x = np.arange(16)
    clean = list(1.0 + np.exp(-((x - 6) ** 2) / 8.0))

    def mic_feed(c, clock):                          # 30 ms frames of a quiet room
        clock.t += 0.03
        c.observe(-60.0 + rng.normal(0, 0.5), False, clock.t)

    def camera_feed(c, clock):                       # 10 fps, still scene
        clock.t += 0.1
        image = np.clip(frame + rng.normal(0, 2, frame.shape), 0, 255).astype(np.uint8)
        c.observe(image, c.motion.detect(image).motion, clock.t)

    def graphene_feed(c, clock):                     # one sweep per second, clean air
        clock.t += 1.0
        c.observe(clean, "clean_air")

    class MQ:
        def __init__(self):
            self.options = {"r0_ohm": 15_000.0, "rl_ohm": 10_000.0, "vc": 5.0, "clean_air_ratio": 1.0}

    def mos_feed(c, clock):
        clock.t += 1.0
        c.observe(2.0)

    def factory_feed(c, clock):
        clock.t += 1.0
        c.state.tick()

    def maker(cls, kind, kw):
        return lambda store, clock: cls(config, f"sim-{kind}", f"sim-{kind}", lambda p: None, clock, store,
                                        **({"kind": kind} if cls is Calibrator else {}), **kw(clock))

    return {
        "mic": (maker(MicCalibrator, "mic", lambda clock: {"background": BackgroundLevel()}), mic_feed),
        "camera": (maker(CameraCalibrator, "camera", lambda clock: {"motion": MotionDetector(min_area=50, fps=10)}),
                   camera_feed),
        "graphene_fet": (maker(GrapheneCalibrator, "graphene_fet", lambda clock: {"voltages": voltages}),
                         graphene_feed),
        "mos": (maker(MosCalibrator, "mos", lambda clock: {"driver": MQ()}), mos_feed),
        "electrochemical": (maker(Calibrator, "electrochemical", lambda clock: {}), factory_feed),
        "particle": (maker(Calibrator, "particle", lambda clock: {}), factory_feed),
    }


def measure(config: Dict[str, Any]) -> Dict[str, Any]:
    store = CalibrationStore(None)
    rows = {}
    for kind, (make, feed) in scenarios(config).items():
        first = run(make, feed, store)
        restart = run(make, feed, store)             # one minute later: same uid, the stored record
        rows[kind] = {"first_start": first, "restart": restart}
    return {"rows": rows,
            "total_first_sec": round(max(r["first_start"]["ready_after_sec"] or 0 for r in rows.values()), 2),
            "total_restart_sec": round(max(r["restart"]["ready_after_sec"] or 0 for r in rows.values()), 2)}


def main(argv=None) -> int:
    from core.config import load_config

    parser = argparse.ArgumentParser(description="Ready time of every sensor kind: first start vs restart")
    parser.add_argument("--out", default=os.path.join("reports", "calibration_restart"))
    args = parser.parse_args(argv)
    config = load_config("config.yaml")
    result = {"measured_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "policy": (config.get("sensor_calibration") or {}).get("kinds"), **measure(config)}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "restart.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    lines = ["# Ready time after a restart (C1 / TOD-54)", "",
             "Simulated clock and synthetic signals through the real calibrators "
             "(`python -m tools.calibration_restart`); policy from `config.yaml` `sensor_calibration`. The live "
             "system reports the same per device as `ready_after_sec` (dashboard «Калібрування»).", "",
             "| Sensor | First start: ready after, s | full calibration | Restart after 1 min: ready after, s | "
             "full calibration | restart status |", "| -- | -- | -- | -- | -- | -- |"]
    for kind, r in result["rows"].items():
        f, s = r["first_start"], r["restart"]
        lines.append(f"| {kind} | {f['ready_after_sec']} | {'yes' if f['full_calibration'] else 'no'} | "
                     f"{s['ready_after_sec']} | {'yes' if s['full_calibration'] else 'no'} | {s['text']} |")
    lines += ["", f"All devices ready: **{result['total_first_sec']} s** on the first start, "
                  f"**{result['total_restart_sec']} s** after a restart (devices start in parallel: the slowest one "
                  "decides). Warm-up times (MOS heater, electrochemical stabilisation, particle fan) are physical "
                  "and stay; during them the stored calibration is used with a wider uncertainty."]
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
