"""
Vision benchmark for the F1 report: motion FPS, YOLO FPS and peak RAM on the current machine.

    python -m modules.vision.bench                       # camera from config.yaml
    python -m modules.vision.bench --source video.mp4    # video file
    python -m modules.vision.bench --source synthetic    # generated frames (no camera needed)

Results are printed and written to reports/vision_benchmark.json. Run it on the Raspberry Pi.
"""
import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

import cv2
import numpy as np
import psutil

from core.config import load_config
from modules.vision.capture import OpenCVSource, source_factory
from modules.vision.motion import MotionDetector
from modules.vision.yolo import ObjectDetector


def synthetic_frames(width: int, height: int) -> Iterator[np.ndarray]:
    """Noisy background with a moving rectangle."""
    rng = np.random.default_rng(0)
    background = rng.integers(40, 80, (height, width, 3), dtype=np.uint8)
    t = 0
    while True:
        frame = background.copy()
        x = (t * 8) % (width - 120)
        cv2.rectangle(frame, (x, 150), (x + 120, 400), (200, 180, 160), -1)
        frame = cv2.add(frame, rng.integers(0, 6, frame.shape, dtype=np.uint8))
        t += 1
        yield frame


def frame_iterator(source: str, vision_cfg: Dict[str, Any]) -> Iterator[np.ndarray]:
    width, height = vision_cfg.get("resolution", [640, 480])
    if source == "synthetic":
        yield from synthetic_frames(width, height)
        return
    if source == "camera":
        src = source_factory(vision_cfg)()
    else:
        src = OpenCVSource(source, width, height, realtime=False, loop=True)
    src.open()
    try:
        while True:
            frame = src.read()
            if frame is not None:
                yield frame
    finally:
        src.close()


def measure(frames: Iterator[np.ndarray], seconds: float, fn) -> Dict[str, float]:
    count, busy, start = 0, 0.0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        frame = next(frames)
        t = time.perf_counter()
        fn(frame)
        busy += time.perf_counter() - t
        count += 1
    elapsed = time.perf_counter() - start
    return {"frames": count, "fps_end_to_end": round(count / elapsed, 2),
            "ms_per_frame": round(1000 * busy / max(1, count), 2)}


def cpu_temperature() -> Optional[float]:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000
    except OSError:
        return None


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Vision FPS / RAM benchmark")
    parser.add_argument("--source", default="camera", help="camera | synthetic | <video path>")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--model", help="Overrides vision.model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="reports/vision_benchmark.json")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)["vision"]
    if args.model:
        cfg["model"] = args.model
    process = psutil.Process()
    frames = frame_iterator(args.source, cfg)

    motion = MotionDetector(cfg.get("motion_min_area", 800), cfg.get("motion_history", 300),
                            cfg.get("motion_var_threshold", 25), cfg.get("motion_warmup_frames", 30))
    motion_result = measure(frames, args.seconds, motion.detect)

    detector = ObjectDetector(cfg["model"], cfg.get("imgsz", 320), cfg.get("confidence_threshold", 0.4),
                              cfg.get("classes"), cfg.get("tracker", "bytetrack.yaml"))
    detector.load()
    detector.track(next(frames))  # warm-up
    yolo_result = measure(frames, args.seconds, detector.track)

    report = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": f"{platform.node()} {platform.machine()} {platform.system()} {platform.release()}",
        "source": args.source,
        "resolution": cfg.get("resolution"),
        "model": cfg["model"],
        "imgsz": cfg.get("imgsz"),
        "motion": motion_result,
        "yolo_track": yolo_result,
        "peak_rss_mb": round(process.memory_info().rss / 1e6, 1),
        "cpu_temp_c": cpu_temperature(),
        "targets": {"motion_fps": 15, "yolo_fps": 5, "ram_mb": 1500},
    }
    report["meets_targets"] = {
        "motion_fps": motion_result["fps_end_to_end"] >= 15,
        "yolo_fps": yolo_result["fps_end_to_end"] >= 5,
        "ram": report["peak_rss_mb"] <= 1500,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
