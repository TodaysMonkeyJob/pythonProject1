"""Local check of the vision module: prints events, shows the annotated frame from shared memory.

Run from the repo root:  python -m modules.vision.demo [camera index | video path] [--no-window] [--seconds N]
Press q in the window to stop.
"""
import argparse
import os
import queue
import threading
import time

import cv2
import numpy as np

from core.bus import CONTROL_STOP
from core.config import load_config
from core.events import Event
from modules.vision.detector import VisionDetector
from modules.vision.shared_frame import SharedFrameBuffer
from system_setup.logger_setup import init_logger, set_logger_params

parser = argparse.ArgumentParser()
parser.add_argument("device", nargs="?", default="0")
parser.add_argument("--model", default="yolo11n.pt")
parser.add_argument("--pose-model", help="also run action recognition (F2), e.g. yolo11n-pose.pt")
parser.add_argument("--no-identity", action="store_true",
                    help="don't run face identification (F3); by default it runs when identity.enabled and the "
                         "face models exist, saving unknown people to the identity registry automatically")
parser.add_argument("--no-window", action="store_true")
parser.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = until q / Ctrl+C)")
args = parser.parse_args()

set_logger_params(init_logger(), log_level="INFO", no_log_file=True)
config = load_config()
vision = config["vision"]
vision.update(camera="opencv", device=args.device, model=args.model, stats_interval_sec=10,
              shared_buffer="iot_vision_demo", log_events=True)
width, height = vision["resolution"]

events, control = queue.Queue(), queue.Queue()
detector = VisionDetector(events, control, config)
thread = threading.Thread(target=detector.run, daemon=True)
thread.start()

actions_control = queue.Queue()
if args.pose_model:
    from modules.vision.actions import ActionRecognizer

    config["actions"]["pose_model"] = args.pose_model
    threading.Thread(target=ActionRecognizer(events, actions_control, config).run, daemon=True).start()

identity_control = queue.Queue()
identity_cfg = config.get("identity", {})
face_models = [identity_cfg.get("detector_model", ""), identity_cfg.get("recognizer_model", "")]
models_ready = all(os.path.isfile(m) for m in face_models)
run_identity = not args.no_identity and identity_cfg.get("enabled", True) and models_ready
if not args.no_identity and not models_ready:
    print("Face identification off: run `python -m modules.vision.face_models` to download the models")
if run_identity:
    from modules.vision.identity import IdentityRecognizer

    print(f"Face identification on: unknown people are saved to {identity_cfg.get('registry_path')} "
          f"(name them: python -m tools.identity list / rename unknown_N <name>)")

    threading.Thread(target=IdentityRecognizer(events, identity_control, config).run, daemon=True).start()

buffer, started = None, time.monotonic()
try:
    while thread.is_alive():
        while not events.empty():
            raw_event = events.get()
            event = Event.from_json(raw_event)
            if event.type == "object_detected":
                actions_control.put(raw_event)  # what the orchestrator will do: forward to F2 / F3
                identity_control.put(raw_event)
            if event.type != "heartbeat":
                print(f"{event.timestamp}  {event.type:16} {event.payload}")
        if buffer is None:
            try:
                buffer = SharedFrameBuffer.attach("iot_vision_demo", width, height)
            except FileNotFoundError:
                pass
        if not args.no_window and buffer is not None:
            frame = buffer.read_annotated()
            if frame:
                cv2.imshow("vision (q to quit)", cv2.imdecode(np.frombuffer(frame[2], np.uint8), cv2.IMREAD_COLOR))
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
        else:
            time.sleep(0.05)
        if args.seconds and time.monotonic() - started > args.seconds:
            break
except KeyboardInterrupt:
    pass
finally:
    control.put(CONTROL_STOP)
    actions_control.put(CONTROL_STOP)
    identity_control.put(CONTROL_STOP)
    thread.join(timeout=5)
    if buffer is not None:
        buffer.unlink()
        buffer.close()
    cv2.destroyAllWindows()
