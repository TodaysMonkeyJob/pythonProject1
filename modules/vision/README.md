# Vision module (F1 / TOD-12)

Camera → motion detection on every frame (MOG2) → YOLO + ByteTrack only on motion or every
`yolo_interval_sec` → change events on the bus + latest frames in shared memory.

| File | Purpose |
|---|---|
| `detector.py` | `VisionDetector` (`run(event_queue, control_queue, config)`), event logic, FPS stats |
| `capture.py` | `Picamera2Source`, `OpenCVSource` (camera index or video file), `ReconnectingSource` |
| `motion.py` | `MotionDetector`: MOG2 + morphology + `motion_min_area` filter |
| `state.py` | `TrackRegistry` (appeared/disappeared, id-switch merge), `MotionState`, `SceneState` |
| `yolo.py` | `ObjectDetector`: ultralytics `model.track(persist=True, tracker="bytetrack.yaml")` |
| `shared_frame.py` | `SharedFrameBuffer`: annotated JPEG + raw frame + person boxes for F2/F3/F7 |
| `export.py` | `python -m modules.vision.export` → `models/yolo11n_ncnn_model` |
| `pose.py` | `PoseEstimator` (yolo11n-pose, 17 COCO keypoints) + skeleton geometry (joint/torso angles) |
| `actions.py` | F2: `ActionEngine` rules + `ActionRecognizer` module (reads the shared buffer) |
| `eval_actions.py` | F2 acceptance: labelled videos → `reports/actions_eval.json` |
| `bench.py` | `python -m modules.vision.bench` → `reports/vision_benchmark.json` |

## Events (source `camera`, only on changes)

| type | payload |
|---|---|
| `motion` | `{"state": "start" \| "end", "area_ratio", "boxes"}` |
| `object_detected` | `{"track_id", "class", "bbox", "state": "appeared" \| "disappeared"}`, `confidence` |
| `scene_active` | `{"active": true \| false}` — false `scene_timeout_sec` after motion and objects are gone |
| `alert` | `{"reason": "camera_lost" \| "camera_restored"}` |

An object that stays in frame produces one `appeared`: the track is kept while YOLO re-checks it every
`yolo_interval_sec` (must be < `track_lost_sec`), and a new ByteTrack id on top of an unmatched known track
of the same class (IoU ≥ `reid_iou`) is merged into it.

## Shared memory for F2 / F3 / F7

```python
from modules.vision.shared_frame import SharedFrameBuffer
buf = SharedFrameBuffer.attach("iot_vision", 640, 480)
frame_id, ts, jpeg = buf.read_annotated()            # dashboard
frame_id, ts, frame, meta = buf.read_raw()           # meta["persons"] = [{"track_id", "bbox", "confidence"}]
```

The block outlives the vision process (readers stay valid across restarts); the orchestrator calls
`unlink()` on shutdown.

## Raspberry Pi setup (Pi OS 64-bit Bookworm)

```bash
sudo apt update
sudo apt install -y python3-picamera2 libcap-dev     # picamera2 only comes from apt
python3 -m venv --system-site-packages .venv         # so the venv sees picamera2
source .venv/bin/activate
pip install -r requirements.txt
python -m modules.vision.export --model yolo11n.pt --imgsz 320   # → models/yolo11n_ncnn_model
python -m modules.vision.bench --seconds 60                      # camera from config.yaml
```

USB camera instead of the Pi camera: set `vision.camera: opencv` and `vision.device: 0`.
Replay a recording: `vision.device: path/to/video.mp4`.

## Benchmark report

Targets: motion ≥ 15 FPS, YOLO ≥ 5 FPS, RAM ≤ 1.5 GB. Run `bench.py` on the Pi and record the result here
(and commit `reports/vision_benchmark.json`).

| Machine | Model | imgsz | Motion FPS | YOLO+track FPS | Peak RSS | CPU °C |
|---|---|---|---|---|---|---|
| MacBook (arm64, dev reference, synthetic frames) | yolo11n.pt | 320 | 451 | 64 | 449 MB | — |
| MacBook (arm64, dev reference, synthetic frames) | yolo11n NCNN | 320 | — | 102 | 393 MB | — |
| Raspberry Pi — *to be measured* | yolo11n NCNN | 320 | | | | |

## Action recognition (F2 / TOD-13)

`ActionRecognizer` runs in its own process, reads the raw frame + `persons` from the shared buffer and runs
the pose model only when a person is present (at most `actions.action_fps`). Poses are matched to track ids by
IoU; per track ~2.5 s of keypoints are kept.

| Action | Per-frame rule (thresholds in `config.yaml` → `actions`) |
|---|---|
| `standing` | torso ≤ 25° from vertical, knees > 160° (or legs not visible) |
| `sitting` | knee 40–130° and thigh ≤ 45° from horizontal; or box < 75 % of this track's standing height with bent / hidden legs; or upright with the legs hidden by something inside the frame (desk) and not moving |
| `walking` | least-squares box centre speed ≥ 0.3 body heights/s, or box height changing ≥ 25 %/s (towards / away from the camera), or ≥ half the speed with alternating ankles |
| `lying` | torso ≥ 60° from vertical (event severity `warning`, possible fall) |
| `possible_smoking` | a wrist reaches the nose (≤ 0.35 torso lengths) and leaves again (> 0.6) within 5 s, ≥ 3 times in 30 s; `gas_confirmed` if the gas module reported a rise in the last 60 s |

Legs cut by the bottom edge of the frame (close-up) give `unknown`, not `standing`. Resting the head on the
hands (long contact) and keypoint drop-outs do not count as puffs.

An action is confirmed when it wins ≥ 70 % of the known frames in the window; an `action` event is published
only when the confirmed action of a track changes:

```json
{"track_id": 7, "action": "sitting", "previous": "walking", "duration_sec": 0, "previous_duration_sec": 12.4}
{"track_id": 7, "action": "possible_smoking", "previous": "sitting", "duration_sec": 0,
 "gas_confirmed": true, "wrist_to_mouth_count": 4}
```

`possible_smoking` is re-sent once when gas confirmation arrives later (severity `warning` → `critical`).
The orchestrator must forward gas events and `object_detected` events to this module's control queue
(as event JSON strings); `disappeared` drops the track's history.

```bash
python -m modules.vision.export --model yolo11n-pose.pt --imgsz 320   # → models/yolo11n-pose_ncnn_model
python -m modules.vision.demo 0 --pose-model yolo11n-pose.pt          # live check with the webcam
python -m modules.vision.eval_actions data/actions/manifest.csv       # acceptance on labelled videos
```

### Results on `data/human_behaviour_video` (internet clips, 19 posture segments + 7 hand-to-mouth shots)

| Rules / pose input size | Posture segments ≥ 80 % | Mean score | Coverage | Hand-to-mouth shots found | Counted mouth episodes / min (no smoking) | False `possible_smoking` |
|---|---|---|---|---|---|---|
| first version, 320 | 7 / 19 | 0.39 | 0.71 | 1 / 7 | — | — |
| tuned rules, 320 | 10 / 19 | 0.66 | 0.68 | 1 / 7 | 1.6 | 1 in 3.8 min |
| tuned rules, 640 | 14 / 19 | 0.80 | 0.76 | 1 / 7 | 2.1 | 1 in 3.8 min |

Reports: `reports/actions_eval_human_behaviour_*.json`. What the clips showed and what changed:

* People at desks (legs hidden) were "standing" → legs hidden by furniture now mean sitting, legs cut by the
  frame edge mean unknown. Sitting on the floor needed knee angles down to 40°.
* Handheld / zooming cameras faked walking → speed and scale are least-squares slopes over 1 s.
* Sleeping with the head on the hands and keypoint drop-outs created hand-to-mouth episodes → hysteresis on
  the wrist that reached the mouth, episodes longer than 5 s ignored.
* Film close-ups: the pose model rarely finds the wrist next to the face (5 of 7 smoking shots), so smoking
  can't be validated on that montage; it needs footage from the real camera position.
* Remaining failures are footage artefacts: ByteTrack keeps a track id across a scene cut (the walker inherits
  "sitting"), people standing behind a table read as sitting, tiny people in portrait video at 320 px.
* A 640 px pose input fixes the small-people cases (+4 segments) at ~4× the pose cost — measure on the Pi
  before switching `actions.imgsz`.

Acceptance set: 10 videos, 2 per action (standing, sitting, walking, lying, possible_smoking), manifest
`video,expected`. Record the result here:

| Date | Machine | Videos passed | Mean score |
|---|---|---|---|
| *to be measured* | | | |
