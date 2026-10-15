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
