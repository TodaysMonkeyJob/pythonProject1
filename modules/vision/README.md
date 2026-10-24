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
| `eval_actions.py` | F2 acceptance: labelled videos → `reports/actions_eval.json` (`--classifier` / `--cigarette`: + learned layer) |
| `action_features.py`, `action_classifier.py` | learned layer: pose-window features, the skeleton action classifier, the cigarette detector |
| `identity.py` | F3: `FaceEngine` (YuNet + SFace), `IdentityEngine` / `IdentityRecognizer` on the F9 registry (`core/identity.py`) |
| `legacy_faces.py` | One-time import of the pre-registry `faces.npz` + `data/faces/unknown/` (`tools/migrate_faces_npz.py`) |
| `face_models.py` | `python -m modules.vision.face_models` → YuNet / SFace ONNX into `models/` (SHA-256 checked) |
| `consumer.py` | `SharedFrameConsumer`: base for F2 / F3 processes reading the shared buffer |
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
| `possible_smoking` | a wrist reaches the nose (≤ 0.35 torso lengths) and leaves again (> 0.6) within 5 s, ≥ 3 times in 30 s (gas readings never confirm it: the sensor measures solvent / ammonia gases, not smoke) |

Legs cut by the bottom edge of the frame (close-up) give `unknown`, not `standing`. Resting the head on the
hands (long contact) and keypoint drop-outs do not count as puffs.

An action is confirmed when it wins ≥ 70 % of the known frames in the window; an `action` event is published
only when the confirmed action of a track changes:

```json
{"track_id": 7, "action": "sitting", "previous": "walking", "duration_sec": 0, "previous_duration_sec": 12.4}
{"track_id": 7, "action": "possible_smoking", "previous": "sitting", "duration_sec": 0,
 "wrist_to_mouth_count": 4, "evidence": ["hand_to_mouth"], "cigarette_seen": false}
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

## Person identification (F3 / TOD-14)

`IdentityRecognizer` (own process, `identity.identity_fps` = 2) reads the shared frame + `persons`, detects all
faces in the frame once (YuNet) and shares them out one-to-one over the person boxes (`assign_faces`: the face
must be centred in the top part of the box; best-fitting pairs first), so with several people — or overlapping /
huge boxes — nobody gets someone else's face. Each assigned face's SFace embedding is compared with `faces.npz`.

* known (similarity ≥ `face_threshold` 0.45) → identified once, no more work for that track;
* otherwise `unknown_N` — embeddings of unknown faces are cached in memory for the session, so the same
  stranger keeps the same label on a new track; an unknown track can later be upgraded to a known name;
* no face (turned away, too small, low score) → `not_visible`, retried every `retry_sec` while the track lives;
* identified tracks are re-checked every `verify_sec` (5 s; unknown ones every `retry_sec`): F1 can keep the
  same `track_id` when another person takes the same place, so after `switch_confirmations` (2) checks that
  see a different face the label changes (`previous` = old label). A check that sees the same person is cheap
  and stores nothing new, except more samples for an unknown person (the best `max_embeddings` are kept);
* events are written to `logs/YYYY-MM-DD/identity.jsonl` (F2: `actions.jsonl`).

```json
{"track_id": 7, "person_id": "3f2a…", "identity": "Oleh", "known": true, "similarity": 0.71, "previous": "not_visible"}
```

Severity is `warning` for `unknown_N`, `info` otherwise. The orchestrator forwards `object_detected`
events to the control queue (`disappeared` drops the track) and the registry's `identity_*` events.

```bash
python -m modules.vision.face_models                                # download the two ONNX models
python -m tools.enroll --name Oleh --faces data/faces/Oleh/         # or --camera 0 --frames 10; --voice a.wav
python -m tools.identity list                                       # rename / merge / delete / history
python -m modules.vision.demo 0                                     # live check (identity on by default)
python -m tools.eval_faces data/faces --people 5 --enroll 5         # accuracy / false accepts
```

### One identity registry for faces and voices (F9 / TOD-20)

Everyone the system knows — enrolled people and strangers — is a person in `data/identity.db` (`core/identity.py`):
`person_id` (uuid, stable key in every event) + `display_name` (`unknown_N` from one global counter shared by faces
and voices, or a name). Face (SFace, 128-d) and voice (ECAPA, 192-d) embeddings are attached to the person; the best
`max_embeddings` per modality are kept and matched by their centroid. Every create / rename / merge / link / delete is
recorded in `identity_events` (audit trail) and published as an `identity_*` event.

Naming someone happens in one place — the dashboard **People** page (click the name → type → Enter), or
`python -m tools.identity rename unknown_3 "Oleh"`. Every module follows at once: the identity process re-sends
`person_identified` (`"renamed_from": "unknown_3"`) for live tracks, the camera overlay shows the new name, the
aggregator resolves names at snapshot time, the browser re-renders old log and transcript lines (they carry the
`person_id`). CLI edits are not on the bus: modules notice them within `sync_sec`.

* **Merge** (`POST /api/persons/{id}/merge`, `tools.identity merge`) — the same person was created twice (e.g. once by
  face, once by voice): all embeddings and face crops move to the target, old ids keep resolving via `merged_into`.
* **Voice ↔ face auto-link** (`core/cooccurrence.py`) — a new voice heard `link_min_cooccurrences` (3) times while
  exactly one identified person is in view (≥ 80 % of each segment) and never with somebody else is merged into that
  face identity; from then on transcripts show the name even off camera. Two people in view → no link. Known
  identities are never reassigned automatically.
* **Delete** — embeddings and face crops removed (privacy / GDPR request).

The old `faces.npz` and `data/faces/unknown/unknown_N/` folders are imported automatically into an empty registry
(numbers kept) or with `python -m tools.migrate_faces_npz`; they are not deleted. Renaming folders no longer names
people — use the People page or `tools.identity`.

**Privacy.** `data/identity.db` and `data/faces/` (face crops in `data/faces/persons/<person_id>/`, legacy files)
are git-ignored and stay on the device; only person_id + name reach the bus (and the name, in the text digest,
Azure). Unknown people did not consent: they are deleted after `unknown_retention_days` (30) without a sighting, and
`store_faces: false` turns storing face crops off. Enroll only people
who gave written consent (thesis ethics section).

### No duplicate unknowns (F14 / TOD-25)

Partial faces (cut off, side view, blurred, small, a hat) give embeddings that match nobody, which used to create a
new `unknown_N` for people already registered. Four deterministic layers on top of YuNet + SFace:

1. **Quality gate** (`face_quality.py`, `identity.quality.*`) before any embedding: YuNet score ≥ 0.85, face ≥ 80 px
   and eye distance ≥ 30 px, face box + 5 landmarks inside the frame (5 % margin) and the landmarks inside the person
   box, yaw ±35° / pitch ±25° estimated from the landmarks, Laplacian variance of the aligned crop, brightness
   40–220. Tiers: `good` (may create / enrich a person), `usable` (pose up to ±50° / ±35° or slightly blurred: matched,
   never stored, never creates anyone), `reject` (not embedded at all). Embeddings always come from
   `FaceRecognizerSF.alignCrop`.
2. **Track-level decisions** (`track_identity.py`): a track is assigned when one face is ≥ `strong_threshold` (0.60)
   similar or ≥ 60 % of its good faces vote for the same person; it then keeps the identity (later fragments are not
   matched on their own; good faces still verify it, 2 consecutive mismatches → collect again). A new `unknown_N`
   needs ≥ 3 good faces over ≥ 1 s that match nobody — before that the track is `unidentified` (dashboard: "person").
3. **Multi-prototype gallery** (`core/gallery.py`): ≤ 30 face embeddings per person with yaw / pitch / brightness
   buckets, matched by the mean of the top-3 similarities (a person in glasses matches their "glasses" prototypes).
   Admission: good faces only, skip anything ≥ 0.9 similar to a stored prototype, a full gallery drops its most
   redundant prototype while keeping ≥ 3 frontal and ≤ 8 per side. Identified tracks' good faces (≥ 0.30 similar)
   enrich the gallery, so it learns variations over time but stays bounded.
4. **Dedup job** (`dedup.py`, every 10 min and "Перевірити зараз" on the People page): unknown↔known and
   unknown↔unknown pairs scored by gallery similarity (top-5 pairwise + centroid) with bonuses for track continuity
   (the unknown was created ≤ 5 s after a track of that person was lost nearby) and a shared voice. Hard blocks:
   cannot-link (both face-confirmed in one frame, "Це різні люди", or an undone merge). ≥ 0.62 → auto-merge,
   0.42–0.62 → "Possible duplicates" review queue; average linkage + a merged-centroid check stop chain merges.
   A merge moves only admitted embeddings and archives the rest, so it can be undone for 7 days (People page →
   "Історія об'єднань" → Скасувати, or `tools.identity undo <id>`). Junk unknowns (< 3 faces, no voice, unseen
   3 days) are deleted.

Metrics for the thesis: `reports/identity_metrics.json` (per UTC day: unknowns created per hour, rejected faces by
reason, auto / review / manual merges, undos, gallery sizes; no names). Audit-derived numbers cover the days before
this feature too — `python -m tools.identity metrics` recomputes them, giving the before / after duplicate rate.

```bash
python -m tools.identity dedup --dry-run     # what the job would merge / queue / delete
python -m tools.identity duplicates          # the review queue
python -m tools.identity undo 42             # undo a merge (event id from history / the dashboard)
```

Checked on the dev Mac: zidane.jpg (YuNet + SFace) — the frontal face is `good`, the profile is rejected (`pose`,
`small`); an unknown appears only on the 3rd good sample; ≈ 28 ms per frame with two persons. Live camera, real
registry copy: the registered person was recognised (0.61), no new unknown. On the user's registry the largest
similarity between different people is 0.31 (review threshold 0.42). With a 640×480 camera the 80 px face limit
means about 1.5–2 m: farther people stay `unidentified` instead of becoming duplicates — lower `quality.min_face_size`
if that is too strict for the mounting distance.

### Results

LFW, frontal press photos (`tools/eval_faces.py`, reports `reports/identity_eval_lfw*.json`):

| Setup | Threshold 0.363 | 0.40 | **0.45 (default)** | 0.55 |
|---|---|---|---|---|
| 5 people × 5 enrolled photos, 150 tests / 500 strangers | 100 % / 0.8 % FA | 100 % / 0.2 % | **100 % / 0 %** | 99.3 % / 0 % |
| 10 people × 3 enrolled photos, 300 tests / 1000 strangers | 100 % / 2.0 % | 100 % / 0.3 % | **100 % / 0 %** | 99.3 % / 0 % |

(known accuracy / false accept rate; detection failures 0; YuNet + SFace ≈ 12 ms per face on the dev Mac.)

F1 frame rate with the identity process running next to it (same video, separate processes): 11.5–11.6 FPS
without, 11.2–12.2 FPS with — within measurement noise (target: ≤ 20 % drop). Re-check both on the Pi, and with
the real camera: faces smaller than `quality.min_face_size` (80 px) at the mounting distance are never embedded.

## Learned layer (optional)

Trained on public data, loaded only when the files exist (else the rules alone run):

```bash
curl -L -o data/action_datasets/hmdb51_2d.pkl https://download.openmmlab.com/mmaction/v1.0/skeleton/data/hmdb51_2d.pkl
curl -L -o data/action_datasets/ntu60_2d.pkl https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl
python -m tools.train_actions          # → models/action_classifier.joblib (HGB, ~10 MB), reports/action_classifier/
python -m tools.train_cigarette        # CigDet → models/cigarette_yolo11n.pt (5 MB), reports/cigarette_detector/
python -m modules.vision.eval_actions data/human_behaviour_video/labels.csv \
    --classifier models/action_classifier.joblib --cigarette models/cigarette_yolo11n.pt
```

| | result |
| -- | -- |
| classifier, official test splits (HMDB51 split 1 + NTU60 xsub, 13.7 k windows) | macro-F1 0.79, accuracy 0.85; walking 0.96, staggering 0.94, standing_up 0.92, distress 0.91, sitting_down 0.91, falling 0.90, fighting 0.84 … smoking 0.61, running 0.43 |
| cigarette detector, CigDet test (111 photos) | P 0.88, R 0.78, mAP50 0.84; best-F1 confidence 0.48 at 320 px |
| own labelled videos, smoking segments (7) | rules 0 raised → learned layer 5 |
| own labelled videos, 3.8 min without smoking | rules 1 false alarm → learned layer 1 (the same rule one) |
| group scene (office, people close together) | hugging / fighting gone with `classifier_fight_min_prob` 0.9 and the second-participant rule |

Datasets: HMDB51 — CC BY-NC-SA (research); NTU RGB+D — ROSE Lab terms (non-commercial research); CigDet — CC BY 4.0.

