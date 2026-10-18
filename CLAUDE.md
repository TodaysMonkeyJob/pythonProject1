# CLAUDE.md

## Purpose
Local voice speech-to-text: microphone → VAD (WebRTC + Silero) → Whisper transcription
(local faster-whisper, or Azure OpenAI `whisper-1`) → speaker identification (SpeechBrain ECAPA).
Plus scripts that compare Whisper models (accuracy / latency / memory).

## Architecture
- `main.py` — orchestrator entry point (F4 / TOD-15): `python main.py [--only gas,vision]` runs every enabled module
  in its own process (`core/orchestrator.py`). The original standalone recorder is `python -m speech_to_text_whisper`.
- `core/orchestrator.py` — `Orchestrator`: `DEFAULT_MODULES` (name → `pkg.mod:run`, config section, forwarded event
  types), spawn processes, router (every event → `logs/<date>/events.jsonl` + fan-out to subscribers' control
  queues), watchdog (dead process or 3 missed heartbeats → restart with backoff, `system` `alert`), graceful
  SIGINT/SIGTERM, creates/unlinks the vision shared buffer. Modules whose model files are missing are skipped with
  a hint; F5–F7 are picked up automatically once `modules.aggregator` / `modules.llm` / `modules.dashboard` have `run`.
  With `network.discovery` (default) vision / audio are not static modules but per-device pipelines
  (`add_module` / `remove_module` at runtime); `device_*` events go to every module. `--only` limits discovery too.
  Module names must equal the `BaseModule.name` (heartbeats are matched by it). `deploy/iot-guard.service` — systemd.
- `speech_to_text_whisper/` — the package:
  - `audio_recorder.py` — `AudioToTextRecorder`: orchestrates processes/threads (`torch.multiprocessing`),
    audio capture (PyAudio), recording state, halo spinner.
  - `transcription_worker.py` — separate transcription process, talks to the recorder over a `Pipe`.
    `api_connection=True` → Azure OpenAI, otherwise faster-whisper warmed up on `warmup_audio.wav`.
    Inside the worker `print` is redirected into the pipe.
  - `vad_webrtc.py`, `vad_silero.py` — voice activity detectors.
  - `recognize_speaker.py` — `SpeakerRecognizer`; reference voices in `speech_to_text_whisper/voices/*.wav`
    (path is relative to cwd → always run from the repo root).
  - `speech_config.py` — all default constants (model, VAD thresholds, 16 kHz sample rate).
  - `records.py` — helper script that records a 5 s reference voice sample.
  - `audio_recorder_testing.py` — old/experimental recorder (wake words, porcupine). Not imported.
  - `parameters_explanation.txt` — recorder parameter reference (partly outdated).
- `core/` — IoT security framework (Linear project fixit-mobile, Step 0 / TOD-11):
  - `events.py` — pydantic `Event` (unified schema v2: schema_version, event_id, timestamp UTC ms `Z`, source, type,
    severity, payload, confidence + required `device_id`, `node_id`, `location_id`), `make_event()` (device / node /
    location from `set_context()` unless given), `to_json()`/`from_json()` (v1 log lines still readable).
    New event types go into `EVENT_TYPES`.
  - `bus.py` — `multiprocessing.Queue` carrying events as JSON strings (`publish`/`receive`, `CONTROL_STOP`).
  - `logger.py` — `JsonlLogger`: `logs/YYYY-MM-DD/<module>.jsonl`, UTC daily rotation, total size cap
    (oldest days deleted first, current day never).
  - `config.py` — `load_config()`: `config.yaml` + secrets from env/`.env` via `ENV_MAP`; fails fast on
    missing sections or `required_env`.
  - `module.py` — `BaseModule`: `setup/step/teardown` loop, heartbeat every 5 s, stops on `"stop"`. Per-device
    instances get `config["_device"]` / `config["_module"]` (`vision@camera-…`); `emit()` stamps device + location;
    a forwarded `device_updated` for its device changes `self.location_id`.
  - `devices.py` — F11 (TOD-22) `DeviceRegistry`: SQLite `data/devices.db` (devices: stable `device_id` = kind +
    hash(node, hw key), kind camera/mic/gas, transport, location (new → `unassigned`), enabled, status; locations).
    Publishes `device_online` / `device_offline` / `device_updated`.
  - `discovery.py` — local cameras (macOS `system_profiler` = AVFoundation / OpenCV order, Continuity iPhone →
    `continuity`; Linux `/sys/class/video4linux` + picamera2) and mics (sounddevice, virtual ones ignored); `diff()`.
  - `device_manager.py` — in the orchestrator: scans every `network.discovery_interval_sec`, starts / stops /
    restarts one `vision@<id>` (own shared buffer `iotv_<hash>`) or `audio@<id>` process per enabled device; also for
    remote `browser` / `rtsp` devices (on their device_online / offline). Child processes get small OMP / BLAS
    thread pools via env (`orchestrator.child_threads`, `<section>.threads`) — without them 4 cameras were 14× slower.
  - `tls.py` (self-signed HTTPS cert via the system openssl), `onvif.py` (WS-Discovery probe, stdlib).
  - `locations.py` — F12 (TOD-23) `LocationRegistry`: the `locations` table of `devices.db` (order, icon, rules:
    `unknown_person_after`, `gas_thresholds`, `attention_window_min`, `cooldown_min`); delete moves devices to
    `unassigned`; seed `config/locations.yaml` (example: `config/locations.example.yaml`).
  - Remote nodes (see `deploy/NETWORK.md`, no MQTT / extra programs): `dashboard/node.py` on the HTTPS port 8443 —
    pairing codes → tokens (hashes in `devices.db`), phone page `static/node.html` (JPEG `0x01` + PCM `0x02` over WSS →
    `IngestSource` shared buffer / `SocketSource` unix socket), Pi agents (`node_agent/agent.py` + `modules/uplink.py`:
    events + previews over WSS, hub found by UDP announce, cert pinned), RTSP / ONVIF endpoints.
- `modules/<name>/__init__.py` exposes `run(event_queue, control_queue, config)`.
  - `modules/gas` — gas classifier (TOD-19). `data.py` loads `data/gas/data.xls` (4 sheets = gases, row = R at
    16 gate voltages Ug 0.0..1.5 V, col 0 = C %; C = 0 rows → class `clean_air`) and writes `data_clean.xlsx` /
    `replay_demo.csv`. `train.py` splits ORIGINAL rows first, augments the train part only, grid-searches
    scaled models → `models/*.joblib` (git-ignored) + `reports/gas_metrics.json`. `inference.py` →
    `GasPredictor` (no matplotlib). `worker.py` → `run()`, replay or sensor mode
    (`gas.reader: "pkg.module:function"` returning 16 values), publishes `gas_reading` + `alert`.
    `legacy/gas_model.py` — original research script, reference only (excluded from ruff).
  - `modules/vision` — camera module (TOD-12), see `modules/vision/README.md`. `detector.py` →
    `VisionDetector`: MOG2 motion every frame, YOLO + ByteTrack (`yolo.py`) only on motion or every
    `yolo_interval_sec`; publishes `motion`, `object_detected` (appeared/disappeared), `scene_active` on
    changes only; writes annotated JPEG + raw frame + person boxes to `SharedFrameBuffer` (shared memory).
    `export.py` → NCNN model, `bench.py` → FPS/RAM report.
    `actions.py` (F2, TOD-13) → `ActionRecognizer`: pose (`pose.py`, yolo11n-pose) on shared frames with persons,
    rule-based standing/sitting/walking/lying + `possible_smoking` (`gas_confirmed` from gas events forwarded
    via the control queue); `action` events only on change. `eval_actions.py` → labelled-video accuracy.
    `identity.py` (F3, TOD-14) → `IdentityRecognizer`: YuNet + SFace on person boxes matched in the F9 registry,
    `person_identified` (`person_id` + `identity` = display name, `unknown_N` / `not_visible`) on change. F2 and F3
    share `consumer.SharedFrameConsumer`, which with discovery serves EVERY camera buffer in one process
    (`scheduler.FairScheduler`: active cameras at the module fps, idle ones at `idle_fps`, round-robin; per-camera
    engines because track ids are per camera). `tools/eval_faces.py` → accuracy / false accepts on a face folder.
    Identified tracks are re-verified every `verify_sec` (a new person can inherit a track id); `identity_*` events
    (or `sync_sec` polling) relabel live tracks. `legacy_faces.py` imports the old `faces.npz` / unknown folders
    into an empty registry. F2/F3 events go to `logs/<date>/actions.jsonl` / `identity.jsonl`.
  - `core/identity.py` — F9 (TOD-20) `IdentityRegistry`: SQLite `data/identity.db` (WAL; persons, face / voice
    embeddings as float32 BLOBs, top-K per person, identity_events audit, global `unknown_N` counter). Every process
    opens its own connection and cache; `poll()` reloads on `PRAGMA data_version`. Mutations (create / rename / merge /
    link / delete) publish `identity_*` events (source system). `person_id` is the key in events; names are resolved
    at render time. `core/cooccurrence.py`: `PresenceTracker` + `VoiceFaceLinker` (new voice heard 3× while one
    identified person is alone in view → merged into that face; 2+ people → no link; known never reassigned).
    Rename / merge / delete: dashboard People page, `tools/identity.py`; enrollment `tools/enroll.py` (faces, voice).
  - `modules/aggregator/` — F5 (TOD-16): `Aggregator` keeps the window's scene state (people + identity + actions,
    objects, transcripts, sounds, gas, alerts, mic state) and builds a JSON snapshot every `snapshot_interval_sec`
    (if the scene is active / something changed) or at once on a critical event (`severity: critical`,
    `immediate_actions`, unknown person in `unknown_alert_hours`). Validated by `schemas/snapshot.schema.json`,
    shrunk to `max_snapshot_bytes` (2 KB), written to `exports/snapshots/` and queued in the SQLite
    `exports/outbox.db` (`outbox.Outbox`: pending / sent / failed — F6 reads it). Every event also goes to
    `exports/events_YYYY-MM-DD.csv` (gas concentration in ppm = % × 10 000).
    GPT gets only text: `digest.snapshot_digest()` (one line per section, local time, ~100 tokens vs ~260 for the
    JSON) stored in `outbox.digest`. No images or audio are ever sent. A critical snapshot opens a local incident
    (`incidents.IncidentRecorder`): `exports/incidents/<id>.zip` with summary.txt (digests + timeline),
    incident.json, events.jsonl (2 min before … 15 s after) and annotated camera frames sampled from the vision
    shared buffer; repeated critical events extend the open incident; size-capped. Only `incident_id` reaches GPT.
  - `modules/llm/` — F6 (TOD-17): `worker.LLMWorker` reads the SQLite outbox and sends only `digest` (text, ~100
    tokens) with `prompts/security_analyst.md` (~290 tokens) to `gpt_client.GPTClient` (`llm.gpt_deployment`,
    JSON mode with fallback, one clarification retry, token + cost accounting) → `llm_decision` event
    (risk_level, summary in Ukrainian, recommended_action, reasons, follow_up). Critical snapshots immediately,
    regular ones once per `min_llm_interval_sec` (older ones skipped). Offline / 429 / 5xx → backoff, rows stay
    pending; no keys → idles. `azure_client.create_client` is shared with Whisper (AzureOpenAI when
    AZURE_OPENAI_API_VERSION is set, else the v1 `OpenAI(base_url=…)`). `tools/test_llm.py` (`--dry-run`) — 10 cases.
- `modules/locations/` — F12: `state.LocationStateEngine` (safe / attention / danger / offline per location from its
  events + the latest `llm_decision` of that location; danger cools down after `cooldown_min`, Acknowledge clears it;
  per-location rules) and `module.LocationsModule` (emits `location_state_changed`, pushes `location_summary` —
  not logged, dashboard only — every 1.5 s on change; notifies). `modules/notify/service.py`: `data/notifications.db`,
  rate limit per (location, reason), Telegram via plain HTTPS for `danger` (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID).
  The aggregator keeps one scene per location (immediate snapshot of the critical location only, location name +
  rules in the digest, `global` site snapshot every `global_snapshot_min`); the LLM worker throttles per location
  and `llm_decision` carries `location_id`.
- `dashboard/` — F7 (TOD-18): `server.DashboardModule` runs uvicorn (FastAPI) in a background thread on
  `dashboard.port` (8000); the orchestrator forwards all events + heartbeats (`subscribe: ["*", "heartbeat"]`).
  `/api/persons` (F9: list, PATCH rename, POST merge, DELETE, thumbnail) is the single rename point.
  `/ws` pushes events (last 200 from `logs/<date>/events.jsonl` on connect), `/video` MJPEG from the vision shared
  buffer (`dashboard_fps`), `/api/status` (heartbeats incl. `heartbeat_payload()` extras: vision FPS, mic state;
  CPU temp, RAM), `/api/snapshot/{id}` (digest sent to GPT + JSON), `/api/incident/{id}` (zip). HTTP Basic auth from
  DASHBOARD_USER / DASHBOARD_PASS; without them it binds 127.0.0.1 only. `static/index.html`: single page, vanilla
  JS + Chart.js (CDN), all event text inserted via textContent.
  - `modules/audio/mic_controller.py` — F4 microphone: STANDBY (RMS + webrtcvad on 30 ms frames, nothing recorded)
    / ACTIVE (scene_active, loud sound or speech; speech segments → `modules/llm/whisper_client.py` → `speech`
    with `backend`). `whisper_backend: azure` falls back to local faster-whisper when keys are missing or a request
    fails (Azure retried after `azure_retry_sec`); `azure_only` = cloud only, no fallback; `local` = local only;
    `WhisperOptions` (both backends): fixed `language` or detection restricted to `languages` (Azure re-asks in the
    first one), beam 5, no `condition_on_previous_text`, hallucination filter (silence / compression ratio / known
    phrases); `whisper_max_backlog` skips stale segments on a slow Pi. Target device is the Pi: no GPU-only paths.
    `mic_state` events (source system). F10 (TOD-21) `speaker_id: true`: `modules/audio/speaker.py`
    `SpeakerAnalyzer` (16 kHz + 80 Hz high-pass + loudness norm, 1.5 s / 0.75 s windows, ECAPA in mini-batches,
    agglomerative cosine clustering + split pass → speaker turns; per cluster `registry.rank()` → known / unknown_N /
    uncertain; clean single-speaker chunks enrich the profile) and `modules/audio/attribution.py` (Whisper segments
    aligned / split to turns; `TranscriptLog` → `logs/<date>/transcripts.jsonl` + `.txt`). One `speech` event per
    segment: `person_id`, `speaker_name`, `speaker_score`, `speaker_status`, `text`, `start`/`end`, `location_id`,
    `device_id`. Voice ↔ face linking from forwarded `object_detected` / `person_identified`.
    `tools/calibrate_voice.py` → EER + suggested `identity.voice_threshold`. `modules/audio:run` is the legacy `AudioToTextRecorder` wrapper.
- `config.yaml` — non-secret settings per module; `.env.example` — required secret names.
- `system_setup/logger_setup.py` — shared logger: `init_logger()` + `set_logger_params(logger, log_file,
  log_level, no_log_file)`. Re-created after the original was lost; keep this signature stable.
- `analyzing_whisper_models.py` — plots from hard-coded experiment data.
- `results.py` — analysis of `multi_model_sentence_data.xlsx` (pandas/seaborn); writes `test.xlsx` to cwd.

## Environment
- Python 3.12, pip + `requirements.txt` (no lock file, no pyproject).
- System dependency: PortAudio (`brew install portaudio`) for PyAudio. On Apple Silicon build PyAudio with
  `CFLAGS=-I/opt/homebrew/include LDFLAGS=-L/opt/homebrew/lib pip install -r requirements.txt`.

## Commands
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m speech_to_text_whisper    # standalone recorder (needs a microphone and speech_to_text_whisper/voices/*.wav)
python analyzing_whisper_models.py  # plots
python results.py                   # xlsx analysis
```
```bash
pip install -r requirements-dev.txt     # pytest, ruff
python -m pytest                        # tests/ (no microphone, models or network needed)
ruff check core modules tests           # lint new code only
python -m modules.gas.data              # regenerate data/gas/data_clean.xlsx + replay_demo.csv
python -m modules.gas.train             # train gas models (~10 s) → models/, reports/
python -m modules.vision.export         # yolo11n.pt → models/yolo11n_ncnn_model
python -m modules.vision.bench --source synthetic   # vision FPS/RAM (on the Pi: camera)
YOLO_TEST_MODEL=yolo11n.pt python -m pytest -k real_yolo   # optional test with a real model
python main.py                          # whole system (Ctrl+C to stop); events → logs/<date>/events.jsonl
python -m tools.test_llm --dry-run      # GPT example cases: tokens / cost; without --dry-run needs .env keys
python -m modules.vision.demo 0 --pose-model yolo11n-pose.pt   # webcam: objects + actions
python -m tools.identity list                 # F9 registry: rename unknown_3 Oleh / merge / delete / history
python -m tools.enroll --name Oleh --faces data/faces/Oleh/ --voice speech_to_text_whisper/voices/oleh.wav
python -m tools.calibrate_voice --audio <dir>/<name>/*.wav   # voice_threshold from EER (≥ 2 people)
YOLO_POSE_TEST_MODEL=yolo11n-pose.pt python -m pytest -k real_pose
```
- Legacy files have many pre-existing ruff findings (star imports etc.); don't mass-fix them. No CI yet.

## Gotchas
- `speech_to_text_whisper/voices/` must exist with at least one `.wav`, or `SpeakerRecognizer` fails.
- First run downloads models from Hugging Face (`large-v2` ~3 GB; SpeechBrain into `model/`).
- Defaults are CUDA-oriented; on macOS only CPU works.
- `os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'` in the recorder is a deliberate OpenMP workaround — keep it.
- `numpy<2.0.0` and `torch~=2.3.0` are pinned on purpose (torch 2.3 is built against numpy 1.x).
  `pandas~=2.3` keeps it that way — pandas 3 pulls numpy 2 and breaks scipy/torch. Run `pip check` after changes.
- The recorder writes `IoTsecuritySTTx.log` to cwd (git-ignored).
- Installing ultralytics without constraints upgrades numpy to 2.x and torch: keep `torchvision~=0.18.0`
  in requirements. Ultralytics auto-installs missing extras at runtime (`lap`, `ncnn`) — pin them instead.
- Shared memory (`SharedFrameBuffer`) is not cleaned up by process exit; stale blocks live until reboot
  or `unlink()`. On macOS the name must be ≤ 30 chars.
- Gas models are pickles tied to the scikit-learn version: retrain after upgrading sklearn
  (`GasPredictor` logs a warning on mismatch). `SVC(probability=True)` is deprecated in sklearn 1.9.

## Secrets & config
- Azure OpenAI: env vars `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`.
- Never read, print or commit `.env` or any file containing keys.

## Code conventions
- PEP 8, 4 spaces, snake_case; classes PascalCase; constants UPPER_CASE in `speech_config.py`.
- New defaults go into `speech_config.py`, not as literals in code.
- Log via `logger` from `system_setup.logger_setup`, not `print`.
- Docstrings in `:param x:` / `:return:` style (see `recognize_speaker.py`).
- Existing comments may be Ukrainian; don't translate them. New docs/comments in English.
- No mass reformatting or refactoring outside the task scope.

## Git workflow
- Remote `origin` → github.com/TodaysMonkeyJob/pythonProject1, main branch `master`.
- Tasks are tracked in Linear. One branch per task from up-to-date `origin/master`: `feature/tod-<N>`.
- Never commit or push directly to `master`.
- Commit message: `TOD-<N>: <imperative summary>` (subject ≤72 chars; body optional).
- Before committing: `pytest` and `ruff` (on new/changed code) must pass; show the diff.
- `git commit` only after confirmation; `git push` and PRs only on explicit request.

## Do not touch
- `.env` and any secret/credential files.
- `.idea/` (local IDE settings, git-ignored).
- `multi_model_sentence_data.xlsx`, `speech_to_text_whisper/warmup_audio.wav` — data/binaries.
- `speech_to_text_whisper/voices/`, `model/` — reference voices and model cache (never commit).
- `data/devices.db*` — the user's device registry (locations, enabled, node tokens), never delete.
- `data/notifications.db*`, `exports/` — the user's notifications / snapshots / outbox (live runs: redirect them).
- `data/tls/` (hub HTTPS key), `data/node_agent/` (edge token) — secrets, git-ignored, never commit.
- `data/faces/`, `data/identity.db*` — face photos, face / voice embeddings (biometric data, git-ignored, never
  commit). Tests must use `tmp_path` registries and log dirs: `tests/conftest.py` fails the run if real
  `data/identity.db*` or `logs/*/*` change. Never delete them without checking — they hold the user's live data.
- `speech_to_text_whisper/audio_recorder_testing.py` — leave as is.
