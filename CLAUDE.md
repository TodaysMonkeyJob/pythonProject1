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
  - `events.py` — pydantic `Event` (unified schema: event_id, timestamp UTC ms `Z`, source, type, severity,
    payload, confidence), `make_event()`, `to_json()`/`from_json()`. New event types go into `EVENT_TYPES`.
  - `bus.py` — `multiprocessing.Queue` carrying events as JSON strings (`publish`/`receive`, `CONTROL_STOP`).
  - `logger.py` — `JsonlLogger`: `logs/YYYY-MM-DD/<module>.jsonl`, UTC daily rotation, total size cap
    (oldest days deleted first, current day never).
  - `config.py` — `load_config()`: `config.yaml` + secrets from env/`.env` via `ENV_MAP`; fails fast on
    missing sections or `required_env`.
  - `module.py` — `BaseModule`: `setup/step/teardown` loop, heartbeat every 5 s, stops on `"stop"`.
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
    `identity.py` (F3, TOD-14) → `IdentityRecognizer`: YuNet + SFace on person boxes, `faces.npz` database
    (`tools/enroll_face.py`), `person_identified` known / `unknown_N` / `not_visible` on change. F2 and F3 share
    `consumer.SharedFrameConsumer`. `tools/eval_faces.py` → accuracy / false accepts on a face folder.
    `identity_store.py`: unknown people saved to `data/faces/unknown/unknown_N/`; renaming that folder (or
    `enroll_face --from-unknown`) enrolls the person, moves the crops, writes `aliases.json`, relabels live tracks.
    Identified tracks are re-verified every `verify_sec` (a new person can inherit a track id). F2/F3 events go
    to `logs/<date>/actions.jsonl` / `identity.jsonl`.
  - `modules/aggregator/` — F5 (TOD-16): `Aggregator` keeps the window's scene state (people + identity + actions,
    objects, transcripts, sounds, gas, alerts, mic state) and builds a JSON snapshot every `snapshot_interval_sec`
    (if the scene is active / something changed) or at once on a critical event (`severity: critical`,
    `immediate_actions`, unknown person in `unknown_alert_hours`). Validated by `schemas/snapshot.schema.json`,
    shrunk to `max_snapshot_bytes` (2 KB), written to `exports/snapshots/` and queued in the SQLite
    `exports/outbox.db` (`outbox.Outbox`: pending / sent / failed — F6 reads it). Every event also goes to
    `exports/events_YYYY-MM-DD.csv` (gas concentration in ppm = % × 10 000).
  - `modules/audio/mic_controller.py` — F4 microphone: STANDBY (RMS + webrtcvad on 30 ms frames, nothing recorded)
    / ACTIVE (scene_active, loud sound or speech; speech segments → `modules/llm/whisper_client.py` → `speech`
    with `backend`). `whisper_backend: azure` falls back to local faster-whisper when keys are missing or a request
    fails (Azure retried after `azure_retry_sec`); `local` = local only;
    `mic_state` events (source system). `modules/audio:run` is the legacy `AudioToTextRecorder` wrapper.
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
python -m modules.vision.demo 0 --pose-model yolo11n-pose.pt   # webcam: objects + actions
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
- `data/faces/` — face photos and embeddings (biometric data, git-ignored, never commit).
- `speech_to_text_whisper/audio_recorder_testing.py` — leave as is.
