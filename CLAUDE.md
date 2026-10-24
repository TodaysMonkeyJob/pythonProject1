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
  - Learned action layer (`modules/vision/action_features.py` window features, `action_classifier.py`): an sklearn
    classifier on COCO-17 pose windows (2.5 s at 5 fps) trained by `tools/train_actions.py` on public skeletons
    (HMDB51 + NTU RGB+D 60 from OpenMMLab, git-ignored `data/action_datasets/`, research licences) → 18 actions
    (falling, staggering, fighting, distress, running, climbing, throwing, picking_up, sitting_down, standing_up,
    drinking, eating, phone_call, waving, hugging, smoking, walking, other); `action` events with `source:
    classifier`, once per second per track, 2 agreeing windows, fighting / hugging need a second participant, lying
    keeps priority. Smoking evidence: hand-to-mouth rule, classifier, and a yolo11n cigarette detector
    (`tools/train_cigarette.py` on CigDet, CC BY 4.0, trained on CPU — MPS did not converge) run on a head / hands
    crop once per second per person, counting only sightings at the mouth / in a hand (`cigarette_min_hits` in
    10 s, one at the mouth); a confident drinking / eating / phone call vetoes the rule. Models in git-ignored
    `models/` (absent → rules only); metrics in `reports/action_classifier/`, `reports/cigarette_detector/`;
    `eval_actions --classifier --cigarette` compares with the rules.
  - F14 (TOD-25) duplicates: `face_quality.assess` (good / usable / reject before embedding), `track_identity`
    (`TrackIdentityResolver`: strong match or voting per track, inherited; unknown only after 3 good faces over 1 s,
    else `unidentified`), `core/gallery.py` (≤ 30 prototypes, top-3 matching, admission: good + non-redundant +
    bucket quotas), `dedup.DedupJob` (every 10 min in the identity process / dashboard "Перевірити зараз": auto-merge
    ≥ 0.62, review queue 0.42–0.62, cannot-link blocks, average linkage, junk cleanup) and
    `identity_metrics` → `reports/identity_metrics.json`. Registry: `cannot_link` + `merge_candidates` tables, merges
    archive instead of delete (`archived_by`) → `unmerge()` within `undo_days`.
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
- `core/fuzzy/` — F15 (TOD-26) Mamdani fuzzy inference core (numpy only, no network, ≈ 0.25 ms per `infer()` on the
  dev Mac; target ≤ 5 ms on the Pi), shared by the fuzzy trust layer (TOD-27) and the risk corridor (TOD-28):
  `membership.py` (tri / trap / gauss, list = max for wrap-around terms), `variables.py` (`LinguisticVariable`,
  per-gas `variants`, `default` for missing inputs), `rules.py` (`IF … AND / OR / NOT (…) THEN risk IS x [WITH w]`),
  `engine.py` (`FuzzyEngine.from_config(path, location_id)` → `infer()` = crisp + output-term activations + readable
  `fired_rules`; `fuzzify()`), `validate.py` (unknown variables / terms, universe gaps, duplicate / conflicting rules
  → `FuzzyConfigError`). Config `config/fuzzy.yaml` (variables, rules R1–R22, `maps.action_risk`, `locations:`
  overrides: terms / shift / rules / remove_rules / weights; also `fuzzy:` blocks in `config/locations.yaml`).
  `python -m core.fuzzy --check` / `python -m core.fuzzy gas_concentration=4.2 gas=acetone --location warehouse`.
- `modules/trust/` — F16 (TOD-27) fuzzy trust in evidence, run by the orchestrator router on every evidence event
  before it is logged / forwarded (`trust.enabled`): `channels.py` (`channel_of`: physical / identity / speech /
  scene_text; `EvidenceState`: presence per location, cameras + health, gas readings, recent alarms — learned from
  the stream), `claims.py` (UA + EN pattern detector: instruction_to_system > deescalation > escalation > none),
  `scorer.py` (`TrustScorer`: τ = min(μ_source, μ_asr, μ_consistency) with the `trust.source` / `asr` / `identity`
  rule bases in `config/fuzzy.yaml`; adds trust, trust_terms, provenance, claim_type, claim_spans,
  playback_suspected, trust_reasons, trust_components). Speech events now carry `asr_confidence` (Whisper
  avg_logprob / no_speech_prob) and `speaker_overlap`. New event type `scene_text`. See `modules/trust/README.md`.
- `modules/envelope/` — F17 (TOD-28) fuzzy risk corridor in the LLM worker: the orchestrator forwards trust-scored
  evidence (`ENVELOPE_EVENTS`) to the llm module; `state.TrustedState` (τ ≥ θ only, low-trust lowering claims kept as
  "ignored", operator acks), `corridor.py` (risk FIS → r_floor = crisp, r_ceil = α-cut top; `FuzzyEngine.infer` now
  returns `cut`), `confirmation.py` (`envelope.confirmation` rules; ack → c = 1), `projection.py`
  (r_final = max(r_LLM, (1 − c)·r_floor), level maps). `llm_decision` gets risk_llm / risk_floor / risk_ceil /
  confirmation / risk_final / risk_level_final / envelope_applied / envelope_reason; raised → `deescalation_rejected`;
  both → `logs/<date>/envelope.jsonl`. Location state uses `risk_level_final`; the location page shows the corridor.
- F18 (TOD-29) what GPT sees and how its risk is read: `modules/aggregator/llm_view.py` builds snapshot v2's
  `llm_view` (fuzzified terms via `FuzzyEngine.fuzzify`, words for durations / loudness, `trusted_evidence` /
  `untrusted_evidence` split at θ, no raw sensor numbers; the v1 sections stay as the local debug record; also the
  v1 → v2 adapter); the outbox `digest` column holds its compact JSON (`aggregator.llm_format: digest` = old text);
  the LLM worker adds `risk_floor {term, mu}` from the corridor. Prompt: spotlighting + `risk_distribution`
  (`LLMDecision`, renormalised, optional). `modules/llm/calibration.py` (`RiskCalibrator`: r_LLM = centroid of
  Σ p·μ; Brier / ECE / reliability), `tools/calibrate_llm_risk.py` → `config/calibration.yaml` (used by the envelope
  only once it exists), `tools/snapshot_tokens.py` (v1 JSON / v1 digest / v2 tokens), `tools/test_llm.py --repeat`.
- F19 (TOD-30) `modules/llm/gating.py` `LLMGate` in the LLM worker (`llm.gating`): novelty vs the location's last
  30 min (`snapshot_features`), corridor uncertainty, untrusted claims, minutes since the last GPT call, severity →
  `gating` rules in `config/fuzzy.yaml` → call / fuzzy-only. Hard: critical_event and (F28) a critical floor no
  GPT answer within the interval saw → GPT at once; everything else (also untrusted claims) `min_llm_interval_sec`. Fuzzy-only
  → `llm_decision` `source: fuzzy_only`, risk = r_floor, `fuzzy_summary` (Ukrainian template), outbox row
  `mark_handled`; works without Azure keys. `gating.jsonl` (inputs, tokens saved). Dashboard badge LLM / без GPT.
  `tools/gating_replay.py` (synthetic routine hour or a real events.jsonl, read only): calls vs every snapshot.
- `experiments/` — F20 (TOD-31) evaluation harness: `python -m experiments.run --all --seed 42 [--scripted --llm sim]`
  replays every scenario of `data/eval/recorded` (or the generated `data/eval/scripted`) through 4 configurations +
  4 ablations (`configs.py`, `pipeline.py`), GPT cached by payload hash (`experiments/cache/llm_<backend>.sqlite`),
  metrics + bootstrap CIs + McNemar (`metrics.py`) → `reports/experiment_<date>/` (`report.py`, `RESULTS.md`).
  `record.py` cuts / labels the recorded set (raw media in git-ignored `data/eval/raw/`). See `experiments/README.md`.
- F22 (TOD-33) measured trust inputs (`core/calibration.py`): `config/fuzzy.yaml` has no hand-picked trust degree /
  threshold — strings `"@trust_priors:…"` / `"@thresholds:…"` are resolved by `core.fuzzy.load_yaml` (always load
  fuzzy.yaml through it) from `config/trust_priors.yaml` (`tools/trust_priors.py`: one entry per TOD-34 device ×
  channel, r0 / s0 on held-out data → τ0, u0, Wilson LB; no data → `prior_status: unmeasured`, r0 = s0 = 0, never a
  typed value; counts from `tools/measure_channels.py` → `reports/channel_eval/`; `policy.*` = rules + interim
  values, `measured: false`), `config/thresholds.yaml` (`tools/fit_thresholds.py`: min C_miss·P_miss +
  C_fa·P_fa on the calibration split, `calibration.cost_ratio` in config.yaml; reads the GPT cache read-only, misses →
  simulator, level boundaries kept when < 10 scenarios have recorded answers; log `reports/thresholds_log.jsonl`) and
  `trust.calibration` maps `config/calibration/{asr,speaker,face,gas}.yaml` (`tools/calibrate_channels.py`: isotonic /
  logistic / Bayes bins; the scorer falls back to the `trust.asr` / `trust.identity` rule bases without them).
  Data: `tools/synth_speech.py` (macOS `say` voices + noise → git-ignored `data/calibration/speech/`), committed
  `data/calibration/asr_phrases.csv` / `speaker_pairs.csv`. `tools/calibration_report.py` → `reports/calibration_<date>/`.
  Changing trust values changes the GPT payloads: re-run `experiments.run --llm azure` (new calls) before refitting levels.
- F23 (TOD-34) trust registry (`modules/trust/registry.py`, `config/trust_registry.yaml`): trust per device × channel
  (camera objects / motion / pose_action / smoking / face_identity / scene_text; mic vad / asr / speaker_identity /
  face_voice_link; gas gas_class / gas_concentration / gas_trend; identity person_registry), τ = (r+1)/(r+s+2)·h with
  measured priors (unmeasured → r0 = s0 = 0), γ per channel kind, health h from `device_health` events of the workers'
  self-diagnostics (`modules/trust/health.py`, `trust.health`), confirmations / refutations from the stream
  (`evidence.py`) and operator `trust_feedback`; SQLite `trust.registry_db` (`data/trust_registry.db`, never commit;
  replays / tests use `:memory:`, the conftest guard covers it) with `history`; Ukrainian explanations. `trust.base`
  is gone: μ_source = registry τ. Dashboard page `#/trust` «Довіра» (`views/trust.js`, `/api/trust`,
  `/api/trust/history`, POST `/api/trust/feedback`), trust dot on camera tiles, feedback buttons in event dialogs.
  `experiments.run` stores the registry setup in `manifest.json`. Pose / smoking priors come from the 640 px action
  evaluation (`tools/measure_channels.py --actions-report`; deployed `actions.imgsz` is recorded next to it).
- F24 (TOD-35) interval type-2 engine: `core/fuzzy/it2.py` (UMF / LMF = type-1 edges moved ±δ, shoulders on the
  universe edges fixed; Karnik–Mendel centre-of-sets `cos_interval`, `brute_force_cos` for tests) and
  `core/fuzzy/engine2.py` `FuzzyEngine2.infer(inputs, uncertainty={var: u})` → interval [y_l, y_r], type1 COS,
  firing [lo, hi], fou, `widened_by` (inputs ranked by width contribution); δ_var = k_var · u, `it2.k` =
  `@thresholds:values.k_var` (κ · span_var, κ fitted by `tools/fit_thresholds.py` on the interval score with the cost
  ratio as non-coverage penalty). The corridor (`modules/envelope/corridor.py`, `envelope.floor`: it2_mom default |
  it2 | mom ablation) and the confirmation (c = lower end) use it; NO θ cut for physical evidence any more — events
  carry `trust_u` (registry u discounted by health) and `TrustedState.features` returns `facts["uncertainty"]`;
  only speech / scene-text claims use `trust.claim_threshold` (`@thresholds:values.claim_theta`). `llm_decision` has
  `risk_width`, `risk_type1`, `envelope_reason.widened_by`; the location page prints what widened the corridor.
- F25 (TOD-36) two-sided fuzzy corridor (`modules/envelope/fuzzy_corridor.py`): K = plateau on [L, U] with slopes
  of the risk terms' mean edge width, L = (1 − c)·(y_l + λ(y_r − y_l)), U = y_r; μ_GPT = max_t p_t·μ_t (calibrated
  level sets), μ_D = min, h = sup μ_D, r_D = centroid (kept in [L, U]); policy agree (h ≥ h_min → r_D) /
  deescalation_rejected (L) / escalation_review (U, event + dashboard Accept / Reject → POST /api/escalation/{id} →
  `escalation_resolved` → trust registry `llm/risk_assessment`, `RuleSuggestionStore` (F27, now
  `modules/rules/suggestions.py`; `rules.suggestions_db` = data/rule_suggestions.db, guarded) and the location takes GPT's
  level) / clamp / fuzzy_only (no GPT → L, `RiskEnvelope.fallback`). λ, h_min fitted by `tools/fit_thresholds.py`
  (λ: cost of L vs expert level on adversarial / normal scenarios; h_min: end-to-end level cost, Youden J tie-break).
  `llm_decision` carries y_l, y_r, width, L, U, r_llm, r_D, h, policy_branch, c; `envelope_applied` = level raised.
- F26 (TOD-37) rule governance: every rule entry of `config/fuzzy.yaml` has `meta: {source, author, rationale,
  added}` (`core.fuzzy.governance.metadata_errors`; `python -m core.fuzzy --check` fails without it and prints the
  version); `validate.py` catches duplicates / conflicts in any clause order and reports subsumed rules. Version =
  sha256 of the normalised rule set (metadata excluded) → `llm_decision.rules_version`, experiment manifests,
  dashboard page `#/rules` «Правила» (`/api/rules`: rules + metadata + firing stats + changelog, read-only).
  Changes only via `python -m tools.rules propose <file>` (checks + `tools/rules_gate.py`: proposed pipeline on the
  120 scenarios with the cached GPT read-only, blocks ASR > `rules.gate.max_asr` (failing scenario ids) or macro-F1 /
  ECE worse than the tolerances; per-scenario diff, never-fired rules) → `approve <id> --by --role --comment` →
  `data/rules/versions/<sha>.yaml` + append-only `data/rules/changelog.jsonl`; `rollback <version>`.
  `modules/llm/rule_stats.py` (`rules.stats_db`): fired rules per version + operator outcomes of escalation reviews.
  Never edit `config/fuzzy.yaml` rules by hand (the conftest guard also watches it and `data/rules/*`).
- F27 (TOD-38) GPT rule suggestions, a separate review store (never on the bus, in logs/ or the live feed):
  `modules/rules/suggestions.py` (`RuleSuggestionStore`, SQLite `rules.suggestions_db`: outcomes = cases, jobs,
  suggestions with validator / gate report, status new / rejected / edited / sent_to_approval / approved, dedup by
  canonical rule + location → `support_count`), `suggester.py` (`SuggestionTriggers` in the router via
  `modules.trust.evidence`: accepted escalation review, ≥ `repeat_n` escalation_reviews per location in
  `repeat_window_h`, operator-confirmed physical alarm while the latest decision fired no non-low risk rule;
  `RuleSuggester`: `prompts/rule_suggester.md` + DSL grammar + risk variables → one GPT call per job, JSON mode,
  pydantic `Suggestion`, `validate_rule` — invalid DSL stored as is), run by the LLM worker when the outbox is empty
  (at most every `rules.suggestions.every_sec`) or `python -m tools.suggest_rules --since 7d` (Azure keys).
  `review.py`: reject / edit (re-validated) / `send_to_approval` = candidate copy with rule `GS<id>` inserted as text
  (comments kept) + meta → `tools.rules` `RuleStore.propose`; `config/fuzzy.yaml` only changes on `tools.rules
  approve`. CLI `python -m tools.suggestions list|show|reject|edit|send|export`; dashboard `#/suggestions`
  «Пропозиції правил» (`/api/suggestions…`, send runs the gate in a thread).
- F28 (TOD-39) fixes of C and E after the ablation. C: snapshot v2.1 (`llm.snapshot_version` v1 | v2 | v2.1, wins
  over `aggregator.llm_format`; `llm_view(version=)`): gas item `range_pct` / `trend_range_pct_per_min` (core range
  of the strongest term) + view `gas_scale_pct` (the location's gas terms) — term definitions, never readings;
  `prompts/security_analyst.md` medium / high rubric with `{risk_scale}` filled from the risk output terms
  (`gpt_client.risk_scale`); calibration (`modules/llm/calibration.py`) constrained (centres inside their risk term's
  support, ≥ 0.05 apart, supports ≥ `MIN_WIDTH`, Huber loss), `loo_cv`, `check_constraints`; experiments fit on the
  whole calibration split minus answers the corridor rejected (`deescalation_rejected`) and use the fit only when its
  LOO Brier beats the fixed mapping; `tools/calibrate_llm_risk.py --harness` (cache read-only, ≥ 10 recorded answers).
  E: `LLMGate` inputs `width` (corridor y_r − y_l) and `agreement` (h of GPT's last answer, `gate.record()` from the
  worker / replay; absent = never asked), novelty of a first snapshot = what it shows; bypass of the interval only
  for critical_event and a critical floor no answer within the interval saw (claims / disagreement are fuzzy inputs);
  `reasons` in `gating.jsonl`; `gating.threshold` fitted with `calibration.call_cost` per sent snapshot
  (`fit_cost_threshold(flag_cost=)`). Gating rules G1–G15 went through `tools.rules` propose / approve.
- F29 (TOD-40) experiment v2: `python -m experiments.run --suite v2 --seed 42 [--scripted] [--llm azure]
  [--offline]` → `reports/experiment_v2_<date>/`. `configs.V2`: baselines + `proposed_v2` (= proposed flags) and
  `_type1` (`envelope.floor: mom`, u ignored), `_static_trust` (`experiments/variants.StaticTrustRegistry`: TOD-27
  constants 0.95 / 0.1 / 0.8, no health, u = 0; `TrustScorer.from_config(registry_factory=)`), `_floor_only`
  (`envelope.policy: floor_only` = F17 max-projection, branches floor_raised / floor_kept), `_no_B/C/E`, `_local`
  (no GPT, every snapshot = corridor L). Sensor-failure block (`configs.FAILURE`, `variants.with_gas_failure`: gas
  +1.5 % drift + `baseline_drift` health 0.3) on every evaluation scenario. `Decision` carries width / u / h /
  branch / truth; `metrics.corridor_stats` (mean w, corr(w, u), h quartiles + histogram, branch shares, escalation
  reviews / h and share right = GPT level ≤ expert level) and `failure_stats`; `report.write_v2`: questions.md
  (department questions answered by numbers + McNemar verdict), corridor.md, failure.md, ablation / McNemar vs
  proposed_v2, figures width_vs_u / h_hist / branches / failure; manifest hashes thresholds / trust files.
  `--offline` = `tools.fit_thresholds.OfflineLLM` (cache read-only, misses → simulator, counted).
- `modules/locations/` — F12: `state.LocationStateEngine` (safe / attention / danger / offline per location from its
  events + the latest `llm_decision` of that location; danger cools down after `cooldown_min`, Acknowledge clears it;
  per-location rules) and `module.LocationsModule` (emits `location_state_changed`, pushes `location_summary` —
  not logged, dashboard only — every 1.5 s on change; notifies). `modules/notify/service.py`: `data/notifications.db`,
  rate limit per (location, reason), Telegram via plain HTTPS for `danger` (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID).
  The aggregator keeps one scene per location (immediate snapshot of the critical location only, location name +
  rules in the digest, `global` site snapshot every `global_snapshot_min`); the LLM worker throttles per location
  and `llm_decision` carries `location_id`.
- `dashboard/static/` — F13 (TOD-24), single-screen redesign F21 (TOD-32: top-bar territory tabs + ◀ ▶ / ← → / Home / 1–9, Overview = cameras + cards + newest-first log, territory = camera + tabbed panel Події · Розмова · Газ · Рішення GPT, `ui.LiveList` keeps the scroll position; no page scroll, no scroll-snap) control room, plain ES modules (no build step, no CDN, nothing vendored):
  `app/main.js` (hash router `#/overview` `#/loc/<id>[/details]` `#/people` `#/devices` `#/locations`, carousel ← →,
  swipe, dots, ↓/↑ summary ⇄ details), `store.js` (REST + /ws state, names via person_id), `video.js` (one /ws/video
  socket, subscribes to visible tiles only), `charts.js` (canvas), `views/` (overview, grid, location, admin),
  `app.css` (no page scroll ≥ 1280×720, light/dark). Backend: `dashboard/video.py` (`FrameCache`: one 320 px frame per
  camera frame shared by all pages; `[4-byte len][header JSON][JPEG]`), `/static/{path}` (behind auth),
  `/api/history` (events of a location, read backwards from logs). See `dashboard/README.md`.
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
- `reports/identity_metrics.json` — written by the live identity process (tests: tmp paths, guarded).
- `data/faces/`, `data/identity.db*` — face photos, face / voice embeddings (biometric data, git-ignored, never
  commit). Tests must use `tmp_path` registries and log dirs: `tests/conftest.py` fails the run if real
  `data/identity.db*` or `logs/*/*` change. Never delete them without checking — they hold the user's live data.
- `speech_to_text_whisper/audio_recorder_testing.py` — leave as is.
