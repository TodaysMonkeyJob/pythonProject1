# CLAUDE.md

## Purpose
Local voice speech-to-text: microphone → VAD (WebRTC + Silero) → Whisper transcription
(local faster-whisper, or Azure OpenAI `whisper-1`) → speaker identification (SpeechBrain ECAPA).
Plus scripts that compare Whisper models (accuracy / latency / memory).

## Architecture
- `main.py` — entry point: `AudioToTextRecorder(api_connection=False)` looping `recorder.text(cb)`.
  The docstring below the code holds raw experiment logs, not code.
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
python main.py                      # run (needs a microphone and speech_to_text_whisper/voices/*.wav)
python analyzing_whisper_models.py  # plots
python results.py                   # xlsx analysis
```
- No tests, linters, formatters or CI yet. Pre-commit check for now:
  `python3 -m py_compile $(git ls-files '*.py')`.

## Gotchas
- `speech_to_text_whisper/voices/` must exist with at least one `.wav`, or `SpeakerRecognizer` fails.
- First run downloads models from Hugging Face (`large-v2` ~3 GB; SpeechBrain into `model/`).
- Defaults are CUDA-oriented; on macOS only CPU works.
- `os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'` in the recorder is a deliberate OpenMP workaround — keep it.
- `numpy<2.0.0` and `torch~=2.3.0` are pinned on purpose (torch 2.3 is built against numpy 1.x).
  `pandas~=2.3` keeps it that way — pandas 3 pulls numpy 2 and breaks scipy/torch. Run `pip check` after changes.
- The recorder writes `IoTsecuritySTTx.log` to cwd (git-ignored).

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
- Before committing: py_compile (later `pytest`) must pass; show the diff.
- `git commit` only after confirmation; `git push` and PRs only on explicit request.

## Do not touch
- `.env` and any secret/credential files.
- `.idea/` (local IDE settings, git-ignored).
- `multi_model_sentence_data.xlsx`, `speech_to_text_whisper/warmup_audio.wav` — data/binaries.
- `speech_to_text_whisper/voices/`, `model/` — reference voices and model cache (never commit).
- `speech_to_text_whisper/audio_recorder_testing.py` — leave as is.
