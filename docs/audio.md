# Audio subsystem (A4 / TOD-44)

Risk 3 of the autonomous hierarchical fuzzy system: `intensity × sound_type_danger × keyword_danger ×
speaker_unknown` → `config/fuzzy/audio.yaml` (108-rule table of `modules/hfs/policy.py` `audio()`, exactly
monotonic: `reports/surfaces/`).

| Input | Source | Notes |
| -- | -- | -- |
| `intensity` | `modules/audio/intensity.py` → `audio_level` events | dB above **this microphone's** background: the 20th percentile of per-second levels over 5 min, or a calibrated `audio.mic.background_db`; smoothed 2 s |
| `sound_type_danger` | `modules/audio/noise_stage.py` + `sound_type.py` → `sound_type` events | the background around every speech segment and every non-speech burst (+10 dB, 0.3–3 s) classified; danger from `config/audio_types.yaml` |
| `keyword_danger` | `modules/audio/keywords.py`, `config/danger_words.yaml` | graded words (alarming 0.4, dangerous 0.7, critical 1.0), UA + EN; exact / stem + ending / edit distance; `danger_words` in every `speech` event |
| `speaker_unknown` | speaker identification (F10) | known 0, uncertain 0.5, unknown 1 |

## Sound types: no trained model yet

The recorded noise set of the previous paper (keyboard, walking, fan) and scream samples are not in the repository
or on this machine. Until they are, **no model ships**: every `sound_type` event says `unknown` with
`danger: null`, and the fuzzy system ignores it (sound_type_danger = 0). Train once the recordings are available:

```bash
# <dir>/keyboard/*.wav, <dir>/walking/*.wav, <dir>/fan/*.wav, <dir>/scream/*.wav (+ impact / glass_breaking / alarm)
python -m tools.train_sound_types --data <dir>      # → models/sound_types.joblib (git-ignored)
                                                     #   reports/sound_types/{metrics.json,README.md}: per-type
                                                     #   accuracy, precision, confusion, share below min_confidence
```

The split is by file, so 2-second clips of one recording never land on both sides. The model is a scaler plus a
single-threaded random forest (~3 ms per clip). Gradient boosting took 50 ms and spun every core, which is too much
next to Whisper on the Pi. The classifier's pipeline is tested end to end on synthetic sounds (hum, clicks, thumps,
a pitched scream) in `tests/test_audio_a4.py`; those numbers are no measure of real accuracy.

## Danger words

Matching order:
1. Exact form, or a listed form.
2. Stem plus a Ukrainian ending («пожеж·ею»).
3. Levenshtein distance ≤ 1 (5–7 letters) or ≤ 2 (8+) for Whisper's near misses («допаможіть»).

Words of up to 4 letters match exactly only, and a fuzzy hit that is a common word (`common_words`: «газета», «раз»,
"fine" …) is rejected. This is not a real lemmatiser: Ukrainian morphology would need a dictionary (pymorphy),
which we don't ship, so irregular forms are listed per entry.

## Noise stage and denoising

`NoiseProfile` averages the power spectrum of the non-speech frames and keeps their last 3 s for classification.
`wiener_denoise` (soft Wiener mask, √Hann overlap-add) can clean speech before Whisper with `audio.mic.denoise: true`.
It is off by default: Whisper is trained on noisy speech and a mask can remove consonants. Measure the word error
rate before switching it on.

## Latency

`python -m tools.audio_bench` → `reports/audio_bench/`. Measured on the dev Mac only: a microphone frame costs
0.03 ms of the 30 ms budget, a sound type 6–9 ms per segment (own thread), danger words 2 ms, Whisper tiny 0.21×
real time. **The Raspberry Pi has not been measured**: run the same command there.
