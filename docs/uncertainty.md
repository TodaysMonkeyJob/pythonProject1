# Trust from measurement error: the fuzzy corridor (A5 / TOD-45)

**Slide 22.** The Beta reputation model (confirmations, refutations, forgetting; TOD-34) fits a simulation, not
real sensors. They rarely fail, and their readings can rarely be confirmed or refuted, so the counters stay empty.
In Article 1 (`mode: autonomous`) trust comes from what is actually known about every sensor and decision: **its
accuracy, i.e. its measurement error**. The Beta trust registry is not created or imported in this mode; a test
checks it.

## From an error to the corridor

1. Every input of the hierarchical fuzzy system has a half-width **δ**, in the input's own units, with a
   documented source (`config/uncertainty.yaml`, written by `python -m tools.fit_uncertainty`).
2. δ is the **footprint of uncertainty** of an interval type-2 set (`core/fuzzy/it2.py`):
   - Triangles and trapezoids (the default): the upper membership function moves every sloped edge outward by δ,
     the lower one inward; shoulders on the universe edge stay fixed.
   - Gaussians: either an uncertain width σ ∈ [σ(1 − k), σ(1 + k)] (`sigma_k`) or an uncertain centre ± δ.
3. Every rule fires with an interval [f̲, f̄]. **Karnik–Mendel** centre-of-sets type reduction, weighted by the
   consequents' areas, gives the **corridor [y_l, y_r]** of each subsystem (`core/fuzzy/engine2.py`):
   - With δ = 0 it reproduces the type-1 risk exactly (tested on all four rule bases).
   - Its width never shrinks when δ grows (property test).
   - KM equals the brute-force switch-point search on random cases.
4. The top level takes risks 1–3 as intervals: the midpoint, with δ = half the subsystem corridor. Its corridor is
   the corridor of the decision; `widened_by` names the inputs or subsystems that widen it most.
5. The crisp risk (the level and location state) stays the type-1 result. **The corridor width shows how much the
   decision depends on sensor errors.** The dashboard draws it as a band on every risk meter ("±w").

## Where every δ comes from

| Input | δ | Source |
| -- | -- | -- |
| gas concentration → `max_norm`, `mixture_index` | δ_c = max(abs, rel · c) of the reading's sensor, ÷ C_crit; the mixture sums the δ (worst case) | graphene FET: regressor MAE per gas, `reports/gas_metrics.json` (0.0074–0.0176 vol %). DFRobot SEN0469 NH₃: ±10 % or ±5 ppm (datasheet / vendor). SDS011 / SPS30 PM2.5: datasheets (`docs/sensors.md`). DFRobot SEN0466 CO: **unmeasured** (accuracy not published) |
| `dynamics` | δ_c_norm · 60 / √Σ(t − t̄)² per minute | propagated from the concentration error and the sample times (least-squares slope error) |
| `unknown_share` | binomial: the 95 % quantile f of wrong identities among n → f / n | identification accuracy p = 0.994: Wilson lower bound of 650/650 correct decisions at threshold 0.45, `reports/identity_eval_lfw.json` |
| `motion_intensity` | 0.0066 of the frame | the largest moving share the adaptive detector reported on scenes without anyone (`reports/camera_bench/motion.json`, synthetic clips) |
| `activity_time` | 0 | the schedule and the clock: exact |
| `intensity` | half the spread (p30 − p10) / 2 of the microphone's per-second background levels | live statistics of each microphone; the absolute mic calibration is **unmeasured** (no reference microphone) |
| `sound_type_danger` | 1 − per-type accuracy (Wilson LB) once a model is trained, else 1 − the classifier's confidence | **no trained model yet** (TOD-44): sound types are not evidence |
| `keyword_danger` | 1 − accuracy = 0.465 (only when somebody spoke) | danger-dictionary words said vs heard by faster-whisper-base on 320 synthetic TTS phrases at −5 … 30 dB SNR: 17 right, 6 missed or false, Wilson lower bound 0.535 (`data/calibration/asr_phrases.csv`) |
| `speaker_unknown` | 1 − accuracy = 0.071 (only when somebody spoke) | held-out speaker verification EER 5.3 % on 825 pairs (`config/calibration/speaker.yaml`, synthetic TTS voices) |

**Decision-type inputs** (identity, words, speaker, sound type) use one rule: δ = (probability of a wrong
decision) × (the largest change a wrong decision makes on the 0..1 input). Accuracy is the Wilson 95 % lower bound,
so a perfect small sample (650/650) does not claim zero error. **Unmeasured** errors are δ = 0 and are listed as
such; they are never typed in.

## Examples (`config/uncertainty.yaml` of 2026-11-07)

| Situation | Risk | Corridor | Why that wide |
| -- | -- | -- | -- |
| NH₃ 150 ppm on the electrochemical module | 0.55 (high) | 0.46 – 0.62 | ±10 % of the reading = ±0.05 C_crit |
| NH₃ 0.015 vol % on the graphene FET | 0.55 (high) | 0.13 – 0.87 | its error, 108 ppm, is a third of C_crit (300 ppm): it cannot place a reading near the limit |
| 1 known + 1 unknown person, 20:30 | 0.28 (medium) | 0.22 – 0.32 | motion noise; p = 0.994 leaves no doubt about two identities |
| «Здається, пожежа» from a known voice | 0.87 (critical) | 0.35 – 0.87 | one ASR decision, danger-word accuracy measured on few noisy words |
| the three together (top level) | 0.87 (critical) | 0.65 – 0.87 | the audio corridor, damped by the other subsystems |

The last row shows what the corridor is for. The decision is critical, but the corridor says it rests on one
recognised word, whose recognition accuracy is poorly measured. Recording more danger words (and the scream /
noise set of TOD-44) is what narrows it.

## Device health: exclusion, no counters

A device that is offline, or whose self-diagnostics (`device_health`) report an excluding check, is **excluded**
until it is back and healthy: no frames, frozen frame, lens covered, stuck silence, wrong sample rate, sensor
error, flat readings, out of range (`hfs.health.exclude`).
- Its evidence is dropped and its subsystem works with the remaining devices, marked `partial`.
- If no device of that subsystem is left, the subsystem is missing and the top level is partial.
- `hfs_risk.excluded` lists the device and the reason, and the dashboard explanation shows it.

Softer findings (too dark, blurred, noisy) do not exclude a device. Turning them into a larger δ is future work.

## Reproduce

```bash
python -m tools.camera_bench          # motion noise on static scenes → reports/camera_bench/motion.json
python -m tools.fit_uncertainty       # → config/uncertainty.yaml (sources, dates, values)
python -m pytest tests/test_uncertainty.py
```
