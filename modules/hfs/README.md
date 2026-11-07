# Autonomous hierarchical fuzzy system (A1 / TOD-41)

Article 1: every decision is made locally by a two-level fuzzy system — no LLM, no network.

```
gas_reading ─▶ gas    (max_norm, dynamics, mixture_index: c / c_crit)          ─▶ risk 1 ┐
motion(_level), object_detected, person_identified ─▶ camera (unknown_share,       ─▶ risk 2 ├─▶ top ─▶ overall risk
                                              motion_intensity, activity_time)           │        per location
audio_level, sound_type, speech ─▶ audio (intensity, sound_type_danger,             ─▶ risk 3 ┘   (hfs_risk event)
                                    keyword_danger, speaker_unknown)
```

| Level | Rule base | Rules | Update |
| -- | -- | -- | -- |
| 1 gas | `config/fuzzy/gas.yaml` | 48 | 2 s |
| 1 camera | `config/fuzzy/camera.yaml` | 27 | 1 s |
| 1 audio | `config/fuzzy/audio.yaml` | 81 | 1 s |
| 2 top | `config/fuzzy/top.yaml` | 64 | 2 s |

**Mode.** `config.yaml` → `mode: autonomous` (`core/mode.py`): the LLM worker, GPT gating, the risk envelope and GPT
rule suggestions are not started or imported, the aggregator queues nothing for GPT, Whisper runs locally. The
location state follows `hfs_risk` (high / critical → danger, medium → attention). `mode: hybrid` (default) keeps the
GPT pipeline; the HFS then runs as a shadow (logged, shown on the dashboard, no effect on the state).

**Contract.** `FuzzySubsystem` (`base.py`): `observe(event)`, `features(location) → inputs | None` (None = no data),
`infer(inputs) → {risk, level, interval, fired_rules, inputs, partial}`, own `update_sec`. `TopLevel.combine()`
(`top.py`) → `{risk1, risk2, risk3, overall, level, interval, partial, missing, subsystems, fired_rules_by_level}`.

**Monotonic by construction.** Every rule base is the complete table of a policy in `policy.py` (each input term
step changes the level by at most one: smooth), on Ruspini partitions of triangles / trapezoids, with product AND /
implication, sum aggregation and equal-area output terms. The centroid is then Σ w·c, a multilinear interpolation of
the table: a riskier input never lowers the risk. `python -m tools.plot_surfaces` → `reports/surfaces/` (surfaces +
monotonicity table: 0 falls at every level). The first sparse expert version with min / max fell on up to 10 % of the
grid steps. Tests check every table cell against its policy.

**Missing subsystem.** No device online / stale data → that risk is `null`, its top input takes the default 0 (low).
Because the top level is monotonic, a missing input neither escalates nor blocks anything, so the result rests on the
remaining subsystems. It is marked `partial` and lists `missing`.

**Corridor (A5 / TOD-45).** Every input carries its measured error δ (`config/uncertainty.yaml`,
`python -m tools.fit_uncertainty`); the interval type-2 inference (`core/fuzzy/engine2.py`, Karnik–Mendel,
area-weighted: δ = 0 is exactly type-1) gives each subsystem's corridor [y_l, y_r]; the top level takes them as
interval inputs. Offline / failing devices are excluded (no Beta counters in autonomous mode). `docs/uncertainty.md`.

```bash
python -m modules.hfs --check                                  # validate + metadata of the four rule bases
python -m modules.hfs top risk_gas=0.87 risk_camera=0.1        # explain one inference
python -m tools.plot_surfaces                                  # reports/surfaces/
```

Gas (A2 / TOD-42): every substance as c / c_crit from `config/gases.yaml` (NIOSH IDLH; EPA AQI Hazardous for PM2.5;
`gas_normalize.py`: vol % / ppm / mg/m³ / µg/m³). `max_norm` = max, `mixture_index` = Σ (two gases at 0.6 → critical,
one → high), `dynamics` = least-squares slope of max_norm per minute over 60 s. Sensor survey, periods and the
graphene range problem (its whole dataset is above C_crit): `docs/sensors.md`.

Camera (A3 / TOD-43): `unknown_share` = N_unknown / (N_unknown + N_known) in view, with an interval from the
identification accuracy (Wilson bound of the LFW evaluation, Binomial number of wrong identities → `input_intervals`
for the TOD-45 FOU); `motion_intensity` = smoothed moving share of the frame from the adaptive detector's
`motion_level` events; `activity_time` = signed hours to the location's schedule (`core/schedule.py`: windows,
weekdays, `object_type` presets apartment / office / warehouse / factory). No people count, no actions: the actions
mode is a per-location toggle that only logs. Benchmarks: `reports/camera_bench/`.

Audio (A4 / TOD-44): intensity above each microphone's background, sound types of the noise stage (no trained model
yet: `unknown`, not evidence), graded danger words, speaker identity — `docs/audio.md`. Change a table by
changing its policy in `policy.py` and the YAML together (the test compares them).
