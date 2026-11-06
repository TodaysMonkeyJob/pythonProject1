# Autonomous hierarchical fuzzy system (A1 / TOD-41)

Article 1: every decision is made locally by a two-level fuzzy system — no LLM, no network.

```
gas_reading ─▶ gas    (gas_level, gas_dynamics, mixture_index)                 ─▶ risk 1 ┐
motion, object_detected, person_identified ─▶ camera (unknown_share,              ─▶ risk 2 ├─▶ top ─▶ overall risk
                                              motion_intensity, off_hours)               │        per location
sound, speech ─▶ audio (sound_level, sound_type, danger_keywords, speaker_unknown) ─▶ risk 3 ┘   (hfs_risk event)
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

**Interval.** Subsystem: α-cut (0.5) of the aggregated output set. Top: the subsystem intervals propagated through
the monotonic top level. A5 / TOD-45 replaces both with the IT2 corridor.

```bash
python -m modules.hfs --check                                  # validate + metadata of the four rule bases
python -m modules.hfs top risk_gas=0.87 risk_camera=0.1        # explain one inference
python -m tools.plot_surfaces                                  # reports/surfaces/
```

Inputs are first versions; A2 (gas, PM2.5, IDLH limits), A3 (camera) and A4 (audio) refine them. Change a table by
changing its policy in `policy.py` and the YAML together (the test compares them).
