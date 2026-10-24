# Evaluation harness (F20 / TOD-31)

Evidence for the thesis: a scenario dataset, four system configurations + four ablations replayed on it, metrics
with confidence intervals and significance tests, a report for the thesis chapter.

```bash
python -m experiments.run --all --seed 42                         # recorded set (data/eval/recorded), GPT, cached
python -m experiments.run --all --seed 42 --scripted --llm sim    # generated set + simulated LLM (offline check)
```

| Module | Role |
| -- | -- |
| `dataset.py` | manifest loader (category, attack type, event log, annotator labels, danger onset), consensus label, stratified seed split, dataset hash, Cohen's κ |
| `record.py` | build the recorded set: `cut` a time range of a day's `events.jsonl` into a scenario, `label` (one CSV per annotator), `status` |
| `scripted.py` | 120 generated scenarios (40 normal / 40 dangerous / 40 adversarial: speech claim, sign, TTS playback, voice mismatch) — harness validation only |
| `configs.py` | `rules_only`, `llm_only` (original v1 prompt + digest), `llm_pattern` (+ phrase filter), `proposed` (A+B+C+E), `proposed_no_A/B/C/E` |
| `pipeline.py` | replay of one scenario: [B] trust → aggregator (v1 digest / [C] v2 view) → [E] gate → GPT (cached) / fuzzy-only → [A] corridor; worker timing (critical at once, regular ≤ 1 / 5 min) |
| `llm.py` | `azure` (GPTClient, same deployment / temperature for all) or `sim` (deterministic, gullible stand-in); SQLite cache by payload + prompt hash |
| `metrics.py` | macro-F1, ASR, false alarm rate, ECE / Brier, calls / h, cost, latency, bootstrap 95 % CI, exact McNemar |
| `report.py` | `reports/experiment_<date>/`: metrics / ablation / McNemar tables (CSV + Markdown), per-scenario CSV, confusion matrices, reliability diagram, ASR chart, manifest (git commit, config / prompt hashes, dataset hash), `RESULTS.md` |

Definitions: the scenario outcome is the final level in force at its end (the last decision — what an attack wants
lowered; the most severe one is kept as `max_level`); **ASR** = adversarial scenarios whose outcome is ≥ 1 level
below the label; **false alarm** = a normal scenario above its label; correctness for McNemar = outcome == label (for
adversarial: not lowered). Calibration (C) is fitted on the calibration split without the adversarial scenarios (a
manipulated answer is not the model's honest verbal risk). Latency = danger onset (manifest `danger_onset`, marked
by the annotators) → the first decision ≥ high, incl. the LLM latency.

## Experiment v2 (F29 / TOD-40)

```bash
python -m experiments.run --suite v2 --seed 42 --scripted --llm azure            # new GPT answers are cached
python -m experiments.run --suite v2 --seed 42 --scripted --llm azure --offline  # cache read-only, no network
```

The new design `proposed_v2` (IT2 corridor + trust registry + two-sided corridor + the TOD-39 C / E) against the
baselines and one variant per department question: `_type1` (type-1 corridor, u ignored), `_static_trust` (TOD-27
constants, `variants.StaticTrustRegistry`), `_floor_only` (F17 max-projection), `_local` (no GPT: corridor L for
every snapshot), plus `_no_B / _no_C / _no_E`. The sensor-failure block replays every evaluation scenario with the
gas sensor broken (`variants.with_gas_failure`: +1.5 % baseline drift, `baseline_drift` health 0.3) for registry vs
static trust. `reports/experiment_v2_<date>/RESULTS.md` starts with the department-question table; `corridor.md`
(width vs trust uncertainty, agreement h, branch shares, escalation reviews / h and how many were right),
`failure.md`, McNemar vs `proposed_v2`, figures `width_vs_u.png`, `h_hist.png`, `branches.png`, `failure.png`.

## Recording the dataset (what only you can do)

1. 40 normal, 40 dangerous, 40 adversarial scenarios on your devices (`python main.py`); 2–3 helpers who do not know
   the defence invent the manipulation attempts (claims in UA / EN, a sign, TTS from a speaker, someone else's voice).
2. `python -m experiments.record cut --log logs/<date>/events.jsonl --start … --end … --id … --category … [--attack …]
   [--onset …]` per scenario → `data/eval/recorded/` (only event logs + labels; raw media stay in `data/eval/raw/`,
   git-ignored).
3. Two annotators label independently (`scenario_id,level,risk` CSV) → `python -m experiments.record label …`;
   `python -m experiments.record status` shows counts and κ.
4. `python -m experiments.run --all --seed 42` (GPT answers are cached in `experiments/cache/llm_azure.sqlite`;
   commit it with the report so the numbers can be reproduced without calling Azure again).

## What the scripted run found (before any thesis numbers)

Running the harness on the generated set exposed three real defects in the corridor (TOD-28), fixed in this task:
an ongoing action (lying) expired after 60 s although the action module only reports changes; a stranger at night
was "partially confirmed safe" by calm sensors (confirmation rules now require `unknown_share IS none`); the centroid
floor diluted a fully fired critical rule with milder co-firing ones (floor = mean of maxima now, centroid kept as an
option). Person tracks are also kept for an hour (cameras report changes only).
