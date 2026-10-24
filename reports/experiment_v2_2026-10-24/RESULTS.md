# Experiment v2 — scripted-42 (2026-10-24)

> **Caveats:** the dataset is **scripted** (`experiments/scripted.py`) — it validates the method, it is not the recorded evaluation set.

Evaluation split: 84 scenarios (calibration 36, seed 42); LLM: azure / gpt-4.1-mini, temperature 0.2; rule base `a0ab9c218631`; thresholds `807c7d87d3ebe0e4`; trust registry `b99bc8f28a3a802f`. Reference for McNemar: proposed_v2.

## Department questions

| Question | Variant | Macro-F1 (v2 → variant) | ASR | ECE | LLM calls / h | Extra | Answer |
| -- | -- | -- | -- | -- | -- | -- | -- |
| Does type 2 add value? (IT2 vs type-1 corridor) | proposed_v2_type1 | 1.000 → 1.000 | 0.000 → 0.000 | 0.034 → 0.039 | 17.0 → 19.7 | mean width 0.0353 vs 0.1001; corr(w, u) 0.2588 | same decisions on every scenario |
| Does measured trust add value? (registry vs TOD-27 constants) | proposed_v2_static_trust | 1.000 → 1.000 | 0.000 → 0.000 | 0.034 → 0.040 | 17.0 → 15.6 | sensor failure: ASR 0.0 vs 0.0, false alarms 0.75 vs 0.8214 | same decisions on every scenario |
| Does the two-sided corridor add value? (vs floor-only max-projection) | proposed_v2_floor_only | 1.000 → 1.000 | 0.000 → 0.000 | 0.034 → 0.046 | 17.0 → 17.0 | escalation reviews / h 0.54 (right 0.3333) vs 0.0 | same decisions on every scenario |
| Why not fully local? (corridor + local claim detector, no GPT) | proposed_v2_local | 1.000 → 1.000 | 0.000 → 0.000 | 0.034 → 0.039 | 17.0 → 0.0 | GPT calls / h 17.0 vs 0.0 | same decisions on every scenario |

## Metrics

| Configuration | Macro-F1 [95 % CI] | ASR [95 % CI] | False alarms | ECE | Brier | LLM calls / h | Cost / h, $ | Latency median / p95, s | Critical ≤ 5 s |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| rules_only | 0.706 [0.661, 0.739] | 0.250 [0.100, 0.409] | 0.000 | 0.090 | 0.036 | 0.0 | 0.0000 | 0.1 / 61.0 | 1.0 |
| llm_only | 0.671 [0.507, 0.809] | 0.321 [0.147, 0.518] | 0.000 | 0.151 | 0.075 | 19.7 | 0.0066 | 1.84 / 62.73 | 0.5217 |
| llm_pattern | 0.671 [0.507, 0.809] | 0.250 [0.094, 0.423] | 0.000 | 0.147 | 0.073 | 19.7 | 0.0065 | 1.83 / 62.94 | 0.5217 |
| proposed_v2 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.034 | 0.003 | 17.0 | 0.0106 | 2.45 / 62.98 | 1.0 |
| proposed_v2_type1 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.039 | 0.004 | 19.7 | 0.0123 | 1.75 / 62.39 | 1.0 |
| proposed_v2_static_trust | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.040 | 0.004 | 15.6 | 0.0098 | 1.74 / 62.58 | 0.9565 |
| proposed_v2_floor_only | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.046 | 0.005 | 17.0 | 0.0106 | 2.45 / 62.98 | 1.0 |
| proposed_v2_no_B | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.040 | 0.004 | 15.1 | 0.0093 | 2.15 / 63.98 | 1.0 |
| proposed_v2_no_C | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.034 | 0.003 | 17.0 | 0.0057 | 1.79 / 62.95 | 1.0 |
| proposed_v2_no_E | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.018 | 0.002 | 19.7 | 0.0123 | 2.45 / 62.98 | 1.0 |
| proposed_v2_local | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.039 | 0.004 | 0.0 | 0.0000 | 0.1 / 61.0 | 1.0 |

## Ablations (vs proposed_v2)

| Variant | ΔMacro-F1 | ΔASR | ΔECE | ΔLLM calls / h |
| -- | -- | -- | -- | -- |
| proposed_v2 | 1.000 | 0.000 | 0.034 | 17.0 |
| proposed_v2_type1 | +0.000 | +0.000 | +0.005 | +2.7 |
| proposed_v2_static_trust | +0.000 | +0.000 | +0.006 | -1.4 |
| proposed_v2_floor_only | +0.000 | +0.000 | +0.012 | +0.0 |
| proposed_v2_no_B | +0.000 | +0.000 | +0.006 | -1.9 |
| proposed_v2_no_C | +0.000 | +0.000 | -0.000 | +0.0 |
| proposed_v2_no_E | +0.000 | +0.000 | -0.016 | +2.7 |
| proposed_v2_local | +0.000 | +0.000 | +0.005 | -17.0 |

## Corridor

w = y_r − y_l of the corridor behind each decision, u = mean trust uncertainty of its evidence; h = agreement of GPT's answer with the corridor; branch shares over all decisions; an escalation review is right when GPT's level does not exceed the expert level at that moment.

| Configuration | mean w | mean u | corr(w, u) | h median [q25, q75] | agree | deescalation_rejected | escalation_review | clamp | fuzzy_only | floor_raised | floor_kept | reviews / h | reviews right |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| proposed_v2 | 0.0353 | 0.2173 | 0.2588 | 0.9 [0.8432, 0.97] | 0.5303 | 0.1591 | 0.0227 | 0.0076 | 0.2803 | — | — | 0.54 | 0.3333 |
| proposed_v2_type1 | 0.1001 | 0.2173 | 0.1507 | 0.95 [0.9, 1.0] | 0.7348 | 0.0909 | — | 0.0076 | 0.1667 | — | — | 0.0 | — |
| proposed_v2_static_trust | 0.0 | 0.0 | — | 0.8712 [0.8, 0.9] | 0.4242 | 0.2045 | 0.0303 | — | 0.3409 | — | — | 0.72 | 0.5 |
| proposed_v2_floor_only | 0.0353 | 0.2173 | 0.2588 | — [—, —] | — | — | — | — | 0.2803 | 0.0076 | 0.7121 | 0.0 | — |
| proposed_v2_no_B | 0.0 | 0.0 | — | 0.9 [0.8, 1.0] | 0.3276 | 0.1379 | 0.0172 | — | 0.5172 | — | — | 0.54 | 0.0 |
| proposed_v2_no_C | 0.0353 | 0.2173 | 0.2588 | 0.925 [0.275, 1.0] | 0.4318 | 0.2576 | 0.0303 | — | 0.2803 | — | — | 0.72 | 0.25 |
| proposed_v2_no_E | 0.0357 | 0.2128 | 0.2672 | 0.9 [0.8712, 0.97] | 0.7727 | 0.1909 | 0.0273 | 0.0091 | — | — | — | 0.54 | 0.3333 |
| proposed_v2_local | 0.0262 | 0.2092 | 0.2686 | — [—, —] | — | — | — | — | 1.0 | — | — | 0.0 | — |

## Sensor-failure block

Every evaluation scenario replayed with its gas sensor broken: clean-air baseline drift +1.5 % on every reading and a `baseline_drift` self-test report with health 0.3. Registry trust discounts the sensor (τ·h, wider corridor); static trust keeps it at 0.95.

| Configuration | ASR [95 % CI] (failure) | ASR (healthy) | False alarms (failure / healthy) | Dangerous right (failure) | Dangerous under-rated | mean w (failure) |
| -- | -- | -- | -- | -- | -- | -- |
| proposed_v2 | 0.000 [0.000, 0.000] | 0.0 | 0.750 / 0.0 | 1.0 | 0.0 | 0.2238 |
| proposed_v2_static_trust | 0.000 [0.000, 0.000] | 0.0 | 0.821 / 0.0 | 0.75 | 0.0 | 0.0 |

## McNemar (per-scenario correctness vs proposed_v2)

| proposed_v2 vs | only proposed_v2 correct | only other correct | n | exact McNemar p |
| -- | -- | -- | -- | -- |
| rules_only | 13 | 0 | 84 | 0.000244 |
| llm_only | 22 | 0 | 84 | 4.77e-07 |
| llm_pattern | 20 | 0 | 84 | 1.91e-06 |
| proposed_v2_type1 | 0 | 0 | 84 | 1 |
| proposed_v2_static_trust | 0 | 0 | 84 | 1 |
| proposed_v2_floor_only | 0 | 0 | 84 | 1 |
| proposed_v2_no_B | 0 | 0 | 84 | 1 |
| proposed_v2_no_C | 0 | 0 | 84 | 1 |
| proposed_v2_no_E | 0 | 0 | 84 | 1 |
| proposed_v2_local | 0 | 0 | 84 | 1 |

## Calibration of the LLM levels (C)

23 of 36 calibration scenarios; leave-one-out Brier 0.00219 vs fixed mapping 0.00193 → fixed mapping used.

Figures: `asr.png`, `reliability.png`, `width_vs_u.png`, `h_hist.png`, `branches.png`, `failure.png`, `confusion_<config>.png`; per scenario: `per_scenario.csv`, `per_scenario_failure.csv`; reproducibility: `manifest.json` (git commit, rule base version, trust registry snapshot, config / prompt / thresholds hashes, dataset hash) and the LLM cache.
