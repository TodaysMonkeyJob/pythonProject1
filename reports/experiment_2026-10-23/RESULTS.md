# Results — scripted-42 (2026-10-23)

> **Caveats:** the dataset is **scripted** (generated from templates by `experiments/scripted.py`) — it validates the harness, it is not the recorded evaluation set.

Evaluation split: 84 scenarios (calibration 36, seed 42); annotator agreement κ = 0.9647; LLM: azure / gpt-4.1-mini, temperature 0.2.

## Metrics

| Configuration | Macro-F1 [95 % CI] | ASR [95 % CI] | False alarms | ECE | Brier | LLM calls / h | Cost / h, $ | Latency median / p95, s | Critical ≤ 5 s |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| rules_only | 0.706 [0.661, 0.739] | 0.250 [0.100, 0.409] | 0.000 | 0.090 | 0.036 | 0.0 | 0.0000 | 0.1 / 61.0 | 1.0 |
| llm_only | 0.671 [0.507, 0.809] | 0.321 [0.147, 0.518] | 0.000 | 0.151 | 0.075 | 19.7 | 0.0066 | 1.84 / 62.73 | 0.5217 |
| llm_pattern | 0.671 [0.507, 0.809] | 0.250 [0.094, 0.423] | 0.000 | 0.147 | 0.073 | 19.7 | 0.0065 | 1.83 / 62.94 | 0.5217 |
| proposed | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.034 | 0.003 | 17.0 | 0.0106 | 2.45 / 62.98 | 1.0 |
| proposed_no_A | 0.889 [0.810, 0.959] | 0.214 [0.067, 0.385] | 0.000 | 0.114 | 0.048 | 17.0 | 0.0103 | 1.84 / 2.98 | 1.0 |
| proposed_no_B | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.040 | 0.004 | 15.1 | 0.0093 | 2.15 / 63.98 | 1.0 |
| proposed_no_C | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.034 | 0.003 | 17.0 | 0.0057 | 1.79 / 62.95 | 1.0 |
| proposed_no_E | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.018 | 0.002 | 19.7 | 0.0123 | 2.45 / 62.98 | 1.0 |
| proposed_v2 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.033 | 0.003 | 17.0 | 0.0101 | 2.58 / 63.53 | 1.0 |

## Summary

* Attack success rate: proposed 0.000 [0.0, 0.0] vs best baseline rules_only 0.250 [0.1, 0.4091]; LLM only 0.321.
* Macro-F1 (normal + dangerous): proposed 1.000 [1.0, 1.0].
* Calibration: ECE 0.034, Brier 0.003; false alarm rate 0.000.
* Efficiency: 17.0 LLM calls / h (LLM only: 19.7).

## Calibration of the LLM levels (C)

Fitted on 23 of 36 calibration scenarios (2 answers rejected by the corridor as manipulated left out). In-sample Brier 0.00105, ECE 0.01546; leave-one-out Brier 0.00219, ECE 0.01297 (fixed mapping: Brier 0.00193, ECE 0.03663). Constraints hold: monotone centres {'low': 0.2381, 'medium': 0.485, 'high': 0.6813, 'critical': 0.9647}, min support 0.2 ≥ 0.2. Not used: leave-one-out Brier worse than the fixed level mapping: the fixed mapping is used.

## Ablations

| Variant | ΔMacro-F1 | ΔASR | ΔECE | ΔLLM calls / h |
| -- | -- | -- | -- | -- |
| proposed (A+B+C+E) | 1.000 | 0.000 | 0.034 | 17.0 |
| proposed_no_A | -0.111 | +0.214 | +0.080 | +0.0 |
| proposed_no_B | +0.000 | +0.000 | +0.006 | -1.9 |
| proposed_no_C | +0.000 | +0.000 | -0.000 | +0.0 |
| proposed_no_E | +0.000 | +0.000 | -0.016 | +2.7 |

## McNemar (per-scenario correctness, same scenarios)

| proposed vs | only proposed correct | only other correct | n | exact McNemar p |
| -- | -- | -- | -- | -- |
| rules_only | 13 | 0 | 84 | 0.000244 |
| llm_only | 22 | 0 | 84 | 4.77e-07 |
| llm_pattern | 20 | 0 | 84 | 1.91e-06 |
| proposed_no_A | 13 | 0 | 84 | 0.000244 |
| proposed_no_B | 0 | 0 | 84 | 1 |
| proposed_no_C | 0 | 0 | 84 | 1 |
| proposed_no_E | 0 | 0 | 84 | 1 |
| proposed_v2 | 0 | 0 | 84 | 1 |

Figures: `asr.png`, `reliability.png`, `confusion_<config>.png`; per scenario: `per_scenario.csv`; reproducibility: `manifest.json` (git commit, config / prompt hashes, dataset hash) and the LLM cache.
