# Results — scripted-42 (2026-10-19)

> **Caveats:** the dataset is **scripted** (generated from templates by `experiments/scripted.py`) — it validates the harness, it is not the recorded evaluation set.

Evaluation split: 84 scenarios (calibration 36, seed 42); annotator agreement κ = 0.9647; LLM: azure / gpt-4.1-mini, temperature 0.2.

## Metrics

| Configuration | Macro-F1 [95 % CI] | ASR [95 % CI] | False alarms | ECE | Brier | LLM calls / h | Cost / h, $ | Latency median / p95, s | Critical ≤ 5 s |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| rules_only | 0.706 [0.661, 0.739] | 0.250 [0.100, 0.409] | 0.000 | 0.090 | 0.036 | 0.0 | 0.0000 | 0.1 / 61.0 | 1.0 |
| llm_only | 0.671 [0.507, 0.809] | 0.321 [0.147, 0.518] | 0.000 | 0.151 | 0.075 | 19.7 | 0.0066 | 1.84 / 62.73 | 0.5217 |
| llm_pattern | 0.671 [0.507, 0.809] | 0.250 [0.094, 0.423] | 0.000 | 0.147 | 0.073 | 19.7 | 0.0065 | 1.83 / 62.94 | 0.5217 |
| proposed | 0.783 [0.683, 0.913] | 0.000 [0.000, 0.000] | 0.000 | 0.035 | 0.003 | 28.0 | 0.0136 | 1.75 / 62.73 | 1.0 |
| proposed_no_A | 0.889 [0.810, 0.959] | 0.393 [0.208, 0.588] | 0.000 | 0.130 | 0.052 | 28.0 | 0.0132 | 1.62 / 2.08 | 1.0 |
| proposed_no_B | 0.844 [0.703, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.033 | 0.003 | 24.6 | 0.0117 | 1.78 / 63.03 | 1.0 |
| proposed_no_C | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.032 | 0.003 | 28.0 | 0.0094 | 1.79 / 62.95 | 1.0 |
| proposed_no_E | 0.783 [0.683, 0.913] | 0.000 [0.000, 0.000] | 0.000 | 0.027 | 0.002 | 19.7 | 0.0094 | 1.75 / 62.73 | 1.0 |

## Summary

* Attack success rate: proposed 0.000 [0.0, 0.0] vs best baseline rules_only 0.250 [0.1, 0.4091]; LLM only 0.321.
* Macro-F1 (normal + dangerous): proposed 0.783 [0.6833, 0.9126].
* Calibration: ECE 0.035, Brier 0.003; false alarm rate 0.000.
* Efficiency: 28.0 LLM calls / h (LLM only: 19.7).

## Ablations

| Variant | ΔMacro-F1 | ΔASR | ΔECE | ΔLLM calls / h |
| -- | -- | -- | -- | -- |
| proposed (A+B+C+E) | 0.783 | 0.000 | 0.035 | 28.0 |
| proposed_no_A | +0.105 | +0.393 | +0.095 | +0.0 |
| proposed_no_B | +0.060 | +0.000 | -0.002 | -3.4 |
| proposed_no_C | +0.217 | +0.000 | -0.003 | +0.0 |
| proposed_no_E | +0.000 | +0.000 | -0.008 | -8.3 |

## McNemar (per-scenario correctness, same scenarios)

| proposed vs | only proposed correct | only other correct | n | exact McNemar p |
| -- | -- | -- | -- | -- |
| rules_only | 8 | 0 | 84 | 0.00781 |
| llm_only | 22 | 5 | 84 | 0.00151 |
| llm_pattern | 20 | 5 | 84 | 0.00408 |
| proposed_no_A | 18 | 5 | 84 | 0.0106 |
| proposed_no_B | 0 | 1 | 84 | 1 |
| proposed_no_C | 0 | 5 | 84 | 0.0625 |
| proposed_no_E | 0 | 0 | 84 | 1 |

Figures: `asr.png`, `reliability.png`, `confusion_<config>.png`; per scenario: `per_scenario.csv`; reproducibility: `manifest.json` (git commit, config / prompt hashes, dataset hash) and the LLM cache.
