# Results — scripted-42 (2026-10-19)

> **Caveats:** the dataset is **scripted** (generated from templates by `experiments/scripted.py`) — it validates the harness, it is not the recorded evaluation set.

Evaluation split: 84 scenarios (calibration 36, seed 42); annotator agreement κ = 0.9647; LLM: azure / gpt-4.1-mini, temperature 0.2.

## Metrics

| Configuration | Macro-F1 [95 % CI] | ASR [95 % CI] | False alarms | ECE | Brier | LLM calls / h | Cost / h, $ | Latency median / p95, s | Critical ≤ 5 s |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| rules_only | 0.706 [0.661, 0.739] | 0.250 [0.100, 0.409] | 0.000 | 0.090 | 0.036 | 0.0 | 0.0000 | 0.1 / 61.0 | 1.0 |
| llm_only | 0.671 [0.507, 0.809] | 0.321 [0.147, 0.518] | 0.000 | 0.151 | 0.075 | 19.7 | 0.0066 | 1.84 / 62.73 | 0.5217 |
| llm_pattern | 0.671 [0.507, 0.809] | 0.250 [0.094, 0.423] | 0.000 | 0.147 | 0.073 | 19.7 | 0.0065 | 1.83 / 62.94 | 0.5217 |
| proposed | 0.792 [0.679, 0.890] | 0.000 [0.000, 0.000] | 0.000 | 0.067 | 0.008 | 28.0 | 0.0135 | 2.17 / 63.33 | 1.0 |
| proposed_no_A | 0.889 [0.810, 0.959] | 0.536 [0.350, 0.724] | 0.000 | 0.123 | 0.051 | 28.0 | 0.0131 | 2.19 / 3.1 | 1.0 |
| proposed_no_B | 0.930 [0.843, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.063 | 0.007 | 24.6 | 0.0116 | 2.08 / 63.04 | 1.0 |
| proposed_no_C | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.032 | 0.003 | 28.0 | 0.0094 | 1.79 / 62.95 | 1.0 |
| proposed_no_E | 0.792 [0.679, 0.890] | 0.000 [0.000, 0.000] | 0.000 | 0.060 | 0.007 | 19.7 | 0.0093 | 2.17 / 63.33 | 1.0 |

## Summary

* Attack success rate: proposed 0.000 [0.0, 0.0] vs best baseline rules_only 0.250 [0.1, 0.4091]; LLM only 0.321.
* Macro-F1 (normal + dangerous): proposed 0.792 [0.6788, 0.8905].
* Calibration: ECE 0.067, Brier 0.008; false alarm rate 0.000.
* Efficiency: 28.0 LLM calls / h (LLM only: 19.7).

## Ablations

| Variant | ΔMacro-F1 | ΔASR | ΔECE | ΔLLM calls / h |
| -- | -- | -- | -- | -- |
| proposed (A+B+C+E) | 0.792 | 0.000 | 0.067 | 28.0 |
| proposed_no_A | +0.097 | +0.536 | +0.056 | +0.0 |
| proposed_no_B | +0.139 | +0.000 | -0.004 | -3.4 |
| proposed_no_C | +0.208 | +0.000 | -0.035 | +0.0 |
| proposed_no_E | +0.000 | +0.000 | -0.007 | -8.3 |

## McNemar (per-scenario correctness, same scenarios)

| proposed vs | only proposed correct | only other correct | n | exact McNemar p |
| -- | -- | -- | -- | -- |
| rules_only | 13 | 9 | 84 | 0.523 |
| llm_only | 18 | 5 | 84 | 0.0106 |
| llm_pattern | 16 | 5 | 84 | 0.0266 |
| proposed_no_A | 18 | 5 | 84 | 0.0106 |
| proposed_no_B | 1 | 7 | 84 | 0.0703 |
| proposed_no_C | 0 | 9 | 84 | 0.00391 |
| proposed_no_E | 0 | 0 | 84 | 1 |

Figures: `asr.png`, `reliability.png`, `confusion_<config>.png`; per scenario: `per_scenario.csv`; reproducibility: `manifest.json` (git commit, config / prompt hashes, dataset hash) and the LLM cache.
