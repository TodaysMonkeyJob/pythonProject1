# Results — scripted-42 (2026-10-19)

> **Caveats:** the dataset is **scripted** (generated from templates by `experiments/scripted.py`) — it validates the harness, it is not the recorded evaluation set.

Evaluation split: 84 scenarios (calibration 36, seed 42); annotator agreement κ = 0.9647; LLM: azure / gpt-4.1-mini, temperature 0.2.

## Metrics

| Configuration | Macro-F1 [95 % CI] | ASR [95 % CI] | False alarms | ECE | Brier | LLM calls / h | Cost / h, $ | Latency median / p95, s | Critical ≤ 5 s |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| rules_only | 0.706 [0.661, 0.739] | 0.250 [0.100, 0.409] | 0.000 | 0.090 | 0.036 | 0.0 | 0.0000 | 0.1 / 61.0 | 1.0 |
| llm_only | 0.671 [0.507, 0.809] | 0.321 [0.147, 0.518] | 0.000 | 0.151 | 0.075 | 19.7 | 0.0066 | 1.84 / 62.73 | 0.5217 |
| llm_pattern | 0.671 [0.507, 0.809] | 0.250 [0.094, 0.423] | 0.000 | 0.147 | 0.073 | 19.7 | 0.0065 | 1.83 / 62.94 | 0.5217 |
| proposed | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.036 | 0.003 | 28.0 | 0.0139 | 1.78 / 62.68 | 1.0 |
| proposed_no_A | 0.889 [0.810, 0.959] | 0.250 [0.094, 0.429] | 0.000 | 0.140 | 0.048 | 28.0 | 0.0134 | 1.68 / 2.0 | 1.0 |
| proposed_no_B | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.038 | 0.003 | 24.6 | 0.0118 | 1.81 / 62.95 | 1.0 |
| proposed_no_C | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.032 | 0.003 | 28.0 | 0.0094 | 1.79 / 62.95 | 1.0 |
| proposed_no_E | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.029 | 0.002 | 19.7 | 0.0095 | 1.78 / 62.68 | 1.0 |

## Summary

* Attack success rate: proposed 0.000 [0.0, 0.0] vs best baseline rules_only 0.250 [0.1, 0.4091]; LLM only 0.321.
* Macro-F1 (normal + dangerous): proposed 1.000 [1.0, 1.0].
* Calibration: ECE 0.036, Brier 0.003; false alarm rate 0.000.
* Efficiency: 28.0 LLM calls / h (LLM only: 19.7).

## Ablations

| Variant | ΔMacro-F1 | ΔASR | ΔECE | ΔLLM calls / h |
| -- | -- | -- | -- | -- |
| proposed (A+B+C+E) | 1.000 | 0.000 | 0.036 | 28.0 |
| proposed_no_A | -0.111 | +0.250 | +0.103 | +0.0 |
| proposed_no_B | +0.000 | +0.000 | +0.002 | -3.4 |
| proposed_no_C | +0.000 | +0.000 | -0.005 | +0.0 |
| proposed_no_E | +0.000 | +0.000 | -0.007 | -8.3 |

## McNemar (per-scenario correctness, same scenarios)

| proposed vs | only proposed correct | only other correct | n | exact McNemar p |
| -- | -- | -- | -- | -- |
| rules_only | 13 | 0 | 84 | 0.000244 |
| llm_only | 22 | 0 | 84 | 4.77e-07 |
| llm_pattern | 20 | 0 | 84 | 1.91e-06 |
| proposed_no_A | 14 | 0 | 84 | 0.000122 |
| proposed_no_B | 0 | 0 | 84 | 1 |
| proposed_no_C | 0 | 0 | 84 | 1 |
| proposed_no_E | 0 | 0 | 84 | 1 |

Figures: `asr.png`, `reliability.png`, `confusion_<config>.png`; per scenario: `per_scenario.csv`; reproducibility: `manifest.json` (git commit, config / prompt hashes, dataset hash) and the LLM cache.
