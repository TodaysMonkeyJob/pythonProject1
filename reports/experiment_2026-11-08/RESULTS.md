# Results — scripted-42 (2026-11-08)

> **Caveats:** the dataset is **scripted** (generated from templates by `experiments/scripted.py`) — it validates the harness, it is not the recorded evaluation set; **offline run: GPT took no part** (no call, no cache, no simulator) — only the configurations that decide without GPT; not run: llm_only, llm_pattern, proposed, proposed_no_A, proposed_no_B, proposed_no_C, proposed_no_E.

Evaluation split: 84 scenarios (calibration 36, seed 42); annotator agreement κ = 0.9647; LLM: none (offline).

## Metrics

| Configuration | Macro-F1 [95 % CI] | ASR [95 % CI] | False alarms | ECE | Brier | LLM calls / h | Cost / h, $ | Latency median / p95, s | Critical ≤ 5 s |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| rules_only | 1.000 [1.000, 1.000] | 0.250 [0.100, 0.409] | 0.000 | 0.074 | 0.032 | 0.0 | 0.0000 | 20.1 / 61.0 | 0.6087 |
| proposed_local | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 0.038 | 0.004 | 0.0 | 0.0000 | 20.1 / 61.0 | 0.6087 |

## McNemar (per-scenario correctness, same scenarios)

| proposed_local vs | only proposed_local correct | only other correct | n | exact McNemar p |
| -- | -- | -- | -- | -- |
| rules_only | 7 | 0 | 84 | 0.0156 |

Figures: `asr.png`, `reliability.png`, `confusion_<config>.png`; per scenario: `per_scenario.csv`; reproducibility: `manifest.json` (git commit, config / prompt hashes, dataset hash) and the LLM cache.
