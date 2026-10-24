| Question | Variant | Macro-F1 (v2 → variant) | ASR | ECE | LLM calls / h | Extra | Answer |
| -- | -- | -- | -- | -- | -- | -- | -- |
| Does type 2 add value? (IT2 vs type-1 corridor) | proposed_v2_type1 | 1.000 → 1.000 | 0.000 → 0.000 | 0.034 → 0.039 | 17.0 → 19.7 | mean width 0.0353 vs 0.1001; corr(w, u) 0.2588 | same decisions on every scenario |
| Does measured trust add value? (registry vs TOD-27 constants) | proposed_v2_static_trust | 1.000 → 1.000 | 0.000 → 0.000 | 0.034 → 0.040 | 17.0 → 15.6 | sensor failure: ASR 0.0 vs 0.0, false alarms 0.75 vs 0.8214 | same decisions on every scenario |
| Does the two-sided corridor add value? (vs floor-only max-projection) | proposed_v2_floor_only | 1.000 → 1.000 | 0.000 → 0.000 | 0.034 → 0.046 | 17.0 → 17.0 | escalation reviews / h 0.54 (right 0.3333) vs 0.0 | same decisions on every scenario |
| Why not fully local? (corridor + local claim detector, no GPT) | proposed_v2_local | 1.000 → 1.000 | 0.000 → 0.000 | 0.034 → 0.039 | 17.0 → 0.0 | GPT calls / h 17.0 vs 0.0 | same decisions on every scenario |
