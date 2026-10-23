# TOD-39 (F28): components C and E after the fixes

Harness: `python -m experiments.run --scripted --llm azure --seed 42` on scripted-42 (120 scenarios, 36 calibration /
84 evaluation), gpt-4.1-mini, temperature 0.2. 371 new GPT calls on 2026-10-23 (new prompt and payloads); the final
run is answered entirely from `experiments/cache/llm_azure.sqlite` (838 / 838 hits). Rule base `a0ab9c218631`
(approved through `tools.rules`), thresholds refit by `tools.fit_thresholds` on the recorded answers. The dataset is
scripted: these numbers validate the pipeline, not the recorded evaluation set (TOD-40).

| Configuration | Macro-F1 | ASR | ECE | LLM calls / h |
| -- | -- | -- | -- | -- |
| llm_only (reference) | 0.671 | 0.321 | 0.151 | 19.7 |
| proposed (v2.1 snapshot, new gating) | **1.000** | **0.000** | 0.034 | **17.0** |
| proposed_v2 (v2 snapshot) | 1.000 | 0.000 | 0.033 | 17.0 |
| proposed_no_C (v1 digest, old prompt) | 1.000 | 0.000 | 0.034 | 17.0 |
| proposed_no_E (no gating) | 1.000 | 0.000 | 0.018 | 19.7 |

Before (ablation of 2026-10-22 quoted in TOD-39): proposed with C 0.783 vs 1.000 without; gating 28.0 vs 19.7 calls / h.

## C — snapshot, prompt, calibration

* Medium scenarios are no longer escalated to high: F1 with C = without C (criterion: ≥ without C − 0.02). On this
  set v1, v2 and v2.1 give the same decisions, so the default (`llm.snapshot_version: v2.1`) is not yet supported by
  a measurable gain — the recorded set of TOD-40 has to decide it.
* The fix that mattered: the medium / high rubric in the prompt (risk scale from the rule base) and no more inflated
  calibrated risk. The constrained calibration (centres in their risk term, supports ≥ 0.2, Huber loss) was fitted
  on 23 of 36 calibration scenarios (2 answers rejected by the corridor as manipulated, 11 without a GPT answer at
  the end): in-sample Brier 0.0010 / ECE 0.0155, leave-one-out Brier 0.0022 / ECE 0.0130, fixed mapping Brier
  0.0019 / ECE 0.0366. Leave-one-out Brier is worse than the fixed mapping, so the fixed mapping is used (no
  `config/calibration.yaml` written); the constraints hold. Report: `calibration_loo.json`.

## E — gating

* 17.0 calls / h vs 19.7 without gating (−14 %) with ASR 0 and the same F1. Reasons: only `critical_event` and an
  unseen critical floor bypass the interval (repeated untrusted claims and re-asked critical floors used to: 74 of 156 calls in the simulator replay);
  certain, low scenes are left to the corridor; the threshold (0.68) is fitted with `calibration.call_cost = 0.2`.
* Cost: ECE 0.034 vs 0.018 without gating — fuzzy-only decisions take the corridor's L, which is less well
  calibrated than GPT's answer on the scenarios it skips.
* Agreement threshold h_min refit on recorded answers: 0.4 → 0.85.
