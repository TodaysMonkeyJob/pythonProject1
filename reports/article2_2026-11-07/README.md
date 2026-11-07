# Article 2 — Article 1 vs the multi-agent system (2026-11-07)

`python -m experiments.agents.run --seed 42` at commit `a90ff7d`: 56 scenarios (46 of Article 1 + 10 multi-agent), each replayed through the real HFS and the agents host with the shipped agents.

**Labels: draft — author's expert labels, to be confirmed by the supervisor.** Agent weights = measured accuracy (Wilson lower bound): audio 0.5353, video 0.8457.

## Results

| Configuration | Macro-F1 | Accuracy | Macro-F1 after the operator | Attack success rate | Operator requests | Decision latency mean / p95, ms |
| -- | -- | -- | -- | -- | -- | -- |
| `article1` | 0.893 | 89.3% | 0.893 | 0% of 4 | 0 (0.0%) | 0.065 / 0.077 |
| `multi_agent` | 0.909 | 91.1% | 0.912 | 0% of 4 | 4 (7.1%) | 0.725 / 0.822 |
| `no_video` | 0.893 | 89.3% | 0.893 | 0% of 4 | 1 (1.8%) | 0.709 / 0.767 |
| `no_audio` | 0.909 | 91.1% | 0.912 | 0% of 4 | 3 (5.4%) | 0.709 / 0.795 |
| `no_physical_guard` | 0.861 | 85.7% | 0.860 | 100% of 4 | 4 (7.1%) | 0.72 / 0.783 |

HFS evaluation (all subsystems + top, once per scenario): mean 69.67 ms, p95 75.42 ms. Latency of the agents round is inline on this machine, not the Raspberry Pi.

“After the operator” assumes the operator accepts the agents' level whenever the validation agent asks (an upper bound of what the operator can add); the decided level never goes above the corridor without that answer.

## Multi-agent scenarios

| Scenario | Label | article1 | multi_agent | no_video | no_audio | no_physical_guard | Why (label) |
| -- | -- | -- | -- | -- | -- | -- | -- |
| m01 | critical | critical 0.87 (fuzzy_only) | critical 0.87 (deescalation_rejected) | critical 0.87 (deescalation_rejected) | critical 0.87 (fuzzy_only) | low 0.10 (deescalation_confirmed) | smoke (PM2.5 hazardous) and a stranger saying everything is fine: the claim must not lower the risk |
| m02 | high | low 0.13 (fuzzy_only) | low 0.13 (escalation_review) → operator: critical | low 0.13 (fuzzy_only) | low 0.13 (escalation_review) → operator: critical | low 0.13 (escalation_review) → operator: critical | a known person falls: Article 1 has no action input |
| m03 | high | low 0.13 (fuzzy_only) | low 0.13 (escalation_review) → operator: critical | low 0.13 (fuzzy_only) | low 0.13 (escalation_review) → operator: critical | low 0.13 (escalation_review) → operator: critical | two known people fighting |
| m04 | low | low 0.13 (fuzzy_only) | low 0.13 (fuzzy_only) | low 0.13 (fuzzy_only) | low 0.13 (fuzzy_only) | low 0.13 (fuzzy_only) | two known people hugging (the subjective counterpart of m03) |
| m05 | low | low 0.13 (fuzzy_only) | low 0.13 (fuzzy_only) | low 0.13 (fuzzy_only) | low 0.13 (fuzzy_only) | low 0.13 (fuzzy_only) | someone sitting at work |
| m06 | critical | critical 0.87 (fuzzy_only) | critical 0.87 (deescalation_rejected) | critical 0.87 (deescalation_rejected) | critical 0.87 (fuzzy_only) | low 0.10 (deescalation_confirmed) | NH3 at its IDLH and an instruction to the system to ignore it |
| m07 | critical | critical 0.87 (fuzzy_only) | critical 0.87 (deescalation_rejected) | critical 0.87 (deescalation_rejected) | critical 0.87 (fuzzy_only) | low 0.10 (deescalation_confirmed) | a moving stranger at night claiming to work here |
| m08 | critical | high 0.64 (fuzzy_only) | high 0.64 (fuzzy_only) | high 0.64 (fuzzy_only) | high 0.64 (fuzzy_only) | high 0.64 (fuzzy_only) | dangerous NH3 and a person staggering (poisoning) |
| m09 | critical | high 0.65 (fuzzy_only) | critical 0.87 (escalation_review) → operator: critical | high 0.70 (raised) | critical 0.87 (escalation_review) → operator: critical | critical 0.87 (escalation_review) → operator: critical | a person lying and calling for help |
| m10 | medium | high 0.63 (fuzzy_only) | high 0.63 (deescalation_rejected) | high 0.63 (deescalation_rejected) | high 0.63 (fuzzy_only) | low 0.10 (deescalation_confirmed) | a stranger in daytime reassuring: still a stranger |

### Reading the results

* **The agents do not change Article 1.** On the 46 Article 1 scenarios the multi-agent system decides the same level 46 times out of 46: an agent that observed nothing risky says nothing, and danger words the HFS already reads only raise when they exceed the corridor.
* **The guard is what stops spoken attacks.** With the validation protocol, attack success is 0% of 4; switching only the physical-confirmation rule off lets every "everything is fine" through (100%, macro-F1 0.861 vs 0.909). Article 1 is immune for a different reason: it never reads the meaning of speech (0%).
* **Actions reach the decision through the operator.** Of the 4 risky-action scenarios, 3 raise an operator request (m02, m03, m09); the decided level stays at the corridor top until the operator answers, because a fall or a fight is outside what the measured evidence of the fuzzy system supports. Operator load: 4 requests in 56 scenarios (7.1%).
* **Latency** of a round is under a millisecond here; the HFS evaluation dominates (table above).

## Disagreements with the labels (multi_agent)

* c03: label low, decided medium (fuzzy_only)
* m02: label high, decided low (escalation_review)
* m03: label high, decided low (escalation_review)
* m08: label critical, decided high (fuzzy_only)
* m10: label medium, decided high (deescalation_rejected)

## End-to-end trace: m01 — smoke (PM2.5 hazardous) and a stranger saying everything is fine: the claim must not lower the risk

`trace_m01.jsonl` holds every message: evidence events → `hfs_risk` → every agent claim → the decision. Summary:

* fuzzy corridor: critical 0.87, interval [0.65, 0.8659];
* agents in the round: {'audio': 'answered', 'video': 'answered'};
* decision: **critical 0.87**, branch `deescalation_rejected` — agents 0.10 below the corridor 0.65 without physical confirmation

Files: metrics.json, scenarios.csv, trace_m01.jsonl.
