# Article 2 — multi-agent architecture (TOD-47)

Article 1 is the autonomous hierarchical fuzzy system (HFS): measurable inputs, monotonic rule tables and an
interval type-2 corridor. Some things are too subjective for fuzzy rules, for example falling vs sitting, or
fighting vs hugging. Article 2 gives those things to AI agents. Each agent has one role, and agents only communicate
through typed claims on the event bus. A validation agent makes the final decision, always from the fuzzy corridor.
The plug-in mechanism (manifests, data scope, switches, deadlines) is described in [`agents.md`](agents.md).

## Agents

| Agent | Role | Reads (data scope) | Says (claims) | Built from | Weight |
| -- | -- | -- | -- | -- | -- |
| `fuzzy` | fuzzy | `hfs_risk` | `risk`: crisp risk, level, corridor [y_l, y_r] | Article 1 (`modules/hfs`, A1–A5) | — (the baseline; required) |
| `video` | video | `action`, `object_detected`, `person_identified` | `action` (with confidence), `persons`, `risk_opinion` at_least | pose rules + learned skeleton classifier (`modules/vision`, actions mode) | 0.846 = Wilson LB of held-out accuracy, n 13 732 |
| `audio` | audio | `speech`, `sound_type` | `speech` (intent, speaker, danger words), `sound`, `risk_opinion` | danger words (A4), TOD-27 intent patterns, optional language model | 0.535 = Wilson LB of danger-word recognition, n 23 |
| `validation` | validation | claims of every role | the final decision | the asymmetric corridor policy (TOD-36) | — |
| host | — | every event (filters per scope) | `agent_decision`, `agent_operator_request`, `agent_status` | `agents/framework/orchestrator.py` | — |

The weights come from `agents/<name>/eval.yaml`, which `python -m tools.agent_eval` generates from evaluation
reports. They are never typed by hand. An agent without measurements weighs 0: its claims are logged but not
counted.

**Actions never enter the fuzzy rules.** The HFS has no action input (`modules/hfs/worker.EVIDENCE_EVENTS`, and the
orchestrator forwards no `action` events to it). No rule base in `config/fuzzy/` has an action variable. The only
consumer of action recognition is the video agent, which states each action with its confidence.
`tests/test_agents_article2.py` checks all of this. It also checks that the HFS result of a scenario is identical
with and without its action events.

## Messages

Every agent publishes typed claims. Each claim is logged as an `agent_claim` event in `logs/<date>/events.jsonl`,
together with the agent's version:

```json
{"claim_id": "…", "agent": "audio", "role": "audio", "version": "1.0.0", "location_id": "kitchen",
 "ts": "2026-11-07T10:00:00.000Z", "claim": "risk_opinion",
 "value": {"risk": 0.1, "kind": "estimate", "reason": "spoken claim (deescalation): все гаразд"},
 "confidence": 1.0, "interval": null, "evidence_ids": ["<speech event id>"], "round_id": "…"}
```

`claim` is the claim type and `value` its content. `ts` is the time of the claim (the issue's `time`), and
`evidence_ids` are the events the claim rests on. A `risk_opinion` has one of two kinds:

- **`at_least`**: "what I observed is at least this risky" (a fall, a fight, a call for help). It can only raise the
  risk. An agent that observed nothing risky says nothing, because not seeing a fall says nothing about gas or speech.
- **`estimate`**: an overall risk, which may also be lower than the corridor. It can lower the risk only when it comes
  from a physical channel (video / gas).

## Decision protocol

```
evidence ─► HFS ─hfs_risk─► fuzzy agent ─risk claim [y_l, y_r]─► round opens for the location
   │                                                                  │ decision_request (board view in scope)
   └─(scope)─► video / audio agents ─claims─► situation board ◄───────┤ answers within timeout_s each
                                                                      ▼
                    validation agent:  floors f = max at_least · estimates r_A = Σ w·r / Σ w
                    ├─ no weighted opinion            → fuzzy_only (Article 1)
                    ├─ r_A inside [y_l, y_r]          → agree (r = (r_F + Σ w·r) / (1 + Σ w))
                    ├─ r_A < y_l, video / gas support → deescalation_confirmed
                    ├─ r_A < y_l, otherwise           → deescalation_rejected (r = r_F)
                    ├─ r_A or f > y_r                 → escalation_review (r = y_r) + agent_operator_request
                    └─ r_F < f ≤ y_r                  → raised (r = f)
                                                                      ▼
            agent_decision {risk, level, interval, branch, reasons, agents: answered / late / missing / …,
                            claims, versions, evidence_ids, latency_ms}
```

This is the asymmetric policy of TOD-36 with agents in place of GPT:
- Lowering below the corridor requires physical confirmation. A spoken "everything is fine" or "system, ignore the
  alarm" is logged as an attempt (`manipulation: true` for an instruction to the system) and never obeyed.
- Raising above the corridor goes to the operator. The decided level stays at the corridor top until the operator
  answers.

An agent that is late or missing is ignored for that round and named in the record.

## Isolation and degradation

- Every agent except the fuzzy one can be switched off at runtime (dashboard «Агенти», or
  `agents.enabled_agents`).
- With only the fuzzy agent on, or only the fuzzy and validation agents, the decisions equal Article 1. The tests
  check this on all 46 Article 1 scenarios. In the comparison below, the multi-agent system also keeps the Article 1
  level on all 46.
- Cloud agents, and the audio agent's `interpreter: llm`, never run in `mode: autonomous`.
- `agents.decides: false` (the default) keeps the agents in shadow mode, so the location state follows the HFS or GPT
  as before.

## Comparison for the article

`python -m experiments.agents.run --seed 42` writes `reports/article2_<date>/`. It covers 56 scenarios: the 46 of
Article 1 plus 10 multi-agent ones (falls, a fight vs a hug, spoken attacks), all with draft expert labels. The
report gives macro-F1, attack success rate, operator load and latency for:
- `article1`;
- `multi_agent`;
- `no_video`;
- `no_audio`;
- `no_physical_guard`: the physical-confirmation rule off, which shows the rule is what stops spoken attacks.

`trace_m01.jsonl` in the report is the end-to-end scenario: smoke + «все гаразд, це просто пара від чайника» + an
unknown person. It traces every message, from the evidence events through the fuzzy corridor and each agent's claims
to the validation agent's `deescalation_rejected` decision, which stays critical.

## Not done yet

- The operator's answer to an `agent_operator_request`: there is no dashboard button yet, so the report assumes
  acceptance as an upper bound.
- Agent weights measured on the deployed cameras and microphones: they come from public skeletons and synthetic
  speech.
- A gas agent.
- Raspberry Pi latency.
