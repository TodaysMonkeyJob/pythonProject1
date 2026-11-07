# Agents (B2 / TOD-49)

Article 2 adds AI agents to the system. Each agent has one role, and agents only talk to each other through typed
claims on the event bus. The core does not change when an agent is added. To add one, drop in a folder with an
`agent.yaml` and the code. The fuzzy agent (the Article 1 system) is always on. When every other agent is off, the
decisions are exactly the Article 1 decisions (`tests/test_agents.py` checks this on the 46 Article 1 scenarios).

```
evidence events ──► hfs module ──hfs_risk──► fuzzy agent ──claim "risk" (corridor)──┐
       │                                                                             ▼
       └──(data scope)──► video / audio / … agents ──claims──► situation board ──► decision round
                                                                                     │ deadline per agent
                                                                       validation agent: final decision
                                                                                     ▼
                                                     agent_decision (+ agent_operator_request)
```

## Where things are

| Path | What |
| -- | -- |
| `agents/<name>/agent.yaml` | the manifest: what the agent is, what it may read and say, its timeout |
| `agents/<name>/agent.py` | the code, a subclass of `agents.framework.base.Agent` |
| `agents/framework/manifest.py` | manifest schema (pydantic), the agent's evaluation file → weight |
| `agents/framework/base.py` | `Agent` interface, `Message`, `ValidationAgent` |
| `agents/framework/registry.py` | discovery (folders + entry points `iot_guard.agents`), on / off switches |
| `agents/framework/bus_adapter.py` | `Claim`, data-scope filter, situation board |
| `agents/framework/orchestrator.py` | `AgentHost`: routing, decision rounds, deadlines, decision record |
| `agents/module.py` | the `agents` module process (one thread per agent) |
| `agents/fuzzy`, `agents/validation` | fuzzy agent (`hfs_risk` → corridor claim), validation agent (final decision) |
| `agents/video`, `agents/audio` | TOD-47: actions / persons, meaning of speech — see `docs/article2_architecture.md` |
| `tools/agent_eval.py` | writes `agents/<name>/eval.yaml` (measured weights) from the evaluation reports |
| `agents/echo` | the template agent |

## Writing an agent

1. Copy `agents/echo/` to `agents/<name>/`. The folder name must equal `name:`.
2. Edit `agent.yaml`:

```yaml
name: falls
role: video                  # fuzzy | video | audio | gas | validation | explanation | rule_suggestion | other
version: 0.1.0               # stored with every claim
description: Falls from pose windows
entry: agent:FallAgent       # <module in the folder>:<class>
subscribes: [action, "claim:risk"]   # event types + claim types it wants
publishes: [risk_opinion]    # the only claim types it may publish
backend: {kind: local_model, models: [models/falls.joblib]}   # rules | local_model | cloud_model
resources: {ram_mb: 120, gpu: false}
data_scope:                  # the bus never delivers anything outside this
  events: [action]
  claims: [fuzzy]            # roles whose claims it may read ("*" = all)
timeout_s: 1.0               # answer a decision request within this, or the round goes on without you
enabled_default: false
eval_file: eval.yaml         # measured accuracy: its weight in validation
config: {threshold: 0.6}     # defaults; overridden by agents.settings.falls in config.yaml
```

3. Write the class:

```python
from agents.framework.base import Agent


class FallAgent(Agent):
    def load(self):                      # only while enabled: load models here
        import joblib
        self.model = joblib.load(self.manifest.backend.models[0])

    def unload(self):                    # disabled: drop every reference, the memory is freed
        self.model = None

    def on_message(self, message):
        if message.kind == "decision_request":         # a round is open for message.location_id
            p = self.probability(message.location_id)
            return [self.claim("risk_opinion", message.location_id, {"risk": p}, confidence=0.9,
                               round_id=message.round_id)]
        if message.kind == "event":                    # message.event: an action event in scope
            ...
        return None
```

The agent shows up on the dashboard page **Агенти** after a restart. No other file changes.

### Messages an agent receives

| `message.kind` | Contents | When |
| -- | -- | -- |
| `event` | `message.event` (an `Event`) | an event type the agent subscribed to and that is inside its `data_scope.events` |
| `claim` | `message.claim` | another agent's claim of a subscribed type (`claim:<type>`) from a role inside `data_scope.claims` |
| `decision_request` | `round_id`, `location_id`, `corridor` (the fuzzy claim, if `fuzzy` is in its claim scope), `board` (live claims in scope) | a decision round opened; answer within `timeout_s` |

`tick(now)` runs about every 0.1 s for periodic work. `health()` returns `{status, detail}` for the dashboard.

### Claims

```json
{"claim_id": "…", "agent": "falls", "role": "video", "version": "0.1.0", "location_id": "kitchen",
 "ts": "2026-11-07T10:00:00.000Z", "claim": "risk_opinion", "value": {"risk": 0.8}, "confidence": 0.9,
 "interval": null, "evidence_ids": ["<event ids it used>"], "round_id": "…"}
```

`claim` is the claim type and `value` its content. The validation agent counts `risk_opinion` claims with
`value.risk` in [0, 1]. Their `value.kind` is either `at_least` ("what I saw is at least this risky", which only
raises) or `estimate` (the default: an overall risk, which may lower only when it comes from video / gas). Every claim goes on the bus as an `agent_claim` event, so it is logged in
`logs/<date>/events.jsonl` together with the agent's version.

## Rules the framework enforces

- **Data scope.** An agent receives an event type only if it subscribed to it *and* it is in `data_scope.events`. A
  manifest that subscribes to something outside its scope is rejected. Claims and the situation board are filtered
  the same way, using `data_scope.claims`. For example, an audio agent never sees face data.
- **Boundaries.** Agents only return claims. A claim type outside `publishes` is dropped and counted as an error.
  Agents get the configuration read-only and never change rules, thresholds or configuration.
- **Final decision.** Only the validation agent makes it, always from the fuzzy agent's corridor. If no validation
  agent is on, the corridor is the decision (`fuzzy_only`).
- **Weights.** An agent's weight is its measured accuracy from `eval_file` (`accuracy`, `n`, `source`, `date`). With
  no file the agent is "unmeasured": weight 0, so its claims are logged but not counted. Weights are never typed by
  hand.
- **Timeouts.** Each agent has its own deadline per round. An answer after it is `late` and is ignored for that
  decision, but its claims still reach the board. An agent that never answers is `missing`. If a new corridor for the
  same location arrives first, the absent agents of the old round are `superseded`. All of this is listed in the
  decision record.
- **Switches.**
  - Precedence, lowest first: `enabled_default` in the manifest, `agents.enabled_agents` in config.yaml, then the
    dashboard switch, which is saved in `agents.state_path` (git-ignored) and survives restarts.
  - `required` agents (the fuzzy agent) cannot be switched off.
  - `cloud_model` agents never start in `mode: autonomous`.
  - At most one agent of role `fuzzy` / `validation` is on at a time.
  - Code is imported only when the agent is enabled. Disabling calls `stop()` and drops the object.
- **Broken plug-ins.** An invalid manifest or an agent that fails to start is listed on the page with the error. The
  other agents run normally.

## Decision round

1. The fuzzy agent turns each `hfs_risk` into a `risk` claim: risk, level, and the IT2 corridor `[y_l, y_r]`. This
   opens a round for that location.
2. Every enabled contributing agent gets a `decision_request` (all roles except fuzzy and validation, unless the
   manifest sets `contributes`).
3. The round closes when everyone has answered or at the last deadline. The validation agent then decides (see
   `agents/validation/agent.py`):
   - no weighted opinion → `fuzzy_only` (Article 1);
   - the agents' weighted risk inside the corridor → `agree`;
   - an `at_least` floor above the fuzzy risk raises the risk to it (`raised`), or goes to the operator when the
     floor is above the corridor;
   - below the corridor → `deescalation_confirmed` only if video / gas opinions support it; otherwise
     `deescalation_rejected` and the fuzzy risk stays (a spoken "everything is fine" never lowers it);
   - above the corridor → `escalation_review`: risk = `y_r` plus an `agent_operator_request` event.
4. An `agent_decision` event is published with:
   - round_id, risk, level, interval, branch, reasons, decided_by;
   - corridor, `agents` (answered / late / missing / error / superseded), missing;
   - claims (the ids used), versions (of every agent that took part), evidence_ids, latency_ms.

In `mode: multiagent` (TOD-52, [`modes.md`](modes.md)) the location state follows `agent_decision`; in
`mode: autonomous` there are no rounds and the `hfs_risk` decides. A decision falls back to the fuzzy corridor
(`mode_effective: autonomous`) when the validator, the network or a required agent is missing.

## Dashboard

The page `#/agents` («Агенти») shows every discovered agent:
- role, version, backend, status;
- latency, timeout, claims, errors and late answers;
- measured accuracy and the last claims;
- an on / off switch (`POST /api/agents/{name}` → `agent_control` event → agents module).

It also lists the manifests that failed to load and the latest decisions. The status comes from `agent_status`
events, published every `agents.status_interval_sec`; these are not logged.
