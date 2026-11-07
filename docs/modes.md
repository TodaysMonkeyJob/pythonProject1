# Modes: autonomous (Article 1) and multi-agent (Article 2) — M1 / TOD-52

| Mode | What runs | Who makes the final decision |
| -- | -- | -- |
| `autonomous` — «Автономний (стаття 1)» | gas, camera and audio subsystems + the top level; no agents, no GPT, no network | the top-level fuzzy system (`hfs_risk`) |
| `multiagent` — «Мультиагентний (стаття 2)» | the same fuzzy system (as the fuzzy agent) + the enabled AI agents; the GPT pipeline runs but only logs its `llm_decision` | the validation agent, always within the fuzzy corridor (`agent_decision`) |

`hybrid`, the name used by TOD-41, is read as `multiagent`. The fuzzy system always runs in both modes, so a switch
never leaves a location without a decision.

## How the mode is chosen at start

The first of these that is set wins:

1. `python main.py --mode autonomous|multiagent`
2. the environment variable `IOT_MODE`
3. the choice saved from the dashboard: `data/mode.json`, together with the per-location overrides (git-ignored)
4. `mode` in `config.yaml` (shipped: `autonomous`)

A config without the `mode` key (library use, tests) means `multiagent`, which is the TOD-41 default. The orchestrator
logs which source it used.

**When started autonomous, the cloud paths are off for the whole run.** The LLM worker, GPT gating, the risk envelope
and GPT rule suggestions are not started or imported. Whisper runs locally and the trust scorer is not created. A
later runtime switch to multiagent starts the LLM worker, and the aggregator queues snapshots for it again. The
following stay as they were set at start until a restart:
- the Whisper backend;
- the trust scorer in the router.

## Switching at runtime

- **Where:** the mode chip in the dashboard header («режим: …») opens the switch dialog. Choose the mode for the whole
  system, or for one location (an override, e.g. the warehouse autonomous and the office multi-agent). Every change
  is confirmed first.
- **Who:** only the admin can switch.
  - The admin is the `DASHBOARD_USER` account. Without credentials configured, the local operator on 127.0.0.1 is
    the admin.
  - `DASHBOARD_VIEWERS="name:password,…"` in `.env` adds read-only accounts. They see the mode and get 403 on a
    switch.
- **What happens:**
  1. `POST /api/mode {mode, location_id?, confirm: true}` publishes a `mode_control` event.
  2. The orchestrator saves the choice in `data/mode.json` and appends `{at, by, scope, location_id, from, to}` to
     `data/mode_audit.jsonl`. It also starts or stops the LLM worker (it runs while any location is multiagent).
  3. The orchestrator broadcasts `mode_changed`. The dialog shows the last 10 switches.
- **When it takes effect:** every module applies the switch from its next decision cycle.
  - **HFS:** `hfs_risk` carries `mode_requested`. It has its level's severity for an autonomous location, and is
    `info` otherwise.
  - **Agents:** an autonomous location gets no decision round, and agents receive nothing for it. While no location
    is multiagent, the contributing agents are stopped and their models unloaded. Switching back starts the enabled
    ones.
  - **Locations:** follow the decider of each location.
  - **Aggregator:** queues snapshots for GPT only while some location is multiagent.

Each `hfs_risk` cycle produces exactly one deciding record: the `hfs_risk` itself for an autonomous location, or the
one `agent_decision` of its round for a multiagent one. A switch never leaves a cycle undecided or decided twice
(`tests/test_modes.py` switches twice during a running scenario).

## Automatic fallback

In multiagent mode a decision falls back to the fuzzy system (`branch: fallback`, `mode_effective: autonomous`,
`fallback_reason`) when:

- the validation agent is switched off («агент валідації вимкнений») or unhealthy («агент валідації недоступний»);
- the network is down («немає мережі»): a TCP probe to `agents.network_probe`, at most every 30 s, run only while some
  location is multiagent;
- an agent listed in `agents.required` did not answer within its deadline («не відповіли вчасно: …»);
- an enabled agent is not healthy yet («агенти запускаються: …»), e.g. right after a switch while its models load.

If the agents process itself goes silent, the locations module falls back too. That happens when no `agent_decision`
arrives within `locations.agents_fallback_sec` (10 s) of an `hfs_risk`; the `hfs_risk` then decides, with the reason
«агенти не відповідають».

While any location runs on fallback, the dashboard shows the banner «Працює автономний режим: <локація> — <причина>»,
and the location card shows the reason. The next decision returns to multiagent by itself once the cause is gone.

## Marking decisions

- **`agent_decision`** carries:
  - `mode_requested`, `mode_effective` (they differ on fallback) and `fallback_reason`;
  - `contributing_agents` (the agents that answered);
  - the agents' `versions`;
  - `rules_version`: the gas / camera / audio / top rule base hashes, from the corridor.
- **`hfs_risk`** carries `mode_requested` and `rules_version`.
- **The location summary** carries `mode: {requested, effective, reason}`.

The Article 2 comparison (`python -m experiments.agents.run`) runs `article1` as `mode: autonomous`, where the
`hfs_risk` decides. It splits every configuration's metrics by `mode_effective`, so a multi-agent decision that fell
back counts under autonomous, and Article 1 and Article 2 results are never mixed.
