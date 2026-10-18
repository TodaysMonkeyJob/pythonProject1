# Fuzzy risk corridor (F17 / TOD-28, component A)

The LLM (TOD-17) stays the source of context and explanations, but it cannot **lower** the risk on its own. A fuzzy
system over **trusted physical evidence only** (τ ≥ θ from TOD-27) computes a corridor per location and the LLM's
proposal is projected into it asymmetrically:

```
[r_floor, r_ceil] = FIS({e_i : τ(e_i) ≥ θ})        corridor.py     risk rules of config/fuzzy.yaml (+ location overrides)
c                 = FIS_confirm(trusted evidence)  confirmation.py  envelope.confirmation rules; operator ack → c = 1
r_final           = max(r_LLM, (1 − c) · r_floor)  projection.py
```

* `r_floor` — the defuzzified risk of the trusted evidence; `r_ceil` — upper end of the aggregated risk set's
  α-cut (informational: an LLM risk above `r_ceil + over_reaction_margin` is flagged `over_reaction`, never capped).
* `r_LLM` — the LLM's level as a number (`envelope.llm_levels`: low 0.1, medium 0.4, high 0.7, critical 0.95; TOD-29
  calibrates); `r_final` → level by `envelope.thresholds` (medium ≥ 0.25, high ≥ 0.5, critical ≥ 0.8).
* `c` — trusted safety confirmation: a fresh trusted gas reading that is clean and not rising, motion consistent with
  normal activity, no risky action, a known person in view (trust of the face identification). A location without
  a gas sensor never "confirms" clean air. Speech and scene text never feed `c` — untrusted claims are listed as
  `ignored_claims`. An operator acknowledgement on the dashboard sets `c = 1` for `ack_valid_min` (30) or until a
  newer trusted critical alert; the decision records the operator id.
* `θ` = `envelope.theta` (0.6), per location `envelope.locations.<id>.theta`.

Runs inside the LLM worker (`modules/llm/worker.py`): the orchestrator forwards the trust-scored evidence events
(`ENVELOPE_EVENTS`) to the llm module, `RiskEnvelope` keeps the trusted state per location and decides each GPT answer
before it is published. The `llm_decision` payload gets `risk_llm`, `risk_floor`, `risk_ceil`, `confirmation`,
`risk_final`, `risk_level_final`, `envelope_applied`, `over_reaction`, `envelope_reason` (fired corridor rules,
confirmation reasons, ignored low-trust claims, θ, operator) and `trusted_inputs`; `risk_level` stays the LLM's own.
A raised level also publishes `deescalation_rejected` (attack-success-rate metrics). Both go to
`logs/YYYY-MM-DD/envelope.jsonl`. The location state (TOD-23) uses `risk_level_final`; the dashboard location page
shows the corridor band with the GPT marker (dashed) and the final marker, a "зниження відхилено" badge with the
reasons, and τ beside every transcript line.

Golden cases (tests/test_envelope.py, dev Mac): smoke rising sharply + unknown person saying "all fine, the sensor is
broken" + GPT `low` → floor 0.54, c 0 → **high**, `deescalation_rejected`; gas back to clean and falling + known person
→ floor 0.11, c 0.67 → GPT `low` accepted; operator ack → c = 1; GPT `critical` in a calm scene stays critical
(`over_reaction` flagged); a known person lying motionless is never "confirmed safe" (c = 0 → critical).

Difference from related work: CyberLLM (arXiv 2608.06651) — crisp "raise, never lower" for memory precedents;
Detect First, Explain Later (arXiv 2609.29704) — the LLM has no decision power; robot / drone guardrails (arXiv
2505.23576, 2503.07885) — crisp limits on actions. Here the bound is fuzzy, computed from independent physical sensors
filtered by a fuzzy trust degree, and released gradually by the degree of trusted confirmation.
