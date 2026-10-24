"""
System configurations compared in the thesis (F20 / TOD-31) — feature flags over the existing modules.

| name | what decides |
| -- | -- |
| rules_only | deterministic rules: gas alerts / thresholds, lying, smoking, unknown person at night |
| llm_only | GPT on the v1 text digest with the original prompt (TOD-17 as built) |
| llm_pattern | llm_only + a pattern filter that drops transcripts / scene text matching injection phrases |
| proposed | A + B + C + E: trust (TOD-27), corridor (TOD-28), tagged view + calibration (TOD-29), gating (TOD-30) |
| proposed_no_A / _no_B / _no_C / _no_E | ablations: one component off |

Without B every event is treated as trusted (trust 1, no claim detection); without C GPT gets the v1 digest with the
original prompt and the fixed level mapping; without A the LLM's level is final; without E every snapshot goes
to GPT. F28 (TOD-39): C sends snapshot v2.1 (``Config.snapshot``); ``proposed_v2_view`` keeps the plain v2 view for
the v2 → v2.1 comparison (not part of ``--all``).

Experiment v2 (F29 / TOD-40, ``--suite v2``) — the new design (IT2 corridor + trust registry + two-sided corridor +
the TOD-39 C / E) and one question per variant:

| name | what it isolates |
| -- | -- |
| proposed_v2 | the new design (same flags as proposed) |
| proposed_v2_type1 | type-1 corridor (``envelope.floor: mom``, trust uncertainty u ignored) — does type 2 add value? |
| proposed_v2_static_trust | TOD-27 constant trust instead of the registry (``experiments.variants``) |
| proposed_v2_floor_only | the F17 max-projection (lower bound only, no fuzzy intersection / upper bound) |
| proposed_v2_no_B / _no_C / _no_E | ablations as before |
| proposed_v2_local | no GPT: the corridor's L decides every snapshot, language goes to the local claim detector (B) |
"""
from dataclasses import asdict, dataclass, replace
from typing import Dict, List


@dataclass(frozen=True)
class Config:
    name: str
    llm: bool = True              # GPT takes part
    pattern_filter: bool = False  # baseline injection filter on transcripts
    trust: bool = False           # B — fuzzy trust in evidence
    envelope: bool = False        # A — fuzzy risk corridor
    view: bool = False            # C — fuzzified trust-tagged view, spotlighting prompt, calibrated r_LLM
    snapshot: str = "v2.1"        # C's view version (F28 / TOD-39: v2.1 = v2 + numeric term ranges); no C → v1
    type1: bool = False           # F29: type-1 corridor (MOM floor, α-cut ceiling, u ignored)
    static_trust: bool = False    # F29: constant TOD-27 trust instead of the measured registry
    floor_only: bool = False      # F29: max(r_LLM, (1 − c)·y_l) instead of the two-sided corridor
    local: bool = False           # F29: no GPT — the corridor's L decides every snapshot
    gating: bool = False          # E — fuzzy gating of LLM calls

    def flags(self) -> Dict[str, bool]:
        return {k: v for k, v in asdict(self).items() if k != "name"}


RULES_ONLY = Config("rules_only", llm=False)
LLM_ONLY = Config("llm_only")
LLM_PATTERN = Config("llm_pattern", pattern_filter=True)
PROPOSED = Config("proposed", trust=True, envelope=True, view=True, gating=True)
ABLATIONS = [replace(PROPOSED, name="proposed_no_A", envelope=False),
             replace(PROPOSED, name="proposed_no_B", trust=False),
             replace(PROPOSED, name="proposed_no_C", view=False),
             replace(PROPOSED, name="proposed_no_E", gating=False)]
MAIN = [RULES_ONLY, LLM_ONLY, LLM_PATTERN, PROPOSED]
ALL: List[Config] = MAIN + ABLATIONS
EXTRA = [replace(PROPOSED, name="proposed_v2_view", snapshot="v2")]
PROPOSED_V2 = replace(PROPOSED, name="proposed_v2")
V2 = [RULES_ONLY, LLM_ONLY, LLM_PATTERN, PROPOSED_V2,
      replace(PROPOSED_V2, name="proposed_v2_type1", type1=True),
      replace(PROPOSED_V2, name="proposed_v2_static_trust", static_trust=True),
      replace(PROPOSED_V2, name="proposed_v2_floor_only", floor_only=True),
      replace(PROPOSED_V2, name="proposed_v2_no_B", trust=False),
      replace(PROPOSED_V2, name="proposed_v2_no_C", view=False),
      replace(PROPOSED_V2, name="proposed_v2_no_E", gating=False),
      replace(PROPOSED_V2, name="proposed_v2_local", local=True, gating=False)]
FAILURE = ["proposed_v2", "proposed_v2_static_trust"]       # the sensor-failure block: registry vs static trust
BY_NAME = {c.name: c for c in ALL + EXTRA + V2}
