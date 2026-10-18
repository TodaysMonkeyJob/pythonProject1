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
to GPT.
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
BY_NAME = {c.name: c for c in ALL}
