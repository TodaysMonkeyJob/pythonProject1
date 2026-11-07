"""
Validation agent (B2 / TOD-49): the only agent that issues final decisions.

Inputs: the fuzzy agent's corridor (crisp risk r_F, level, [y_l, y_r]) and the ``risk_opinion`` claims of this
round (``value.risk`` ∈ [0, 1]), each weighted by w = measured accuracy (manifest ``eval_file``) × confidence.
Unmeasured agents weigh 0: their claims are logged, never counted.

* no weighted opinion → ``fuzzy_only``: the corridor decides (r = r_F — Article 1);
* r_A = Σ w·r / Σ w inside [y_l, y_r] → ``agree``: r = (r_F + Σ w·r) / (1 + Σ w), level from the HFS risk terms;
* r_A below y_l → only opinions of physical channels (video / gas) may lower: without them ``deescalation_rejected``
  (r = r_F); with them ``deescalation_confirmed`` (r = their weighted mean) — a spoken "everything is fine" never
  lowers the risk;
* r_A above y_r → ``escalation_review``: r = y_r (the corridor top) and an operator request with the agents' risk.

The decision interval is always the fuzzy corridor. No GPT here and no ``modules.envelope`` import, so it also runs in
``mode: autonomous``.
"""
from typing import Any, Dict, List

from agents.framework.base import ValidationAgent
from agents.framework.bus_adapter import Claim

PHYSICAL = ("video", "gas")
OPINION = "risk_opinion"


class CorridorValidationAgent(ValidationAgent):
    def load(self) -> None:
        from core.fuzzy import FuzzyEngine, load_yaml

        self.engine = FuzzyEngine.from_dict(load_yaml(self.config.get("risk_terms", "config/fuzzy/top.yaml")))

    def unload(self) -> None:
        self.engine = None

    def level(self, risk: float) -> str:
        from modules.hfs.base import level_of

        return level_of(self.engine, risk)

    def decide(self, corridor: Claim, claims: List[Claim], weights: Dict[str, float],
               confirmations: List[Claim]) -> Dict[str, Any]:
        r_f = float(corridor.value["risk"])
        y_l, y_r = corridor.interval
        base = {"interval": [y_l, y_r], "risk_fuzzy": r_f}
        reasons, opinions = [], []
        for c in claims:
            if c.claim != OPINION or not isinstance(c.value.get("risk"), (int, float)):
                continue
            w = weights.get(c.agent, 0.0) * c.confidence
            if w <= 0:
                reasons.append(f"{c.agent}: unmeasured accuracy, not counted")
                continue
            opinions.append((c, w, float(c.value["risk"])))
        if not opinions:
            return {**base, "risk": r_f, "level": corridor.value["level"], "branch": "fuzzy_only",
                    "reasons": reasons + ["the fuzzy corridor decides"]}
        total = sum(w for _, w, _ in opinions)
        r_a = sum(w * r for _, w, r in opinions) / total
        base["risk_agents"] = round(r_a, 4)
        if r_a < y_l:
            physical = [(c, w, r) for c, w, r in opinions if c.role in PHYSICAL and r < y_l]
            if not physical:
                return {**base, "risk": r_f, "level": corridor.value["level"], "branch": "deescalation_rejected",
                        "reasons": reasons + [f"agents {r_a:.2f} below the corridor {y_l:.2f} without physical "
                                              "confirmation"]}
            r = sum(w * r for _, w, r in physical) / sum(w for _, w, _ in physical)
            return {**base, "risk": round(r, 4), "level": self.level(r), "branch": "deescalation_confirmed",
                    "reasons": reasons + [f"lowered by {', '.join(c.agent for c, _, _ in physical)}"]}
        if r_a > y_r:
            return {**base, "risk": y_r, "level": self.level(y_r), "branch": "escalation_review",
                    "reasons": reasons + [f"agents {r_a:.2f} above the corridor {y_r:.2f}: the operator decides"],
                    "operator_request": {"risk_agents": round(r_a, 4), "level_agents": self.level(r_a),
                                         "corridor": [y_l, y_r], "claims": [c.claim_id for c, _, _ in opinions],
                                         "agents": sorted({c.agent for c, _, _ in opinions})}}
        r = (r_f + total * r_a) / (1 + total)
        return {**base, "risk": round(r, 4), "level": self.level(r), "branch": "agree",
                "reasons": reasons + [f"agents {r_a:.2f} inside the corridor"]}
