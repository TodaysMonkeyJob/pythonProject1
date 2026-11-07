"""
Validation agent (B2 / TOD-49, TOD-47): the only agent that issues final decisions.

Inputs: the fuzzy agent's corridor (crisp risk r_F, level, [y_l, y_r]) and the ``risk_opinion`` claims of this
round (``value.risk`` ∈ [0, 1]), each weighted by w = measured accuracy (manifest ``eval_file``) × confidence.
Unmeasured agents weigh 0: their claims are logged, never counted. ``value.kind`` says what an opinion means:

* ``at_least`` — "what I observed is at least this risky" (a fall, a fight, a call for help). It can only raise:
  the strongest weighted floor f above r_F raises the risk to f inside the corridor (``raised``); above y_r it goes
  to the operator. An agent that saw nothing risky says nothing — the absence of a fall is not evidence of safety
  (a camera that sees calm people knows nothing about the gas);
* ``estimate`` (default) — an overall risk. r_A = Σ w·r / Σ w:
  * inside [y_l, y_r] → ``agree``: r = (r_F + Σ w·r) / (1 + Σ w);
  * below y_l → only estimates of physical channels (video / gas) may lower: without them ``deescalation_rejected``
    (r = r_F) — a spoken "everything is fine" never lowers the risk; with them ``deescalation_confirmed``;
  * above y_r → ``escalation_review``.
* no weighted opinion → ``fuzzy_only``: the corridor decides (r = r_F — Article 1).

``escalation_review``: r = y_r (the corridor top) and an operator request with the agents' risk and level — raising
beyond what the measured evidence supports is the operator's call. The decision interval is always the corridor.
``require_physical: false`` (ablation only, TOD-47 comparison) lets any estimate lower the risk.
No GPT here and no ``modules.envelope`` import, so it also runs in ``mode: autonomous``.
"""
from typing import Any, Dict, List, Optional, Tuple

from agents.framework.base import ValidationAgent
from agents.framework.bus_adapter import Claim

PHYSICAL = ("video", "gas")
OPINION = "risk_opinion"
AT_LEAST, ESTIMATE = "at_least", "estimate"


class CorridorValidationAgent(ValidationAgent):
    def load(self) -> None:
        from core.fuzzy import FuzzyEngine, load_yaml

        self.engine = FuzzyEngine.from_dict(load_yaml(self.config.get("risk_terms", "config/fuzzy/top.yaml")))

    def unload(self) -> None:
        self.engine = None

    def level(self, risk: float) -> str:
        from modules.hfs.base import level_of

        return level_of(self.engine, risk)

    def review(self, base: Dict[str, Any], y_l: float, y_r: float, r_a: float, opinions: List[tuple],
               reasons: List[str]) -> Dict[str, Any]:
        return {**base, "risk": y_r, "level": self.level(y_r), "branch": "escalation_review",
                "reasons": reasons + [f"agents {r_a:.2f} above the corridor {y_r:.2f}: the operator decides"],
                "operator_request": {"risk_agents": round(r_a, 4), "level_agents": self.level(r_a),
                                     "corridor": [y_l, y_r], "claims": [c.claim_id for c, _, _ in opinions],
                                     "agents": sorted({c.agent for c, _, _ in opinions}),
                                     "reasons": [str(c.value.get("reason") or c.claim) for c, _, _ in opinions]}}

    def decide(self, corridor: Claim, claims: List[Claim], weights: Dict[str, float],
               confirmations: List[Claim]) -> Dict[str, Any]:
        r_f = float(corridor.value["risk"])
        y_l, y_r = corridor.interval
        base = {"interval": [y_l, y_r], "risk_fuzzy": r_f}
        reasons: List[str] = []
        estimates, floors = [], []
        for c in claims:
            if c.claim != OPINION or not isinstance(c.value.get("risk"), (int, float)):
                continue
            w = weights.get(c.agent, 0.0) * c.confidence
            if w <= 0:
                reasons.append(f"{c.agent}: unmeasured accuracy, not counted")
                continue
            (floors if c.value.get("kind") == AT_LEAST else estimates).append((c, w, float(c.value["risk"])))
        decision = self._estimates(base, corridor, r_f, y_l, y_r, estimates, reasons)
        floor = self._floor(floors)
        if floor is None or floor[2] <= decision["risk"]:
            return decision
        c, _, f = floor
        if f > y_r:
            if decision["branch"] == "escalation_review" and decision["operator_request"]["risk_agents"] >= f:
                return decision
            return self.review(base, y_l, y_r, f, floors, decision["reasons"])
        return {**base, "risk": round(f, 4), "level": self.level(f), "branch": "raised",
                "reasons": decision["reasons"] + [f"raised to {f:.2f} by {c.agent}: {c.value.get('reason', c.claim)}"]}

    @staticmethod
    def _floor(floors: List[tuple]) -> Optional[Tuple[Claim, float, float]]:
        return max(floors, key=lambda item: item[2]) if floors else None

    def _estimates(self, base, corridor, r_f, y_l, y_r, opinions, reasons) -> Dict[str, Any]:
        if not opinions:
            return {**base, "risk": r_f, "level": corridor.value["level"], "branch": "fuzzy_only",
                    "reasons": reasons + ["the fuzzy corridor decides"]}
        total = sum(w for _, w, _ in opinions)
        r_a = sum(w * r for _, w, r in opinions) / total
        base = {**base, "risk_agents": round(r_a, 4)}
        if r_a < y_l:
            physical = [(c, w, r) for c, w, r in opinions if r < y_l and (c.role in PHYSICAL or not self.config.get(
                "require_physical", True))]
            if not physical:
                return {**base, "risk": r_f, "level": corridor.value["level"], "branch": "deescalation_rejected",
                        "reasons": reasons + [f"agents {r_a:.2f} below the corridor {y_l:.2f} without physical "
                                              "confirmation"]}
            r = sum(w * r for _, w, r in physical) / sum(w for _, w, _ in physical)
            return {**base, "risk": round(r, 4), "level": self.level(r), "branch": "deescalation_confirmed",
                    "reasons": reasons + [f"lowered by {', '.join(c.agent for c, _, _ in physical)}"]}
        if r_a > y_r:
            return self.review(base, y_l, y_r, r_a, opinions, reasons)
        r = (r_f + total * r_a) / (1 + total)
        return {**base, "risk": round(r, 4), "level": self.level(r), "branch": "agree",
                "reasons": reasons + [f"agents {r_a:.2f} inside the corridor"]}
