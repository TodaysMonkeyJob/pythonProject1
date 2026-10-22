"""
Fuzzy risk corridor (F17 / TOD-28, component A): the LLM stays the source of context and explanations but cannot
lower the risk on its own.

    [r_floor, r_ceil] = IT2-FIS({e_i, u_i})               corridor.py — all physical evidence, u widens the FOU (F24)
    c                 = lower end of IT2-FIS_confirm     confirmation.py — operator acknowledgement → 1
    L, U              = (1 − c)·(y_l + λ(y_r − y_l)), y_r  fuzzy_corridor.py (F25 / TOD-36) — two active bounds
    r_final           = r_D (GPT ∩ corridor, h ≥ h_min) | L (GPT below) | U (GPT above → escalation_review)

``RiskEnvelope`` runs inside the LLM worker: it follows the forwarded (trust-scored) events and decides every
``llm_decision`` before it is published.
"""
import time
from datetime import datetime
from typing import Any, Dict, Optional

from core.events import Event
from core.fuzzy import load_yaml
from modules.envelope import fuzzy_corridor
from modules.envelope.confirmation import Confirmation
from modules.envelope.corridor import Corridor
from modules.envelope.projection import LEVELS, DEFAULT_LLM_LEVELS, DEFAULT_THRESHOLDS, level_of, level_value, project
from modules.envelope.state import TrustedState
from modules.llm.calibration import RiskCalibrator

# Events the envelope needs (the orchestrator forwards them to the llm module)
ENVELOPE_EVENTS = ["gas_reading", "motion", "object_detected", "person_identified", "action", "alert", "speech",
                   "scene_text", "notification_acked"]


class RiskEnvelope:
    def __init__(self, root: Dict[str, Any], path: str = "config/fuzzy.yaml",
                 locations_path: Optional[str] = "config/locations.yaml"):
        """:param root: The parsed ``config/fuzzy.yaml`` (``envelope`` section + the risk rule base)"""
        cfg = root.get("envelope") or {}
        self.cfg = cfg
        self.llm_levels = {**DEFAULT_LLM_LEVELS, **(cfg.get("llm_levels") or {})}
        self.thresholds = {**DEFAULT_THRESHOLDS, **(cfg.get("thresholds") or {})}
        # F24: no θ cut — only language / scene-text claims need trust.claim_threshold to count
        self.state = TrustedState(float((root.get("trust") or {}).get("claim_threshold", 0.6)))
        self.corridor = Corridor(path, locations_path, float(cfg.get("alpha", 0.5)), cfg.get("floor", "it2"))
        self.confirmation = Confirmation(root, cfg["confirmation"], float(cfg.get("ack_valid_min", 30)))
        self.action_map = (root.get("maps") or {}).get("action_risk") or {}
        # F18: r_LLM from the model's risk distribution once a calibration was fitted (else the fixed mapping)
        self.calibrator = RiskCalibrator.from_file(cfg.get("calibration_path", "config/calibration.yaml"))
        # F25: the two-sided corridor — risk attitude λ, agreement threshold h_min (cost-fitted), K's slope width
        self.lam = float(cfg.get("lambda", 0.0))
        self.h_min = float(cfg.get("h_min", 0.5))
        self.slope = fuzzy_corridor.output_slope(self.corridor.engine(None).variables[
            self.corridor.engine(None).output].terms)

    @classmethod
    def from_config(cls, path: str = "config/fuzzy.yaml",
                    locations_path: Optional[str] = "config/locations.yaml") -> "RiskEnvelope":
        return cls(load_yaml(path), path, locations_path)

    def observe(self, event: Event) -> None:
        self.state.observe(event)

    def assess(self, location_id: str, now: Optional[float] = None, local_hour: Optional[float] = None):
        """:return: (trusted inputs, facts incl. ignored low-trust claims, corridor) of a location — F19 gating"""
        return self._trusted(location_id, time.time() if now is None else now, local_hour)

    def _trusted(self, location_id: str, now: float, local_hour: Optional[float]):
        if local_hour is None:
            moment = datetime.fromtimestamp(now)
            local_hour = moment.hour + moment.minute / 60
        inputs, facts = self.state.features(location_id, now, self.action_map, local_hour)
        return inputs, facts, self.corridor.compute(location_id, inputs, facts.get("uncertainty"))

    def floor_term(self, location_id: Optional[str], now: Optional[float] = None,
                   local_hour: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """:return: The corridor floor as the LLM sees it (F18): {"term": strongest risk term, "mu": its degree}"""
        if not location_id:
            return None
        now = time.time() if now is None else now
        floor = self._trusted(location_id, now, local_hour)[2]["floor"]
        terms = self.corridor.engine(location_id).fuzzify("risk", floor)
        term = max(terms, key=terms.get)
        return {"term": term, "mu": round(terms[term], 2)}

    def level(self, risk: float) -> str:
        """:return: The level of a risk number with this envelope's thresholds (``envelope.thresholds``)"""
        return level_of(risk, self.thresholds)

    def level_sets(self) -> Dict[str, Any]:
        """The calibrated GPT level sets (F18 fit when present, else the risk output terms)"""
        from modules.llm.calibration import DEFAULT_PARAMS

        return self.calibrator.params if self.calibrator is not None else DEFAULT_PARAMS

    def fallback(self, location_id: str, now: Optional[float] = None,
                 local_hour: Optional[float] = None) -> Dict[str, Any]:
        """F25: no GPT answer (offline / gated) → r_final = L of the location's corridor"""
        now = time.time() if now is None else now
        inputs, facts, corridor = self._trusted(location_id, now, local_hour)
        c, reasons, _ = self.confirmation.compute(inputs, facts, now)
        y_l, y_r = corridor["floor"], corridor["ceil"]
        out = fuzzy_corridor.decide(y_l, y_r, c, None, None, self.level_sets(), self.lam, self.h_min, self.slope)
        return {**out, "y_l": y_l, "y_r": y_r, "width": corridor["width"], "c": round(c, 3), "corridor": corridor,
                "confirmation_reasons": reasons}

    def r_llm(self, risk_level: str, distribution: Optional[Dict[str, float]] = None):
        """:return: (r_LLM, source): calibrated centroid of the distribution, or the fixed level number"""
        if distribution and self.calibrator is not None:
            return self.calibrator.risk(distribution), "calibrated"
        return level_value(risk_level, self.llm_levels), "fixed"

    def decide(self, location_id: Optional[str], risk_level: str, now: Optional[float] = None,
               local_hour: Optional[float] = None, distribution: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        :param location_id: The snapshot's location (None = the site-wide summary: no corridor)
        :param risk_level: The LLM's level (low / medium / high / critical)
        :param now: Decision time, seconds since the epoch
        :param local_hour: Local hour 0..24 for ``time_of_day_risk`` (default: from ``now``)
        :param distribution: The model's risk_distribution (F18) — calibrated r_LLM when a calibration exists
        :return: Fields for the ``llm_decision`` payload (risk_llm, risk_floor, risk_ceil, confirmation, risk_final,
                 risk_level_final, envelope_applied, envelope_reason, …)
        """
        now = time.time() if now is None else now
        r_llm, source = self.r_llm(risk_level, distribution)
        if not location_id:
            return {"risk_llm": r_llm, "risk_llm_source": source, "risk_floor": 0.0, "risk_ceil": 1.0,
                    "confirmation": None, "risk_final": r_llm, "risk_level_final": risk_level,
                    "envelope_applied": False,
                    "envelope_reason": {"note": "site-wide summary: no per-location corridor"}}
        inputs, facts, corridor = self._trusted(location_id, now, local_hour)
        c, confirm_reasons, operator = self.confirmation.compute(inputs, facts, now)
        y_l, y_r = corridor["floor"], corridor["ceil"]
        fc = fuzzy_corridor.decide(y_l, y_r, c, distribution, risk_level, self.level_sets(), self.lam, self.h_min,
                                   self.slope)
        final_level = level_of(fc["r_final"], self.thresholds)
        reason = {"fired": [r["explanation"] for r in corridor["fired_rules"][:3]],
                  "confirmation": confirm_reasons,
                  "ignored_claims": [{k: v for k, v in claim.items() if k != "at"} for claim in facts["ignored"]],
                  "claim_threshold": self.state.claim_threshold,
                  "interval": corridor["interval"], "widened_by": corridor["widened_by"][:3]}
        if operator:
            reason["operator"] = operator
        reason.update(h=fc["h"], h_min=self.h_min, policy_branch=fc["policy_branch"], **{"lambda": self.lam})
        raised = LEVELS.index(final_level) > LEVELS.index(risk_level)
        return {"risk_llm": fc["r_llm"], "risk_llm_source": source, "risk_llm_level": r_llm,
                "y_l": y_l, "y_r": y_r, "width": corridor["width"], "L": fc["L"], "U": fc["U"], "r_D": fc["r_D"],
                "h": fc["h"], "policy_branch": fc["policy_branch"], "c": round(c, 3),
                "risk_floor": fc["L"], "risk_ceil": fc["U"], "risk_width": corridor["width"],
                "risk_type1": corridor["type1"], "confirmation": round(c, 3), "risk_final": fc["r_final"],
                "risk_level_final": final_level,
                "envelope_applied": raised,              # the corridor lifted GPT's level (L, or r_D clamped to L)
                "escalation_review": fc["policy_branch"] == "escalation_review",
                "over_reaction": fc["policy_branch"] == "escalation_review",
                "envelope_reason": reason, "trusted_inputs": corridor["inputs"]}


__all__ = ["ENVELOPE_EVENTS", "RiskEnvelope", "project"]
