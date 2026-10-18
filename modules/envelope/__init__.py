"""
Fuzzy risk corridor (F17 / TOD-28, component A): the LLM stays the source of context and explanations but cannot
lower the risk on its own.

    [r_floor, r_ceil] = FIS({e_i : τ(e_i) ≥ θ})          corridor.py — risk rules over trusted evidence only
    c                 = FIS_confirm(trusted evidence)    confirmation.py — operator acknowledgement → 1
    r_final           = max(r_LLM, (1 − c) · r_floor)    projection.py

``RiskEnvelope`` runs inside the LLM worker: it follows the forwarded (trust-scored) events and decides every
``llm_decision`` before it is published.
"""
import time
from datetime import datetime
from typing import Any, Dict, Optional

import yaml

from core.events import Event
from modules.envelope.confirmation import Confirmation
from modules.envelope.corridor import Corridor
from modules.envelope.projection import DEFAULT_LLM_LEVELS, DEFAULT_THRESHOLDS, level_value, project
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
        self.margin = float(cfg.get("over_reaction_margin", 0.3))
        thetas = {loc: float(v["theta"]) for loc, v in (cfg.get("locations") or {}).items() if "theta" in v}
        self.state = TrustedState(float(cfg.get("theta", 0.6)), thetas)
        self.corridor = Corridor(path, locations_path, float(cfg.get("alpha", 0.5)))
        self.confirmation = Confirmation(root, cfg["confirmation"], float(cfg.get("ack_valid_min", 30)))
        self.action_map = (root.get("maps") or {}).get("action_risk") or {}
        # F18: r_LLM from the model's risk distribution once a calibration was fitted (else the fixed mapping)
        self.calibrator = RiskCalibrator.from_file(cfg.get("calibration_path", "config/calibration.yaml"))

    @classmethod
    def from_config(cls, path: str = "config/fuzzy.yaml",
                    locations_path: Optional[str] = "config/locations.yaml") -> "RiskEnvelope":
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f) or {}, path, locations_path)

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
        return inputs, facts, self.corridor.compute(location_id, inputs)

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
        projection = project(r_llm, corridor["floor"], corridor["ceil"], c, self.thresholds, self.margin)
        reason = {"fired": [r["explanation"] for r in corridor["fired_rules"][:3]],
                  "confirmation": confirm_reasons,
                  "ignored_claims": [{k: v for k, v in claim.items() if k != "at"} for claim in facts["ignored"]],
                  "theta": self.state.theta_for(location_id)}
        if operator:
            reason["operator"] = operator
        return {"risk_llm": r_llm, "risk_llm_source": source, "risk_floor": corridor["floor"],
                "risk_ceil": corridor["ceil"],
                "confirmation": round(c, 3), "risk_final": projection["risk_final"],
                "risk_level_final": projection["risk_level_final"],
                "envelope_applied": projection["envelope_applied"], "over_reaction": projection["over_reaction"],
                "envelope_reason": reason, "trusted_inputs": corridor["inputs"]}


__all__ = ["ENVELOPE_EVENTS", "RiskEnvelope", "project"]
