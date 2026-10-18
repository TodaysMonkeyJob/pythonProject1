"""
Trusted safety confirmation ``c ∈ [0, 1]`` (F17 / TOD-28): how strongly trusted evidence confirms the situation is
safe, so the corridor floor may be released gradually — ``r_final = max(r_LLM, (1 − c) · r_floor)``.

Its own rule base (``envelope.confirmation`` in ``config/fuzzy.yaml``) over trusted inputs only: gas clean and not
rising (only from a fresh trusted reading — a location without a gas sensor never "confirms" clean air), motion
consistent with normal activity, a known person present (trust of their face identification). Untrusted claims
(τ < θ) never contribute. An operator acknowledgement on the dashboard sets c = 1 until it expires
(``ack_valid_min``) or a newer trusted critical alert arrives.
"""
import copy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.fuzzy import FuzzyEngine

GAS_VARIABLES = ("gas_concentration", "gas_trend")


class Confirmation:
    def __init__(self, root: Dict[str, Any], cfg: Dict[str, Any], ack_valid_min: float = 30.0):
        """
        :param root: The whole ``config/fuzzy.yaml`` (shared variables are inherited from it)
        :param cfg: Its ``envelope.confirmation`` section
        """
        variables = {}
        for name in cfg.get("inherit", []):
            variables[name] = copy.deepcopy(root["variables"][name])
            if name in GAS_VARIABLES:
                variables[name].pop("default", None)      # no trusted reading → gas rules cannot confirm anything
        variables.update(copy.deepcopy(cfg.get("variables") or {}))
        self.engine = FuzzyEngine.from_dict({"output": cfg.get("output", "confirmation"), "variables": variables,
                                             "rules": cfg.get("rules") or [], "default_output": 0.0})
        self.ack_valid_sec = float(ack_valid_min) * 60

    def compute(self, inputs: Dict[str, Any], facts: Dict[str, Any],
                now: float) -> Tuple[float, List[str], Optional[str]]:
        """
        :param inputs: Trusted FIS inputs of the location (``TrustedState.features``)
        :param facts: known_present, ack, last_alarm
        :return: (c, reasons, operator id when an acknowledgement released the floor)
        """
        ack = facts.get("ack")
        if ack and now - ack["at"] <= self.ack_valid_sec and ack["at"] >= (facts.get("last_alarm") or 0.0):
            when = datetime.fromtimestamp(ack["at"]).strftime("%H:%M")
            return 1.0, [f"operator {ack['by']} acknowledged at {when}"], ack["by"]
        fis_inputs = {k: v for k, v in inputs.items() if k in self.engine.variables or k == "gas"}
        fis_inputs["known_present"] = float(facts.get("known_present") or 0.0)
        result = self.engine.infer(fis_inputs)
        reasons = [rule["explanation"] for rule in result["fired_rules"][:3]]
        return float(result["crisp"]), reasons, None
