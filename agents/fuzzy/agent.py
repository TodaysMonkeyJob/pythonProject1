"""Fuzzy agent (B2 / TOD-49): the ``hfs_risk`` of the Article 1 hierarchical fuzzy system as a ``risk`` claim."""
from typing import List, Optional

from agents.framework.base import Agent, Message
from agents.framework.bus_adapter import Claim

KEPT = ("risk1", "risk2", "risk3", "partial", "missing", "corridor_width", "widened_by", "excluded", "rules_version",
        "safety", "explosive")      # G1 (TOD-53): the gas safety invariant travels with the corridor


class FuzzyAgent(Agent):
    def on_message(self, message: Message) -> Optional[List[Claim]]:
        if message.kind != "event" or message.event.type != "hfs_risk":
            return None
        p = message.event.payload
        interval = list(p.get("interval") or [p["overall"], p["overall"]])
        value = {"risk": p["overall"], "level": p["level"], **{k: p[k] for k in KEPT if k in p}}
        return [self.claim("risk", message.event.location_id, value, confidence=1.0, interval=interval,
                           evidence_ids=[message.event.event_id])]
