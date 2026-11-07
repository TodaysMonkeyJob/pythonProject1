"""
Video agent (TOD-47): persons and actions of one location as claims.

* ``action`` events (pose rules or the learned classifier, with ``probability`` / event confidence) → claim
  ``action`` {action, track_id, device_id, source} with that confidence; the latest per track is kept for
  ``stale_sec`` (``disappeared`` drops it);
* ``object_detected`` / ``person_identified`` → claim ``persons`` {in_view, known, unknown};
* decision round → ``risk_opinion`` ``at_least`` = the riskiest live action's risk (``maps.action_risk``),
  confidence = its recognition confidence; nothing when every action is everyday activity (< ``min_risk``) — not
  seeing a fall says nothing about gas or speech.

Action recognition is available to the rest of the system only through these claims: the HFS has no action input
(``modules.hfs.worker.EVIDENCE_EVENTS``), so a subjective label like "fighting vs hugging" can never move the fuzzy
corridor; the validation agent weighs it by the classifier's measured accuracy.
"""
import time
from typing import Any, Dict, List, Optional

import yaml

from agents.framework.base import Agent, Message
from agents.framework.bus_adapter import Claim


class VideoAgent(Agent):
    def load(self) -> None:
        path = self.config.get("action_risk_from", "config/fuzzy.yaml")
        with open(path, encoding="utf-8") as f:
            self.action_risk: Dict[str, float] = {k: float(v) for k, v in
                                                  ((yaml.safe_load(f) or {}).get("maps") or {})
                                                  .get("action_risk", {}).items()}
        self.actions: Dict[str, Dict[tuple, Dict[str, Any]]] = {}       # location → (device, track) → action
        self.persons: Dict[str, Dict[tuple, str]] = {}                  # location → (device, track) → identity
        self.clock = time.time

    def unload(self) -> None:
        self.actions = self.persons = self.action_risk = None

    def on_message(self, message: Message) -> Optional[List[Claim]]:
        if message.kind == "decision_request":
            return self._opinion(message)
        if message.kind != "event":
            return None
        e, p = message.event, message.event.payload
        key = (e.device_id, p.get("track_id"))
        if e.type == "action":
            confidence = p.get("probability", e.confidence)
            confidence = float(self.config.get("default_confidence", 0.7) if confidence is None else confidence)
            self.actions.setdefault(e.location_id, {})[key] = {
                "action": p.get("action"), "confidence": confidence, "at": self.clock(), "event_id": e.event_id,
                "track_id": p.get("track_id"), "device_id": e.device_id, "source": p.get("source", "rules")}
            return [self.claim("action", e.location_id, {"action": p.get("action"), "track_id": p.get("track_id"),
                                                         "device_id": e.device_id, "source": p.get("source", "rules"),
                                                         "risk": self.action_risk.get(p.get("action"), 0.0)},
                               confidence=confidence, evidence_ids=[e.event_id])]
        persons = self.persons.setdefault(e.location_id, {})
        if e.type == "object_detected" and p.get("class") == "person":
            if p.get("state") == "disappeared":
                persons.pop(key, None)
                self.actions.get(e.location_id, {}).pop(key, None)
            else:
                persons.setdefault(key, "unidentified")
        elif e.type == "person_identified" and p.get("track_id") is not None:
            identity = str(p.get("identity") or "")
            persons[key] = "unknown" if identity.startswith("unknown") else \
                "unidentified" if identity in ("not_visible", "unidentified", "") else "known"
        else:
            return None
        counts = {kind: sum(v == kind for v in persons.values()) for kind in ("known", "unknown", "unidentified")}
        return [self.claim("persons", e.location_id, {"in_view": len(persons), **counts},
                           evidence_ids=[e.event_id])]

    def live_actions(self, location_id: str) -> List[Dict[str, Any]]:
        now, stale = self.clock(), float(self.config.get("stale_sec", 30))
        return [a for a in (self.actions.get(location_id) or {}).values() if now - a["at"] <= stale]

    def _opinion(self, message: Message) -> Optional[List[Claim]]:
        risky = [(self.action_risk.get(a["action"], 0.0), a) for a in self.live_actions(message.location_id)]
        risky = [(r, a) for r, a in risky if r >= float(self.config.get("min_risk", 0.3))]
        if not risky:
            return None
        risk, action = max(risky, key=lambda item: (item[0], item[1]["confidence"]))
        reason = f"{action['action']} seen ({action['source']}, p {action['confidence']:.2f})"
        return [self.claim("risk_opinion", message.location_id,
                           {"risk": risk, "kind": "at_least", "action": action["action"],
                            "track_id": action["track_id"], "reason": reason},
                           confidence=action["confidence"], evidence_ids=[action["event_id"]],
                           round_id=message.round_id)]

    def health(self) -> Dict[str, Any]:
        if getattr(self, "action_risk", None) is None:
            return {"status": "stopped"}
        return {"status": "ok", "tracks": sum(len(v) for v in self.actions.values())}
