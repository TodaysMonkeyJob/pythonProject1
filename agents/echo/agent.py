"""Echo agent: the template of docs/agents.md — every hook of the interface, nothing clever."""
from typing import List, Optional

from agents.framework.base import Agent, Message
from agents.framework.bus_adapter import Claim


class EchoAgent(Agent):
    def load(self) -> None:
        self.model = {"heard": {}}          # a real agent loads its model here (only while enabled)

    def unload(self) -> None:
        self.model = None                   # drop every reference so the memory is freed

    def on_message(self, message: Message) -> Optional[List[Claim]]:
        if message.kind == "event":
            text = str(message.event.payload.get("text") or "")[: int(self.config.get("max_chars", 80))]
            heard = self.model["heard"]
            heard[message.location] = heard.get(message.location, 0) + 1
            return [self.claim("echo", message.location, {"text": text}, confidence=1.0,
                               evidence_ids=[message.event.event_id])]
        if message.kind == "decision_request":
            count = self.model["heard"].get(message.location_id, 0)
            return [self.claim("echo", message.location_id, {"heard": count}, round_id=message.round_id)]
        return None

    def health(self):
        return {"status": "ok" if getattr(self, "model", None) is not None else "stopped"}
