"""
Audio agent (TOD-47): transcripts, danger words, sound types and the meaning of speech as claims.

* ``speech`` → claim ``speech`` {text, speaker_status, person_id, intent, spans, danger (grade + words),
  interpreter}; confidence = the event's ``asr_confidence`` when present;
* ``sound_type`` → claim ``sound`` {type, danger} with the classifier's confidence;
* decision round, from the speech of the last ``window_sec``:
  * a danger word (grade 0.4 / 0.7 / 1.0, ``config/danger_words.yaml``) or an escalation phrase → ``risk_opinion``
    ``at_least`` (raises only);
  * a de-escalation claim / an instruction to the system ("все гаразд", "ignore the alarm") → ``risk_opinion``
    ``estimate`` = ``calm_claim_risk`` — what the speaker asserts. The validation agent rejects lowering without
    physical confirmation, so this is logged as an attempt (``manipulation: true`` for instructions), never obeyed.

Intent: the local pattern detector of TOD-27 (``modules/trust/claims.py``, Ukrainian + English, stdlib only — loaded
by file so ``modules.trust``'s scorer is never imported) or, with ``interpreter: llm`` in hybrid mode,
``agents.audio.interpret.LLMInterpreter`` (only the transcript text is sent; failures fall back to the patterns).
"""
import importlib.util
import os
import sys
import time
from typing import Any, Dict, List, Optional

from agents.framework.base import Agent, Message
from agents.framework.bus_adapter import Claim
from core.mode import is_autonomous

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALM = ("deescalation", "instruction_to_system")


def claim_detector():
    """``modules.trust.claims.ClaimDetector`` without importing the ``modules.trust`` package (its scorer)."""
    name = "_iot_trust_claims"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "modules", "trust", "claims.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module.ClaimDetector()


class AudioAgent(Agent):
    def load(self) -> None:
        from modules.audio.keywords import DangerWords

        self.words = DangerWords.load(self.config.get("danger_words", "config/danger_words.yaml"))
        self.detector = claim_detector()
        self.interpreter, self.problem = None, None
        if self.config.get("interpreter", "rules") == "llm":
            if is_autonomous(self.full_config):
                self.problem = "interpreter llm ignored in mode autonomous (no cloud calls): local patterns"
            else:
                from agents.audio.interpret import LLMInterpreter

                self.interpreter = LLMInterpreter(self.full_config)
        self.speech: Dict[str, List[Dict[str, Any]]] = {}       # location → recent interpreted speech
        self.clock = time.time

    def unload(self) -> None:
        self.words = self.detector = self.interpreter = self.speech = None

    def interpret(self, text: str) -> Dict[str, Any]:
        """:return: {intent, spans, interpreter, risk?}"""
        if self.interpreter is not None:
            try:
                return {**self.interpreter.interpret(text), "interpreter": "llm"}
            except Exception as e:          # offline / quota: the local patterns still work
                self.problem = f"llm interpreter failed ({type(e).__name__}): local patterns"
        found = self.detector.detect(text)
        return {"intent": found.claim_type, "spans": [s for _, s in found.spans][:5], "interpreter": "rules"}

    def on_message(self, message: Message) -> Optional[List[Claim]]:
        if message.kind == "decision_request":
            return self._opinion(message)
        if message.kind != "event":
            return None
        e, p = message.event, message.event.payload
        if e.type == "sound_type":
            return [self.claim("sound", e.location_id, {"type": p.get("type"), "danger": p.get("danger")},
                               confidence=float(p.get("confidence") or 0.0), evidence_ids=[e.event_id])]
        text = str(p.get("text") or "")[: int(self.config.get("max_text", 200))]
        if not text.strip():
            return None
        grade, matches = self.words.grade(text)
        meaning = self.interpret(text)
        item = {"text": text, "speaker_status": p.get("speaker_status"), "person_id": p.get("person_id"),
                **meaning, "danger": {"grade": grade, "words": [m.entry for m in matches[:3]]}}
        self.speech.setdefault(e.location_id, []).append({**item, "at": self.clock(), "event_id": e.event_id})
        confidence = p.get("asr_confidence")
        return [self.claim("speech", e.location_id, item, confidence=1.0 if confidence is None else float(confidence),
                           evidence_ids=[e.event_id])]

    def _opinion(self, message: Message) -> Optional[List[Claim]]:
        window = float(self.config.get("window_sec", 30))
        recent = [s for s in self.speech.get(message.location_id, []) if self.clock() - s["at"] <= window]
        self.speech[message.location_id] = recent
        out = []
        escalation = float(self.config.get("escalation_risk", 0.7))

        def floor(item: Dict[str, Any]) -> float:
            risk = item["danger"]["grade"]
            if item["intent"] == "escalation":
                risk = max(risk, escalation, float(item.get("risk") or 0.0))
            return risk

        raising = [(floor(s), s) for s in recent]
        raising = [(r, s) for r, s in raising if r > 0]
        if raising:
            risk, s = max(raising, key=lambda item: item[0])
            words = ", ".join(s["danger"]["words"]) or s["intent"]
            out.append(self.claim("risk_opinion", message.location_id,
                                  {"risk": risk, "kind": "at_least", "reason": f"said: {words}"},
                                  evidence_ids=[s["event_id"]], round_id=message.round_id))
        calm = [s for s in recent if s["intent"] in CALM]
        if calm:
            s = calm[-1]
            out.append(self.claim("risk_opinion", message.location_id,
                                  {"risk": float(self.config.get("calm_claim_risk", 0.1)), "kind": "estimate",
                                   "reason": f"spoken claim ({s['intent']}): {s['text'][:60]}",
                                   "manipulation": s["intent"] == "instruction_to_system"},
                                  evidence_ids=[s["event_id"]], round_id=message.round_id))
        return out or None

    def health(self) -> Dict[str, Any]:
        if getattr(self, "words", None) is None:
            return {"status": "stopped"}
        return {"status": "degraded" if self.problem else "ok", **({"detail": self.problem} if self.problem else {})}
