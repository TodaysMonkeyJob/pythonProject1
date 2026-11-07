"""
Optional language-model interpretation of a transcript for the audio agent (TOD-47, ``interpreter: llm``, hybrid
mode only). Sends the transcript text — nothing else — with ``prompts/speech_interpreter.md`` and expects JSON
``{"intent": escalation | deescalation | instruction_to_system | none, "risk": 0..1, "why": "…"}``. The deployment and
keys are the LLM worker's (``llm`` section, ``modules.llm.azure_client``).
"""
import json
import os
from typing import Any, Dict

INTENTS = ("none", "escalation", "deescalation", "instruction_to_system")
PROMPT = os.path.join("prompts", "speech_interpreter.md")


class LLMInterpreter:
    def __init__(self, config: Dict[str, Any], client=None):
        """:param client: a pre-built openai client (tests); otherwise made from the ``llm`` section"""
        llm = config.get("llm", {})
        self.deployment = llm.get("gpt_deployment") or llm.get("chat_deployment", "gpt-4o-mini")
        with open(PROMPT, encoding="utf-8") as f:
            self.prompt = f.read()
        if client is None:
            from modules.llm.azure_client import create_client

            client = create_client(llm)
        self.client = client

    def interpret(self, text: str) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.deployment, temperature=0, max_tokens=120, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": self.prompt}, {"role": "user", "content": text}])
        data = json.loads(response.choices[0].message.content or "{}")
        intent = data.get("intent") if data.get("intent") in INTENTS else "none"
        risk = min(1.0, max(0.0, float(data.get("risk") or 0.0)))
        return {"intent": intent, "risk": risk, "spans": [str(data.get("why") or "")[:120]]}
