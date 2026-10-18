"""
GPT situation analysis (F6): the text digest of a snapshot in, a validated ``LLMDecision`` out.

* Client: ``modules.llm.azure_client`` (same Azure resource as Whisper); model = ``llm.gpt_deployment``.
* System prompt: ``prompts/<llm.prompt>.md`` — editable without touching code.
* Strict JSON: ``response_format={"type": "json_object"}``; if the deployment rejects it, the request is repeated
  without it (and remembered). An invalid answer gets one retry with a clarification, then ``InvalidResponse``.
* Token usage and cost (``llm.price_per_1k_input`` / ``price_per_1k_output``) come back with every decision.
"""
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator

from modules.llm.azure_client import create_client
from system_setup.logger_setup import init_logger

logger = init_logger()


RISK_LEVELS = ("low", "medium", "high", "critical")


class LLMDecision(BaseModel):
    risk_level: Literal["low", "medium", "high", "critical"]
    summary: str = Field(min_length=1, max_length=600)
    recommended_action: Literal["ignore", "log", "notify", "alarm"]
    reasons: List[str] = Field(default_factory=list, max_length=10)
    follow_up: Optional[str] = Field(default=None, max_length=300)
    # F18: the model's probability per level (renormalised to sum 1); optional — older answers stay valid
    risk_distribution: Optional[Dict[str, float]] = None

    @field_validator("risk_distribution", mode="before")
    @classmethod
    def _normalise(cls, value):
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("risk_distribution must be an object {low, medium, high, critical}")
        unknown = set(value) - set(RISK_LEVELS)
        if unknown:
            raise ValueError(f"unknown levels in risk_distribution: {sorted(unknown)}")
        probs = {level: max(0.0, float(value.get(level) or 0.0)) for level in RISK_LEVELS}
        total = sum(probs.values())
        if total <= 0:
            return None                      # nothing usable: fall back to risk_level
        return {level: round(p / total, 4) for level, p in probs.items()}

    def distribution(self) -> Dict[str, float]:
        """:return: risk_distribution, or a one-hot of risk_level when the model gave none"""
        return self.risk_distribution or {level: float(level == self.risk_level) for level in RISK_LEVELS}


class InvalidResponse(RuntimeError):
    """The model did not return a valid decision even after the clarification retry."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    requests: int = 0


def load_prompt(name: str, prompts_dir: str = "prompts") -> str:
    """
    :param name: File name without extension, e.g. "security_analyst"
    :return: The prompt text
    """
    path = os.path.join(prompts_dir, f"{name}.md")
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def parse_decision(text: str) -> LLMDecision:
    """
    :param text: Model output (a JSON object, possibly wrapped in a ```json fence)
    """
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    return LLMDecision.model_validate(json.loads(text))


class GPTClient:
    def __init__(self, config: Dict[str, Any], client=None):
        """
        :param client: Pre-built openai client (tests); created from the llm config otherwise
        """
        llm = config.get("llm", {})
        self.deployment = llm.get("gpt_deployment") or llm.get("chat_deployment", "gpt-4o-mini")
        self.temperature = float(llm.get("temperature", 0.2))
        self.max_output_tokens = int(llm.get("max_output_tokens", 300))
        self.price_in = float(llm.get("price_per_1k_input", 0.0))
        self.price_out = float(llm.get("price_per_1k_output", 0.0))
        self.system_prompt = load_prompt(llm.get("prompt", "security_analyst"), llm.get("prompts_dir", "prompts"))
        self.client = client if client is not None else create_client(llm)
        self.json_mode = True
        self.total = Usage()

    def analyze(self, digest: str) -> Tuple[LLMDecision, Usage]:
        """
        :param digest: Snapshot text digest (modules.aggregator.digest)
        :return: (decision, usage of this call incl. a clarification retry)
        :raises InvalidResponse: no valid decision after one retry; openai errors propagate (caller backs off)
        """
        usage = Usage()
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": digest}]
        text = self._complete(messages, usage)
        try:
            decision = parse_decision(text)
        except (ValueError, ValidationError) as first_error:
            logger.warning(f"Invalid GPT answer, asking again: {str(first_error)[:200]}")
            messages += [{"role": "assistant", "content": text or ""},
                         {"role": "user", "content": "Your previous reply was not a valid JSON object with keys "
                                                     "risk_level, risk_distribution, summary, recommended_action, "
                                                     "reasons, follow_up "
                                                     f"({str(first_error)[:200]}). Reply with that JSON only."}]
            text = self._complete(messages, usage)
            try:
                decision = parse_decision(text)
            except (ValueError, ValidationError) as e:
                raise InvalidResponse(f"Invalid GPT answer after retry: {str(e)[:300]}") from e
        for key in ("prompt_tokens", "completion_tokens", "cost_usd", "latency_ms", "requests"):
            setattr(self.total, key, getattr(self.total, key) + getattr(usage, key))
        logger.info(f"GPT {self.deployment}: {usage.prompt_tokens}+{usage.completion_tokens} tokens, "
                    f"${usage.cost_usd:.5f}, {usage.latency_ms} ms (total ${self.total.cost_usd:.4f})")
        return decision, usage

    def _complete(self, messages: List[Dict[str, str]], usage: Usage) -> str:
        from openai import BadRequestError

        kwargs = {"model": self.deployment, "messages": messages, "temperature": self.temperature,
                  "max_tokens": self.max_output_tokens}
        started = time.monotonic()
        try:
            response = self.client.chat.completions.create(
                **kwargs, **({"response_format": {"type": "json_object"}} if self.json_mode else {}))
        except BadRequestError as e:
            if not self.json_mode or "response_format" not in str(e):
                raise
            logger.warning(f"Deployment {self.deployment} does not support JSON mode; validating plain answers")
            self.json_mode = False
            response = self.client.chat.completions.create(**kwargs)
        usage.latency_ms += int((time.monotonic() - started) * 1000)
        usage.requests += 1
        if response.usage is not None:
            usage.prompt_tokens += response.usage.prompt_tokens
            usage.completion_tokens += response.usage.completion_tokens
            usage.cost_usd += (response.usage.prompt_tokens * self.price_in
                               + response.usage.completion_tokens * self.price_out) / 1000
        return response.choices[0].message.content or ""
