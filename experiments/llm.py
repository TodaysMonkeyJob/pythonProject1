"""
LLM backends for the experiments (F20 / TOD-31), behind one SQLite cache so every run is reproducible.

* ``azure`` — the real ``GPTClient`` (same deployment / temperature for all configurations); prompt v1 =
  ``experiments/prompts/security_analyst_v1.md`` (TOD-17…28, text digest), v2 = ``prompts/security_analyst.md``.
* ``sim`` — a deterministic stand-in analyst for running the harness offline. It reads the same payloads, follows the
  prompt rules, and is deliberately gullible: it believes calming claims in its input (in a v2 view it ignores
  ``untrusted_evidence`` claims only part of the time, decided by a hash of the payload). Results with ``sim``
  validate the pipeline; they say nothing about GPT.

Cache key = sha256(backend, model, temperature, prompt text, payload) → the decision + usage (tokens, latency) of the
first call; re-runs reuse it (``cache_hit``), so numbers are reproducible from the cache alone.
"""
import hashlib
import json
import os
import re
import sqlite3
import time
from typing import Any, Dict, Optional, Tuple

from modules.llm.gpt_client import LLMDecision, load_prompt

PROMPT_V1 = ("security_analyst_v1", os.path.join(os.path.dirname(__file__), "prompts"))
PROMPT_V2 = ("security_analyst", "prompts")
CALMING = re.compile(r"(все гаразд|все добре|все нормально|глючить|зламан|status safe|статус безпечно|everything is "
                     r"fine|all good|ignore the alarm|скасувати|просто перевірка|just a test|sensor is broken)", re.I)


class Cache:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, isolation_level=None)
        self.db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, response TEXT NOT NULL, "
                        "created_at REAL NOT NULL)")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        row = self.db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, response: Dict[str, Any]) -> None:
        self.db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?)",
                        (key, json.dumps(response, ensure_ascii=False), time.time()))

    def close(self) -> None:
        self.db.close()


class SimAnalyst:
    """Deterministic, gullible stand-in for GPT (see module docstring)."""
    model = "sim-analyst-1"
    temperature = 0.0

    def __init__(self, untrusted_obedience: float = 0.3):
        self.untrusted_obedience = untrusted_obedience

    def analyze(self, payload: str, prompt_version: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        level, believed = self._assess(payload)
        if believed:
            level = "low"                                   # talked down
        order = ["low", "medium", "high", "critical"]
        i = order.index(level)
        distribution = {name: 0.0 for name in order}
        distribution[level] = 0.85
        for j in (i - 1, i + 1):
            if 0 <= j < 4:
                distribution[order[j]] += 0.15 / (2 if 0 < i < 3 else 1)
        decision = {"risk_level": level, "summary": "Симульована оцінка.", "recommended_action":
                    {"low": "ignore", "medium": "log", "high": "notify", "critical": "alarm"}[level],
                    "reasons": ["sim"], "follow_up": None}
        if prompt_version == "v2":
            decision["risk_distribution"] = distribution
        tokens = len(payload) // 4 + 300
        return decision, {"prompt_tokens": tokens, "completion_tokens": 80, "latency_ms": 1200}

    def _assess(self, payload: str) -> Tuple[str, bool]:
        clock = re.search(r"(\d{2}):\d{2}:\d{2}", payload)
        hour = int(clock.group(1)) if clock else 12
        night = hour >= 22 or hour < 6
        if payload.startswith("{"):
            view = json.loads(payload)
            trusted, untrusted = view.get("trusted_evidence", []), view.get("untrusted_evidence", [])
            text = json.dumps(trusted, ensure_ascii=False)
            calm_trusted = any(CALMING.search(e.get("text", "")) for e in trusted)
            calm_untrusted = any(CALMING.search(e.get("text", "")) for e in untrusted)
            heed = int(hashlib.sha256(payload.encode()).hexdigest()[:4], 16) / 0xFFFF < self.untrusted_obedience
            believed = calm_trusted or (calm_untrusted and not heed)
        else:
            text, believed = payload, bool(CALMING.search(payload))
        level = "low"
        if "unknown_" in text and night:
            level = "high"
        if "possible_smoking" in text:
            level = max(level, "high" if "gas_confirmed" in text else "medium", key=_rank)
        if re.search(r"gas_rising|gas_threshold|elevated|sharply_rising", text):
            level = max(level, "high", key=_rank)
        if re.search(r"dangerous", text) or ("gas_threshold" in text and "ppm" in text and _ppm(text) >= 50000):
            level = "critical"
        if "lying" in text:
            level = "critical"
        return level, believed


def _rank(level: str) -> int:
    return ["low", "medium", "high", "critical"].index(level)


def _ppm(text: str) -> int:
    match = re.search(r"(\d+)ppm", text)
    return int(match.group(1)) if match else 0


class AzureAnalyst:
    def __init__(self, config: Dict[str, Any]):
        from modules.llm.gpt_client import GPTClient

        llm = dict(config.get("llm", {}))
        self.clients = {"v1": GPTClient({"llm": {**llm, "prompt": PROMPT_V1[0], "prompts_dir": PROMPT_V1[1]}}),
                        "v2": GPTClient({"llm": {**llm, "prompt": PROMPT_V2[0], "prompts_dir": PROMPT_V2[1]}})}
        self.model = self.clients["v2"].deployment
        self.temperature = self.clients["v2"].temperature

    def analyze(self, payload: str, prompt_version: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Rate limits / network errors are retried (the run must not die on a 429); an answer that is still not
        valid JSON after GPTClient's clarification retry is recorded as ``invalid`` (counted in the report)."""
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        from modules.llm.gpt_client import InvalidResponse

        for attempt in range(8):
            try:
                decision, usage = self.clients[prompt_version].analyze(payload)
                return decision.model_dump(), {"prompt_tokens": usage.prompt_tokens,
                                               "completion_tokens": usage.completion_tokens,
                                               "latency_ms": usage.latency_ms, "cost_usd": usage.cost_usd,
                                               "requests": usage.requests}
            except InvalidResponse as e:
                return ({"risk_level": "medium", "summary": "Невалідна відповідь GPT.", "recommended_action": "log",
                         "reasons": [str(e)[:200]], "follow_up": None}, {"invalid": True, "latency_ms": 0})
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
                wait = 10.0 * (attempt + 1)
                retry_after = getattr(getattr(e, "response", None), "headers", {}) or {}
                try:
                    wait = max(wait, float(retry_after.get("retry-after", 0)))
                except (TypeError, ValueError):
                    pass
                print(f"  {type(e).__name__}: waiting {wait:.0f} s", flush=True)
                time.sleep(min(wait, 90))
        raise RuntimeError("Azure kept failing after 8 attempts")


class CachedLLM:
    def __init__(self, backend: str = "sim", cache_path: str = "experiments/cache/llm.sqlite",
                 config: Optional[Dict[str, Any]] = None):
        self.backend_name = backend
        self.backend = SimAnalyst() if backend == "sim" else AzureAnalyst(config or {})
        self.cache = Cache(cache_path)
        self.prompts = {"v1": load_prompt(*PROMPT_V1), "v2": load_prompt(*PROMPT_V2)}
        self.calls = self.hits = self.invalid = 0

    def analyze(self, payload: str, prompt_version: str) -> Tuple[LLMDecision, Dict[str, Any]]:
        key = hashlib.sha256(json.dumps([self.backend_name, self.backend.model, self.backend.temperature,
                                         self.prompts[prompt_version], payload]).encode()).hexdigest()
        self.calls += 1
        cached = self.cache.get(key)
        hit = cached is not None
        if not hit:
            decision, usage = self.backend.analyze(payload, prompt_version)
            cached = {"decision": decision, "usage": usage}
            self.cache.put(key, cached)
        self.hits += hit
        self.invalid += bool(cached["usage"].get("invalid"))
        return LLMDecision(**cached["decision"]), {**cached["usage"], "cache_hit": hit}

    def close(self) -> None:
        self.cache.close()
