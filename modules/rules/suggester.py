"""
GPT rule suggestions (F27 / TOD-38): when to ask, what to ask, how the answer is checked. GPT never changes a rule —
its suggestions only land in the separate review store (``modules.rules.suggestions``).

Triggers (``SuggestionTriggers``, in the orchestrator router next to the trust evidence):

* an ``escalation_review`` the operator **accepted** (GPT saw a higher risk than the corridor and was right);
* over-corridor escalations repeating ≥ ``repeat_n`` times in a location within ``repeat_window_h``, even without an
  operator (one job per window and location);
* an operator-confirmed alarm (``trust_feedback`` confirm on a physical channel) in a location whose latest decision
  fired no risk rule above ``low`` — a coverage gap;
* on demand: ``python -m tools.suggest_rules --since 7d``.

``RuleSuggester`` (the LLM worker when idle, or the CLI) turns a job into one GPT call with
``prompts/rule_suggester.md``: the DSL grammar, the risk variables and their terms (from ``config/fuzzy.yaml``) and the
supporting cases; the strict JSON answer is validated (pydantic), the rule checked offline with the engine's validator
(invalid DSL is stored with the report, never fixed silently) and stored — deduplicated by rule + location. Nothing is
published on the event bus or written to the JSONL logs; the process log only gets counts.
"""
import json
import os
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from modules.rules.suggestions import RuleSuggestionStore
from system_setup.logger_setup import init_logger

logger = init_logger()

GRAMMAR = """IF <antecedent> THEN risk IS <term> [WITH <weight 0..1>]
antecedent := and ("OR" and)* ;  and := unary ("AND" unary)* ;  unary := "NOT" unary | "(" antecedent ")" | clause
clause := <variable> IS [NOT] <term>     (keywords upper case; AND binds tighter than OR)"""
PHYSICAL_KINDS = ("objects", "motion", "pose_action", "smoking", "gas_class", "gas_concentration", "gas_trend", "vad")


class Suggestion(BaseModel):
    rule: str = Field(default="")
    location: str = Field(default="*")
    rationale: str = Field(default="")
    supporting_cases: List[Union[int, str]] = Field(default_factory=list)
    new_variable_needed: Optional[str] = None
    confidence: Literal["low", "medium", "high"] = "low"


def parse_suggestion(text: str) -> Suggestion:
    """:raise ValueError / ValidationError: not the strict JSON object of the prompt"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("the answer is not a JSON object")
    return Suggestion.model_validate(data)


def validate_rule(rule: str, location: str, fuzzy_path: str = "config/fuzzy.yaml",
                  new_variable_needed: Optional[str] = None) -> Dict[str, Any]:
    """
    Offline check of a suggested rule against the current rule base (the location's overrides for a location rule).

    :return: {"valid", "errors", "warnings"} — never modifies the rule
    """
    from core.fuzzy import FuzzyConfigError, FuzzyEngine, load_yaml
    from core.fuzzy.engine import location_overrides
    from core.fuzzy.rules import parse_rule
    from core.fuzzy.validate import subsumed

    if not rule.strip():
        if new_variable_needed:
            return {"valid": False, "errors": [f"no rule: a new variable is needed ({new_variable_needed})"],
                    "warnings": []}
        return {"valid": False, "errors": ["empty rule"], "warnings": []}
    try:
        candidate = parse_rule({"id": "SUGGESTED", "rule": rule}, 0)
    except ValueError as e:
        return {"valid": False, "errors": [f"syntax: {e}"], "warnings": []}
    data = load_yaml(fuzzy_path)
    spec = {"id": "SUGGESTED", "rule": rule}
    try:
        if location and location != "*":
            overrides = location_overrides(data, location, None) + [{"rules": [spec]}]
            engine = FuzzyEngine.from_dict(data, location, overrides)
        else:
            engine = FuzzyEngine.from_dict({**data, "rules": list(data.get("rules") or []) + [spec]})
    except FuzzyConfigError as e:
        return {"valid": False, "errors": list(e.errors), "warnings": []}
    warnings = [w for w in subsumed(engine.rules) if "SUGGESTED" in w]
    if candidate.output != engine.output:
        return {"valid": False, "errors": [f"concludes on {candidate.output}, must be {engine.output}"],
                "warnings": warnings}
    return {"valid": True, "errors": [], "warnings": warnings}


def variables_text(fuzzy_path: str = "config/fuzzy.yaml") -> str:
    """The risk FIS inputs and output with their terms (what a suggested rule may use)"""
    from core.fuzzy import FuzzyEngine, load_yaml

    engine = FuzzyEngine.from_dict(load_yaml(fuzzy_path))
    lines = []
    for name, variable in engine.variables.items():
        role = "output" if name == engine.output else "input"
        unit = f" ({variable.unit})" if variable.unit else ""
        lines.append(f"- {name}{unit} [{role}]: {', '.join(variable.terms)}")
    return "\n".join(lines)


class SuggestionTriggers:
    """Router side: records cases and queues suggester jobs. Wraps the store (``record_outcome`` stays compatible)."""

    def __init__(self, store: RuleSuggestionStore, cfg: Optional[Dict[str, Any]] = None,
                 fuzzy_path: str = "config/fuzzy.yaml", registry=None):
        cfg = cfg or {}
        self.store = store
        self.repeat_n = int(cfg.get("repeat_n", 3))
        self.repeat_window = float(cfg.get("repeat_window_h", 24)) * 3600
        self.registry = registry
        self.reviews: Dict[str, Deque[tuple]] = defaultdict(deque)     # location → (t, case id)
        self.repeat_job_at: Dict[str, float] = {}
        self.last_fired: Dict[str, List[str]] = {}                      # location → fired risk rule ids
        self.rule_terms: Dict[str, str] = {}
        try:
            from core.fuzzy import load_yaml
            from core.fuzzy.governance import iter_rules
            from core.fuzzy.rules import parse_rule

            for section, spec in iter_rules(load_yaml(fuzzy_path)):
                if section == "risk" or section.startswith("locations."):
                    rule = parse_rule(spec, 0)
                    self.rule_terms[rule.id] = rule.term
        except (OSError, ValueError, KeyError) as e:
            logger.warning(f"Rule suggestions: risk rules unavailable ({e})")

    # the F25 call in modules.trust.evidence (operator answered an escalation review)
    def record_outcome(self, review_id: str, accepted: bool, at: float, by: Optional[str] = None,
                       location_id: Optional[str] = None, risk_level_llm: Optional[str] = None,
                       risk_level_final: Optional[str] = None, reason: Optional[str] = None,
                       context: Optional[Dict[str, Any]] = None) -> int:
        case = self.store.record_outcome(review_id, accepted, at, by, location_id, risk_level_llm, risk_level_final,
                                         reason, context)
        if accepted:
            self.store.add_job("accepted_escalation", location_id, [case], at)
        return case

    def observe(self, event, now: float) -> None:
        p, t = event.payload, event.type
        location = p.get("location_id") or event.location_id
        if t == "llm_decision" and location:
            self.last_fired[location] = list(p.get("fired_ids") or [])
        elif t == "escalation_review" and location:
            case = self.store.record_outcome(str(p.get("review_id")), False, now, None, location,
                                             p.get("risk_level_llm"), p.get("risk_level_final"), None,
                                             {k: p.get(k) for k in ("summary", "reasons", "U", "h", "risk_llm",
                                                                    "inputs", "snapshot_id")},
                                             kind="repeated_escalation")
            seen = self.reviews[location]
            seen.append((now, case))
            while seen and now - seen[0][0] > self.repeat_window:
                seen.popleft()
            if len(seen) >= self.repeat_n and now - self.repeat_job_at.get(location, float("-inf")) > \
                    self.repeat_window:
                self.repeat_job_at[location] = now
                self.store.add_job("repeated_escalation", location, [c for _, c in seen], now)
        elif t == "trust_feedback" and p.get("verdict") == "confirm":
            self._coverage(p, now)

    def _coverage(self, p: Dict[str, Any], now: float) -> None:
        channel = str(p.get("channel_id") or "")
        device, _, kind = channel.rpartition("/")
        if kind not in PHYSICAL_KINDS:
            return
        location = p.get("location_id")
        if not location and self.registry is not None:
            location = (self.registry.devices.get(device) or {}).get("location_id")
        if not location:
            return
        fired = self.last_fired.get(location, [])
        if any(self.rule_terms.get(rid, "low") != "low" for rid in fired):
            return                                           # a risk rule covered it
        case = self.store.record_outcome(str(p.get("event_id") or channel), True, now, p.get("by"), location, None,
                                         None, p.get("reason"), {"channel_id": channel, "fired_rules": fired},
                                         kind="coverage_gap")
        self.store.add_job("coverage_gap", location, [case], now)


class RuleSuggester:
    """Asks GPT for one rule per job and stores the checked answer."""

    def __init__(self, store: RuleSuggestionStore, gpt, fuzzy_path: str = "config/fuzzy.yaml",
                 prompt_path: str = "prompts/rule_suggester.md", max_tokens: int = 500):
        """:param gpt: ``modules.llm.gpt_client.GPTClient`` (its openai client and deployment) or a test double"""
        self.store, self.gpt, self.fuzzy_path, self.max_tokens = store, gpt, fuzzy_path, max_tokens
        with open(prompt_path, encoding="utf-8") as f:
            self.template = f.read()

    def prompt(self) -> str:
        return self.template.replace("{grammar}", GRAMMAR).replace("{variables}", variables_text(self.fuzzy_path))

    def cases(self, job: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{"case_id": c["id"], "kind": c["kind"], "location": c["location_id"], "gpt_level": c["risk_level_llm"],
                 "final_level": c["risk_level_final"], "operator_accepted": c["accepted"], **c["context"]}
                for c in self.store.outcomes(ids=job["case_ids"], limit=50)]

    def ask(self, job: Dict[str, Any]) -> str:
        messages = [{"role": "system", "content": self.prompt()},
                    {"role": "user", "content": json.dumps({"trigger": job["trigger"], "location": job["location_id"],
                                                            "cases": self.cases(job)}, ensure_ascii=False,
                                                           default=str)}]
        response = self.gpt.client.chat.completions.create(
            model=self.gpt.deployment, messages=messages, temperature=0, max_tokens=self.max_tokens,
            response_format={"type": "json_object"})
        return response.choices[0].message.content or ""

    def run(self, job: Dict[str, Any]) -> Optional[int]:
        """:return: the suggestion id (None when GPT's answer was not the strict JSON — the job fails)"""
        try:
            suggestion = parse_suggestion(self.ask(job))
        except (ValueError, ValidationError) as e:
            self.store.finish_job(job["id"], error=f"invalid answer: {str(e)[:200]}")
            return None
        data = suggestion.model_dump()
        if not data["supporting_cases"]:
            data["supporting_cases"] = list(job["case_ids"])
        if data["location"] in ("", None):
            data["location"] = job.get("location_id") or "*"
        report = validate_rule(data["rule"], data["location"], self.fuzzy_path, data["new_variable_needed"])
        stored = self.store.add_suggestion(data, report, job["trigger"])
        self.store.finish_job(job["id"], stored["id"])
        return stored["id"]

    def process_pending(self, limit: int = 1) -> int:
        """:return: how many jobs were handled (counts only go to the process log)"""
        done = 0
        for job in self.store.pending_jobs(limit):
            self.run(job)
            done += 1
        if done:
            logger.info(f"Rule suggester: {done} job(s) handled")
        return done


def manual_job(store: RuleSuggestionStore, since_sec: float, location: Optional[str] = None) -> Optional[int]:
    """On demand: one job over the stored cases of the last ``since_sec`` (accepted escalations, gaps, repeats)."""
    now = time.time()
    cases = [c["id"] for c in store.outcomes(location, limit=50, since=now - since_sec)
             if c["accepted"] or c["kind"] in ("repeated_escalation", "coverage_gap")]
    return store.add_job("manual", location, cases, now) if cases else None


def store_from_config(config: Dict[str, Any]) -> RuleSuggestionStore:
    rules = config.get("rules") or {}
    path = rules.get("suggestions_db") or (config.get("envelope") or {}).get("suggestions_db") or ":memory:"
    return RuleSuggestionStore(path if path == ":memory:" else os.path.abspath(path))
