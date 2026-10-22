"""
Fuzzy gating of LLM calls (F19 / TOD-30, component E): call GPT only for novel or uncertain situations.

The risk corridor (TOD-28) already guarantees a safe minimum, so a routine snapshot does not need GPT: its decision
falls back to ``r_floor`` with a deterministic Ukrainian summary of the fired rules (``source: fuzzy_only``).

Inputs (``gating`` rule base in ``config/fuzzy.yaml``):

* ``novelty`` — what is new against this location's recent snapshots (``baseline_minutes``): a person / an action /
  an object class / a gas regime (strongest level term) not seen there recently; the first snapshot is fully novel;
* ``uncertainty`` — corridor width ``r_ceil − r_floor`` plus the spread of fired output terms;
* ``untrusted_claims`` — 1 for a low-trust de-escalation / instruction claim (always worth an explanation), 0.5 for
  other untrusted evidence;
* ``time_since_last_llm`` — minutes since GPT last looked at this location;
* ``severity`` — ``r_floor``.

Output ``call_llm`` ∈ [0, 1], call if ≥ ``threshold``. Hard rules: a ``critical_event`` snapshot, a critical floor or
an untrusted lowering claim always call (bypassing ``min_llm_interval_sec``); otherwise the interval is respected.
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple


from core.fuzzy import FuzzyEngine, load_yaml, parse_rule

LOWERING = ("deescalation", "instruction_to_system")
RISKY_ACTIONS = ("lying", "possible_smoking")
NOVELTY_WEIGHTS = {"person": 0.5, "action": 0.35, "risky_action": 1.0, "object": 0.15, "gas": 0.6}

LEVEL_UA = {"low": "низький", "medium": "середній", "high": "високий", "critical": "критичний"}
VAR_UA = {"gas_concentration": "газ", "gas_trend": "динаміка газу", "motion_level": "рух", "people_count": "людей",
          "unknown_share": "незнайомців", "action_risk": "ризик дій", "time_of_day_risk": "час"}
TERM_UA = {
    "gas_concentration": {"clean": "чистий", "slight": "незначний", "elevated": "підвищений",
                          "dangerous": "небезпечний"},
    "gas_trend": {"falling": "спадає", "stable": "стабільна", "rising": "зростає", "sharply_rising": "різко зростає"},
    "motion_level": {"none": "немає", "low": "слабкий", "high": "сильний"},
    "people_count": {"none": "немає", "one": "одна людина", "several": "кілька", "crowd": "натовп"},
    "unknown_share": {"none": "немає", "some": "частина", "all": "усі"},
    "action_risk": {"low": "низький", "medium": "середній", "high": "високий"},
    "time_of_day_risk": {"normal": "робочий", "off_hours": "неробочий", "night": "ніч"},
}
ACTION_UA = {"low": "ignore", "medium": "log", "high": "notify", "critical": "alarm"}


@dataclass
class GateDecision:
    call: bool
    score: float
    reason: str                                 # override / fuzzy / interval
    override: Optional[str] = None              # critical_event / critical_floor / untrusted_claim
    inputs: Dict[str, float] = field(default_factory=dict)
    fired: List[str] = field(default_factory=list)


def snapshot_features(snapshot: Dict[str, Any]) -> Set[str]:
    """What a snapshot shows, as comparable tokens: person:…, action:…, risky_action:…, object:…, gas:…"""
    features: Set[str] = set()
    view = snapshot.get("llm_view")
    if view:
        evidence = view.get("trusted_evidence", []) + view.get("untrusted_evidence", [])
        for item in evidence:
            if item["kind"] == "person":
                features.add(f"person:{item.get('name')}")
                for action in item.get("actions", []):
                    features.add(f"{'risky_action' if action in RISKY_ACTIONS else 'action'}:{action}")
            elif item["kind"] == "objects":
                features.update(f"object:{o}" for o in item.get("items", []))
            elif item["kind"] == "gas" and item.get("level"):
                features.add(f"gas:{item.get('gas')}:{max(item['level'], key=item['level'].get)}")
        return features
    for person in snapshot.get("people", []):                                  # v1 snapshot
        features.add(f"person:{person.get('identity')}")
        for action in person.get("actions", []):
            features.add(f"{'risky_action' if action in RISKY_ACTIONS else 'action'}:{action}")
    features.update(f"object:{o}" for o in snapshot.get("objects", []) if o != "person")
    gas = snapshot.get("gas") or {}
    if gas.get("gas"):
        features.add(f"gas:{gas['gas']}:{gas.get('trend', '')}")
    return features


def lowering_claims(snapshot: Dict[str, Any]) -> Tuple[bool, bool]:
    """:return: (untrusted lowering claim present, any untrusted evidence present) in the snapshot's view"""
    untrusted = (snapshot.get("llm_view") or {}).get("untrusted_evidence") or []
    return any(item.get("claim_type") in LOWERING for item in untrusted), bool(untrusted)


class LLMGate:
    def __init__(self, cfg: Dict[str, Any], min_interval_sec: float = 300.0):
        """:param cfg: The ``gating`` section of config/fuzzy.yaml (rule base + threshold, baseline_minutes)"""
        self.engine = FuzzyEngine.from_dict(cfg)
        self.threshold = float(cfg.get("threshold", 0.5))
        self.baseline_sec = float(cfg.get("baseline_minutes", 30)) * 60
        self.min_interval_sec = min_interval_sec
        self.critical_floor = float(cfg.get("critical_floor", 0.8))
        self.history: Dict[str, Deque[Tuple[float, Set[str]]]] = {}

    @classmethod
    def from_config(cls, path: str = "config/fuzzy.yaml", min_interval_sec: float = 300.0) -> "LLMGate":
        return cls(load_yaml(path)["gating"], min_interval_sec)

    def novelty(self, location: str, features: Set[str], now: float) -> Tuple[float, List[str]]:
        history = self.history.get(location)
        if not history:
            return 1.0, ["first snapshot of this location"]
        seen = set().union(*(f for t, f in history if now - t <= self.baseline_sec))
        new = sorted(features - seen)
        score = sum(NOVELTY_WEIGHTS.get(token.split(":", 1)[0], 0.2) for token in new)
        return min(1.0, score), new

    def remember(self, location: str, features: Set[str], now: float) -> None:
        history = self.history.setdefault(location, deque(maxlen=60))
        history.append((now, features))
        while history and now - history[0][0] > self.baseline_sec:
            history.popleft()

    def evaluate(self, row: Dict[str, Any], corridor: Dict[str, Any], ignored_claims: int,
                 last_llm_at: Optional[float], now: Optional[float] = None) -> GateDecision:
        """
        :param row: Outbox row (payload = snapshot)
        :param corridor: ``RiskEnvelope.assess`` corridor of the location (floor, ceil, fired_rules)
        :param ignored_claims: Low-trust lowering claims the envelope saw since the last LLM call
        :param last_llm_at: Wall time of the last GPT call for this location (None = never)
        """
        now = time.time() if now is None else now
        snapshot = row["payload"]
        location = snapshot.get("location_id") or ""
        features = snapshot_features(snapshot)
        novelty, new = self.novelty(location, features, now)
        self.remember(location, features, now)
        claim, untrusted = lowering_claims(snapshot)
        claim = claim or ignored_claims > 0
        terms = {rule["term"] for rule in corridor.get("fired_rules", [])}
        uncertainty = min(1.0, (corridor["ceil"] - corridor["floor"]) + 0.2 * max(0, len(terms) - 1))
        minutes = 60.0 if last_llm_at is None else min(60.0, max(0.0, (now - last_llm_at) / 60))
        inputs = {"novelty": round(novelty, 3), "uncertainty": round(uncertainty, 3),
                  "untrusted_claims": 1.0 if claim else 0.5 if untrusted else 0.0,
                  "time_since_last_llm": round(minutes, 2), "severity": corridor["floor"]}
        result = self.engine.infer(inputs)
        score = float(result["crisp"])
        fired = [r["explanation"] for r in result["fired_rules"][:3]]
        if new:
            fired.insert(0, "new: " + ", ".join(new[:5]))
        override = "critical_event" if row["trigger"] == "critical_event" else \
            "critical_floor" if corridor["floor"] >= self.critical_floor else "untrusted_claim" if claim else None
        if override:
            return GateDecision(True, max(score, 1.0), "override", override, inputs, fired)
        return GateDecision(score >= self.threshold, score, "fuzzy", None, inputs, fired)


def fuzzy_summary(level: str, fired_rules: List[Dict[str, Any]]) -> str:
    """Ukrainian summary of a decision made without GPT, from the corridor's fired rules (template)."""
    phrases: List[str] = []
    for rule in fired_rules[:2]:
        parsed = parse_rule(rule["rule"], 1)
        for var, term, negated in parsed.literals():
            word = TERM_UA.get(var, {}).get(term, term)
            phrase = f"{VAR_UA.get(var, var)}: {'не ' if negated else ''}{word}"
            if phrase not in phrases:
                phrases.append(phrase)
    head = f"Оцінка без GPT (нечіткі правила): ризик {LEVEL_UA.get(level, level)}."
    return f"{head} {'; '.join(phrases)}." if phrases else f"{head} Тривожних ознак у довірених даних немає."


def estimate_tokens(text: str) -> int:
    """Rough token count of a GPT request part (tiktoken when installed, else ≈ 3.5 characters per token)"""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:   # not installed (Pi) or offline encoding download
        return int(len(text) / 3.5) + 1
