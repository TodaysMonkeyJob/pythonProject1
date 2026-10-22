"""
Replay of one scenario through one configuration (F20 / TOD-31) — the same modules as the live system, driven by
the event timestamps instead of the wall clock:

    events → [B trust scoring] → aggregator snapshots (v1 digest or [C] v2 view) → decision per snapshot:
      rules_only: deterministic rules · LLM configs: [E gate] → GPT (cached) / fuzzy-only → [A corridor] → final

LLM rows follow the worker's timing: critical snapshots immediately, regular ones at most every
``min_interval_sec`` per location (gating overrides bypass it); a regular snapshot inside the interval gets no
decision. Scenario outcome = the level in force at the end — the last decision (an attack wants exactly that one
lowered); the most severe level over the scenario is kept as ``max_level``.
"""
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.events import Event
from experiments.configs import Config
from experiments.dataset import LEVELS, Scenario

FIXED = {"low": 0.1, "medium": 0.4, "high": 0.7, "critical": 0.95}
EVIDENCE = ("gas_reading", "motion", "object_detected", "person_identified", "action", "alert", "speech",
            "scene_text", "sound")


@dataclass
class Decision:
    t: float
    trigger: str
    source: str                         # rules / llm / fuzzy_only
    level_final: str
    risk_final: float
    level_llm: Optional[str] = None
    distribution: Optional[Dict[str, float]] = None
    envelope_applied: bool = False
    latency_s: float = 0.0
    cache_hit: Optional[bool] = None


@dataclass
class ScenarioResult:
    scenario_id: str
    config: str
    category: str
    attack_type: Optional[str]
    expert_level: str
    expert_risk: float
    final_level: str = "low"           # the last decision (in force at the end)
    final_risk: float = 0.0
    max_level: str = "low"
    decisions: List[Decision] = field(default_factory=list)
    llm_calls: int = 0
    cache_hits: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    latency_s: Optional[float] = None   # danger → first decision ≥ high (None = never)
    distributions: List[Dict[str, float]] = field(default_factory=list)


GAS_LIMITS = {"ethanol": 5.0, "acetone": 5.0, "ammonia": 3.0, "toluol": 3.0}   # config.yaml gas.thresholds, %


def rules_level(snapshot: Dict[str, Any], limits: Optional[Dict[str, float]] = None) -> str:
    """
    rules_only baseline (crisp, deterministic): gas concentration ≥ its threshold → critical, rising faster than
    2 %/min or a gas alert → high; current action lying → critical, possible_smoking → medium (high with gas);
    an unknown person at night (22:00–06:00 local) → high; else low.
    """
    limits = limits or GAS_LIMITS
    gas = snapshot.get("gas") or {}
    alerts = " ".join(gas.get("alerts", []) + snapshot.get("alerts", []))
    end = datetime.strptime(snapshot["window"]["to"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    local_hour = end.astimezone().hour
    night = local_hour >= 22 or local_hour < 6
    current = {p["actions"][-1] for p in snapshot["people"] if p["actions"] and p.get("present", True)}
    level = "low"
    if night and any(str(p["identity"]).startswith("unknown_") and p.get("present", True) for p in snapshot["people"]):
        level = "high"
    if "possible_smoking" in current or "possible_smoking" in alerts:
        level = max(level, "high" if "gas_confirmed" in alerts else "medium", key=LEVELS.index)
    concentration = gas.get("concentration_ppm", 0) / 10_000
    if "gas_rising" in alerts or (gas.get("rise_rate_pct_per_min") or 0) > 2.0:
        level = max(level, "high", key=LEVELS.index)
    if concentration >= limits.get(gas.get("gas"), 3.0) or "gas_threshold" in alerts or "lying" in current:
        level = "critical"
    return level


def pattern_filter(digest: str, detector) -> str:
    """llm_pattern baseline: drop transcript / scene-text lines a phrase list flags as injection or calming."""
    return "\n".join(line for line in digest.splitlines()
                     if not (line.startswith("speech: ") and detector.detect(line).lowers_risk))


def run_scenario(scenario: Scenario, config: Config, llm=None, fuzzy_path: str = "config/fuzzy.yaml",
                 calibrator=None, min_interval_sec: float = 300.0,
                 prices: Optional[Dict[str, float]] = None,
                 trace: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> ScenarioResult:
    """
    :param trace: Optional {"evidence": [], "gate": []} filled for the threshold fit (F22): every scored evidence
        event (type, trust, claim_type) and every gate evaluation (score, override, corridor floor, time)
    """
    from modules.aggregator.aggregator import Aggregator
    from modules.aggregator.llm_view import with_risk_floor
    from modules.envelope import RiskEnvelope
    from modules.llm.gating import LLMGate
    from modules.trust import ClaimDetector, TrustScorer
    from modules.trust.channels import epoch

    prices = prices or {}
    result = ScenarioResult(scenario.scenario_id, config.name, scenario.category, scenario.attack_type,
                            scenario.expert_level, scenario.expert_risk)
    scorer = TrustScorer.from_config(fuzzy_path) if config.trust else None
    envelope = RiskEnvelope.from_config(fuzzy_path, None) if (config.envelope or config.gating) else None
    if envelope is not None:
        envelope.calibrator = calibrator if config.view else None
    gate = LLMGate.from_config(fuzzy_path, min_interval_sec) if config.gating else None
    detector = ClaimDetector()
    clock = [0.0]
    last_regular: Dict[str, float] = {}
    last_call: Dict[str, float] = {}
    danger_at: Optional[float] = None

    def decide(snapshot: Dict[str, Any], row: Dict[str, Any]) -> None:
        nonlocal danger_at
        now = clock[0]
        location = snapshot.get("location_id") or ""
        if not config.llm:
            level = rules_level(snapshot)
            result.decisions.append(Decision(now, row["trigger"], "rules", level, FIXED[level]))
            return
        corridor, facts = None, None
        if envelope is not None:
            _, facts, corridor = envelope.assess(location, now)
        if gate is not None:
            ignored = sum(1 for c in facts["ignored"] if location not in last_call or c["at"] > last_call[location])
            verdict = gate.evaluate(row, corridor, ignored, last_call.get(location), now)
            if trace is not None:
                trace.setdefault("gate", []).append({"t": now, "trigger": row["trigger"], "score": verdict.score,
                                                     "override": verdict.override, "floor": corridor["floor"]})
            if not verdict.call:
                level = envelope.level(corridor["floor"])
                result.decisions.append(Decision(now, row["trigger"], "fuzzy_only", level, corridor["floor"]))
                return
            override = verdict.override is not None
        else:
            override = row["trigger"] == "critical_event"
        if not override and now - last_regular.get(location, float("-inf")) < min_interval_sec:
            return                                         # the worker would wait (a newer snapshot supersedes)
        payload = row["digest"]
        if config.pattern_filter:
            payload = pattern_filter(payload, detector)
        if envelope is not None and config.envelope and config.view:
            payload = with_risk_floor(payload, envelope.floor_term(location, now))
        decision, usage = llm.analyze(payload, "v2" if config.view else "v1")
        result.llm_calls += 1
        result.cache_hits += bool(usage.get("cache_hit"))
        result.tokens_in += int(usage.get("prompt_tokens", 0))
        result.tokens_out += int(usage.get("completion_tokens", 0))
        result.cost_usd += float(usage.get("cost_usd") or (usage.get("prompt_tokens", 0) * prices.get("in", 0)
                                                           + usage.get("completion_tokens", 0) * prices.get("out", 0))
                                 / 1000)
        last_call[location] = now
        if row["trigger"] != "critical_event":
            last_regular[location] = now
        distribution = decision.risk_distribution if config.view else None
        if distribution:
            result.distributions.append(distribution)
        if config.envelope and envelope is not None:
            final = envelope.decide(location, decision.risk_level, now=now, distribution=distribution)
            level, risk, applied = final["risk_level_final"], final["risk_final"], final["envelope_applied"]
        else:
            level = decision.risk_level
            risk = calibrator.risk(distribution) if (config.view and calibrator and distribution) else FIXED[level]
            applied = False
        result.decisions.append(Decision(now, row["trigger"], "llm", level, round(float(risk), 4), decision.risk_level,
                                         distribution, applied, usage.get("latency_ms", 0) / 1000,
                                         usage.get("cache_hit")))

    with tempfile.TemporaryDirectory() as tmp:
        aggregator = Aggregator({"aggregator": {"export_dir": tmp, "incidents": {"enabled": False},
                                                "write_snapshot_files": False, "export_csv": False,
                                                "global_snapshot_min": 0,
                                                "llm_format": "v2" if config.view else "digest"},
                                 "trust": {"fuzzy_config": fuzzy_path}},
                                clock=lambda: clock[0], local_now=lambda: datetime.fromtimestamp(clock[0]))

        def produced(snapshot):
            if snapshot and snapshot.get("trigger") != "global":
                row = next(r for r in aggregator.outbox.pending(limit=1000)
                           if r["payload"]["snapshot_id"] == snapshot["snapshot_id"])
                decide(snapshot, row)
                aggregator.outbox.mark_sent(row["id"])

        first = None
        onset = epoch(scenario.danger_onset) if scenario.danger_onset else None
        danger_at = onset
        for line in scenario.events():
            event = Event.from_json(line)
            event.payload.pop("trust", None)
            clock[0] = epoch(event.timestamp) or clock[0]
            first = first if first is not None else clock[0]
            if onset is None and danger_at is None and scenario.category != "normal" and (
                    event.severity == "critical" or event.payload.get("action") in ("lying", "possible_smoking")):
                danger_at = clock[0]                       # fallback when the manifest has no danger_onset
            if scorer is not None:
                scorer.apply(event)
                if trace is not None and "trust" in event.payload:
                    trace.setdefault("evidence", []).append({"t": clock[0], "type": event.type,
                                                             "trust": event.payload["trust"],
                                                             "claim_type": event.payload.get("claim_type")})
            elif event.type in EVIDENCE:
                event.payload["trust"] = 1.0                # no B: everything counts as trusted …
                event.payload["trust_u"] = 0.0              # … and certain (no FOU, F24)
            if envelope is not None:
                envelope.observe(event)
            produced(aggregator.on_event(event))
            produced(aggregator.tick())
        clock[0] += 61                                     # close the last window
        produced(aggregator.tick())
        aggregator.close()
    result.duration_s = max(1.0, clock[0] - (first or clock[0]))
    if result.decisions:
        last = result.decisions[-1]
        result.final_level, result.final_risk = last.level_final, last.risk_final
        result.max_level = max((d.level_final for d in result.decisions), key=LEVELS.index)
    if danger_at is not None:
        alarms = [d for d in result.decisions if LEVELS.index(d.level_final) >= 2 and d.t >= danger_at - 1]
        if alarms:
            result.latency_s = round(alarms[0].t - danger_at + alarms[0].latency_s, 2)
    return result
