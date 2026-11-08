"""
Agents host (B2 / TOD-49): routes events and claims to the enabled agents within their data scope, runs decision
rounds with deadlines and records every decision. Final decisions come only from the validation agent.

Decision round (one per fuzzy ``risk`` claim of a location):

1. the fuzzy agent's ``risk`` claim (crisp risk, level, corridor ``interval`` [y_l, y_r]) opens a round for its
   location; an older open round of that location is closed first (its absent agents are noted as ``superseded``);
2. every enabled contributing agent gets a ``decision_request`` (its situation-board view + the corridor); the round
   closes when all answered or at the latest deadline (``timeout_s`` of each agent); an agent that answers after
   its own deadline is ``late`` — ignored for that decision, its claims still reach the board;
3. the validation agent ``decide``s with the corridor, the answers, the agents' measured weights and the physical
   confirmations on the board. Without a validation agent the corridor itself is the decision (``fuzzy_only``):
   with every agent but the fuzzy one off the decisions equal Article 1;
4. ``agent_decision`` event: risk, level, interval, branch, reasons, ``agents`` {name: answered / late / missing /
   error / superseded}, ``claims`` (the ids used), versions, ``decided_by``; ``agent_operator_request`` when the
   validation agent asks the operator.

M1 (TOD-52) modes: a location whose requested mode is ``autonomous`` gets no round (the fuzzy system decides there);
while no location is multiagent the contributing agents are suspended (stopped, models unloaded) and receive nothing.
Every decision carries ``mode_requested`` / ``mode_effective``: it falls back to the fuzzy corridor (``branch``
``fallback``, ``mode_effective`` autonomous, ``fallback_reason``) when the validation agent is off or unhealthy, the
network is down, an agent of ``agents.required`` did not answer in time, or an enabled agent is not healthy yet
(still starting); the next decision returns to multiagent when the cause is gone.

Agents run in their own thread each (``threaded=True``, the module) or inline (tests, replays: an answer is late when
``on_message`` took longer than the timeout). Agents never change rules, thresholds or configuration: they only
return claims, and a claim type outside the manifest's ``publishes`` is dropped.
"""
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from agents.framework.base import Agent, Message
from agents.framework.bus_adapter import Claim, ScopeFilter, SituationBoard
from agents.framework.registry import AgentRegistry
from core.mode import AUTONOMOUS, MULTIAGENT
from modules.hfs.gas_safety import enforce
from system_setup.logger_setup import init_logger

logger = init_logger()

PHYSICAL_ROLES = ("video", "gas")         # their claims can confirm a lower risk (evidence a sensor measured)
CORRIDOR_CLAIM = "risk"


@dataclass
class AgentStats:
    messages: int = 0
    claims: int = 0
    errors: int = 0
    late: int = 0
    latency_ms: Optional[float] = None
    last_error: Optional[str] = None
    last_claims: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, elapsed: float, claims: List[Claim]) -> None:
        ms = elapsed * 1000
        self.latency_ms = ms if self.latency_ms is None else round(0.8 * self.latency_ms + 0.2 * ms, 3)
        self.messages += 1
        self.claims += len(claims)
        for c in claims:
            self.last_claims = ([{k: c.payload()[k] for k in ("claim", "location_id", "value", "confidence", "ts")}]
                                + self.last_claims)[:5]


@dataclass
class Result:
    agent: str
    message: Message
    claims: List[Claim]
    elapsed: float
    error: Optional[str] = None


class AgentRunner:
    """One agent's inbox: a worker thread (``threaded``) or inline calls."""

    def __init__(self, agent: Agent, results: "queue.Queue[Result]", threaded: bool):
        self.agent, self.results, self.threaded = agent, results, threaded
        self.inbox: "queue.Queue[Optional[tuple]]" = queue.Queue()      # (call, message)
        self.thread: Optional[threading.Thread] = None
        if threaded:
            self.thread = threading.Thread(target=self._loop, name=f"agent-{agent.name}", daemon=True)
            self.thread.start()

    def _call(self, fn: Callable[[], Optional[List[Claim]]], message: Message) -> None:
        start = time.perf_counter()
        try:
            claims, error = list(fn() or []), None
        except Exception as e:
            logger.exception(f"agent {self.agent.name} failed on {message.kind}")
            claims, error = [], f"{type(e).__name__}: {e}"
        self.results.put(Result(self.agent.name, message, claims, time.perf_counter() - start, error))

    def _put(self, fn: Callable[[], Optional[List[Claim]]], message: Message) -> None:
        if self.threaded:
            self.inbox.put((fn, message))
        else:
            self._call(fn, message)

    def submit(self, message: Message) -> None:
        self._put(lambda: self.agent.on_message(message), message)

    def tick(self, now: float) -> None:
        self._put(lambda: self.agent.tick(now), Message("tick"))

    def _loop(self) -> None:
        while True:
            item = self.inbox.get()
            if item is None:
                return
            self._call(*item)

    def close(self) -> None:
        if self.thread is not None:
            self.inbox.put(None)
            self.thread.join(timeout=2)


@dataclass
class Round:
    round_id: str
    location_id: str
    corridor: Claim
    opened_at: float
    deadlines: Dict[str, float]                  # agent → its deadline
    answers: Dict[str, List[Claim]] = field(default_factory=dict)
    status: Dict[str, str] = field(default_factory=dict)

    def complete(self) -> bool:
        return all(name in self.status for name in self.deadlines)


class AgentHost:
    def __init__(self, registry: AgentRegistry, emit: Callable[[str, Dict[str, Any], str, str], None],
                 threaded: bool = False, clock: Callable[[], float] = time.time, board_max_age_s: float = 120.0,
                 decides: Optional[bool] = None, mode: Callable[[str], str] = lambda location_id: MULTIAGENT,
                 network_ok: Callable[[], Optional[bool]] = lambda: None, required: Sequence[str] = ()):
        """
        :param emit: (event type, payload, severity, location_id) → publishes on the bus
        :param decides: None = decisions carry the severity of their level (they are the location's decision);
                        False = a shadow, severity info (tests / replays)
        :param mode: location → its requested mode (TOD-52)
        :param network_ok: → False when the network is down (fallback), None = not checked
        :param required: agents whose missing answer makes the decision fall back
        """
        self.registry, self.emit, self.threaded, self.clock = registry, emit, threaded, clock
        self.decides = decides
        self.mode, self.network_ok, self.required = mode, network_ok, tuple(required)
        self.suspended = False
        self.scope = ScopeFilter(registry.manifests.values())
        self.board = SituationBoard(board_max_age_s, clock)
        self.results: "queue.Queue[Result]" = queue.Queue()
        self.runners: Dict[str, AgentRunner] = {}
        self.stats: Dict[str, AgentStats] = {name: AgentStats() for name in registry.manifests}
        self.rounds: Dict[str, Round] = {}                  # location → open round
        self.decisions: List[Dict[str, Any]] = []          # recent (tests, dashboard)
        self.problems: Dict[str, str] = {}

    # --- lifecycle ----------------------------------------------------------------------------------------------
    def start(self) -> None:
        self.problems = self.registry.start_enabled()
        self._sync_runners()

    def stop(self) -> None:
        for runner in self.runners.values():
            runner.close()
        self.runners.clear()
        self.registry.stop_all()

    def _sync_runners(self) -> None:
        for name in [n for n in self.runners if n not in self.registry.instances]:
            self.runners.pop(name).close()
            self.board.forget_agent(name)
        for name, agent in self.registry.instances.items():
            if name not in self.runners:
                self.runners[name] = AgentRunner(agent, self.results, self.threaded)

    def set_enabled(self, name: str, on: bool, by: str = "system") -> Optional[str]:
        reason = self.registry.set_enabled(name, on)
        if reason is None:
            self.problems.pop(name, None)
        self._sync_runners()
        self.emit("agent_status", self.status(), "info", None)
        logger.info(f"agent {name} {'on' if on else 'off'} by {by}" + (f" refused: {reason}" if reason else ""))
        return reason

    # --- input --------------------------------------------------------------------------------------------------
    def suspend(self, suspended: bool) -> None:
        """No location multiagent → stop the contributing agents (models unloaded); back → start the enabled ones."""
        if suspended == self.suspended:
            return
        self.suspended = suspended
        if suspended:
            self.registry.suspend()
        else:
            self.problems = self.registry.start_enabled()
        self._sync_runners()
        self.emit("agent_status", self.status(), "info", None)

    def observe(self, event) -> None:
        """A bus event: control messages for the host, evidence for the agents (within their scope)."""
        if event.type == "agent_control":
            p = event.payload
            self.set_enabled(str(p.get("agent")), bool(p.get("enabled")), str(p.get("by") or "dashboard"))
            return
        if event.type != "hfs_risk" and self.mode(event.location_id) == AUTONOMOUS:
            return                                       # agents receive nothing for an autonomous location
        for name in self.scope.event_receivers(event.type, self.runners):
            self.runners[name].submit(Message("event", event=event))
        self.drain()

    def tick(self) -> None:
        now = self.clock()
        for runner in list(self.runners.values()):
            runner.tick(now)
        self.drain()

    # --- results ------------------------------------------------------------------------------------------------
    def drain(self) -> None:
        """Processes every finished agent call, then closes rounds that are complete or past their deadline."""
        while True:
            try:
                result = self.results.get_nowait()
            except queue.Empty:
                break
            self._result(result)
        now = self.clock()
        for rnd in list(self.rounds.values()):
            if rnd.complete() or now >= max(rnd.deadlines.values(), default=rnd.opened_at):
                self._close(rnd, now)

    def _result(self, result: Result) -> None:
        stats = self.stats[result.agent]
        if result.error:
            stats.errors += 1
            stats.last_error = result.error
        if result.message.kind != "tick" or result.claims or result.error:   # an idle tick is not a message
            stats.record(result.elapsed, result.claims)
        manifest = self.registry.manifests[result.agent]
        rnd = self.rounds.get(result.message.location_id or "") if result.message.kind == "decision_request" else None
        in_round = rnd is not None and rnd.round_id == result.message.round_id
        late = result.message.kind == "decision_request" and (
            not in_round or result.elapsed > manifest.timeout_s
            or (self.threaded and self.clock() > rnd.deadlines.get(result.agent, 0)))
        accepted = [c for c in result.claims if self._accept(c)]
        if result.message.kind != "decision_request":
            return
        if result.error:
            status = "error"
        elif late:
            status, stats.late = "late", stats.late + 1
        else:
            status = "answered"
        if in_round and result.agent not in rnd.status:
            rnd.status[result.agent] = status
            if status == "answered":
                rnd.answers[result.agent] = accepted

    def _accept(self, claim: Claim) -> bool:
        """Boundary check, then board + bus + delivery to subscribers; a fuzzy risk claim opens a round."""
        if not self.scope.may_publish(claim):
            logger.warning(f"agent {claim.agent}: claim {claim.claim!r} not in its manifest's publishes — dropped")
            self.stats[claim.agent].errors += 1
            return False
        claim.at = self.clock()
        self.board.update(claim)
        self.emit("agent_claim", claim.payload(), "info", claim.location_id)
        for name in self.scope.claim_receivers(claim, self.runners):
            self.runners[name].submit(Message("claim", claim=claim))
        if claim.role == "fuzzy" and claim.claim == CORRIDOR_CLAIM and claim.interval is not None \
                and self.mode(claim.location_id) != AUTONOMOUS:
            self._open(claim)
        return True

    # --- rounds -------------------------------------------------------------------------------------------------
    def _open(self, corridor: Claim) -> None:
        now = self.clock()
        old = self.rounds.get(corridor.location_id)
        if old is not None:
            for name in old.deadlines:
                old.status.setdefault(name, "superseded")
            self._close(old, now)
        contributors = [n for n in self.runners if self.registry.manifests[n].contributes]
        rnd = Round(uuid.uuid4().hex[:12], corridor.location_id, corridor, now,
                    {n: now + self.registry.manifests[n].timeout_s for n in contributors})
        self.rounds[corridor.location_id] = rnd
        for name in contributors:
            manifest = self.registry.manifests[name]
            view = self.board.view(corridor.location_id, manifest)
            self.runners[name].submit(Message("decision_request", round_id=rnd.round_id,
                                              location_id=corridor.location_id,
                                              corridor=corridor if manifest.data_scope.allows_claims_of("fuzzy")
                                              else None, board=[c for c in view if c.agent != name]))
        if not self.threaded:
            self.drain()

    def _close(self, rnd: Round, now: float) -> None:
        if self.rounds.get(rnd.location_id) is rnd:
            del self.rounds[rnd.location_id]
        for name in rnd.deadlines:
            rnd.status.setdefault(name, "missing")
        claims = [c for answers in rnd.answers.values() for c in answers]
        validator = self.registry.first("validation")
        corridor = rnd.corridor
        base = {"risk": corridor.value.get("risk"), "level": corridor.value.get("level"),
                "interval": corridor.interval}
        fallback = self.fallback_reasons(rnd, validator)
        if fallback:
            decision = {**base, "branch": "fallback", "reasons": ["працює автономний режим: " + "; ".join(fallback)]}
            decided_by = corridor.agent
        else:
            confirmations = [c for c in self.board.view(rnd.location_id) if c.role in PHYSICAL_ROLES]
            try:
                decision = validator.decide(corridor, claims, self.registry.weights(), confirmations)
                decided_by = validator.name
            except Exception as e:      # a broken validator never blocks the safety baseline
                logger.exception("validation agent failed; the fuzzy corridor decides")
                decision = {**base, "branch": "fuzzy_only", "reasons": [f"validation failed: {e}"]}
                decided_by = corridor.agent
        operator = decision.pop("operator_request", None) if not fallback else None
        record = {"round_id": rnd.round_id, "location_id": rnd.location_id, **decision,
                  "decided_by": decided_by, "corridor": {"claim_id": corridor.claim_id, **base},
                  "mode_requested": MULTIAGENT, "mode_effective": AUTONOMOUS if fallback else MULTIAGENT,
                  "fallback_reason": "; ".join(fallback) or None,
                  "contributing_agents": sorted(n for n, s in rnd.status.items() if s == "answered"),
                  "rules_version": corridor.value.get("rules_version"),
                  "agents": dict(sorted(rnd.status.items())),
                  "missing": sorted(n for n, s in rnd.status.items() if s != "answered"),
                  "claims": [c.claim_id for c in claims],
                  "versions": {n: self.registry.manifests[n].version for n in
                               sorted({corridor.agent, decided_by, *rnd.answers})},
                  "evidence_ids": sorted({e for c in [corridor, *claims] for e in c.evidence_ids}),
                  "latency_ms": round((now - rnd.opened_at) * 1000, 1)}
        # G1 (TOD-53): a gas at / above C_crit keeps the decision critical whatever the agents said (hard floor)
        enforce(record, corridor.value.get("safety"), f"agent {decided_by}",
                float(corridor.value.get("risk") or 0.0))
        self.decisions = (self.decisions + [record])[-200:]
        decides = self.decides is not False
        severity = {"critical": "critical", "high": "warning"}.get(str(record.get("level")), "info") \
            if decides else "info"
        self.emit("agent_decision", record, severity, rnd.location_id)
        if operator:
            self.emit("agent_operator_request", {"round_id": rnd.round_id, "location_id": rnd.location_id,
                                                 **operator}, "warning" if decides else "info", rnd.location_id)

    def fallback_reasons(self, rnd: Round, validator: Optional[Agent]) -> List[str]:
        """:return: why this decision cannot be multi-agent (empty = it can) — TOD-52 automatic fallback"""
        reasons = []
        if validator is None:
            reasons.append("агент валідації вимкнений")
        elif self._health(validator) not in ("ok",):
            reasons.append("агент валідації недоступний")
        if self.network_ok() is False:
            reasons.append("немає мережі")
        late = [n for n in self.required if rnd.status.get(n) != "answered"]
        if late:
            reasons.append(f"не відповіли вчасно: {', '.join(late)}")
        starting = [n for n, agent in self.registry.instances.items()
                    if self.registry.manifests[n].contributes and self._health(agent) not in ("ok", "degraded")]
        if starting:
            reasons.append(f"агенти запускаються: {', '.join(sorted(starting))}")
        return reasons

    @staticmethod
    def _health(agent: Agent) -> str:
        try:
            return str((agent.health() or {}).get("status", "ok"))
        except Exception:
            return "error"

    # --- status (dashboard page «Агенти») -----------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        agents = []
        for name, m in sorted(self.registry.manifests.items()):
            agent, stats = self.registry.instances.get(name), self.stats[name]
            try:
                health = agent.health() if agent is not None else {"status": "off"}
            except Exception as e:
                health = {"status": "error", "detail": str(e)}
            agents.append({"name": name, "role": m.role, "version": m.version, "description": m.description,
                           "backend": m.backend.kind, "enabled": agent is not None, "required": m.required,
                           "blocked": self.registry.blocked(name, agent is None), "problem": self.problems.get(name),
                           "health": health, "timeout_s": m.timeout_s, "latency_ms": stats.latency_ms,
                           "messages": stats.messages, "claims": stats.claims, "errors": stats.errors,
                           "late": stats.late, "last_error": stats.last_error, "last_claims": stats.last_claims,
                           "evaluation": m.evaluation(), "data_scope": m.data_scope.model_dump(),
                           "publishes": m.publishes, "resources": m.resources.model_dump()})
        return {"agents": agents, "errors": self.registry.errors,
                "decisions": [{k: d.get(k) for k in ("round_id", "location_id", "level", "risk", "branch",
                                                     "decided_by", "missing")} for d in self.decisions[-10:]]}
