"""
LLM worker (F6): sends snapshot digests from the SQLite outbox to GPT and publishes ``llm_decision`` events.

* Critical snapshots go out immediately; non-critical ones at most once per ``min_llm_interval_sec`` PER LOCATION
  (F12; the global "site" snapshot counts as its own location) — only the newest pending one, older non-critical rows
  of that location are skipped (cost control). ``llm_decision`` carries the snapshot's ``location_id``.
* Network / rate-limit / 5xx / timeout errors: exponential backoff, the row stays ``pending`` (nothing is lost
  offline). Credential / deployment errors: logged, retried rarely, rows stay pending.
* A decision that is not valid JSON after one clarification retry marks the row ``failed``.
* F17 (TOD-28): before publishing, the fuzzy risk corridor (``modules.envelope.RiskEnvelope``, fed with the forwarded
  trust-scored events) projects the LLM's risk: ``r_final = max(r_LLM, (1 − c) · r_floor)``. The decision carries
  risk_llm / risk_floor / risk_ceil / confirmation / risk_final / risk_level_final / envelope_applied /
  envelope_reason; a raised level also publishes ``deescalation_rejected``. Every decision → ``envelope.jsonl``.
* F19 (TOD-30): with ``llm.gating`` a fuzzy controller (``modules.llm.gating.LLMGate``) decides per location snapshot
  whether GPT is needed (novelty, corridor uncertainty, untrusted claims, time since the last call, severity). Not
  needed → ``llm_decision`` with ``source: fuzzy_only``, risk = r_floor and a Ukrainian summary of the fired rules
  (works without Azure keys too). critical_event / critical floor / untrusted lowering claim → always GPT. Every
  gating decision → ``gating.jsonl`` (inputs, estimated tokens saved).
"""
import os
import time
from typing import Any, Dict, List, Optional

from core.events import Event, make_event
from core.logger import JsonlLogger
from core.module import BaseModule
from modules.aggregator.outbox import Outbox
from system_setup.logger_setup import init_logger

logger = init_logger()

def location_key(row: Dict[str, Any]) -> str:
    return "__site__" if row["trigger"] == "global" else row["payload"].get("location_id") or ""


SEVERITY = {"low": "info", "medium": "warning", "high": "critical", "critical": "critical"}


class LLMWorker(BaseModule):
    name = "llm"
    source = "llm"

    def __init__(self, event_queue, control_queue, config: Dict[str, Any], gpt=None, outbox: Optional[Outbox] = None,
                 clock=time.monotonic, envelope=None, wall_clock=time.time, gate=None):
        """
        :param gpt: object with analyze(digest) -> (LLMDecision, Usage); GPTClient from config when None
        :param outbox: snapshot outbox; aggregator.outbox_path when None
        :param envelope: ``RiskEnvelope`` (tests); from ``trust.fuzzy_config`` when None and ``envelope.enabled``
        :param gate: ``LLMGate`` (tests); from the ``gating`` section when None, ``llm.gating`` and an envelope
        """
        super().__init__(event_queue, control_queue, config)
        llm = config.get("llm", {})
        self.min_interval = float(llm.get("min_llm_interval_sec", 300))
        self.poll_sec = float(llm.get("poll_sec", 1.0))
        self.max_backoff = float(llm.get("max_backoff_sec", 300))
        self.max_attempts = int(llm.get("max_attempts", 3))
        self.gpt = gpt
        self.outbox = outbox
        self.clock = clock
        self._next_poll = 0.0
        self._last_regular: Dict[str, float] = {}
        self._backoff = 0.0
        self._blocked_until = 0.0
        self.unavailable: Optional[str] = None
        self.envelope = envelope
        self.wall_clock = wall_clock
        self.envelope_log: Optional[JsonlLogger] = None
        self.gate = gate
        self.gating_enabled = bool(llm.get("gating", True))
        self.gating_log: Optional[JsonlLogger] = None
        self._gated: Dict[str, Any] = {}               # row id → GateDecision (each snapshot is judged once)
        self._last_llm_wall: Dict[str, float] = {}     # location → wall time of the last GPT call
        self._prompt_tokens: Optional[int] = None

    def setup(self) -> None:
        if self.envelope is None and self.config.get("envelope", {}).get("enabled", True):
            from modules.envelope import RiskEnvelope

            path = self.config.get("trust", {}).get("fuzzy_config", "config/fuzzy.yaml")
            if os.path.isfile(path):
                self.envelope = RiskEnvelope.from_config(path)
            else:
                logger.warning(f"Risk envelope off: {path} not found — LLM decisions are published unbounded")
        if self.envelope is not None and self.config.get("envelope", {}).get("log", True):
            log_cfg = self.config.get("logging", {})
            self.envelope_log = JsonlLogger("envelope", log_cfg.get("dir", "logs"), log_cfg.get("max_total_mb", 2048))
        if self.gate is None and self.gating_enabled and self.envelope is not None:
            from modules.llm.gating import LLMGate

            path = self.config.get("trust", {}).get("fuzzy_config", "config/fuzzy.yaml")
            self.gate = LLMGate.from_config(path, self.min_interval)
        if self.gate is not None:
            log_cfg = self.config.get("logging", {})
            self.gating_log = JsonlLogger("gating", log_cfg.get("dir", "logs"), log_cfg.get("max_total_mb", 2048))
        if self.outbox is None:
            agg = self.config.get("aggregator", {})
            self.outbox = Outbox(agg.get("outbox_path", os.path.join(agg.get("export_dir", "exports"), "outbox.db")))
        if self.gpt is None:
            from modules.llm.azure_client import MissingCredentials
            from modules.llm.gpt_client import GPTClient

            try:
                self.gpt = GPTClient(self.config)
            except (MissingCredentials, OSError) as e:
                self.unavailable = str(e)
                logger.error(f"GPT analysis disabled: {e}; snapshots stay pending in the outbox")

    def teardown(self) -> None:
        if self.outbox is not None:
            self.outbox.close()
        if self.envelope_log is not None:
            self.envelope_log.close()
        if self.gating_log is not None:
            self.gating_log.close()

    def handle_control(self, message: Any) -> None:
        """Forwarded (trust-scored) evidence events feed the risk envelope."""
        if not (isinstance(message, str) and message.startswith("{")):
            super().handle_control(message)
            return
        if self.envelope is not None:
            self.envelope.observe(Event.from_json(message))

    def step(self) -> None:
        now = self.clock()
        if (self.gpt is None and self.gate is None) or now < max(self._next_poll, self._blocked_until):
            time.sleep(0.1)
            return
        self._next_poll = now + self.poll_sec
        for row in self.select(self.outbox.pending(limit=50, max_attempts=self.max_attempts), now):
            if self.gpt is None:
                break            # no keys: fuzzy-only decisions still flow, rows that need GPT stay pending
            if not self.send(row):
                break

    def select(self, rows: List[Dict[str, Any]], now: float) -> List[Dict[str, Any]]:
        """
        Critical rows first (all, oldest first); then the newest regular row if the interval allows it.
        Older regular rows are skipped.
        """
        critical = [r for r in rows if r["trigger"] == "critical_event"]
        chosen = list(critical)
        by_location: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            if row["trigger"] != "critical_event":
                by_location.setdefault(location_key(row), []).append(row)
        if self.gate is not None:
            for row in critical:
                self._gate(row)                        # logged + remembered as the location's baseline
        for key, regular in by_location.items():
            interval_ok = now - self._last_regular.get(key, float("-inf")) >= self.min_interval
            if self.gate is None or regular[-1]["trigger"] == "global":
                if interval_ok:
                    for row in regular[:-1]:
                        self.outbox.mark_skipped(row["id"], "superseded by a newer snapshot")
                    chosen.append(regular[-1])
                continue
            for row in regular[:-1]:
                self.outbox.mark_skipped(row["id"], "superseded by a newer snapshot")
                self._gated.pop(row["id"], None)
            newest = regular[-1]
            decision = self._gate(newest)
            if not decision.call:
                self.fuzzy_only(newest, decision)
            elif decision.override or interval_ok:
                chosen.append(newest)
        return chosen

    # --- F19 gating ---------------------------------------------------------------------------------------------

    def _gate(self, row: Dict[str, Any]):
        """Judges a snapshot once (cached by row id) and logs the decision."""
        if row["id"] in self._gated:
            return self._gated[row["id"]]
        now = self.wall_clock()
        location = row["payload"].get("location_id") or ""
        _, facts, corridor = self.envelope.assess(location, now)
        last = self._last_llm_wall.get(location)
        ignored = sum(1 for claim in facts["ignored"] if last is None or claim["at"] > last)
        decision = self.gate.evaluate(row, corridor, ignored, last, now)
        decision.corridor = corridor
        self._gated[row["id"]] = decision
        if len(self._gated) > 500:
            self._gated.pop(next(iter(self._gated)))
        saved = 0 if decision.call else self._request_tokens(row)
        if self.gating_log is not None:
            self.gating_log.write(make_event("llm", "llm_gating", {
                "snapshot_id": row["id"], "location_id": location, "trigger": row["trigger"],
                "call_llm": decision.call, "score": round(decision.score, 3), "reason": decision.reason,
                "override": decision.override, "inputs": decision.inputs, "fired": decision.fired,
                "tokens_saved_est": saved}, location_id=location or None))
        return decision

    def _request_tokens(self, row: Dict[str, Any]) -> int:
        """Estimated prompt + answer tokens a GPT call for this row would have cost"""
        from modules.llm.gating import estimate_tokens

        if self._prompt_tokens is None:
            try:
                from modules.llm.gpt_client import load_prompt

                llm = self.config.get("llm", {})
                prompt = load_prompt(llm.get("prompt", "security_analyst"), llm.get("prompts_dir", "prompts"))
                self._prompt_tokens = estimate_tokens(prompt)
            except OSError:
                self._prompt_tokens = 0
        return self._prompt_tokens + estimate_tokens(row.get("digest") or "") + 150

    def fuzzy_only(self, row: Dict[str, Any], decision) -> Event:
        """Publishes the decision of a snapshot GPT did not need: risk = r_floor, summary from the fired rules."""
        from modules.envelope.projection import level_of
        from modules.llm.gating import ACTION_UA, fuzzy_summary

        corridor = decision.corridor
        location_id = row["payload"].get("location_id")
        floor = float(corridor["floor"])
        level = level_of(floor, self.envelope.thresholds)
        payload = {"snapshot_id": row["id"], "trigger": row["trigger"], "source": "fuzzy_only", "risk_level": level,
                   "summary": fuzzy_summary(level, corridor["fired_rules"]),
                   "recommended_action": ACTION_UA[level],
                   "reasons": [r["explanation"] for r in corridor["fired_rules"][:3]], "follow_up": None,
                   "risk_llm": None, "risk_floor": corridor["floor"], "risk_ceil": corridor["ceil"],
                   "confirmation": None, "risk_final": round(floor, 3), "risk_level_final": level,
                   "envelope_applied": False,
                   "gating": {"score": round(decision.score, 3), "inputs": decision.inputs, "fired": decision.fired}}
        if location_id:
            payload["location_id"] = location_id
        self.outbox.mark_handled(row["id"], "fuzzy_only")
        self._gated.pop(row["id"], None)
        event = self.emit("llm_decision", payload, severity=SEVERITY[level], location_id=location_id)
        if self.envelope_log is not None:
            self.envelope_log.write(event)
        return event

    def send(self, row: Dict[str, Any]) -> bool:
        """
        :return: False when sending should pause (network / quota / credentials problem)
        """
        from openai import (APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, NotFoundError,
                            PermissionDeniedError, RateLimitError)

        from modules.llm.gpt_client import InvalidResponse

        digest = row.get("digest") or row["payload"].get("reason") or ""
        location_id = row["payload"].get("location_id")
        if self.envelope is not None and row["trigger"] != "global":   # F18: the corridor floor, for consistency
            from modules.aggregator.llm_view import with_risk_floor

            digest = with_risk_floor(digest, self.envelope.floor_term(location_id, now=self.wall_clock()))
        try:
            decision, usage = self.gpt.analyze(digest)
        except InvalidResponse as e:
            logger.error(f"Snapshot {row['id']}: {e}")
            self.outbox.mark_failed(row["id"], str(e))
            return True
        except (AuthenticationError, PermissionDeniedError, NotFoundError) as e:
            self._blocked_until = self.clock() + self.max_backoff
            logger.error(f"GPT request rejected ({type(e).__name__}: {e}); check .env keys and llm.gpt_deployment. "
                         f"Retrying in {self.max_backoff:.0f} s, snapshots stay pending")
            return False
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            self._pause(e)
            return False
        except APIStatusError as e:
            if e.status_code >= 500:
                self._pause(e)
                return False
            logger.error(f"Snapshot {row['id']} rejected by GPT ({e.status_code}): {e}")
            self.outbox.mark_failed(row["id"], f"{e.status_code}: {e}")
            return True

        self._backoff = 0.0
        self._gated.pop(row["id"], None)
        self._last_llm_wall[row["payload"].get("location_id") or ""] = self.wall_clock()
        if row["trigger"] != "critical_event":
            self._last_regular[location_key(row)] = self.clock()
        self.outbox.mark_sent(row["id"])
        payload = {"snapshot_id": row["id"], "trigger": row["trigger"], "source": "llm", **decision.model_dump(),
                   "model": getattr(self.gpt, "deployment", None), "prompt_tokens": usage.prompt_tokens,
                   "completion_tokens": usage.completion_tokens, "cost_usd": round(usage.cost_usd, 6),
                   "latency_ms": usage.latency_ms}
        if row["payload"].get("incident_id"):
            payload["incident_id"] = row["payload"]["incident_id"]
        if location_id:
            payload["location_id"] = location_id
        level = decision.risk_level
        if self.envelope is not None:
            payload.update(self.envelope.decide(None if row["trigger"] == "global" else location_id, level,
                                                now=self.wall_clock(), distribution=decision.risk_distribution))
            level = payload["risk_level_final"]
        event = self.emit("llm_decision", payload, severity=SEVERITY[level], location_id=location_id)
        if self.envelope_log is not None:
            self.envelope_log.write(event)
        if payload.get("envelope_applied"):
            rejected = self.emit("deescalation_rejected", {
                "snapshot_id": row["id"], "decision_event_id": event.event_id, "location_id": location_id,
                "risk_level_llm": decision.risk_level, "risk_level_final": level, "risk_llm": payload["risk_llm"],
                "risk_floor": payload["risk_floor"], "confirmation": payload["confirmation"],
                "risk_final": payload["risk_final"], "reasons": payload["envelope_reason"]},
                severity="warning", location_id=location_id)
            logger.warning(f"Envelope raised the GPT risk for {location_id}: {decision.risk_level} → {level}")
            if self.envelope_log is not None:
                self.envelope_log.write(rejected)
        return True

    def _pause(self, error: Exception) -> None:
        self._backoff = min(self.max_backoff, max(2.0, self._backoff * 2))
        self._blocked_until = self.clock() + self._backoff
        logger.warning(f"GPT unreachable ({type(error).__name__}); retrying in {self._backoff:.0f} s, "
                       f"snapshots stay pending")


def run(event_queue, control_queue, config):
    """Module entry point for the orchestrator (own process; works from the outbox, needs no forwarded events)."""
    LLMWorker(event_queue, control_queue, config).run()
