"""
LLM worker (F6): sends snapshot digests from the SQLite outbox to GPT and publishes ``llm_decision`` events.

* Critical snapshots go out immediately; non-critical ones at most once per ``min_llm_interval_sec`` PER LOCATION
  (F12; the global "site" snapshot counts as its own location) — only the newest pending one, older non-critical rows
  of that location are skipped (cost control). ``llm_decision`` carries the snapshot's ``location_id``.
* Network / rate-limit / 5xx / timeout errors: exponential backoff, the row stays ``pending`` (nothing is lost
  offline). Credential / deployment errors: logged, retried rarely, rows stay pending.
* A decision that is not valid JSON after one clarification retry marks the row ``failed``.
"""
import os
import time
from typing import Any, Dict, List, Optional

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
                 clock=time.monotonic):
        """
        :param gpt: object with analyze(digest) -> (LLMDecision, Usage); GPTClient from config when None
        :param outbox: snapshot outbox; aggregator.outbox_path when None
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

    def setup(self) -> None:
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

    def step(self) -> None:
        now = self.clock()
        if self.gpt is None or now < max(self._next_poll, self._blocked_until):
            time.sleep(0.1)
            return
        self._next_poll = now + self.poll_sec
        for row in self.select(self.outbox.pending(limit=50, max_attempts=self.max_attempts), now):
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
        for key, regular in by_location.items():
            if now - self._last_regular.get(key, float("-inf")) >= self.min_interval:
                for row in regular[:-1]:
                    self.outbox.mark_skipped(row["id"], "superseded by a newer snapshot")
                chosen.append(regular[-1])
        return chosen

    def send(self, row: Dict[str, Any]) -> bool:
        """
        :return: False when sending should pause (network / quota / credentials problem)
        """
        from openai import (APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, NotFoundError,
                            PermissionDeniedError, RateLimitError)

        from modules.llm.gpt_client import InvalidResponse

        digest = row.get("digest") or row["payload"].get("reason") or ""
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
        if row["trigger"] != "critical_event":
            self._last_regular[location_key(row)] = self.clock()
        self.outbox.mark_sent(row["id"])
        payload = {"snapshot_id": row["id"], "trigger": row["trigger"], **decision.model_dump(),
                   "model": getattr(self.gpt, "deployment", None), "prompt_tokens": usage.prompt_tokens,
                   "completion_tokens": usage.completion_tokens, "cost_usd": round(usage.cost_usd, 6),
                   "latency_ms": usage.latency_ms}
        if row["payload"].get("incident_id"):
            payload["incident_id"] = row["payload"]["incident_id"]
        location_id = row["payload"].get("location_id")
        if location_id:
            payload["location_id"] = location_id
        self.emit("llm_decision", payload, severity=SEVERITY[decision.risk_level], location_id=location_id)
        return True

    def _pause(self, error: Exception) -> None:
        self._backoff = min(self.max_backoff, max(2.0, self._backoff * 2))
        self._blocked_until = self.clock() + self._backoff
        logger.warning(f"GPT unreachable ({type(error).__name__}); retrying in {self._backoff:.0f} s, "
                       f"snapshots stay pending")


def run(event_queue, control_queue, config):
    """Module entry point for the orchestrator (own process; works from the outbox, needs no forwarded events)."""
    LLMWorker(event_queue, control_queue, config).run()
