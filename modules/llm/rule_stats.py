"""
Fired-rule statistics per rule base version (F26 / TOD-37 monitoring), kept by the LLM worker.

Every decision counts the risk rules that fired in its corridor (``fired``). When the operator answers an escalation
review (F25) the rules of that decision get an outcome: the corridor's bound kept (GPT's escalation rejected) →
``confirmed``; GPT's higher level accepted (the rules underestimated the situation) → ``rejected``. SQLite
(``rules.stats_db``, default ``data/rules/firing.db``); the dashboard page «Правила» reads it.
"""
import os
import sqlite3
import threading
from typing import Dict, Iterable, List

SCHEMA = """
CREATE TABLE IF NOT EXISTS firing (version TEXT NOT NULL, rule_id TEXT NOT NULL, fired INTEGER NOT NULL DEFAULT 0,
    confirmed INTEGER NOT NULL DEFAULT 0, rejected INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (version, rule_id));
"""


class RuleStats:
    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _bump(self, version: str, rule_ids: Iterable[str], column: str) -> None:
        with self.lock:
            for rule_id in dict.fromkeys(rule_ids):
                self.db.execute(f"INSERT INTO firing (version, rule_id, {column}) VALUES (?, ?, 1) "
                                f"ON CONFLICT (version, rule_id) DO UPDATE SET {column} = {column} + 1",
                                (version, rule_id))

    def record(self, version: str, rule_ids: Iterable[str]) -> None:
        self._bump(version, rule_ids, "fired")

    def outcome(self, version: str, rule_ids: Iterable[str], bound_kept: bool) -> None:
        self._bump(version, rule_ids, "confirmed" if bound_kept else "rejected")

    def table(self, version: str = None) -> List[Dict[str, object]]:
        sql = "SELECT version, rule_id, fired, confirmed, rejected FROM firing"
        rows = self.db.execute(sql + (" WHERE version = ?" if version else "") + " ORDER BY fired DESC",
                               (version,) if version else ()).fetchall()
        return [dict(zip(("version", "rule_id", "fired", "confirmed", "rejected"), r, strict=True)) for r in rows]

    def close(self) -> None:
        self.db.close()


def read_stats(path: str, version: str = None) -> List[Dict[str, object]]:
    """Read-only access for the dashboard"""
    if not os.path.isfile(path):
        return []
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        sql = "SELECT version, rule_id, fired, confirmed, rejected FROM firing"
        rows = db.execute(sql + (" WHERE version = ?" if version else "") + " ORDER BY fired DESC",
                          (version,) if version else ()).fetchall()
    finally:
        db.close()
    return [dict(zip(("version", "rule_id", "fired", "confirmed", "rejected"), r, strict=True)) for r in rows]
