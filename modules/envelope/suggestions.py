"""
Outcomes of operator escalation reviews (F25 / TOD-36) — the input store of the rule-suggestion work (TOD-38).

When GPT sees a higher risk than the corridor allows (``escalation_review``) the operator accepts or rejects GPT's
level on the dashboard (``escalation_resolved``). Each outcome is kept with the context of the review (GPT's level,
reasons, the corridor bounds, the fuzzy inputs of the location) so that recurring accepted escalations can later be
turned into suggested rules. SQLite (``envelope.suggestions_db``, default ``data/rule_suggestions.db``; tests and
replays ``:memory:``), append-only.
"""
import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, review_id TEXT NOT NULL, at REAL NOT NULL, location_id TEXT,
    accepted INTEGER NOT NULL, by TEXT, risk_level_llm TEXT, risk_level_final TEXT, reason TEXT,
    context_json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS outcomes_location ON outcomes (location_id, at);
"""


class RuleSuggestionStore:
    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def record_outcome(self, review_id: str, accepted: bool, at: float, by: Optional[str] = None,
                       location_id: Optional[str] = None, risk_level_llm: Optional[str] = None,
                       risk_level_final: Optional[str] = None, reason: Optional[str] = None,
                       context: Optional[Dict[str, Any]] = None) -> int:
        """:return: The row id of the stored outcome"""
        with self.lock:
            cursor = self.db.execute(
                "INSERT INTO outcomes (review_id, at, location_id, accepted, by, risk_level_llm, risk_level_final, "
                "reason, context_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (review_id, at, location_id, int(bool(accepted)), by, risk_level_llm, risk_level_final, reason,
                 json.dumps(context or {}, ensure_ascii=False)))
            return int(cursor.lastrowid)

    def outcomes(self, location_id: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
        sql = "SELECT review_id, at, location_id, accepted, by, risk_level_llm, risk_level_final, reason, " \
              "context_json FROM outcomes"
        args: tuple = ()
        if location_id is not None:
            sql, args = sql + " WHERE location_id = ?", (location_id,)
        rows = self.db.execute(sql + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        keys = ("review_id", "at", "location_id", "accepted", "by", "risk_level_llm", "risk_level_final", "reason")
        return [{**dict(zip(keys, row[:8], strict=True)), "accepted": bool(row[3]), "context": json.loads(row[8])}
                for row in rows]

    def close(self) -> None:
        self.db.close()
