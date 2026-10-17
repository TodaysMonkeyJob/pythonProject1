"""
SQLite outbox for snapshots: the aggregator (F5) adds rows, the GPT worker (F6) sends them.

Table ``outbox(id, created_at, trigger, payload_json, status, attempts, last_error, sent_at)``;
status is ``pending`` → ``sent`` or ``failed`` (retried until ``max_attempts``). Nothing is lost while offline:
rows stay pending until F6 can deliver them.
"""
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    trigger TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    sent_at REAL
);
CREATE INDEX IF NOT EXISTS outbox_status_created ON outbox (status, created_at);
"""


class Outbox:
    def __init__(self, path: str):
        """
        :param path: SQLite file (created with its folder if missing)
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        # Autocommit; WAL lets F5 write while F6 reads from another process
        self.db = sqlite3.connect(path, isolation_level=None, timeout=10, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def add(self, snapshot: Dict[str, Any], trigger: str, created_at: Optional[float] = None) -> str:
        """:return: the snapshot id (row id)"""
        self.db.execute("INSERT INTO outbox (id, created_at, trigger, payload_json) VALUES (?, ?, ?, ?)",
                        (snapshot["snapshot_id"], created_at or time.time(), trigger,
                         json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))))
        return snapshot["snapshot_id"]

    def pending(self, limit: int = 10, max_attempts: int = 5) -> List[Dict[str, Any]]:
        """
        :return: Oldest rows to send: pending, or failed with attempts left
        """
        rows = self.db.execute(
            "SELECT id, created_at, trigger, payload_json, status, attempts FROM outbox "
            "WHERE status = 'pending' OR (status = 'failed' AND attempts < ?) ORDER BY created_at LIMIT ?",
            (max_attempts, limit)).fetchall()
        return [{"id": r[0], "created_at": r[1], "trigger": r[2], "payload": json.loads(r[3]), "status": r[4],
                 "attempts": r[5]} for r in rows]

    def mark_sent(self, snapshot_id: str) -> None:
        self.db.execute("UPDATE outbox SET status = 'sent', attempts = attempts + 1, sent_at = ?, last_error = NULL "
                        "WHERE id = ?", (time.time(), snapshot_id))

    def mark_failed(self, snapshot_id: str, error: str) -> None:
        self.db.execute("UPDATE outbox SET status = 'failed', attempts = attempts + 1, last_error = ? WHERE id = ?",
                        (error[:500], snapshot_id))

    def counts(self) -> Dict[str, int]:
        return dict(self.db.execute("SELECT status, COUNT(*) FROM outbox GROUP BY status").fetchall())

    def close(self) -> None:
        self.db.close()
