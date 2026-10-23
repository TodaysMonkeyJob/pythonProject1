"""
Rule-suggestion store (F27 / TOD-38): cases, suggester jobs and GPT's rule suggestions, separate from logs and the
event bus — SQLite ``rules.suggestions_db`` (default ``data/rule_suggestions.db``), reachable only through
``tools/suggestions.py`` and the dashboard page «Пропозиції правил».

* ``outcomes`` — cases: operator outcomes of escalation reviews (F25), repeated over-corridor escalations, operator-
  confirmed alarms without a fired risk rule (coverage gaps); context = GPT's reasons, the corridor, fuzzy inputs;
* ``jobs`` — what the suggester still has to ask GPT about (trigger, location, case ids), done by the LLM worker;
* ``suggestions`` — GPT's answer: the rule, location, rationale, supporting cases, new_variable_needed, confidence,
  the validator report (invalid DSL is stored as is, never fixed silently), an optional gate report, status
  ``new | rejected | edited | sent_to_approval | approved``, reviewer; ``dedup_key`` = normalised rule + location →
  a repeat raises ``support_count`` instead of a new row.

Nothing changes ``config/fuzzy.yaml`` here: sending a suggestion to approval creates a TOD-37 proposal.
"""
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, review_id TEXT NOT NULL, at REAL NOT NULL, location_id TEXT,
    accepted INTEGER NOT NULL, by TEXT, risk_level_llm TEXT, risk_level_final TEXT, reason TEXT,
    context_json TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'escalation_review');
CREATE INDEX IF NOT EXISTS outcomes_location ON outcomes (location_id, at);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL, trigger TEXT NOT NULL, location_id TEXT,
    case_ids TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', error TEXT, suggestion_id INTEGER);
CREATE TABLE IF NOT EXISTS suggestions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, location TEXT NOT NULL, rule TEXT NOT NULL,
    rationale TEXT, supporting_cases TEXT NOT NULL, new_variable_needed TEXT, confidence TEXT,
    validator_report TEXT NOT NULL, gate_report TEXT, status TEXT NOT NULL DEFAULT 'new', reviewer TEXT,
    reviewed_at TEXT, review_reason TEXT, dedup_key TEXT NOT NULL, support_count INTEGER NOT NULL DEFAULT 1,
    trigger TEXT, proposal_id TEXT, original_rule TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS suggestions_dedup ON suggestions (dedup_key);
"""
STATUSES = ("new", "rejected", "edited", "sent_to_approval", "approved")
COLUMNS = ("id", "created_at", "location", "rule", "rationale", "supporting_cases", "new_variable_needed",
           "confidence", "validator_report", "gate_report", "status", "reviewer", "reviewed_at", "review_reason",
           "dedup_key", "support_count", "trigger", "proposal_id", "original_rule")
JSON_COLUMNS = ("supporting_cases", "validator_report", "gate_report")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dedup_key(rule: str, location: str) -> str:
    """Normalised rule (canonical DSL when it parses, else whitespace / case folded) + location"""
    try:
        from core.fuzzy.rules import parse_rule, render

        parsed = parse_rule(rule, 0)
        text = f"IF {render(parsed.antecedent)} THEN {parsed.output} IS {parsed.term}"
    except ValueError:
        text = " ".join(str(rule).split()).upper()
    return f"{location or '*'}|{text}"


class RuleSuggestionStore:
    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=10)
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")      # the router and the LLM worker write concurrently
        self.db.executescript(SCHEMA)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(outcomes)")}
        if "kind" not in columns:                       # stores created by F25 (TOD-36)
            self.db.execute("ALTER TABLE outcomes ADD COLUMN kind TEXT NOT NULL DEFAULT 'escalation_review'")
        self.lock = threading.Lock()

    # --- cases ---------------------------------------------------------------------------------------------------

    def record_outcome(self, review_id: str, accepted: bool, at: float, by: Optional[str] = None,
                       location_id: Optional[str] = None, risk_level_llm: Optional[str] = None,
                       risk_level_final: Optional[str] = None, reason: Optional[str] = None,
                       context: Optional[Dict[str, Any]] = None, kind: str = "escalation_review") -> int:
        """:return: the case id"""
        with self.lock:
            cursor = self.db.execute(
                "INSERT INTO outcomes (review_id, at, location_id, accepted, by, risk_level_llm, risk_level_final, "
                "reason, context_json, kind) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (review_id, at, location_id, int(bool(accepted)), by, risk_level_llm, risk_level_final, reason,
                 json.dumps(context or {}, ensure_ascii=False, default=str), kind))
            return int(cursor.lastrowid)

    def outcomes(self, location_id: Optional[str] = None, limit: int = 500, since: Optional[float] = None,
                 ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        sql = "SELECT id, review_id, at, location_id, accepted, by, risk_level_llm, risk_level_final, reason, " \
              "context_json, kind FROM outcomes WHERE 1 = 1"
        args: list = []
        if location_id is not None:
            sql, args = sql + " AND location_id = ?", args + [location_id]
        if since is not None:
            sql, args = sql + " AND at >= ?", args + [since]
        if ids:
            sql += f" AND id IN ({','.join('?' * len(ids))})"
            args += list(ids)
        rows = self.db.execute(sql + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        keys = ("id", "review_id", "at", "location_id", "accepted", "by", "risk_level_llm", "risk_level_final",
                "reason")
        return [{**dict(zip(keys, row[:9], strict=True)), "accepted": bool(row[4]), "context": json.loads(row[9]),
                 "kind": row[10]} for row in rows]

    # --- jobs ----------------------------------------------------------------------------------------------------

    def add_job(self, trigger: str, location_id: Optional[str], case_ids: List[int], at: float) -> int:
        with self.lock:
            cursor = self.db.execute("INSERT INTO jobs (created_at, trigger, location_id, case_ids) VALUES (?,?,?,?)",
                                     (at, trigger, location_id, json.dumps(list(case_ids))))
            return int(cursor.lastrowid)

    def pending_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        rows = self.db.execute("SELECT id, created_at, trigger, location_id, case_ids FROM jobs WHERE status = "
                               "'pending' ORDER BY id LIMIT ?", (limit,)).fetchall()
        return [{"id": r[0], "created_at": r[1], "trigger": r[2], "location_id": r[3],
                 "case_ids": json.loads(r[4])} for r in rows]

    def finish_job(self, job_id: int, suggestion_id: Optional[int] = None, error: Optional[str] = None) -> None:
        with self.lock:
            self.db.execute("UPDATE jobs SET status = ?, suggestion_id = ?, error = ? WHERE id = ?",
                            ("failed" if error else "done", suggestion_id, error, job_id))

    # --- suggestions ---------------------------------------------------------------------------------------------

    def add_suggestion(self, suggestion: Dict[str, Any], validator_report: Dict[str, Any],
                       trigger: str = "manual") -> Dict[str, Any]:
        """Stores a GPT suggestion (or raises support_count of the same rule + location). :return: {id, new}"""
        location = suggestion.get("location") or "*"
        key = dedup_key(suggestion["rule"], location)
        with self.lock:
            row = self.db.execute("SELECT id, supporting_cases FROM suggestions WHERE dedup_key = ?", (key,)).fetchone()
            if row:
                cases = sorted(set(json.loads(row[1])) | set(suggestion.get("supporting_cases") or []))
                self.db.execute("UPDATE suggestions SET support_count = support_count + 1, supporting_cases = ? "
                                "WHERE id = ?", (json.dumps(cases), row[0]))
                return {"id": row[0], "new": False}
            cursor = self.db.execute(
                "INSERT INTO suggestions (created_at, location, rule, rationale, supporting_cases, "
                "new_variable_needed, confidence, validator_report, dedup_key, trigger, original_rule) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (now_iso(), location, suggestion["rule"], suggestion.get("rationale"),
                 json.dumps(suggestion.get("supporting_cases") or []), suggestion.get("new_variable_needed"),
                 suggestion.get("confidence"), json.dumps(validator_report, ensure_ascii=False), key, trigger,
                 suggestion["rule"]))
            return {"id": int(cursor.lastrowid), "new": True}

    def get(self, suggestion_id: int) -> Optional[Dict[str, Any]]:
        row = self.db.execute(f"SELECT {', '.join(COLUMNS)} FROM suggestions WHERE id = ?",
                              (suggestion_id,)).fetchone()
        return self._row(row) if row else None

    def list(self, status: Optional[str] = None, location: Optional[str] = None, confidence: Optional[str] = None,
             min_support: int = 1, limit: int = 200) -> List[Dict[str, Any]]:
        sql, args = f"SELECT {', '.join(COLUMNS)} FROM suggestions WHERE support_count >= ?", [min_support]
        for column, value in (("status", status), ("location", location), ("confidence", confidence)):
            if value:
                sql, args = sql + f" AND {column} = ?", args + [value]
        rows = self.db.execute(sql + " ORDER BY support_count DESC, id DESC LIMIT ?", (*args, limit)).fetchall()
        return [self._row(r) for r in rows]

    def update(self, suggestion_id: int, **fields) -> None:
        if not fields:
            return
        for key in fields:
            if key not in COLUMNS or key in ("id", "dedup_key"):
                raise ValueError(f"cannot update {key!r}")
        values = [json.dumps(v, ensure_ascii=False, default=str) if k in JSON_COLUMNS and v is not None else v
                  for k, v in fields.items()]
        with self.lock:
            self.db.execute(f"UPDATE suggestions SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?",
                            (*values, suggestion_id))

    def _row(self, row) -> Dict[str, Any]:
        item = dict(zip(COLUMNS, row, strict=True))
        for key in JSON_COLUMNS:
            if item.get(key):
                item[key] = json.loads(item[key])
        return item

    def close(self) -> None:
        self.db.close()
