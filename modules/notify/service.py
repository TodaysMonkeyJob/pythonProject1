"""
Notifications (F12 / TOD-23): one per location state change worth telling someone about.

Every notification is stored in ``data/notifications.db`` and published as a ``notification`` event (dashboard toast,
notification centre, browser Notification API). ``danger`` notifications also go to Telegram when
``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID`` are set (plain HTTPS to the Bot API, no extra package). The same
location + reason is notified at most once per ``repeat_min``. ``ack()`` records who acknowledged and when.
"""
import json
import os
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
import uuid
from typing import Any, Callable, Dict, List, Optional

from core.events import Event, make_event
from system_setup.logger_setup import init_logger

logger = init_logger()

SCHEMA = """
CREATE TABLE IF NOT EXISTS notifications (
    id TEXT PRIMARY KEY, ts REAL NOT NULL, location_id TEXT NOT NULL, level TEXT NOT NULL, title TEXT NOT NULL,
    body TEXT NOT NULL, reason TEXT NOT NULL, event_ids_json TEXT, channels_json TEXT,
    acknowledged_by TEXT, acknowledged_at REAL);
CREATE INDEX IF NOT EXISTS notifications_ts ON notifications (ts);
"""
ICON = {"danger": "🔴", "attention": "🟡", "offline": "⚪", "safe": "🟢"}


def telegram_sender(token: str, chat_id: str, timeout: float = 10.0) -> Callable[[str], bool]:
    def send(text: str) -> bool:
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": text, "disable_web_page_preview": "true"}).encode()
        try:
            with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data,
                                        timeout=timeout) as response:
                return json.load(response).get("ok", False)
        except Exception as e:  # offline / wrong token: the dashboard still has it
            logger.warning(f"Telegram notification failed: {type(e).__name__}")
            return False
    return send


class NotificationService:
    def __init__(self, path: str = "data/notifications.db", publish: Optional[Callable[[Event], None]] = None,
                 repeat_min: float = 5.0, telegram: Optional[Callable[[str], bool]] = None,
                 telegram_levels=("danger",), clock: Callable[[], float] = time.time):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, timeout=10, isolation_level=None, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self.publish = publish
        self.repeat_sec = float(repeat_min) * 60
        self.telegram = telegram
        self.telegram_levels = tuple(telegram_levels)
        self.clock = clock

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], publish=None) -> "NotificationService":
        """:param cfg: The ``locations`` config section (``notify``, telegram credentials from .env)"""
        notify = cfg.get("notify", {}) or {}
        token, chat = cfg.get("telegram_bot_token"), cfg.get("telegram_chat_id")
        telegram = telegram_sender(token, str(chat)) if token and chat and notify.get("telegram", True) else None
        return cls(notify.get("path", "data/notifications.db"), publish, notify.get("repeat_min", 5), telegram,
                   tuple(notify.get("telegram_levels", ("danger",))))

    def close(self) -> None:
        self.db.close()

    def notify(self, location_id: str, level: str, title: str, body: str, reason: str,
               event_ids: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """:return: the notification, or None when the same location + reason was notified recently"""
        now = self.clock()
        with self._lock:
            recent = self.db.execute("SELECT 1 FROM notifications WHERE location_id = ? AND reason = ? AND ts > ?",
                                     (location_id, reason, now - self.repeat_sec)).fetchone()
            if recent:
                return None
            channels = ["dashboard", "browser"]
            if self.telegram is not None and level in self.telegram_levels:
                channels.append("telegram")
            note = {"id": str(uuid.uuid4()), "ts": now, "location_id": location_id, "level": level, "title": title,
                    "body": body, "reason": reason, "event_ids": list(event_ids or [])[-10:], "channels": channels,
                    "acknowledged_by": None, "acknowledged_at": None}
            self.db.execute("INSERT INTO notifications VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (note["id"], now, location_id, level, title, body, reason, json.dumps(note["event_ids"]),
                             json.dumps(channels), None, None))
        if self.publish is not None:
            self.publish(make_event("system", "notification", note, location_id=location_id,
                                    severity="critical" if level == "danger" else "warning"))
        if "telegram" in channels:
            threading.Thread(target=self.telegram, args=(f"{ICON.get(level, '')} {title}\n{body}",),
                             daemon=True).start()
        return note

    def ack(self, notification_id: str, by: str = "dashboard") -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.db.execute("SELECT location_id, acknowledged_at FROM notifications WHERE id = ?",
                                  (notification_id,)).fetchone()
            if row is None:
                return None
            if row[1] is None:
                self.db.execute("UPDATE notifications SET acknowledged_by = ?, acknowledged_at = ? WHERE id = ?",
                                (by[:64], self.clock(), notification_id))
        return self.get(notification_id)

    def ack_location(self, location_id: str, by: str = "dashboard") -> int:
        with self._lock:
            return self.db.execute("UPDATE notifications SET acknowledged_by = ?, acknowledged_at = ? "
                                   "WHERE location_id = ? AND acknowledged_at IS NULL",
                                   (by[:64], self.clock(), location_id)).rowcount

    def get(self, notification_id: str) -> Optional[Dict[str, Any]]:
        rows = self.list(where="id = ?", args=(notification_id,))
        return rows[0] if rows else None

    def list(self, unacked: bool = False, limit: int = 100, where: str = "1=1", args=()) -> List[Dict[str, Any]]:
        query = f"SELECT id, ts, location_id, level, title, body, reason, event_ids_json, channels_json, " \
                f"acknowledged_by, acknowledged_at FROM notifications WHERE {where}"
        if unacked:
            query += " AND acknowledged_at IS NULL"
        with self._lock:
            rows = self.db.execute(query + " ORDER BY ts DESC LIMIT ?", (*args, limit)).fetchall()
        return [{"id": r[0], "ts": r[1], "location_id": r[2], "level": r[3], "title": r[4], "body": r[5],
                 "reason": r[6], "event_ids": json.loads(r[7] or "[]"), "channels": json.loads(r[8] or "[]"),
                 "acknowledged_by": r[9], "acknowledged_at": r[10]} for r in rows]
