"""
Device registry (F11 / TOD-22): every camera, microphone and sensor of the network, with its location.

SQLite ``data/devices.db`` (WAL, like the identity registry): the orchestrator writes what discovery finds, the
dashboard Devices page changes locations / names / ``enabled``, modules read it. A device is identified by a stable
``device_id`` = kind + hash of (node, hardware key), so the same USB camera keeps its id and location across
re-plugs and restarts. New devices are registered as ``unassigned`` and ``enabled``.

Changes are published as ``device_online`` / ``device_offline`` / ``device_updated`` events (source ``system``).
"""
import hashlib
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.events import UNASSIGNED, Event, make_event

KINDS = ("camera", "mic", "gas")
TRANSPORTS = ("local", "continuity", "browser", "rtsp", "edge", "picamera")
SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    device_id TEXT PRIMARY KEY, node_id TEXT NOT NULL, kind TEXT NOT NULL, transport TEXT NOT NULL,
    name TEXT NOT NULL, hw_key TEXT NOT NULL, location_id TEXT NOT NULL DEFAULT 'unassigned',
    enabled INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL DEFAULT 'offline', fps REAL, last_seen REAL,
    created_at REAL NOT NULL, info_json TEXT);
CREATE TABLE IF NOT EXISTS locations (location_id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL, token_sha256 TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL, last_seen REAL);
CREATE TABLE IF NOT EXISTS pairing_codes (code TEXT PRIMARY KEY, expires_at REAL NOT NULL);
"""
COLUMNS = ("device_id", "node_id", "kind", "transport", "name", "hw_key", "location_id", "enabled", "status", "fps",
           "last_seen", "created_at", "info_json")


def make_device_id(kind: str, node_id: str, hw_key: str) -> str:
    """Stable id, e.g. ``camera-3f2a9c01d4``."""
    return f"{kind}-{hashlib.sha1(f'{node_id}|{hw_key}'.encode()).hexdigest()[:10]}"


def slug(name: str) -> str:
    """Location id from a name: ``Room B`` → ``room_b`` (Cyrillic kept)."""
    value = re.sub(r"[^\w]+", "_", name.strip().lower(), flags=re.UNICODE).strip("_")
    return value[:64] or UNASSIGNED


class DeviceRegistry:
    def __init__(self, path: str = "data/devices.db", node_id: Optional[str] = None,
                 publish: Optional[Callable[[Event], None]] = None, clock: Callable[[], float] = time.time):
        """
        :param node_id: This machine (devices it registers belong to it)
        :param publish: Called with every device_* event; None = no bus
        """
        from core.events import default_node_id

        self.path = path
        self.node_id = node_id or default_node_id()
        self.publish = publish
        self.clock = clock
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(path, timeout=10, isolation_level=None, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=10000")
        self.db.executescript(SCHEMA)
        self.db.execute("INSERT OR IGNORE INTO locations (location_id, name, created_at) VALUES (?, ?, ?)",
                        (UNASSIGNED, "Unassigned", self.clock()))

    def close(self) -> None:
        with self._lock:
            self.db.close()

    # --- reads ------------------------------------------------------------------------------------------------------

    def _row(self, row) -> Dict[str, Any]:
        device = dict(zip(COLUMNS, row, strict=True))
        device["enabled"] = bool(device["enabled"])
        device["info"] = json.loads(device.pop("info_json") or "{}")
        return device

    def get(self, device_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.db.execute(f"SELECT {', '.join(COLUMNS)} FROM devices WHERE device_id = ?",
                                  (device_id,)).fetchone()
        return self._row(row) if row else None

    def list_devices(self, kind: Optional[str] = None, status: Optional[str] = None,
                     node_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query, args = f"SELECT {', '.join(COLUMNS)} FROM devices WHERE 1=1", []
        for column, value in (("kind", kind), ("status", status), ("node_id", node_id)):
            if value is not None:
                query += f" AND {column} = ?"
                args.append(value)
        with self._lock:
            rows = self.db.execute(query + " ORDER BY kind, created_at", args).fetchall()
        return [self._row(r) for r in rows]

    def list_locations(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.db.execute("SELECT location_id, name FROM locations ORDER BY location_id = ? DESC, name",
                                   (UNASSIGNED,)).fetchall()
        return [{"location_id": r[0], "name": r[1]} for r in rows]

    # --- writes -----------------------------------------------------------------------------------------------------

    def seen(self, kind: str, transport: str, name: str, hw_key: str, info: Optional[Dict[str, Any]] = None,
             location_id: Optional[str] = None, enabled: Optional[bool] = None,
             node_id: Optional[str] = None) -> Tuple[Dict[str, Any], bool]:
        """
        Registers (if new) and marks a device online.

        :param node_id: The node the device belongs to (a paired phone / Pi); this registry's node by default
        :param location_id: Location for a NEW device (a seed / the node's choice); existing devices keep theirs
        :param enabled: ``enabled`` for a NEW device
        :return: (device, changed) — changed = it was new, offline before, or its info changed
        """
        if kind not in KINDS or transport not in TRANSPORTS:
            raise ValueError(f"kind {kind!r} / transport {transport!r} not supported")
        node_id = node_id or self.node_id
        device_id = make_device_id(kind, node_id, hw_key)
        now, info_json = self.clock(), json.dumps(info or {}, sort_keys=True)
        with self._lock:
            old = self.get(device_id)
            if old is None:
                location = location_id or UNASSIGNED
                self._ensure_location(location)
                self.db.execute(f"INSERT INTO devices ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                                (device_id, node_id, kind, transport, name, hw_key, location,
                                 1 if enabled is None else int(enabled), "online", None, now, now, info_json))
            else:
                self.db.execute("UPDATE devices SET status = 'online', last_seen = ?, transport = ?, info_json = ? "
                                "WHERE device_id = ?", (now, transport, info_json, device_id))
        device = self.get(device_id)
        changed = old is None or old["status"] != "online" or old["info"] != device["info"]
        if old is None or old["status"] != "online":
            self._emit("device_online", device, new=old is None)
        return device, changed

    def offline(self, device_id: str, status: str = "offline") -> Optional[Dict[str, Any]]:
        """Marks a device offline (unplugged) or ``error`` (its pipeline keeps failing)."""
        with self._lock:
            old = self.get(device_id)
            if old is None or old["status"] == status:
                return old
            self.db.execute("UPDATE devices SET status = ? WHERE device_id = ?", (status, device_id))
        device = self.get(device_id)
        self._emit("device_offline", device)
        return device

    def update(self, device_id: str, location_id: Optional[str] = None, name: Optional[str] = None,
               enabled: Optional[bool] = None, fps: Optional[float] = None) -> Dict[str, Any]:
        """Dashboard / node edits; emits ``device_updated`` (fps-only updates are silent)."""
        old = self.get(device_id)
        if old is None:
            raise KeyError(f"Unknown device {device_id!r}")
        # Only real changes (an echo of the same values from a node must not bounce back and forth)
        location_id = None if location_id == old["location_id"] else location_id
        name = None if name is not None and name.strip() == old["name"] else name
        enabled = None if enabled is not None and bool(enabled) == old["enabled"] else enabled
        sets, args = [], []
        if location_id is not None:
            self._ensure_location(location_id)
            sets.append("location_id = ?")
            args.append(location_id)
        if name is not None:
            if not name.strip():
                raise ValueError("Empty device name")
            sets.append("name = ?")
            args.append(name.strip()[:80])
        if enabled is not None:
            sets.append("enabled = ?")
            args.append(int(enabled))
        if fps is not None:
            sets.append("fps = ?")
            args.append(round(float(fps), 1))
        if sets:
            with self._lock:
                self.db.execute(f"UPDATE devices SET {', '.join(sets)} WHERE device_id = ?", (*args, device_id))
        device = self.get(device_id)
        if location_id is not None or name is not None or enabled is not None:
            self._emit("device_updated", device)
        return device

    # --- pairing (nodes in other rooms: phones, Raspberry Pis) ---------------------------------------------------

    def pairing_code(self, ttl_sec: float = 600) -> str:
        """A one-time 6-digit code shown on the dashboard; a node exchanges it for a token within ``ttl_sec``."""
        code = f"{secrets.randbelow(1_000_000):06d}"
        with self._lock:
            self.db.execute("DELETE FROM pairing_codes WHERE expires_at < ?", (self.clock(),))
            self.db.execute("INSERT OR REPLACE INTO pairing_codes VALUES (?, ?)", (code, self.clock() + ttl_sec))
        return code

    def pair(self, code: str, name: str, kind: str = "browser") -> Tuple[str, str]:
        """
        :return: (node_id, token) — the token is shown once; only its hash is stored
        :raise PermissionError: wrong or expired code
        """
        with self._lock:
            row = self.db.execute("SELECT expires_at FROM pairing_codes WHERE code = ?",
                                  (str(code).strip(),)).fetchone()
            self.db.execute("DELETE FROM pairing_codes WHERE code = ?", (str(code).strip(),))  # one-time
            if row is None or row[0] < self.clock():
                raise PermissionError("wrong or expired pairing code")
            token = secrets.token_urlsafe(32)
            node_id = f"{kind}-{slug(name)[:24]}-{secrets.token_hex(3)}"
            self.db.execute("INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?)",
                            (node_id, name.strip()[:64] or node_id, kind, _sha256(token), self.clock(), None))
        return node_id, token

    def node_for_token(self, token: Optional[str]) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        with self._lock:
            row = self.db.execute("SELECT node_id, name, kind FROM nodes WHERE token_sha256 = ?",
                                  (_sha256(token),)).fetchone()
            if row:
                self.db.execute("UPDATE nodes SET last_seen = ? WHERE node_id = ?", (self.clock(), row[0]))
        return {"node_id": row[0], "name": row[1], "kind": row[2]} if row else None

    def list_nodes(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.db.execute("SELECT node_id, name, kind, created_at, last_seen FROM nodes "
                                   "ORDER BY name").fetchall()
        return [dict(zip(("node_id", "name", "kind", "created_at", "last_seen"), r, strict=True)) for r in rows]

    def revoke_node(self, node_id: str) -> bool:
        """Forgets a node's token (lost phone): it has to pair again."""
        with self._lock:
            return self.db.execute("DELETE FROM nodes WHERE node_id = ?", (node_id,)).rowcount > 0

    def add_location(self, name: str) -> Dict[str, Any]:
        location_id = slug(name)
        with self._lock:
            self.db.execute("INSERT OR IGNORE INTO locations (location_id, name, created_at) VALUES (?, ?, ?)",
                            (location_id, name.strip()[:64], self.clock()))
        return {"location_id": location_id, "name": name.strip()[:64]}

    def _ensure_location(self, location_id: str) -> None:
        with self._lock:
            self.db.execute("INSERT OR IGNORE INTO locations (location_id, name, created_at) VALUES (?, ?, ?)",
                            (location_id, location_id.replace("_", " ").title(), self.clock()))

    def _emit(self, type: str, device: Dict[str, Any], **extra) -> None:
        if self.publish is None:
            return
        payload = {key: device[key] for key in ("device_id", "node_id", "kind", "transport", "name", "location_id",
                                                "enabled", "status")}
        self.publish(make_event("system", type, {**payload, **extra}, device_id=device["device_id"],
                                node_id=device["node_id"], location_id=device["location_id"]))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
