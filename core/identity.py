"""
Unified identity registry (F9 / TOD-20): one place that owns every person the system knows about.

Faces (SFace, 128 float32) and voices (ECAPA, 192 float32) are embeddings attached to a person; ``person_id`` (uuid)
is the stable key in every event and log, ``display_name`` (``unknown_N`` with one global counter, or a name) is
only for display and is looked up at render time, so a rename changes old lines in the UI too.

SQLite (``data/identity.db``, WAL): several processes open it (faces, voices, aggregator, dashboard, CLI), every
write is one short ``BEGIN IMMEDIATE`` transaction. Each process keeps an in-memory cache of names and per-person
centroids; ``poll()`` reloads it when another connection committed (``PRAGMA data_version``), and mutations publish
``identity_*`` bus events so the other modules refresh at once. Every mutation is written to ``identity_events``.

Privacy: everything stays on the device (``data/`` is git-ignored); ``delete()`` removes a person's embeddings and
face crops.
"""
import json
import os
import re
import shutil
import socket
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.events import Event, make_event

MODALITIES = {"face": 128, "voice": 192}
UNKNOWN_RE = re.compile(r"^unknown_\d+$")
TOUCH_INTERVAL_SEC = 10.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    person_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, is_known INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL, updated_at REAL NOT NULL, merged_into TEXT, last_seen REAL, last_location TEXT);
CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT, person_id TEXT NOT NULL, vector BLOB NOT NULL, device_id TEXT,
    location_id TEXT, quality REAL NOT NULL DEFAULT 1.0, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS voice_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT, person_id TEXT NOT NULL, vector BLOB NOT NULL, device_id TEXT,
    location_id TEXT, duration_sec REAL NOT NULL DEFAULT 0, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS identity_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, action TEXT NOT NULL, person_id TEXT,
    details_json TEXT);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS face_person ON face_embeddings(person_id);
CREATE INDEX IF NOT EXISTS voice_person ON voice_embeddings(person_id);
"""
# Column that decides which K embeddings are kept per person
SCORE_COLUMN = {"face": "quality", "voice": "duration_sec"}


class NameTaken(ValueError):
    """Another person already has this name: merge them instead."""


def normalize(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def valid_name(name: str) -> bool:
    name = name.strip()
    return 0 < len(name) <= 64 and not UNKNOWN_RE.match(name) and not name.startswith((".", "_")) \
        and "/" not in name and "\\" not in name


def registry_from_config(cfg: Dict[str, Any], publish: Optional[Callable[[Event], None]] = None) -> "IdentityRegistry":
    """:param cfg: The ``identity`` config section"""
    faces_dir = cfg.get("faces_dir", "data/faces/persons") if cfg.get("store_faces", True) else None
    return IdentityRegistry(cfg.get("registry_path", "data/identity.db"), cfg, publish=publish, faces_dir=faces_dir)


class IdentityRegistry:
    def __init__(self, path: str = "data/identity.db", config: Optional[Dict[str, Any]] = None,
                 publish: Optional[Callable[[Event], None]] = None, faces_dir: Optional[str] = None,
                 clock: Callable[[], float] = time.time):
        """
        :param path: SQLite file (``:memory:`` for tests)
        :param config: The ``identity`` config section (thresholds, max_embeddings, device_id, location_id)
        :param publish: Called with every identity_* event (BaseModule.publish); None = no bus
        :param faces_dir: Folder for per-person face crops (``<faces_dir>/<person_id>/``); None = never stored
        """
        cfg = config or {}
        self.path = path
        self.publish = publish
        self.faces_dir = faces_dir
        self.clock = clock
        self.face_threshold = float(cfg.get("face_threshold", 0.45))
        self.voice_threshold = float(cfg.get("voice_threshold", 0.30))
        self.max_embeddings = int(cfg.get("max_embeddings", 20))
        self.device_id = cfg.get("device_id") or socket.gethostname()
        self.location_id = cfg.get("location_id") or "home"
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(path, timeout=10, isolation_level=None, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=10000")
        self.db.executescript(SCHEMA)
        self.persons: Dict[str, Dict[str, Any]] = {}
        self.counts: Dict[str, Dict[str, int]] = {}
        self.centroids: Dict[str, Tuple[List[str], np.ndarray]] = {}
        self._data_version: Optional[int] = None
        self.refresh()

    def close(self) -> None:
        with self._lock:
            self.db.close()

    # --- cache ------------------------------------------------------------------------------------------------------

    def refresh(self) -> None:
        """Reloads names, sample counts and centroids from the database."""
        with self._lock:
            rows = self.db.execute("SELECT person_id, display_name, is_known, created_at, updated_at, merged_into, "
                                   "last_seen, last_location FROM persons").fetchall()
            self.persons = {r[0]: {"person_id": r[0], "display_name": r[1], "is_known": bool(r[2]),
                                   "created_at": r[3], "updated_at": r[4], "merged_into": r[5],
                                   "last_seen": r[6], "last_location": r[7]} for r in rows}
            self.counts = {pid: {"face": 0, "voice": 0} for pid in self.persons}
            for modality, size in MODALITIES.items():
                vectors: Dict[str, List[np.ndarray]] = {}
                for pid, blob in self.db.execute(f"SELECT person_id, vector FROM {modality}_embeddings"):
                    vectors.setdefault(pid, []).append(np.frombuffer(blob, np.float32))
                pids = [pid for pid in vectors if pid in self.persons and not self.persons[pid]["merged_into"]]
                for pid in pids:
                    self.counts[pid][modality] = len(vectors[pid])
                matrix = np.array([normalize(np.mean(vectors[pid], axis=0)) for pid in pids], np.float32)
                self.centroids[modality] = (pids, matrix.reshape(-1, size))
            self._data_version = self.db.execute("PRAGMA data_version").fetchone()[0]

    def poll(self) -> bool:
        """
        Refreshes the cache when another connection (process) changed the database.

        :return: True if it was refreshed
        """
        with self._lock:
            if self.db.execute("PRAGMA data_version").fetchone()[0] == self._data_version:
                return False
            self.refresh()
            return True

    def handle_event(self, event: Event) -> bool:
        """Bus hook for reader processes: identity_* events refresh the cache. :return: True if handled"""
        if not event.type.startswith("identity_"):
            return False
        self.poll()
        return True

    # --- lookups ----------------------------------------------------------------------------------------------------

    def resolve(self, person_id: Optional[str]) -> Optional[str]:
        """:return: The live person a (possibly merged) id belongs to, or None if unknown / deleted"""
        seen = set()
        while person_id in self.persons and person_id not in seen:
            seen.add(person_id)
            target = self.persons[person_id]["merged_into"]
            if not target:
                return person_id
            person_id = target
        return None

    def name(self, person_id: Optional[str], default: Optional[str] = None) -> Optional[str]:
        pid = self.resolve(person_id)
        return self.persons[pid]["display_name"] if pid else default

    def is_known(self, person_id: Optional[str]) -> bool:
        pid = self.resolve(person_id)
        return bool(pid and self.persons[pid]["is_known"])

    def get(self, person_id: str) -> Optional[Dict[str, Any]]:
        pid = self.resolve(person_id)
        if pid is None:
            return None
        return {**self.persons[pid], "face_samples": self.counts[pid]["face"],
                "voice_samples": self.counts[pid]["voice"], "has_thumbnail": self.thumbnail(pid) is not None}

    def find(self, ref: str) -> Optional[str]:
        """
        :param ref: person_id, an id prefix (≥ 8 chars) or a display name (case-insensitive)
        :return: Live person id or None
        """
        if ref in self.persons:
            return self.resolve(ref)
        live = [p for p in self.persons.values() if not p["merged_into"]]
        by_name = [p["person_id"] for p in live if p["display_name"].casefold() == ref.strip().casefold()]
        if by_name:
            return by_name[0]
        by_prefix = [p["person_id"] for p in live if len(ref) >= 8 and p["person_id"].startswith(ref)]
        return by_prefix[0] if len(by_prefix) == 1 else None

    def list_persons(self) -> List[Dict[str, Any]]:
        """:return: Live persons, known first, then most recently seen"""
        people = [self.get(pid) for pid, p in self.persons.items() if not p["merged_into"]]
        return sorted(people, key=lambda p: (not p["is_known"], -(p["last_seen"] or p["created_at"])))

    def centroid(self, person_id: str, modality: str) -> Optional[np.ndarray]:
        pid = self.resolve(person_id)
        pids, matrix = self.centroids.get(modality, ([], None))
        return matrix[pids.index(pid)] if pid in pids else None

    def match(self, modality: str, vector, threshold: Optional[float] = None,
              known: Optional[bool] = None) -> Tuple[Optional[str], float]:
        """
        Cosine similarity against per-person centroids.

        :param known: True / False to search only known / unknown people
        :return: (person_id or None when below the threshold, best similarity)
        """
        pids, matrix = self.centroids.get(modality, ([], None))
        if known is not None:
            keep = [i for i, pid in enumerate(pids) if self.persons[pid]["is_known"] == known]
            pids, matrix = [pids[i] for i in keep], matrix[keep]
        if not pids:
            return None, 0.0
        similarities = matrix @ normalize(vector)
        best = int(np.argmax(similarities))
        similarity = float(similarities[best])
        if threshold is None:
            threshold = self.face_threshold if modality == "face" else self.voice_threshold
        return (pids[best] if similarity >= threshold else None), similarity

    def rank(self, modality: str, vector, k: int = 2) -> List[Tuple[str, float]]:
        """:return: The k most similar live persons as [(person_id, similarity)], best first"""
        pids, matrix = self.centroids.get(modality, ([], None))
        if not pids:
            return []
        similarities = matrix @ normalize(vector)
        order = np.argsort(-similarities)[:k]
        return [(pids[i], float(similarities[i])) for i in order]

    def match_face(self, vector, threshold: Optional[float] = None,
                   known: Optional[bool] = None) -> Tuple[Optional[str], float]:
        return self.match("face", vector, threshold, known)

    def match_voice(self, vector, threshold: Optional[float] = None,
                    known: Optional[bool] = None) -> Tuple[Optional[str], float]:
        return self.match("voice", vector, threshold, known)

    # --- mutations --------------------------------------------------------------------------------------------------

    @contextmanager
    def _transaction(self):
        with self._lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def _audit(self, db, action: str, person_id: Optional[str], **details) -> None:
        db.execute("INSERT INTO identity_events (ts, action, person_id, details_json) VALUES (?, ?, ?, ?)",
                   (self.clock(), action, person_id, json.dumps(details, ensure_ascii=False)))

    def _emit(self, type: str, person_id: str, **payload) -> Event:
        event = make_event("system", type, {"person_id": person_id, "display_name": self.name(person_id),
                                            **payload})
        if self.publish is not None:
            self.publish(event)
        return event

    def _insert_embedding(self, db, person_id: str, modality: str, vector, quality: float, duration_sec: float,
                          device_id: Optional[str], location_id: Optional[str]) -> None:
        vector = normalize(vector)
        if modality not in MODALITIES or vector.size != MODALITIES[modality]:
            raise ValueError(f"{modality} embedding must have {MODALITIES.get(modality)} values, got {vector.size}")
        score_column = SCORE_COLUMN[modality]
        db.execute(f"INSERT INTO {modality}_embeddings (person_id, vector, device_id, location_id, {score_column}, "
                   f"created_at) VALUES (?, ?, ?, ?, ?, ?)",
                   (person_id, vector.tobytes(), device_id or self.device_id, location_id or self.location_id,
                    float(quality if modality == "face" else duration_sec), self.clock()))
        # Keep the K best (ties: the newest)
        db.execute(f"DELETE FROM {modality}_embeddings WHERE person_id = ? AND id NOT IN (SELECT id FROM "
                   f"{modality}_embeddings WHERE person_id = ? ORDER BY {score_column} DESC, id DESC LIMIT ?)",
                   (person_id, person_id, self.max_embeddings))

    def create_person(self, display_name: str) -> str:
        """Creates a known person (enrollment). :return: person_id"""
        if not valid_name(display_name):
            raise ValueError(f"Invalid name {display_name!r}")
        with self._transaction() as db:
            self._check_free(db, display_name.strip())
            person_id, now = str(uuid.uuid4()), self.clock()
            db.execute("INSERT INTO persons (person_id, display_name, is_known, created_at, updated_at) "
                       "VALUES (?, ?, 1, ?, ?)", (person_id, display_name.strip(), now, now))
            self._audit(db, "create", person_id, display_name=display_name.strip(), modality=None)
        self.refresh()
        self._emit("identity_created", person_id, is_known=True)
        return person_id

    def create_unknown(self, modality: str, vector, device_id: Optional[str] = None,
                       location_id: Optional[str] = None, quality: float = 1.0, duration_sec: float = 0.0) -> str:
        """
        New ``unknown_N`` person (N from one global counter shared by faces and voices; never reused).

        :return: person_id
        """
        with self._transaction() as db:
            row = db.execute("SELECT value FROM meta WHERE key = 'unknown_counter'").fetchone()
            number = (int(row[0]) if row else 0) + 1
            db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('unknown_counter', ?)", (str(number),))
            person_id, now = str(uuid.uuid4()), self.clock()
            name = f"unknown_{number}"
            db.execute("INSERT INTO persons (person_id, display_name, is_known, created_at, updated_at, last_seen, "
                       "last_location) VALUES (?, ?, 0, ?, ?, ?, ?)",
                       (person_id, name, now, now, now, location_id or self.location_id))
            self._insert_embedding(db, person_id, modality, vector, quality, duration_sec, device_id, location_id)
            self._audit(db, "create", person_id, display_name=name, modality=modality)
        self.refresh()
        self._emit("identity_created", person_id, is_known=False, modality=modality)
        return person_id

    def ensure_counter(self, at_least: int) -> None:
        """Makes the next ``unknown_N`` larger than ``at_least`` (imports keep their old numbers)."""
        with self._transaction() as db:
            row = db.execute("SELECT value FROM meta WHERE key = 'unknown_counter'").fetchone()
            if not row or int(row[0]) < at_least:
                db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('unknown_counter', ?)", (str(at_least),))

    def add_embedding(self, person_id: str, modality: str, vector, quality: float = 1.0, duration_sec: float = 0.0,
                      device_id: Optional[str] = None, location_id: Optional[str] = None) -> None:
        """Adds a sample (keeps the K best by quality / duration) and recomputes the centroid. No bus event."""
        pid = self._require(person_id)
        with self._transaction() as db:
            self._insert_embedding(db, pid, modality, vector, quality, duration_sec, device_id, location_id)
        self.refresh()

    def link(self, person_id: str, modality: str, vector, quality: float = 1.0, duration_sec: float = 0.0) -> Event:
        """Attaches a voice to a face identity (or vice versa); emits ``identity_linked``."""
        pid = self._require(person_id)
        with self._transaction() as db:
            self._insert_embedding(db, pid, modality, vector, quality, duration_sec, None, None)
            self._audit(db, f"link_{modality}", pid)
        self.refresh()
        return self._emit("identity_linked", pid, modality=modality)

    def rename(self, person_id: str, new_name: str) -> Event:
        """
        Sets the display name and marks the person known; emits ``identity_renamed``.

        :raise NameTaken: another person has this name (merge them instead)
        """
        pid = self._require(person_id)
        new_name = new_name.strip()
        if not valid_name(new_name):
            raise ValueError(f"Invalid name {new_name!r} (1–64 chars, not unknown_N)")
        previous = self.persons[pid]["display_name"]
        with self._transaction() as db:
            self._check_free(db, new_name, pid)
            db.execute("UPDATE persons SET display_name = ?, is_known = 1, updated_at = ? WHERE person_id = ?",
                       (new_name, self.clock(), pid))
            self._audit(db, "rename", pid, previous=previous, display_name=new_name)
        self.refresh()
        return self._emit("identity_renamed", pid, previous=previous)

    def merge(self, source_id: str, target_id: str, action: str = "merge") -> Event:
        """
        Moves all face + voice embeddings (and face crops) of ``source`` to ``target``; source gets ``merged_into``
        so old person_ids in logs still resolve. A known source names an unknown target.

        :param action: ``merge`` (user) or ``link_voice`` (automatic face ↔ voice link) — audit + event type
        """
        source, target = self._require(source_id), self._require(target_id)
        if source == target:
            raise ValueError("Cannot merge a person into itself")
        src, dst = self.persons[source], self.persons[target]
        with self._transaction() as db:
            for modality in MODALITIES:
                db.execute(f"UPDATE {modality}_embeddings SET person_id = ? WHERE person_id = ?", (target, source))
            now = self.clock()
            db.execute("UPDATE persons SET merged_into = ?, updated_at = ? WHERE person_id = ?", (target, now, source))
            if src["is_known"] and not dst["is_known"]:
                db.execute("UPDATE persons SET display_name = ?, is_known = 1 WHERE person_id = ?",
                           (src["display_name"], target))
            last = max((src["last_seen"] or 0, src["last_location"]), (dst["last_seen"] or 0, dst["last_location"]),
                       key=lambda x: x[0])
            db.execute("UPDATE persons SET updated_at = ?, last_seen = ?, last_location = ? WHERE person_id = ?",
                       (now, last[0] or None, last[1], target))
            for modality in MODALITIES:  # the merged person may now exceed K samples
                db.execute(f"DELETE FROM {modality}_embeddings WHERE person_id = ? AND id NOT IN (SELECT id FROM "
                           f"{modality}_embeddings WHERE person_id = ? ORDER BY {SCORE_COLUMN[modality]} DESC, "
                           f"id DESC LIMIT ?)", (target, target, self.max_embeddings))
            self._audit(db, action, target, source_id=source, source_name=src["display_name"],
                        target_name=dst["display_name"])
        self._move_crops(source, target)
        self.refresh()
        event_type = "identity_linked" if action == "link_voice" else "identity_merged"
        return self._emit(event_type, target, source_id=source, source_name=src["display_name"])

    def delete(self, person_id: str, reason: str = "user") -> Event:
        """Removes a person, its embeddings and face crops (privacy / GDPR request); emits ``identity_deleted``."""
        pid = self._require(person_id)
        name = self.persons[pid]["display_name"]
        merged = [p for p, row in self.persons.items() if row["merged_into"] and self.resolve(p) == pid]
        with self._transaction() as db:
            for victim in [pid] + merged:
                for modality in MODALITIES:
                    db.execute(f"DELETE FROM {modality}_embeddings WHERE person_id = ?", (victim,))
                db.execute("DELETE FROM persons WHERE person_id = ?", (victim,))
            self._audit(db, "delete", pid, display_name=name, reason=reason)
        if self.faces_dir:
            shutil.rmtree(os.path.join(self.faces_dir, pid), ignore_errors=True)
        self.refresh()
        event = make_event("system", "identity_deleted", {"person_id": pid, "display_name": name, "reason": reason})
        if self.publish is not None:
            self.publish(event)
        return event

    def touch(self, person_id: str, location_id: Optional[str] = None) -> None:
        """Records "last seen" (written at most every 10 s per person)."""
        pid = self.resolve(person_id)
        if pid is None:
            return
        now = self.clock()
        if now - (self.persons[pid]["last_seen"] or 0) < TOUCH_INTERVAL_SEC:
            return
        location = location_id or self.location_id
        with self._lock:
            self.db.execute("UPDATE persons SET last_seen = ?, last_location = ? WHERE person_id = ?",
                            (now, location, pid))
            self._data_version = self.db.execute("PRAGMA data_version").fetchone()[0]
        self.persons[pid].update(last_seen=now, last_location=location)

    def purge_unknowns(self, retention_days: float) -> List[str]:
        """Deletes unknown people not seen for ``retention_days`` (consent / GDPR). :return: deleted ids"""
        cutoff = self.clock() - retention_days * 86400
        old = [pid for pid, p in self.persons.items() if not p["merged_into"] and not p["is_known"]
               and (p["last_seen"] or p["created_at"]) < cutoff]
        for pid in old:
            self.delete(pid, reason="retention")
        return old

    def history(self, person_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """:return: identity_events rows, newest first"""
        query, args = "SELECT ts, action, person_id, details_json FROM identity_events", []
        if person_id:
            query, args = query + " WHERE person_id = ?", [person_id]
        with self._lock:
            rows = self.db.execute(query + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        return [{"ts": r[0], "action": r[1], "person_id": r[2], **json.loads(r[3] or "{}")} for r in rows]

    # --- face crops (thumbnails for the dashboard; only when faces_dir is set) ---------------------------------------

    def save_face_crop(self, person_id: str, image, max_images: int = 10) -> Optional[str]:
        """:return: Path of the stored crop, or None (storage off or enough crops)"""
        pid = self.resolve(person_id)
        if not self.faces_dir or pid is None or image is None:
            return None
        import cv2

        folder = os.path.join(self.faces_dir, pid)
        os.makedirs(folder, exist_ok=True)
        existing = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))
        if len(existing) >= max_images:
            return None
        path = os.path.join(folder, f"face_{int(self.clock() * 1000)}_{len(existing):03d}.jpg")
        cv2.imwrite(path, image)
        return path

    def thumbnail(self, person_id: str) -> Optional[str]:
        if not self.faces_dir:
            return None
        folder = os.path.join(self.faces_dir, person_id)
        files = sorted(f for f in os.listdir(folder) if f.endswith(".jpg")) if os.path.isdir(folder) else []
        return os.path.join(folder, files[0]) if files else None

    def _move_crops(self, source: str, target: str) -> None:
        if not self.faces_dir or not os.path.isdir(os.path.join(self.faces_dir, source)):
            return
        os.makedirs(os.path.join(self.faces_dir, target), exist_ok=True)
        for file in os.listdir(os.path.join(self.faces_dir, source)):
            shutil.move(os.path.join(self.faces_dir, source, file), os.path.join(self.faces_dir, target, file))
        shutil.rmtree(os.path.join(self.faces_dir, source), ignore_errors=True)

    # --- helpers ----------------------------------------------------------------------------------------------------

    def _require(self, person_id: str) -> str:
        pid = self.resolve(person_id)
        if pid is None:
            self.refresh()  # created by another process a moment ago?
            pid = self.resolve(person_id)
        if pid is None:
            raise KeyError(f"Unknown person {person_id!r}")
        return pid

    @staticmethod
    def _check_free(db, name: str, own_id: Optional[str] = None) -> None:
        row = db.execute("SELECT person_id FROM persons WHERE display_name = ? COLLATE NOCASE AND merged_into IS NULL "
                         "AND person_id != ?", (name, own_id or "")).fetchone()
        if row:
            raise NameTaken(f"{name!r} already exists ({row[0][:8]}): merge the two people instead")
