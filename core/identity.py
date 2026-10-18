"""
Unified identity registry (F9 / TOD-20): one place that owns every person the system knows about.

Faces (SFace, 128 float32) and voices (ECAPA, 192 float32) are embeddings attached to a person; ``person_id`` (uuid)
is the stable key in every event and log, ``display_name`` (``unknown_N`` with one global counter, or a name) is
only for display and is looked up at render time, so a rename changes old lines in the UI too.

SQLite (``data/identity.db``, WAL): several processes open it (faces, voices, aggregator, dashboard, CLI), every
write is one short ``BEGIN IMMEDIATE`` transaction. Each process keeps an in-memory cache of names and per-person
centroids; ``poll()`` reloads it when another connection committed (``PRAGMA data_version``), and mutations publish
``identity_*`` bus events so the other modules refresh at once. Every mutation is written to ``identity_events``.

Faces (F14 / TOD-25) are a multi-prototype gallery (``core.gallery``): up to ``max_face_embeddings_per_person``
vectors with tier / yaw / pitch / brightness buckets, matched by the mean of the top-3 similarities; new vectors go
through the admission rule (good quality, not redundant, bucket quotas). ``cannot_link`` holds pairs that are never
the same person (seen in one frame, or a user's "not the same person"); ``merge_candidates`` is the duplicates
review queue of ``modules.vision.dedup``. A merge moves only admitted face embeddings and archives the rest
(``archived_by`` = the merge's ``identity_events`` id), so ``unmerge()`` restores both people within ``undo_days``.

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
from core.gallery import (ADMISSIBLE_TIERS, Prototype, Quotas, admit, brightness_bucket, pitch_bucket, top_k_mean,
                          yaw_bucket)

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
CREATE TABLE IF NOT EXISTS cannot_link (
    person_a TEXT NOT NULL, person_b TEXT NOT NULL, reason TEXT NOT NULL, created_at REAL NOT NULL,
    PRIMARY KEY (person_a, person_b));
CREATE TABLE IF NOT EXISTS merge_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT, person_a TEXT NOT NULL, person_b TEXT NOT NULL, score REAL NOT NULL,
    evidence_json TEXT, status TEXT NOT NULL DEFAULT 'pending', created_at REAL NOT NULL, updated_at REAL NOT NULL,
    decided_by TEXT);
CREATE INDEX IF NOT EXISTS face_person ON face_embeddings(person_id);
CREATE INDEX IF NOT EXISTS voice_person ON voice_embeddings(person_id);
"""
# Columns added after the first release (ALTER TABLE on open)
MIGRATIONS = {
    "face_embeddings": {"tier": "TEXT", "yaw_bucket": "TEXT", "pitch_bucket": "TEXT", "brightness_bucket": "TEXT",
                        "archived_by": "INTEGER"},
    "voice_embeddings": {"archived_by": "INTEGER"},
}
MERGE_ACTIONS = ("merge", "auto_merge", "review_merge", "link_voice")
TOP_K = 3   # face match score = mean of the top-3 prototype similarities


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
    # F14: without thumbnails, unknowns still keep a few aligned crops so a human can review possible duplicates
    review_dir = cfg.get("review_crops_dir", "data/faces/review") \
        if faces_dir is None and cfg.get("store_review_crops", True) else None
    return IdentityRegistry(cfg.get("registry_path", "data/identity.db"), cfg, publish=publish, faces_dir=faces_dir,
                            review_dir=review_dir)


class IdentityRegistry:
    def __init__(self, path: str = "data/identity.db", config: Optional[Dict[str, Any]] = None,
                 publish: Optional[Callable[[Event], None]] = None, faces_dir: Optional[str] = None,
                 clock: Callable[[], float] = time.time, review_dir: Optional[str] = None):
        """
        :param path: SQLite file (``:memory:`` for tests)
        :param config: The ``identity`` config section (thresholds, max_embeddings, device_id, location_id)
        :param publish: Called with every identity_* event (BaseModule.publish); None = no bus
        :param faces_dir: Folder for per-person face crops (``<faces_dir>/<person_id>/``); None = never stored
        :param review_dir: Used instead when there is no faces_dir: ≤ 3 aligned crops per UNKNOWN person, deleted
                           once it is named, merged, rejected as a duplicate or deleted
        """
        cfg = config or {}
        self.path = path
        self.publish = publish
        self.faces_dir = faces_dir
        self.review_dir = review_dir
        self.clock = clock
        self.face_threshold = float(cfg.get("face_threshold", 0.45))
        self.voice_threshold = float(cfg.get("voice_threshold", 0.30))
        self.max_embeddings = int(cfg.get("max_embeddings", 20))        # voices (K best by duration)
        self.max_faces = int(cfg.get("max_face_embeddings_per_person", 30))
        self.redundancy = float(cfg.get("redundant_similarity", 0.9))
        self.quotas = Quotas.from_dict(cfg.get("gallery_quotas"))
        self.undo_days = float(cfg.get("undo_days", 7))
        self.device_id = cfg.get("device_id") or socket.gethostname()
        self.location_id = cfg.get("location_id") or "home"
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(path, timeout=10, isolation_level=None, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=10000")
        self.db.executescript(SCHEMA)
        self._migrate()
        self.persons: Dict[str, Dict[str, Any]] = {}
        self.counts: Dict[str, Dict[str, int]] = {}
        self.centroids: Dict[str, Tuple[List[str], np.ndarray]] = {}
        self.galleries: Dict[str, List[Prototype]] = {}     # live person → face prototypes
        self._gallery_matrix: Tuple[List[str], np.ndarray, np.ndarray] = ([], np.zeros((0, 128), np.float32),
                                                                          np.zeros(0, np.int32))
        self.cannot_links: set = set()                      # frozenset({id_a, id_b}) as stored
        self._blocked_live: set = set()
        self._data_version: Optional[int] = None
        self.refresh()

    def _migrate(self) -> None:
        for table, columns in MIGRATIONS.items():
            existing = {row[1] for row in self.db.execute(f"PRAGMA table_info({table})")}
            for column, kind in columns.items():
                if column not in existing:
                    try:
                        self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
                    except sqlite3.OperationalError as e:   # another process migrated a moment ago
                        if "duplicate column" not in str(e):
                            raise

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
            live = {pid for pid, p in self.persons.items() if not p["merged_into"]}
            self.galleries = {}
            for row_id, pid, blob, tier, yaw, pitch, light in self.db.execute(
                    "SELECT id, person_id, vector, tier, yaw_bucket, pitch_bucket, brightness_bucket "
                    "FROM face_embeddings WHERE archived_by IS NULL ORDER BY id"):
                if pid in live:
                    self.galleries.setdefault(pid, []).append(Prototype(
                        row_id, np.frombuffer(blob, np.float32), yaw,
                        {"tier": tier, "pitch_bucket": pitch, "brightness_bucket": light}))
            for modality, size in MODALITIES.items():
                vectors: Dict[str, List[np.ndarray]] = {}
                for pid, blob in self.db.execute(f"SELECT person_id, vector FROM {modality}_embeddings "
                                                 f"WHERE archived_by IS NULL"):
                    vectors.setdefault(pid, []).append(np.frombuffer(blob, np.float32))
                pids = [pid for pid in vectors if pid in live]
                for pid in pids:
                    self.counts[pid][modality] = len(vectors[pid])
                matrix = np.array([normalize(np.mean(vectors[pid], axis=0)) for pid in pids], np.float32)
                self.centroids[modality] = (pids, matrix.reshape(-1, size))
            owners = list(self.galleries)
            stacked = [p.vector for pid in owners for p in self.galleries[pid]]
            self._gallery_matrix = (owners, np.array(stacked, np.float32).reshape(-1, MODALITIES["face"]),
                                    np.array([i for i, pid in enumerate(owners) for _ in self.galleries[pid]],
                                             np.int32))
            self.cannot_links = {frozenset(r) for r in self.db.execute("SELECT person_a, person_b FROM cannot_link")}
            self._blocked_live = set()      # the stored pairs mapped onto the live people they belong to now
            for pair in self.cannot_links:
                resolved = frozenset(self.resolve(pid) for pid in pair)
                if len(resolved) == 2 and None not in resolved:
                    self._blocked_live.add(resolved)
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

    def scores(self, modality: str, vector) -> Tuple[List[str], np.ndarray]:
        """
        Similarity of a query to every live person: faces = mean of the top-3 prototype similarities, voices =
        cosine to the centroid.

        :return: (person_ids, scores)
        """
        query = normalize(vector)
        if modality != "face":
            pids, matrix = self.centroids.get(modality, ([], None))
            return (list(pids), matrix @ query) if pids else ([], np.zeros(0, np.float32))
        owners, matrix, index = self._gallery_matrix
        if not owners:
            return [], np.zeros(0, np.float32)
        similarities = matrix @ query
        bounds = np.searchsorted(index, np.arange(len(owners) + 1))
        return owners, np.array([top_k_mean(similarities[bounds[i]:bounds[i + 1]], TOP_K)
                                 for i in range(len(owners))], np.float32)

    def match(self, modality: str, vector, threshold: Optional[float] = None,
              known: Optional[bool] = None) -> Tuple[Optional[str], float]:
        """
        Best live person by ``scores()``.

        :param known: True / False to search only known / unknown people
        :return: (person_id or None when below the threshold, best similarity)
        """
        pids, similarities = self.scores(modality, vector)
        if known is not None:
            keep = [i for i, pid in enumerate(pids) if self.persons[pid]["is_known"] == known]
            pids, similarities = [pids[i] for i in keep], similarities[keep]
        if not pids:
            return None, 0.0
        best = int(np.argmax(similarities))
        similarity = float(similarities[best])
        if threshold is None:
            threshold = self.face_threshold if modality == "face" else self.voice_threshold
        return (pids[best] if similarity >= threshold else None), similarity

    def similarity(self, person_id: str, modality: str, vector) -> float:
        """:return: ``scores()`` of one person (0 when it has no samples)"""
        pid = self.resolve(person_id)
        if modality == "face":
            gallery = self.galleries.get(pid)
            return top_k_mean(np.stack([p.vector for p in gallery]) @ normalize(vector), TOP_K) if gallery else 0.0
        reference = self.centroid(pid, modality) if pid else None
        return float(reference @ normalize(vector)) if reference is not None else 0.0

    def rank(self, modality: str, vector, k: int = 2) -> List[Tuple[str, float]]:
        """:return: The k most similar live persons as [(person_id, similarity)], best first"""
        pids, similarities = self.scores(modality, vector)
        if not pids:
            return []
        order = np.argsort(-similarities)[:k]
        return [(pids[i], float(similarities[i])) for i in order]

    def gallery(self, person_id: str) -> np.ndarray:
        """:return: (n, 128) live face prototypes of a person"""
        pid = self.resolve(person_id)
        prototypes = self.galleries.get(pid, [])
        return np.array([p.vector for p in prototypes], np.float32).reshape(-1, MODALITIES["face"])

    def blocked(self, person_a: str, person_b: str) -> bool:
        """:return: True when the two (live) people must never be merged (cannot-link on any of their ids)"""
        a, b = self.resolve(person_a), self.resolve(person_b)
        return a is not None and b is not None and a != b and frozenset((a, b)) in self._blocked_live

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
                          device_id: Optional[str], location_id: Optional[str],
                          meta: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """
        Faces go through the gallery admission rule (evicted prototypes are deleted); voices keep the K longest.

        :param meta: Face quality (``modules.vision.face_quality.assess``): tier, yaw, pitch, brightness
        :return: New row id, or None when the face was not admitted
        """
        vector = normalize(vector)
        if modality not in MODALITIES or vector.size != MODALITIES[modality]:
            raise ValueError(f"{modality} embedding must have {MODALITIES.get(modality)} values, got {vector.size}")
        if modality == "face":
            meta = meta or {}
            buckets = (yaw_bucket(meta.get("yaw")), pitch_bucket(meta.get("pitch")),
                       brightness_bucket(meta.get("brightness")))
            decision = admit(self._face_rows(db, person_id), vector, meta.get("tier"), buckets[0], self.max_faces,
                             self.redundancy, self.quotas)
            if not decision.accepted:
                return None
            for victim in decision.evict:
                db.execute("DELETE FROM face_embeddings WHERE id = ?", (victim,))
            cursor = db.execute(
                "INSERT INTO face_embeddings (person_id, vector, device_id, location_id, quality, created_at, tier, "
                "yaw_bucket, pitch_bucket, brightness_bucket) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (person_id, vector.tobytes(), device_id or self.device_id, location_id or self.location_id,
                 float(meta.get("score", quality)), self.clock(), meta.get("tier"), *buckets))
            return cursor.lastrowid
        cursor = db.execute("INSERT INTO voice_embeddings (person_id, vector, device_id, location_id, duration_sec, "
                            "created_at) VALUES (?, ?, ?, ?, ?, ?)",
                            (person_id, vector.tobytes(), device_id or self.device_id,
                             location_id or self.location_id, float(duration_sec), self.clock()))
        # Keep the K best (ties: the newest)
        db.execute("DELETE FROM voice_embeddings WHERE person_id = ? AND archived_by IS NULL AND id NOT IN (SELECT id "
                   "FROM voice_embeddings WHERE person_id = ? AND archived_by IS NULL ORDER BY duration_sec DESC, "
                   "id DESC LIMIT ?)", (person_id, person_id, self.max_embeddings))
        return cursor.lastrowid

    @staticmethod
    def _face_rows(db, person_id: str) -> List[Prototype]:
        return [Prototype(r[0], np.frombuffer(r[1], np.float32), r[2], {"tier": r[3]}) for r in db.execute(
            "SELECT id, vector, yaw_bucket, tier FROM face_embeddings WHERE person_id = ? AND archived_by IS NULL "
            "ORDER BY id", (person_id,))]

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
                       location_id: Optional[str] = None, quality: float = 1.0, duration_sec: float = 0.0,
                       meta: Optional[Dict[str, Any]] = None, details: Optional[Dict[str, Any]] = None) -> str:
        """
        New ``unknown_N`` person (N from one global counter shared by faces and voices; never reused).

        :param meta: Face quality (tier must be admissible: a ``usable`` face never creates a person)
        :param details: Extra audit details (e.g. ``continuity``: tracks lost just before, for the dedup job)
        :return: person_id
        """
        if modality == "face" and (meta or {}).get("tier") not in ADMISSIBLE_TIERS:
            raise ValueError(f"A {meta.get('tier')!r} face cannot create a person")
        with self._transaction() as db:
            row = db.execute("SELECT value FROM meta WHERE key = 'unknown_counter'").fetchone()
            number = (int(row[0]) if row else 0) + 1
            db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('unknown_counter', ?)", (str(number),))
            person_id, now = str(uuid.uuid4()), self.clock()
            name = f"unknown_{number}"
            db.execute("INSERT INTO persons (person_id, display_name, is_known, created_at, updated_at, last_seen, "
                       "last_location) VALUES (?, ?, 0, ?, ?, ?, ?)",
                       (person_id, name, now, now, now, location_id or self.location_id))
            self._insert_embedding(db, person_id, modality, vector, quality, duration_sec, device_id, location_id,
                                   meta)
            self._audit(db, "create", person_id, display_name=name, modality=modality,
                        device_id=device_id or self.device_id, **(details or {}))
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
                      device_id: Optional[str] = None, location_id: Optional[str] = None,
                      meta: Optional[Dict[str, Any]] = None) -> bool:
        """
        Adds a sample (faces: gallery admission; voices: the K longest) and refreshes the cache. No bus event.

        :return: True if it was stored
        """
        pid = self._require(person_id)
        with self._transaction() as db:
            stored = self._insert_embedding(db, pid, modality, vector, quality, duration_sec, device_id, location_id,
                                            meta) is not None
        if stored:
            self.refresh()
        return stored

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
        self.drop_review_crops(pid)
        self.refresh()
        return self._emit("identity_renamed", pid, previous=previous)

    def merge(self, source_id: str, target_id: str, action: str = "merge", **details) -> Event:
        """
        Merges ``source`` into ``target``: every source face embedding goes through the target's gallery admission
        (moved if it adds something, else archived), voices move (the K longest stay live), face crops move. Source
        gets ``merged_into`` so old person_ids in logs still resolve; a known source names an unknown target.
        Nothing is deleted: ``unmerge(event_id)`` restores both people within ``undo_days``.

        :param action: ``merge`` (user), ``auto_merge`` / ``review_merge`` (dedup job / review queue) or
                       ``link_voice`` (automatic face ↔ voice link) — audit + event type
        :param details: Extra audit details (score, evidence)
        :return: ``identity_merged`` (``identity_linked`` for link_voice) with ``event_id`` for undo
        """
        source, target = self._require(source_id), self._require(target_id)
        if source == target:
            raise ValueError("Cannot merge a person into itself")
        src, dst = self.persons[source], self.persons[target]
        crops = self._crop_files(source)
        with self._transaction() as db:
            now = self.clock()
            event_id = db.execute("INSERT INTO identity_events (ts, action, person_id, details_json) "
                                  "VALUES (?, ?, ?, '{}')", (now, action, target)).lastrowid
            moved, dropped, evicted = [], [], []
            for row_id, blob, yaw, tier in db.execute(
                    "SELECT id, vector, yaw_bucket, tier FROM face_embeddings WHERE person_id = ? AND archived_by "
                    "IS NULL ORDER BY quality DESC, id", (source,)).fetchall():
                vector = np.frombuffer(blob, np.float32)
                decision = admit(self._face_rows(db, target), vector, tier, yaw, self.max_faces, self.redundancy,
                                 self.quotas)
                if not decision.accepted:
                    dropped.append(row_id)
                    db.execute("UPDATE face_embeddings SET archived_by = ? WHERE id = ?", (event_id, row_id))
                    continue
                for victim in decision.evict:
                    evicted.append(victim)
                    db.execute("UPDATE face_embeddings SET archived_by = ? WHERE id = ?", (event_id, victim))
                moved.append(row_id)
                db.execute("UPDATE face_embeddings SET person_id = ? WHERE id = ?", (target, row_id))
            voices = [r[0] for r in db.execute("SELECT id FROM voice_embeddings WHERE person_id = ? AND archived_by "
                                               "IS NULL", (source,))]
            db.execute("UPDATE voice_embeddings SET person_id = ? WHERE person_id = ? AND archived_by IS NULL",
                       (target, source))
            # The merged person may now exceed K voice samples: archive the shortest
            db.execute("UPDATE voice_embeddings SET archived_by = ? WHERE person_id = ? AND archived_by IS NULL AND "
                       "id NOT IN (SELECT id FROM voice_embeddings WHERE person_id = ? AND archived_by IS NULL "
                       "ORDER BY duration_sec DESC, id DESC LIMIT ?)", (event_id, target, target, self.max_embeddings))
            db.execute("UPDATE persons SET merged_into = ?, updated_at = ? WHERE person_id = ?", (target, now, source))
            named = bool(src["is_known"] and not dst["is_known"])
            if named:
                db.execute("UPDATE persons SET display_name = ?, is_known = 1 WHERE person_id = ?",
                           (src["display_name"], target))
            last = max((src["last_seen"] or 0, src["last_location"]), (dst["last_seen"] or 0, dst["last_location"]),
                       key=lambda x: x[0])
            db.execute("UPDATE persons SET updated_at = ?, last_seen = ?, last_location = ? WHERE person_id = ?",
                       (now, last[0] or None, last[1], target))
            db.execute("UPDATE merge_candidates SET status = 'merged', updated_at = ?, decided_by = ? WHERE status = "
                       "'pending' AND ((person_a = ? AND person_b = ?) OR (person_a = ? AND person_b = ?))",
                       (now, action, source, target, target, source))
            record = {"source_id": source, "source_name": src["display_name"], "target_name": dst["display_name"],
                      "named_target": named, "target_last_seen": dst["last_seen"],
                      "target_last_location": dst["last_location"], "moved_face": moved, "dropped_face": dropped,
                      "evicted_face": evicted, "moved_voice": voices, "crops": crops, **details}
            db.execute("UPDATE identity_events SET details_json = ? WHERE id = ?",
                       (json.dumps(record, ensure_ascii=False), event_id))
        self._move_crops(source, target, crops)
        self.drop_review_crops(source, target if named or dst["is_known"] else None)
        self.refresh()
        event_type = "identity_linked" if action == "link_voice" else "identity_merged"
        return self._emit(event_type, target, source_id=source, source_name=src["display_name"], event_id=event_id,
                          action=action, moved=len(moved), dropped=len(dropped) + len(evicted))

    def merge_record(self, event_id: int) -> Optional[Dict[str, Any]]:
        """:return: A merge's identity_events row (+ ``undone``, ``undoable``), or None"""
        with self._lock:
            row = self.db.execute("SELECT id, ts, action, person_id, details_json FROM identity_events WHERE id = ?",
                                  (int(event_id),)).fetchone()
            undone = {json.loads(r[0] or "{}").get("merge_event_id") for r in self.db.execute(
                "SELECT details_json FROM identity_events WHERE action = 'unmerge'")}
        if not row or row[2] not in MERGE_ACTIONS:
            return None
        record = {"event_id": row[0], "ts": row[1], "action": row[2], "person_id": row[3],
                  **json.loads(row[4] or "{}")}
        source = self.persons.get(record.get("source_id"))
        record["undone"] = row[0] in undone
        record["undoable"] = bool(not record["undone"] and source and source["merged_into"] == row[3]
                                  and row[3] in self.persons and not self.persons[row[3]]["merged_into"]
                                  and self.clock() - row[1] <= self.undo_days * 86400)
        return record

    def merges(self, limit: int = 50) -> List[Dict[str, Any]]:
        """:return: Recent merges (newest first) with undo state, for the dashboard merge history"""
        with self._lock:
            ids = [r[0] for r in self.db.execute(
                f"SELECT id FROM identity_events WHERE action IN ({','.join('?' * len(MERGE_ACTIONS))}) "
                f"ORDER BY id DESC LIMIT ?", (*MERGE_ACTIONS, limit))]
        return [r for r in (self.merge_record(i) for i in ids) if r]

    def unmerge(self, event_id: int, by: str = "user") -> Event:
        """
        Undoes a merge: the moved embeddings and face crops go back, archived ones are restored, the source is a
        live person again and the pair gets a cannot-link (the dedup job will not merge it again).

        :raise ValueError: not a merge, already undone, older than ``undo_days`` or the target was merged further
        """
        self.refresh()
        record = self.merge_record(event_id)
        if record is None:
            raise KeyError(f"No merge with event id {event_id}")
        if not record["undoable"]:
            raise ValueError("This merge cannot be undone (already undone, too old, or merged further since)")
        source, target = record["source_id"], record["person_id"]
        with self._transaction() as db:
            for column, ids in (("face", record.get("moved_face", [])), ("voice", record.get("moved_voice", []))):
                for row_id in ids:
                    db.execute(f"UPDATE {column}_embeddings SET person_id = ? WHERE id = ?", (source, row_id))
            for modality in MODALITIES:
                db.execute(f"UPDATE {modality}_embeddings SET archived_by = NULL WHERE archived_by = ?",
                           (int(event_id),))
            now = self.clock()
            db.execute("UPDATE persons SET merged_into = NULL, updated_at = ? WHERE person_id = ?", (now, source))
            if record.get("named_target") and self.persons[target]["display_name"] == record["source_name"]:
                db.execute("UPDATE persons SET display_name = ?, is_known = 0 WHERE person_id = ?",
                           (record["target_name"], target))
            db.execute("UPDATE persons SET updated_at = ?, last_seen = ?, last_location = ? WHERE person_id = ?",
                       (now, record.get("target_last_seen"), record.get("target_last_location"), target))
            self._link_apart(db, source, target, "undo")
            self._audit(db, "unmerge", target, merge_event_id=int(event_id), source_id=source, by=by)
        self._move_crops(target, source, record.get("crops", []))
        self.refresh()
        return self._emit("identity_unmerged", target, source_id=source, source_name=self.name(source),
                          event_id=int(event_id))

    # --- cannot-link + duplicates review queue (F14) -----------------------------------------------------------------

    @staticmethod
    def _pair(a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def _link_apart(self, db, a: str, b: str, reason: str) -> bool:
        cursor = db.execute("INSERT OR IGNORE INTO cannot_link (person_a, person_b, reason, created_at) "
                            "VALUES (?, ?, ?, ?)", (*self._pair(a, b), reason, self.clock()))
        return cursor.rowcount > 0

    def add_cannot_link(self, person_ids: List[str], reason: str = "same_frame") -> int:
        """
        Records that these people are pairwise different (seen in one frame, or a user's "not the same person").
        Cheap to call every frame: pairs already known are skipped without touching the database.

        :return: Number of new pairs
        """
        live = sorted({pid for pid in (self.resolve(p) for p in person_ids) if pid})
        pairs = [(a, b) for i, a in enumerate(live) for b in live[i + 1:]
                 if frozenset((a, b)) not in self._blocked_live]
        if not pairs:
            return 0
        with self._transaction() as db:
            added = sum(self._link_apart(db, a, b, reason) for a, b in pairs)
        for a, b in pairs:      # no full refresh: only the pair sets change
            self.cannot_links.add(frozenset((a, b)))
            self._blocked_live.add(frozenset((a, b)))
        with self._lock:
            self._data_version = self.db.execute("PRAGMA data_version").fetchone()[0]
        return added

    def propose(self, person_a: str, person_b: str, score: float, evidence: Dict[str, Any]) -> Optional[int]:
        """
        Puts a pair into the review queue (or updates its score). Pairs with a cannot-link are never proposed.

        :return: Candidate id, or None if blocked
        """
        a, b = self._require(person_a), self._require(person_b)
        if a == b or self.blocked(a, b):
            return None
        a, b = self._pair(a, b)
        with self._transaction() as db:
            now = self.clock()
            row = next(([c["id"]] for c in self.candidates() if {c["person_a"], c["person_b"]} == {a, b}), None) or \
                db.execute("SELECT id FROM merge_candidates WHERE person_a = ? AND person_b = ? AND status = "
                           "'pending'", (a, b)).fetchone()
            if row:
                db.execute("UPDATE merge_candidates SET score = ?, evidence_json = ?, updated_at = ? WHERE id = ?",
                           (float(score), json.dumps(evidence), now, row[0]))
                return row[0]
            return db.execute("INSERT INTO merge_candidates (person_a, person_b, score, evidence_json, created_at, "
                              "updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                              (a, b, float(score), json.dumps(evidence), now, now)).lastrowid

    def candidates(self, status: str = "pending") -> List[Dict[str, Any]]:
        """:return: Review-queue rows (live people only), best score first"""
        with self._lock:
            rows = self.db.execute("SELECT id, person_a, person_b, score, evidence_json, status, created_at, "
                                   "updated_at, decided_by FROM merge_candidates WHERE status = ? ORDER BY score DESC",
                                   (status,)).fetchall()
        out, seen = [], set()
        for r in rows:
            a, b = self.resolve(r[1]), self.resolve(r[2])
            if status == "pending" and (a is None or b is None or a == b or self.blocked(a, b)
                                        or frozenset((a, b)) in seen):
                continue        # gone, merged meanwhile, blocked, or the same live pair as a better row
            seen.add(frozenset((a, b)))
            out.append({"id": r[0], "person_a": a or r[1], "person_b": b or r[2], "score": round(r[3], 3),
                        "evidence": json.loads(r[4] or "{}"), "status": r[5], "created_at": r[6],
                        "updated_at": r[7], "decided_by": r[8]})
        return out

    def candidate(self, candidate_id: int) -> Optional[Dict[str, Any]]:
        return next((c for c in self.candidates() if c["id"] == int(candidate_id)), None)

    def merge_target(self, person_a: str, person_b: str) -> Tuple[str, str]:
        """:return: (source, target): a known person is the target, else the older unknown"""
        a, b = self._require(person_a), self._require(person_b)
        pa, pb = self.persons[a], self.persons[b]
        if pa["is_known"] != pb["is_known"]:
            return (b, a) if pa["is_known"] else (a, b)
        return (b, a) if pa["created_at"] <= pb["created_at"] else (a, b)

    def accept_candidate(self, candidate_id: int, by: str = "user") -> Event:
        """Merges a queued pair (into the known / older person)."""
        candidate = self.candidate(candidate_id)
        if candidate is None:
            raise KeyError(f"No pending duplicate {candidate_id}")
        source, target = self.merge_target(candidate["person_a"], candidate["person_b"])
        event = self.merge(source, target, action="review_merge", candidate_id=candidate["id"],
                           score=candidate["score"], by=by)
        self._decide(candidate["id"], "merged", by)
        return event

    def reject_candidate(self, candidate_id: int, by: str = "user") -> None:
        """"Not the same person": a permanent cannot-link, the pair is never proposed again."""
        candidate = self.candidate(candidate_id)
        if candidate is None:
            raise KeyError(f"No pending duplicate {candidate_id}")
        with self._transaction() as db:
            self._link_apart(db, candidate["person_a"], candidate["person_b"], "user")
            self._audit(db, "not_same", candidate["person_a"], other_id=candidate["person_b"], by=by,
                        candidate_id=candidate["id"])
        self._decide(candidate["id"], "rejected", by)
        self.drop_review_crops(candidate["person_a"], candidate["person_b"])
        self.refresh()

    def _decide(self, candidate_id: int, status: str, by: str) -> None:
        with self._transaction() as db:
            db.execute("UPDATE merge_candidates SET status = ?, updated_at = ?, decided_by = ? WHERE id = ?",
                       (status, self.clock(), by, int(candidate_id)))

    def purge_archived(self) -> int:
        """Deletes embeddings archived by merges older than ``undo_days`` (their undo window is over)."""
        cutoff = self.clock() - self.undo_days * 86400
        with self._transaction() as db:
            old = [r[0] for r in db.execute("SELECT id FROM identity_events WHERE ts < ? AND action IN "
                                            f"({','.join('?' * len(MERGE_ACTIONS))})", (cutoff, *MERGE_ACTIONS))]
            removed = 0
            for event_id in old:
                for modality in MODALITIES:
                    removed += db.execute(f"DELETE FROM {modality}_embeddings WHERE archived_by = ?",
                                          (event_id,)).rowcount
        return removed

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
                db.execute("DELETE FROM merge_candidates WHERE status = 'pending' AND (person_a = ? OR person_b = ?)",
                           (victim, victim))
            self._audit(db, "delete", pid, display_name=name, reason=reason)
        if self.faces_dir:
            shutil.rmtree(os.path.join(self.faces_dir, pid), ignore_errors=True)
        self.drop_review_crops(pid, *merged)
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
        """:return: Path of the stored crop, or None (storage off, enough crops, or a known person without faces_dir)"""
        pid = self.resolve(person_id)
        if pid is None or image is None:
            return None
        if self.faces_dir:
            root = self.faces_dir
        elif self.review_dir and not self.persons[pid]["is_known"]:
            root, max_images = self.review_dir, min(max_images, 3)
        else:
            return None
        import cv2

        folder = os.path.join(root, pid)
        os.makedirs(folder, exist_ok=True)
        existing = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))
        if len(existing) >= max_images:
            return None
        path = os.path.join(folder, f"face_{int(self.clock() * 1000)}_{len(existing):03d}.jpg")
        cv2.imwrite(path, image)
        return path

    def thumbnail(self, person_id: str) -> Optional[str]:
        for root in (self.faces_dir, self.review_dir):
            folder = os.path.join(root, person_id) if root else None
            files = sorted(f for f in os.listdir(folder) if f.endswith(".jpg")) \
                if folder and os.path.isdir(folder) else []
            if files:
                return os.path.join(folder, files[0])
        return None

    def drop_review_crops(self, *person_ids: str) -> None:
        """Review crops exist only while a decision is pending (named / merged / rejected / deleted → removed)."""
        if not self.review_dir:
            return
        for pid in person_ids:
            if pid:
                shutil.rmtree(os.path.join(self.review_dir, os.path.basename(pid)), ignore_errors=True)

    def _crop_files(self, person_id: str) -> List[str]:
        folder = os.path.join(self.faces_dir, person_id) if self.faces_dir else None
        return sorted(os.listdir(folder)) if folder and os.path.isdir(folder) else []

    def _move_crops(self, source: str, target: str, files: Optional[List[str]] = None) -> None:
        """Moves the given (default: all) face crops of ``source`` into ``target``'s folder."""
        if not self.faces_dir or not os.path.isdir(os.path.join(self.faces_dir, source)):
            return
        os.makedirs(os.path.join(self.faces_dir, target), exist_ok=True)
        for file in self._crop_files(source) if files is None else files:
            path = os.path.join(self.faces_dir, source, file)
            if os.path.isfile(path):
                shutil.move(path, os.path.join(self.faces_dir, target, file))
        if not os.listdir(os.path.join(self.faces_dir, source)):
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
