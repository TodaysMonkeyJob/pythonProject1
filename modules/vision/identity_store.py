"""
Persistent store of unknown people and promotion of a renamed unknown to a known person.

Every unknown face gets a folder ``<unknown_dir>/unknown_N/`` with aligned face crops (``face_NNN.jpg``, at most
``max_images``), ``embeddings.npy`` and ``meta.json`` (label, first/last seen). Labels survive restarts.

To name a person, rename the folder (``unknown_3`` → ``Oleh``) inside ``unknown_dir`` or move it to
``<faces_dir>/Oleh``. ``find_renamed()`` spots folders whose ``meta.json`` label differs from the folder name and
``promote()`` then: adds the embeddings to the face database (merging with an existing person), moves the images
to ``<faces_dir>/<name>/``, and records ``unknown_3 → Oleh`` in ``aliases.json`` so components that stored the
old label (logs, snapshots, dashboard) can resolve it. Event logs are not rewritten — they stay an audit trail.

Unknown faces are biometric data of people who did not consent: everything stays under ``data/faces``
(git-ignored), and folders not seen for ``retention_days`` are deleted.
"""
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from system_setup.logger_setup import init_logger

logger = init_logger()

UNKNOWN_RE = re.compile(r"^unknown_(\d+)$")
META_FILE = "meta.json"
COUNTER_FILE = "_counter.json"
EMBEDDINGS_FILE = "embeddings.npy"
MAX_EMBEDDINGS = 50


@dataclass
class Promotion:
    old_label: str
    name: str
    embeddings: np.ndarray
    folder: str          # final folder of the person's images


def _read_json(path: str, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def valid_name(name: str) -> bool:
    """A folder name usable as a person's name (not an unknown label, not hidden / temporary)."""
    return bool(name.strip()) and name != "unknown" and not UNKNOWN_RE.match(name) \
        and not name.startswith((".", "_")) and "/" not in name


class AliasRegistry:
    """``aliases.json``: old unknown label → person name."""

    def __init__(self, path: str):
        self.path = path
        self.aliases: Dict[str, str] = _read_json(path, {})

    def add(self, old_label: str, name: str) -> None:
        self.aliases[old_label] = name
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        _write_json(self.path, self.aliases)

    def resolve(self, label: str) -> str:
        """
        :return: The current name for a label (follows chains, e.g. after a second rename)
        """
        seen = set()
        while label in self.aliases and label not in seen:
            seen.add(label)
            label = self.aliases[label]
        return label


class UnknownStore:
    def __init__(self, unknown_dir: str, faces_dir: str, max_images: int = 10, retention_days: float = 30):
        self.unknown_dir = unknown_dir
        self.faces_dir = faces_dir
        self.max_images = max_images
        self.retention_sec = retention_days * 86400
        os.makedirs(unknown_dir, exist_ok=True)

    # --- unknowns ------------------------------------------------------------------------------------------------

    def folder(self, label: str) -> str:
        return os.path.join(self.unknown_dir, label)

    def load(self) -> List[Tuple[str, np.ndarray, int]]:
        """
        :return: [(label, normalised mean embedding, sample count)] of the stored unknowns
        """
        unknowns = []
        for name in sorted(os.listdir(self.unknown_dir)):
            path = self.folder(name)
            if not UNKNOWN_RE.match(name) or not os.path.isfile(os.path.join(path, META_FILE)):
                continue
            embeddings = self._embeddings(path)
            if len(embeddings):
                mean = embeddings.mean(axis=0)
                unknowns.append((name, mean / np.linalg.norm(mean), len(embeddings)))
        return unknowns

    def next_label(self) -> str:
        """
        :return: A new unknown_N; numbers are never reused (a renamed unknown_3 stays an alias of its person)
        """
        counter_path = os.path.join(self.unknown_dir, COUNTER_FILE)
        used = [int(m.group(1)) for m in map(UNKNOWN_RE.match, os.listdir(self.unknown_dir)) if m]
        number = max([_read_json(counter_path, {}).get("last", 0)] + used) + 1
        _write_json(counter_path, {"last": number})
        return f"unknown_{number}"

    def save(self, label: str, embedding: np.ndarray, face_image: Optional[np.ndarray] = None,
             now: Optional[float] = None) -> None:
        """Adds one sighting of an unknown person."""
        now = time.time() if now is None else now
        path = self.folder(label)
        os.makedirs(path, exist_ok=True)
        meta_path = os.path.join(path, META_FILE)
        meta = _read_json(meta_path, {"label": label, "first_seen": now, "sightings": 0, "images": 0})
        meta["last_seen"] = now
        meta["sightings"] += 1

        embeddings = self._embeddings(path)
        embeddings = np.vstack([embeddings, embedding.reshape(1, -1)])[-MAX_EMBEDDINGS:]
        np.save(os.path.join(path, EMBEDDINGS_FILE), embeddings.astype(np.float32))

        if face_image is not None and meta["images"] < self.max_images:
            meta["images"] += 1
            cv2.imwrite(os.path.join(path, f"face_{meta['images']:03d}.jpg"), face_image)
        _write_json(meta_path, meta)

    def purge(self, now: Optional[float] = None) -> List[str]:
        """Deletes unknown folders not seen for retention_days. :return: removed labels"""
        now = time.time() if now is None else now
        removed = []
        for name in os.listdir(self.unknown_dir):
            meta = _read_json(os.path.join(self.folder(name), META_FILE), None)
            if UNKNOWN_RE.match(name) and meta and now - meta.get("last_seen", now) > self.retention_sec:
                shutil.rmtree(self.folder(name), ignore_errors=True)
                removed.append(name)
        if removed:
            logger.info(f"Removed unknown faces older than the retention period: {removed}")
        return removed

    # --- renames -------------------------------------------------------------------------------------------------

    def find_renamed(self) -> List[Tuple[str, str, str]]:
        """
        :return: [(old label, new name, folder)] for unknown folders renamed in unknown_dir or moved to faces_dir
        """
        found = []
        for root in (self.unknown_dir, self.faces_dir):
            if not os.path.isdir(root):
                continue
            for name in sorted(os.listdir(root)):
                path = os.path.join(root, name)
                meta = _read_json(os.path.join(path, META_FILE), None) if os.path.isdir(path) else None
                if not meta or meta.get("promoted_to"):
                    continue
                label = meta.get("label", "")
                if UNKNOWN_RE.match(label) and name != label and valid_name(name):
                    found.append((label, name, path))
        return found

    def promote(self, old_label: str, name: str, source: Optional[str] = None) -> Promotion:
        """
        Moves an unknown person's data to ``<faces_dir>/<name>/`` (merging with existing photos).

        :param old_label: unknown_N
        :param name: New person name
        :param source: Current folder (renamed / moved); defaults to unknown_dir/old_label
        :return: Promotion with the embeddings to add to the face database
        """
        if not valid_name(name):
            raise ValueError(f"Invalid person name {name!r}")
        source = source or self.folder(old_label)
        if not os.path.isdir(source):
            raise FileNotFoundError(f"No stored unknown {old_label!r} ({source})")
        embeddings = self._embeddings(source)
        target = os.path.join(self.faces_dir, name)
        os.makedirs(target, exist_ok=True)
        for file in sorted(os.listdir(source)):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            destination = os.path.join(target, f"{old_label}_{file}")
            if os.path.abspath(os.path.join(source, file)) != os.path.abspath(destination):
                shutil.move(os.path.join(source, file), destination)
        meta = _read_json(os.path.join(source, META_FILE), {})
        meta.update({"promoted_to": name, "promoted_at": time.time()})
        _write_json(os.path.join(target, f"{old_label}_{META_FILE}"), meta)
        if os.path.abspath(source) != os.path.abspath(target):
            shutil.rmtree(source, ignore_errors=True)
        else:  # the folder itself was moved to faces_dir/<name>: drop the per-unknown files
            for file in (META_FILE, EMBEDDINGS_FILE):
                if os.path.exists(os.path.join(target, file)):
                    os.remove(os.path.join(target, file))
        return Promotion(old_label, name, embeddings, target)

    @staticmethod
    def _embeddings(path: str) -> np.ndarray:
        file = os.path.join(path, EMBEDDINGS_FILE)
        if not os.path.isfile(file):
            return np.zeros((0, 128), np.float32)
        return np.load(file).astype(np.float32).reshape(-1, 128)
