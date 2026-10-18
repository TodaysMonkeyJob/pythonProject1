"""
Imports the pre-registry face data (TOD-14) into the identity registry (F9 / TOD-20):

* ``faces.npz`` (names, averaged embeddings) → known persons; photos in ``<npz folder>/<name>/*.jpg`` become their
  thumbnails;
* ``unknown/unknown_N/`` folders (``embeddings.npy``, ``face_*.jpg``) → unknown persons that keep their number;
* ``aliases.json`` is not needed any more (old person_ids resolve through ``merged_into``), it is left untouched.

Run once via ``python -m tools.migrate_faces_npz``; the identity module also does it automatically when the
registry is empty. The legacy files are not deleted.
"""
import glob
import json
import os
import re
import shutil
from typing import Dict, Optional

import numpy as np

from core.identity import IdentityRegistry, valid_name

UNKNOWN_RE = re.compile(r"^unknown_(\d+)$")


def _copy_crops(registry: IdentityRegistry, person_id: str, folder: str, limit: int = 10) -> int:
    if not registry.faces_dir or not os.path.isdir(folder):
        return 0
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")))[:limit]
    target = os.path.join(registry.faces_dir, person_id)
    os.makedirs(target, exist_ok=True)
    for path in files:
        shutil.copy2(path, os.path.join(target, os.path.basename(path)))
    return len(files)


def migrate(registry: IdentityRegistry, npz_path: Optional[str], unknown_dir: Optional[str] = None) -> Dict[str, int]:
    """
    :param npz_path: Legacy ``faces.npz`` (skipped when missing)
    :param unknown_dir: Legacy unknown folders; defaults to ``<npz folder>/unknown``
    :return: Counts of imported known / unknown people and images
    """
    report = {"known": 0, "unknown": 0, "images": 0, "skipped": 0}
    root = os.path.dirname(npz_path) if npz_path else None
    if npz_path and os.path.isfile(npz_path):
        data = np.load(npz_path, allow_pickle=False)
        for name, embedding in zip(data["names"], data["embeddings"], strict=True):
            name = str(name)
            if not valid_name(name) or registry.find(name):
                report["skipped"] += 1
                continue
            person_id = registry.create_person(name)
            registry.add_embedding(person_id, "face", embedding.astype(np.float32))
            report["known"] += 1
            report["images"] += _copy_crops(registry, person_id, os.path.join(root, name))

    unknown_dir = unknown_dir or (os.path.join(root, "unknown") if root else None)
    if unknown_dir and os.path.isdir(unknown_dir):
        folders = sorted((int(m.group(1)), name) for name in os.listdir(unknown_dir)
                         if (m := UNKNOWN_RE.match(name)))
        for number, name in folders:
            path = os.path.join(unknown_dir, name, "embeddings.npy")
            if not os.path.isfile(path):
                continue
            embeddings = np.load(path).astype(np.float32).reshape(-1, 128)
            if not len(embeddings) or registry.find(name):
                report["skipped"] += 1
                continue
            registry.ensure_counter(number - 1)
            person_id = registry.create_unknown("face", embeddings[-1])
            for embedding in embeddings[:-1][-registry.max_faces:]:
                registry.add_embedding(person_id, "face", embedding)
            report["unknown"] += 1
            report["images"] += _copy_crops(registry, person_id, os.path.join(unknown_dir, name))
        try:  # numbers are never reused, even of unknowns that were renamed or deleted earlier
            with open(os.path.join(unknown_dir, "_counter.json"), encoding="utf-8") as f:
                last = int(json.load(f).get("last", 0))
        except (OSError, ValueError):
            last = 0
        registry.ensure_counter(max([last] + [number for number, _ in folders]))
    return report
