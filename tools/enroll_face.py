"""
Enrolls a known person into the face database (``identity.db_path``, default data/faces/faces.npz).

    python -m tools.enroll_face --name Oleh --images data/faces/Oleh/
    python -m tools.enroll_face --name Oleh --camera 0 --frames 10
    python -m tools.enroll_face --from-unknown unknown_3 --name Oleh   # name a stored unknown person
    python -m tools.enroll_face --list
    python -m tools.enroll_face --remove Oleh

Embeddings of all usable images are averaged and added to (or merged into) the person's entry.
Only enroll people who gave written consent; the database stays on the device.
"""
import argparse
import glob
import os
import sys
import time
from typing import List, Optional

import cv2
import numpy as np

if __package__ in (None, ""):  # allow `python tools/enroll_face.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config  # noqa: E402
from modules.vision.identity import FaceDatabase, FaceEngine, largest_face  # noqa: E402
from modules.vision.identity_store import AliasRegistry, UnknownStore  # noqa: E402

IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def embeddings_from_images(engine: FaceEngine, folder: str, min_size: int) -> List[np.ndarray]:
    paths = sorted(p for pattern in IMAGE_PATTERNS for p in glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise SystemExit(f"No images in {folder}")
    result = []
    for path in paths:
        image = cv2.imread(path)
        face = None if image is None else largest_face(engine.detect(image), min_size)
        if face is None:
            print(f"  skip (no face): {path}")
            continue
        result.append(engine.embed(image, face))
    return result


def embeddings_from_camera(engine: FaceEngine, device: int, frames: int, min_size: int,
                           interval_sec: float = 0.5) -> List[np.ndarray]:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {device}")
    result, deadline = [], time.monotonic() + 30
    try:
        while len(result) < frames and time.monotonic() < deadline:
            ok, frame = cap.read()
            if not ok:
                continue
            face = largest_face(engine.detect(frame), min_size)
            if face is not None:
                result.append(engine.embed(frame, face))
                print(f"  captured {len(result)}/{frames}")
                time.sleep(interval_sec)  # different head positions between captures
    finally:
        cap.release()
    return result


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Enroll a person into the face database")
    parser.add_argument("--name")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--images", help="Folder with photos of the person")
    source.add_argument("--camera", type=int, help="Camera index to capture from")
    source.add_argument("--from-unknown", metavar="LABEL", help="Stored unknown person (unknown_N) to name")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--remove", metavar="NAME")
    parser.add_argument("--db", help="Overrides identity.db_path")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)["identity"]
    database = FaceDatabase(args.db or cfg["db_path"])
    if args.list:
        for name, count in zip(database.names, database.counts, strict=True):
            print(f"{name}: {count} images")
        return
    if args.remove:
        if not database.remove(args.remove):
            raise SystemExit(f"{args.remove} is not enrolled")
        database.save()
        print(f"Removed {args.remove}")
        return
    if not args.name or (args.images is None and args.camera is None and args.from_unknown is None):
        parser.error("--name and one of --images / --camera / --from-unknown are required")

    if args.from_unknown:
        store = UnknownStore(cfg.get("unknown_dir", "data/faces/unknown"), cfg.get("faces_dir", "data/faces"))
        promotion = store.promote(args.from_unknown, args.name)
        if len(promotion.embeddings):
            database.add(args.name, list(promotion.embeddings))
            database.save()
        AliasRegistry(cfg.get("aliases_path", "data/faces/aliases.json")).add(args.from_unknown, args.name)
        print(f"{args.from_unknown} is now {args.name}: {len(promotion.embeddings)} embeddings enrolled, images in "
              f"{promotion.folder} (a running identity module picks this up within sync_sec)")
        return

    engine = FaceEngine(cfg["detector_model"], cfg["recognizer_model"], cfg.get("detection_score", 0.8))
    min_size = cfg.get("min_face_size", 40)
    if args.images:
        embeddings = embeddings_from_images(engine, args.images, min_size)
    else:
        embeddings = embeddings_from_camera(engine, args.camera, args.frames, min_size)
    if not embeddings:
        raise SystemExit("No usable face found")
    database.add(args.name, embeddings)
    database.save()
    print(f"Enrolled {args.name} from {len(embeddings)} images -> {database.path} ({len(database)} people)")


if __name__ == "__main__":
    main()
