"""
Evaluates face identification on a folder of people (``<root>/<name>/*.jpg``, e.g. data/faces or LFW).

The ``--people`` persons with the most photos are "known": ``--enroll`` photos each are enrolled, the rest are
test images. Everyone else is a stranger. For each similarity threshold it reports

* known accuracy — test photos identified as the right person (acceptance: ≥ 90 % for 5 people);
* false accept rate — stranger photos matched to any known person (should be ~0);
* face detection failures and per-face time (YuNet + SFace).

    python -m tools.eval_faces data/faces --people 5 --enroll 5
"""
import argparse
import glob
import json
import os
import random
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from core.config import load_config
from modules.vision.identity import FaceDatabase, FaceEngine, largest_face

THRESHOLDS = (0.30, 0.363, 0.40, 0.45, 0.50, 0.55)


def load_people(root: str) -> Dict[str, List[str]]:
    people = {}
    for folder in sorted(glob.glob(os.path.join(root, "*"))):
        images = sorted(p for ext in ("jpg", "jpeg", "png") for p in glob.glob(os.path.join(folder, f"*.{ext}")))
        if os.path.isdir(folder) and images:
            people[os.path.basename(folder)] = images
    return people


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Face identification accuracy / false accept rate")
    parser.add_argument("root")
    parser.add_argument("--people", type=int, default=5)
    parser.add_argument("--enroll", type=int, default=5, help="Photos enrolled per known person")
    parser.add_argument("--max-test", type=int, default=30, help="Test photos per known person")
    parser.add_argument("--strangers", type=int, default=300, help="Stranger photos (one per other person first)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="reports/identity_eval.json")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)["identity"]
    engine = FaceEngine(cfg["detector_model"], cfg["recognizer_model"], cfg.get("detection_score", 0.8))
    min_size = cfg.get("min_face_size", 40)
    rng = random.Random(args.seed)
    people = load_people(args.root)
    ranked = sorted(people, key=lambda n: -len(people[n]))
    known = ranked[:args.people]
    others = ranked[args.people:]

    timings: List[float] = []
    failures = 0

    def embed(path: str) -> Optional[np.ndarray]:
        nonlocal failures
        image = cv2.imread(path)
        start = time.perf_counter()
        face = None if image is None else largest_face(engine.detect(image), min_size)
        if face is None:
            failures += 1
            return None
        embedding = engine.embed(image, face)
        timings.append((time.perf_counter() - start) * 1000)
        return embedding

    database = FaceDatabase()
    tests = []
    for name in known:
        photos = people[name][:]
        rng.shuffle(photos)
        enrolled = [e for e in (embed(p) for p in photos[:args.enroll]) if e is not None]
        database.add(name, enrolled)
        tests += [(name, p) for p in photos[args.enroll:args.enroll + args.max_test]]

    stranger_photos = [people[n][0] for n in others]
    extra = [p for n in others for p in people[n][1:]]
    rng.shuffle(extra)
    stranger_photos = (stranger_photos + extra)[:args.strangers]

    known_results = [(name, database.match(e)) for name, e in ((n, embed(p)) for n, p in tests) if e is not None]
    stranger_results = [database.match(e) for e in (embed(p) for p in stranger_photos) if e is not None]

    rows = []
    for threshold in THRESHOLDS:
        correct = sum(match == name and sim >= threshold for name, (match, sim) in known_results)
        false_accepts = sum(sim >= threshold for _, sim in stranger_results)
        rows.append({
            "threshold": threshold,
            "known_accuracy": round(correct / len(known_results), 3) if known_results else None,
            "false_accept_rate": round(false_accepts / len(stranger_results), 4) if stranger_results else None,
        })
    report = {
        "root": args.root, "known_people": known, "enrolled_per_person": args.enroll,
        "known_test_images": len(known_results), "stranger_images": len(stranger_results),
        "stranger_people": len(others), "detection_failures": failures,
        "face_ms": {"mean": round(float(np.mean(timings)), 2), "p95": round(float(np.percentile(timings, 95)), 2)}
        if timings else None,
        "config_threshold": cfg.get("face_threshold"), "by_threshold": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
