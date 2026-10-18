"""
Enrolls a known person into the identity registry (F9 / TOD-20): face photos and / or voice samples.

    python -m tools.enroll --name Oleh --faces data/faces/Oleh/
    python -m tools.enroll --name Oleh --camera 0 --frames 10
    python -m tools.enroll --name Oleh --voice speech_to_text_whisper/voices/oleh.wav
    python -m tools.enroll --name Oleh --faces photos/ --voice a.wav b.wav

An existing person with this name gets the new samples added (the best ``identity.max_embeddings`` per modality are
kept). Unknown people seen by the camera are named on the dashboard People page or with ``python -m tools.identity``.
Only enroll people who gave written consent; everything stays on the device.
"""
import argparse
import glob
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

if __package__ in (None, ""):  # allow `python tools/enroll.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config  # noqa: E402
from core.identity import registry_from_config  # noqa: E402
from modules.vision.identity import FaceEngine, largest_face  # noqa: E402

IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def embeddings_from_images(engine: FaceEngine, folder: str, min_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """:return: [(embedding, aligned face crop)]"""
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
        result.append((engine.embed(image, face), engine.aligned_crop(image, face)))
    return result


def embeddings_from_camera(engine: FaceEngine, device: int, frames: int, min_size: int,
                           interval_sec: float = 0.5) -> List[Tuple[np.ndarray, np.ndarray]]:
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
                result.append((engine.embed(frame, face), engine.aligned_crop(frame, face)))
                print(f"  captured {len(result)}/{frames}")
                time.sleep(interval_sec)  # different head positions between captures
    finally:
        cap.release()
    return result


def voice_embedding(embedder, path: str) -> Tuple[np.ndarray, float]:
    """:return: (ECAPA embedding, duration in seconds) of a WAV file"""
    import torchaudio

    waveform, sample_rate = torchaudio.load(path)
    audio = waveform.mean(dim=0).numpy()
    return embedder(audio, sample_rate), len(audio) / sample_rate


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Enroll a person (faces and / or voice) into the identity registry")
    parser.add_argument("--name", required=True)
    faces = parser.add_mutually_exclusive_group()
    faces.add_argument("--faces", metavar="DIR", help="Folder with photos of the person")
    faces.add_argument("--camera", type=int, help="Camera index to capture the face from")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--voice", nargs="+", metavar="WAV", help="Voice sample(s), ≥ 3 s of speech each")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    if args.faces is None and args.camera is None and not args.voice:
        parser.error("give --faces, --camera and / or --voice")

    config = load_config(args.config)
    cfg = config["identity"]
    registry = registry_from_config(cfg)
    try:
        person_id = registry.find(args.name)
        if person_id is not None and not registry.is_known(person_id):
            raise SystemExit(f"{args.name} is an unknown_N label: rename it instead (python -m tools.identity)")
        created = person_id is None
        if args.faces is not None or args.camera is not None:
            engine = FaceEngine(cfg["detector_model"], cfg["recognizer_model"], cfg.get("detection_score", 0.8))
            min_size = cfg.get("min_face_size", 40)
            samples = embeddings_from_images(engine, args.faces, min_size) if args.faces is not None \
                else embeddings_from_camera(engine, args.camera, args.frames, min_size)
            if not samples:
                raise SystemExit("No usable face found")
            person_id = person_id or registry.create_person(args.name)
            for embedding, crop in samples:
                registry.add_embedding(person_id, "face", embedding, quality=1.0,   # enrolled photos: always admissible
                                       meta={"tier": "enrolled", "score": 1.0})
                registry.save_face_crop(person_id, crop, cfg.get("face_images", 10))
            print(f"  {len(samples)} face samples")
        if args.voice:
            from modules.audio.speaker import ecapa_embedder

            embedder = ecapa_embedder(config.get("audio", {}).get("mic", {}).get("speaker_model_dir",
                                                                                 "model/speechbrain"))
            person_id = person_id or registry.create_person(args.name)
            for path in args.voice:
                vector, duration = voice_embedding(embedder, path)
                registry.add_embedding(person_id, "voice", vector, duration_sec=duration)
                print(f"  voice {path}: {duration:.1f} s")
        print(f"{'Enrolled' if created else 'Updated'} {args.name} ({person_id[:8]}): faces "
              f"{registry.counts[person_id]['face']}, voices {registry.counts[person_id]['voice']} → {registry.path}")
    finally:
        registry.close()


if __name__ == "__main__":
    main()
