"""
Face quality gate (F14 / TOD-25): decides, before any embedding is computed, whether a YuNet face is good enough.

Partial faces (cut off at the frame or person-box edge, side views, blurred, tiny, badly lit) give SFace embeddings
that match nobody, which used to create a new ``unknown_N`` for people already registered. Only YuNet's output and
OpenCV are used — no extra model:

* detector score, face size and inter-eye distance;
* truncation: face box + 5 landmarks inside the frame (margin = ``truncation_margin`` × face size) and the
  landmarks inside the person box;
* pose from the 5 landmarks: yaw = atan(2 · nose offset along the eye line / eye distance), pitch = atan of the
  eye→nose vs nose→mouth imbalance (both ≈ 0 for the SFace alignment template; the factor assumes a nose depth of
  about half the eye distance — an approximation, good enough for ±35° / ±25° gates);
* sharpness: variance of the Laplacian of the aligned 112×112 crop; exposure: its mean brightness.

Tiers: ``good`` (everything passes) may create / enrich identities; ``usable`` (only pose and / or sharpness slightly
out) may be matched against the gallery but never creates a person and is never stored; ``reject`` is not embedded.
"""
import math
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np

# Expected (eye→nose − nose→mouth) / (eye→mouth) of a frontal face: SFace's alignment template
# (eyes y ≈ 51.7, nose 71.7, mouth corners 92.4 in the 112 px crop)
FRONTAL_PITCH_RATIO = (71.7 - 51.7 - (92.4 - 71.7)) / (92.4 - 51.7)


@dataclass
class QualityConfig:
    """``identity.quality`` config section; tune on your camera (``python -m tools.eval_faces``)."""
    min_score: float = 0.85          # YuNet score
    min_face_size: int = 80          # px, face box width and height
    min_eye_distance: float = 30.0   # px
    truncation_margin: float = 0.05  # face box + landmarks this far (× face size) inside the frame
    max_yaw: float = 35.0            # degrees: good
    max_pitch: float = 25.0
    usable_yaw: float = 50.0         # degrees: still usable for matching
    usable_pitch: float = 35.0
    min_sharpness: float = 60.0      # Laplacian variance of the aligned crop
    usable_sharpness: float = 25.0
    min_brightness: float = 40.0     # mean grey level of the aligned crop
    max_brightness: float = 220.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "QualityConfig":
        data = data or {}
        return cls(**{f.name: type(f.default)(data[f.name]) for f in fields(cls) if f.name in data})


def landmarks(face: np.ndarray) -> np.ndarray:
    """:return: (5, 2) YuNet landmarks: right eye, left eye, nose tip, right / left mouth corner"""
    return np.asarray(face[4:14], np.float32).reshape(5, 2)


def pose(face: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Yaw / pitch estimated from the 5 landmarks.

    :return: {"yaw", "pitch", "eye_distance"} in degrees / px, or None for degenerate landmarks
    """
    right_eye, left_eye, nose, mouth_r, mouth_l = landmarks(face)
    eye_line = left_eye - right_eye
    eye_distance = float(np.linalg.norm(eye_line))
    if eye_distance < 1e-3:
        return None
    ux = eye_line / eye_distance
    uy = np.array([-ux[1], ux[0]], np.float32)       # perpendicular, pointing down the face
    eye_mid, mouth_mid = (right_eye + left_eye) / 2, (mouth_r + mouth_l) / 2
    upper, lower = float((nose - eye_mid) @ uy), float((mouth_mid - nose) @ uy)
    if upper + lower <= 1e-3:
        return None
    yaw = math.degrees(math.atan(2 * float((nose - eye_mid) @ ux) / eye_distance))
    pitch = math.degrees(math.atan((upper - lower) / (upper + lower) - FRONTAL_PITCH_RATIO))
    return {"yaw": yaw, "pitch": pitch, "eye_distance": eye_distance}


def _inside(points: np.ndarray, box: Sequence[float], margin: float) -> bool:
    x1, y1, x2, y2 = box
    return bool(np.all(points[:, 0] >= x1 + margin) and np.all(points[:, 0] <= x2 - margin)
                and np.all(points[:, 1] >= y1 + margin) and np.all(points[:, 1] <= y2 - margin))


def assess(frame_shape: Sequence[int], face: np.ndarray, person_bbox: Optional[Sequence[float]] = None,
           aligned: Optional[np.ndarray] = None, cfg: Optional[QualityConfig] = None) -> Dict[str, Any]:
    """
    Quality of one detected face.

    :param frame_shape: ``frame.shape`` of the image the face row refers to
    :param face: YuNet row (x, y, w, h, 5 landmarks, score)
    :param person_bbox: [x1, y1, x2, y2] of the person the face was assigned to; None = frame only
    :param aligned: ``FaceRecognizerSF.alignCrop`` result (sharpness / exposure); None = those checks skipped
    :return: {"tier": good|usable|reject, "score", "reasons", "yaw", "pitch", "sharpness", "brightness"}
    """
    cfg = cfg or QualityConfig()
    x, y, w, h = (float(v) for v in face[:4])
    score = float(face[-1])
    hard, soft = [], []           # reject reasons / "slightly out" reasons
    if score < cfg.min_score:
        hard.append("score")
    if min(w, h) < cfg.min_face_size:
        hard.append("small")
    height, width = int(frame_shape[0]), int(frame_shape[1])
    points = np.vstack([landmarks(face), [[x, y], [x + w, y + h]]])
    if not _inside(points, (0, 0, width - 1, height - 1), cfg.truncation_margin * min(w, h)):
        hard.append("truncated")
    elif person_bbox is not None and not _inside(landmarks(face), person_bbox, 0.0):
        hard.append("outside_person")
    angles = pose(face)
    yaw = pitch = None
    if angles is None:
        hard.append("landmarks")
    else:
        yaw, pitch = angles["yaw"], angles["pitch"]
        if angles["eye_distance"] < cfg.min_eye_distance:
            hard.append("small")
        if abs(yaw) > cfg.usable_yaw or abs(pitch) > cfg.usable_pitch:
            hard.append("pose")
        elif abs(yaw) > cfg.max_yaw or abs(pitch) > cfg.max_pitch:
            soft.append("pose")
    sharpness = brightness = None
    if aligned is not None and aligned.size:
        grey = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) if aligned.ndim == 3 else aligned
        sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        brightness = float(grey.mean())
        if sharpness < cfg.usable_sharpness:
            hard.append("blur")
        elif sharpness < cfg.min_sharpness:
            soft.append("blur")
        if not cfg.min_brightness <= brightness <= cfg.max_brightness:
            hard.append("exposure")
    tier = "reject" if hard else ("usable" if soft else "good")
    # 0..1 summary used to rank a track's faces (best first)
    quality = score * min(1.0, min(w, h) / 112)
    if yaw is not None:
        quality *= max(0.0, 1 - abs(yaw) / 90) * max(0.0, 1 - abs(pitch) / 90)
    if sharpness is not None:
        quality *= min(1.0, sharpness / (2 * cfg.min_sharpness))
    return {"tier": tier, "score": round(quality, 4), "reasons": sorted(set(hard + soft)),
            "yaw": None if yaw is None else round(yaw, 1), "pitch": None if pitch is None else round(pitch, 1),
            "sharpness": None if sharpness is None else round(sharpness, 1),
            "brightness": None if brightness is None else round(brightness, 1)}
