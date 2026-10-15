"""
Pose estimation (ultralytics YOLO pose, 17 COCO keypoints) and skeleton geometry helpers.

Keypoints are an array (17, 2) in pixels with NaN for points below the confidence threshold.
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
N_KEYPOINTS = 17


@dataclass
class Pose:
    bbox: List[int]          # [x1, y1, x2, y2]
    keypoints: np.ndarray    # (17, 2), NaN = not visible
    confidence: float = 1.0


def mask_keypoints(xy: np.ndarray, conf: Optional[np.ndarray], min_conf: float) -> np.ndarray:
    """
    :param xy: (17, 2) keypoint coordinates
    :param conf: (17,) keypoint confidences or None
    :param min_conf: Points below this confidence become NaN
    :return: (17, 2) float array
    """
    points = np.array(xy, dtype=float).reshape(N_KEYPOINTS, 2)
    if conf is not None:
        points[np.asarray(conf) < min_conf] = np.nan
    points[(points == 0).all(axis=1)] = np.nan  # ultralytics reports invisible points as (0, 0)
    return points


def visible(*points: np.ndarray) -> bool:
    return all(p is not None and not np.isnan(p).any() for p in points)


def midpoint(kp: np.ndarray, a: int, b: int) -> Optional[np.ndarray]:
    """
    :return: Midpoint of two keypoints, the single visible one, or None
    """
    pa, pb = kp[a], kp[b]
    if visible(pa, pb):
        return (pa + pb) / 2
    if visible(pa):
        return pa
    if visible(pb):
        return pb
    return None


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    :return: Angle ABC in degrees (at vertex b), NaN if a point is missing
    """
    if not visible(a, b, c):
        return math.nan
    v1, v2 = a - b, c - b
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return math.nan
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0))))


def angle_from_vertical(top: Optional[np.ndarray], bottom: Optional[np.ndarray]) -> float:
    """
    :return: 0° for a vertical segment, 90° for a horizontal one; NaN if missing
    """
    if top is None or bottom is None or not visible(top, bottom):
        return math.nan
    dx, dy = top - bottom
    if dx == 0 and dy == 0:
        return math.nan
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def torso_angle(kp: np.ndarray) -> float:
    """Torso (shoulder centre → hip centre) angle from vertical, degrees."""
    return angle_from_vertical(midpoint(kp, L_SHOULDER, R_SHOULDER), midpoint(kp, L_HIP, R_HIP))


def knee_angle(kp: np.ndarray) -> float:
    """Hip-knee-ankle angle, mean of the visible legs (180° = straight leg)."""
    angles = [joint_angle(kp[L_HIP], kp[L_KNEE], kp[L_ANKLE]), joint_angle(kp[R_HIP], kp[R_KNEE], kp[R_ANKLE])]
    angles = [a for a in angles if not math.isnan(a)]
    return float(np.mean(angles)) if angles else math.nan


def thigh_angle_from_horizontal(kp: np.ndarray) -> float:
    """Hip→knee angle from horizontal, mean of visible legs (0° = thigh horizontal)."""
    angles = [90.0 - angle_from_vertical(kp[h], kp[k]) for h, k in ((L_HIP, L_KNEE), (R_HIP, R_KNEE))
              if visible(kp[h], kp[k])]
    return float(np.mean(angles)) if angles else math.nan


def torso_length(kp: np.ndarray, bbox: Sequence[float]) -> float:
    """
    Body scale: shoulder centre to hip centre distance. Without hips (close-ups, desks) the shoulder width is
    used (≈ 0.85 torso length when facing the camera), then 30 % of the bbox height.
    """
    top, bottom = midpoint(kp, L_SHOULDER, R_SHOULDER), midpoint(kp, L_HIP, R_HIP)
    if top is not None and bottom is not None:
        length = float(np.linalg.norm(top - bottom))
        if length > 0:
            return length
    if visible(kp[L_SHOULDER], kp[R_SHOULDER]):
        width = float(np.linalg.norm(kp[L_SHOULDER] - kp[R_SHOULDER]))
        if width > 0:
            return width / 0.85
    return 0.3 * max(1.0, bbox[3] - bbox[1])


def legs_visible(kp: np.ndarray) -> bool:
    """At least one knee or ankle is visible."""
    return any(visible(kp[i]) for i in (L_KNEE, R_KNEE, L_ANKLE, R_ANKLE))


def mouth_point(kp: np.ndarray) -> Optional[np.ndarray]:
    """Mouth proxy: nose, or the eye centre when the nose is not visible."""
    if visible(kp[NOSE]):
        return kp[NOSE]
    return midpoint(kp, L_EYE, R_EYE)


def wrist_distances(kp: np.ndarray, bbox: Sequence[float]) -> Dict[int, float]:
    """
    :return: {L_WRIST: d, R_WRIST: d} — wrist-to-mouth distances in torso lengths, NaN when not visible
    """
    mouth = mouth_point(kp)
    scale = torso_length(kp, bbox)
    return {w: float(np.linalg.norm(kp[w] - mouth)) / scale if mouth is not None and visible(kp[w]) else math.nan
            for w in (L_WRIST, R_WRIST)}


def wrist_to_mouth(kp: np.ndarray, bbox: Sequence[float]) -> float:
    """
    :return: Distance of the closer wrist to the mouth in torso lengths, NaN if not visible
    """
    distances = [d for d in wrist_distances(kp, bbox).values() if not math.isnan(d)]
    return min(distances) if distances else math.nan


class PoseEstimator:
    """
    ultralytics YOLO pose model, loaded lazily (NCNN export directory on the Pi or a .pt file).
    """

    def __init__(self, model_path: str, imgsz: int = 320, conf: float = 0.4, keypoint_conf: float = 0.4):
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.keypoint_conf = keypoint_conf
        self.model = None

    def load(self) -> None:
        from ultralytics import YOLO

        self.model = YOLO(self.model_path, task="pose")

    def estimate(self, frame: np.ndarray) -> List[Pose]:
        """
        :param frame: BGR frame
        :return: One Pose per detected person
        """
        if self.model is None:
            self.load()
        result = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy().round().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        xy = result.keypoints.xy.cpu().numpy()
        kconf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else [None] * len(xy)
        return [Pose([int(v) for v in box], mask_keypoints(points, c, self.keypoint_conf), round(float(s), 3))
                for box, s, points, c in zip(boxes, scores, xy, kconf, strict=True)]
