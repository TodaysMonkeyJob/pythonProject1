"""
Learned additions to the rule-based action module: a skeleton action classifier and a cigarette detector.

* ``ActionClassifier`` — ``models/action_classifier.joblib`` from ``tools/train_actions.py`` (HMDB51 + NTU RGB+D 60
  skeletons, COCO-17): a track's last ``window_sec`` of poses → probabilities of walking, running, sitting_down,
  standing_up, falling, staggering, fighting, smoking, drinking, eating, phone_call, waving, picking_up, throwing,
  climbing, hugging, distress, other. sklearn only (the gas models use it on the Pi too).
* ``CigaretteDetector`` — ``models/cigarette_yolo11n.pt`` from ``tools/train_cigarette.py`` (CigDet): run on a crop
  around a person's head and hands, it tells whether a cigarette is actually visible — the visual evidence the
  wrist-to-mouth rule and the classifier's ``smoking`` lack.
"""
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from modules.vision.action_features import window_features
from system_setup.logger_setup import init_logger

logger = init_logger()

HEAD_HANDS = (0, 1, 2, 3, 4, 5, 6, 9, 10)          # nose, eyes, ears, shoulders, wrists (COCO-17)


class ActionClassifier:
    def __init__(self, path: str):
        import joblib
        import sklearn

        artifact = joblib.load(path)
        if artifact.get("sklearn_version") != sklearn.__version__:
            logger.warning(f"Action classifier trained with scikit-learn {artifact.get('sklearn_version')}, "
                           f"running {sklearn.__version__}; retrain if predictions look wrong")
        self.model = artifact["model"]
        self.classes: List[str] = list(self.model.classes_)
        self.window_sec = float(artifact.get("window_sec", 2.5))
        self.min_samples = int(artifact.get("min_samples", 6))
        self.min_score = float(artifact.get("min_score", 0.4))

    @classmethod
    def from_path(cls, path: Optional[str]) -> Optional["ActionClassifier"]:
        if not path or not os.path.isfile(path):
            return None
        try:
            return cls(path)
        except Exception as e:  # a broken model must not stop the rule-based recognition
            logger.error(f"Action classifier {path} unavailable: {e}")
            return None

    def predict(self, samples: Sequence[Tuple[float, np.ndarray]],
                others: Optional[Sequence[Sequence[np.ndarray]]] = None) -> Optional[Dict[str, float]]:
        """
        :param samples: [(t, (17, 2) keypoints with NaN = not visible)] of one track, oldest first
        :param others: per sample, the keypoints of the other persons in that frame
        :return: {class: probability}, or None for too few samples
        """
        if len(samples) < self.min_samples:
            return None
        rows = [(t, np.nan_to_num(kp), (~np.isnan(kp).any(axis=1)).astype(float)) for t, kp in samples]
        feature = window_features(rows, others, self.min_score)
        if feature is None:
            return None
        proba = self.model.predict_proba(feature[None, :])[0]
        return {c: float(p) for c, p in zip(self.classes, proba, strict=True)}


def head_hands_crop(frame: np.ndarray, keypoints: np.ndarray,
                    margin: float = 0.6) -> Optional[Tuple[int, int, int, int]]:
    """:return: (x1, y1, x2, y2) around the visible head / shoulder / wrist keypoints, widened by ``margin``"""
    pts = keypoints[list(HEAD_HANDS)]
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) < 3:
        return None
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    size = max(x1 - x0, y1 - y0, 24.0) * (1 + margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h, w = frame.shape[:2]
    box = (int(max(0, cx - size / 2)), int(max(0, cy - size / 2)), int(min(w, cx + size / 2)),
           int(min(h, cy + size / 2)))
    return box if box[2] - box[0] >= 16 and box[3] - box[1] >= 16 else None


class CigaretteDetector:
    def __init__(self, path: str, imgsz: int = 320, conf: float = 0.35):
        self.path, self.imgsz, self.conf = path, imgsz, conf
        self.model = None

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> Optional["CigaretteDetector"]:
        path = cfg.get("cigarette_model")
        if not path or not os.path.exists(path):
            return None
        return cls(path, int(cfg.get("cigarette_imgsz", 320)), float(cfg.get("cigarette_conf", 0.35)))

    def detect(self, frame: np.ndarray, keypoints: np.ndarray) -> float:
        """:return: the best cigarette confidence in the head / hands crop (0 = none)"""
        return self.locate(frame, keypoints)[0]

    def locate(self, frame: np.ndarray, keypoints: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """:return: (best cigarette confidence, its box centre in frame pixels) in the head / hands crop"""
        box = head_hands_crop(frame, keypoints)
        if box is None:
            return 0.0, None
        if self.model is None:
            from ultralytics import YOLO

            self.model = YOLO(self.path, task="detect")
        crop = frame[box[1]:box[3], box[0]:box[2]]
        result = self.model.predict(crop, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return 0.0, None
        best = int(result.boxes.conf.argmax())
        x1, y1, x2, y2 = (float(v) for v in result.boxes.xyxy[best])
        return float(result.boxes.conf[best]), np.array([box[0] + (x1 + x2) / 2, box[1] + (y1 + y2) / 2])
