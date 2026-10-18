"""
YOLO object detection + ByteTrack tracking through ultralytics. The model loads lazily, so importing
this module does not pull torch.
"""
from typing import List, Optional, Sequence

import numpy as np

from modules.vision.state import Detection


def limit_threads(threads) -> None:
    """
    CPU threads for torch / OpenCV in this process. Several camera pipelines each using every core fight over the CPU
    (4 cameras on a 10-core Mac: 570 ms per YOLO frame instead of ~40 ms), so each gets a small share.
    """
    if not threads:
        return
    import cv2
    import torch

    torch.set_num_threads(int(threads))
    cv2.setNumThreads(int(threads))


class ObjectDetector:
    def __init__(self, model_path: str, imgsz: int = 320, conf: float = 0.4,
                 classes: Optional[Sequence[int]] = None, tracker: str = "bytetrack.yaml"):
        """
        :param model_path: NCNN export directory (Pi) or .pt file
        :param imgsz: Inference size
        :param conf: Confidence threshold
        :param classes: COCO class ids to keep; None = all
        :param tracker: Ultralytics tracker config
        """
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.classes = list(classes) if classes else None
        self.tracker = tracker
        self.model = None

    def load(self) -> None:
        from ultralytics import YOLO

        self.model = YOLO(self.model_path, task="detect")

    def track(self, frame: np.ndarray) -> List[Detection]:
        """
        :param frame: BGR frame
        :return: Detections with ByteTrack ids (track_id None until the tracker confirms a track)
        """
        if self.model is None:
            self.load()
        result = self.model.track(frame, persist=True, tracker=self.tracker, imgsz=self.imgsz, conf=self.conf,
                                  classes=self.classes, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy().round().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(cls)
        return [Detection(None if track_id is None else int(track_id), result.names[int(c)],
                          [int(v) for v in box], round(float(p), 3))
                for box, c, p, track_id in zip(xyxy, cls, conf, ids, strict=True)]
