"""
Cheap per-frame motion detection: MOG2 background subtraction, morphology, contour area filter.
"""
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

Box = List[int]  # [x1, y1, x2, y2]


@dataclass
class MotionResult:
    motion: bool
    boxes: List[Box] = field(default_factory=list)
    area_ratio: float = 0.0  # moving area / frame area


class MotionDetector:
    def __init__(self, min_area: int = 800, history: int = 300, var_threshold: float = 25,
                 warmup_frames: int = 30, blur_kernel: int = 5):
        """
        :param min_area: Ignore contours smaller than this many pixels
        :param history: MOG2 history length, frames
        :param var_threshold: MOG2 variance threshold
        :param warmup_frames: Frames used to learn the background before reporting motion
        :param blur_kernel: Gaussian blur kernel (odd), suppresses sensor noise
        """
        self.min_area = min_area
        self.warmup_frames = warmup_frames
        self.blur_kernel = blur_kernel
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold,
                                                             detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.frames_seen = 0

    def detect(self, frame: np.ndarray) -> MotionResult:
        """
        :param frame: BGR frame
        :return: MotionResult with bounding boxes of moving regions
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        mask = self.subtractor.apply(gray)
        self.frames_seen += 1
        if self.frames_seen <= self.warmup_frames:
            return MotionResult(False)

        # Shadows are 127 in the MOG2 mask; keep only confident foreground
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes, area = [], 0.0
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([int(x), int(y), int(x + w), int(y + h)])
            area += contour_area
        return MotionResult(bool(boxes), boxes, round(area / (frame.shape[0] * frame.shape[1]), 4))
