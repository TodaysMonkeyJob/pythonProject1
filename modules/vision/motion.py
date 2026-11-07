"""
Cheap per-frame motion detection: MOG2 background subtraction, morphology, contour area filter.

A3 (TOD-43) adaptive measures (``adaptive=True``, the default; ``False`` = the original detector, kept for the
benchmark ``tools/camera_bench.py``):

* **learning rate tied to the frame rate** — the background adapts over ``background_sec`` seconds whatever the fps
  (MOG2's own rate depends on frames: at 2 fps a 300-frame history is 150 s, at 30 fps 10 s);
* **illumination jumps** — a brightness change of more than ``illumination_k`` × the recent frame-to-frame jitter
  (and at least ``illumination_min`` grey levels) in ``illumination_share`` of the grid cells, all the same way (a
  person close to the camera changes only the cells it covers) re-learns the background at once and suppresses motion
  for
  ``illumination_holdoff_sec`` (learning fast meanwhile); slow changes (clouds, dusk) are followed by the learning rate;
* **adaptive per-region threshold** — the frame is split into ``grid`` × ``grid`` cells; each cell keeps a slow average
  of its foreground share while nothing real moves there (its noise level: foliage, curtains, a monitor). A cell
  counts as moving only when its share exceeds ``noise_k`` × its noise level + ``cell_min_share``, so a tree that sways
  all day stops reporting motion while a person crossing the same cell still does;
* **persistence** — motion must be seen in ``persist_frames`` of the last ``persist_window`` frames (single-frame
  flicker, rain drops);
* **masks** — ``mask_rects`` ([x1, y1, x2, y2] in pixels) are never analysed (a window onto a road).

``MotionResult.area_ratio`` is the moving share of the frame after these filters (the HFS camera subsystem's
``motion_intensity``, smoothed by the detector module).
"""
import collections
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence

import cv2
import numpy as np

Box = List[int]  # [x1, y1, x2, y2]


@dataclass
class MotionResult:
    motion: bool
    boxes: List[Box] = field(default_factory=list)
    area_ratio: float = 0.0  # moving area / frame area
    illumination_change: bool = False


class MotionDetector:
    def __init__(self, min_area: int = 800, history: int = 300, var_threshold: float = 25,
                 warmup_frames: int = 30, blur_kernel: int = 5, adaptive: bool = True, fps: float = 15.0,
                 background_sec: float = 20.0, illumination_k: float = 6.0, illumination_min: float = 6.0,
                 illumination_holdoff_sec: float = 2.0, illumination_share: float = 0.7, grid: int = 8,
                 noise_k: float = 3.0,
                 cell_min_share: float = 0.02, noise_adapt_sec: float = 30.0, persist_frames: int = 2,
                 persist_window: int = 3, mask_rects: Optional[Sequence[Sequence[int]]] = None):
        """
        :param min_area: Ignore contours smaller than this many pixels
        :param history: MOG2 history length, frames (non-adaptive mode only sets the learning rate through it)
        :param var_threshold: MOG2 variance threshold
        :param warmup_frames: Frames used to learn the background before reporting motion
        :param blur_kernel: Gaussian blur kernel (odd), suppresses sensor noise
        :param adaptive: The A3 measures below (False = the original detector)
        :param fps: Frame rate the detector is fed at (learning rate, hold-off and noise adaptation in seconds)
        :param background_sec: The background adapts over this many seconds
        :param illumination_k: Brightness jump threshold in units of the recent frame-to-frame jitter …
        :param illumination_min: … and at least this many grey levels
        :param illumination_holdoff_sec: No motion this long after a jump (the model re-learns)
        :param illumination_share: A jump changes at least this share of the grid cells, all the same way
        :param grid: Cells per side for the adaptive per-region threshold
        :param noise_k: A cell moves when its foreground share > noise_k × its noise level + cell_min_share
        :param cell_min_share: Minimum foreground share of a cell
        :param noise_adapt_sec: Time constant of a cell's noise level
        :param persist_frames: Motion needs this many moving frames …
        :param persist_window: … among the last this many
        :param mask_rects: Regions never analysed, [x1, y1, x2, y2] in pixels
        """
        self.min_area = min_area
        self.warmup_frames = warmup_frames
        self.blur_kernel = blur_kernel
        self.adaptive = adaptive
        self.fps = max(0.5, float(fps))
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold,
                                                             detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.frames_seen = 0
        self.learning_rate = min(1.0, 1.0 / (self.fps * background_sec))
        self.illumination_k, self.illumination_min = illumination_k, illumination_min
        self.holdoff_frames = max(1, int(round(illumination_holdoff_sec * self.fps)))
        self.grid, self.noise_k, self.cell_min_share = grid, noise_k, cell_min_share
        self.noise_alpha = min(1.0, 1.0 / (self.fps * noise_adapt_sec))
        self.persist_frames, self.persist_window = persist_frames, persist_window
        self.mask_rects = [list(map(int, r)) for r in mask_rects or []]
        self.illumination_share = illumination_share
        self._cells: Optional[np.ndarray] = None
        self._jitter = 1.0
        self._holdoff = 0
        self._noise: Optional[np.ndarray] = None
        self._recent: Deque[bool] = collections.deque(maxlen=persist_window)
        self._mask: Optional[np.ndarray] = None

    def _static_mask(self, shape) -> Optional[np.ndarray]:
        if not self.mask_rects:
            return None
        if self._mask is None or self._mask.shape != shape:
            self._mask = np.full(shape, 255, np.uint8)
            for x1, y1, x2, y2 in self.mask_rects:
                self._mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 0
        return self._mask

    def _illumination_jump(self, gray: np.ndarray) -> bool:
        """
        A brightness change across most of the frame, far above the recent frame-to-frame jitter (lights on / off,
        auto exposure). A large object close to the camera changes the mean too, but only in the cells it covers:
        a jump needs ``illumination_share`` of the grid cells to change in the same direction.
        """
        cells = cv2.resize(gray.astype(np.float32), (self.grid, self.grid), interpolation=cv2.INTER_AREA)
        previous, self._cells = self._cells, cells
        if previous is None:
            return False
        delta = cells - previous
        median = float(np.median(delta))
        threshold = max(self.illumination_min, self.illumination_k * self._jitter)
        same_way = np.mean(np.abs(delta) > threshold) >= self.illumination_share and \
            np.mean(np.sign(delta) == np.sign(median)) >= self.illumination_share
        jump = bool(abs(median) > threshold and same_way)
        if not jump:
            self._jitter = 0.95 * self._jitter + 0.05 * max(abs(median), 0.2)
        return jump

    def detect(self, frame: np.ndarray) -> MotionResult:
        """
        :param frame: BGR frame
        :return: MotionResult with bounding boxes of moving regions
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        if not self.adaptive:
            return self._legacy(gray)
        jump = self._illumination_jump(gray)
        if jump:
            self._holdoff = self.holdoff_frames
        # a jump: learn the new lighting at once, then fast (~0.5 s) through the hold-off so the variances of moving
        # textures (foliage) are learned again before motion counts
        rate = 1.0 if jump else max(self.learning_rate, 2.0 / self.fps) if self._holdoff > 0 else self.learning_rate
        mask = self.subtractor.apply(gray, learningRate=rate if self.frames_seen > self.warmup_frames else -1)
        self.frames_seen += 1
        if self.frames_seen <= self.warmup_frames:
            return MotionResult(False)
        if self._holdoff > 0:
            self._holdoff -= 1
            self._recent.append(False)
            return MotionResult(False, illumination_change=True)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        static = self._static_mask(mask.shape)
        if static is not None:
            mask = cv2.bitwise_and(mask, static)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = self._adaptive_cells(mask)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        boxes, area = self._boxes(mask)
        self._recent.append(bool(boxes))
        persistent = sum(self._recent) >= min(self.persist_frames, len(self._recent))
        if not persistent:
            return MotionResult(False)
        return MotionResult(bool(boxes), boxes, round(area / (frame.shape[0] * frame.shape[1]), 4))

    def _adaptive_cells(self, mask: np.ndarray) -> np.ndarray:
        """Zeroes cells whose foreground share does not exceed their own noise level; updates the noise levels."""
        share = cv2.resize(mask.astype(np.float32) / 255.0, (self.grid, self.grid), interpolation=cv2.INTER_AREA)
        if self._noise is None:
            self._noise = np.zeros_like(share)
        moving = share > self.noise_k * self._noise + self.cell_min_share
        # the noise level follows cells that do not move now, and — slower — chronic movers (foliage) too, so a
        # region that "moves" for minutes becomes background noise; a person passes in seconds
        alpha = np.where(moving, self.noise_alpha * 0.25, self.noise_alpha)
        self._noise = (1 - alpha) * self._noise + alpha * share
        if moving.all():
            return mask
        h, w = mask.shape
        keep = cv2.resize(moving.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return mask * keep

    def _boxes(self, mask: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes, area = [], 0.0
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([int(x), int(y), int(x + w), int(y + h)])
            area += contour_area
        return boxes, area

    def _legacy(self, gray: np.ndarray) -> MotionResult:
        """The original detector (before A3): MOG2 at its own rate, fixed threshold, area filter."""
        mask = self.subtractor.apply(gray)
        self.frames_seen += 1
        if self.frames_seen <= self.warmup_frames:
            return MotionResult(False)
        # Shadows are 127 in the MOG2 mask; keep only confident foreground
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        boxes, area = self._boxes(mask)
        return MotionResult(bool(boxes), boxes, round(area / (gray.shape[0] * gray.shape[1]), 4))
