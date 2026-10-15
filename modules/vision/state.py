"""
State machines that turn per-frame results into change events: tracks appear/disappear, motion
start/end, scene active/inactive. Time is passed in explicitly so the logic is testable.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Box = List[int]


@dataclass
class Detection:
    track_id: Optional[int]
    cls: str
    bbox: Box
    confidence: float


@dataclass
class Track:
    track_id: int
    cls: str
    bbox: Box
    confidence: float
    first_seen: float
    last_seen: float
    aliases: set = field(default_factory=set)  # tracker ids merged into this track


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """
    :return: Intersection over union of two [x1, y1, x2, y2] boxes
    """
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class TrackRegistry:
    """
    Keeps the objects currently in the scene. A track "appears" when a new tracker id shows up and
    "disappears" when it has not been seen for ``lost_timeout_sec``. A new tracker id that overlaps
    (IoU ≥ ``reid_iou``) a known track of the same class that was not matched in this update is treated
    as an id switch of that track, so a static object produces one "appeared" event.
    """

    def __init__(self, lost_timeout_sec: float = 10.0, reid_iou: float = 0.5):
        self.lost_timeout_sec = lost_timeout_sec
        self.reid_iou = reid_iou
        self.tracks: Dict[int, Track] = {}
        self._alias: Dict[int, int] = {}  # tracker id → track id

    @property
    def has_objects(self) -> bool:
        return bool(self.tracks)

    def visible(self, now: float, max_age_sec: float) -> List[Track]:
        """
        :return: Tracks seen within max_age_sec
        """
        return [t for t in self.tracks.values() if now - t.last_seen <= max_age_sec]

    def update(self, detections: Sequence[Detection], now: float) -> Tuple[List[Track], List[Track]]:
        """
        :param detections: Detections of one YOLO run (detections without a track id are ignored)
        :param now: Current time, seconds
        :return: (appeared, disappeared) tracks
        """
        appeared, matched = [], set()
        tracked = [d for d in detections if d.track_id is not None]
        # Known ids first, so an id switch is only matched against tracks that really went unseen
        tracked.sort(key=lambda d: self._alias.get(d.track_id, d.track_id) not in self.tracks)
        for det in tracked:
            track_id = self._alias.get(det.track_id, det.track_id)
            track = self.tracks.get(track_id)
            if track is None:
                track = self._find_switched(det, matched)
                if track is not None:
                    self._alias[det.track_id] = track.track_id
                    track.aliases.add(det.track_id)
            if track is None:
                track = Track(det.track_id, det.cls, list(det.bbox), det.confidence, now, now)
                self.tracks[det.track_id] = track
                appeared.append(track)
            track.bbox, track.confidence, track.last_seen = list(det.bbox), det.confidence, now
            matched.add(track.track_id)
        return appeared, self.expire(now)

    def expire(self, now: float) -> List[Track]:
        """
        :return: Tracks removed because they were not seen for lost_timeout_sec
        """
        gone = [t for t in self.tracks.values() if now - t.last_seen > self.lost_timeout_sec]
        for track in gone:
            del self.tracks[track.track_id]
            for alias in track.aliases:
                self._alias.pop(alias, None)
        return gone

    def _find_switched(self, det: Detection, matched: set) -> Optional[Track]:
        best, best_iou = None, self.reid_iou
        for track in self.tracks.values():
            if track.track_id in matched or track.cls != det.cls:
                continue
            overlap = iou(track.bbox, det.bbox)
            if overlap >= best_iou:
                best, best_iou = track, overlap
        return best


class MotionState:
    """Motion starts on the first moving frame and ends after ``end_delay_sec`` without motion."""

    def __init__(self, end_delay_sec: float = 2.0):
        self.end_delay_sec = end_delay_sec
        self.active = False
        self.last_motion = None

    def update(self, motion: bool, now: float) -> Optional[str]:
        """
        :return: "start", "end" or None when nothing changed
        """
        if motion:
            self.last_motion = now
            if not self.active:
                self.active = True
                return "start"
        elif self.active and now - self.last_motion >= self.end_delay_sec:
            self.active = False
            return "end"
        return None


class SceneState:
    """Scene is active while there is motion or an object; it goes inactive ``timeout_sec`` after both stop."""

    def __init__(self, timeout_sec: float = 10.0):
        self.timeout_sec = timeout_sec
        self.active = False
        self.last_active = None

    def update(self, busy: bool, now: float) -> Optional[bool]:
        """
        :param busy: Motion or at least one object right now
        :return: New state when it changed, otherwise None
        """
        if busy:
            self.last_active = now
            if not self.active:
                self.active = True
                return True
        elif self.active and now - self.last_active >= self.timeout_sec:
            self.active = False
            return False
        return None
