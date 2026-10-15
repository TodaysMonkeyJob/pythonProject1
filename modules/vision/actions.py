"""
Action recognition (F2): rules on top of pose keypoints, no trained model.

Per frame a posture is classified (standing / sitting / lying / walking / unknown); an action is confirmed
when it wins ≥ ``vote_threshold`` of the known frames in the last ``window_sec`` and an ``action`` event is
published only when the confirmed action of a track changes. ``possible_smoking`` is a separate signal:
wrist near the mouth ≥ ``smoking_min_count`` times in ``smoking_window_sec``, with ``gas_confirmed`` set when
the gas module reported a rise in the last ``gas_window_sec``.

Input comes from the vision shared buffer (raw frame + person boxes with track ids). Gas readings and
``object_detected``/``disappeared`` events reach this process through its control queue — the orchestrator
forwards them as event JSON strings.
"""
import collections
import math
import time
from dataclasses import dataclass, field, fields
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.events import Event
from core.module import BaseModule
from modules.vision.pose import (L_ANKLE, R_ANKLE, Pose, PoseEstimator, knee_angle, legs_visible,
                                 thigh_angle_from_horizontal, torso_angle, visible, wrist_distances)
from modules.vision.shared_frame import SharedFrameBuffer
from modules.vision.state import iou
from system_setup.logger_setup import init_logger

logger = init_logger()

POSTURES = ("standing", "sitting", "walking", "lying")
SMOKING = "possible_smoking"


@dataclass
class ActionConfig:
    """Thresholds; every field can be overridden from the ``actions`` config section."""
    window_sec: float = 2.5
    vote_threshold: float = 0.7
    min_votes: int = 3
    match_iou: float = 0.3
    track_lost_sec: float = 10.0
    torso_vertical_max: float = 25.0        # deg from vertical
    lying_torso_min: float = 60.0           # deg from vertical
    standing_knee_min: float = 160.0        # deg
    sitting_knee_range: Tuple[float, float] = (40.0, 130.0)  # 40° covers sitting on the floor
    thigh_horizontal_max: float = 45.0      # deg from horizontal
    frame_edge_margin: float = 0.02         # bbox this close to the bottom edge = cut by the frame, not furniture
    sitting_height_ratio: float = 0.75      # bbox height / this track's standing height
    walking_speed: float = 0.3              # bbox centre speed, body heights per second
    walking_scale_rate: float = 0.25        # |d ln(bbox height)/dt| — walking towards / away from the camera
    speed_window_sec: float = 1.0
    ankle_alternations_min: int = 2
    ankle_deadband: float = 0.03            # fraction of bbox height
    mouth_radius: float = 0.35              # wrist-mouth distance, torso lengths: hand reaches the mouth
    mouth_release: float = 0.6              # hand counts as gone again above this (hysteresis)
    mouth_episode_max_sec: float = 5.0      # longer contact (resting the head on the hands) is not a puff
    smoking_window_sec: float = 30.0
    smoking_min_count: int = 3
    gas_window_sec: float = 60.0
    gas_rise_pct: float = 0.5               # concentration increase within gas_window_sec

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ActionConfig":
        data = data or {}
        kwargs = {f.name: data[f.name] for f in fields(cls) if f.name in data}
        if "sitting_knee_range" in kwargs:
            kwargs["sitting_knee_range"] = tuple(kwargs["sitting_knee_range"])
        return cls(**kwargs)


def bbox_height(bbox: Sequence[float]) -> float:
    return max(1.0, float(bbox[3] - bbox[1]))


def bbox_centre(bbox: Sequence[float]) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=float)


def classify_posture(kp: np.ndarray, bbox: Sequence[float], standing_height: Optional[float],
                     cfg: ActionConfig, frame_size: Optional[Tuple[int, int]] = None) -> str:
    """
    Static posture of one frame.

    :param kp: (17, 2) keypoints, NaN = not visible
    :param bbox: Person box
    :param standing_height: This track's typical standing bbox height, or None
    :param frame_size: (width, height); needed to tell legs hidden by furniture from legs cut by the frame
    :return: "lying" | "sitting" | "sitting_occluded" | "standing" | "unknown".
             "sitting_occluded" = upright, legs hidden by something inside the frame (desk, table) — weak
             evidence that frame_action() turns into "sitting" unless the person moves.
    """
    torso = torso_angle(kp)
    knee = knee_angle(kp)
    thigh = thigh_angle_from_horizontal(kp)
    height = bbox_height(bbox)

    # Only a visible horizontal torso means lying: a wide box alone is usually a person cut by the frame edge
    if not math.isnan(torso) and torso >= cfg.lying_torso_min:
        return "lying"
    knee_low, knee_high = cfg.sitting_knee_range
    if not math.isnan(knee) and knee_low <= knee <= knee_high and not math.isnan(thigh) \
            and thigh <= cfg.thigh_horizontal_max:
        return "sitting"
    legs_unclear = math.isnan(knee) or knee < cfg.standing_knee_min
    if standing_height and height < cfg.sitting_height_ratio * standing_height and legs_unclear \
            and (math.isnan(torso) or torso < cfg.lying_torso_min):
        return "sitting"
    upright = not math.isnan(torso) and torso <= cfg.torso_vertical_max
    if upright and not legs_visible(kp):
        if frame_size is None:
            return "unknown"
        cut_by_frame = bbox[3] >= frame_size[1] * (1 - cfg.frame_edge_margin)
        return "unknown" if cut_by_frame else "sitting_occluded"
    if upright and (math.isnan(knee) or knee >= cfg.standing_knee_min):
        return "standing"
    return "unknown"


def frame_action(posture: str, speed: float, alternations: int, cfg: ActionConfig, scale_rate: float = 0.0) -> str:
    """
    Combines the static posture with motion over the recent history.

    :param speed: bbox centre speed, body heights per second
    :param alternations: Ankle order changes in the window
    :param scale_rate: |d ln(bbox height)/dt| — approaching or leaving the camera
    """
    if posture in ("lying", "sitting"):
        return posture
    moving = speed >= cfg.walking_speed or scale_rate >= cfg.walking_scale_rate or \
        (speed >= cfg.walking_speed / 2 and alternations >= cfg.ankle_alternations_min)
    if moving:
        return "walking"
    return "sitting" if posture == "sitting_occluded" else posture


def ankle_alternations(samples: Sequence["Sample"], deadband: float) -> int:
    """
    Counts how often the leading ankle changes (along the axis where the ankles differ most).
    """
    diffs = [(s.keypoints[L_ANKLE] - s.keypoints[R_ANKLE]) / bbox_height(s.bbox)
             for s in samples if visible(s.keypoints[L_ANKLE], s.keypoints[R_ANKLE])]
    if len(diffs) < 3:
        return 0
    diffs = np.array(diffs)
    series = diffs[:, int(np.argmax(diffs.std(axis=0)))]
    signs = [int(np.sign(v)) for v in series if abs(v) > deadband]
    return sum(1 for a, b in zip(signs, signs[1:], strict=False) if a != b)


def motion_rates(samples: Sequence["Sample"]) -> Tuple[float, float]:
    """
    Least-squares slopes over the samples, robust to per-frame box jitter.

    :return: (centre speed in body heights per second, |d ln(height)/dt| per second); zeros if < 3 samples
    """
    if len(samples) < 3 or samples[-1].t - samples[0].t <= 0:
        return 0.0, 0.0
    t = np.array([s.t for s in samples]) - samples[0].t
    centres = np.array([bbox_centre(s.bbox) for s in samples])
    heights = np.array([bbox_height(s.bbox) for s in samples])
    vx, vy = np.polyfit(t, centres[:, 0], 1)[0], np.polyfit(t, centres[:, 1], 1)[0]
    speed = float(math.hypot(vx, vy) / heights.mean())
    scale_rate = abs(float(np.polyfit(t, np.log(heights), 1)[0]))
    return speed, scale_rate


def vote(labels: Sequence[str], threshold: float, min_votes: int) -> Optional[str]:
    """
    :return: Label winning ≥ threshold of the known (non-"unknown") labels, or None
    """
    known = [label for label in labels if label != "unknown"]
    if len(known) < min_votes or len(known) < len(labels) / 2:
        return None
    label, count = collections.Counter(known).most_common(1)[0]
    return label if count / len(known) >= threshold else None


def match_poses(persons: Sequence[Dict[str, Any]], poses: Sequence[Pose], min_iou: float) -> List[Tuple[int, Pose]]:
    """
    Greedy IoU matching of pose detections to tracked persons.

    :param persons: [{"track_id", "bbox"}] from the vision module
    :return: [(track_id, pose)]
    """
    pairs = sorted(((iou(p["bbox"], pose.bbox), i, j) for i, p in enumerate(persons) for j, pose in enumerate(poses)),
                   reverse=True)
    used_p, used_q, result = set(), set(), []
    for overlap, i, j in pairs:
        if overlap < min_iou:
            break
        if i in used_p or j in used_q:
            continue
        used_p.add(i)
        used_q.add(j)
        result.append((int(persons[i]["track_id"]), poses[j]))
    return result


@dataclass
class Sample:
    t: float
    bbox: List[int]
    keypoints: np.ndarray
    label: str


@dataclass
class TrackActions:
    samples: Deque[Sample] = field(default_factory=collections.deque)
    action: Optional[str] = None
    since: float = 0.0
    standing_height: Optional[float] = None
    near_mouth: bool = False
    mouth_wrist: Optional[int] = None     # keypoint index of the wrist at the mouth
    mouth_since: float = 0.0
    mouth_events: Deque[float] = field(default_factory=collections.deque)
    smoking: bool = False
    smoking_gas_confirmed: bool = False
    last_seen: float = 0.0


class GasEvidence:
    """Remembers gas rises reported on the bus (gas_threshold / gas_rising alerts, rising readings)."""

    def __init__(self, window_sec: float = 60.0, rise_pct: float = 0.5):
        self.window_sec = window_sec
        self.rise_pct = rise_pct
        self.rises: Deque[float] = collections.deque()
        self.readings: Deque[Tuple[float, float]] = collections.deque()

    def add(self, event: Event, now: float) -> None:
        if event.source != "gas":
            return
        payload = event.payload
        if event.type == "alert" and payload.get("reason") in ("gas_threshold", "gas_rising"):
            self.rises.append(now)
        elif event.type == "gas_reading":
            concentration = payload.get("concentration_pct")
            if payload.get("trend") == "rising":
                self.rises.append(now)
            if isinstance(concentration, (int, float)):
                self.readings.append((now, float(concentration)))
                self._trim(now)
                if concentration - min(c for _, c in self.readings) >= self.rise_pct:
                    self.rises.append(now)

    def confirmed(self, now: float) -> bool:
        self._trim(now)
        return bool(self.rises)

    def _trim(self, now: float) -> None:
        while self.rises and now - self.rises[0] > self.window_sec:
            self.rises.popleft()
        while self.readings and now - self.readings[0][0] > self.window_sec:
            self.readings.popleft()


class ActionEngine:
    """
    Pure action logic (no I/O): feed person boxes + poses per frame, get event payloads back.
    """

    def __init__(self, cfg: Optional[ActionConfig] = None):
        self.cfg = cfg or ActionConfig()
        self.tracks: Dict[int, TrackActions] = {}
        self.gas = GasEvidence(self.cfg.gas_window_sec, self.cfg.gas_rise_pct)

    def on_event(self, event: Event, now: float) -> None:
        """Gas events feed the smoking confirmation; ``disappeared`` drops the track's history."""
        self.gas.add(event, now)
        if event.type == "object_detected" and event.payload.get("state") == "disappeared":
            self.tracks.pop(event.payload.get("track_id"), None)

    def update(self, now: float, persons: Sequence[Dict[str, Any]], poses: Sequence[Pose],
               frame_size: Optional[Tuple[int, int]] = None) -> List[Tuple[Dict[str, Any], str, Optional[float]]]:
        """
        :param now: Frame time, seconds
        :param persons: [{"track_id", "bbox"}] currently tracked persons
        :param poses: Pose detections of the same frame
        :param frame_size: (width, height) of the frame; lets the rules tell furniture occlusion from the frame edge
        :return: [(payload, severity, confidence)] for the events to publish
        """
        out = []
        for track_id, pose in match_poses(persons, poses, self.cfg.match_iou):
            track = self.tracks.setdefault(track_id, TrackActions(since=now))
            track.last_seen = now
            out.extend(self._update_track(track_id, track, pose, now, frame_size))
        for track_id in [t for t, track in self.tracks.items() if now - track.last_seen > self.cfg.track_lost_sec]:
            del self.tracks[track_id]
        return out

    def _update_track(self, track_id: int, track: TrackActions, pose: Pose, now: float,
                      frame_size: Optional[Tuple[int, int]] = None):
        cfg = self.cfg
        posture = classify_posture(pose.keypoints, pose.bbox, track.standing_height, cfg, frame_size)
        if posture == "standing":
            height = bbox_height(pose.bbox)
            track.standing_height = height if track.standing_height is None else max(
                height, 0.95 * track.standing_height)

        while track.samples and now - track.samples[0].t > cfg.window_sec:
            track.samples.popleft()
        recent = [s for s in track.samples if now - s.t <= cfg.speed_window_sec]
        speed, scale_rate = motion_rates(recent + [Sample(now, list(pose.bbox), pose.keypoints, "")])
        alternations = ankle_alternations(list(track.samples) + [Sample(now, pose.bbox, pose.keypoints, "")],
                                          cfg.ankle_deadband) if recent else 0
        label = frame_action(posture, speed, alternations, cfg, scale_rate)
        track.samples.append(Sample(now, list(pose.bbox), pose.keypoints, label))

        events = []
        confirmed = vote([s.label for s in track.samples], cfg.vote_threshold, cfg.min_votes)
        if confirmed and confirmed != track.action:
            events.append(({"track_id": track_id, "action": confirmed, "previous": track.action,
                            "duration_sec": 0, "previous_duration_sec": round(now - track.since, 1)
                            if track.action else None}, "warning" if confirmed == "lying" else "info",
                           round(sum(s.label == confirmed for s in track.samples) / len(track.samples), 2)))
            track.action, track.since = confirmed, now

        smoking_event = self._update_smoking(track_id, track, pose, now)
        if smoking_event:
            events.append(smoking_event)
        return events

    def _update_smoking(self, track_id: int, track: TrackActions, pose: Pose, now: float):
        cfg = self.cfg
        distances = wrist_distances(pose.keypoints, pose.bbox)
        # Hysteresis on the wrist that reached the mouth: it arrives below mouth_radius and leaves above
        # mouth_release (a missing keypoint changes nothing). An episode counts when the hand leaves within
        # mouth_episode_max_sec — a puff, not a head resting on the hands.
        if not track.near_mouth:
            near = [w for w, d in distances.items() if not math.isnan(d) and d <= cfg.mouth_radius]
            if near:
                track.near_mouth, track.mouth_wrist, track.mouth_since = True, near[0], now
        else:
            distance = distances[track.mouth_wrist]
            if not math.isnan(distance) and distance > cfg.mouth_release:
                track.near_mouth = False
                if now - track.mouth_since <= cfg.mouth_episode_max_sec:
                    track.mouth_events.append(now)
        while track.mouth_events and now - track.mouth_events[0] > cfg.smoking_window_sec:
            track.mouth_events.popleft()

        count = len(track.mouth_events)
        if count < cfg.smoking_min_count:
            track.smoking = False
            return None
        gas_confirmed = self.gas.confirmed(now)
        if track.smoking and (track.smoking_gas_confirmed or not gas_confirmed):
            return None  # already reported with the same (or stronger) evidence
        track.smoking, track.smoking_gas_confirmed = True, gas_confirmed
        confidence = min(0.95, 0.5 + 0.05 * (count - cfg.smoking_min_count) + (0.3 if gas_confirmed else 0.0))
        payload = {"track_id": track_id, "action": SMOKING, "previous": track.action, "duration_sec": 0,
                   "gas_confirmed": gas_confirmed, "wrist_to_mouth_count": count}
        return payload, "critical" if gas_confirmed else "warning", round(confidence, 2)


class ActionRecognizer(BaseModule):
    """
    Reads frames and person boxes from the vision shared buffer, runs pose only when a person is present
    and publishes ``action`` events.
    """
    name = "actions"
    source = "camera"

    def __init__(self, event_queue, control_queue, config: Dict[str, Any], pose_estimator=None,
                 frame_buffer: Optional[SharedFrameBuffer] = None):
        super().__init__(event_queue, control_queue, config)
        self.cfg = config.get("actions", {})
        self.vision_cfg = config.get("vision", {})
        self.engine = ActionEngine(ActionConfig.from_dict(self.cfg))
        self.pose_estimator = pose_estimator
        self.frame_buffer = frame_buffer
        self.interval_sec = 1.0 / float(self.cfg.get("action_fps", 5))
        self._last_frame_id = 0
        self._next_poll = 0.0

    def setup(self) -> None:
        if self.pose_estimator is None:
            self.pose_estimator = PoseEstimator(self.cfg["pose_model"], self.cfg.get("imgsz", 320),
                                                self.cfg.get("pose_conf", 0.4), self.cfg.get("keypoint_conf", 0.4))
            self.pose_estimator.load()

    def teardown(self) -> None:
        if self.frame_buffer is not None:
            self.frame_buffer.close()

    def handle_control(self, message: Any) -> None:
        """Forwarded bus events (JSON strings) update gas evidence and drop disappeared tracks."""
        if isinstance(message, str) and message.startswith("{"):
            try:
                self.engine.on_event(Event.from_json(message), time.time())
            except ValueError as e:
                logger.debug(f"Ignoring malformed forwarded event: {e}")
        else:
            super().handle_control(message)

    def step(self) -> None:
        now = time.monotonic()
        if now < self._next_poll:
            time.sleep(min(0.05, self._next_poll - now))
            return
        self._next_poll = now + self.interval_sec
        if self.frame_buffer is None and not self._attach():
            self._next_poll = now + 1.0
            return
        frame = self.frame_buffer.read_raw()
        if frame is None or frame[0] == self._last_frame_id:
            return
        self.process(*frame)

    def process(self, frame_id: int, timestamp: float, frame: np.ndarray, meta: Dict[str, Any]) -> None:
        """Runs pose for one shared frame and publishes the resulting events."""
        self._last_frame_id = frame_id
        persons = meta.get("persons", [])
        poses = self.pose_estimator.estimate(frame) if persons else []
        frame_size = (frame.shape[1], frame.shape[0])
        for payload, severity, confidence in self.engine.update(timestamp, persons, poses, frame_size):
            self.emit("action", payload, severity=severity, confidence=confidence)

    def _attach(self) -> bool:
        width, height = self.vision_cfg.get("resolution", [640, 480])
        try:
            self.frame_buffer = SharedFrameBuffer.attach(self.vision_cfg.get("shared_buffer", "iot_vision"),
                                                         width, height)
        except FileNotFoundError:
            logger.debug("Vision shared buffer not available yet")
            return False
        return True


def run(event_queue, control_queue, config):
    """
    Module entry point for the orchestrator (own process).

    :param event_queue: Bus queue to publish events to
    :param control_queue: Control messages ("stop") and forwarded bus events (gas, object_detected)
    :param config: Full merged config
    """
    ActionRecognizer(event_queue, control_queue, config).run()
