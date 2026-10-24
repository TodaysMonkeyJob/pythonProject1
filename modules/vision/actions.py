"""
Action recognition (F2): rules on top of pose keypoints, plus an optional learned layer.

Per frame a posture is classified (standing / sitting / lying / walking / unknown); an action is confirmed
when it wins ≥ ``vote_threshold`` of the known frames in the last ``window_sec`` and an ``action`` event is
published only when the confirmed action of a track changes. ``possible_smoking`` is a separate signal:
wrist near the mouth ≥ ``smoking_min_count`` times in ``smoking_window_sec``. Gas readings never confirm it: the gas
sensor classifies solvent / ammonia gases (ethanol, acetone, ammonia, toluol), not smoke, so a concentration rise is
unrelated to smoking (the former ``gas_confirmed`` flag raised false "critical" smoking alerts).

Learned layer (``actions.classifier_model`` / ``actions.cigarette_model``, ``modules.vision.action_classifier``): a
skeleton classifier trained on HMDB51 + NTU RGB+D 60 adds dynamic actions — falling, staggering, fighting, running,
climbing, throwing, picking_up, sitting_down, standing_up, drinking, eating, phone_call, waving, hugging, distress —
published as ``action`` events with ``source: classifier`` once ``classifier_votes`` consecutive windows agree with
p ≥ ``classifier_min_prob`` (the rules keep standing / sitting / walking / lying; a lying person only gets the riskier
falling / distress / fighting; fighting / hugging also need a second track classified as fighting / hugging).
Smoking uses three kinds of evidence: hand-to-mouth episodes (rule), the classifier's
``smoking`` class and a cigarette seen by the detector in the head / hands crop — with
``smoking_requires_cigarette`` (default, owner review 2026-10-24: the first two alone fired on everyday hand movements
almost all the time) only a seen cigarette produces ``possible_smoking``, the others are listed as supporting
evidence; a confident drinking / eating / phone_call without a visible cigarette vetoes the hand-to-mouth rule
(fewer false alarms). ``evidence`` lists which fired. The cigarette detector checks every tracked person once per
``cigarette_every_sec``; a cigarette seen in ``cigarette_min_hits`` checks within ``cigarette_window_sec`` is evidence
on its own; the classifier's smoking needs a hand-to-mouth episode or one sighting as well.

Input comes from the vision shared buffer (raw frame + person boxes with track ids). ``object_detected`` /
``disappeared`` events reach this process through its control queue — the orchestrator forwards them as event JSON
strings.
"""
import collections
import math
from dataclasses import dataclass, field, fields
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.events import Event
from modules.vision.consumer import SharedFrameConsumer
from modules.vision.pose import (L_ANKLE, L_WRIST, NOSE, R_ANKLE, R_WRIST, Pose, PoseEstimator, knee_angle,
                                 legs_visible, thigh_angle_from_horizontal, torso_angle, torso_length, visible,
                                 wrist_distances)
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
    smoking_requires_cigarette: bool = True  # only a cigarette the detector saw counts (pose / classifier alone: no)
    classifier_min_prob: float = 0.55       # learned layer: a window counts for its top class at / above this
    classifier_votes: int = 2               # … and this many consecutive windows must agree
    classifier_fight_min_prob: float = 0.9  # fighting (risk 0.9) needs a far more confident window
    classifier_every_sec: float = 1.0       # one prediction per track per second (the training windows' stride)
    smoking_classifier_prob: float = 0.5    # the classifier's "smoking" counts as smoking evidence at / above this
    cigarette_every_sec: float = 1.0        # the cigarette detector checks every tracked person this often
    cigarette_window_sec: float = 10.0      # … and a cigarette seen in ≥ cigarette_min_hits checks within this counts
    cigarette_min_hits: int = 3

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
    others: List[np.ndarray] = field(default_factory=list)   # keypoints of the other persons in the frame


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
    last_seen: float = 0.0
    predictions: Deque[Optional[str]] = field(default_factory=lambda: collections.deque(maxlen=8))
    smoking_probs: Deque[float] = field(default_factory=lambda: collections.deque(maxlen=8))
    proba: Dict[str, float] = field(default_factory=dict)
    activity: Optional[str] = None          # the last classifier action published
    classified_at: float = float("-inf")
    cigarette_at: float = float("-inf")
    cigarette_score: float = 0.0
    cigarette_seen_at: float = float("-inf")
    cigarette_hits: Deque[Tuple[float, str]] = field(default_factory=collections.deque)   # (t, "mouth" | "hand")


class ActionEngine:
    """
    Pure action logic (no I/O): feed person boxes + poses per frame, get event payloads back.
    """

    RULE_ACTIONS = ("walking",)                     # the rules decide these; the classifier's are not published
    LYING_OVERRIDES = ("falling", "distress", "fighting")
    SMOKE_VETO = ("drinking", "eating", "phone_call")
    PAIR_ACTIONS = ("fighting", "hugging")          # two-person actions: another track must take part too

    def __init__(self, cfg: Optional[ActionConfig] = None, classifier=None, cigarette=None):
        """
        :param classifier: ``action_classifier.ActionClassifier`` or None (rules only)
        :param cigarette: ``action_classifier.CigaretteDetector`` or None
        """
        self.cfg = cfg or ActionConfig()
        self.tracks: Dict[int, TrackActions] = {}
        self.classifier, self.cigarette = classifier, cigarette
        self.frame: Optional[np.ndarray] = None

    def on_event(self, event: Event, now: float) -> None:
        """``disappeared`` drops the track's history."""
        if event.type == "object_detected" and event.payload.get("state") == "disappeared":
            self.tracks.pop(event.payload.get("track_id"), None)

    def update(self, now: float, persons: Sequence[Dict[str, Any]], poses: Sequence[Pose],
               frame_size: Optional[Tuple[int, int]] = None,
               frame: Optional[np.ndarray] = None) -> List[Tuple[Dict[str, Any], str, Optional[float]]]:
        """
        :param now: Frame time, seconds
        :param persons: [{"track_id", "bbox"}] currently tracked persons
        :param poses: Pose detections of the same frame
        :param frame_size: (width, height) of the frame; lets the rules tell furniture occlusion from the frame edge
        :param frame: the BGR frame (only the cigarette detector needs it)
        :return: [(payload, severity, confidence)] for the events to publish
        """
        out = []
        self.frame = frame
        self._poses = poses
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
        others = [p.keypoints for p in getattr(self, "_poses", []) if p is not pose]
        track.samples.append(Sample(now, list(pose.bbox), pose.keypoints, label, others))

        events = []
        confirmed = vote([s.label for s in track.samples], cfg.vote_threshold, cfg.min_votes)
        if confirmed and confirmed != track.action:
            events.append(({"track_id": track_id, "action": confirmed, "previous": track.action,
                            "duration_sec": 0, "previous_duration_sec": round(now - track.since, 1)
                            if track.action else None}, "warning" if confirmed == "lying" else "info",
                           round(sum(s.label == confirmed for s in track.samples) / len(track.samples), 2)))
            track.action, track.since = confirmed, now

        learned = self._update_classifier(track_id, track, now)
        if learned:
            events.append(learned)
        smoking_event = self._update_smoking(track_id, track, pose, now)
        if smoking_event:
            events.append(smoking_event)
        return events

    def _update_classifier(self, track_id: int, track: TrackActions, now: float):
        """The learned layer: a new dynamic action once ``classifier_votes`` consecutive windows agree."""
        if self.classifier is None or now - track.classified_at < self.cfg.classifier_every_sec - 1e-6:
            return None
        window = [s for s in track.samples if now - s.t <= self.classifier.window_sec]
        proba = self.classifier.predict([(s.t, s.keypoints) for s in window], [s.others for s in window])
        if proba is None:
            return None
        track.classified_at = now
        track.proba = proba
        track.smoking_probs.append(proba.get("smoking", 0.0))
        top = max(proba, key=proba.get)
        needed = self.cfg.classifier_fight_min_prob if top == "fighting" else self.cfg.classifier_min_prob
        track.predictions.append(top if proba[top] >= needed else None)
        votes = list(track.predictions)[-self.cfg.classifier_votes:]
        if len(votes) < self.cfg.classifier_votes or len(set(votes)) != 1 or votes[0] is None:
            return None
        label = votes[0]
        if label in ("other", "smoking") or label in self.RULE_ACTIONS or label == track.activity:
            return None                                # smoking is evidence for possible_smoking (below)
        if track.action == "lying" and label not in self.LYING_OVERRIDES:
            return None
        recent = self.cfg.classifier_votes
        if label in self.PAIR_ACTIONS and not any(
                other is not track and set(list(other.predictions)[-recent:]) & set(self.PAIR_ACTIONS)
                for other in self.tracks.values()):
            return None                                # one person alone (or next to bystanders) is not a fight / hug
        previous, track.activity = track.activity, label
        severity = "warning" if label in ("falling", "fighting", "distress", "staggering") else "info"
        payload = {"track_id": track_id, "action": label, "previous": previous, "posture": track.action,
                   "source": "classifier", "probability": round(proba[label], 3), "duration_sec": 0}
        return payload, severity, round(proba[label], 2)

    def _cigarette(self, track: TrackActions, pose: Pose, now: float) -> float:
        """Runs the cigarette detector on the head / hands crop (rate-limited per track)"""
        if self.cigarette is None or self.frame is None or now - track.cigarette_at < self.cfg.cigarette_every_sec:
            return track.cigarette_score if now - track.cigarette_at < self.cfg.cigarette_every_sec else 0.0
        track.cigarette_at = now
        try:
            track.cigarette_score, centre = self.cigarette.locate(self.frame, pose.keypoints)
        except Exception as e:  # the detector must never stop the rules
            logger.error(f"Cigarette detector failed: {e}")
            track.cigarette_score, centre = 0.0, None
        place = self._cigarette_place(pose, centre) if track.cigarette_score > 0 else None
        if place is not None:                    # only at the mouth or in a hand; elsewhere (chair bars, desk) ignored
            track.cigarette_seen_at = now
            track.cigarette_hits.append((now, place))
        while track.cigarette_hits and now - track.cigarette_hits[0][0] > self.cfg.cigarette_window_sec:
            track.cigarette_hits.popleft()
        return track.cigarette_score

    def _cigarette_place(self, pose: Pose, centre: Optional[np.ndarray]) -> Optional[str]:
        """:return: "mouth" / "hand" when the detection is within mouth_radius torso lengths of the nose / a wrist"""
        if centre is None:
            return None
        scale = torso_length(pose.keypoints, pose.bbox)
        nose = pose.keypoints[NOSE]
        if visible(nose) and np.linalg.norm(centre - nose) / scale <= self.cfg.mouth_radius:
            return "mouth"
        for wrist in (L_WRIST, R_WRIST):
            point = pose.keypoints[wrist]
            if visible(point) and np.linalg.norm(centre - point) / scale <= self.cfg.mouth_radius:
                return "hand"
        return None

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
        proba = track.proba or {}
        recent = list(track.smoking_probs)[-cfg.classifier_votes:]
        classifier_smoking = len(recent) >= cfg.classifier_votes and \
            float(np.mean(recent)) >= cfg.smoking_classifier_prob     # mean of the last windows (short puffs)
        self._cigarette(track, pose, now)                       # every person, rate-limited (≈ 12 ms per crop)
        hits = len(track.cigarette_hits)
        at_mouth = any(place == "mouth" for _, place in track.cigarette_hits)
        # a cigarette seen repeatedly, at the mouth at least once — or in a hand that went to the mouth
        cigarette = hits >= cfg.cigarette_min_hits and at_mouth
        top = max(proba, key=proba.get) if proba else None
        vetoed = top in self.SMOKE_VETO and proba[top] >= cfg.classifier_min_prob and not cigarette
        # the classifier alone is not enough (a hand at the face while dozing looks alike): it needs a hand-to-mouth
        # episode or at least one cigarette sighting
        evidence = [name for name, ok in (("hand_to_mouth", count >= cfg.smoking_min_count and not vetoed),
                                          ("classifier", classifier_smoking and (count >= 1 or hits >= 1)),
                                          ("cigarette", cigarette)) if ok]
        if cfg.smoking_requires_cigarette and not cigarette:
            evidence = []                   # hand-to-mouth / the classifier alone fired on drinking, phoning, …
        if not evidence:
            track.smoking = False
            return None
        if track.smoking:
            return None  # already reported
        track.smoking = True
        confidence = min(0.95, 0.4 + 0.15 * len(evidence) + 0.05 * max(0, count - cfg.smoking_min_count))
        payload = {"track_id": track_id, "action": SMOKING, "previous": track.action, "duration_sec": 0,
                   "wrist_to_mouth_count": count, "evidence": evidence,
                   "cigarette_seen": "cigarette" in evidence,
                   **({"smoking_probability": round(proba.get("smoking", 0.0), 3)} if proba else {})}
        return payload, "warning", round(confidence, 2)


class ActionRecognizer(SharedFrameConsumer):
    """
    Reads frames and person boxes from the vision shared buffer, runs pose only when a person is present
    and publishes ``action`` events.
    """
    name = "actions"
    source = "camera"

    def __init__(self, event_queue, control_queue, config: Dict[str, Any], pose_estimator=None,
                 frame_buffer: Optional[SharedFrameBuffer] = None):
        self.cfg = config.get("actions", {})
        super().__init__(event_queue, control_queue, config, self.cfg.get("action_fps", 5), frame_buffer)
        self.classifier, self.cigarette = None, None                    # learned layer, loaded in setup()
        self.engine = ActionEngine(ActionConfig.from_dict(self.cfg))   # the first camera's (single camera: the only)
        self.engines: Dict[str, ActionEngine] = {}                     # device_id → per-camera tracks
        self.pose_estimator = pose_estimator

    def engine_for(self, device_id: str) -> ActionEngine:
        """Each camera has its own tracks (track ids are per camera); one pose model serves all of them."""
        if device_id not in self.engines:
            self.engines[device_id] = self.engine if not self.engines else ActionEngine(
                self.engine.cfg, self.classifier, self.cigarette)
        return self.engines[device_id]

    def on_camera_removed(self, device_id: str) -> None:
        self.engines.pop(device_id, None)

    def setup(self) -> None:
        super().setup()
        if self.pose_estimator is None:
            self.pose_estimator = PoseEstimator(self.cfg["pose_model"], self.cfg.get("imgsz", 320),
                                                self.cfg.get("pose_conf", 0.4), self.cfg.get("keypoint_conf", 0.4))
            self.pose_estimator.load()
            from modules.vision.yolo import limit_threads

            limit_threads(self.cfg.get("threads", 2))  # after loading: importing ultralytics resets torch's threads
        from modules.vision.action_classifier import ActionClassifier, CigaretteDetector

        self.classifier = ActionClassifier.from_path(self.cfg.get("classifier_model"))
        self.cigarette = CigaretteDetector.from_config(self.cfg)
        self.engine.classifier, self.engine.cigarette = self.classifier, self.cigarette
        if self.classifier is not None:
            logger.info(f"Action classifier: {', '.join(self.classifier.classes)}")

    def on_event(self, event: Event, now: float) -> None:
        """
        Forwarded bus events: ``object_detected`` / ``disappeared`` reach the engine of that camera.
        """
        if event.type == "object_detected":
            self.engine_for(event.device_id).on_event(event, now)

    def process(self, frame_id: int, timestamp: float, frame: np.ndarray, meta: Dict[str, Any]) -> None:
        """Runs pose for one shared frame and publishes the resulting events."""
        persons = meta.get("persons", [])
        poses = self.pose_estimator.estimate(frame) if persons else []
        frame_size = (frame.shape[1], frame.shape[0])
        engine = self.engine_for(self.current.device_id) if self.current else self.engine
        for payload, severity, confidence in engine.update(timestamp, persons, poses, frame_size, frame):
            self.emit("action", payload, severity=severity, confidence=confidence)


def run(event_queue, control_queue, config):
    """
    Module entry point for the orchestrator (own process).

    :param event_queue: Bus queue to publish events to
    :param control_queue: Control messages ("stop") and forwarded bus events (object_detected)
    :param config: Full merged config
    """
    ActionRecognizer(event_queue, control_queue, config).run()
