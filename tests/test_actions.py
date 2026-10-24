import math
import os
import queue
import uuid

import numpy as np
import pytest

from core.events import Event, make_event
from modules.vision.actions import (ActionConfig, ActionEngine, ActionRecognizer, classify_posture,
                                    frame_action, match_poses, vote)
from modules.vision.pose import (L_ANKLE, L_WRIST, NOSE, R_ANKLE, Pose, joint_angle, knee_angle, mask_keypoints,
                                 torso_angle, wrist_to_mouth)
from modules.vision.shared_frame import SharedFrameBuffer

CFG = ActionConfig()
FPS = 5
DT = 1 / FPS


# --- synthetic skeletons (image coordinates, y down) ------------------------------------------------------------

def _kp(points):
    kp = np.full((17, 2), np.nan)
    for index, xy in points.items():
        kp[index] = xy
    return kp


def standing(x=300.0, y=100.0, s=2.0, hand_at_mouth=False, step=0.0):
    """Front view, ~290 px tall at s=2. step shifts the ankles forward/back (walking gait)."""
    p = {0: (0, 10), 1: (-3, 8), 2: (3, 8), 3: (-6, 9), 4: (6, 9), 5: (-15, 25), 6: (15, 25), 7: (-18, 45),
         8: (18, 45), 9: (-18, 65), 10: (18, 65), 11: (-10, 70), 12: (10, 70), 13: (-10, 105), 14: (10, 105),
         15: (-10 + step, 140), 16: (10 - step, 140)}
    if hand_at_mouth:
        p[9] = (-2, 14)
    kp = _kp({i: (x + dx * s, y + dy * s) for i, (dx, dy) in p.items()})
    return Pose([int(x - 25 * s), int(y), int(x + 25 * s), int(y + 145 * s)], kp)


def sitting(x=300.0, y=150.0, s=2.0):
    """Side view on a chair: vertical torso, horizontal thigh, shin down (knee 90°)."""
    p = {0: (5, 10), 1: (4, 8), 3: (0, 9), 5: (0, 25), 6: (0, 25), 7: (5, 45), 9: (15, 55), 11: (0, 70),
         12: (0, 70), 13: (35, 72), 14: (35, 72), 15: (35, 105), 16: (35, 105)}
    kp = _kp({i: (x + dx * s, y + dy * s) for i, (dx, dy) in p.items()})
    return Pose([int(x - 10 * s), int(y), int(x + 45 * s), int(y + 107 * s)], kp)


def lying(x=200.0, y=300.0, s=2.0):
    p = {0: (-10, 0), 5: (15, -5), 6: (15, 5), 11: (60, -5), 12: (60, 5), 13: (95, -5), 14: (95, 5),
         15: (130, -5), 16: (130, 5)}
    kp = _kp({i: (x + dx * s, y + dy * s) for i, (dx, dy) in p.items()})
    return Pose([int(x - 15 * s), int(y - 15 * s), int(x + 135 * s), int(y + 15 * s)], kp)


def person(track_id, pose):
    return {"track_id": track_id, "bbox": pose.bbox}


def run_frames(engine, poses, t0=0.0, track_id=1):
    events, t = [], t0
    for pose in poses:
        events += engine.update(t, [person(track_id, pose)], [pose])
        t += DT
    return events, t


def actions_of(events):
    return [payload["action"] for payload, _, _ in events]


# --- geometry ---------------------------------------------------------------------------------------------------

def test_joint_angles():
    a, b = np.array([0.0, 0.0]), np.array([0.0, 10.0])
    assert joint_angle(a, b, np.array([10.0, 10.0])) == pytest.approx(90)
    assert joint_angle(a, b, np.array([0.0, 20.0])) == pytest.approx(180)
    assert math.isnan(joint_angle(a, b, np.array([np.nan, 1.0])))


def test_skeleton_measures():
    kp = standing().keypoints
    assert torso_angle(kp) == pytest.approx(0, abs=1)
    assert knee_angle(kp) == pytest.approx(180, abs=1)
    assert knee_angle(sitting().keypoints) == pytest.approx(90, abs=5)
    assert torso_angle(lying().keypoints) == pytest.approx(90, abs=1)
    assert wrist_to_mouth(standing(hand_at_mouth=True).keypoints, standing().bbox) < CFG.mouth_radius
    assert wrist_to_mouth(kp, standing().bbox) > 1.0


def test_mask_keypoints_drops_low_confidence_and_zero_points():
    xy = np.arange(34, dtype=float).reshape(17, 2) + 1
    xy[3] = 0
    conf = np.full(17, 0.9)
    conf[5] = 0.2
    kp = mask_keypoints(xy, conf, 0.4)
    assert np.isnan(kp[3]).all() and np.isnan(kp[5]).all() and not np.isnan(kp[0]).any()


# --- per-frame rules --------------------------------------------------------------------------------------------

def test_classify_posture():
    assert classify_posture(standing().keypoints, standing().bbox, None, CFG) == "standing"
    assert classify_posture(sitting().keypoints, sitting().bbox, None, CFG) == "sitting"
    assert classify_posture(lying().keypoints, lying().bbox, None, CFG) == "lying"


def upper_body(bottom):
    """Upright person with hips visible and legs hidden; the box ends at `bottom`."""
    pose = standing()
    pose.keypoints[13:] = np.nan
    pose.bbox = [pose.bbox[0], pose.bbox[1], pose.bbox[2], bottom]
    return pose


def test_legs_hidden_by_furniture_vs_frame_edge():
    behind_desk = upper_body(bottom=260)           # box ends inside a 640×480 frame → something hides the legs
    assert classify_posture(behind_desk.keypoints, behind_desk.bbox, None, CFG, (640, 480)) == "sitting_occluded"
    close_up = upper_body(bottom=479)              # box cut by the bottom edge → can't tell
    assert classify_posture(close_up.keypoints, close_up.bbox, None, CFG, (640, 480)) == "unknown"
    assert classify_posture(behind_desk.keypoints, behind_desk.bbox, None, CFG) == "unknown"   # no frame size


def test_sitting_occluded_becomes_sitting_unless_moving():
    assert frame_action("sitting_occluded", 0.0, 0, CFG) == "sitting"
    assert frame_action("sitting_occluded", 0.5, 0, CFG) == "walking"        # legs hidden by others in a crowd
    assert frame_action("sitting_occluded", 0.0, 0, CFG, scale_rate=0.4) == "walking"


def test_lower_than_standing_height_is_sitting():
    pose = upper_body(bottom=479)
    short = [pose.bbox[0], pose.bbox[1] + 120, pose.bbox[2], pose.bbox[3]]
    assert classify_posture(pose.keypoints, short, standing_height=400, cfg=CFG) == "sitting"   # 259 < 0.75·400


def test_sitting_on_the_floor():
    pose = sitting()
    pose.keypoints[[15, 16]] = pose.keypoints[[13, 14]] + np.array([-40.0, 50.0])   # knee ≈ 50°
    assert 40 < knee_angle(pose.keypoints) < 70
    assert classify_posture(pose.keypoints, pose.bbox, None, CFG) == "sitting"


def test_wide_box_without_torso_is_not_lying():
    kp = np.full((17, 2), np.nan)
    kp[NOSE] = (300, 200)
    assert classify_posture(kp, [0, 100, 900, 400], None, CFG) == "unknown"


def test_motion_rates_are_robust_to_jitter():
    from modules.vision.actions import Sample, motion_rates

    jitter = [Sample(i * DT, [100 + (-1) ** i * 6, 100, 200 + (-1) ** i * 6, 400 + (-1) ** i * 15], None, "")
              for i in range(6)]
    speed, scale = motion_rates(jitter)
    assert speed < CFG.walking_speed and scale < CFG.walking_scale_rate
    approach = [Sample(i * DT, [100, 100, 200, int(300 * 1.1 ** i)], None, "") for i in range(6)]
    assert motion_rates(approach)[1] >= CFG.walking_scale_rate
    assert motion_rates(approach[:2]) == (0.0, 0.0)


def test_frame_action_walking():
    assert frame_action("standing", 0.5, 0, CFG) == "walking"
    assert frame_action("unknown", 0.2, 3, CFG) == "walking"     # slow centre motion + alternating ankles
    assert frame_action("standing", 0.2, 0, CFG) == "standing"
    assert frame_action("sitting", 1.0, 5, CFG) == "sitting"


def test_vote():
    assert vote(["sitting"] * 7 + ["standing"] * 3, 0.7, 3) == "sitting"
    assert vote(["sitting"] * 6 + ["standing"] * 4, 0.7, 3) is None
    assert vote(["sitting"] * 2, 0.7, 3) is None
    assert vote(["sitting"] * 3 + ["unknown"] * 5, 0.7, 3) is None   # mostly unknown


def test_match_poses_by_iou():
    a, b = standing(x=150), standing(x=450)
    persons = [{"track_id": 7, "bbox": b.bbox}, {"track_id": 3, "bbox": a.bbox}]
    far = standing(x=1000)
    matched = dict(match_poses(persons, [a, b, far], 0.3))
    assert matched[3] is a and matched[7] is b and len(matched) == 2


# --- engine: events only on change ------------------------------------------------------------------------------

def test_engine_emits_only_on_action_change():
    engine = ActionEngine()
    events, t = run_frames(engine, [standing()] * 50)                 # 10 s standing
    assert actions_of(events) == ["standing"]
    payload, severity, confidence = events[0]
    assert payload == {"track_id": 1, "action": "standing", "previous": None, "duration_sec": 0,
                       "previous_duration_sec": None}
    assert severity == "info" and confidence == 1.0

    events, t = run_frames(engine, [sitting()] * 50, t0=t)
    assert actions_of(events) == ["sitting"]
    assert events[0][0]["previous"] == "standing"
    # Measured between confirmations, so it includes the voting delay (≤ window_sec)
    assert 10 <= events[0][0]["previous_duration_sec"] <= 10 + CFG.window_sec


def test_engine_ignores_flicker():
    engine = ActionEngine()
    _, t = run_frames(engine, [standing()] * 20)
    flicker = [sitting() if i % 4 == 0 else standing() for i in range(40)]  # 25 % sitting frames
    events, _ = run_frames(engine, flicker, t0=t)
    assert events == []


def test_engine_walking_then_standing():
    engine = ActionEngine()
    walk = [standing(x=100 + 30 * i, step=8 * (-1) ** i) for i in range(20)]   # 150 px/s ≈ 0.5 heights/s
    events, t = run_frames(engine, walk)
    assert actions_of(events) == ["walking"]
    events, _ = run_frames(engine, [standing(x=700)] * 25, t0=t)
    assert actions_of(events) == ["standing"]
    assert events[0][0]["previous"] == "walking"


def test_engine_lying_is_a_warning():
    engine = ActionEngine()
    events, _ = run_frames(engine, [lying()] * 15)
    assert actions_of(events) == ["lying"] and events[0][1] == "warning"


def test_tracks_are_independent():
    engine = ActionEngine()
    events = []
    for i in range(20):
        a, b = standing(x=150), sitting(x=450)
        events += engine.update(i * DT, [person(1, a), person(2, b)], [a, b])
    assert sorted((p["track_id"], p["action"]) for p, _, _ in events) == [(1, "standing"), (2, "sitting")]


# --- smoking ----------------------------------------------------------------------------------------------------

def smoking_frames(episodes=3, frames_per_episode=10):
    """Hand to mouth for 2 frames out of every `frames_per_episode`."""
    return [standing(hand_at_mouth=(i % frames_per_episode) < 2) for i in range(episodes * frames_per_episode)]


def test_possible_smoking_without_gas():
    engine = ActionEngine()
    events, _ = run_frames(engine, smoking_frames(3))
    smoking = [e for e in events if e[0]["action"] == "possible_smoking"]
    assert len(smoking) == 1
    payload, severity, confidence = smoking[0]
    assert "gas_confirmed" not in payload and payload["wrist_to_mouth_count"] == 3
    assert severity == "warning" and 0.5 <= confidence < 0.8


def test_resting_head_on_hands_is_not_smoking():
    engine = ActionEngine()
    events, _ = run_frames(engine, [standing(hand_at_mouth=True)] * 150)    # 30 s with the hand at the face
    assert "possible_smoking" not in actions_of(events)


def test_keypoint_dropouts_do_not_create_episodes():
    engine = ActionEngine()
    frames = []
    for i in range(60):                                                     # hand at the mouth, wrist flickers
        pose = standing(hand_at_mouth=True)
        if i % 3 == 0:
            pose.keypoints[L_WRIST] = np.nan
        frames.append(pose)
    run_frames(engine, frames)
    assert len(engine.tracks[1].mouth_events) == 0


def test_two_episodes_are_not_smoking():
    engine = ActionEngine()
    events, _ = run_frames(engine, smoking_frames(2) + [standing()] * 10)
    assert "possible_smoking" not in actions_of(events)


def test_gas_never_confirms_smoking():
    """The gas sensor measures solvent / ammonia gases, not smoke: a gas rise changes nothing about smoking"""
    engine = ActionEngine()
    engine.on_event(make_event("gas", "alert", {"reason": "gas_rising", "gas": "ethanol"}, "critical"), 0.0)
    events, t = run_frames(engine, smoking_frames(3))
    smoking = [e for e in events if e[0]["action"] == "possible_smoking"]
    assert len(smoking) == 1
    payload, severity, confidence = smoking[0]
    assert "gas_confirmed" not in payload and severity == "warning" and confidence < 0.8
    events, _ = run_frames(engine, smoking_frames(2), t0=t)            # same track → no repeat
    assert "possible_smoking" not in actions_of(events)


# --- track lifetime ---------------------------------------------------------------------------------------------

def test_disappeared_event_drops_history():
    engine = ActionEngine()
    run_frames(engine, [standing()] * 20)
    engine.on_event(make_event("camera", "object_detected", {"track_id": 1, "class": "person",
                                                             "bbox": [0, 0, 1, 1], "state": "disappeared"}), 5)
    assert 1 not in engine.tracks
    events, _ = run_frames(engine, [standing()] * 20, t0=10)
    assert actions_of(events) == ["standing"]                          # reported again for the new history


def test_unseen_track_expires():
    engine = ActionEngine()
    run_frames(engine, [standing()] * 5)
    engine.update(30.0, [], [])
    assert engine.tracks == {}


# --- module -----------------------------------------------------------------------------------------------------

class FakePoseEstimator:
    def __init__(self, poses):
        self.poses = poses
        self.calls = 0

    def estimate(self, frame):
        self.calls += 1
        return self.poses


def _drain(q):
    events = []
    while True:
        try:
            events.append(Event.from_json(q.get_nowait()))
        except queue.Empty:
            return events


def test_recognizer_runs_pose_only_with_persons_and_forwards_gas():
    pose = standing()
    estimator = FakePoseEstimator([pose])
    module = ActionRecognizer(queue.Queue(), queue.Queue(), {"actions": {}, "vision": {}}, pose_estimator=estimator)
    frame = np.zeros((480, 640, 3), np.uint8)

    module.process(1, 0.0, frame, {"persons": []})
    assert estimator.calls == 0
    for i in range(20):
        module.process(i + 2, i * DT, frame, {"persons": [person(4, pose)]})
    assert estimator.calls == 20
    (event,) = _drain(module.event_queue)
    assert (event.source, event.type, event.payload["action"], event.payload["track_id"]) == \
        ("camera", "action", "standing", 4)

    module.handle_control(make_event("gas", "alert", {"reason": "gas_threshold"}, "critical").to_json())
    assert not hasattr(module.engine, "gas")                       # gas is not an input of the action module
    module.handle_control("not json")  # ignored


def test_recognizer_reads_shared_buffer_once_per_frame():
    name = f"ta_{uuid.uuid4().hex[:10]}"
    buffer = SharedFrameBuffer.create(name, 640, 480)
    try:
        pose = standing()
        estimator = FakePoseEstimator([pose])
        config = {"actions": {"action_fps": 1000}, "vision": {"shared_buffer": name, "resolution": [640, 480]}}
        module = ActionRecognizer(queue.Queue(), queue.Queue(), config, pose_estimator=estimator)
        module.step()                                                  # attaches; nothing written yet
        frame = np.zeros((480, 640, 3), np.uint8)
        buffer.write(b"jpeg", frame, {"persons": [person(1, pose)]})
        for _ in range(5):
            module._next_poll = 0
            module.step()
        assert estimator.calls == 1                                    # same frame_id is not reprocessed
        module.teardown()
    finally:
        buffer.unlink()
        buffer.close()


def test_action_config_from_dict():
    cfg = ActionConfig.from_dict({"vote_threshold": 0.8, "sitting_knee_range": [60, 120], "unrelated": 1})
    assert cfg.vote_threshold == 0.8 and cfg.sitting_knee_range == (60, 120)


@pytest.mark.skipif(not os.environ.get("YOLO_POSE_TEST_MODEL"),
                    reason="set YOLO_POSE_TEST_MODEL=yolo11n-pose.pt (or NCNN dir)")
def test_real_pose_model_standing_people():
    import cv2
    from ultralytics.utils import ASSETS

    from modules.vision.pose import PoseEstimator

    poses = PoseEstimator(os.environ["YOLO_POSE_TEST_MODEL"], imgsz=640).estimate(cv2.imread(str(ASSETS / "bus.jpg")))
    postures = [classify_posture(p.keypoints, p.bbox, None, CFG) for p in poses]
    assert postures.count("standing") >= 3
    assert "lying" not in postures


def test_ankle_alternation_sign():
    assert standing(step=8).keypoints[L_ANKLE][0] > standing().keypoints[L_ANKLE][0]
    assert standing(step=8).keypoints[R_ANKLE][0] < standing().keypoints[R_ANKLE][0]
    assert standing(hand_at_mouth=True).keypoints[L_WRIST][1] < standing().keypoints[L_WRIST][1]
