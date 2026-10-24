"""Learned action layer: window features, classifier voting in the engine, smoking evidence, the training data map."""
import numpy as np
import pytest

from modules.vision.action_features import FEATURE_NAMES, window_features
from modules.vision.actions import ActionConfig, ActionEngine
from modules.vision.pose import Pose

STAND = np.array([[100, 40], [96, 36], [104, 36], [92, 38], [108, 38], [85, 60], [115, 60], [80, 85], [120, 85],
                  [78, 110], [122, 110], [90, 120], [110, 120], [90, 160], [110, 160], [90, 200], [110, 200]], float)


def samples(kp=STAND, n=12, fps=5.0, shift=(0.0, 0.0)):
    return [(i / fps, kp + np.array(shift) * i, np.ones(17)) for i in range(n)]


def test_features_have_a_fixed_length_and_are_scale_invariant():
    a = window_features(samples())
    b = window_features([(t, kp * 3 + 50, sc) for t, kp, sc in samples()])   # same pose, bigger / shifted
    assert a.shape == (len(FEATURE_NAMES),) and np.isfinite(a).all()
    posture = slice(0, 34)
    assert np.allclose(a[posture], b[posture], atol=1e-4)
    assert window_features(samples(n=2)) is None
    hidden = np.ones(17)
    hidden[[13, 14, 15, 16]] = 0                                              # legs not visible
    c = window_features([(t, kp, hidden) for t, kp, _ in samples()])
    assert c[FEATURE_NAMES.index("ankles_visible")] == 0.0


def test_hand_at_the_mouth_and_motion_features():
    smoking = STAND.copy()
    smoking[10] = smoking[0] + [3, 6]                                          # right wrist at the nose
    f = window_features(samples(smoking))
    assert f[FEATURE_NAMES.index("mouth_share")] == 1.0
    walking = window_features(samples(shift=(8.0, 0.0)))
    assert walking[FEATURE_NAMES.index("centre_speed")] > window_features(samples())[
        FEATURE_NAMES.index("centre_speed")]


def test_interaction_features_need_another_person():
    other = STAND + [30, 0]
    alone = window_features(samples())
    close = window_features(samples(), [[other]] * 12)
    assert alone[FEATURE_NAMES.index("other_min")] == 10.0 and close[FEATURE_NAMES.index("other_min")] < 1.0


class FakeClassifier:
    window_sec, min_samples = 2.5, 3

    def __init__(self, *answers):
        self.answers = list(answers)

    def predict(self, rows, others=None):
        if len(rows) < self.min_samples:
            return None
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return dict(answer)


def run(engine, poses, start=0.0, fps=5.0, frame=None, bbox=(60, 20, 140, 210)):
    events = []
    for i, kp in enumerate(poses):
        pose = Pose(list(bbox), kp)
        events += engine.update(start + i / fps, [{"track_id": 1, "bbox": list(bbox)}], [pose], (640, 480), frame)
    return [e for e, _, _ in events]


def test_classifier_actions_need_consecutive_agreement_and_respect_lying():
    fighting = {"fighting": 0.92, "other": 0.08}
    answers = ({"other": 0.9}, {"other": 0.9}, fighting, {"other": 0.6}, fighting, fighting)
    engine = ActionEngine(ActionConfig(classifier_every_sec=0.0), FakeClassifier(*answers))
    alone = [e for e in run(engine, [STAND] * 8) if e.get("source") == "classifier"]
    assert alone == []                                                        # a fight needs two people
    engine = ActionEngine(ActionConfig(classifier_every_sec=0.0), FakeClassifier(fighting))
    other = STAND + [60, 0]
    events = []
    for i in range(8):
        persons = [{"track_id": 1, "bbox": [60, 20, 140, 210]}, {"track_id": 2, "bbox": [120, 20, 200, 210]}]
        poses = [Pose([60, 20, 140, 210], STAND), Pose([120, 20, 200, 210], other)]
        events += [e for e, _, _ in engine.update(i / 5, persons, poses, (640, 480))]
    learned = [e for e in events if e.get("source") == "classifier"]
    assert {e["action"] for e in learned} == {"fighting"} and learned[0]["probability"] == 0.92
    unsure = ActionEngine(ActionConfig(classifier_every_sec=0.0), FakeClassifier({"fighting": 0.8, "other": 0.2}))
    quiet = []
    for i in range(8):
        quiet += [e for e, _, _ in unsure.update(i / 5, persons, poses, (640, 480))]
    assert not [e for e in quiet if e.get("source") == "classifier"]           # 0.8 < classifier_fight_min_prob
    assert learned[0]["posture"] in (None, "standing")
    lying = STAND[:, ::-1] + [200, 0]                                          # the body horizontal
    box = (int(lying[:, 0].min()), int(lying[:, 1].min()), int(lying[:, 0].max()), int(lying[:, 1].max()))
    engine = ActionEngine(ActionConfig(classifier_every_sec=0.0), FakeClassifier({"drinking": 0.9}))
    events = run(engine, [lying] * 8, bbox=box)
    assert engine.tracks[1].action == "lying"
    assert not [e for e in events if e.get("source") == "classifier"]        # drinking never replaces lying
    falls = ActionEngine(ActionConfig(classifier_every_sec=0.0), FakeClassifier({"falling": 0.9}))
    assert [e["action"] for e in run(falls, [lying] * 8, bbox=box) if e.get("source") == "classifier"] == ["falling"]


def test_smoking_from_the_classifier_and_the_veto_by_drinking():
    cfg = ActionConfig(smoking_min_count=3, classifier_every_sec=0.0, smoking_requires_cigarette=False)
    smoking = ActionEngine(cfg, FakeClassifier({"smoking": 0.7, "other": 0.3}))
    assert not [e for e in run(smoking, [STAND] * 6) if e["action"] == "possible_smoking"]   # the classifier alone
    hand = STAND.copy()
    hand[10] = hand[0] + [2, 4]
    away = STAND.copy()
    with_hand = ActionEngine(cfg, FakeClassifier({"smoking": 0.7, "other": 0.3}))
    events = [e for e in run(with_hand, [STAND] * 4 + [hand, away]) if e["action"] == "possible_smoking"]
    assert events and events[0]["evidence"] == ["classifier"] and events[0]["smoking_probability"] == 0.7
    poses = [hand, away] * 8                                                   # 8 hand-to-mouth episodes
    rule_only = [e for e in run(ActionEngine(cfg), poses) if e["action"] == "possible_smoking"]
    assert rule_only and rule_only[0]["evidence"] == ["hand_to_mouth"]
    drinking = ActionEngine(cfg, FakeClassifier({"drinking": 0.9, "other": 0.1}))
    assert not [e for e in run(drinking, poses) if e["action"] == "possible_smoking"]   # vetoed
    strict = ActionConfig(smoking_min_count=3, classifier_every_sec=0.0)           # default: a cigarette is needed
    both = ActionEngine(strict, FakeClassifier({"smoking": 0.9, "other": 0.1}))
    assert not [e for e in run(both, poses) if e["action"] == "possible_smoking"]


class FakeCigarette:
    def __init__(self, score):
        self.score, self.calls = score, 0

    def detect(self, frame, keypoints):
        return self.locate(frame, keypoints)[0]

    def locate(self, frame, keypoints, where=None):
        self.calls += 1
        return self.score, (keypoints[0] + [2, 6]) if where is None else where   # at the mouth by default


def test_a_visible_cigarette_is_evidence_and_lifts_the_veto():
    cfg = ActionConfig(smoking_min_count=3, cigarette_every_sec=0.0, classifier_every_sec=0.0)
    hand = STAND.copy()
    hand[10] = hand[0] + [2, 4]
    away = STAND.copy()
    cigarette = FakeCigarette(0.8)
    engine = ActionEngine(cfg, FakeClassifier({"drinking": 0.9, "other": 0.1}), cigarette)
    frame = np.zeros((480, 640, 3), np.uint8)
    events = [e for e in run(engine, [hand, away] * 8, frame=frame) if e["action"] == "possible_smoking"]
    assert events and "cigarette" in events[0]["evidence"] and events[0]["cigarette_seen"] and cigarette.calls > 0


def test_training_label_map_covers_the_risk_map():
    from core.fuzzy import load_yaml
    from tools.train_actions import HMDB, HMDB_NAMES, NTU

    risk = load_yaml("config/fuzzy.yaml")["maps"]["action_risk"]
    learned = (set(HMDB.values()) | set(NTU.values())) - {"smoking", "walking"}
    assert learned <= set(risk), learned - set(risk)
    assert len(HMDB_NAMES) == 51 and set(HMDB) <= set(HMDB_NAMES)
    assert all(1 <= k <= 60 for k in NTU)


@pytest.mark.skipif(not __import__("os").path.isfile("models/action_classifier.joblib"),
                    reason="trained model not present (python -m tools.train_actions)")
def test_trained_classifier_loads_and_predicts():
    from modules.vision.action_classifier import ActionClassifier

    clf = ActionClassifier("models/action_classifier.joblib")
    proba = clf.predict([(t, kp) for t, kp, _ in samples()])
    assert proba and abs(sum(proba.values()) - 1) < 1e-6 and "fighting" in proba and "smoking" in proba


def test_a_cigarette_far_from_mouth_and_hands_is_ignored():
    cfg = ActionConfig(smoking_min_count=3, cigarette_every_sec=0.0, classifier_every_sec=0.0)

    class FarAway(FakeCigarette):
        def locate(self, frame, keypoints, where=None):
            self.calls += 1
            return self.score, np.array([600.0, 470.0])                       # a chair bar in the corner

    engine = ActionEngine(cfg, None, FarAway(0.9))
    frame = np.zeros((480, 640, 3), np.uint8)
    assert not [e for e in run(engine, [STAND] * 12, frame=frame) if e["action"] == "possible_smoking"]
    assert not engine.tracks[1].cigarette_hits

