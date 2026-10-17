import numpy as np

from modules.vision.identity import IdentityConfig, IdentityEngine, assign_faces, face_fit
from tests.test_identity import ANNA, FRAME, OLEH, STRANGER, face_row, make_registry, near

CFG = IdentityConfig()
LEFT_BOX, RIGHT_BOX = [40, 60, 240, 460], [380, 60, 580, 460]
HUGE_BOX = [0, 0, 640, 480]                      # e.g. someone very close to the camera / a bad detection
LEFT_FACE, RIGHT_FACE = face_row(x=110, y=80, size=60), face_row(x=450, y=80, size=60)


class FacesByPosition:
    """Fake detector/recognizer: fixed faces in the frame, embedding chosen by the face's x position."""

    def __init__(self, faces):
        self.faces = faces                      # [(row, embedding)]
        self.detect_calls = 0

    def detect(self, image):
        self.detect_calls += 1
        return np.array([row for row, _ in self.faces], np.float32).reshape(-1, 15)

    def embed(self, image, face):
        return next(e for row, e in self.faces if row[0] == face[0])


def engine_with(faces):
    return IdentityEngine(FacesByPosition(faces), make_registry(), IdentityConfig(retry_sec=1.0, verify_sec=1.0))


def identities(events):
    return {p["track_id"]: p["identity"] for p, _, _ in events}


def test_assign_faces_one_to_one():
    persons = [{"track_id": 1, "bbox": LEFT_BOX}, {"track_id": 2, "bbox": RIGHT_BOX}]
    assigned = assign_faces(persons, np.array([RIGHT_FACE, LEFT_FACE]), CFG)
    assert assigned[1][0] == LEFT_FACE[0] and assigned[2][0] == RIGHT_FACE[0]


def test_face_fit_rejects_faces_of_others():
    assert face_fit(LEFT_FACE, LEFT_BOX, CFG) is not None
    assert face_fit(RIGHT_FACE, LEFT_BOX, CFG) is None                          # outside the box
    assert face_fit(face_row(x=110, y=380), LEFT_BOX, CFG) is None              # at the box's feet
    # A huge box may contain the face, but the person's own box fits it much better
    assert face_fit(LEFT_FACE, HUGE_BOX, CFG) > 2 * face_fit(LEFT_FACE, LEFT_BOX, CFG)


def test_two_people_get_their_own_identities():
    engine = engine_with([(LEFT_FACE, near(OLEH, 1)), (RIGHT_FACE, near(ANNA, 2))])
    persons = [{"track_id": 1, "bbox": LEFT_BOX}, {"track_id": 2, "bbox": RIGHT_BOX}]
    assert identities(engine.update(0.0, FRAME, persons)) == {1: "Oleh", 2: "Anna"}
    assert engine.faces.detect_calls == 1                                       # one detection for the frame


def test_huge_overlapping_box_does_not_steal_faces():
    engine = engine_with([(LEFT_FACE, near(OLEH, 1)), (RIGHT_FACE, near(ANNA, 2))])
    persons = [{"track_id": 1, "bbox": LEFT_BOX}, {"track_id": 2, "bbox": RIGHT_BOX},
               {"track_id": 3, "bbox": HUGE_BOX}]
    assert identities(engine.update(0.0, FRAME, persons)) == {1: "Oleh", 2: "Anna", 3: "not_visible"}


def test_one_face_two_overlapping_boxes():
    engine = engine_with([(LEFT_FACE, near(OLEH, 1))])
    shifted = [60, 70, 260, 470]                                                # second detection of the same person
    persons = [{"track_id": 1, "bbox": LEFT_BOX}, {"track_id": 2, "bbox": shifted}]
    result = identities(engine.update(0.0, FRAME, persons))
    assert sorted(result.values()) == ["Oleh", "not_visible"]


def test_three_people_stay_stable_over_time():
    middle_box, middle_face = [210, 60, 410, 460], face_row(x=280, y=80, size=60)
    engine = engine_with([(LEFT_FACE, near(OLEH, 1)), (middle_face, near(STRANGER, 2)), (RIGHT_FACE, near(ANNA, 3))])
    persons = [{"track_id": 1, "bbox": LEFT_BOX}, {"track_id": 2, "bbox": middle_box},
               {"track_id": 3, "bbox": RIGHT_BOX}]
    events = []
    for t in range(30):
        events += engine.update(float(t), FRAME, persons)
    assert identities(events) == {1: "Oleh", 2: "unknown_1", 3: "Anna"}
    assert len(events) == 3                                                     # no flip-flopping
