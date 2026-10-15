import multiprocessing as mp
import os
import queue
import threading
import uuid

import cv2
import numpy as np
import pytest

from core.bus import CONTROL_STOP, create_queue, receive
from core.events import Event
from modules.vision.capture import OpenCVSource, ReconnectingSource
from modules.vision.detector import VisionDetector, annotate
from modules.vision.motion import MotionDetector, MotionResult
from modules.vision.shared_frame import SharedFrameBuffer
from modules.vision.state import Detection, MotionState, SceneState, TrackRegistry, iou

W, H = 640, 480
RNG = np.random.default_rng(0)
BACKGROUND = RNG.integers(40, 80, (H, W, 3), dtype=np.uint8)


def background():
    return cv2.add(BACKGROUND, RNG.integers(0, 4, BACKGROUND.shape, dtype=np.uint8))


def with_rect(x, y=150, w=120, h=250):
    frame = background()
    cv2.rectangle(frame, (x, y), (x + w, y + h), (220, 200, 180), -1)
    return frame


def _drain(q):
    events = []
    while True:
        try:
            events.append(Event.from_json(q.get_nowait()))
        except queue.Empty:
            return events


class FakeClock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


# --- motion -----------------------------------------------------------------------------------------------------

def test_motion_detects_moving_object_only():
    detector = MotionDetector(min_area=800, warmup_frames=20)
    for _ in range(40):
        assert not detector.detect(background()).motion

    result = None
    for i in range(10):
        result = detector.detect(with_rect(100 + 15 * i))
    assert result.motion
    assert any(iou(box, [235, 150, 355, 400]) > 0.3 for box in result.boxes)
    assert 0 < result.area_ratio < 0.5


def test_motion_ignores_small_changes():
    detector = MotionDetector(min_area=800, warmup_frames=20)
    for _ in range(40):
        detector.detect(background())
    for i in range(10):
        frame = background()
        cv2.rectangle(frame, (100 + 3 * i, 100), (110 + 3 * i, 110), (255, 255, 255), -1)  # 10×10 px
        assert not detector.detect(frame).motion


def test_motion_warmup_reports_nothing():
    detector = MotionDetector(warmup_frames=5)
    assert not any(detector.detect(with_rect(50 * i)).motion for i in range(5))


# --- state machines ---------------------------------------------------------------------------------------------

def person(track_id, bbox=(100, 100, 200, 400), conf=0.9):
    return Detection(track_id, "person", list(bbox), conf)


def test_static_object_for_ten_minutes_appears_once():
    registry = TrackRegistry(lost_timeout_sec=10)
    appeared_total, disappeared_total = 0, 0
    for t in np.arange(0, 600, 3.0):  # YOLO every 3 s for 10 minutes
        appeared, disappeared = registry.update([person(1)], t)
        appeared_total += len(appeared)
        disappeared_total += len(disappeared)
    assert (appeared_total, disappeared_total) == (1, 0)


def test_track_disappears_after_timeout():
    registry = TrackRegistry(lost_timeout_sec=10)
    registry.update([person(1)], 0)
    assert registry.update([], 9) == ([], [])
    appeared, disappeared = registry.update([], 10.5)
    assert appeared == [] and [t.track_id for t in disappeared] == [1]
    assert not registry.has_objects


def test_id_switch_on_same_object_is_merged():
    registry = TrackRegistry(lost_timeout_sec=10, reid_iou=0.5)
    registry.update([person(1)], 0)
    appeared, _ = registry.update([person(7, (105, 102, 205, 402))], 3)  # tracker re-assigned the id
    assert appeared == []
    appeared, _ = registry.update([person(7, (110, 102, 210, 402))], 6)
    assert appeared == [] and list(registry.tracks) == [1]
    assert registry.tracks[1].bbox == [110, 102, 210, 402]


def test_new_overlapping_object_while_known_one_is_matched():
    registry = TrackRegistry(lost_timeout_sec=10)
    registry.update([person(1)], 0)
    # Known id 1 is present, so the overlapping new id 2 is a second person, not an id switch
    appeared, _ = registry.update([person(2, (110, 100, 210, 400)), person(1)], 3)
    assert [t.track_id for t in appeared] == [2]


def test_untracked_detections_and_other_classes():
    registry = TrackRegistry()
    appeared, _ = registry.update([person(None), Detection(3, "dog", [0, 0, 50, 50], 0.8)], 0)
    assert [(t.track_id, t.cls) for t in appeared] == [(3, "dog")]
    # Different class at the same place is a new object
    appeared, _ = registry.update([Detection(4, "cat", [0, 0, 50, 50], 0.8)], 1)
    assert [t.cls for t in appeared] == ["cat"]


def test_motion_state():
    state = MotionState(end_delay_sec=2)
    assert state.update(True, 0) == "start"
    assert state.update(True, 0.5) is None
    assert state.update(False, 1.0) is None
    assert state.update(False, 2.4) is None
    assert state.update(False, 2.5) == "end"
    assert state.update(False, 3) is None


def test_scene_state_timeout():
    state = SceneState(timeout_sec=10)
    assert state.update(False, 0) is None
    assert state.update(True, 1) is True
    assert state.update(False, 5) is None
    assert state.update(False, 10.9) is None
    assert state.update(False, 11) is False


# --- shared memory ----------------------------------------------------------------------------------------------

@pytest.fixture
def shm_name():
    name = f"tv_{uuid.uuid4().hex[:10]}"
    yield name
    try:
        SharedFrameBuffer.attach(name, W, H).unlink()
    except FileNotFoundError:
        pass


def _read_in_child(name, out):
    buf = SharedFrameBuffer.attach(name, W, H)
    frame_id, _, raw, meta = buf.read_raw()
    out.put((frame_id, int(raw.sum()), meta, len(buf.read_annotated()[2])))
    buf.close()


def test_shared_buffer_round_trip_across_processes(shm_name):
    writer = SharedFrameBuffer.create(shm_name, W, H)
    assert writer.read_raw() is None and writer.read_annotated() is None

    frame = with_rect(100)
    jpeg = cv2.imencode(".jpg", frame)[1].tobytes()
    meta = {"persons": [{"track_id": 1, "bbox": [1, 2, 3, 4], "confidence": 0.9}]}
    assert writer.write(jpeg, frame, meta) == 1

    frame_id, _, raw, read_meta = writer.read_raw()
    assert frame_id == 1 and np.array_equal(raw, frame) and read_meta == meta
    assert writer.read_annotated()[2] == jpeg

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    child = ctx.Process(target=_read_in_child, args=(shm_name, out))
    child.start()
    child.join(timeout=30)
    assert out.get(timeout=5) == (1, int(frame.sum()), meta, len(jpeg))

    # The block survives the child exiting; a restarted writer reuses it
    writer.close()
    again = SharedFrameBuffer.create(shm_name, W, H)
    assert again.read_raw()[0] == 1
    again.close()


def test_shared_buffer_rejects_bad_input(shm_name):
    buf = SharedFrameBuffer.create(shm_name, W, H, max_jpeg=100)
    with pytest.raises(ValueError, match="JPEG"):
        buf.write(b"x" * 101, background(), {})
    with pytest.raises(ValueError, match="Raw frame"):
        buf.write(b"x", np.zeros((10, 10, 3), np.uint8), {})
    buf.close()


# --- capture ----------------------------------------------------------------------------------------------------

def test_opencv_source_reads_and_loops_video(tmp_path):
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    for i in range(5):
        writer.write(np.full((240, 320, 3), i * 40, np.uint8))
    writer.release()

    source = OpenCVSource(path, W, H, realtime=False, loop=True)
    source.open()
    frames = [source.read() for _ in range(7)]
    source.close()
    assert all(f is not None and f.shape == (H, W, 3) for f in frames)
    assert abs(int(frames[5].mean()) - int(frames[0].mean())) <= 2  # looped back to the first frame


class FlakySource:
    opened = 0

    def __init__(self, fail_open, frames):
        self.fail_open = fail_open
        self.frames = frames

    def open(self):
        FlakySource.opened += 1
        if self.fail_open:
            raise RuntimeError("no camera")

    def read(self):
        return self.frames.pop(0) if self.frames else None

    def close(self):
        pass


def test_reconnecting_source_backs_off_and_recovers():
    clock = FakeClock()
    attempts = iter([FlakySource(True, []), FlakySource(True, []), FlakySource(False, [background()])])
    source = ReconnectingSource(lambda: next(attempts), max_backoff_sec=30, clock=clock)
    FlakySource.opened = 0

    assert source.read() is None and FlakySource.opened == 1   # fails, retry in 1 s
    clock.now = 0.5
    assert source.read() is None and FlakySource.opened == 1   # still waiting
    clock.now = 1.0
    assert source.read() is None and FlakySource.opened == 2   # fails, retry in 2 s
    clock.now = 3.0
    assert source.read() is not None and source.connected


def test_reconnecting_source_reopens_after_missed_frames():
    clock = FakeClock()
    first = FlakySource(False, [background()])
    second = FlakySource(False, [background()])
    attempts = iter([first, second])
    source = ReconnectingSource(lambda: next(attempts), max_missed_frames=3, clock=clock)
    assert source.read() is not None
    assert [source.read() for _ in range(3)] == [None, None, None]
    assert not source.connected and source.last_error == "no frames"
    clock.now = 1.0
    assert source.read() is not None


# --- detector (module) ------------------------------------------------------------------------------------------

class ScriptedSource:
    def __init__(self):
        self.next_frame = None

    def read(self):
        return self.next_frame

    def close(self):
        pass


class FakeObjectDetector:
    def __init__(self):
        self.objects = []
        self.calls = 0

    def track(self, frame):
        self.calls += 1
        return list(self.objects)


VISION_CONFIG = {
    "resolution": [W, H],
    "motion_min_area": 800,
    "motion_warmup_frames": 20,
    "motion_end_sec": 2.0,
    "yolo_interval_sec": 3.0,
    "track_lost_sec": 10.0,
    "scene_timeout_sec": 10,
    "shared_fps": 10,
    "log_events": False,
    "stats_interval_sec": 30,
}


def make_detector(clock, buffer=None):
    source, objects = ScriptedSource(), FakeObjectDetector()
    det = VisionDetector(queue.Queue(), queue.Queue(), {"vision": VISION_CONFIG}, frame_source=source,
                         object_detector=objects, frame_buffer=buffer, clock=clock)
    det.setup()
    return det, source, objects


def feed(det, source, clock, frames, dt):
    for frame in frames:
        source.next_frame = frame
        det.step()
        clock.now += dt


def test_detector_scenario_publishes_changes_only(shm_name):
    clock = FakeClock(1000.0)
    buffer = SharedFrameBuffer.create(shm_name, W, H)
    det, source, objects = make_detector(clock, buffer)

    feed(det, source, clock, [background() for _ in range(30)], 1 / 15)      # warm-up, empty room
    assert _drain(det.event_queue) == []

    objects.objects = [person(1)]
    calls_before = objects.calls
    feed(det, source, clock, [with_rect(50 + 10 * i) for i in range(15)], 1 / 15)  # someone walks in
    assert objects.calls - calls_before == 15                                 # YOLO on every moving frame

    calls_before = objects.calls
    feed(det, source, clock, [background() for _ in range(600)], 1.0)         # stands still for 10 min
    assert objects.calls - calls_before <= 600 / 3 + 2                        # YOLO only every 3 s

    objects.objects = []
    events, seconds_at = [], {}
    for second in range(30):                                                  # leaves
        feed(det, source, clock, [background()], 1.0)
        for event in _drain(det.event_queue):
            seconds_at[(event.type, event.payload.get("state", event.payload.get("active")))] = second
            events.append(event)
    # scene_active goes false scene_timeout_sec (10 s) after the last frame that still had an object
    # (one 1 s frame before the "disappeared" event)
    assert seconds_at[("scene_active", False)] - seconds_at[("object_detected", "disappeared")] in (9, 10)

    summary = [(e.type, e.payload.get("state", e.payload.get("active"))) for e in events]
    assert summary == [
        ("motion", "start"),
        ("object_detected", "appeared"),
        ("scene_active", True),
        ("motion", "end"),
        ("object_detected", "disappeared"),
        ("scene_active", False),
    ]
    assert all(e.source == "camera" for e in events)
    appeared = events[1]
    assert appeared.payload == {"track_id": 1, "class": "person", "bbox": [100, 100, 200, 400], "state": "appeared"}
    assert appeared.confidence == 0.9

    frame_id, _, raw, meta = buffer.read_raw()
    assert frame_id > 0 and raw.shape == (H, W, 3)
    assert meta["persons"] == [] and meta["scene_active"] is False
    buffer.close()


def test_detector_shares_person_boxes(shm_name):
    clock = FakeClock(0.0)
    buffer = SharedFrameBuffer.create(shm_name, W, H)
    det, source, objects = make_detector(clock, buffer)
    objects.objects = [person(5), Detection(6, "backpack", [300, 300, 350, 380], 0.7)]
    feed(det, source, clock, [background()], 0.2)
    _, _, _, meta = buffer.read_raw()
    assert meta["persons"] == [{"track_id": 5, "bbox": [100, 100, 200, 400], "confidence": 0.9}]
    assert {o["class"] for o in meta["objects"]} == {"person", "backpack"}
    jpeg = buffer.read_annotated()[2]
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (H, W, 3)
    buffer.close()


def test_detector_camera_lost_and_restored():
    clock = FakeClock()
    det, source, _ = make_detector(clock)
    source.last_error = "no frames"
    feed(det, source, clock, [None, None], 0.1)                     # still connecting: no alert
    feed(det, source, clock, [background(), None, None, background()], 0.1)
    reasons = [e.payload["reason"] for e in _drain(det.event_queue) if e.type == "alert"]
    assert reasons == ["camera_lost", "camera_restored"]


def test_detector_logs_fps_stats():
    clock = FakeClock()
    det, source, _ = make_detector(clock)
    feed(det, source, clock, [background() for _ in range(31)], 1.0)
    assert det.last_stats["frame_fps"] == pytest.approx(1.0, rel=0.1)
    assert det.last_stats["yolo_fps"] == pytest.approx(1 / 3, rel=0.2)


def test_detector_run_loop_heartbeat_and_stop():
    event_queue, control_queue = create_queue(), create_queue()
    source = ScriptedSource()
    source.next_frame = background()
    det = VisionDetector(event_queue, control_queue, {"vision": VISION_CONFIG}, frame_source=source,
                         object_detector=FakeObjectDetector())
    thread = threading.Thread(target=det.run)
    thread.start()
    first = receive(event_queue, timeout=5)
    control_queue.put(CONTROL_STOP)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert (first.source, first.type) == ("system", "heartbeat")


def test_annotate_does_not_modify_input():
    frame = background()
    original = frame.copy()
    registry = TrackRegistry()
    registry.update([person(1)], 0)
    out = annotate(frame, MotionResult(True, [[10, 10, 50, 50]]), list(registry.tracks.values()))
    assert np.array_equal(frame, original) and not np.array_equal(out, frame)


@pytest.mark.skipif(not os.environ.get("YOLO_TEST_MODEL"), reason="set YOLO_TEST_MODEL=yolo11n.pt (or NCNN dir)")
def test_real_yolo_tracks_people():
    from ultralytics.utils import ASSETS

    from modules.vision.yolo import ObjectDetector

    img = cv2.resize(cv2.imread(str(ASSETS / "bus.jpg")), (W, H))
    detector = ObjectDetector(os.environ["YOLO_TEST_MODEL"], imgsz=320, conf=0.4)
    detector.track(img)
    detections = detector.track(img)
    persons = [d for d in detections if d.cls == "person"]
    assert len(persons) >= 3 and all(d.track_id is not None for d in persons)
