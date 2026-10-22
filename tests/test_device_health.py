"""F23 (TOD-34): self-diagnostics wired into the camera / microphone / gas workers, and the dashboard trust API."""
import base64
import queue
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from core.events import Event, make_event
from modules.audio.attribution import iso
from modules.trust import TrustScorer

REPO = Path(__file__).resolve().parents[1]
HEALTH = {"trust": {"health": True, "registry_config": str(REPO / "config" / "trust_registry.yaml")}}


def drain(q):
    events = []
    while True:
        try:
            events.append(Event.from_json(q.get_nowait()))
        except queue.Empty:
            return events


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


# --- camera ---------------------------------------------------------------------------------------------------------

class Source:
    def __init__(self):
        self.frame = None

    def read(self):
        return self.frame

    def close(self):
        pass


class NoObjects:
    def track(self, frame):
        return []


def test_camera_worker_reports_a_covered_lens():
    from modules.vision.detector import VisionDetector

    clock, source = Clock(100.0), Source()
    det = VisionDetector(queue.Queue(), queue.Queue(), {"vision": {"resolution": [160, 120], "log_events": False},
                                                         **HEALTH}, frame_source=source, object_detector=NoObjects(),
                         clock=clock)
    det.setup()
    source.frame = np.full((120, 160, 3), 90, np.uint8)            # a hand over the lens: one flat colour
    for _ in range(3):
        det.step()
        clock.now += 6
    health = [e for e in drain(det.event_queue) if e.type == "device_health"]
    assert health and "lens_covered" in health[-1].payload["failing"] and health[-1].payload["kind"] == "camera"
    assert health[-1].severity == "warning"


def test_camera_worker_without_trust_health_emits_nothing_extra():
    from modules.vision.detector import VisionDetector

    clock, source = Clock(100.0), Source()
    det = VisionDetector(queue.Queue(), queue.Queue(), {"vision": {"resolution": [160, 120], "log_events": False}},
                         frame_source=source, object_detector=NoObjects(), clock=clock)
    det.setup()
    source.frame = np.full((120, 160, 3), 90, np.uint8)
    det.step()
    assert det.health is None and not [e for e in drain(det.event_queue) if e.type == "device_health"]


# --- microphone -----------------------------------------------------------------------------------------------------

class Mic:
    def __init__(self):
        self.pending = []

    def open(self):
        pass

    def read(self):
        frames, self.pending = self.pending, []
        return frames

    def close(self):
        pass


def test_microphone_worker_reports_clipping_and_vad_refutations():
    from modules.audio.mic_controller import MicController, rms_dbfs

    clock, mic = Clock(), Mic()
    cfg = {"audio": {"mic": {"idle_timeout_sec": 15, "sound_threshold_db": -35, "transcript_log": False}}, **HEALTH}
    controller = MicController(queue.Queue(), queue.Queue(), cfg, frame_source=mic,
                               vad=lambda frame: rms_dbfs(frame) > -30,
                               transcriber=lambda audio, rate: {"text": "", "language": "en"}, clock=clock)
    controller.setup()
    loud = np.clip(np.random.default_rng(0).normal(0, 25000, 480), -32768, 32767).astype(np.int16)
    for _ in range(400):                                             # 12 s of clipped audio
        mic.pending = [loud]
        controller.step()
        clock.now += 0.03
    for _ in range(40):                                              # silence ends the speech segment
        mic.pending = [np.zeros(480, np.int16)]
        controller.step()
        clock.now += 0.03
    controller.executor.shutdown(wait=True)
    events = drain(controller.event_queue)
    health = [e for e in events if e.type == "device_health"]
    assert health and "clipping" in health[0].payload["failing"]
    refutes = [e for e in events if e.type == "trust_evidence"]
    assert refutes and refutes[0].payload["channel"] == "vad" and refutes[0].payload["verdict"] == "refute"
    controller.teardown()


# --- gas ------------------------------------------------------------------------------------------------------------

class Predictor:
    voltages = [round(0.1 * i, 1) for i in range(16)]

    def predict(self, curve):
        return {"gas": "clean_air", "gas_confidence": 0.9, "concentration_pct": 0.0}


def test_gas_worker_self_test_and_flat_readings():
    from modules.gas.data import CLEAN_AIR, load_dataset
    from modules.gas.worker import GasWorker

    dataset = load_dataset(str(REPO / "data" / "gas" / "data.xls"))
    clean = list(dataset.X[dataset.y_gas == CLEAN_AIR].mean(axis=0))
    drifted = [v * 1.8 for v in clean]
    curves = iter([clean] + [drifted] * 40)
    clock = Clock()
    worker = GasWorker(queue.Queue(), queue.Queue(), {"gas": {"sensor_id": "gas-x", "thresholds": {}}, **HEALTH},
                       predictor=Predictor(), reader=lambda: next(curves), clock=clock)
    worker.setup()
    worker.health_cfg = {**worker.health_cfg, "selftest_interval_sec": 10}
    for _ in range(35):
        worker.poll_once()
        clock.now += 1
    health = [e.payload for e in drain(worker.event_queue) if e.type == "device_health"]
    assert health[0]["baseline_ok"] is True                          # the first reading is the training baseline
    assert any(h.get("baseline_ok") is False and "baseline_drift" in h["failing"] for h in health)
    assert "flat_readings" in health[-1]["failing"]


# --- dashboard ------------------------------------------------------------------------------------------------------

def test_dashboard_trust_api_reads_the_registry_and_publishes_feedback(tmp_path):
    from dashboard.server import DashboardState, create_app

    db = str(tmp_path / "trust.db")
    scorer = TrustScorer.from_config(str(REPO / "config" / "fuzzy.yaml"), db)
    online = make_event("system", "device_online", {"device_id": "cam-1", "kind": "camera", "location_id": "lab"},
                        device_id="cam-1", location_id="lab").model_copy(update={"timestamp": iso(1.7e9)})
    scorer.apply(online)
    scorer.registry.close()
    cfg = {"dashboard": {"user": "a", "password": "b"}, "logging": {"dir": str(tmp_path / "logs")},
           "trust": {"registry_db": db}, "network": {"devices_path": str(tmp_path / "devices.db"), "node_port": 0},
           "identity": {}, "locations": {}}
    state = DashboardState(cfg)
    published = []
    state.publish = published.append
    client = TestClient(create_app(state, cfg))
    auth = {"Authorization": "Basic " + base64.b64encode(b"a:b").decode()}
    channels = client.get("/api/trust", headers=auth).json()["channels"]
    assert {c["channel_kind"] for c in channels} == {"objects", "motion", "pose_action", "smoking", "face_identity",
                                                     "scene_text"}
    history = client.get("/api/trust/history", params={"channel_id": "cam-1/objects"}, headers=auth).json()
    assert history["history"][0]["cause_type"] == "created"
    r = client.post("/api/trust/feedback", json={"channel_id": "cam-1/pose_action", "verdict": "refute"}, headers=auth)
    assert r.status_code == 200 and published[0].type == "trust_feedback"
    assert published[0].payload["channel_id"] == "cam-1/pose_action" and published[0].payload["by"] == "a"
    assert client.post("/api/trust/feedback", json={"channel_id": "x"}, headers=auth).status_code == 400
    assert client.get("/api/trust").status_code == 401
    missing = {**cfg, "trust": {"registry_db": str(tmp_path / "none.db")}}
    assert TestClient(create_app(DashboardState(missing), missing)).get("/api/trust", headers=auth).status_code == 503
