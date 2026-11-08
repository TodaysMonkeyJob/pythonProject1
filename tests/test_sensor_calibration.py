"""C1 (TOD-54): calibration is done once per device, kept across restarts, checked briefly on start."""
import json
import os
import time

import numpy as np
import pytest

from core.sensor_calibration import (CALIBRATING, CHECKING, FACTORY, NEEDS_CHECK, READY, WARMING_UP,
                                     CalibrationStore, DeviceCalibration, device_uid)

POLICY = {"max_age_days": 30, "max_failures": 3, "recheck_sec": 60, "warmup_uncertainty_factor": 2.0,
          "needs_check_uncertainty_factor": 2.0}


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def device(store, uid="mic-abc", kind="mic", policy=None, clock=None, wall=None):
    events = []
    calibration = DeviceCalibration(store, uid, kind, "mic-1", {**POLICY, **(policy or {})}, events.append,
                                    clock or Clock(), wall or time.time)
    return calibration, events


# --- store ------------------------------------------------------------------------------------------------------------

def test_store_keeps_versions_restores_and_survives_a_restart(tmp_path):
    store = CalibrationStore(str(tmp_path))
    store.save("mic-abc", "mic", "mic-1", {"background_db": -60.0}, "first")
    store.save("mic-abc", "mic", "mic-1", {"background_db": -55.0}, "second")
    again = CalibrationStore(str(tmp_path))                       # a new process: the file is the state
    record = again.load("mic-abc")
    assert record["values"] == {"background_db": -55.0} and [h["method"] for h in record["history"]] == ["first"]
    restored = again.restore("mic-abc", 0, by="admin")
    assert restored["values"] == {"background_db": -60.0} and restored["restored_by"] == "admin"
    assert [h["method"] for h in again.load("mic-abc")["history"]] == ["second"]
    with pytest.raises(KeyError):
        again.restore("mic-abc", 5)
    assert not [n for n in os.listdir(tmp_path) if n.endswith(".tmp")]           # atomic writes
    assert again.records()[0]["uid"] == "mic-abc" and again.records()[0]["history"][0]["method"] == "second"


def test_corrupt_file_recalibrates_only_that_device(tmp_path):
    store = CalibrationStore(str(tmp_path))
    store.save("good", "mic", "mic-1", {"background_db": -60.0}, "ok")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    bad, _ = device(store, "bad")
    good, _ = device(store, "good")
    assert bad.start() == CALIBRATING and "пошкоджений" in bad.reason
    assert good.start() == CHECKING
    assert not (tmp_path / "bad.json").exists() and any(n.startswith("bad.json.corrupt-") for n in os.listdir(tmp_path))


def test_uid_follows_driver_serial_bus_and_firmware():
    a = device_uid("graphene_fet", "SN1", "/dev/ttyUSB0", "1.0")
    assert a == device_uid("graphene_fet", "SN1", "/dev/ttyUSB0", "1.0") and a.startswith("graphene_fet-")
    assert a != device_uid("graphene_fet", "SN1", "/dev/ttyUSB0", "1.1")            # firmware update → new uid
    assert a != device_uid("graphene_fet", "SN2", "/dev/ttyUSB0", "1.0")            # sensor replaced


# --- state machine --------------------------------------------------------------------------------------------------

def test_new_device_calibrates_once_then_a_restart_only_checks():
    store, clock = CalibrationStore(None), Clock()
    first, events = device(store, clock=clock)
    assert first.start() == CALIBRATING and first.source == "new" and events[-1]["status"] == CALIBRATING
    first.complete({"background_db": -60.0}, "60 s background")
    assert first.state == READY and first.text().startswith("відкалібровано (нова калібровка")
    restart, events = device(store, clock=clock)                  # same uid after a one-minute pause
    clock.t += 60
    assert restart.start() == CHECKING and restart.source == "stored" and restart.values["background_db"] == -60.0
    assert restart.uncertainty_factor() == 2.0                    # not yet checked
    clock.t += 3
    restart.check_result(True, "фон -59 дБ")
    assert restart.state == READY and restart.uncertainty_factor() == 1.0
    assert restart.status()["ready_after_sec"] == 3.0 and "збережено" in restart.text()
    assert not any(e["status"] == CALIBRATING for e in events)    # no full recalibration


def test_too_old_calibration_is_redone_but_used_meanwhile():
    store = CalibrationStore(None)
    store.save("mic-abc", "mic", "mic-1", {"background_db": -60.0}, "old")
    store.memory["mic-abc"]["date"] = "2026-01-01T00:00:00+00:00"
    calibration, _ = device(store, wall=lambda: 1_800_000_000.0)    # 2027: far beyond 30 days
    assert calibration.start() == CALIBRATING and "застаріла" in calibration.reason
    assert calibration.values == {"background_db": -60.0}


def test_failed_check_needs_check_then_recalibrates_after_repeated_failures():
    store, clock = CalibrationStore(None), Clock()
    store.save("mic-abc", "mic", "mic-1", {"background_db": -60.0}, "stored")
    calibration, events = device(store, clock=clock)
    calibration.start()
    assert calibration.check_result(False, "фон -40 дБ") == NEEDS_CHECK          # not silent recalibration
    assert calibration.text() == "потребує перевірки" and calibration.uncertainty_factor() == 2.0
    assert calibration.tick() == NEEDS_CHECK                                       # recheck only after recheck_sec
    clock.t += 61
    assert calibration.tick() == CHECKING
    assert calibration.check_result(False, "фон -40 дБ") == NEEDS_CHECK
    clock.t += 61
    calibration.tick()
    assert calibration.check_result(False, "фон -40 дБ") == CALIBRATING and "3 рази" in calibration.reason
    assert store.load("mic-abc")["validation"]["last_check"]["ok"] is False


def test_warm_up_uses_stored_values_and_factory_kinds_need_nothing():
    store, clock = CalibrationStore(None), Clock()
    store.save("mos-1", "mos", "gas-mq", {"r0_ohm": 9000.0}, "stored")
    mos, _ = device(store, "mos-1", "mos", {"warmup_sec": 180}, clock)
    assert mos.start() == WARMING_UP and mos.text() == "прогрів, 3 хв" and mos.values == {"r0_ohm": 9000.0}
    assert mos.uncertainty_factor() == 2.0
    clock.t += 181
    assert mos.tick() == CHECKING
    factory, _ = device(store, "ec-1", "electrochemical", {"factory": True, "warmup_sec": 60}, clock)
    assert factory.start() == WARMING_UP and factory.source == "factory"
    clock.t += 61
    assert factory.tick() == FACTORY and factory.usable and factory.text() == "заводська калібровка"


def test_manual_request_cancel_and_reload():
    store = CalibrationStore(None)
    store.save("mic-abc", "mic", "mic-1", {"background_db": -60.0}, "v1")
    store.save("mic-abc", "mic", "mic-1", {"background_db": -50.0}, "v2")
    calibration, _ = device(store)
    calibration.start()
    calibration.check_result(True)
    assert calibration.request("admin") == CALIBRATING and "ручне" in calibration.reason
    assert calibration.cancel() == READY and calibration.values == {"background_db": -50.0}
    store.restore("mic-abc", 0)                                    # dashboard «Відновити»
    assert calibration.reload() == CHECKING and calibration.values == {"background_db": -60.0}


# --- calibrators --------------------------------------------------------------------------------------------------

CONFIG = {"sensor_calibration": {"kinds": {"mic": {"check_sec": 3, "tolerance_db": 6, "calibrate_sec": 60},
                                           "camera": {"check_frames": 5, "tolerance": 20, "calibrate_sec": 10},
                                           "graphene_fet": {"calibrate_samples": 5, "tolerance": 0.25}}}}


def test_microphone_restart_skips_the_warm_up():
    from modules.audio.intensity import BackgroundLevel
    from modules.calibration import MicCalibrator

    store, clock, events = CalibrationStore(None), Clock(0.0), []
    first = MicCalibrator(CONFIG, "mic-uid", "mic-1", events.append, clock, store, background=BackgroundLevel())
    first.start()
    for i in range(int(61 / 0.03)):                               # 61 s of quiet room at −60 dBFS
        clock.t = i * 0.03
        first.observe(-60.0 + (i % 3) * 0.5, False, clock.t)
    assert first.state.state == READY and store.load("mic-uid")["values"]["background_db"] == pytest.approx(-60.0)
    background = BackgroundLevel()
    restart = MicCalibrator(CONFIG, "mic-uid", "mic-1", events.append, Clock(0.0), store, background=background)
    restart.start()
    assert background.background == pytest.approx(-60.0)          # at once — no 10 s warm-up
    assert background.relative(-40.0) == pytest.approx(20.0)
    for i in range(int(3.2 / 0.03)):
        restart.clock.t = i * 0.03
        restart.observe(-59.0, False, restart.clock.t)
    assert restart.state.state == READY and restart.state.last_check["ok"]
    loud = MicCalibrator(CONFIG, "mic-uid", "mic-1", events.append, Clock(0.0), store, background=BackgroundLevel())
    loud.start()
    for i in range(int(3.2 / 0.03)):                              # a fan now runs: 20 dB louder
        loud.clock.t = i * 0.03
        loud.observe(-40.0, False, loud.clock.t)
    assert loud.state.state == NEEDS_CHECK


def frames(seed=0, shift=0, n=1):
    rng = np.random.default_rng(seed)
    base = np.zeros((120, 160, 3), np.uint8)
    base[:, :80] = 90
    base[:, 80:] = 170
    base = np.roll(base, shift, axis=1)
    return [np.clip(base + rng.normal(0, 2, base.shape), 0, 255).astype(np.uint8) for _ in range(n)]


def test_camera_restart_restores_the_background_and_detects_a_moved_camera():
    from modules.calibration import CameraCalibrator
    from modules.vision.motion import MotionDetector

    store, clock = CalibrationStore(None), Clock(0.0)
    motion = MotionDetector(min_area=50, fps=10)
    first = CameraCalibrator(CONFIG, "cam-uid", "cam-1", lambda p: None, clock, store, motion=motion)
    first.start()
    for i, frame in enumerate(frames(n=120)):                     # 12 s still scene at 10 fps
        clock.t = i / 10
        result = motion.detect(frame)
        first.observe(frame, result.motion, clock.t)
    record = store.load("cam-uid")
    assert first.state.state == READY and record["values"]["background_png"] and record["values"]["grid"] == 8
    fresh = MotionDetector(min_area=50, fps=10)
    restart = CameraCalibrator(CONFIG, "cam-uid", "cam-1", lambda p: None, Clock(0.0), store, motion=fresh)
    restart.start()
    assert fresh.frames_seen > fresh.warmup_frames and fresh._noise is not None     # no warm-up from scratch
    for frame in frames(seed=1, n=5):
        restart.observe(frame, fresh.detect(frame).motion, 0.0)
    assert restart.state.state == READY
    moved = CameraCalibrator(CONFIG, "cam-uid", "cam-1", lambda p: None, Clock(0.0), store,
                             motion=MotionDetector(min_area=50, fps=10))
    moved.start()
    for frame in frames(seed=2, shift=60, n=5):                   # the camera was turned
        moved.observe(frame, False, 0.0)
    assert moved.state.state == NEEDS_CHECK and "сірого" in moved.state.last_check["detail"]


def curve(peak=2.0, at=6, noise=0.0):
    x = np.arange(16)
    return list(1.0 + (peak - 1.0) * np.exp(-((x - at) ** 2) / 8.0) + noise)


def test_graphene_baseline_is_checked_on_the_first_clean_air_sweep():
    from modules.calibration import GrapheneCalibrator

    store = CalibrationStore(None)
    voltages = [round(0.1 * i, 1) for i in range(16)]
    first = GrapheneCalibrator(CONFIG, "gfet", "gas-1", lambda p: None, Clock(), store, voltages=voltages)
    assert first.start() == CALIBRATING
    first.observe(curve(5.0), "acetone")                           # not clean air: waited out
    for _ in range(5):
        first.observe(curve(), "clean_air")
    values = store.load("gfet")["values"]
    assert first.state.state == READY and values["dirac_v"] == 0.6 and values["peak"] == pytest.approx(2.0)
    restart = GrapheneCalibrator(CONFIG, "gfet", "gas-1", lambda p: None, Clock(), store, voltages=voltages)
    assert restart.start() == CHECKING
    restart.observe(curve(5.0), "acetone")                         # gas present: the check waits
    assert restart.state.state == CHECKING
    restart.observe(curve(2.1), "clean_air")
    assert restart.state.state == READY
    drifted = GrapheneCalibrator(CONFIG, "gfet", "gas-1", lambda p: None, Clock(), store, voltages=voltages)
    drifted.start()
    drifted.observe(curve(3.0, at=10), "clean_air")
    assert drifted.state.state == NEEDS_CHECK and "Дірака" in drifted.state.last_check["detail"]


def test_mos_r0_is_stored_and_replaces_the_declared_one():
    from modules.calibration import MosCalibrator

    class Driver:
        options = {"r0_ohm": 10_000.0, "rl_ohm": 10_000.0, "vc": 5.0, "clean_air_ratio": 1.0}

    store, clock = CalibrationStore(None), Clock()
    policy = {"sensor_calibration": {"kinds": {"mos": {"calibrate_samples": 4, "warmup_sec": 0}}}}
    first = MosCalibrator(policy, "mq", "gas-mq", lambda p: None, clock, store, driver=Driver())
    assert first.start() == CALIBRATING
    for _ in range(4):
        first.observe(2.0)                                         # R_s = 10 kΩ · (5 − 2) / 2 = 15 kΩ
    assert store.load("mq")["values"]["r0_ohm"] == 15_000.0 and first.driver.options["r0_ohm"] == 15_000.0
    driver = Driver()
    driver.options = dict(Driver.options)
    restart = MosCalibrator(policy, "mq", "gas-mq", lambda p: None, clock, store, driver=driver)
    restart.start()
    assert driver.options["r0_ohm"] == 15_000.0
    restart.observe(2.0)
    assert restart.state.state == READY


def test_control_events_reach_only_their_device():
    from modules.calibration import MicCalibrator

    store = CalibrationStore(None)
    store.save("mic-uid", "mic", "mic-1", {"background_db": -60.0}, "stored")
    calibrator = MicCalibrator(CONFIG, "mic-uid", "mic-1", lambda p: None, Clock(), store)
    calibrator.start()
    assert not calibrator.control({"action": "recalibrate", "device_ids": ["mic-2"]})
    assert calibrator.control({"action": "recalibrate", "device_ids": ["mic-1", "cam-1"], "by": "admin"})
    assert calibrator.calibrating
    assert calibrator.control({"action": "cancel", "device_id": "mic-1"}) and not calibrator.calibrating


# --- system: HFS uncertainty, dashboard ------------------------------------------------------------------------------

def test_hfs_widens_the_corridor_of_an_unchecked_device():
    import yaml

    from core.events import make_event
    from modules.hfs.worker import HierarchicalFuzzySystem

    config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    reading = {"gas": "ammonia", "concentration": 120, "unit": "ppm"}

    def gas_result(factor):
        hfs = HierarchicalFuzzySystem(config, clock=lambda: 1000.0)
        if factor:
            hfs.observe(make_event("gas", "calibration_status", {"device_id": "gas-x", "uncertainty_factor": factor},
                                   device_id="gas-x", location_id="lab"))
        hfs.observe(make_event("gas", "gas_reading", reading, device_id="gas-x", location_id="lab"))
        return hfs.tick()[0]["subsystems"]["gas"]

    calm, wide = gas_result(None), gas_result(2.0)
    assert wide["calibration_factor"] == 2.0 and "calibration_factor" not in calm
    assert wide["deltas"]["max_norm"] == pytest.approx(2 * calm["deltas"]["max_norm"])
    assert wide["corridor_width"] >= calm["corridor_width"]


def test_dashboard_calibration_api(tmp_path):
    from fastapi.testclient import TestClient

    from dashboard.server import DashboardState, create_app

    store = CalibrationStore(str(tmp_path / "cal"))
    store.save("mic-uid", "mic", "mic-1", {"background_db": -60.0}, "v1")
    store.save("mic-uid", "mic", "mic-1", {"background_db": -50.0}, "v2")
    config = {"sensor_calibration": {"dir": str(tmp_path / "cal")}, "dashboard": {}, "logging": {"dir": str(tmp_path)}}
    state = DashboardState(config)
    published = []
    state.publish = published.append
    status = DeviceCalibration(store, "mic-uid", "mic", "mic-1", POLICY).status()
    state.on_event(json.dumps({**json.loads(make_status_event(status)), "location_id": "lab"}))
    http = TestClient(create_app(state, config))
    listing = http.get("/api/calibration").json()
    assert listing["devices"][0]["device_id"] == "mic-1" and listing["records"][0]["history"][0]["method"] == "v1"
    assert http.post("/api/calibration/recalibrate", json={"device_id": "mic-1"}).status_code == 400   # confirm
    ok = http.post("/api/calibration/recalibrate", json={"location_id": "lab", "confirm": True}).json()
    assert ok["device_ids"] == ["mic-1"] and published[-1].type == "calibration_control"
    assert published[-1].payload["action"] == "recalibrate"
    restored = http.post("/api/calibration/restore", json={"uid": "mic-uid", "index": 0, "confirm": True}).json()
    assert restored["action"] == "reload" and store.load("mic-uid")["values"] == {"background_db": -60.0}
    assert http.post("/api/calibration/cancel", json={"device_id": "mic-1"}).json()["action"] == "cancel"


def make_status_event(status):
    from core.events import make_event

    return make_event("audio", "calibration_status", status, device_id="mic-1", location_id="lab").to_json()


def test_drivers_poller_calibrates_each_identified_sensor(tmp_path):
    from modules.gas.drivers_mode import DriverPoller
    from tests.test_gas_drivers import FakePredictor, descriptors

    sent, clock = [], Clock(0.0)
    cfg = {"sensors_state": str(tmp_path / "gas_sensors.json"), "generated_model": str(tmp_path / "gen.yaml"),
           "gases": "config/gases.yaml"}
    config = {"sensor_calibration": {"kinds": {"electrochemical": {"factory": True, "warmup_sec": 60},
                                               "particle": {"factory": True, "warmup_sec": 0},
                                               "graphene_fet": {"calibrate_samples": 2}}}}
    store = CalibrationStore(None)
    poller = DriverPoller(cfg, lambda type_, payload, **kw: sent.append((type_, payload, kw)), clock=clock,
                          discovery_factory=lambda: descriptors()[0], config=config, store=store)
    poller.discover()
    for driver in poller.drivers:
        if driver.descriptor.needs_model:
            driver.predictor = FakePredictor()
    by_driver = {s["uid"].split("-")[0]: s for s in poller.statuses()}
    assert by_driver["dfrobot_gravity"]["status"] == WARMING_UP and by_driver["dfrobot_gravity"]["source"] == "factory"
    assert by_driver["sds011"]["status"] == FACTORY
    assert by_driver["graphene_fet"]["status"] == CALIBRATING                    # a new board: its own baseline
    statuses = [p for t, p, _ in sent if t == "calibration_status"]
    assert {p["device_id"] for p in statuses} == {d.descriptor.device_id for d in poller.drivers}   # one each
    clock.t = 61
    poller.step()
    assert {s["uid"].split("-")[0]: s for s in poller.statuses()}["dfrobot_gravity"]["status"] == FACTORY
    poller.handle_event(type("E", (), {"type": "calibration_control", "payload": {
        "action": "recalibrate", "device_ids": [by_driver["graphene_fet"]["device_id"]]}})())
    assert {s["uid"].split("-")[0]: s for s in poller.statuses()}["graphene_fet"]["status"] == CALIBRATING


def test_restart_report_needs_no_full_calibration(tmp_path):
    from core.config import load_config
    from tools.calibration_restart import measure

    result = measure(load_config("config.yaml"))
    for kind, row in result["rows"].items():
        assert row["restart"]["full_calibration"] is False, kind
        assert row["restart"]["ready_after_sec"] <= row["first_start"]["ready_after_sec"], kind
    assert result["rows"]["mic"]["restart"]["ready_after_sec"] < 10                # was a 10 s warm-up + 5 min window
    assert result["rows"]["mos"]["first_start"]["ready_after_sec"] > 180            # calibrated after the heater
