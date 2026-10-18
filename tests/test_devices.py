"""F11 device network: registry, discovery, per-device pipelines, fair scheduling, per-location behaviour."""
import json
import queue
import uuid

import numpy as np
import pytest

from core.device_manager import DeviceManager, buffer_name
from core.devices import DeviceRegistry, make_device_id, slug
from core.discovery import (FoundDevice, diff, linux_cameras, parse_input_devices, parse_system_profiler)
from core.events import Event, make_event
from modules.vision.scheduler import CameraSlot, FairScheduler

FACETIME = FoundDevice("camera", "local", "FaceTime HD Camera", "avfoundation:3F45", {"source": 0})
IPHONE = FoundDevice("camera", "continuity", "Minnor Camera", "avfoundation:0286", {"source": 1})
MAC_MIC = FoundDevice("mic", "local", "MacBook Pro Microphone", "mic:MacBook Pro Microphone",
                      {"device": "MacBook Pro Microphone"})


@pytest.fixture
def registry(tmp_path):
    bus = []
    reg = DeviceRegistry(str(tmp_path / "devices.db"), "hub", publish=bus.append)
    reg.bus = bus
    yield reg
    reg.close()


# --- registry -----------------------------------------------------------------------------------------------------

def test_new_devices_are_unassigned_and_keep_their_id(registry, tmp_path):
    device, changed = registry.seen("camera", "local", "USB cam", "v4l2:usb-1", {"source": "/dev/video0"})
    assert changed and device["location_id"] == "unassigned" and device["enabled"] and device["status"] == "online"
    assert device["device_id"] == make_device_id("camera", "hub", "v4l2:usb-1")
    assert registry.bus[-1].type == "device_online" and registry.bus[-1].device_id == device["device_id"]
    assert registry.seen("camera", "local", "USB cam", "v4l2:usb-1", {"source": "/dev/video0"})[1] is False
    registry.update(device["device_id"], location_id="room_b")
    assert registry.bus[-1].type == "device_updated" and registry.bus[-1].location_id == "room_b"
    registry.offline(device["device_id"])
    assert registry.bus[-1].type == "device_offline"
    again = DeviceRegistry(str(tmp_path / "devices.db"), "hub")                   # restart / re-plug
    device, changed = again.seen("camera", "local", "USB cam", "v4l2:usb-1", {"source": "/dev/video2"})
    assert changed and device["location_id"] == "room_b"                          # location remembered
    assert {loc["location_id"] for loc in again.list_locations()} == {"unassigned", "room_b"}
    again.close()


def test_locations_and_validation(registry):
    assert slug(" Кімната B ") == "кімната_b" and slug("Room B") == "room_b"
    assert registry.add_location("Room B") == {"location_id": "room_b", "name": "Room B"}
    with pytest.raises(KeyError):
        registry.update("camera-nope", location_id="x")
    with pytest.raises(ValueError):
        registry.seen("toaster", "local", "x", "y")
    device, _ = registry.seen("gas", "local", "Gas", "gas:gas-1", location_id="kitchen", enabled=False)
    assert (device["location_id"], device["enabled"]) == ("kitchen", False)


# --- discovery ----------------------------------------------------------------------------------------------------

def test_parse_system_profiler_with_continuity_iphone():
    output = "WARNING: AVCaptureDeviceTypeExternal is deprecated\n" + json.dumps({"SPCameraDataType": [
        {"_name": "FaceTime HD Camera", "spcamera_model-id": "FaceTime HD Camera", "spcamera_unique-id": "3F45"},
        {"_name": "Minnor Camera", "spcamera_model-id": "iPhone18,1", "spcamera_unique-id": "0286"}]})
    cams = parse_system_profiler(output)
    assert [(c.name, c.transport, c.params["source"]) for c in cams] == [("FaceTime HD Camera", "local", 0),
                                                                         ("Minnor Camera", "continuity", 1)]
    assert parse_system_profiler("") == []


def test_linux_cameras_skip_metadata_and_codec_nodes(tmp_path):
    for node, name, index in (("video0", "USB Camera: HD", "0"), ("video1", "USB Camera: HD", "1"),
                              ("video10", "bcm2835-codec-decode", "0")):
        (tmp_path / node).mkdir()
        (tmp_path / node / "name").write_text(name)
        (tmp_path / node / "index").write_text(index)
    cams = linux_cameras(str(tmp_path), picamera_info=lambda: [{"Model": "imx708", "Id": "/base/i2c@88000/imx708"}])
    assert [(c.name, c.params["source"], c.transport) for c in cams] == [
        ("USB Camera: HD", "/dev/video0", "local"), ("Pi camera imx708", "picamera2:0", "picamera")]


def test_parse_microphones_ignores_virtual_and_duplicates():
    devices = [{"name": "MacBook Pro Microphone", "max_input_channels": 1},
               {"name": "MacBook Pro Speakers", "max_input_channels": 0},
               {"name": "Steam Streaming Microphone", "max_input_channels": 2},
               {"name": "Minnor Microphone (iPhone)", "max_input_channels": 1},
               {"name": "MacBook Pro Microphone", "max_input_channels": 1}]
    mics = parse_input_devices(devices)
    assert [(m.name, m.transport) for m in mics] == [("MacBook Pro Microphone", "local"),
                                                     ("Minnor Microphone (iPhone)", "continuity")]


def test_diff_added_removed_changed():
    moved = FoundDevice("camera", "local", "FaceTime HD Camera", "avfoundation:3F45", {"source": 1})
    change = diff([FACETIME, IPHONE, MAC_MIC], [moved, MAC_MIC])
    assert change.removed == [IPHONE] and change.changed == [moved] and change.added == []
    assert not diff([FACETIME], [FACETIME])


# --- device manager -----------------------------------------------------------------------------------------------

class FakeOrchestrator:
    def __init__(self):
        self.modules, self.buffers, self.events = {}, {}, []

    def add_module(self, spec, buffer=None):
        self.modules[spec.name] = spec
        self.buffers[spec.name] = buffer

    def remove_module(self, name):
        self.modules.pop(name)

    def emit(self, event):
        self.events.append(event)


def manager(tmp_path, scans, **network):
    config = {"vision": {"enabled": True, "model": "yolo11n.pt"}, "audio": {"enabled": True},
              "network": {"devices_path": str(tmp_path / "devices.db"), "node_id": "hub", **network}}
    fake = FakeOrchestrator()
    scans = iter(scans)
    return DeviceManager(fake, config, scan=lambda cfg: next(scans)), fake


def test_pipelines_follow_plug_and_unplug(tmp_path):
    mgr, orch = manager(tmp_path, [[FACETIME, IPHONE, MAC_MIC]])
    mgr.start()
    names = sorted(orch.modules)
    assert len(names) == 3 and sum(n.startswith("vision@camera-") for n in names) == 2 and \
        sum(n.startswith("audio@mic-") for n in names) == 1
    facetime = mgr.devices[FACETIME.hw_key]
    spec = orch.modules[f"vision@{facetime}"]
    assert spec.config["vision"] == {"device": 0, "camera": "opencv", "shared_buffer": buffer_name(facetime)}
    assert spec.config["_device"]["location_id"] == "unassigned" and spec.device_id == facetime
    assert orch.buffers[f"vision@{facetime}"] == buffer_name(facetime) and len(buffer_name(facetime)) <= 30
    mic = orch.modules[f"audio@{mgr.devices[MAC_MIC.hw_key]}"]
    assert mic.config["audio"]["mic"]["device"] == "MacBook Pro Microphone"

    mgr.apply([FACETIME, MAC_MIC])                                            # iPhone taken away
    iphone = mgr.devices[IPHONE.hw_key]
    assert f"vision@{iphone}" not in orch.modules and mgr.registry.get(iphone)["status"] == "offline"
    assert orch.events[-1].type == "device_offline"
    moved = FoundDevice("camera", "local", "FaceTime HD Camera", "avfoundation:3F45", {"source": 1})
    mgr.apply([moved, IPHONE, MAC_MIC])                                       # back; FaceTime index shifted
    assert orch.modules[f"vision@{facetime}"].config["vision"]["device"] == 1
    assert f"vision@{iphone}" in orch.modules and mgr.registry.get(iphone)["status"] == "online"
    mgr.close()


def test_disabling_on_the_dashboard_stops_the_pipeline(tmp_path):
    mgr, orch = manager(tmp_path, [[FACETIME, MAC_MIC]])
    mgr.start()
    mic_id = mgr.devices[MAC_MIC.hw_key]
    mgr.registry.update(mic_id, enabled=False)                               # Devices page: switch off
    assert orch.events[-1].type == "device_updated"
    mgr.on_event(Event.from_json(orch.events[-1].to_json()))                 # routed back by the orchestrator
    assert f"audio@{mic_id}" not in orch.modules
    mgr.registry.update(mic_id, enabled=True)
    mgr.on_event(orch.events[-1])
    assert f"audio@{mic_id}" in orch.modules
    mgr.close()


def test_seed_rules_and_disabled_modules(tmp_path):
    mgr, orch = manager(tmp_path, [[FACETIME, MAC_MIC]], seed=[{"match": "facetime", "location": "room_a"},
                                                              {"match": "MacBook", "enabled": False}])
    mgr.start()
    facetime = mgr.registry.get(mgr.devices[FACETIME.hw_key])
    assert facetime["location_id"] == "room_a" and orch.modules[f"vision@{facetime['device_id']}"]
    assert not any(n.startswith("audio@") for n in orch.modules)              # seeded as disabled
    mgr.close()


# --- fair scheduler -----------------------------------------------------------------------------------------------

def test_scheduler_is_fair_and_idle_cameras_run_at_one_fps():
    scheduler = FairScheduler(active_fps=5, idle_fps=1)
    for name, active in (("a", True), ("b", True), ("c", True), ("idle", False)):
        scheduler.add(CameraSlot(name, "x", f"buf_{name}", active=active))
    t = 0.0
    while t < 10.0:
        slot = scheduler.pick(t)
        if slot is not None:
            scheduler.served(slot, t)
        t += 0.01
    served = {s.device_id: s.served for s in scheduler}
    assert all(48 <= served[n] <= 51 for n in "abc")                         # ≈ 5 FPS each
    assert 9 <= served["idle"] <= 11                                          # ≈ 1 FPS
    assert scheduler.pick(10.0).active                                        # active cameras go first


# --- pooled consumer across cameras -------------------------------------------------------------------------------

def test_one_consumer_serves_several_cameras(tmp_path):
    from modules.vision.consumer import SharedFrameConsumer
    from modules.vision.shared_frame import SharedFrameBuffer

    class Recorder(SharedFrameConsumer):
        name = "actions"
        source = "camera"
        seen = []

        def process(self, frame_id, timestamp, frame, meta):
            self.seen.append(self.current.device_id)
            self.emit("action", {"track_id": 1, "action": "standing"})

    devices = DeviceRegistry(str(tmp_path / "devices.db"), "hub")
    config = {"network": {"discovery": True, "devices_path": str(tmp_path / "devices.db"), "node_id": "hub"},
              "vision": {"resolution": [64, 48]}, "actions": {"log_events": False}}
    buffers = []
    try:
        ids = []
        for i, location in enumerate(("room_a", "room_b")):
            device, _ = devices.seen("camera", "local", f"cam{i}", f"test:{uuid.uuid4().hex}", location_id=location)
            ids.append(device["device_id"])
            buffers.append(SharedFrameBuffer.create(buffer_name(device["device_id"]), 64, 48))
            buffers[-1].write(b"j", np.zeros((48, 64, 3), np.uint8), {"persons": [{"track_id": 1}]})
        module = Recorder(queue.Queue(), queue.Queue(), config, fps=100)
        module.setup()
        for _ in range(10):
            module.step()
        assert sorted(set(module.seen)) == sorted(ids)
        events = [Event.from_json(module.event_queue.get_nowait()) for _ in range(module.event_queue.qsize())]
        assert {(e.device_id, e.location_id) for e in events} == {(ids[0], "room_a"), (ids[1], "room_b")}
        offline = devices.offline(ids[1])
        module.handle_control(make_event("system", "device_offline", {**offline}, device_id=ids[1]).to_json())
        assert ids[1] not in module.scheduler.slots
        module.teardown()
    finally:
        for buffer in buffers:
            buffer.close()
            buffer.unlink()
        devices.close()


# --- per-location microphone and aggregation ----------------------------------------------------------------------

def test_microphone_wakes_only_for_cameras_in_its_location():
    from modules.audio.mic_controller import MicController

    config = {"audio": {"mic": {"transcript_log": False}}, "_device": {"device_id": "mic-1", "location_id": "room_a"}}
    mic = MicController(queue.Queue(), queue.Queue(), config, frame_source=object(), vad=lambda f: False,
                        transcriber=lambda a, sr: {"text": ""})
    scene = lambda device, location, active=True: make_event(  # noqa: E731
        "camera", "scene_active", {"active": active}, device_id=device, location_id=location).to_json()
    mic.handle_control(scene("cam-b", "room_b"))
    assert mic.state.state == "standby"                                       # someone in another room
    mic.handle_control(scene("cam-a", "room_a"))
    assert mic.state.state == "active" and mic.scene_active
    mic.handle_control(scene("cam-a", "room_a", active=False))
    assert not mic.scene_active
    moved = make_event("system", "device_updated", {"device_id": "mic-1", "location_id": "room_b"})
    mic.on_device_updated(moved)                                              # moved on the Devices page
    mic.handle_control(scene("cam-b", "room_b"))
    assert mic.location_id == "room_b" and mic.scene_active


def test_aggregator_keeps_tracks_of_different_cameras_apart(tmp_path):
    from datetime import datetime

    from modules.aggregator.aggregator import Aggregator
    from modules.aggregator.digest import snapshot_digest

    config = {"aggregator": {"export_dir": str(tmp_path / "exports"), "incidents": {"enabled": False}}}
    aggregator = Aggregator(config, local_now=lambda: datetime(2026, 10, 18, 12, 0))
    for device, location, name in (("cam-a", "room_a", "Oleh"), ("cam-b", "room_b", "unknown_3")):
        aggregator.on_event(make_event("camera", "object_detected", {"track_id": 1, "class": "person",
                                                                     "state": "appeared"},
                                       device_id=device, location_id=location))
        aggregator.on_event(make_event("camera", "person_identified", {"track_id": 1, "identity": name},
                                       device_id=device, location_id=location))
    a = aggregator.snapshot("interval", aggregator.clock(), location_id="room_a")
    b = aggregator.snapshot("interval", aggregator.clock(), location_id="room_b")
    assert [(p["identity"], p["location_id"]) for p in a["people"]] == [("Oleh", "room_a")]     # per location
    assert [(p["identity"], p["location_id"]) for p in b["people"]] == [("unknown_3", "room_b")]
    assert a["location_id"] == "room_a" and snapshot_digest(a).startswith("room_a | ")
    aggregator.close()
