"""F11 remote nodes without extra programs: pairing, HTTPS certificate, phone, Pi edge node, RTSP / ONVIF."""
import json
import os
import queue
import subprocess
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.devices import DeviceRegistry
from core.events import Event, make_event
from dashboard.server import DashboardState, create_app

AUTH = ("admin", "s3cret")


def setup_app(tmp_path, user="admin", password="s3cret"):
    config = {"dashboard": {"user": user, "password": password}, "logging": {"dir": str(tmp_path / "logs")},
              "aggregator": {"export_dir": str(tmp_path / "exports")}, "vision": {"resolution": [64, 48]},
              "network": {"devices_path": str(tmp_path / "devices.db"), "node_id": "hub", "node_port": 0}}
    state = DashboardState(config)
    bus = []
    state.devices = DeviceRegistry(str(tmp_path / "devices.db"), "hub", publish=bus.append)
    state.node_hub.publish = bus.append
    return TestClient(create_app(state, config)), state, bus


def paired(http, state, kind="browser", name="iPhone B"):
    code = http.post("/api/pairing-code", auth=AUTH).json()["code"]
    answer = http.post("/api/pair", json={"code": code, "name": name, "kind": kind}).json()
    return answer["node_id"], answer["token"]


# --- pairing ------------------------------------------------------------------------------------------------------

def test_pairing_codes_are_one_time_and_expire(tmp_path):
    clock = [1000.0]
    registry = DeviceRegistry(str(tmp_path / "devices.db"), "hub", clock=lambda: clock[0])
    code = registry.pairing_code(ttl_sec=600)
    node_id, token = registry.pair(code, "Pi kitchen", "edge")
    assert node_id.startswith("edge-pi_kitchen-") and registry.node_for_token(token)["node_id"] == node_id
    with pytest.raises(PermissionError):
        registry.pair(code, "again")                                     # one-time
    late = registry.pairing_code(ttl_sec=600)
    clock[0] += 601
    with pytest.raises(PermissionError):
        registry.pair(late, "late")                                      # expired
    assert registry.node_for_token("guess") is None and registry.node_for_token(None) is None
    raw = registry.db.execute("SELECT token_sha256 FROM nodes").fetchone()[0]
    assert token not in raw                                               # only the hash is stored
    assert registry.revoke_node(node_id) and registry.node_for_token(token) is None
    registry.close()


def test_pairing_over_http(tmp_path):
    http, state, _ = setup_app(tmp_path)
    assert http.post("/api/pairing-code").status_code == 401             # dashboard login needed for a code
    assert http.post("/api/pair", json={"code": "000000", "name": "x"}).status_code == 403
    node_id, token = paired(http, state)
    assert [n["node_id"] for n in http.get("/api/nodes", auth=AUTH).json()["nodes"]] == [node_id]
    assert http.get("/api/node/locations", params={"token": token}).status_code == 200
    assert http.get("/api/node/locations", params={"token": "nope"}).status_code == 401
    assert http.post("/api/node/locations", json={"token": token, "name": "Room B"}).json()["location_id"] == "room_b"
    assert "getUserMedia" in http.get("/node").text                       # the phone page needs no dashboard login


def test_https_port_never_serves_an_open_dashboard(tmp_path):
    http, _, _ = setup_app(tmp_path, user=None, password=None)
    assert http.get("/api/status").status_code == 200                     # http://127.0.0.1 without credentials
    https = TestClient(http.app, base_url="https://testserver")
    assert https.get("/api/status").status_code == 401                    # the LAN-facing node port
    assert https.get("/node").status_code == 200


def test_self_signed_certificate_with_the_lan_address(tmp_path):
    from core.tls import ensure_certificate

    names = ["localhost", "hub.local", "127.0.0.1", "192.168.0.112"]
    cert, key = ensure_certificate(str(tmp_path), names)
    text = subprocess.run(["openssl", "x509", "-in", cert, "-noout", "-text"], capture_output=True, text=True).stdout
    assert "IP Address:192.168.0.112" in text and "DNS:hub.local" in text
    assert oct(os.stat(key).st_mode)[-3:] == "600"
    mtime = os.path.getmtime(cert)
    assert ensure_certificate(str(tmp_path), names) == (cert, key) and os.path.getmtime(cert) == mtime   # reused
    ensure_certificate(str(tmp_path), names + ["10.0.0.5"])               # the hub moved to another network
    assert "10.0.0.5" in subprocess.run(["openssl", "x509", "-in", cert, "-noout", "-text"],
                                        capture_output=True, text=True).stdout


# --- browser node (phone) -----------------------------------------------------------------------------------------

def test_phone_streams_camera_and_microphone(tmp_path):
    from modules.audio.mic_controller import SocketSource
    from modules.vision.capture import IngestSource

    http, state, bus = setup_app(tmp_path)
    node_id, token = paired(http, state)
    with http.websocket_connect(f"/node/ws?token={token}") as ws:
        ws.send_json({"name": "iPhone B", "location_id": "room_b", "camera": True, "mic": True})
        ready = ws.receive_json()
        camera, mic = state.devices.get(ready["camera"]), state.devices.get(ready["mic"])
        assert (camera["transport"], camera["location_id"], camera["node_id"]) == ("browser", "room_b", node_id)
        assert camera["info"]["source"].startswith("ingest:iotin_") and mic["info"]["device"].startswith("socket:")
        assert {e.type for e in bus} >= {"device_online"}

        source = IngestSource(camera["info"]["source"][len("ingest:"):], 64, 48, timeout_sec=2)
        source.open()
        mic_source = SocketSource(mic["info"]["device"][len("socket:"):], 480)
        mic_source.open()
        try:
            jpeg = cv2.imencode(".jpg", np.full((480, 640, 3), 120, np.uint8))[1].tobytes()
            ws.send_bytes(bytes([0x01]) + jpeg)
            pcm = (np.sin(np.arange(960) / 5) * 1000).astype(np.int16)
            ws.send_bytes(bytes([0x02]) + pcm.tobytes())
            frame = source.read()
            assert frame.shape == (48, 64, 3) and abs(int(frame.mean()) - 120) < 5      # resized for the hub
            deadline = time.monotonic() + 2
            frames = []
            while len(frames) < 2 and time.monotonic() < deadline:
                frames += mic_source.read()
            assert len(frames) == 2 and np.array_equal(np.concatenate(frames), pcm)
        finally:
            source.close()
            mic_source.close()
        ws.send_text(json.dumps({"location_id": "kitchen"}))                       # moved on the phone
        time.sleep(0.2)
    assert state.devices.get(camera["device_id"])["location_id"] == "kitchen"
    assert state.devices.get(camera["device_id"])["status"] == "offline"           # page closed
    with pytest.raises(FileNotFoundError):
        from modules.vision.shared_frame import SharedFrameBuffer
        SharedFrameBuffer.attach(camera["info"]["source"][len("ingest:"):], 64, 48)  # buffer released


def test_unpaired_phone_is_rejected(tmp_path):
    from starlette.websockets import WebSocketDisconnect

    http, _, _ = setup_app(tmp_path)
    with pytest.raises(WebSocketDisconnect):
        with http.websocket_connect("/node/ws?token=nope") as ws:
            ws.receive_json()


# --- edge node (Raspberry Pi) -------------------------------------------------------------------------------------

def test_edge_node_events_previews_and_location_back(tmp_path):
    http, state, bus = setup_app(tmp_path)
    node_id, token = paired(http, state, kind="edge", name="Pi kitchen")
    with http.websocket_connect(f"/node/edge?token={token}") as ws:
        ws.send_text(json.dumps({"type": "devices", "devices": [
            {"kind": "gas", "transport": "local", "name": "Gas sensor gas-1", "hw_key": "gas:gas-1",
             "location_id": "unassigned", "enabled": True},
            {"kind": "camera", "transport": "picamera", "name": "Pi camera", "hw_key": "picamera:0",
             "location_id": "unassigned", "enabled": True}]}))
        time.sleep(0.2)
        gas = next(d for d in state.devices.list_devices(node_id=node_id) if d["kind"] == "gas")
        assert gas["transport"] == "edge" and gas["status"] == "online"
        state.devices.update(gas["device_id"], location_id="kitchen")            # chosen on the hub dashboard
        state.node_hub.forward_to_edge(bus[-1])
        assert json.loads(ws.receive_text()) == {"type": "device_updated", "device_id": gas["device_id"],
                                                 "location_id": "kitchen", "enabled": True,
                                                 "name": "Gas sensor gas-1"}
        reading = make_event("gas", "gas_reading", {"gas": "acetone", "concentration_pct": 3.2},
                             device_id=gas["device_id"], node_id=node_id, location_id="unassigned")
        spoofed = make_event("gas", "alert", {"reason": "x"}, device_id="camera-hub", node_id="hub")
        for event in (reading, spoofed):
            ws.send_text(json.dumps({"type": "event", "event": json.loads(event.to_json())}))
        from modules.uplink import preview_message

        camera = next(d for d in state.devices.list_devices(node_id=node_id) if d["kind"] == "camera")
        ws.send_bytes(preview_message(camera["device_id"], b"\xff\xd8jpeg"))
        time.sleep(0.3)
    injected = [e for e in bus if e.type == "gas_reading"]
    assert len(injected) == 1 and injected[0].location_id == "kitchen" and injected[0].received_at
    assert not [e for e in bus if e.type == "alert"]                               # a node can't speak for the hub
    assert state.node_hub.previews[camera["device_id"]] == b"\xff\xd8jpeg"
    assert state.devices.get(gas["device_id"])["status"] == "offline"              # disconnected


def test_uplink_session_sends_events_devices_and_applies_hub_changes(tmp_path):
    from modules.uplink import Uplink, ws_url

    registry = DeviceRegistry(str(tmp_path / "devices.db"), "edge-pi-1")
    device, _ = registry.seen("gas", "local", "Gas", "gas:gas-1")
    registry.close()

    class FakeWs:
        def __init__(self, module):
            self.sent, self.module = [], module
            self.incoming = [json.dumps({"type": "device_updated", "device_id": device["device_id"],
                                         "location_id": "kitchen", "enabled": True})]

        def send(self, message):
            self.sent.append(message)

        def recv(self, timeout):
            if self.incoming:
                return self.incoming.pop(0)
            self.module._stop.set()
            raise TimeoutError

    config = {"network": {"devices_path": str(tmp_path / "devices.db"), "node_id": "edge-pi-1"},
              "uplink": {"hub": "https://192.168.0.112:8443", "token": "t"}}
    module = Uplink(queue.Queue(), queue.Queue(), config, connect=lambda url: None)
    from core.devices import DeviceRegistry as Registry
    module.registry = Registry(config["network"]["devices_path"], "edge-pi-1", publish=module.publish)
    module.handle_control(make_event("gas", "gas_reading", {"gas": "acetone"}).to_json())
    module.handle_control(make_event("system", "heartbeat", {"module": "gas"}).to_json())   # stays local
    module.handle_control(make_event("system", "device_updated", {"device_id": "x"}).to_json())  # → device list
    ws = FakeWs(module)
    module.session(ws)
    kinds = [json.loads(m)["type"] for m in ws.sent if isinstance(m, str)]
    assert kinds == ["devices", "event"]
    assert json.loads(ws.sent[0])["devices"][0]["hw_key"] == "gas:gas-1"
    assert module.registry.get(device["device_id"])["location_id"] == "kitchen"   # applied locally
    assert Event.from_json(module.event_queue.get_nowait()).type == "device_updated"   # → local modules
    assert ws_url("https://h:8443", "/node/edge", "t") == "wss://h:8443/node/edge?token=t"
    module.registry.close()


def test_agent_config_for_an_edge_node():
    from node_agent.agent import edge_config

    config = edge_config({"network": {}, "dashboard": {"enabled": True}, "identity": {"enabled": True}},
                         {"hub": "https://h:8443", "node_id": "edge-pi-1", "token": "t", "cafile": "hub.pem"})
    assert config["network"]["node_id"] == "edge-pi-1" and config["network"]["discovery"]
    assert not config["dashboard"]["enabled"] and not config["identity"]["enabled"]
    assert config["uplink"] == {"hub": "https://h:8443", "token": "t", "cafile": "hub.pem"}


# --- RTSP / ONVIF and hub pipelines for remote devices ------------------------------------------------------------

def test_rtsp_camera_and_browser_devices_get_hub_pipelines(tmp_path):
    from core.device_manager import DeviceManager
    from tests.test_devices import FakeOrchestrator

    http, state, bus = setup_app(tmp_path)
    added = http.post("/api/devices/rtsp", json={"name": "Gate", "url": "rtsp://u:p@192.168.0.50:554/s1",
                                                 "location_id": "yard"}, auth=AUTH).json()
    device = state.devices.get(added["device_id"])
    assert device["transport"] == "rtsp" and "u:p" not in device["hw_key"] and device["location_id"] == "yard"
    assert http.post("/api/devices/rtsp", json={"name": "x", "url": "file:///etc/passwd"}, auth=AUTH).status_code == 422

    orch = FakeOrchestrator()
    config = {"vision": {"enabled": True, "model": "yolo11n.pt"}, "audio": {"enabled": True},
              "network": {"devices_path": str(tmp_path / "devices.db"), "node_id": "hub"}}
    manager = DeviceManager(orch, config, scan=lambda cfg: [])
    manager.start()                                                         # RTSP cameras start with the hub
    spec = orch.modules[f"vision@{device['device_id']}"]
    assert spec.config["vision"]["device"] == "rtsp://u:p@192.168.0.50:554/s1"
    node_id, token = paired(http, state)
    with http.websocket_connect(f"/node/ws?token={token}") as ws:
        ws.send_json({"name": "iPhone", "camera": True, "mic": False})
        camera_id = ws.receive_json()["camera"]
        manager.on_event(next(e for e in reversed(bus) if e.type == "device_online"))
        assert orch.modules[f"vision@{camera_id}"].config["vision"]["camera"] == "ingest"
    manager.on_event(next(e for e in reversed(bus) if e.type == "device_offline"))
    assert f"vision@{camera_id}" not in orch.modules
    manager.close()


def test_onvif_probe_match_parsing():
    from core.onvif import parse_probe_match

    answer = ('<d:ProbeMatch><d:Scopes>onvif://www.onvif.org/name/Gate%20Cam onvif://www.onvif.org/hardware/IPC-1'
              '</d:Scopes><d:XAddrs>http://192.168.0.50/onvif/device_service http://[fe80::1]/onvif</d:XAddrs>'
              '</d:ProbeMatch>')
    assert parse_probe_match(answer) == {"address": "http://192.168.0.50/onvif/device_service",
                                         "host": "192.168.0.50", "name": "Gate Cam", "hardware": "IPC-1"}
    assert parse_probe_match("<nothing/>") == {}


def test_opencv_source_treats_urls_as_streams():
    from modules.vision.capture import OpenCVSource

    stream, video = OpenCVSource("rtsp://cam/1"), OpenCVSource("clip.mp4")
    assert stream.is_stream and not stream.is_file and video.is_file and not video.is_stream
