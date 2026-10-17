import asyncio
import base64
import json
import queue
import socket
import threading
import time
import uuid
import zipfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.events import make_event
from dashboard.server import DashboardModule, DashboardState, create_app, load_history, mjpeg_frames
from modules.aggregator.outbox import Outbox

AUTH = ("admin", "s3cret")


def config(tmp_path, **dashboard):
    return {"dashboard": {"user": AUTH[0], "password": AUTH[1], "dashboard_fps": 50, **dashboard},
            "logging": {"dir": str(tmp_path / "logs")},
            "aggregator": {"export_dir": str(tmp_path / "exports"),
                           "outbox_path": str(tmp_path / "exports" / "outbox.db"),
                           "incidents": {"dir": str(tmp_path / "exports" / "incidents")}},
            "gas": {"thresholds": {"acetone": 5.0}}, "vision": {},
            "identity": {"registry_path": str(tmp_path / "identity.db"), "faces_dir": str(tmp_path / "persons")}}


def client(tmp_path, **dashboard):
    cfg = config(tmp_path, **dashboard)
    state = DashboardState(cfg)
    return TestClient(create_app(state, cfg)), state


def auth_header():
    return {"Authorization": "Basic " + base64.b64encode(f"{AUTH[0]}:{AUTH[1]}".encode()).decode()}


def write_log(tmp_path, events):
    day = tmp_path / "logs" / "2026-10-17"
    day.mkdir(parents=True, exist_ok=True)
    with open(day / "events.jsonl", "w", encoding="utf-8") as f:
        f.writelines(e.to_json() + "\n" for e in events)


# --- auth & pages -----------------------------------------------------------------------------------------------

def test_basic_auth_required(tmp_path):
    http, _ = client(tmp_path)
    assert http.get("/").status_code == 401
    assert http.get("/api/status", auth=("admin", "wrong")).status_code == 401
    page = http.get("/", auth=AUTH)
    assert page.status_code == 200 and "IoT Guard" in page.text and "/ws" in page.text


def test_status_reports_modules_and_system(tmp_path):
    http, state = client(tmp_path)
    vision = make_event("system", "heartbeat", {"module": "vision", "frame_fps": 14.2, "yolo_fps": 6.1})
    state.on_event(vision.to_json())
    state.on_event(make_event("system", "heartbeat", {"module": "audio", "mic_state": "standby"}).to_json())
    state.heartbeats["gas"] = {"module": "gas", "at": time.time() - 60}                 # silent for a minute
    status = http.get("/api/status", auth=AUTH).json()
    assert status["video_fps"] == 14.2 and status["yolo_fps"] == 6.1 and status["mic_state"] == "standby"
    assert status["modules"]["vision"]["alive"] and not status["modules"]["gas"]["alive"]
    assert status["ram_total_mb"] > 0 and status["gas_thresholds"] == {"acetone": 5.0}


def test_snapshot_shows_digest_sent_to_gpt(tmp_path):
    http, _ = client(tmp_path)
    box = Outbox(str(tmp_path / "exports" / "outbox.db"))
    box.add({"snapshot_id": "abc", "trigger": "interval", "people": []}, "interval", digest="10:00-10:01 UTC+3")
    box.close()
    data = http.get("/api/snapshot/abc", auth=AUTH).json()
    assert data["digest_sent_to_gpt"] == "10:00-10:01 UTC+3" and data["snapshot"]["people"] == []
    assert http.get("/api/snapshot/nope", auth=AUTH).status_code == 404


def test_incident_download(tmp_path):
    http, _ = client(tmp_path)
    folder = tmp_path / "exports" / "incidents"
    folder.mkdir(parents=True)
    with zipfile.ZipFile(folder / "20261017-120000-ab12.zip", "w") as z:
        z.writestr("summary.txt", "hello")
    response = http.get("/api/incident/20261017-120000-ab12", auth=AUTH)
    assert response.status_code == 200 and response.headers["content-type"] == "application/zip"
    assert http.get("/api/incident/..%2F..%2Fetc", auth=AUTH).status_code == 404       # no path traversal


# --- websocket --------------------------------------------------------------------------------------------------

def test_websocket_sends_history_then_live_events_fast(tmp_path):
    history = [make_event("gas", "gas_reading", {"gas": "acetone", "concentration_pct": i / 10, "trend": "stable"})
               for i in range(250)]
    write_log(tmp_path, history)
    http, state = client(tmp_path, history_events=200)
    with http.websocket_connect("/ws", headers=auth_header()) as ws:
        received = [json.loads(ws.receive_text()) for _ in range(200)]
        values = [r["payload"]["concentration_pct"] for r in received]
        assert values[0] == 5.0 and values[-1] == 24.9                                     # the newest 200
        speech = make_event("audio", "speech", {"text": "<script>alert(1)</script>", "language": "uk"})
        sent_at = time.monotonic()
        threading.Thread(target=state.on_event, args=(speech.to_json(),)).start()   # from the module thread
        live = json.loads(ws.receive_text())
        latency = time.monotonic() - sent_at
    assert live["type"] == "speech" and live["payload"]["text"] == "<script>alert(1)</script>"
    assert latency <= 1.0
    assert not state.clients                                                            # tab closed → client gone


def test_websocket_rejects_missing_auth(tmp_path):
    http, _ = client(tmp_path)
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with http.websocket_connect("/ws") as ws:
            ws.receive_text()


def test_heartbeats_not_shown_in_the_event_stream(tmp_path):
    _, state = client(tmp_path)
    state.on_event(make_event("system", "heartbeat", {"module": "gas"}).to_json())
    assert not state.recent and "gas" in state.heartbeats


def test_load_history_spans_days(tmp_path):
    for day, n in (("2026-10-16", 150), ("2026-10-17", 100)):
        (tmp_path / day).mkdir()
        with open(tmp_path / day / "events.jsonl", "w") as f:
            f.writelines(f'{{"n": "{day}-{i}"}}\n' for i in range(n))
    lines = load_history(str(tmp_path), 200)
    assert len(lines) == 200
    assert json.loads(lines[0])["n"] == "2026-10-16-50" and json.loads(lines[-1])["n"] == "2026-10-17-99"


# --- video ------------------------------------------------------------------------------------------------------

def parse_parts(data):
    frames = []
    while b"\r\n\r\n" in data:
        head, _, rest = data.partition(b"\r\n\r\n")
        length = int(head.split(b"Content-Length: ")[1].split(b"\r\n")[0])
        if len(rest) < length + 2:
            break
        frames.append(rest[:length])
        data = rest[length + 2:]
    return frames


def collect(vision, count, fps=100):
    async def run():
        return [part async for part in mjpeg_frames(vision, fps, max_frames=count)]
    return parse_parts(b"".join(asyncio.run(run())))


def test_video_streams_new_shared_frames_only(tmp_path):
    from modules.vision.shared_frame import SharedFrameBuffer

    name = f"tdash_{uuid.uuid4().hex[:8]}"
    buffer = SharedFrameBuffer.create(name, 64, 48)
    try:
        jpeg = cv2.imencode(".jpg", np.full((48, 64, 3), 200, np.uint8))[1].tobytes()
        buffer.write(jpeg, np.zeros((48, 64, 3), np.uint8), {})
        vision = {"shared_buffer": name, "resolution": [64, 48]}
        assert collect(vision, 1) == [jpeg]

        async def two_frames():
            parts, generator = [], mjpeg_frames(vision, 100, max_frames=2)
            parts.append(await generator.__anext__())
            buffer.write(b"second-frame", np.zeros((48, 64, 3), np.uint8), {})   # a new frame arrives
            parts.append(await generator.__anext__())
            await generator.aclose()
            return parse_parts(b"".join(parts))
        assert asyncio.run(two_frames()) == [jpeg, b"second-frame"]         # the same frame is not re-sent
    finally:
        buffer.unlink()
        buffer.close()


def test_video_placeholder_without_camera():
    (frame,) = collect({}, 1)
    assert cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR) is not None


# --- module process ---------------------------------------------------------------------------------------------

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_module_serves_and_stops_cleanly(tmp_path):
    import httpx

    cfg = config(tmp_path, host="127.0.0.1", port=free_port())
    module = DashboardModule(queue.Queue(), queue.Queue(), cfg)
    thread = threading.Thread(target=module.run)
    thread.start()
    url = f"http://127.0.0.1:{cfg['dashboard']['port']}"
    try:
        for _ in range(50):
            try:
                assert httpx.get(url + "/api/status", auth=AUTH, timeout=1).status_code == 200
                break
            except httpx.TransportError:
                time.sleep(0.1)
        else:
            pytest.fail("dashboard did not start")
        with httpx.stream("GET", url + "/video", auth=AUTH, timeout=5) as response:
            first = next(response.iter_bytes())
        assert response.headers["content-type"].startswith("multipart/x-mixed-replace") and b"image/jpeg" in first
    finally:
        module.control_queue.put("stop")
        thread.join(timeout=10)
    assert not thread.is_alive()
    with pytest.raises(httpx.TransportError):
        httpx.get(url + "/api/status", auth=AUTH, timeout=1)


def test_without_credentials_binds_localhost_only(tmp_path):
    cfg = config(tmp_path, port=free_port())
    cfg["dashboard"].update(user=None, password=None)
    module = DashboardModule(queue.Queue(), queue.Queue(), cfg)
    module.setup()
    try:
        assert module.server.config.host == "127.0.0.1"
    finally:
        module.teardown()


# --- people (F9 identity registry) -------------------------------------------------------------------------------

def people_client(tmp_path):
    from core.identity import IdentityRegistry
    from tests.test_identity_registry import OLEH_FACE, OLEH_VOICE

    http, state = client(tmp_path)
    bus = []
    state.registry = IdentityRegistry(str(tmp_path / "identity.db"), publish=bus.append,
                                      faces_dir=str(tmp_path / "persons"))
    face_id = state.registry.create_unknown("face", OLEH_FACE)
    voice_id = state.registry.create_unknown("voice", OLEH_VOICE)
    bus.clear()
    return http, state, bus, face_id, voice_id


def test_persons_list_rename_merge_delete(tmp_path):
    http, state, bus, face_id, voice_id = people_client(tmp_path)
    listing = http.get("/api/persons", auth=AUTH).json()
    assert {p["display_name"] for p in listing["persons"]} == {"unknown_1", "unknown_2"}
    assert http.get("/api/persons", auth=("admin", "wrong")).status_code == 401

    renamed = http.patch(f"/api/persons/{face_id}", json={"display_name": "Oleh"}, auth=AUTH)
    assert renamed.status_code == 200 and renamed.json()["display_name"] == "Oleh" and renamed.json()["is_known"]
    assert bus[-1].type == "identity_renamed"                                   # → every module refreshes
    assert http.patch(f"/api/persons/{voice_id}", json={"display_name": "OLEH"}, auth=AUTH).status_code == 409
    assert http.patch(f"/api/persons/{voice_id}", json={"display_name": ""}, auth=AUTH).status_code == 422
    assert http.patch("/api/persons/nope", json={"display_name": "X"}, auth=AUTH).status_code == 404

    merged = http.post(f"/api/persons/{voice_id}/merge", json={"target_id": face_id}, auth=AUTH).json()
    assert merged["face_samples"] == 1 and merged["voice_samples"] == 1 and bus[-1].type == "identity_merged"
    listing = http.get("/api/persons", auth=AUTH).json()
    assert [p["person_id"] for p in listing["persons"]] == [face_id] and listing["aliases"] == {voice_id: face_id}

    thumb = http.get(f"/api/persons/{face_id}/thumbnail", auth=AUTH)
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"   # placeholder
    assert http.delete(f"/api/persons/{face_id}", auth=AUTH).status_code == 200
    assert http.get("/api/persons", auth=AUTH).json()["persons"] == [] and bus[-1].type == "identity_deleted"


def test_identity_events_from_other_processes_refresh_the_cache(tmp_path):
    from core.identity import IdentityRegistry

    http, state, bus, face_id, _ = people_client(tmp_path)
    other = IdentityRegistry(str(tmp_path / "identity.db"))                     # e.g. the CLI
    event = other.rename(face_id, "Anna")
    state.on_event(event.to_json())
    assert state.registry.name(face_id) == "Anna"
    assert http.get("/api/persons", auth=AUTH).json()["persons"][0]["display_name"] == "Anna"


def test_page_has_people_panel(tmp_path):
    page = client(tmp_path)[0].get("/", auth=AUTH).text
    assert 'id="people"' in page and "/api/persons" in page and "renderNames" in page
