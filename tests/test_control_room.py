"""F13 control room backend: one video WebSocket for all tiles, shared small-frame cache, history for detail views."""
import json
import uuid

import cv2
import numpy as np
from fastapi.testclient import TestClient

from core.events import make_event
from dashboard.server import DashboardState, create_app, query_history, reverse_lines
from dashboard.video import FrameCache, pack, shrink, unpack
from modules.vision.shared_frame import SharedFrameBuffer

AUTH = ("admin", "s3cret")


def jpeg(value, width=640, height=480):
    return cv2.imencode(".jpg", np.full((height, width, 3), value, np.uint8))[1].tobytes()


def test_pack_unpack_and_shrink():
    header, data = unpack(pack({"device_id": "cam-1", "ts": 1.5, "size": "small"}, b"\xff\xd8x"))
    assert header == {"device_id": "cam-1", "ts": 1.5, "size": "small"} and data == b"\xff\xd8x"
    small = cv2.imdecode(np.frombuffer(shrink(jpeg(90)), np.uint8), cv2.IMREAD_COLOR)
    assert small.shape[:2] == (240, 320)
    tiny = jpeg(10, 200, 100)
    assert shrink(tiny) is tiny                                                   # already small


def test_frame_cache_makes_one_small_frame_per_camera_frame():
    name = f"tcr_{uuid.uuid4().hex[:8]}"
    buffer = SharedFrameBuffer.create(name, 640, 480)
    try:
        cache = FrameCache((640, 480), buffer_name=lambda device: name)
        assert cache.frame("cam") is None                                          # nothing written yet
        buffer.write(jpeg(100), np.zeros((480, 640, 3), np.uint8), {})
        first = cache.frame("cam")
        assert cache.frame("cam")[2] is first[2]                                   # cached, not re-encoded
        assert len(cache.frame("cam", "full")[2]) > len(first[2])
        buffer.write(jpeg(200), np.zeros((480, 640, 3), np.uint8), {})
        assert cache.frame("cam")[0] == first[0] + 1
        cache.close()
    finally:
        buffer.unlink()
        buffer.close()
    previews = {"edge-cam": jpeg(50, 320, 240)}
    cache = FrameCache((640, 480), previews=previews)
    frame_id = cache.frame("edge-cam")[0]
    assert cache.frame("edge-cam")[0] == frame_id                                  # same preview: same id
    previews["edge-cam"] = jpeg(60, 320, 240)
    assert cache.frame("edge-cam")[0] == frame_id + 1


def test_one_socket_streams_the_subscribed_cameras(tmp_path):
    from core.device_manager import buffer_name

    names = {f"camera-{i:010x}": None for i in range(3)}
    buffers = {}
    for device in names:
        buffers[device] = SharedFrameBuffer.create(buffer_name(device), 640, 480)
        buffers[device].write(jpeg(40 * (len(buffers) + 1)), np.zeros((480, 640, 3), np.uint8), {})
    config = {"dashboard": {"user": AUTH[0], "password": AUTH[1], "grid_fps": 50, "focus_fps": 50},
              "logging": {"dir": str(tmp_path / "logs")}, "aggregator": {"export_dir": str(tmp_path / "exports")},
              "vision": {"resolution": [640, 480]}, "network": {"node_port": 0}}
    http = TestClient(create_app(DashboardState(config), config))
    try:
        cams = list(names)
        with http.websocket_connect("/ws/video", headers={"Authorization": "Basic YWRtaW46czNjcmV0"}) as ws:
            ws.send_text(json.dumps({"subscribe": cams[:2], "focus": cams[1]}))
            got = {}
            while len(got) < 2:
                header, data = unpack(ws.receive_bytes())
                got[header["device_id"]] = (header["size"], cv2.imdecode(np.frombuffer(data, np.uint8), 1).shape[1])
        assert got == {cams[0]: ("small", 320), cams[1]: ("full", 640)}            # cams[2] not visible: not sent
    finally:
        for buffer in buffers.values():
            buffer.unlink()
            buffer.close()


def test_video_socket_requires_login(tmp_path):
    import pytest
    from starlette.websockets import WebSocketDisconnect

    config = {"dashboard": {"user": AUTH[0], "password": AUTH[1]}, "logging": {"dir": str(tmp_path)},
              "aggregator": {"export_dir": str(tmp_path)}, "vision": {}, "network": {"node_port": 0}}
    with pytest.raises(WebSocketDisconnect):
        with TestClient(create_app(DashboardState(config), config)).websocket_connect("/ws/video") as ws:
            ws.receive_bytes()


def test_history_reads_newest_first_and_filters(tmp_path):
    day = tmp_path / "2026-10-18"
    day.mkdir()
    lines = []
    for i in range(3000):
        loc = "warehouse" if i % 2 else "office"
        kind = "speech" if i % 10 == 0 else "gas_reading"
        event = make_event("audio", kind, {"n": i}, location_id=loc)
        lines.append(event.model_copy(update={"timestamp": f"2026-10-18T10:{i // 60 % 60:02d}:{i % 60:02d}.000Z"})
                     .to_json())
    (day / "events.jsonl").write_text("\n".join(lines) + "\n")
    newest = list(reverse_lines(str(day / "events.jsonl"), chunk=4096))[:3]
    assert [json.loads(line)["payload"]["n"] for line in newest] == [2999, 2998, 2997]
    speech = query_history(str(tmp_path), "office", ["speech"], limit=5)
    assert [e["payload"]["n"] for e in speech] == [2990, 2980, 2970, 2960, 2950]
    recent = query_history(str(tmp_path), "warehouse", since="2026-10-18T10:49:58.000Z")
    assert [e["payload"]["n"] for e in recent] == [2999]
