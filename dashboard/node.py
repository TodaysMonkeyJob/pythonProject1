"""
Remote nodes of the device network (F11 / TOD-22), served by the dashboard — no MQTT broker or other programs.

* Pairing: the dashboard shows a one-time 6-digit code (``POST /api/pairing-code``); a node exchanges it for a token
  (``POST /api/pair``). Every node connection carries the token; only its hash is stored (``devices.db``).
* Browser node (``https://<hub>:8443/node``, any phone / laptop): ``/node/ws`` — after a JSON hello
  ``{name, location_id, camera, mic, width, height}`` the page streams binary messages: ``0x01`` + JPEG frame,
  ``0x02`` + 16 kHz int16 PCM. Frames are decoded into a per-device shared memory buffer (the hub's ``vision@`` pipeline
  reads it via ``IngestSource``), audio goes to the ``audio@`` pipeline's unix socket (``SocketSource``). The devices
  are registered as transport ``browser`` and go offline when the page disconnects.
* Edge node (a Raspberry Pi running ``node_agent``): ``/node/edge`` — JSON ``{"type": "devices", ...}`` (its device
  list), ``{"type": "event", "event": {...}}`` (events of its own full pipeline, re-published on the hub bus with
  ``received_at`` and the location from the hub registry) and binary ``0x03`` + device id + JPEG previews. Location /
  enabled changes made on the hub dashboard are sent back as ``{"type": "device_updated", ...}``.
* RTSP cameras: ``POST /api/devices/rtsp`` (URL), ``POST /api/devices/onvif-scan`` (WS-Discovery on the LAN).
* ``announce()``: UDP broadcast of the hub's HTTPS address every few seconds, so a Pi agent finds the hub without
  mDNS.
"""
import asyncio
import hashlib
import json
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.events import Event
from system_setup.logger_setup import init_logger

logger = init_logger()

FRAME, AUDIO, PREVIEW = 0x01, 0x02, 0x03
ANNOUNCE_PORT = 47800


def ingest_buffer_name(device_id: str) -> str:
    """Shared memory the dashboard writes a browser camera's frames to (≤ 30 chars for macOS)."""
    return f"iotin_{device_id.split('-')[-1][:12]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class NodeHub:
    """State shared by the node endpoints (dashboard process)."""

    def __init__(self, state, config: Dict[str, Any]):
        self.state = state            # DashboardState: .devices (DeviceRegistry), .loop, publish via .publish
        self.config = config
        self.width, self.height = config.get("vision", {}).get("resolution", [640, 480])
        self.previews: Dict[str, bytes] = {}          # edge device_id → latest JPEG
        self.edge_sockets: Dict[str, Any] = {}        # edge node_id → websocket
        self.publish = None                           # set by the module (BaseModule.publish)

    def registry(self):
        if self.state.devices is None:
            raise RuntimeError("device registry not available")
        return self.state.devices

    # --- browser nodes --------------------------------------------------------------------------------------------

    def browser_connected(self, node: Dict[str, Any], hello: Dict[str, Any]):
        """Registers the page's camera / microphone. :return: (camera device | None, buffer | None, mic | None)"""
        from core.devices import make_device_id
        from modules.audio.mic_controller import audio_socket_path
        from modules.vision.shared_frame import SharedFrameBuffer

        registry, node_id = self.registry(), node["node_id"]
        location = str(hello.get("location_id") or "") or None
        name = str(hello.get("name") or node["name"])[:60]
        camera = buffer = mic = None
        if hello.get("camera", True):
            device_id = make_device_id("camera", node_id, f"browser:{node_id}:camera")
            buffer_name = ingest_buffer_name(device_id)
            try:
                SharedFrameBuffer.attach(buffer_name, self.width, self.height).unlink()   # stale
            except FileNotFoundError:
                pass
            buffer = SharedFrameBuffer.create(buffer_name, self.width, self.height)  # before the pipeline starts
            camera, _ = registry.seen("camera", "browser", f"{name} camera", f"browser:{node_id}:camera",
                                      {"source": f"ingest:{buffer_name}"}, location_id=location, node_id=node_id)
        if hello.get("mic", True):
            device_id = make_device_id("mic", node_id, f"browser:{node_id}:mic")
            mic, _ = registry.seen("mic", "browser", f"{name} microphone", f"browser:{node_id}:mic",
                                   {"device": f"socket:{audio_socket_path(device_id)}"}, location_id=location,
                                   node_id=node_id)
        for device in (camera, mic):
            if device is not None and location and device["location_id"] != location:
                registry.update(device["device_id"], location_id=location)   # chosen on the phone
        return camera, buffer, mic

    def browser_frame(self, buffer, data: bytes) -> bool:
        import cv2
        import numpy as np

        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return False
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        buffer.write(data, frame, {})
        return True

    def browser_disconnected(self, camera, buffer, mic) -> None:
        registry = self.registry()
        for device in (camera, mic):
            if device is not None:
                registry.offline(device["device_id"])
        if buffer is not None:
            buffer.close()
            try:
                buffer.unlink()
            except FileNotFoundError:
                pass

    # --- edge nodes -----------------------------------------------------------------------------------------------

    def edge_devices(self, node: Dict[str, Any], devices) -> int:
        registry = self.registry()
        for d in devices or []:
            if d.get("kind") in ("camera", "mic", "gas") and d.get("hw_key"):
                registry.seen(d["kind"], "edge", str(d.get("name", d["hw_key"]))[:80], str(d["hw_key"]),
                              {"edge_transport": d.get("transport")}, location_id=d.get("location_id") or None,
                              node_id=node["node_id"], enabled=d.get("enabled"))
        return len(devices or [])

    def edge_event(self, node: Dict[str, Any], raw: Dict[str, Any]) -> Optional[Event]:
        """Re-publishes an edge node's event on the hub bus (location from the hub registry, ``received_at``)."""
        event = Event.from_json(json.dumps(raw))
        if event.node_id != node["node_id"] or event.type == "heartbeat" or event.type.startswith("device_"):
            return None  # a node only speaks for itself; heartbeats stay local; devices come as the device list
        device = self.registry().get(event.device_id)
        update = {"received_at": utc_now()}
        if device is not None:
            update["location_id"] = device["location_id"]
        event = event.model_copy(update=update)
        if self.publish is not None:
            self.publish(event)
        return event

    def edge_disconnected(self, node: Dict[str, Any]) -> None:
        self.edge_sockets.pop(node["node_id"], None)
        registry = self.registry()
        for device in registry.list_devices(node_id=node["node_id"], status="online"):
            registry.offline(device["device_id"])

    def forward_to_edge(self, event: Event) -> None:
        """Hub-side location / enabled changes of an edge device → that node (called in the module thread)."""
        if event.type != "device_updated":
            return
        websocket = self.edge_sockets.get(event.payload.get("node_id", ""))
        if websocket is None or self.state.loop is None:
            return
        message = {"type": "device_updated", **{k: event.payload.get(k) for k in ("device_id", "location_id",
                                                                                   "enabled", "name")}}
        asyncio.run_coroutine_threadsafe(websocket.send_text(json.dumps(message)), self.state.loop)


async def preview_frames(hub: NodeHub, device_id: str, fps: float, max_frames: Optional[int] = None):
    """MJPEG of an edge device's previews (a new one every ~2 s)."""
    last, sent = None, 0
    while max_frames is None or sent < max_frames:
        jpeg = hub.previews.get(device_id)
        if jpeg is not None and jpeg is not last:
            last, sent = jpeg, sent + 1
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode()
                   + b"\r\n\r\n" + jpeg + b"\r\n")
        await asyncio.sleep(1.0 / fps)


def add_node_routes(app, hub: NodeHub, auth, static_dir: str) -> None:
    from typing import Annotated

    from fastapi import Body, Depends, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse

    def node_for(token: Optional[str]) -> Dict[str, Any]:
        node = hub.registry().node_for_token(token)
        if node is None:
            raise HTTPException(401, "pair this node first (pairing code from the dashboard)")
        return node

    @app.get("/node")
    async def node_page():
        return FileResponse(f"{static_dir}/node.html")

    @app.post("/api/pairing-code", dependencies=[Depends(auth)])
    def api_pairing_code():
        ttl = 600
        return {"code": hub.registry().pairing_code(ttl), "expires_in_sec": ttl}

    @app.post("/api/pair")
    def api_pair(code: Annotated[str, Body()], name: Annotated[str, Body()] = "phone",
                 kind: Annotated[str, Body()] = "browser"):
        if kind not in ("browser", "edge"):
            raise HTTPException(422, "kind must be browser or edge")
        try:
            node_id, token = hub.registry().pair(code, name, kind)
        except PermissionError as e:
            time.sleep(1.0)  # slows down guessing the 6 digits
            raise HTTPException(403, str(e)) from e
        return {"node_id": node_id, "token": token}

    @app.get("/api/nodes", dependencies=[Depends(auth)])
    def api_nodes():
        return {"nodes": hub.registry().list_nodes()}

    @app.delete("/api/nodes/{node_id}", dependencies=[Depends(auth)])
    def api_revoke(node_id: str):
        if not hub.registry().revoke_node(node_id):
            raise HTTPException(404, "unknown node")
        return {"revoked": node_id}

    @app.get("/api/node/locations")
    def api_node_locations(token: str):
        node_for(token)
        return {"locations": hub.registry().list_locations()}

    @app.post("/api/node/locations")
    def api_node_add_location(token: Annotated[str, Body()], name: Annotated[str, Body()]):
        node_for(token)
        if not name.strip():
            raise HTTPException(422, "empty name")
        return hub.registry().add_location(name)

    @app.post("/api/devices/rtsp", dependencies=[Depends(auth)])
    def api_add_rtsp(name: Annotated[str, Body()], url: Annotated[str, Body()],
                     location_id: Annotated[Optional[str], Body()] = None):
        if not url.startswith(("rtsp://", "rtsps://", "http://", "https://")):
            raise HTTPException(422, "URL must start with rtsp:// (or http(s):// for MJPEG)")
        hw_key = "rtsp:" + hashlib.sha256(url.encode()).hexdigest()[:16]   # the URL (credentials) is not an id
        device, _ = hub.registry().seen("camera", "rtsp", name.strip()[:60] or "IP camera", hw_key, {"source": url},
                                        location_id=location_id or None)
        return {k: device[k] for k in ("device_id", "name", "location_id", "status")}

    @app.post("/api/devices/onvif-scan", dependencies=[Depends(auth)])
    def api_onvif_scan():
        from core.onvif import discover

        return {"found": discover(3.0)}

    @app.websocket("/node/ws")
    async def node_stream(websocket: WebSocket, token: str = ""):
        node = hub.registry().node_for_token(token)
        if node is None:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        if hub.state.loop is None:
            hub.state.loop = asyncio.get_running_loop()
        camera = buffer = mic = None
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.setblocking(False)
        try:
            hello = await websocket.receive_json()
            camera, buffer, mic = hub.browser_connected(node, hello)
            await websocket.send_json({"type": "ready", "camera": camera and camera["device_id"],
                                       "mic": mic and mic["device_id"]})
            audio_path = mic["info"]["device"][len("socket:"):] if mic else None
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data:
                    if data[0] == FRAME and buffer is not None:
                        hub.browser_frame(buffer, data[1:])
                    elif data[0] == AUDIO and audio_path:
                        try:
                            sock.sendto(data[1:], audio_path)
                        except OSError:
                            pass  # the audio pipeline is (re)starting
                elif message.get("text"):
                    change = json.loads(message["text"])
                    if change.get("location_id"):
                        for device in (camera, mic):
                            if device is not None:
                                hub.registry().update(device["device_id"], location_id=change["location_id"])
        except WebSocketDisconnect:
            pass
        finally:
            sock.close()
            hub.browser_disconnected(camera, buffer, mic)

    @app.websocket("/node/edge")
    async def edge_stream(websocket: WebSocket, token: str = ""):
        node = hub.registry().node_for_token(token)
        if node is None or node["kind"] != "edge":
            await websocket.close(code=1008)
            return
        await websocket.accept()
        if hub.state.loop is None:
            hub.state.loop = asyncio.get_running_loop()
        hub.edge_sockets[node["node_id"]] = websocket
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data and data[0] == PREVIEW:
                    size = int.from_bytes(data[1:3], "big")
                    hub.previews[data[3:3 + size].decode()] = data[3 + size:]
                elif message.get("text"):
                    payload = json.loads(message["text"])
                    if payload.get("type") == "devices":
                        hub.edge_devices(node, payload.get("devices"))
                    elif payload.get("type") == "event":
                        try:
                            hub.edge_event(node, payload["event"])
                        except ValueError as e:
                            logger.warning(f"Bad event from {node['node_id']}: {e}")
        except WebSocketDisconnect:
            pass
        finally:
            hub.edge_disconnected(node)


def announce(https_url: str, node_id: str, stop: threading.Event, interval_sec: float = 5.0,
             port: int = ANNOUNCE_PORT) -> None:
    """UDP broadcast ``{"service": "iotguard", "hub": <https url>}`` until ``stop`` is set (thread target)."""
    message = json.dumps({"service": "iotguard", "hub": https_url, "node_id": node_id}).encode()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while not stop.is_set():
            try:
                sock.sendto(message, ("255.255.255.255", port))
            except OSError as e:  # no network right now
                logger.debug(f"Hub announce failed: {e}")
            stop.wait(interval_sec)


def find_hub(timeout_sec: float = 30.0, port: int = ANNOUNCE_PORT) -> Optional[str]:
    """Edge agent side: waits for a hub announcement. :return: the hub's https URL or None"""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", port))
        sock.settimeout(1.0)
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                data, _ = sock.recvfrom(4096)
                message = json.loads(data)
                if message.get("service") == "iotguard" and message.get("hub"):
                    return message["hub"]
            except (socket.timeout, ValueError):
                continue
    return None
