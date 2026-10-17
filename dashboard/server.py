"""
Real-time monitoring dashboard (F7): FastAPI + uvicorn in its own process.

* ``/`` — single page (``dashboard/static/index.html``): camera, system status, transcripts, event log, gas chart,
  GPT decisions.
* ``/ws`` — every bus event (forwarded by the orchestrator to the control queue) is pushed to the browsers; on
  connect the last ``history_events`` events from ``logs/<date>/events.jsonl`` are sent first.
* ``/video`` — MJPEG (``multipart/x-mixed-replace``) of the annotated frames in the vision shared buffer, at most
  ``dashboard_fps``; the JPEGs are already encoded by F1, so streaming costs almost no CPU.
* ``/api/status`` — module heartbeats (alive / age, FPS, mic state), CPU temperature, RAM, CPU load.
* ``/api/snapshot/{id}`` — what GPT got (``digest``) and the full snapshot JSON from the outbox.
* ``/api/incident/{id}`` — download the local incident archive (zip).
* ``/api/persons`` (F9) — the identity registry: ``GET`` list (+ ``aliases`` of merged ids), ``PATCH /{id}``
  ``{display_name}``, ``POST /{id}/merge`` ``{target_id}``, ``DELETE /{id}``, ``GET /{id}/thumbnail``. The single
  rename point: changes are published as ``identity_*`` events, so every module and browser updates at once.

HTTP Basic auth with ``DASHBOARD_USER`` / ``DASHBOARD_PASS`` (.env). Without credentials the server only listens on
127.0.0.1. The web server runs in a background thread; the module loop keeps sending heartbeats and handles "stop",
so a closed browser tab never affects other modules.
"""
import asyncio
import base64
import collections
import glob
import json
import os
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Annotated, Any, Deque, Dict, List, Optional, Set

from core.events import Event
from core.module import BaseModule
from system_setup.logger_setup import init_logger

logger = init_logger()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
HEARTBEAT_ALIVE_SEC = 15


def cpu_temperature() -> Optional[float]:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read()) / 1000, 1)
    except (OSError, ValueError):
        return None


def load_history(log_dir: str, limit: int) -> List[str]:
    """
    :return: The last `limit` events (JSON strings) from the newest events.jsonl files
    """
    lines: List[str] = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*", "events.jsonl")), reverse=True):
        try:
            with open(path, encoding="utf-8") as f:
                lines = [line.rstrip("\n") for line in f if line.strip()] + lines
        except OSError:
            continue
        if len(lines) >= limit:
            break
    return lines[-limit:]


class DashboardState:
    """Thread-safe-enough shared state: written by the module thread, read by the web server thread."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.heartbeats: Dict[str, Dict[str, Any]] = {}
        self.mic_state: Optional[str] = None
        self.started = time.time()
        self.clients: Set[Any] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.recent: Deque[str] = collections.deque(maxlen=500)
        self.registry = None  # core.identity.IdentityRegistry (F9), set by the module

    def on_event(self, data: str) -> None:
        """Called in the module thread for every forwarded event (JSON string)."""
        try:
            event = Event.from_json(data)
        except ValueError:
            return
        if event.type == "heartbeat":
            self.heartbeats[event.payload.get("module", "?")] = {**event.payload, "at": time.time()}
            return
        if event.type == "mic_state":
            self.mic_state = event.payload.get("state")
        elif event.type.startswith("identity_") and self.registry is not None:
            self.registry.poll()
        self.recent.append(data)
        if self.loop is not None and self.clients:
            asyncio.run_coroutine_threadsafe(self.broadcast(data), self.loop)

    async def broadcast(self, data: str) -> None:
        for websocket in list(self.clients):
            try:
                await websocket.send_text(data)
            except Exception:
                self.clients.discard(websocket)

    def status(self) -> Dict[str, Any]:
        import psutil

        now = time.time()
        modules = {name: {**{k: v for k, v in hb.items() if k not in ("module", "at")},
                          "alive": now - hb["at"] <= HEARTBEAT_ALIVE_SEC, "age_sec": round(now - hb["at"], 1)}
                   for name, hb in sorted(self.heartbeats.items())}
        memory = psutil.virtual_memory()
        return {"modules": modules,
                "mic_state": self.mic_state or self.heartbeats.get("audio", {}).get("mic_state"),
                "video_fps": self.heartbeats.get("vision", {}).get("frame_fps"),
                "yolo_fps": self.heartbeats.get("vision", {}).get("yolo_fps"),
                "cpu_temp_c": cpu_temperature(), "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_used_mb": round((memory.total - memory.available) / 1e6),
                "ram_total_mb": round(memory.total / 1e6),
                "uptime_sec": round(now - self.started),
                "gas_thresholds": self.config.get("gas", {}).get("thresholds", {})}


def placeholder_jpeg(text: str = "no camera") -> bytes:
    import cv2
    import numpy as np

    image = np.full((240, 320, 3), 30, np.uint8)
    cv2.putText(image, text, (90, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2)
    return cv2.imencode(".jpg", image)[1].tobytes()


async def mjpeg_frames(vision: Dict[str, Any], fps: float, max_frames: Optional[int] = None):
    """
    MJPEG parts from the vision shared buffer: only new frames, at most `fps` per second; a placeholder until
    the camera has written its first frame.

    :param max_frames: stop after this many parts (tests); endless when None
    """
    from modules.vision.shared_frame import SharedFrameBuffer

    buffer, last_id, idle, sent = None, 0, placeholder_jpeg(), 0
    width, height = vision.get("resolution", [640, 480])
    try:
        while max_frames is None or sent < max_frames:
            jpeg = None
            if buffer is None and vision.get("shared_buffer"):
                try:
                    buffer = SharedFrameBuffer.attach(vision["shared_buffer"], width, height)
                except FileNotFoundError:
                    buffer = None
            if buffer is not None:
                frame = buffer.read_annotated()
                if frame is not None and frame[0] != last_id:
                    last_id, jpeg = frame[0], frame[2]
            if jpeg is None and last_id == 0:
                jpeg = idle
            if jpeg is not None:
                sent += 1
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode()
                       + b"\r\n\r\n" + jpeg + b"\r\n")
            await asyncio.sleep(1.0 / fps)
    finally:
        if buffer is not None:
            buffer.close()


def person_json(person: Dict[str, Any]) -> Dict[str, Any]:
    def iso(ts):
        return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if ts else None
    return {"person_id": person["person_id"], "display_name": person["display_name"], "is_known": person["is_known"],
            "face_samples": person["face_samples"], "voice_samples": person["voice_samples"],
            "has_thumbnail": person["has_thumbnail"], "created_at": iso(person["created_at"]),
            "last_seen": iso(person["last_seen"]), "last_location": person["last_location"]}


def create_app(state: DashboardState, config: Dict[str, Any]):
    from fastapi import Body, Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
    from fastapi.responses import FileResponse, Response, StreamingResponse
    from fastapi.security import HTTPBasic, HTTPBasicCredentials

    from core.identity import NameTaken

    cfg = config.get("dashboard", {})
    user, password = cfg.get("user"), cfg.get("password")
    fps = float(cfg.get("dashboard_fps", 8))
    history = int(cfg.get("history_events", 200))
    log_dir = config.get("logging", {}).get("dir", "logs")
    aggregator = config.get("aggregator", {})
    export_dir = aggregator.get("export_dir", "exports")
    outbox_path = aggregator.get("outbox_path", os.path.join(export_dir, "outbox.db"))
    incidents_dir = (aggregator.get("incidents") or {}).get("dir", os.path.join(export_dir, "incidents"))
    vision = config.get("vision", {})
    security = HTTPBasic(auto_error=False)

    def valid(username: str, secret: str) -> bool:
        return bool(user and password) and secrets.compare_digest(username.encode(), str(user).encode()) \
            and secrets.compare_digest(secret.encode(), str(password).encode())

    def auth(credentials: Annotated[Optional[HTTPBasicCredentials], Depends(security)] = None) -> None:
        if not (user and password):
            return  # no credentials configured: the server is bound to 127.0.0.1 only
        if credentials is None or not valid(credentials.username, credentials.password):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": 'Basic realm="dashboard"'})

    app = FastAPI(title="IoT security dashboard", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", dependencies=[Depends(auth)])
    async def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/api/status", dependencies=[Depends(auth)])
    async def api_status():
        return state.status()

    @app.get("/api/snapshot/{snapshot_id}", dependencies=[Depends(auth)])
    async def api_snapshot(snapshot_id: str):
        if not os.path.exists(outbox_path):
            raise HTTPException(404, "no outbox yet")
        with sqlite3.connect(f"file:{outbox_path}?mode=ro", uri=True) as db:
            row = db.execute("SELECT payload_json, digest, status, trigger FROM outbox WHERE id = ?",
                             (snapshot_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "unknown snapshot")
        return {"id": snapshot_id, "status": row[2], "trigger": row[3], "digest_sent_to_gpt": row[1],
                "snapshot": json.loads(row[0])}

    @app.get("/api/incident/{incident_id}", dependencies=[Depends(auth)])
    async def api_incident(incident_id: str):
        name = os.path.basename(incident_id)
        path = os.path.join(incidents_dir, f"{name}.zip")
        if not os.path.isfile(path):
            raise HTTPException(404, "unknown incident (still recording, or deleted by the size cap)")
        return FileResponse(path, media_type="application/zip", filename=f"{name}.zip")

    # --- identity registry (F9): the single place to rename / merge / delete people ---------------------------------

    def registry():
        if state.registry is None:
            raise HTTPException(503, "identity registry not available")
        state.registry.poll()
        return state.registry

    def person_or_404(person_id: str) -> str:
        pid = registry().resolve(person_id)
        if pid is None:
            raise HTTPException(404, "unknown person")
        return pid

    @app.get("/api/persons", dependencies=[Depends(auth)])
    def api_persons():
        reg = registry()
        aliases = {pid: reg.resolve(pid) for pid, p in reg.persons.items() if p["merged_into"] and reg.resolve(pid)}
        return {"persons": [person_json(p) for p in reg.list_persons()], "aliases": aliases}

    @app.get("/api/persons/{person_id}/thumbnail", dependencies=[Depends(auth)])
    def api_thumbnail(person_id: str):
        path = registry().thumbnail(person_or_404(person_id))
        if path is None:
            return Response(placeholder_jpeg("no photo"), media_type="image/jpeg")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.patch("/api/persons/{person_id}", dependencies=[Depends(auth)])
    def api_rename(person_id: str, display_name: Annotated[str, Body(embed=True)]):
        pid = person_or_404(person_id)
        try:
            registry().rename(pid, display_name)
        except NameTaken as e:
            raise HTTPException(409, str(e)) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return person_json(registry().get(pid))

    @app.post("/api/persons/{person_id}/merge", dependencies=[Depends(auth)])
    def api_merge(person_id: str, target_id: Annotated[str, Body(embed=True)]):
        source, target = person_or_404(person_id), person_or_404(target_id)
        try:
            registry().merge(source, target)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return person_json(registry().get(target))

    @app.delete("/api/persons/{person_id}", dependencies=[Depends(auth)])
    def api_delete(person_id: str):
        registry().delete(person_or_404(person_id))
        return {"deleted": person_id}

    @app.get("/video", dependencies=[Depends(auth)])
    async def video():
        return StreamingResponse(mjpeg_frames(vision, fps), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        if user and password:
            header = websocket.headers.get("authorization", "")
            try:
                username, _, secret = base64.b64decode(header.split(" ", 1)[1]).decode().partition(":")
            except Exception:
                username = secret = ""
            if not valid(username, secret):
                await websocket.close(code=1008)
                return
        await websocket.accept()
        state.loop = asyncio.get_running_loop()  # events from the module thread are scheduled on this loop
        for data in load_history(log_dir, history):
            await websocket.send_text(data)
        state.clients.add(websocket)
        try:
            while True:
                await websocket.receive_text()  # keeps the connection open; the page sends nothing
        except WebSocketDisconnect:
            pass
        finally:
            state.clients.discard(websocket)

    return app


class DashboardModule(BaseModule):
    name = "dashboard"
    source = "system"

    def __init__(self, event_queue, control_queue, config: Dict[str, Any]):
        super().__init__(event_queue, control_queue, config)
        self.state = DashboardState(config)
        self.server = None
        self.thread: Optional[threading.Thread] = None

    def setup(self) -> None:
        import uvicorn

        cfg = self.config.get("dashboard", {})
        host = cfg.get("host", "0.0.0.0")
        if not (cfg.get("user") and cfg.get("password")):
            host = "127.0.0.1"
            logger.warning("DASHBOARD_USER / DASHBOARD_PASS not set: dashboard only on http://127.0.0.1 (set them "
                           "in .env to open it from the local network)")
        try:
            from core.identity import registry_from_config

            self.state.registry = registry_from_config(self.config.get("identity", {}), publish=self.publish)
        except Exception as e:  # the rest of the dashboard works without it
            logger.error(f"Identity registry unavailable: {e}")
        app = create_app(self.state, self.config)
        self.server = uvicorn.Server(uvicorn.Config(app, host=host, port=int(cfg.get("port", 8000)),
                                                    log_level="warning", access_log=False))
        self.thread = threading.Thread(target=self.server.run, name="uvicorn", daemon=True)
        self.thread.start()
        logger.warning(f"Dashboard on http://{host}:{cfg.get('port', 8000)}")

    def handle_control(self, message: Any) -> None:
        if isinstance(message, str) and message.startswith("{"):
            self.state.on_event(message)
        else:
            super().handle_control(message)

    def step(self) -> None:
        if self.thread is not None and not self.thread.is_alive():
            logger.error("Dashboard web server stopped unexpectedly")
            self.running = False  # the orchestrator restarts the module
        time.sleep(0.05)

    def teardown(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=5)
        if self.state.registry is not None:
            self.state.registry.close()


def run(event_queue, control_queue, config):
    """Module entry point for the orchestrator (own process; all events + heartbeats forwarded)."""
    DashboardModule(event_queue, control_queue, config).run()
