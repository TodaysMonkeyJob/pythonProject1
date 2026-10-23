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
  F14: ``GET /api/persons/duplicates`` (review queue), ``POST …/duplicates/scan`` (run the dedup job now),
  ``POST …/duplicates/{id}/merge`` / ``/reject`` ("not the same person" = permanent cannot-link),
  ``GET /api/identity/merges`` (history) and ``POST /api/identity/merges/{event_id}/undo``.
* ``/api/devices`` (F11) — cameras / microphones / sensors with status, location, enabled and the live state of
  their pipelines; ``PATCH /api/devices/{id}`` ``{location_id, name, enabled}``, ``POST /api/locations`` ``{name}``.
  ``/video?device=<id>`` streams one camera.
* F12 (TOD-23): ``/api/locations`` CRUD (+ ``/reorder``, ``/{id}/summary``, ``/{id}/ack``) — live summaries come as
  ``location_summary`` events and are pushed to the browsers; ``/api/notifications`` (``?unacked=true``,
  ``POST /{id}/ack``). Acknowledging publishes ``notification_acked`` (the locations module clears the alerts).

HTTP Basic auth with ``DASHBOARD_USER`` / ``DASHBOARD_PASS`` (.env). Without credentials the server only listens on
127.0.0.1. F11: a second HTTPS server (``network.node_port``, 8443, self-signed certificate from the system openssl)
serves remote nodes (``dashboard/node.py``: phone page ``/node``, Pi agents) on the LAN with pairing tokens; on that
port the dashboard itself needs the credentials. The web servers run in background threads; the module loop keeps
sending heartbeats and handles "stop", so a closed browser tab never affects other modules.
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


def reverse_lines(path: str, chunk: int = 1 << 20):
    """Lines of a file from the last one to the first, read in chunks (big event logs are not loaded whole)."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        position, rest = f.tell(), b""
        while position > 0:
            step = min(chunk, position)
            position -= step
            f.seek(position)
            lines = (f.read(step) + rest).split(b"\n")
            rest = lines[0]
            for line in reversed(lines[1:]):
                if line.strip():
                    yield line.decode("utf-8", "replace")
        if rest.strip():
            yield rest.decode("utf-8", "replace")


def query_history(log_dir: str, location_id: Optional[str] = None, types: Optional[List[str]] = None,
                  since: Optional[str] = None, limit: int = 1000, max_days: int = 2) -> List[Dict[str, Any]]:
    """
    F13 details pages: events of a location from ``logs/<day>/events.jsonl``, newest first.

    :param types: event types to keep (cheap substring pre-filter before parsing)
    :param since: ISO timestamp; older events end the scan
    """
    needles = [f'"type":"{t}"' for t in types or []]
    out: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*", "events.jsonl")), reverse=True)[:max_days]:
        for line in reverse_lines(path):
            if needles and not any(n in line for n in needles):
                continue
            if location_id and f'"location_id":"{location_id}"' not in line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if since and event.get("timestamp", "") < since:
                return out
            if location_id and event.get("location_id") != location_id:
                continue
            out.append(event)
            if len(out) >= limit:
                return out
    return out


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
        self.devices = None   # core.devices.DeviceRegistry (F11), set by the module
        self.locations = None      # core.locations.LocationRegistry (F12)
        self.notifications = None  # modules.notify.service.NotificationService (F12; list / acknowledge)
        self.summaries: Dict[str, Dict[str, Any]] = {}   # location_id → latest location_summary
        self.publish = None        # BaseModule.publish of the dashboard module
        from dashboard.node import NodeHub

        self.node_hub = NodeHub(self, config)

    def on_event(self, data: str) -> None:
        """Called in the module thread for every forwarded event (JSON string)."""
        try:
            event = Event.from_json(data)
        except ValueError:
            return
        if event.type == "heartbeat":
            self.heartbeats[event.payload.get("module", "?")] = {**event.payload, "at": time.time()}
            return
        if event.type == "location_summary":  # live, every ~2 s: pushed, not kept in the event history
            self.summaries[event.payload["location_id"]] = event.payload
            if self.loop is not None and self.clients:
                asyncio.run_coroutine_threadsafe(self.broadcast(data), self.loop)
            return
        if event.type == "mic_state":
            self.mic_state = event.payload.get("state")
        elif event.type.startswith("identity_") and self.registry is not None:
            self.registry.poll()
        elif event.type == "device_updated":
            self.node_hub.forward_to_edge(event)  # a location / enabled change for a Pi's device
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
                "video_fps": self._vision().get("frame_fps"),
                "yolo_fps": self._vision().get("yolo_fps"),
                "cpu_temp_c": cpu_temperature(), "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_used_mb": round((memory.total - memory.available) / 1e6),
                "ram_total_mb": round(memory.total / 1e6),
                "uptime_sec": round(now - self.started),
                "gas_thresholds": self.config.get("gas", {}).get("thresholds", {})}


    def _vision(self) -> Dict[str, Any]:
        """Heartbeat of the vision module (single camera) or the first camera pipeline (``vision@<device>``)."""
        if "vision" in self.heartbeats:
            return self.heartbeats["vision"]
        return next((hb for name, hb in sorted(self.heartbeats.items()) if name.startswith("vision@")), {})


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
    from fastapi import Body, Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
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

    def auth(request: Request,
             credentials: Annotated[Optional[HTTPBasicCredentials], Depends(security)] = None) -> None:
        if not (user and password):
            if request.url.scheme == "https":  # the LAN-facing node server: never an open dashboard
                raise HTTPException(status.HTTP_401_UNAUTHORIZED, "set DASHBOARD_USER / DASHBOARD_PASS in .env to "
                                    "use the dashboard over the network")
            return  # no credentials configured: the HTTP server is bound to 127.0.0.1 only
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

    # --- F14: possible duplicates (review queue of modules.vision.dedup) + merge history with undo ------------------

    def duplicate_json(reg, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        a, b = reg.get(candidate["person_a"]), reg.get(candidate["person_b"])
        if a is None or b is None:
            return None
        return {"id": candidate["id"], "score": candidate["score"], "evidence": candidate["evidence"],
                "a": person_json(a), "b": person_json(b)}

    @app.get("/api/persons/duplicates", dependencies=[Depends(auth)])
    def api_duplicates():
        reg = registry()
        items = [d for d in (duplicate_json(reg, c) for c in reg.candidates()) if d]
        return {"duplicates": items, "count": len(items)}

    @app.post("/api/persons/duplicates/scan", dependencies=[Depends(auth)])
    def api_duplicates_scan():
        from modules.vision.dedup import DedupConfig, DedupJob

        reg = registry()
        cfg = DedupConfig.from_dict(config.get("identity", {}).get("dedup"))
        report = DedupJob(reg, cfg, publish=reg.publish).run()
        return {"merged": len(report["merged"]), "queued": len(report["queued"]), "deleted": len(report["deleted"])}

    @app.post("/api/persons/duplicates/{candidate_id}/merge", dependencies=[Depends(auth)])
    def api_duplicate_merge(candidate_id: int, request: Request):
        try:
            event = registry().accept_candidate(candidate_id, by=user_of(request))
        except KeyError as e:
            raise HTTPException(404, "no such pending duplicate") from e
        return {"merged_into": event.payload["person_id"], "event_id": event.payload.get("event_id")}

    @app.post("/api/persons/duplicates/{candidate_id}/reject", dependencies=[Depends(auth)])
    def api_duplicate_reject(candidate_id: int, request: Request):
        try:
            registry().reject_candidate(candidate_id, by=user_of(request))
        except KeyError as e:
            raise HTTPException(404, "no such pending duplicate") from e
        return {"rejected": candidate_id}

    @app.get("/api/identity/merges", dependencies=[Depends(auth)])
    def api_merges(limit: int = 30):
        reg = registry()
        return {"merges": [{"event_id": m["event_id"], "ts": m["ts"], "action": m["action"],
                            "target_id": m["person_id"], "target_name": reg.name(m["person_id"], m.get("target_name")),
                            "source_name": m.get("source_name"), "score": m.get("score"),
                            "moved": len(m.get("moved_face", [])), "dropped": len(m.get("dropped_face", []))
                            + len(m.get("evicted_face", [])), "undone": m["undone"], "undoable": m["undoable"]}
                           for m in reg.merges(min(max(1, limit), 200))]}

    @app.post("/api/identity/merges/{event_id}/undo", dependencies=[Depends(auth)])
    def api_merge_undo(event_id: int, request: Request):
        try:
            event = registry().unmerge(event_id, by=user_of(request))
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        return {"restored": event.payload["source_id"], "target": event.payload["person_id"]}

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

    # --- devices (F11) ------------------------------------------------------------------------------------------

    def devices():
        if state.devices is None:
            raise HTTPException(503, "device registry not available")
        return state.devices

    def device_json(device: Dict[str, Any]) -> Dict[str, Any]:
        module = f"{'vision' if device['kind'] == 'camera' else 'audio'}@{device['device_id']}"
        heartbeat = state.heartbeats.get(module) or (state.heartbeats.get("gas") if device["kind"] == "gas" else None)
        alive = bool(heartbeat) and time.time() - heartbeat["at"] <= HEARTBEAT_ALIVE_SEC
        return {**{k: device[k] for k in ("device_id", "node_id", "kind", "transport", "name", "location_id",
                                          "enabled", "status", "last_seen")},
                "running": alive, "fps": (heartbeat or {}).get("frame_fps"),
                "mic_state": (heartbeat or {}).get("mic_state")}

    @app.get("/api/devices", dependencies=[Depends(auth)])
    def api_devices():
        reg = devices()
        from core.tls import lan_ip

        port = config.get("network", {}).get("node_port", 8443)
        return {"devices": [device_json(d) for d in reg.list_devices()], "locations": reg.list_locations(),
                "node_url": f"https://{lan_ip()}:{port}/node" if port else None}

    @app.patch("/api/devices/{device_id}", dependencies=[Depends(auth)])
    def api_device_update(device_id: str, location_id: Annotated[Optional[str], Body()] = None,
                          name: Annotated[Optional[str], Body()] = None,
                          enabled: Annotated[Optional[bool], Body()] = None):
        try:
            return device_json(devices().update(device_id, location_id=location_id, name=name, enabled=enabled))
        except KeyError as e:
            raise HTTPException(404, "unknown device") from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    # --- locations + notifications (F12) -------------------------------------------------------------------------

    def locations():
        if state.locations is None:
            raise HTTPException(503, "location registry not available")
        return state.locations

    def location_json(location: Dict[str, Any]) -> Dict[str, Any]:
        summary = state.summaries.get(location["location_id"], {})
        return {**location, "state": summary.get("state", "safe"), "summary": summary or None}

    def user_of(request: Request) -> str:
        header = request.headers.get("authorization", "")
        try:
            return base64.b64decode(header.split(" ", 1)[1]).decode().partition(":")[0] or "dashboard"
        except Exception:
            return "dashboard"

    def acked(location_id: str, by: str, notification_id: Optional[str] = None) -> None:
        if state.publish is not None:
            from core.events import make_event

            state.publish(make_event("system", "notification_acked", {"location_id": location_id, "by": by,
                                                                       "id": notification_id},
                                     location_id=location_id))

    @app.get("/api/locations", dependencies=[Depends(auth)])
    def api_locations():
        locations().poll()
        return {"locations": [location_json(loc) for loc in locations().list()]}

    @app.post("/api/locations", dependencies=[Depends(auth)])
    def api_add_location(name: Annotated[str, Body()], icon: Annotated[str, Body()] = "",
                         order: Annotated[Optional[int], Body()] = None,
                         rules: Annotated[Optional[Dict[str, Any]], Body()] = None):
        try:
            return location_json(locations().create(name, order, icon, rules))
        except ValueError as e:
            raise HTTPException(422 if not name.strip() else 409, str(e)) from e

    @app.patch("/api/locations/{location_id}", dependencies=[Depends(auth)])
    def api_update_location(location_id: str, name: Annotated[Optional[str], Body()] = None,
                            icon: Annotated[Optional[str], Body()] = None,
                            order: Annotated[Optional[int], Body()] = None,
                            rules: Annotated[Optional[Dict[str, Any]], Body()] = None):
        try:
            return location_json(locations().update(location_id, name, order, icon, rules))
        except KeyError as e:
            raise HTTPException(404, "unknown location") from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.delete("/api/locations/{location_id}", dependencies=[Depends(auth)])
    def api_delete_location(location_id: str):
        try:
            moved = locations().delete(location_id)
        except KeyError as e:
            raise HTTPException(404, "unknown location") from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        state.summaries.pop(location_id, None)
        return {"deleted": location_id, "devices_moved_to_unassigned": moved}

    @app.post("/api/locations/reorder", dependencies=[Depends(auth)])
    def api_reorder(order: Annotated[List[str], Body(embed=True)]):
        return {"locations": [location_json(loc) for loc in locations().reorder(order)]}

    @app.get("/api/locations/{location_id}/summary", dependencies=[Depends(auth)])
    def api_location_summary(location_id: str):
        if location_id not in state.summaries:
            raise HTTPException(404, "no summary yet")
        return state.summaries[location_id]

    @app.post("/api/locations/{location_id}/ack", dependencies=[Depends(auth)])
    def api_ack_location(location_id: str, request: Request):
        by = user_of(request)
        count = state.notifications.ack_location(location_id, by) if state.notifications is not None else 0
        acked(location_id, by)
        return {"acknowledged": location_id, "notifications": count, "by": by}

    @app.get("/api/notifications", dependencies=[Depends(auth)])
    def api_notifications(unacked: bool = False, limit: int = 100):
        if state.notifications is None:
            raise HTTPException(503, "notifications not available")
        return {"notifications": state.notifications.list(unacked=unacked, limit=min(limit, 500))}

    @app.post("/api/notifications/{notification_id}/ack", dependencies=[Depends(auth)])
    def api_ack_notification(notification_id: str, request: Request):
        if state.notifications is None:
            raise HTTPException(503, "notifications not available")
        by = user_of(request)
        note = state.notifications.ack(notification_id, by)
        if note is None:
            raise HTTPException(404, "unknown notification")
        acked(note["location_id"], by, notification_id)
        return note

    # --- trust registry (F23 / TOD-34): read-only here, the router writes; feedback goes back as an event --------
    registry_db = config.get("trust", {}).get("registry_db", "data/trust_registry.db")

    def trust_db() -> str:
        if not os.path.isfile(registry_db):
            raise HTTPException(503, "trust registry not available yet (the orchestrator creates it)")
        return registry_db

    @app.get("/api/trust", dependencies=[Depends(auth)])
    def api_trust():
        from modules.trust.registry import read_channels

        return {"channels": read_channels(trust_db())}

    @app.get("/api/trust/history", dependencies=[Depends(auth)])
    def api_trust_history(channel_id: str, limit: int = 500):
        from modules.trust.registry import read_history

        return {"channel_id": channel_id, "history": read_history(trust_db(), channel_id, min(limit, 5000))}

    @app.post("/api/trust/feedback", dependencies=[Depends(auth)])
    def api_trust_feedback(body: Dict[str, Any], request: Request):
        channel_id = str(body.get("channel_id") or "")
        verdict = body.get("verdict")
        if "/" not in channel_id or (verdict not in ("confirm", "refute") and body.get("corrected") is None):
            raise HTTPException(400, "channel_id and verdict confirm / refute (or original + corrected) required")
        if state.publish is None:
            raise HTTPException(503, "event bus not available")
        from core.events import make_event

        payload = {"channel_id": channel_id, "verdict": verdict, "by": user_of(request),
                   "reason": str(body.get("reason") or "")[:200]}
        for key in ("original", "corrected", "event_id"):
            if body.get(key) is not None:
                payload[key] = str(body[key])[:2000]
        state.publish(make_event("system", "trust_feedback", payload))
        return {"accepted": True, **payload}

    # --- F26 (TOD-37): rule governance, read-only (changes go through python -m tools.rules approve) -------------
    rules_cfg = config.get("rules") or {}

    @app.get("/api/rules", dependencies=[Depends(auth)])
    def api_rules():
        from core.fuzzy import load_yaml
        from core.fuzzy.governance import rules_table, short, version_hash
        from modules.llm.rule_stats import read_stats

        fuzzy = config.get("trust", {}).get("fuzzy_config", "config/fuzzy.yaml")
        data = load_yaml(fuzzy)
        version = version_hash(data)
        rules_dir = rules_cfg.get("dir", "data/rules")
        changelog = []
        path = os.path.join(rules_dir, "changelog.jsonl")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                changelog = [json.loads(line) for line in f if line.strip()]
        stats = {r["rule_id"]: r for r in read_stats(rules_cfg.get("stats_db", "data/rules/firing.db"), short(version))}
        rules = [{**row, **{k: (stats.get(row["id"]) or {}).get(k, 0) for k in ("fired", "confirmed", "rejected")}}
                 for row in rules_table(data)]
        return {"version": version, "short": short(version),
                "registered": os.path.isfile(os.path.join(rules_dir, "versions", f"{version}.yaml")),
                "rules": rules, "changelog": changelog[::-1][:100]}

    # --- F27 (TOD-38): GPT rule suggestions — a separate review store, not the live feed, no events --------------
    suggestions_lock = threading.Lock()
    suggestions_state: Dict[str, Any] = {"store": None, "sending": set()}

    def suggestion_store():
        with suggestions_lock:
            if suggestions_state["store"] is None:
                from modules.rules.suggester import store_from_config

                suggestions_state["store"] = store_from_config(config)
            return suggestions_state["store"]

    def fuzzy_path() -> str:
        return config.get("trust", {}).get("fuzzy_config", "config/fuzzy.yaml")

    def review_call(fn, *args):
        from modules.rules.review import ReviewError

        try:
            return fn(*args)
        except ReviewError as e:
            raise HTTPException(409, str(e)) from e

    @app.get("/api/suggestions", dependencies=[Depends(auth)])
    def api_suggestions(status: Optional[str] = None, location: Optional[str] = None,
                        confidence: Optional[str] = None, min_support: int = 1):
        from modules.rules.review import mark_approved

        store = suggestion_store()
        mark_approved(store, rules_cfg.get("dir", "data/rules"))
        return {"suggestions": store.list(status, location, confidence, min_support),
                "sending": sorted(suggestions_state["sending"])}

    @app.get("/api/suggestions/{suggestion_id}", dependencies=[Depends(auth)])
    def api_suggestion(suggestion_id: int):
        store = suggestion_store()
        item = store.get(suggestion_id)
        if item is None:
            raise HTTPException(404, "no such suggestion")
        item["cases"] = store.outcomes(ids=item["supporting_cases"] or [-1], limit=50)
        item["sending"] = suggestion_id in suggestions_state["sending"]
        return item

    @app.post("/api/suggestions/{suggestion_id}/validate", dependencies=[Depends(auth)])
    def api_suggestion_validate(suggestion_id: int, body: Dict[str, Any]):
        from modules.rules.suggester import validate_rule

        item = api_suggestion(suggestion_id)
        return validate_rule(str(body.get("rule") or ""), str(body.get("location") or item["location"]),
                             fuzzy_path(), item.get("new_variable_needed"))

    @app.post("/api/suggestions/{suggestion_id}/reject", dependencies=[Depends(auth)])
    def api_suggestion_reject(suggestion_id: int, body: Dict[str, Any], request: Request):
        from modules.rules.review import reject

        return review_call(reject, suggestion_store(), suggestion_id, user_of(request),
                           str(body.get("reason") or ""))

    @app.post("/api/suggestions/{suggestion_id}/edit", dependencies=[Depends(auth)])
    def api_suggestion_edit(suggestion_id: int, body: Dict[str, Any], request: Request):
        from modules.rules.review import edit

        if not str(body.get("rule") or "").strip():
            raise HTTPException(400, "rule required")
        return review_call(edit, suggestion_store(), suggestion_id, str(body["rule"]), user_of(request),
                           fuzzy_path(), body.get("location") or None)

    @app.post("/api/suggestions/{suggestion_id}/send", dependencies=[Depends(auth)])
    def api_suggestion_send(suggestion_id: int, request: Request):
        """Checks + regression gate take a while: run in a thread, the page polls the suggestion."""
        from modules.rules.review import send_to_approval

        store, by = suggestion_store(), user_of(request)
        item = api_suggestion(suggestion_id)
        if not item["validator_report"].get("valid"):
            raise HTTPException(409, "the rule does not validate — edit it first")
        with suggestions_lock:
            if suggestion_id in suggestions_state["sending"]:
                return {"started": False, "sending": True}
            suggestions_state["sending"].add(suggestion_id)

        def work():
            try:
                send_to_approval(store, suggestion_id, by, fuzzy_path(), rules_cfg.get("dir", "data/rules"),
                                 rules_cfg.get("gate_dataset", "data/eval/scripted"), rules_cfg.get("gate"))
            except Exception as e:  # shown on the page through gate_report
                store.update(suggestion_id, gate_report={"passed": False, "errors": [str(e)[:500]]})
            finally:
                suggestions_state["sending"].discard(suggestion_id)

        threading.Thread(target=work, name=f"suggestion-{suggestion_id}", daemon=True).start()
        return {"started": True, "sending": True}

    # --- F25 (TOD-36): operator decision on GPT's escalation above the corridor -----------------------------------
    @app.post("/api/escalation/{review_id}", dependencies=[Depends(auth)])
    def api_escalation(review_id: str, body: Dict[str, Any], request: Request):
        if not isinstance(body.get("accepted"), bool):
            raise HTTPException(400, "accepted: true / false required")
        if state.publish is None:
            raise HTTPException(503, "event bus not available")
        review = None
        for data in reversed(state.recent):
            if review_id in data and '"escalation_review"' in data:
                candidate = Event.from_json(data)
                if candidate.type == "escalation_review" and candidate.payload.get("review_id") == review_id:
                    review = candidate
                    break
        from core.events import make_event

        location_id = (review.payload.get("location_id") if review else None) or body.get("location_id")
        payload = {"review_id": review_id, "accepted": body["accepted"], "by": user_of(request),
                   "reason": str(body.get("reason") or "")[:200],
                   "risk_level_llm": (review.payload.get("risk_level_llm") if review else None)
                   or body.get("risk_level_llm"), "location_id": location_id}
        state.publish(make_event("system", "escalation_resolved", payload,
                                 **({"location_id": location_id} if location_id else {})))
        return {"resolved": True, **payload}

    from dashboard.node import add_node_routes, preview_frames

    add_node_routes(app, state.node_hub, auth, STATIC_DIR)

    @app.get("/video", dependencies=[Depends(auth)])
    async def video(device: Optional[str] = None):
        if device and device in state.node_hub.previews:  # a Raspberry Pi camera: its previews (~ every 2 s)
            return StreamingResponse(preview_frames(state.node_hub, device, 2.0),
                                     media_type="multipart/x-mixed-replace; boundary=frame")
        source = vision
        if state.devices is not None and (device or config.get("network", {}).get("discovery")):
            from core.device_manager import buffer_name

            cameras = [d for d in state.devices.list_devices(kind="camera") if d["status"] == "online"]
            chosen = next((d for d in cameras if d["device_id"] == device), cameras[0] if cameras else None)
            if chosen is not None:
                source = {**vision, "shared_buffer": buffer_name(chosen["device_id"])}
        return StreamingResponse(mjpeg_frames(source, fps), media_type="multipart/x-mixed-replace; boundary=frame")

    def ws_allowed(websocket: WebSocket) -> bool:
        if not (user and password):
            return websocket.url.scheme != "wss"  # the LAN-facing node port never serves an open dashboard
        header = websocket.headers.get("authorization", "")
        try:
            username, _, secret = base64.b64decode(header.split(" ", 1)[1]).decode().partition(":")
        except Exception:
            return False
        return valid(username, secret)

    # --- F13 control room: static app, history for detail views, one video WebSocket per page ---------------------

    @app.get("/static/{path:path}", dependencies=[Depends(auth)])
    async def static_file(path: str):
        full = os.path.realpath(os.path.join(STATIC_DIR, path))
        if not full.startswith(os.path.realpath(STATIC_DIR) + os.sep) or not os.path.isfile(full):
            raise HTTPException(404, "not found")
        media = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}.get(os.path.splitext(full)[1])
        return FileResponse(full, media_type=media, headers={"Cache-Control": "no-cache"})

    @app.get("/api/history", dependencies=[Depends(auth)])
    def api_history(location_id: Optional[str] = None, types: Optional[str] = None, since: Optional[str] = None,
                    limit: int = 1000):
        return {"events": query_history(log_dir, location_id, [t for t in (types or "").split(",") if t], since,
                                        max(1, min(limit, 5000)))}

    from dashboard.video import FrameCache, stream

    frames = FrameCache(tuple(vision.get("resolution", [640, 480])), state.node_hub.previews)

    @app.websocket("/ws/video")
    async def video_socket(websocket: WebSocket):
        if not ws_allowed(websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            await stream(websocket, frames, float(cfg.get("grid_fps", 4)), float(cfg.get("focus_fps", 10)))
        except WebSocketDisconnect:
            pass

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        if not ws_allowed(websocket):
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
        self.node_server = None
        self.node_thread: Optional[threading.Thread] = None
        self._announce_stop = threading.Event()

    def start_node_server(self, app) -> None:
        """HTTPS for phones / Pi agents on the LAN (``network.node_port``) + UDP announcements of the hub."""
        import uvicorn

        from core.tls import ensure_certificate, lan_ip
        from dashboard.node import announce

        network = self.config.get("network", {})
        port = network.get("node_port", 8443)
        if not port:
            return
        try:
            cert, key = ensure_certificate(network.get("tls_dir", "data/tls"))
        except Exception as e:  # no openssl: the hub still works, only remote browser nodes don't
            logger.error(f"No HTTPS certificate ({e}); remote nodes disabled")
            return
        self.node_server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=int(port), ssl_certfile=cert,
                                                         ssl_keyfile=key, log_level="warning", access_log=False))
        self.node_thread = threading.Thread(target=self.node_server.run, name="uvicorn-nodes", daemon=True)
        self.node_thread.start()
        url = f"https://{lan_ip()}:{port}"
        logger.warning(f"Remote nodes: open {url}/node on a phone (pairing code on the dashboard Devices page)")
        if network.get("announce", True):
            threading.Thread(target=announce, args=(url, self.device_id.split(":")[0], self._announce_stop),
                             name="announce", daemon=True).start()

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
        try:
            from core.devices import DeviceRegistry

            network = self.config.get("network", {})
            self.state.devices = DeviceRegistry(network.get("devices_path", "data/devices.db"), network.get("node_id"),
                                                publish=self.publish)
        except Exception as e:
            logger.error(f"Device registry unavailable: {e}")
        try:
            from core.locations import LocationRegistry
            from modules.notify.service import NotificationService

            network = self.config.get("network", {})
            self.state.locations = LocationRegistry(network.get("devices_path", "data/devices.db"),
                                                    publish=self.publish)
            self.state.notifications = NotificationService.from_config(
                {**self.config.get("locations", {}), "telegram_bot_token": None})  # sending is the locations module's
        except Exception as e:
            logger.error(f"Locations / notifications unavailable: {e}")
        self.state.publish = self.publish
        self.state.node_hub.publish = self.publish
        app = create_app(self.state, self.config)
        self.start_node_server(app)
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
        self._announce_stop.set()
        if self.node_server is not None:
            self.node_server.should_exit = True
        if self.node_thread is not None:
            self.node_thread.join(timeout=5)
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=5)
        if self.state.registry is not None:
            self.state.registry.close()
        if self.state.devices is not None:
            self.state.devices.close()
        for resource in (self.state.locations, self.state.notifications):
            if resource is not None:
                resource.close()


def run(event_queue, control_queue, config):
    """Module entry point for the orchestrator (own process; all events + heartbeats forwarded)."""
    DashboardModule(event_queue, control_queue, config).run()
