"""
Uplink of an edge node (F11 / TOD-22): a Raspberry Pi runs its own full pipeline and this module forwards it to the hub.

Over one WSS connection to ``<hub>/node/edge`` (token from pairing, the hub's self-signed certificate pinned at
pairing): the node's device list, every local event (heartbeats and device_* stay local), and a JPEG preview of each
camera every ``preview_sec``. The hub sends back ``device_updated`` (location / enabled chosen on the hub dashboard),
which is applied to the local device registry and so reaches the local modules. Offline: events are buffered (up to
``max_buffer``) and the connection is retried with backoff.
"""
import collections
import json
import ssl
import threading
import time
from typing import Any, Deque, Dict, List, Optional

from core.events import Event
from core.module import BaseModule
from system_setup.logger_setup import init_logger

logger = init_logger()

PREVIEW = 0x03


def ws_url(hub: str, path: str, token: str) -> str:
    base = hub.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
    return f"{base}{path}?token={token}"


def preview_message(device_id: str, jpeg: bytes) -> bytes:
    raw = device_id.encode()
    return bytes([PREVIEW]) + len(raw).to_bytes(2, "big") + raw + jpeg


class Uplink(BaseModule):
    name = "uplink"
    source = "system"

    def __init__(self, event_queue, control_queue, config: Dict[str, Any], connect=None):
        """:param connect: url → websocket-like (send / recv(timeout) / close); the websockets sync client by default"""
        super().__init__(event_queue, control_queue, config)
        self.cfg = config.get("uplink", {})
        self.connect = connect
        self.outbox: Deque[str] = collections.deque(maxlen=int(self.cfg.get("max_buffer", 5000)))
        self.preview_sec = float(self.cfg.get("preview_sec", 2.0))
        self.registry = None
        self.connected = False
        self._devices_changed = True
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def heartbeat_payload(self) -> Dict[str, Any]:
        return {"hub_connected": self.connected, "buffered": len(self.outbox)}

    def setup(self) -> None:
        from core.devices import DeviceRegistry

        network = self.config.get("network", {})
        self.registry = DeviceRegistry(network.get("devices_path", "data/devices.db"), network.get("node_id"),
                                       publish=self.publish)
        if self.connect is None:
            from websockets.sync.client import connect as ws_connect

            context = ssl.create_default_context(cafile=self.cfg.get("cafile")) if self.cfg.get("cafile") else None
            self.connect = lambda url: ws_connect(url, ssl=context, open_timeout=10, max_size=2 ** 22)
        self._thread = threading.Thread(target=self._loop, name="uplink", daemon=True)
        self._thread.start()

    def teardown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self.registry is not None:
            self.registry.close()

    def handle_control(self, message: Any) -> None:
        if not (isinstance(message, str) and message.startswith("{")):
            super().handle_control(message)
            return
        event = Event.from_json(message)
        if event.type.startswith("device_"):
            self._devices_changed = True  # the device list is sent instead (no echo loops)
        elif event.type != "heartbeat":
            with self._lock:
                self.outbox.append(json.dumps({"type": "event", "event": json.loads(message)}))

    # --- connection (own thread) -----------------------------------------------------------------------------------

    def device_list(self) -> List[Dict[str, Any]]:
        return [{k: d[k] for k in ("kind", "transport", "name", "hw_key", "location_id", "enabled")}
                for d in self.registry.list_devices(node_id=self.registry.node_id)]

    def previews(self) -> List[bytes]:
        from core.device_manager import buffer_name
        from modules.vision.shared_frame import SharedFrameBuffer

        width, height = self.config.get("vision", {}).get("resolution", [640, 480])
        out = []
        for device in self.registry.list_devices(kind="camera", status="online", node_id=self.registry.node_id):
            try:
                buffer = SharedFrameBuffer.attach(buffer_name(device["device_id"]), width, height)
            except FileNotFoundError:
                continue
            try:
                frame = buffer.read_annotated()
                if frame is not None:
                    out.append(preview_message(device["device_id"], frame[2]))
            finally:
                buffer.close()
        return out

    def apply(self, message: Dict[str, Any]) -> None:
        """A change made on the hub → the local registry (→ device_updated on the local bus)."""
        if message.get("type") == "device_updated" and self.registry.get(message.get("device_id", "")):
            self.registry.update(message["device_id"], location_id=message.get("location_id"),
                                 enabled=message.get("enabled"))

    def session(self, ws) -> None:
        """One connected session: sends until the connection breaks or the module stops."""
        next_preview = 0.0
        self._devices_changed = True
        while not self._stop.is_set():
            if self._devices_changed:
                self._devices_changed = False
                ws.send(json.dumps({"type": "devices", "devices": self.device_list()}))
            while self.outbox:
                with self._lock:
                    message = self.outbox[0]
                ws.send(message)
                with self._lock:
                    self.outbox.popleft()  # removed only after it was sent
            if time.monotonic() >= next_preview:
                next_preview = time.monotonic() + self.preview_sec
                for preview in self.previews():
                    ws.send(preview)
            try:
                incoming = ws.recv(timeout=0.1)
            except TimeoutError:
                continue
            if isinstance(incoming, str):
                self.apply(json.loads(incoming))

    def _loop(self) -> None:
        url = ws_url(self.cfg["hub"], "/node/edge", self.cfg["token"])
        backoff = 1.0
        while not self._stop.is_set():
            try:
                ws = self.connect(url)
            except Exception as e:
                logger.warning(f"Hub not reachable ({e}); retrying in {backoff:.0f} s ({len(self.outbox)} buffered)")
                self._stop.wait(backoff)
                backoff = min(60.0, backoff * 2)
                continue
            self.connected, backoff = True, 1.0
            logger.warning(f"Connected to the hub {self.cfg['hub']}")
            try:
                self.session(ws)
            except Exception as e:
                logger.warning(f"Hub connection lost: {e}")
            finally:
                self.connected = False
                try:
                    ws.close()
                except Exception:
                    pass


def run(event_queue, control_queue, config):
    Uplink(event_queue, control_queue, config).run()
