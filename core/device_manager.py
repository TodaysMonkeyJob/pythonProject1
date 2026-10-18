"""
Runs one pipeline per discovered device (F11 / TOD-22), inside the orchestrator process.

A background thread re-scans every ``network.discovery_interval_sec`` (10 s); ``tick()`` (main loop) applies the
diff: a new / re-plugged camera gets its own ``vision@<device_id>`` process (capture + motion + YOLO, own shared frame
buffer), a microphone an ``audio@<device_id>`` mic controller; an unplugged device is stopped and marked offline; a
device whose OpenCV index shifted is restarted. The pooled F2 / F3 consumers (actions, identity) read every camera
buffer. Devices disabled on the dashboard are not started; ``device_updated`` events start / stop them at once.
"""
import copy
import queue
import threading
from typing import Any, Callable, Dict, List, Optional

from core import discovery
from core.devices import DeviceRegistry
from core.events import Event
from system_setup.logger_setup import init_logger

logger = init_logger()


HUB_PROCESSED_REMOTE = ("browser", "rtsp")   # remote devices whose frames / audio the hub processes


def buffer_name(device_id: str) -> str:
    """Shared memory name for a camera (macOS limit: ≤ 30 chars)."""
    return f"iotv_{device_id.split('-')[-1][:12]}"


def device_overrides(device: Dict[str, Any], params: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Child config sections for the pipeline of one device (merged over the normal child config)."""
    ref = {k: device[k] for k in ("device_id", "node_id", "kind", "transport", "name", "location_id")}
    if device["kind"] == "camera":
        source = params.get("source", 0)
        picamera = isinstance(source, str) and source.startswith("picamera2")
        ingest = isinstance(source, str) and source.startswith("ingest:")
        camera = "picamera2" if picamera else "ingest" if ingest else "opencv"
        return {"_device": ref, "vision": {"device": 0 if picamera else source[len("ingest:"):] if ingest else source,
                                           "camera": camera, "shared_buffer": buffer_name(device["device_id"])}}
    if device["kind"] == "mic":
        return {"_device": ref, "audio": {"mic": {"device": params.get("device"), "device_id": device["device_id"]}}}
    return {"_device": ref}


def merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        out[key] = merge(out.get(key, {}), value) if isinstance(value, dict) and isinstance(out.get(key), dict) \
            else value
    return out


class DeviceManager:
    def start_remote(self) -> None:
        """At start: RTSP cameras added earlier (they have no discovery; browser nodes reconnect on their own)."""
        for device in self.registry.list_devices(kind="camera"):
            if device["transport"] == "rtsp" and device["enabled"]:
                self.registry.seen("camera", "rtsp", device["name"], device["hw_key"], device["info"],
                                   node_id=device["node_id"])
                self._sync(self.registry.get(device["device_id"]), device["info"])

    def __init__(self, orchestrator, config: Dict[str, Any], scan: Callable[[Dict[str, Any]], List] = None,
                 registry: Optional[DeviceRegistry] = None):
        """
        :param orchestrator: Has add_module(spec, buffer=None), remove_module(name), emit(event)
        :param scan: network config → [FoundDevice] (tests inject a fake)
        """
        self.orchestrator = orchestrator
        self.config = config
        self.cfg = config.get("network", {})
        self.scan = scan or discovery.scan
        self.registry = registry or DeviceRegistry(self.cfg.get("devices_path", "data/devices.db"),
                                                   self.cfg.get("node_id"), publish=orchestrator.emit)
        self.interval = float(self.cfg.get("discovery_interval_sec", 10))
        self.found: Dict[str, discovery.FoundDevice] = {}          # hw_key → device of the last applied scan
        self.devices: Dict[str, str] = {}                           # hw_key → device_id
        self.pipelines: Dict[str, str] = {}                         # device_id → module name
        self.results: "queue.Queue[List]" = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Devices that were online when the hub stopped are offline until seen again
        for device in self.registry.list_devices(status="online"):
            if device["kind"] != "gas" and (device["node_id"] == self.registry.node_id
                                            or device["transport"] in HUB_PROCESSED_REMOTE):
                self.registry.offline(device["device_id"])

    # --- scanning -----------------------------------------------------------------------------------------------

    def start(self) -> None:
        self.apply(self.scan(self.cfg))  # first scan synchronously: pipelines start with the hub
        self.start_remote()
        self._thread = threading.Thread(target=self._loop, name="discovery", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                found = self.scan(self.cfg)
            except Exception as e:  # discovery must never take the hub down
                logger.warning(f"Discovery scan failed: {e}")
                continue
            try:
                self.results.put_nowait(found)
            except queue.Full:
                pass

    def tick(self) -> None:
        """Main loop: applies the latest finished scan."""
        try:
            self.apply(self.results.get_nowait())
        except queue.Empty:
            pass

    # --- applying -----------------------------------------------------------------------------------------------

    def apply(self, found: List[discovery.FoundDevice]) -> discovery.Diff:
        change = discovery.diff(self.found.values(), found)
        for gone in change.removed:
            device_id = self.devices.get(gone.hw_key)
            self._stop_pipeline(device_id)
            if device_id:
                self.registry.offline(device_id)
                logger.warning(f"Device gone: {gone.kind} {gone.name}")
        for item in change.added + change.changed:
            seed = self._seed(item.name)
            device, _ = self.registry.seen(item.kind, item.transport, item.name, item.hw_key, item.params,
                                           location_id=seed.get("location"), enabled=seed.get("enabled"))
            self.devices[item.hw_key] = device["device_id"]
            if item in change.changed:
                self._stop_pipeline(device["device_id"])
            self._sync(device, item.params)
            logger.warning(f"Device {'changed' if item in change.changed else 'found'}: {item.kind} {item.name} "
                           f"({device['device_id']}, {device['location_id']})")
        self.found = {d.hw_key: d for d in found}
        return change

    def on_event(self, event: Event) -> None:
        """
        ``device_updated`` from the dashboard: enabled toggled → start / stop the pipeline. Devices of remote thin
        nodes processed here (a phone's browser camera / mic, an RTSP camera — registered by the dashboard) start on
        ``device_online`` and stop on ``device_offline``. Edge nodes (a Pi) process their devices themselves.
        """
        if not event.type.startswith("device_"):
            return
        device = self.registry.get(event.payload.get("device_id", ""))
        if device is None:
            return
        if device["transport"] in HUB_PROCESSED_REMOTE:
            if event.type == "device_offline":
                self._stop_pipeline(device["device_id"])
            else:
                self._sync(device, device["info"])
            return
        if event.type != "device_updated" or device["node_id"] != self.registry.node_id:
            return
        found = next((f for f in self.found.values() if self.devices.get(f.hw_key) == device["device_id"]), None)
        if found is not None:
            self._sync(device, found.params)

    def _sync(self, device: Dict[str, Any], params: Dict[str, Any]) -> None:
        running = device["device_id"] in self.pipelines
        if device["enabled"] and device["status"] == "online" and not running:
            self._start_pipeline(device, params)
        elif not device["enabled"] and running:
            self._stop_pipeline(device["device_id"])

    def _start_pipeline(self, device: Dict[str, Any], params: Dict[str, Any]) -> None:
        from core.orchestrator import ModuleSpec, pipeline_template

        template = pipeline_template(device["kind"], self.config)
        if template is None:
            return
        target, subscribe = template
        name = f"{'vision' if device['kind'] == 'camera' else 'audio'}@{device['device_id']}"
        overrides = device_overrides(device, params, self.config)
        spec = ModuleSpec(name, target, list(subscribe), config=overrides, device_id=device["device_id"])
        buffer = overrides.get("vision", {}).get("shared_buffer")
        self.orchestrator.add_module(spec, buffer=buffer)
        self.pipelines[device["device_id"]] = name

    def _stop_pipeline(self, device_id: Optional[str]) -> None:
        name = self.pipelines.pop(device_id, None) if device_id else None
        if name:
            self.orchestrator.remove_module(name)

    def _seed(self, name: str) -> Dict[str, Any]:
        """``network.seed``: [{match, location, enabled}] — defaults for devices seen the first time."""
        for rule in self.cfg.get("seed") or []:
            if str(rule.get("match", "")).lower() in name.lower():
                return rule
        return {}

    def register_static(self, kind: str, name: str, hw_key: str) -> Dict[str, Any]:
        """A configured device that is not discovered (the gas sensor): registered so it has an id and location."""
        device, _ = self.registry.seen(kind, "local", name, hw_key, {})
        return device

    def close(self) -> None:
        self.stop()
        if self._thread is not None:
            self._thread.join(timeout=2)
        for device_id in set(self.devices.values()):  # hub stopped: nothing of this node is online any more
            self.registry.offline(device_id)
        self.registry.close()

