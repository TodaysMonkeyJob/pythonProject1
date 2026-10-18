"""
Orchestrator (F4): runs every enabled module in its own process, routes events, restarts failed modules.

* Each module is ``"package.module:function"`` with the Step 0 signature ``run(event_queue, control_queue, config)``.
  Processes use the ``spawn`` start method (safe with torch / OpenCV threads) and ignore SIGINT; shutdown is
  coordinated here.
* Router: reads the shared event queue, writes every non-heartbeat event to ``logs/YYYY-MM-DD/events.jsonl``
  and forwards it (as the JSON string) to the control queues of modules subscribed to its type.
* Watchdog: a module whose process died, or that sent no heartbeat for ``missed_heartbeats`` × 5 s (after a
  start-up grace period), is restarted with exponential backoff; every restart is a ``system`` ``alert`` event.
* The vision shared-memory buffer is created before the modules start and unlinked on shutdown.
"""
import copy
import importlib
import multiprocessing as mp
import os
import queue
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from core.bus import CONTROL_STOP, publish
from core.events import Event, make_event
from core.logger import JsonlLogger
from core.module import HEARTBEAT_INTERVAL
from system_setup.logger_setup import init_logger

logger = init_logger()

# Registry changes (F9): modules holding person names refresh them at once
IDENTITY_EVENTS = ["identity_created", "identity_renamed", "identity_merged", "identity_linked", "identity_deleted"]
# name → (target, config section with `enabled`, event types forwarded to its control queue; "*" = all)
DEFAULT_MODULES = {
    "gas": ("modules.gas:run", "gas", []),
    "vision": ("modules.vision.detector:run", "vision", ["person_identified"]),       # names on the overlay
    "actions": ("modules.vision.actions:run", "actions", ["gas_reading", "alert", "object_detected"]),
    "identity": ("modules.vision.identity:run", "identity", ["object_detected", *IDENTITY_EVENTS]),
    "audio": ("modules.audio.mic_controller:run", "audio",
              ["scene_active", "object_detected", "person_identified", *IDENTITY_EVENTS]),
    "aggregator": ("modules.aggregator:run", "aggregator", ["*"]),
    "llm": ("modules.llm:run", "llm", [            # works from the SQLite outbox; F17 envelope needs the evidence
        "gas_reading", "motion", "object_detected", "person_identified", "action", "alert", "speech", "scene_text",
        "notification_acked"]),
    "locations": ("modules.locations:run", "locations", ["*"]),   # F12 per-location state + notifications
    "dashboard": ("dashboard:run", "dashboard", ["*", "heartbeat", "location_summary"]),
}
SELF_LOGGING_SECTIONS = ("vision", "actions", "identity")
PER_DEVICE_MODULES = ("vision", "audio")   # with network.discovery: one process per camera / microphone
UNLOGGED_EVENTS = ("heartbeat", "location_summary")   # frequent status: explicit subscribers only, not logged
# Files a module can't start without: (config section, key, how to get it)
REQUIRED_FILES = {
    "gas": [("gas", "classifier_path", "python -m modules.gas.train"),
            ("gas", "regressor_path", "python -m modules.gas.train")],
    "vision": [("vision", "model", "python -m modules.vision.export (or set vision.model: yolo11n.pt)")],
    "actions": [("actions", "pose_model",
                 "python -m modules.vision.export --model yolo11n-pose.pt (or set actions.pose_model)")],
    "identity": [("identity", "detector_model", "python -m modules.vision.face_models"),
                 ("identity", "recognizer_model", "python -m modules.vision.face_models")],
}  # their own JSONL logging is off under the orchestrator


THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def thread_env(config: Dict[str, Any], section: Optional[str]) -> Dict[str, str]:
    """
    CPU thread limits for a module process, set BEFORE torch / numpy are imported. Without them every process starts
    an OpenMP pool as big as the machine and busy-waits: 4 camera pipelines on a 10-core Mac took 570 ms per YOLO
    frame instead of 40 ms. ``<section>.threads`` or ``orchestrator.child_threads`` (default 2).
    """
    threads = (config.get(section or "", {}) or {}).get("threads") or config.get("orchestrator", {}).get(
        "child_threads", 2)
    env = {name: str(int(threads)) for name in THREAD_ENV}
    env["OMP_WAIT_POLICY"] = "PASSIVE"  # idle threads sleep instead of spinning on cores other processes need
    return env


def _child_main(target: str, event_queue, control_queue, config: Dict[str, Any]) -> None:
    """Process entry point: Ctrl+C is handled by the orchestrator, which sends "stop"."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for name, value in config.get("_thread_env", {}).items():
        os.environ.setdefault(name, value)  # an explicit OMP_NUM_THREADS from the shell wins
    from core.events import default_node_id, set_context

    network, device = config.get("network", {}), config.get("_device") or {}
    node = network.get("node_id") or default_node_id()
    set_context(node_id=node, device_id=device.get("device_id") or f"{node}:{config.get('_module', 'module')}",
                location_id=device.get("location_id") or network.get("hub_location") or "unassigned")
    module_name, _, function = target.partition(":")
    getattr(importlib.import_module(module_name), function)(event_queue, control_queue, config)


def missing_files(name: str, config: Dict[str, Any]) -> List[str]:
    """
    :return: Human-readable problems for files the module needs; bare model names (``yolo11n.pt``) are
             downloaded by ultralytics and don't count as missing
    """
    problems = []
    for section, key, hint in REQUIRED_FILES.get(name, []):
        path = config.get(section, {}).get(key)
        if not path or (os.sep not in str(path) and str(path).endswith(".pt")):
            continue
        if not os.path.exists(path):
            problems.append(f"{section}.{key} = {path} not found ({hint})")
    return problems


def target_available(target: str) -> bool:
    module_name, _, function = target.partition(":")
    try:
        return callable(getattr(importlib.import_module(module_name), function, None))
    except ImportError:
        return False


def pipeline_template(kind: str, config: Dict[str, Any]):
    """:return: (target, subscribe) of the per-device pipeline for a camera / mic, or None when that module is off"""
    name, section = {"camera": ("vision", "vision"), "mic": ("audio", "audio")}.get(kind, (None, None))
    if name is None or not config.get(section, {}).get("enabled", False):
        return None
    problems = missing_files(name, config)
    if problems:
        logger.warning(f"{name} pipelines not started: {'; '.join(problems)}")
        return None
    target, _, subscribe = DEFAULT_MODULES[name]
    override = (config.get("orchestrator", {}).get("modules", {}) or {}).get(name, {}) or {}
    return override.get("target", target), list(override.get("subscribe", subscribe))


@dataclass
class ModuleSpec:
    name: str
    target: str
    subscribe: List[str] = field(default_factory=list)
    config: Optional[Dict[str, Any]] = None     # merged over the child config (per-device pipelines)
    device_id: Optional[str] = None

    def wants(self, event_type: str) -> bool:
        return "*" in self.subscribe or event_type in self.subscribe


@dataclass
class ModuleProcess:
    spec: ModuleSpec
    control: Any
    process: Optional[mp.Process] = None
    started_at: float = 0.0
    last_heartbeat: Optional[float] = None
    restarts: int = 0
    next_start: float = 0.0


class Orchestrator:
    def __init__(self, config: Dict[str, Any], modules: Optional[Sequence[ModuleSpec]] = None,
                 clock=time.monotonic, device_scan=None):
        """
        :param config: Full merged config
        :param modules: Module specs; built from DEFAULT_MODULES + ``orchestrator.modules`` + ``enabled`` flags if None
        :param device_scan: network config → [FoundDevice]; ``core.discovery.scan`` by default (tests inject a fake)
        """
        self.device_scan = device_scan
        self.config = config
        self.cfg = config.get("orchestrator", {})
        self.ctx = mp.get_context("spawn")
        self.events = self.ctx.Queue(maxsize=int(self.cfg.get("event_queue_size", 10000)))
        self.clock = clock
        self.heartbeat_timeout = float(self.cfg.get("missed_heartbeats", 3)) * HEARTBEAT_INTERVAL
        self.startup_grace = float(self.cfg.get("startup_grace_sec", 120))
        self.max_backoff = float(self.cfg.get("max_restart_backoff_sec", 60))
        self.stop_timeout = float(self.cfg.get("stop_timeout_sec", 10))
        self.child_config = self._child_config(config)
        self.modules: Dict[str, ModuleProcess] = {
            spec.name: ModuleProcess(spec, self.ctx.Queue(maxsize=1000))
            for spec in (modules if modules is not None else self.module_specs(config))}
        log_cfg = config.get("logging", {})
        self.log = JsonlLogger("events", log_cfg.get("dir", "logs"), log_cfg.get("max_total_mb", 2048),
                               log_cfg.get("check_every", 100)) if self.cfg.get("log_events", True) else None
        self.frame_buffer = None
        self.device_buffers: Dict[str, Any] = {}    # module name → SharedFrameBuffer of its camera
        self.devices = None                         # DeviceManager (network.discovery)
        self.discovery = bool(config.get("network", {}).get("discovery", False))
        self._stopping = False
        self.routed = 0
        self.trust = self._trust_scorer(config)    # F16: trust degree on evidence events, before logging

    # --- setup -----------------------------------------------------------------------------------------------

    @staticmethod
    def module_specs(config: Dict[str, Any]) -> List[ModuleSpec]:
        overrides = config.get("orchestrator", {}).get("modules", {}) or {}
        discovery = config.get("network", {}).get("discovery", False)
        specs = []
        for name, (target, section, subscribe) in DEFAULT_MODULES.items():
            if discovery and name in PER_DEVICE_MODULES:
                continue  # one pipeline per discovered camera / microphone instead (core/device_manager.py)
            override = overrides.get(name, {}) or {}
            if override.get("enabled") is False or not config.get(section, {}).get("enabled", False):
                continue
            target = override.get("target", target)
            if not target_available(target):
                logger.info(f"Module {name} ({target}) not available yet — skipped")
                continue
            problems = missing_files(name, config) if "target" not in override else []
            if problems:
                logger.warning(f"Module {name} skipped: {'; '.join(problems)}")
                continue
            specs.append(ModuleSpec(name, target, list(override.get("subscribe", subscribe))))
        return specs

    @staticmethod
    def _child_config(config: Dict[str, Any]) -> Dict[str, Any]:
        child = copy.deepcopy(config)
        for section in SELF_LOGGING_SECTIONS:
            child.setdefault(section, {})["log_events"] = False  # the router logs every event once
        return child

    def _create_frame_buffer(self) -> None:
        vision = self.config.get("vision", {})
        if "vision" not in self.modules or not vision.get("shared_buffer"):
            return
        from modules.vision.shared_frame import SharedFrameBuffer

        width, height = vision.get("resolution", [640, 480])
        self.frame_buffer = SharedFrameBuffer.create(vision["shared_buffer"], width, height)

    # --- processes -------------------------------------------------------------------------------------------

    @staticmethod
    def _trust_scorer(config: Dict[str, Any]):
        """F16 (TOD-27) ``modules.trust.TrustScorer`` from ``trust.fuzzy_config``; None when off or misconfigured."""
        cfg = config.get("trust", {})
        if not cfg.get("enabled", True):
            return None
        path = cfg.get("fuzzy_config", "config/fuzzy.yaml")
        if not os.path.isfile(path):
            logger.warning(f"Trust scoring off: {path} not found")
            return None
        from modules.trust import TrustScorer

        return TrustScorer.from_config(path)    # a broken trust rule base fails the start loudly

    def start(self) -> None:
        self._create_frame_buffer()
        if self.discovery:
            from core.device_manager import DeviceManager

            self.devices = DeviceManager(self, self.config, scan=self.device_scan)
            if self.trust is not None:
                self.trust.state.seed_devices(self.devices.registry.list_devices())
            if "gas" in self.modules:
                sensor = self.config.get("gas", {}).get("sensor_id", "gas-1")
                device = self.devices.register_static("gas", f"Gas sensor {sensor}", f"gas:{sensor}")
                self.modules["gas"].spec.config = {"_device": {k: device[k] for k in ("device_id", "location_id")}}
                self.modules["gas"].spec.device_id = device["device_id"]
        for module in self.modules.values():
            self._start(module)
        logger.warning(f"Orchestrator started: {', '.join(self.modules) or 'no modules'}")
        if self.devices is not None:
            self.devices.start()

    def add_module(self, spec: ModuleSpec, buffer: Optional[str] = None) -> None:
        """Starts a module at runtime (a device pipeline); ``buffer`` = shared frame buffer to create for it."""
        if spec.name in self.modules:
            return
        if buffer:
            from modules.vision.shared_frame import SharedFrameBuffer

            width, height = self.config.get("vision", {}).get("resolution", [640, 480])
            try:
                SharedFrameBuffer.attach(buffer, width, height).unlink()  # stale block of a crashed run
            except FileNotFoundError:
                pass
            self.device_buffers[spec.name] = SharedFrameBuffer.create(buffer, width, height)
        module = ModuleProcess(spec, self.ctx.Queue(maxsize=1000))
        self.modules[spec.name] = module
        self._start(module)

    def remove_module(self, name: str) -> None:
        """Stops a module at runtime (device unplugged / disabled) and releases its frame buffer."""
        module = self.modules.pop(name, None)
        if module is not None and module.process is not None:
            if module.process.is_alive():
                module.control.put(CONTROL_STOP)
                module.process.join(self.stop_timeout / 2)
            self._kill(module)
            logger.info(f"Stopped {name}")
        buffer = self.device_buffers.pop(name, None)
        if buffer is not None:
            buffer.close()
            try:
                buffer.unlink()
            except FileNotFoundError:
                pass

    def _start(self, module: ModuleProcess) -> None:
        self._drain(module.control)
        config = self.child_config
        if module.spec.config:
            from core.device_manager import merge

            config = merge(config, module.spec.config)
        name = module.spec.name.split("@")[0]                       # vision@camera-… → vision
        section = DEFAULT_MODULES[name][1] if name in DEFAULT_MODULES else name
        config = {**config, "_module": module.spec.name, "_thread_env": thread_env(config, section)}
        module.process = self.ctx.Process(target=_child_main, name=module.spec.name, daemon=False,
                                          args=(module.spec.target, self.events, module.control, config))
        module.process.start()
        module.started_at = self.clock()
        module.last_heartbeat = None
        logger.info(f"Started {module.spec.name} (pid {module.process.pid})")

    def _restart(self, module: ModuleProcess, reason: str) -> None:
        self._kill(module)
        module.restarts += 1
        delay = min(self.max_backoff, 2 ** (module.restarts - 1)) if module.restarts > 1 else 0.0
        module.next_start = self.clock() + delay
        logger.warning(f"Restarting {module.spec.name} ({reason}), restart #{module.restarts}, delay {delay:.0f} s")
        self._route(make_event("system", "alert", {"reason": "module_restarted", "module": module.spec.name,
                                                   "cause": reason, "restarts": module.restarts,
                                                   "delay_sec": delay}, severity="warning").to_json())
        module.process = None

    def _kill(self, module: ModuleProcess, timeout: float = 5.0) -> None:
        process = module.process
        if process is None:
            return
        if process.is_alive():
            process.terminate()
            process.join(timeout)
        if process.is_alive():
            process.kill()
            process.join(2)

    def check_health(self) -> None:
        """Restarts dead or silent modules; starts modules whose backoff elapsed."""
        now = self.clock()
        for module in self.modules.values():
            if module.process is None:
                if now >= module.next_start and not self._stopping:
                    self._start(module)
                continue
            if not module.process.is_alive():
                self._restart(module, f"exited with code {module.process.exitcode}")
            elif module.last_heartbeat is None:
                if now - module.started_at > self.startup_grace:
                    self._restart(module, f"no heartbeat within {self.startup_grace:.0f} s of start")
            elif now - module.last_heartbeat > self.heartbeat_timeout:
                self._restart(module, f"no heartbeat for {now - module.last_heartbeat:.0f} s")
            if module.spec.device_id and module.restarts == 5 and self.devices is not None:
                self.devices.registry.offline(module.spec.device_id, status="error")  # keeps failing

    # --- routing ---------------------------------------------------------------------------------------------

    def route_pending(self, timeout: float = 0.1, limit: int = 1000) -> int:
        """Routes queued events. :return: number of events routed"""
        count = 0
        try:
            data = self.events.get(timeout=timeout)
        except queue.Empty:
            return 0
        while data is not None and count < limit:
            self._route(data)
            count += 1
            try:
                data = self.events.get_nowait()
            except queue.Empty:
                data = None
        return count

    def _route(self, data: str) -> None:
        try:
            event = Event.from_json(data)
        except ValueError as e:
            logger.warning(f"Dropping malformed event: {e}")
            return
        if event.type in UNLOGGED_EVENTS:
            module = self.modules.get(event.payload.get("module")) if event.type == "heartbeat" else None
            if module is not None:
                module.last_heartbeat = self.clock()
            for subscriber in self.modules.values():  # only explicit subscribers (the dashboard), never via "*"
                if event.type in subscriber.spec.subscribe:
                    try:
                        subscriber.control.put_nowait(data)
                    except queue.Full:
                        pass
            return
        self.routed += 1
        if self.trust is not None:
            try:
                if self.trust.apply(event):
                    data = event.to_json()
            except Exception as e:  # scoring must never stop the router
                logger.exception(f"Trust scoring failed for {event.type}: {e}")
        if self.log is not None:
            self.log.write(event)
        if event.type.startswith("device_"):  # every module may hold devices (locations, buffers to read)
            if self.devices is not None:
                self.devices.on_event(event)
            for module in self.modules.values():
                try:
                    module.control.put_nowait(data)
                except queue.Full:
                    pass
            return
        for module in self.modules.values():
            if module.spec.wants(event.type):
                try:
                    module.control.put_nowait(data)
                except queue.Full:
                    logger.warning(f"Control queue of {module.spec.name} is full; event dropped")

    def poll(self, timeout: float = 0.05) -> int:
        """One main-loop pass: routes queued events and applies the latest device scan. :return: events routed"""
        count = self.route_pending(timeout=timeout)
        if self.devices is not None:
            self.devices.tick()
        return count

    def emit(self, event: Event) -> None:
        """Publishes an orchestrator event into the router (logged and forwarded like any other)."""
        publish(self.events, event)

    # --- lifecycle -------------------------------------------------------------------------------------------

    def run(self, duration: Optional[float] = None) -> None:
        """
        Main loop until SIGINT / SIGTERM (or `duration` seconds, for tests).
        """
        def request_stop(signum, frame):
            logger.warning(f"Signal {signum} received, shutting down")
            self._stopping = True

        previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        started = self.clock()
        self.start()
        try:
            last_check = 0.0
            while not self._stopping and (duration is None or self.clock() - started < duration):
                self.poll(timeout=0.05)
                if self.clock() - last_check >= 0.5:
                    last_check = self.clock()
                    self.check_health()
        finally:
            self.shutdown()
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def shutdown(self) -> None:
        """Sends "stop" to every module, waits, then terminates stragglers; releases shared memory."""
        self._stopping = True
        if self.devices is not None:
            self.devices.stop()
        for module in self.modules.values():
            if module.process is not None and module.process.is_alive():
                module.control.put(CONTROL_STOP)
        deadline = self.clock() + self.stop_timeout
        for module in self.modules.values():
            if module.process is not None:
                module.process.join(max(0.1, deadline - self.clock()))
                if module.process.is_alive():
                    logger.warning(f"{module.spec.name} did not stop in time, terminating")
                    self._kill(module)
        while self.route_pending(timeout=0.05):  # last events (e.g. final mic_state) still get logged
            pass
        if self.devices is not None:
            manager, self.devices = self.devices, None   # routing below must not reach the closed registry
            manager.close()
            while self.route_pending(timeout=0.2):  # device_offline events
                pass
        for buffer in self.device_buffers.values():
            buffer.close()
            try:
                buffer.unlink()
            except FileNotFoundError:
                pass
        self.device_buffers.clear()
        if self.log is not None:
            self.log.close()
        if self.frame_buffer is not None:
            self.frame_buffer.close()
            try:
                self.frame_buffer.unlink()
            except FileNotFoundError:
                pass
            self.frame_buffer = None
        logger.warning("Orchestrator stopped")

    @staticmethod
    def _drain(q) -> None:
        while True:
            try:
                q.get_nowait()
            except (queue.Empty, OSError):
                return
