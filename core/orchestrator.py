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

# name → (target, config section with `enabled`, event types forwarded to its control queue; "*" = all)
DEFAULT_MODULES = {
    "gas": ("modules.gas:run", "gas", []),
    "vision": ("modules.vision.detector:run", "vision", []),
    "actions": ("modules.vision.actions:run", "actions", ["gas_reading", "alert", "object_detected"]),
    "identity": ("modules.vision.identity:run", "identity", ["object_detected"]),
    "audio": ("modules.audio.mic_controller:run", "audio", ["scene_active"]),
    "aggregator": ("modules.aggregator:run", "aggregator", ["*"]),
    "llm": ("modules.llm:run", "llm", ["*"]),
    "dashboard": ("modules.dashboard:run", "dashboard", ["*"]),
}
SELF_LOGGING_SECTIONS = ("vision", "actions", "identity")
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


def _child_main(target: str, event_queue, control_queue, config: Dict[str, Any]) -> None:
    """Process entry point: Ctrl+C is handled by the orchestrator, which sends "stop"."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
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


@dataclass
class ModuleSpec:
    name: str
    target: str
    subscribe: List[str] = field(default_factory=list)

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
                 clock=time.monotonic):
        """
        :param config: Full merged config
        :param modules: Module specs; built from DEFAULT_MODULES + ``orchestrator.modules`` + ``enabled`` flags if None
        """
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
        self._stopping = False
        self.routed = 0

    # --- setup -----------------------------------------------------------------------------------------------

    @staticmethod
    def module_specs(config: Dict[str, Any]) -> List[ModuleSpec]:
        overrides = config.get("orchestrator", {}).get("modules", {}) or {}
        specs = []
        for name, (target, section, subscribe) in DEFAULT_MODULES.items():
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

    def start(self) -> None:
        self._create_frame_buffer()
        for module in self.modules.values():
            self._start(module)
        logger.warning(f"Orchestrator started: {', '.join(self.modules) or 'no modules'}")

    def _start(self, module: ModuleProcess) -> None:
        self._drain(module.control)
        module.process = self.ctx.Process(target=_child_main, name=module.spec.name, daemon=False,
                                          args=(module.spec.target, self.events, module.control, self.child_config))
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
        if event.type == "heartbeat":
            module = self.modules.get(event.payload.get("module"))
            if module is not None:
                module.last_heartbeat = self.clock()
            return
        self.routed += 1
        if self.log is not None:
            self.log.write(event)
        for module in self.modules.values():
            if module.spec.wants(event.type):
                try:
                    module.control.put_nowait(data)
                except queue.Full:
                    logger.warning(f"Control queue of {module.spec.name} is full; event dropped")

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
                self.route_pending(timeout=0.05)
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
