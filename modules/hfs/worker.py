"""
Hierarchical fuzzy system module (A1 / TOD-41, own process): the evidence events of every location feed the three
level-1 subsystems (gas, camera, audio), each re-evaluated at its own rate; the top level combines their risks and
publishes one ``hfs_risk`` event per location when it changes (level, partial / missing, overall by ≥ ``min_delta``)
and every ``keepalive_sec`` anyway::

    {"risk1": 0.1, "risk2": 0.84, "risk3": null, "overall": 0.82, "level": "critical", "interval": [0.78, 1.0],
     "partial": true, "missing": ["audio"], "subsystems": {...}, "fired_rules_by_level": {"gas": [...], ...},
     "mode": "autonomous"}

For a location in mode autonomous (TOD-52: global or its override, ``mode_changed`` switches at runtime) the event
carries the severity of its level (critical → critical, high → warning) and is the decision; in multiagent it is the
fuzzy agent's corridor (severity info) and the validation agent decides. Every event carries ``mode_requested`` and
``rules_version`` (gas / camera / audio / top). numpy + the fuzzy core only: no network, no LLM.
"""
import time
from typing import Any, Callable, Dict, List, Optional

from core.events import Event, make_event
from core.mode import AUTONOMOUS, location_mode
from core.module import BaseModule
from modules.hfs.audio import AudioSubsystem
from modules.hfs.base import FuzzySubsystem
from modules.hfs.camera import CameraSubsystem
from modules.hfs.gas import GasSubsystem
from modules.hfs.top import SUBSYSTEMS, TopLevel
from system_setup.logger_setup import init_logger

logger = init_logger()

EVIDENCE_EVENTS = ["gas_reading", "motion", "object_detected", "person_identified", "sound", "speech"]
# A5 (TOD-45): device self-diagnostics (device_health) that make a device's data unusable — no counters, no graded
# trust: the device is excluded until it reports healthy, its subsystem works with the rest (partial)
EXCLUDING_CHECKS = ("no_frames", "frozen_frame", "lens_covered", "stuck_silence", "wrong_sample_rate",
                    "sensor_error", "flat_readings", "out_of_range")
KIND_SUBSYSTEM = {"gas": "gas", "camera": "camera", "mic": "audio"}
SEVERITY = {"critical": "critical", "high": "warning"}


class HierarchicalFuzzySystem:
    """The whole HFS without the process loop (tests, replays): ``observe`` events, ``tick`` → hfs_risk payloads."""

    def __init__(self, config: Dict[str, Any], rules: Callable[[str], Dict[str, Any]] = lambda location_id: {},
                 devices: Callable[[], List[Dict[str, Any]]] = lambda: [], clock: Callable[[], float] = time.time,
                 ignore=("hub", "unknown"), names: Callable[[str], str] = lambda location_id: location_id,
                 **camera_kwargs):
        """
        :param rules: location_id → location rules (``activity_hours``, ``object_type``, ``gas_benign``)
        :param devices: → device records (kind, location_id, status, enabled, node_id) of ``core.devices``
        :param names: location_id → display name (explanations)
        """
        cfg = config.get("hfs") or {}
        self.subsystems: Dict[str, FuzzySubsystem] = {
            "gas": GasSubsystem(config, rules=rules, clock=clock),
            "camera": CameraSubsystem(config, rules=rules, clock=clock, **camera_kwargs),
            "audio": AudioSubsystem(config, clock=clock)}
        self.top = TopLevel(config=config)
        self.rules, self.names = rules, names
        self.devices, self.clock, self.ignore = devices, clock, ignore
        self.min_delta = float(cfg.get("min_delta", 0.05))
        self.keepalive = float(cfg.get("keepalive_sec", 30))
        self.exclude_checks = tuple((cfg.get("health") or {}).get("exclude") or EXCLUDING_CHECKS)
        # device → {"kind", "location_id", "reasons"}: offline / failing an excluding check
        self.excluded: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {}    # location → subsystem → result
        self.due: Dict[tuple, float] = {}                                     # (location, subsystem / top) → time
        self.published: Dict[str, Dict[str, Any]] = {}                         # location → last payload
        self.published_at: Dict[str, float] = {}

    def observe(self, event: Event) -> None:
        now = self.clock()
        if event.type in ("device_offline", "device_online", "device_health"):
            self._health(event)
            return
        if event.device_id in self.excluded:
            return                                   # an excluded device's evidence does not count
        for subsystem in self.subsystems.values():
            if subsystem.wants(event):
                subsystem.observe(event, now)

    def _health(self, event: Event) -> None:
        """Offline / failing devices are excluded (A5), with the reason, until they are back and healthy."""
        p = event.payload
        device = p.get("device_id") or event.device_id
        if event.type == "device_offline":
            reasons = ["offline"]
        elif event.type == "device_online":
            reasons = [r for r in (self.excluded.get(device) or {}).get("reasons", []) if r != "offline"]
        else:
            failing = p.get("failing") or {}
            reasons = [f"{check}" + (f" ({info.get('detail')})" if (info or {}).get("detail") else "")
                       for check, info in failing.items() if check in self.exclude_checks]
        if reasons:
            if device not in self.excluded:
                for subsystem in self.subsystems.values():
                    subsystem.forget_device(device)
            self.excluded[device] = {"device_id": device, "kind": p.get("kind"), "location_id": event.location_id
                                     if event.type == "device_health" else p.get("location_id", event.location_id),
                                     "reasons": reasons}
        else:
            self.excluded.pop(device, None)

    def excluded_in(self, location_id: str, subsystem: Optional[str] = None) -> List[Dict[str, Any]]:
        return [e for e in self.excluded.values() if e.get("location_id") == location_id
                and (subsystem is None or KIND_SUBSYSTEM.get(e.get("kind")) == subsystem)]

    def locations(self) -> List[str]:
        found = {d.get("location_id") for d in self.devices() if d.get("enabled", True)}
        for subsystem in self.subsystems.values():
            found.update(subsystem.locations())
        return sorted(loc for loc in found if loc and loc not in self.ignore)

    def available(self, kind: Optional[str], location_id: str) -> bool:
        return any(d.get("kind") == kind and d.get("location_id") == location_id and d.get("status") == "online"
                   and d.get("enabled", True) and d.get("device_id") not in self.excluded for d in self.devices())

    def evaluate(self, location_id: str, now: float) -> Optional[Dict[str, Any]]:
        """Re-runs the subsystems that are due, then the top level. :return: the hfs_risk payload or None"""
        results = self.results.setdefault(location_id, {})
        for name, subsystem in self.subsystems.items():
            if now >= self.due.get((location_id, name), 0.0):
                self.due[(location_id, name)] = now + subsystem.update_sec
                results[name] = subsystem.evaluate(location_id, now, self.available(subsystem.device_kind,
                                                                                     location_id))
                excluded = self.excluded_in(location_id, name)
                if excluded and results[name] is not None:          # the rest of its devices still report
                    results[name] = {**results[name], "partial": True, "excluded": excluded}
        payload = self.combine(location_id, results, now)
        if payload is not None:
            payload["excluded"] = self.excluded_in(location_id)
        return payload

    def combine(self, location_id: str, results: Dict[str, Optional[Dict[str, Any]]],
                now: float) -> Optional[Dict[str, Any]]:
        """The top level with the cross-subsystem escalation context of the location (A9 / TOD-51)."""
        camera = self.subsystems["camera"]
        try:
            inside = camera.activity_time(location_id) < 0          # negative = inside the activity hours
        except Exception:                                           # no schedule → no benign exception
            inside = None
        nodes = {d.get("device_id"): d.get("node_id") for d in self.devices() if d.get("node_id")}
        analysis = self.top.escalation.analyse(
            {name: results.get(name) for name in SUBSYSTEMS}, location_id, now, inside_hours=inside,
            object_type=(self.rules(location_id) or {}).get("object_type"),
            devices={name: s.devices_of(location_id, now) for name, s in self.subsystems.items()}, nodes=nodes,
            location_name=self.names(location_id))
        return self.top.combine({name: results.get(name) for name in SUBSYSTEMS}, analysis)

    def tick(self) -> List[Dict[str, Any]]:
        """:return: payloads (with ``location_id``) to publish now"""
        now, out = self.clock(), []
        for location_id in self.locations():
            if now < self.due.get((location_id, "top"), 0.0):
                continue
            self.due[(location_id, "top")] = now + self.top.update_sec
            payload = self.evaluate(location_id, now)
            if payload is None or not self._changed(location_id, payload, now):
                continue
            self.published[location_id], self.published_at[location_id] = payload, now
            out.append({**payload, "location_id": location_id})
        return out

    def _changed(self, location_id: str, payload: Dict[str, Any], now: float) -> bool:
        last = self.published.get(location_id)
        if last is None or now - self.published_at.get(location_id, 0.0) >= self.keepalive:
            return True
        return (last["level"] != payload["level"] or last["missing"] != payload["missing"]
                or last["partial"] != payload["partial"] or last.get("excluded") != payload.get("excluded")
                or last.get("explosive") != payload.get("explosive")
                or abs(last["overall"] - payload["overall"]) >= self.min_delta
                or abs(last.get("corridor_width", 0) - payload.get("corridor_width", 0)) >= self.min_delta)


def rules_versions(system: HierarchicalFuzzySystem) -> Dict[str, str]:
    """:return: the rule base version (F26 hash of the rules) of every level — carried by every decision (TOD-52)"""
    from core.fuzzy import load_yaml
    from core.fuzzy.governance import short, version_hash

    paths = {name: s.path for name, s in system.subsystems.items()}
    paths["top"] = system.top.path
    return {name: short(version_hash({"rules": load_yaml(path).get("rules")})) for name, path in paths.items()}


class HFSModule(BaseModule):
    name = "hfs"
    source = "system"

    def __init__(self, event_queue, control_queue, config: Dict[str, Any],
                 system: Optional[HierarchicalFuzzySystem] = None, clock=time.time):
        super().__init__(event_queue, control_queue, config)
        self.system = system
        self.clock = clock
        self.locations = None
        self._versions: Optional[Dict[str, str]] = None
        self._devices: List[Dict[str, Any]] = []
        self._devices_at = 0.0

    @property
    def versions(self) -> Dict[str, str]:
        if self._versions is None:
            self._versions = rules_versions(self.system)
        return self._versions

    def devices(self) -> List[Dict[str, Any]]:
        """Device records (cached 2 s; device_* events refresh at once)."""
        if self.locations is not None and self.clock() - self._devices_at > 2:
            self._devices, self._devices_at = self.locations.devices.list_devices(), self.clock()
        return self._devices

    def setup(self) -> None:
        if self.system is None:
            from core.locations import LocationRegistry

            network = self.config.get("network", {})
            self.locations = LocationRegistry(network.get("devices_path", "data/devices.db"))
            ignore = (network.get("hub_location") or "hub", "unknown")
            self.system = HierarchicalFuzzySystem(self.config, self.locations.rules, self.devices, self.clock,
                                                  ignore=ignore, names=self.locations.name)
        logger.info(f"HFS ready (mode {self.config.get('mode') or 'multiagent'}"
                    + (f", per location {self.config['mode_locations']}" if self.config.get("mode_locations") else "")
                    + ")")

    def teardown(self) -> None:
        if self.locations is not None:
            self.locations.close()

    def handle_control(self, message: Any) -> None:
        if not (isinstance(message, str) and message.startswith("{")):
            super().handle_control(message)
            return
        event = Event.from_json(message)
        if event.type == "mode_changed":          # TOD-52: from the next decision cycle
            self.config = {**self.config, "mode": event.payload["mode"],
                           "mode_locations": event.payload.get("locations") or {}}
            return
        if event.type.startswith("device_"):
            self._devices_at = 0.0
        self.system.observe(event)

    def step(self) -> None:
        for payload in self.system.tick():
            location_id = payload.pop("location_id")
            requested = location_mode(self.config, location_id)
            severity = SEVERITY.get(payload["level"], "info") if requested == AUTONOMOUS else "info"
            self.publish(make_event("system", "hfs_risk", {**payload, "mode_requested": requested,
                                                           "rules_version": self.versions},
                                    severity=severity, device_id=self.device_id, location_id=location_id))
        time.sleep(0.1)


def run(event_queue, control_queue, config):
    HFSModule(event_queue, control_queue, config).run()
