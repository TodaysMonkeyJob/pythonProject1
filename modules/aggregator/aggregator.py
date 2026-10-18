"""
Aggregator (F5): turns the event stream into compact JSON snapshots for GPT (F6) and a flat CSV for analysis.

* Scene state for the current window: people (track id, identity, actions, time in frame), object classes,
  transcripts, loud sounds, latest gas reading and alerts, microphone state.
* A snapshot is built every ``snapshot_interval_sec`` when the scene is active or something changed, and right
  away on a critical event (``severity: critical``, an action from ``immediate_actions``, an unknown person within
  ``unknown_alert_hours``). It is validated against ``schemas/snapshot.schema.json``, shrunk to
  ``max_snapshot_bytes``, written to ``exports/snapshots/`` and queued in the SQLite outbox for F6.
* Every event is also appended to ``exports/events_YYYY-MM-DD.csv`` (UTF-8, one header per file).
"""
import csv
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.events import Event
from core.module import BaseModule
from modules.aggregator.digest import snapshot_digest
from modules.aggregator.llm_view import llm_view, payload_text
from modules.aggregator.incidents import IncidentRecorder
from modules.aggregator.outbox import Outbox
from system_setup.logger_setup import init_logger

logger = init_logger()

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "schemas", "snapshot.schema.json")
CSV_FIELDS = ["timestamp", "source", "type", "severity", "track_id", "identity", "action", "text", "gas_ppm",
              "confidence", "person_id", "speaker_name", "speaker_score", "location_id", "device_id"]
PPM_PER_PERCENT = 10_000


def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hms(event: Event) -> str:
    """HH:MM:SS (UTC) of an event, for the compact snapshot lists."""
    return event.timestamp[11:19]


def event_time(event: Event) -> float:
    return datetime.fromisoformat(event.timestamp.replace("Z", "+00:00")).timestamp()


def cpu_temperature() -> Optional[float]:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read()) / 1000, 1)
    except (OSError, ValueError):
        return None


def in_hours(now_local: datetime, hours: Optional[List[str]]) -> bool:
    """
    :param hours: ["22:00", "06:00"] (may wrap over midnight) or None
    """
    if not hours:
        return False
    start, end = (datetime.strptime(h, "%H:%M").time() for h in hours)
    current = now_local.time()
    return start <= current < end if start <= end else current >= start or current < end


@dataclass
class Person:
    track_id: int
    first_seen: float
    last_seen: float
    identity: str = "unidentified"
    known: bool = False
    person_id: Optional[str] = None     # identity registry id (F9); the name is resolved at snapshot time
    location_id: Optional[str] = None   # F11: where (several cameras / rooms)
    actions: List[str] = field(default_factory=list)
    present: bool = True
    trust: Optional[float] = None       # F16 trust of the face identification (F18: trusted / untrusted evidence)


@dataclass
class AggregatorConfig:
    snapshot_interval_sec: float = 60.0
    max_snapshot_bytes: int = 2048
    immediate_actions: Tuple[str, ...] = ("possible_smoking", "lying")
    unknown_alert_hours: Optional[List[str]] = None
    immediate_cooldown_sec: float = 3.0
    max_text_chars: int = 200

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AggregatorConfig":
        data = data or {}
        cfg = cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})
        cfg.immediate_actions = tuple(cfg.immediate_actions)
        return cfg


class SceneState:
    """Everything the next snapshot needs; ``reset_window()`` after each snapshot keeps only what is still present."""

    def __init__(self, started: float):
        self.window_start = started
        self.people: Dict[tuple, Person] = {}
        self.object_tracks: Dict[tuple, str] = {}    # live non-person tracks: (device, track id) → class
        self.objects_seen: List[str] = []
        self.speech: List[Dict[str, Any]] = []
        self.sounds: List[Dict[str, Any]] = []
        self.alerts: List[str] = []
        self.gas: Dict[str, Any] = {}
        self.gas_alerts: List[str] = []
        self.mic_state: Optional[str] = None
        self.scene_active = False
        self.events = 0
        self.changed = False

    def apply(self, event: Event, now: float) -> None:
        p = event.payload
        self.events += 1
        if event.type != "gas_reading":
            self.changed = True
        # Track ids are per camera (F11): people / objects are keyed by (device, track id)
        key = (event.device_id, p.get("track_id"))
        if event.type == "object_detected":
            cls = p.get("class", "object")
            if cls == "person":
                person = self._person(key, event, now)
                person.present, person.last_seen = p.get("state") != "disappeared", now
            elif p.get("state") == "disappeared":
                self.object_tracks.pop(key, None)
            else:
                self.object_tracks[key] = cls
            if cls not in self.objects_seen:
                self.objects_seen.append(cls)
        elif event.type == "person_identified":
            if p.get("track_id") is None and p.get("renamed_from"):        # unknown_N named → update everywhere
                for person in self.people.values():
                    if person.identity == p["renamed_from"]:
                        person.identity, person.known = p["identity"], True
            elif p.get("identity") not in ("not_visible", "unidentified") or key not in self.people:
                person = self._person(key, event, now)
                person.identity, person.known, person.last_seen = p.get("identity", person.identity), \
                    bool(p.get("known")), now
                person.person_id = p.get("person_id")
                if p.get("trust") is not None:
                    person.trust = float(p["trust"])
        elif event.type == "action":
            person = self._person(key, event, now)
            person.last_seen = now
            if p.get("action") and p["action"] not in person.actions:
                person.actions.append(p["action"])
            if p.get("action") == "possible_smoking":
                self.alerts.append(f"possible_smoking #{p['track_id']}"
                                   + (" gas_confirmed" if p.get("gas_confirmed") else ""))
            elif p.get("action") == "lying":
                self.alerts.append(f"lying #{p['track_id']}")
        elif event.type == "speech":
            entry = {"t": hms(event), "text": p.get("text", ""), "language": p.get("language")}
            if p.get("person_id"):
                entry.update(person_id=p["person_id"], speaker=p.get("speaker_name") or p.get("speaker"))
            elif p.get("speaker_status") == "uncertain":
                entry["speaker"] = "?"                                    # overlapping / too short to tell
            for key in ("trust", "claim_type", "provenance"):            # F16 tags → F18 trusted / untrusted split
                if p.get(key) is not None:
                    entry[key] = p[key]
            if p.get("trust_reasons"):
                entry["trust_reasons"] = [str(r)[:100] for r in p["trust_reasons"][:3]]
            self.speech.append(entry)
        elif event.type == "sound":
            self.sounds.append({"t": hms(event), "label": p.get("label", "loud_noise"), "db": p.get("rms_db", 0)})
        elif event.type == "gas_reading":
            self.gas = p
        elif event.type == "alert":
            reason = p.get("reason", "alert")
            detail = p.get("gas") or p.get("module") or ""
            text = f"{reason} {detail}".strip()
            (self.gas_alerts if event.source == "gas" else self.alerts).append(text)
        elif event.type == "scene_text":                                  # F16: signs / screens — always weak evidence
            self.speech.append({"t": hms(event), "text": p.get("text", ""), "kind": "scene_text",
                                **{k: p[k] for k in ("trust", "claim_type", "provenance") if p.get(k) is not None}})
        elif event.type == "mic_state":
            self.mic_state = p.get("state")
        elif event.type == "scene_active":
            self.scene_active = bool(p.get("active"))

    def _person(self, key, event: Event, now: float) -> Person:
        person = self.people.setdefault(key, Person(key[1], now, now))
        person.location_id = event.location_id
        return person

    def reset_window(self, now: float) -> None:
        self.window_start = now
        self.people = {tid: p for tid, p in self.people.items() if p.present}
        for person in self.people.values():
            person.actions = person.actions[-1:]                          # current action carries over
        self.objects_seen = sorted(set(self.object_tracks.values()) | ({"person"} if self.people else set()))
        self.speech, self.sounds, self.alerts, self.gas_alerts = [], [], [], []
        self.events, self.changed = 0, False


class SnapshotBuilder:
    def __init__(self, cfg: AggregatorConfig, schema_path: str = SCHEMA_PATH):
        import jsonschema

        with open(schema_path, encoding="utf-8") as f:
            self.schema = json.load(f)
        self.validator = jsonschema.Draft202012Validator(self.schema,
                                                         format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER)
        self.cfg = cfg

    def build(self, state: SceneState, trigger: str, now: float, reason: Optional[str] = None,
              incident_id: Optional[str] = None, registry=None, location_id: Optional[str] = None,
              location_name: Optional[str] = None) -> Dict[str, Any]:
        """
        :param registry: ``IdentityRegistry`` resolving person_id → current display name, so a rename made a moment
                         ago is already in this snapshot
        """
        cfg = self.cfg

        def names(person_id, fallback):
            return registry.name(person_id, fallback) if registry is not None and person_id else fallback

        def known(person_id, fallback):
            return registry.is_known(person_id) if registry is not None and registry.resolve(person_id) else fallback
        people = sorted(state.people.values(), key=lambda p: (not p.present, -p.last_seen))[:20]
        snapshot: Dict[str, Any] = {
            "snapshot_id": str(uuid.uuid4()),
            "window": {"from": utc_iso(state.window_start), "to": utc_iso(now)},
            "trigger": trigger,
            "people": [{"track_id": int(p.track_id), "identity": names(p.person_id, p.identity)[:64],
                        "known": known(p.person_id, p.known), "actions": p.actions[-10:], "present": p.present,
                        "in_frame_sec": int(max(0, (now if p.present else p.last_seen) - p.first_seen)),
                        **({"person_id": p.person_id} if p.person_id else {}),
                        **({"trust": round(p.trust, 3)} if p.trust is not None else {}),
                        **({"location_id": p.location_id[:64]}
                           if p.location_id and p.location_id != "unassigned" else {})}
                       for p in people],
            "objects": state.objects_seen[:30],
            "speech": [{**s, "text": s["text"][:cfg.max_text_chars],
                        **({"speaker": (names(s["person_id"], s.get("speaker")) or "")[:64]} if s.get("person_id")
                           else {"speaker": s["speaker"]} if s.get("speaker") else {})}
                       for s in state.speech[-20:]],
            "sounds": state.sounds[-20:],
            "gas": self._gas(state),
            "alerts": state.alerts[-20:],
            "system": {"mic_state": state.mic_state, "scene_active": state.scene_active,
                       "cpu_temp_c": cpu_temperature(), "events_in_window": state.events},
        }
        if snapshot["system"]["mic_state"] is None:
            del snapshot["system"]["mic_state"]
        if reason:
            snapshot["reason"] = reason[:120]
        if incident_id:
            snapshot["incident_id"] = incident_id
        if location_id:
            snapshot["location_id"], snapshot["location_name"] = location_id[:64], (location_name or location_id)[:64]
        self.shrink(snapshot)
        self.validator.validate(snapshot)
        return snapshot

    @staticmethod
    def _gas(state: SceneState) -> Dict[str, Any]:
        gas: Dict[str, Any] = {}
        if state.gas:
            gas["gas"] = str(state.gas.get("gas", "unknown"))[:32]
            if isinstance(state.gas.get("concentration_pct"), (int, float)):
                gas["concentration_ppm"] = int(round(state.gas["concentration_pct"] * PPM_PER_PERCENT))
            if state.gas.get("trend") in ("rising", "falling", "stable"):
                gas["trend"] = state.gas["trend"]
            if isinstance(state.gas.get("gas_confidence"), (int, float)):
                gas["confidence"] = round(float(state.gas["gas_confidence"]), 2)
            if isinstance(state.gas.get("rise_rate_pct_per_min"), (int, float)):
                gas["rise_rate_pct_per_min"] = round(float(state.gas["rise_rate_pct_per_min"]), 3)
            if isinstance(state.gas.get("trust"), (int, float)):
                gas["trust"] = round(float(state.gas["trust"]), 3)
        if state.gas_alerts:
            gas["alerts"] = list(dict.fromkeys(state.gas_alerts))[-10:]
        return gas

    def size(self, snapshot: Dict[str, Any]) -> int:
        return len(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    def shrink(self, snapshot: Dict[str, Any]) -> None:
        """Drops the least useful details until the snapshot fits max_snapshot_bytes (sets ``truncated``)."""
        for step in self._shrink_steps():
            while self.size(snapshot) > self.cfg.max_snapshot_bytes and step(snapshot):
                snapshot["truncated"] = True
            if self.size(snapshot) <= self.cfg.max_snapshot_bytes:
                return
        logger.warning(f"Snapshot still {self.size(snapshot)} bytes after truncation")

    @staticmethod
    def _shrink_steps():
        """Each step removes one detail and returns True, or returns False when it has nothing left to remove."""
        def pop_first(key, keep):
            def step(s):
                if len(s[key]) > keep:
                    s[key].pop(0)
                    return True
                return False
            return step

        def shorten_speech(s):
            long = [x for x in s["speech"] if len(x["text"]) > 80]
            for x in long:
                x["text"] = x["text"][:80]
            return bool(long)

        def drop_absent_person(s):
            absent = [p for p in s["people"] if not p["present"]]
            if absent:
                s["people"].remove(absent[-1])
            return bool(absent)

        def trim_actions(s):
            long = [p for p in s["people"] if len(p["actions"]) > 3]
            for p in long:
                p["actions"] = p["actions"][-3:]
            return bool(long)

        def pop_last(key, keep):
            def step(s):
                if len(s[key]) > keep:
                    s[key].pop()
                    return True
                return False
            return step

        # oldest transcripts first, then shorter texts, sounds, old alerts, people who left, long action lists, ...
        return [pop_first("speech", 5), shorten_speech, pop_first("sounds", 0), pop_first("alerts", 5),
                drop_absent_person, trim_actions, pop_first("speech", 0), pop_first("alerts", 0),
                pop_last("objects", 10), pop_last("people", 5)]


class EventCsv:
    """exports/events_YYYY-MM-DD.csv — one row per event, header once per file, new file per UTC day."""

    def __init__(self, export_dir: str):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
        self._day: Optional[str] = None
        self._file = None
        self._writer = None

    def write(self, event: Event) -> None:
        day = event.timestamp[:10]
        if day != self._day:
            self._open(day)
        p = event.payload
        gas_pct = p.get("concentration_pct")
        text = p.get("text") or p.get("reason") or ""
        if "class" in p:
            text = f"{p['class']} {p.get('state', '')}".strip()
        elif event.type == "mic_state":
            text = f"{p.get('state')} ({p.get('reason')})"
        self._writer.writerow({
            "timestamp": event.timestamp, "source": event.source, "type": event.type, "severity": event.severity,
            "track_id": p.get("track_id", ""), "identity": p.get("identity", ""), "action": p.get("action", ""),
            "text": text, "gas_ppm": int(round(gas_pct * PPM_PER_PERCENT)) if isinstance(gas_pct, (int, float)) else "",
            "confidence": "" if event.confidence is None else event.confidence,
            "person_id": p.get("person_id") or "", "speaker_name": p.get("speaker_name") or "",
            "speaker_score": p.get("speaker_score", ""), "location_id": p.get("location_id") or "",
            "device_id": p.get("device_id") or "",
        })
        self._file.flush()

    def _open(self, day: str) -> None:
        self.close()
        path = os.path.join(self.export_dir, f"events_{day}.csv")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, encoding="utf-8") as f:
                if f.readline().strip() != ",".join(CSV_FIELDS):   # written by an older version: don't mix columns
                    path = os.path.join(self.export_dir, f"events_{day}_v2.csv")
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._file = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        if new:
            self._writer.writeheader()
        self._day = day

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


SCENE_TYPES = ("object_detected", "person_identified", "action", "speech", "sound", "gas_reading", "alert",
               "mic_state", "scene_active", "scene_text")


class Aggregator:
    """
    Snapshot / CSV logic without the process loop. F12: one scene per location — snapshots are per location
    (``location_id`` / ``location_name``), a critical event in X triggers an immediate snapshot of X only, and every
    ``global_snapshot_min`` a short "whole site" snapshot lists every active location in one line.
    """

    def __init__(self, config: Dict[str, Any], clock=time.time, local_now=datetime.now, registry=None,
                 locations=None):
        """
        :param registry: ``core.identity.IdentityRegistry`` to resolve display names at snapshot time
        :param locations: ``core.locations.LocationRegistry`` for location names / rules (ids are used without it)
        """
        agg = config.get("aggregator", {})
        self.registry = registry
        self.locations = locations
        self.cfg = AggregatorConfig.from_dict(agg)
        export_dir = agg.get("export_dir", "exports")
        self.snapshot_dir = agg.get("snapshot_dir", os.path.join(export_dir, "snapshots"))
        self.write_files = agg.get("write_snapshot_files", True)
        self.outbox = Outbox(agg.get("outbox_path", os.path.join(export_dir, "outbox.db")))
        self.csv = EventCsv(export_dir) if agg.get("export_csv", True) else None
        self.builder = SnapshotBuilder(self.cfg, agg.get("schema_path", SCHEMA_PATH))
        self.clock, self.local_now = clock, local_now
        incidents = agg.get("incidents", {}) or {}
        self.incidents = IncidentRecorder({"dir": os.path.join(export_dir, "incidents"), **incidents}, clock) \
            if incidents.get("enabled", True) else None
        self.ignore_locations = (config.get("network", {}).get("hub_location") or "hub", "unknown")
        self.global_interval = float(agg.get("global_snapshot_min", 10)) * 60
        self.states: Dict[str, SceneState] = {}
        self.last_snapshot: Dict[str, float] = {}
        self.last_immediate: Dict[str, float] = {}
        self.pending: Dict[str, str] = {}
        self.last_global = clock()
        self.recent: Dict[str, Dict[str, Any]] = {}       # location → summary of its last snapshot (global digest)
        # F18: what GPT gets — "v2" (fuzzified, trust-tagged JSON view) or "digest" (the v1 text, for ablations)
        self.llm_format = agg.get("llm_format", "v2")
        self.fuzzy_path = config.get("trust", {}).get("fuzzy_config", "config/fuzzy.yaml")
        self._fuzzy: Dict[str, Any] = {}
        self._fuzzy_root: Optional[Dict[str, Any]] = None

    @property
    def state(self) -> SceneState:
        """The single location's scene (one-location setups, tests)."""
        return next(iter(self.states.values())) if self.states else self._scene("unassigned", self.clock())

    def _scene(self, location_id: str, now: float) -> SceneState:
        if location_id not in self.states:
            self.states[location_id] = SceneState(now)
            self.last_snapshot[location_id] = now
        return self.states[location_id]

    def location_name(self, location_id: str) -> str:
        return self.locations.name(location_id) if self.locations is not None else location_id

    def on_event(self, event: Event) -> Optional[Dict[str, Any]]:
        """:return: the snapshot if this event produced one immediately"""
        now = self.clock()
        if self.csv is not None:
            self.csv.write(event)
        if self.incidents is not None:
            self.incidents.add_event(event)
        if event.type not in SCENE_TYPES or event.location_id in self.ignore_locations:
            return None   # hub software events (module restarts, GPT, registry) are not about a place
        self._scene(event.location_id, now).apply(event, now)
        reason = self.critical_reason(event)
        if reason:
            self.pending.setdefault(event.location_id, reason)
            return self.tick(event.location_id)
        return None

    def critical_reason(self, event: Event) -> Optional[str]:
        p = event.payload
        if event.source == "llm":
            return None  # GPT decisions about a snapshot must not trigger new snapshots
        if event.type == "action" and p.get("action") in self.cfg.immediate_actions:
            return f"{p['action']} #{p.get('track_id')}"
        if event.severity == "critical":
            return f"{event.type} {p.get('reason') or p.get('action') or ''}".strip()
        if event.type == "person_identified" and str(p.get("identity", "")).startswith("unknown_") \
                and in_hours(self.local_now(), self.cfg.unknown_alert_hours):
            return f"unknown person {p['identity']} at {self.local_now().strftime('%H:%M')}"
        return None

    def tick(self, only: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Produces one due snapshot (critical first, then the regular interval, then the global one).

        :param only: Look at this location only (an immediate snapshot of the location with the critical event)
        :return: the snapshot or None
        """
        now = self.clock()
        if self.incidents is not None and only is None:
            self.incidents.tick()
        locations = [only] if only else list(self.states)
        for location_id in locations:
            if location_id in self.pending and \
                    now - self.last_immediate.get(location_id, float("-inf")) >= self.cfg.immediate_cooldown_sec:
                reason = self.pending.pop(location_id)
                self.last_immediate[location_id] = now
                return self.snapshot("critical_event", now, reason, location_id)
        if only:
            return None
        for location_id in locations:
            state = self.states[location_id]
            if now - self.last_snapshot[location_id] >= self.cfg.snapshot_interval_sec:
                if state.scene_active or state.changed:
                    return self.snapshot("interval", now, location_id=location_id)
                self.last_snapshot[location_id] = now
                state.reset_window(now)                                    # empty window: nothing to report
        if self.global_interval and now - self.last_global >= self.global_interval:
            since, self.last_global = self.last_global, now
            return self.global_snapshot(now, since)
        return None

    def add_frame(self, jpeg: bytes) -> None:
        """Annotated camera frame for incident archives (only kept locally)."""
        if self.incidents is not None:
            self.incidents.add_frame(jpeg)

    def snapshot(self, trigger: str, now: float, reason: Optional[str] = None,
                 location_id: Optional[str] = None) -> Dict[str, Any]:
        location_id = location_id or next(iter(self.states), "unassigned")
        state = self._scene(location_id, now)
        incident_id = self.incidents.start(reason or trigger) \
            if self.incidents is not None and trigger == "critical_event" else None
        if self.registry is not None:
            self.registry.poll()
        name = self.location_name(location_id)
        snapshot = self.builder.build(state, trigger, now, reason, incident_id, self.registry, location_id, name)
        rules = self.locations.rules(location_id) if self.locations is not None else {}
        if self.llm_format == "digest":
            digest = snapshot_digest(snapshot, incident_id, rules=rules)
        else:
            engine, theta = self._fuzzy_for(location_id)
            snapshot["schema_version"] = 2
            snapshot["llm_view"] = llm_view(snapshot, engine, theta, rules)
            self.builder.validator.validate(snapshot)
            digest = payload_text(snapshot["llm_view"])
        if incident_id:
            self.incidents.attach(snapshot, digest)
        self.outbox.add(snapshot, trigger, created_at=now, digest=digest)
        self._write(snapshot)
        logger.info(f"Snapshot {trigger} {location_id} ({self.builder.size(snapshot)} B)"
                    f"{f': {reason}' if reason else ''}")
        self.recent[location_id] = {"at": now, "name": name, "people": snapshot["people"], "alerts": snapshot["alerts"],
                                    "gas": snapshot.get("gas") or {}}
        self.last_snapshot[location_id] = now
        state.reset_window(now)
        return snapshot

    def _fuzzy_for(self, location_id: str):
        """:return: (the location's FuzzyEngine or None, trust threshold θ) for the v2 view"""
        if location_id not in self._fuzzy:
            engine, theta = None, 0.6
            try:
                import yaml

                from core.fuzzy import FuzzyEngine

                if self._fuzzy_root is None:
                    with open(self.fuzzy_path, encoding="utf-8") as f:
                        self._fuzzy_root = yaml.safe_load(f) or {}
                envelope = self._fuzzy_root.get("envelope") or {}
                theta = float(((envelope.get("locations") or {}).get(location_id) or {}).get(
                    "theta", envelope.get("theta", 0.6)))
                engine = FuzzyEngine.from_config(self.fuzzy_path, location_id)
            except (OSError, ValueError, KeyError) as e:
                logger.warning(f"Fuzzy terms unavailable for {location_id} ({e}); the LLM view keeps labels only")
            self._fuzzy[location_id] = (engine, theta)
        return self._fuzzy[location_id]

    def global_snapshot(self, now: float, since: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """One line per location that reported since the previous global snapshot — the "whole site" view for GPT."""
        since = now - self.global_interval if since is None else since
        active = {loc: info for loc, info in self.recent.items() if info["at"] >= since}
        if len(active) < 2:
            return None  # one location: its own snapshots already say everything
        lines = [f"site overview, last {int(self.global_interval // 60)} min, {len(active)} locations"]
        for info in active.values():
            people = ", ".join(f"{p['identity']} {'>'.join(p['actions'][-2:])}".strip() for p in info["people"][:5])
            gas = info["gas"]
            line = f"{info['name']}: {len(info['people'])} people" + (f" ({people})" if people else "")
            if gas:
                line += f"; gas {gas.get('gas')} {gas.get('trend')}"
            if info["alerts"]:
                line += f"; alerts {', '.join(info['alerts'][-3:])}"
            lines.append(line)
        snapshot = {"snapshot_id": str(uuid.uuid4()), "trigger": "global", "window": {
            "from": utc_iso(since), "to": utc_iso(now)}, "locations": list(active)}
        digest = "\n".join(lines)[:1200]
        self.outbox.add(snapshot, "global", created_at=now, digest=digest)
        self._write(snapshot)
        return snapshot

    def _write(self, snapshot: Dict[str, Any]) -> None:
        if not self.write_files:
            return
        os.makedirs(self.snapshot_dir, exist_ok=True)
        name = f"{snapshot['window']['to'].replace(':', '').replace('-', '')}_{snapshot['snapshot_id'][:8]}.json"
        with open(os.path.join(self.snapshot_dir, name), "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=1)

    def close(self) -> None:
        if self.incidents is not None:
            self.incidents.finish()                                       # an open incident is not lost
        if self.csv is not None:
            self.csv.close()
        self.outbox.close()
        if self.locations is not None:
            self.locations.close()


class AggregatorModule(BaseModule):
    name = "aggregator"
    source = "system"

    def __init__(self, event_queue, control_queue, config: Dict[str, Any], aggregator: Optional[Aggregator] = None,
                 frame_buffer=None):
        super().__init__(event_queue, control_queue, config)
        self.aggregator = aggregator
        self.frame_buffer = frame_buffer
        self._last_frame_id = 0
        self._next_attach = 0.0

    def setup(self) -> None:
        if self.aggregator is None:
            registry = None
            if self.config.get("identity", {}).get("enabled", False):
                from core.identity import registry_from_config

                registry = registry_from_config(self.config["identity"])
            locations = None
            if "network" in self.config:  # the device network is configured (always in config.yaml)
                try:
                    from core.locations import LocationRegistry

                    locations = LocationRegistry(self.config["network"].get("devices_path", "data/devices.db"))
                except Exception as e:  # snapshots keep working with location ids
                    logger.warning(f"Location registry unavailable: {e}")
            self.aggregator = Aggregator(self.config, registry=registry, locations=locations)

    def teardown(self) -> None:
        if self.aggregator is not None:
            self.aggregator.close()
        if self.frame_buffer is not None:
            self.frame_buffer.close()

    def handle_control(self, message: Any) -> None:
        """Every bus event arrives here (the orchestrator forwards all types to the aggregator)."""
        if isinstance(message, str) and message.startswith("{"):
            try:
                self.aggregator.on_event(Event.from_json(message))
            except Exception as e:
                logger.error(f"Aggregator failed on an event: {e}", exc_info=True)
        else:
            super().handle_control(message)

    def step(self) -> None:
        self.aggregator.tick()
        self._sample_frame()
        time.sleep(0.1)

    def _sample_frame(self) -> None:
        """Feeds the incident recorder with an annotated frame every frame_interval_sec."""
        incidents = self.aggregator.incidents
        if incidents is None or not incidents.wants_frame:
            return
        if self.frame_buffer is None:
            if time.monotonic() < self._next_attach:
                return
            self._next_attach = time.monotonic() + 5
            vision = self.config.get("vision", {})
            if not vision.get("shared_buffer"):
                return
            from modules.vision.shared_frame import SharedFrameBuffer

            width, height = vision.get("resolution", [640, 480])
            try:
                self.frame_buffer = SharedFrameBuffer.attach(vision["shared_buffer"], width, height)
            except FileNotFoundError:
                return
        frame = self.frame_buffer.read_annotated()
        if frame is not None and frame[0] != self._last_frame_id:
            self._last_frame_id = frame[0]
            self.aggregator.add_frame(frame[2])


def run(event_queue, control_queue, config):
    """Module entry point for the orchestrator (own process; subscribed to all events)."""
    AggregatorModule(event_queue, control_queue, config).run()
