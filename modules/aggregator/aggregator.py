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
from modules.aggregator.incidents import IncidentRecorder
from modules.aggregator.outbox import Outbox
from system_setup.logger_setup import init_logger

logger = init_logger()

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "schemas", "snapshot.schema.json")
CSV_FIELDS = ["timestamp", "source", "type", "severity", "track_id", "identity", "action", "text", "gas_ppm",
              "confidence"]
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
    actions: List[str] = field(default_factory=list)
    present: bool = True


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
        self.people: Dict[int, Person] = {}
        self.object_tracks: Dict[int, str] = {}      # live non-person tracks: track id → class
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
        if event.type == "object_detected":
            track_id, cls = p.get("track_id"), p.get("class", "object")
            if cls == "person":
                person = self.people.setdefault(track_id, Person(track_id, now, now))
                person.present, person.last_seen = p.get("state") != "disappeared", now
            elif p.get("state") == "disappeared":
                self.object_tracks.pop(track_id, None)
            else:
                self.object_tracks[track_id] = cls
            if cls not in self.objects_seen:
                self.objects_seen.append(cls)
        elif event.type == "person_identified":
            if p.get("track_id") is None and p.get("renamed_from"):        # unknown_N named → update everywhere
                for person in self.people.values():
                    if person.identity == p["renamed_from"]:
                        person.identity, person.known = p["identity"], True
            elif p.get("identity") != "not_visible" or p.get("track_id") not in self.people:
                person = self.people.setdefault(p["track_id"], Person(p["track_id"], now, now))
                person.identity, person.known, person.last_seen = p.get("identity", person.identity), \
                    bool(p.get("known")), now
        elif event.type == "action":
            person = self.people.setdefault(p["track_id"], Person(p["track_id"], now, now))
            person.last_seen = now
            if p.get("action") and p["action"] not in person.actions:
                person.actions.append(p["action"])
            if p.get("action") == "possible_smoking":
                self.alerts.append(f"possible_smoking #{p['track_id']}"
                                   + (" gas_confirmed" if p.get("gas_confirmed") else ""))
            elif p.get("action") == "lying":
                self.alerts.append(f"lying #{p['track_id']}")
        elif event.type == "speech":
            self.speech.append({"t": hms(event), "text": p.get("text", ""), "language": p.get("language")})
        elif event.type == "sound":
            self.sounds.append({"t": hms(event), "label": p.get("label", "loud_noise"), "db": p.get("rms_db", 0)})
        elif event.type == "gas_reading":
            self.gas = p
        elif event.type == "alert":
            reason = p.get("reason", "alert")
            detail = p.get("gas") or p.get("module") or ""
            text = f"{reason} {detail}".strip()
            (self.gas_alerts if event.source == "gas" else self.alerts).append(text)
        elif event.type == "mic_state":
            self.mic_state = p.get("state")
        elif event.type == "scene_active":
            self.scene_active = bool(p.get("active"))

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
              incident_id: Optional[str] = None) -> Dict[str, Any]:
        cfg = self.cfg
        people = sorted(state.people.values(), key=lambda p: (not p.present, -p.last_seen))[:20]
        snapshot: Dict[str, Any] = {
            "snapshot_id": str(uuid.uuid4()),
            "window": {"from": utc_iso(state.window_start), "to": utc_iso(now)},
            "trigger": trigger,
            "people": [{"track_id": int(p.track_id), "identity": p.identity[:64], "known": p.known,
                        "actions": p.actions[-10:], "present": p.present,
                        "in_frame_sec": int(max(0, (now if p.present else p.last_seen) - p.first_seen))}
                       for p in people],
            "objects": state.objects_seen[:30],
            "speech": [{**s, "text": s["text"][:cfg.max_text_chars]} for s in state.speech[-20:]],
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
        })
        self._file.flush()

    def _open(self, day: str) -> None:
        self.close()
        path = os.path.join(self.export_dir, f"events_{day}.csv")
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


class Aggregator:
    """Snapshot / CSV logic without the process loop."""

    def __init__(self, config: Dict[str, Any], clock=time.time, local_now=datetime.now):
        agg = config.get("aggregator", {})
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
        self.state = SceneState(clock())
        self.last_snapshot = clock()
        self.last_immediate = float("-inf")
        self.pending_reason: Optional[str] = None

    def on_event(self, event: Event) -> Optional[Dict[str, Any]]:
        """:return: the snapshot if this event produced one immediately"""
        now = self.clock()
        if self.csv is not None:
            self.csv.write(event)
        if self.incidents is not None:
            self.incidents.add_event(event)
        self.state.apply(event, now)
        reason = self.critical_reason(event)
        if reason:
            self.pending_reason = self.pending_reason or reason
            return self.tick()
        return None

    def critical_reason(self, event: Event) -> Optional[str]:
        p = event.payload
        if event.type == "action" and p.get("action") in self.cfg.immediate_actions:
            return f"{p['action']} #{p.get('track_id')}"
        if event.severity == "critical":
            return f"{event.type} {p.get('reason') or p.get('action') or ''}".strip()
        if event.type == "person_identified" and str(p.get("identity", "")).startswith("unknown_") \
                and in_hours(self.local_now(), self.cfg.unknown_alert_hours):
            return f"unknown person {p['identity']} at {self.local_now().strftime('%H:%M')}"
        return None

    def tick(self) -> Optional[Dict[str, Any]]:
        """Produces a due snapshot (critical first, then the regular interval). :return: the snapshot or None"""
        now = self.clock()
        if self.incidents is not None:
            self.incidents.tick()
        if self.pending_reason and now - self.last_immediate >= self.cfg.immediate_cooldown_sec:
            reason, self.pending_reason = self.pending_reason, None
            self.last_immediate = now
            return self.snapshot("critical_event", now, reason)
        if now - self.last_snapshot >= self.cfg.snapshot_interval_sec:
            if self.state.scene_active or self.state.changed:
                return self.snapshot("interval", now)
            self.last_snapshot = now
            self.state.reset_window(now)                                   # empty window: nothing to report
        return None

    def add_frame(self, jpeg: bytes) -> None:
        """Annotated camera frame for incident archives (only kept locally)."""
        if self.incidents is not None:
            self.incidents.add_frame(jpeg)

    def snapshot(self, trigger: str, now: float, reason: Optional[str] = None) -> Dict[str, Any]:
        incident_id = self.incidents.start(reason or trigger) \
            if self.incidents is not None and trigger == "critical_event" else None
        snapshot = self.builder.build(self.state, trigger, now, reason, incident_id)
        digest = snapshot_digest(snapshot, incident_id)
        if incident_id:
            self.incidents.attach(snapshot, digest)
        self.outbox.add(snapshot, trigger, created_at=now, digest=digest)
        if self.write_files:
            os.makedirs(self.snapshot_dir, exist_ok=True)
            name = f"{snapshot['window']['to'].replace(':', '').replace('-', '')}_{snapshot['snapshot_id'][:8]}.json"
            with open(os.path.join(self.snapshot_dir, name), "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=1)
        logger.info(f"Snapshot {trigger} ({self.builder.size(snapshot)} B){f': {reason}' if reason else ''}")
        self.last_snapshot = now
        self.state.reset_window(now)
        return snapshot

    def close(self) -> None:
        if self.incidents is not None:
            self.incidents.finish()                                       # an open incident is not lost
        if self.csv is not None:
            self.csv.close()
        self.outbox.close()


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
            self.aggregator = Aggregator(self.config)

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
