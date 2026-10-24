"""
Local incident archives for later human analysis — never sent to GPT (only the incident id is).

The recorder keeps the last ``context_sec`` of events and a ring of annotated camera frames (one per
``frame_interval_sec`` for ``pre_frames_sec``) in memory. A critical snapshot opens an incident; events and frames
keep being collected for ``post_sec``; then ``exports/incidents/<id>.zip`` is written:

* ``summary.txt`` — reasons, the GPT digest(s) and a human-readable timeline;
* ``incident.json`` — id, reasons, times, snapshots;
* ``events.jsonl`` — every event from ``context_sec`` before to ``post_sec`` after;
* ``frames/NNN_HHMMSS.jpg`` — annotated frames before and after the trigger.

Critical events while an incident is open extend it (one archive per situation, not per alert). Oldest archives
are deleted above ``max_total_mb``. Frames contain faces: archives stay on the device (``exports/`` is git-ignored).
"""
import collections
import json
import os
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from core.events import Event
from system_setup.logger_setup import init_logger

logger = init_logger()


@dataclass
class Incident:
    incident_id: str
    started_at: float
    deadline: float
    reasons: List[str]
    events: List[Tuple[float, str]]
    frames: List[Tuple[float, bytes]]
    snapshots: List[Dict[str, Any]] = field(default_factory=list)
    digests: List[str] = field(default_factory=list)


def timeline_line(event: Event, local_offset) -> str:
    """One readable line per event for summary.txt."""
    t = (datetime.fromisoformat(event.timestamp.replace("Z", "+00:00")) + local_offset).strftime("%H:%M:%S")
    p = event.payload
    detail = {
        "object_detected": lambda: f"#{p.get('track_id')} {p.get('class')} {p.get('state')}",
        "person_identified": lambda: f"#{p.get('track_id')} {p.get('identity')}"
                                     f"{' (was ' + p['previous'] + ')' if p.get('previous') else ''}",
        "action": lambda: f"#{p.get('track_id')} {p.get('action')}",
        "speech": lambda: f"\"{p.get('text', '')}\"",
        "sound": lambda: f"{p.get('rms_db')} dB",
        "gas_reading": lambda: f"{p.get('gas')} {p.get('concentration_pct')}% {p.get('trend')}",
        "alert": lambda: " ".join(str(x) for x in (p.get("reason"), p.get("gas") or p.get("module") or "") if x),
        "mic_state": lambda: f"{p.get('state')} ({p.get('reason')})",
        "scene_active": lambda: "active" if p.get("active") else "idle",
        "motion": lambda: str(p.get("state")),
    }.get(event.type, lambda: json.dumps(p, ensure_ascii=False)[:120])()
    severity = "" if event.severity == "info" else f" [{event.severity}]"
    return f"{t} {event.source:6} {event.type:17} {detail}{severity}"


class IncidentRecorder:
    def __init__(self, cfg: Dict[str, Any], clock: Callable[[], float],
                 local_now: Callable[[], datetime] = lambda: datetime.now().astimezone()):
        self.dir = cfg.get("dir", "exports/incidents")
        self.context_sec = float(cfg.get("context_sec", 120))
        self.post_sec = float(cfg.get("post_sec", 15))
        self.frame_interval_sec = float(cfg.get("frame_interval_sec", 2))
        self.pre_frames_sec = float(cfg.get("pre_frames_sec", 30))
        self.max_frames = int(cfg.get("max_frames", 40))
        self.max_total_bytes = int(float(cfg.get("max_total_mb", 1024)) * 1024 * 1024)
        self.clock = clock
        self.local_now = local_now
        self.events: Deque[Tuple[float, str]] = collections.deque()
        self.frames: Deque[Tuple[float, bytes]] = collections.deque()
        self.active: Optional[Incident] = None
        self._last_frame = float("-inf")

    @property
    def wants_frame(self) -> bool:
        return self.clock() - self._last_frame >= self.frame_interval_sec

    def add_event(self, event: Event) -> None:
        now = self.clock()
        item = (now, event.to_json())
        self.events.append(item)
        while self.events and now - self.events[0][0] > self.context_sec:
            self.events.popleft()
        if self.active is not None:
            self.active.events.append(item)

    def add_frame(self, jpeg: bytes) -> None:
        now = self.clock()
        self._last_frame = now
        self.frames.append((now, jpeg))
        while self.frames and now - self.frames[0][0] > self.pre_frames_sec:
            self.frames.popleft()
        if self.active is not None and len(self.active.frames) < self.max_frames:
            self.active.frames.append((now, jpeg))

    def start(self, reason: str) -> str:
        """
        Opens an incident (or extends the open one).

        :return: incident id
        """
        now = self.clock()
        if self.active is not None:
            self.active.deadline = now + self.post_sec
            if reason not in self.active.reasons:
                self.active.reasons.append(reason)
            return self.active.incident_id
        incident_id = f"{self.local_now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
        self.active = Incident(incident_id, now, now + self.post_sec, [reason], list(self.events),
                               list(self.frames)[-self.max_frames:])
        logger.warning(f"Incident {incident_id} opened: {reason}")
        return incident_id

    def attach(self, snapshot: Dict[str, Any], digest: str) -> None:
        if self.active is not None:
            self.active.snapshots.append(snapshot)
            self.active.digests.append(digest)

    def tick(self) -> Optional[str]:
        """Writes the open incident once its post-event time is over. :return: archive path or None"""
        if self.active is not None and self.clock() >= self.active.deadline:
            return self.finish()
        return None

    def finish(self) -> Optional[str]:
        incident, self.active = self.active, None
        if incident is None:
            return None
        os.makedirs(self.dir, exist_ok=True)
        path = os.path.join(self.dir, f"{incident.incident_id}.zip")
        offset = self.local_now().utcoffset()
        events = [Event.from_json(data) for _, data in incident.events]
        summary = [f"Incident {incident.incident_id}", "Reasons: " + "; ".join(incident.reasons),
                   f"Events: {len(events)}  Frames: {len(incident.frames)}", ""]
        for digest in incident.digests:
            summary += ["GPT digest:", digest, ""]
        summary += ["Timeline (local time):"] + [timeline_line(e, offset) for e in events if e.type != "gas_reading"
                                                  or e.severity != "info"]
        meta = {"incident_id": incident.incident_id, "reasons": incident.reasons,
                "started_at": datetime.fromtimestamp(incident.started_at).astimezone().isoformat(timespec="seconds"),
                "frames": len(incident.frames), "events": len(events), "snapshots": incident.snapshots}
        tmp = path + ".part"
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("summary.txt", "\n".join(summary) + "\n")
            archive.writestr("incident.json", json.dumps(meta, ensure_ascii=False, indent=1))
            archive.writestr("events.jsonl", "".join(data + "\n" for _, data in incident.events))
            for i, (t, jpeg) in enumerate(incident.frames):
                stamp = datetime.fromtimestamp(t).strftime("%H%M%S")
                marker = "before" if t < incident.started_at else "after"
                archive.writestr(f"frames/{i:03d}_{stamp}_{marker}.jpg", jpeg, compress_type=zipfile.ZIP_STORED)
        os.replace(tmp, path)
        logger.warning(f"Incident archive {path} ({os.path.getsize(path) // 1024} KB, {len(events)} events, "
                       f"{len(incident.frames)} frames)")
        self.enforce_size_limit()
        return path

    def enforce_size_limit(self) -> None:
        archives = sorted((os.path.join(self.dir, f) for f in os.listdir(self.dir) if f.endswith(".zip")),
                          key=os.path.getmtime)
        total = sum(os.path.getsize(a) for a in archives)
        while archives and total > self.max_total_bytes and len(archives) > 1:
            oldest = archives.pop(0)
            total -= os.path.getsize(oldest)
            os.remove(oldest)
            logger.info(f"Removed old incident archive {oldest} (size cap)")
