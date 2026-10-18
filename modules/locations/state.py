"""
Per-location security state (F12 / TOD-23): every room / territory is ``safe``, ``attention``, ``danger`` or
``offline``, computed from its own events.

* ``danger`` — an unacknowledged critical event (gas threshold, lying, smoking + gas, a per-location gas threshold …)
  newer than ``cooldown_min``, or a GPT risk ``high`` / ``critical`` for this location (valid ``llm_valid_min``).
* ``attention`` — warnings in the last ``attention_window_min``, unacknowledged older alerts, or GPT risk ``medium``.
* ``offline`` — every device of the location is offline (and nothing is pending).
* ``safe`` — none of the above.

``danger`` de-escalates only after ``cooldown_min`` without new critical events; acknowledging clears the location's
alerts (and the GPT risk) at once. Every transition is returned as a ``location_state_changed`` payload; ``summary()``
is the live object pushed to the dashboard.

Per-location rules (``core.locations``): ``unknown_person_after`` ("20:00": an unknown person only counts as a warning
from then until 06:00), ``gas_thresholds`` ({gas: %} → critical), ``attention_window_min``, ``cooldown_min``.
"""
import collections
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from core.events import Event, utc_timestamp

SAFE, ATTENTION, DANGER, OFFLINE = "safe", "attention", "danger", "offline"
RANK = {SAFE: 0, OFFLINE: 0, ATTENTION: 1, DANGER: 2}
IGNORED_TYPES = ("location_state_changed", "location_summary", "notification", "notification_acked", "heartbeat")


@dataclass
class Alert:
    event_id: str
    ts: float
    reason: str
    acknowledged: bool = False


@dataclass
class LocationLive:
    location_id: str
    state: str = SAFE
    reason: str = ""
    since: float = 0.0
    people: Dict[Tuple[str, Any], Dict[str, Any]] = field(default_factory=dict)
    last_speech: Optional[Dict[str, Any]] = None
    gas: Optional[Dict[str, Any]] = None
    last_llm: Optional[Dict[str, Any]] = None
    alerts: List[Alert] = field(default_factory=list)
    warnings: Deque[Tuple[float, str, str]] = field(default_factory=collections.deque)
    last_critical: float = float("-inf")
    acknowledged_at: float = float("-inf")
    updated_at: float = 0.0


def in_evening(now_local: datetime, after: str, until: str = "06:00") -> bool:
    start = datetime.strptime(after, "%H:%M").time()
    end = datetime.strptime(until, "%H:%M").time()
    current = now_local.time()
    return current >= start or current < end if start > end else start <= current < end


def describe(event: Event) -> str:
    p = event.payload
    if event.type == "action":
        return f"{p.get('action')} #{p.get('track_id')}" + (" + gas" if p.get("gas_confirmed") else "")
    if event.type == "alert":
        return " ".join(str(x) for x in (p.get("reason", "alert"), p.get("gas") or p.get("module") or "") if x)
    if event.type == "person_identified":
        return f"{p.get('identity')} #{p.get('track_id')}"
    if event.type == "gas_reading":
        return f"{p.get('gas')} {p.get('concentration_pct')}% {p.get('trend', '')}".strip()
    return event.type


class LocationStateEngine:
    def __init__(self, names: Callable[[str], str], rules: Callable[[str], Dict[str, Any]],
                 devices: Callable[[], List[Dict[str, Any]]], cfg: Optional[Dict[str, Any]] = None,
                 local_now: Callable[[], datetime] = datetime.now, ignore: Tuple[str, ...] = ("hub", "unknown")):
        """
        :param names: location_id → display name
        :param rules: location_id → rule overrides
        :param devices: → current device records (location_id, status, enabled)
        :param ignore: locations of software modules (their events are not about a place)
        """
        cfg = cfg or {}
        self.names, self.rules, self.devices = names, rules, devices
        self.attention_window = float(cfg.get("attention_window_min", 5)) * 60
        self.cooldown = float(cfg.get("cooldown_min", 2)) * 60
        self.llm_valid = float(cfg.get("llm_valid_min", 10)) * 60
        self.local_now = local_now
        self.ignore = ignore
        self.live: Dict[str, LocationLive] = {}
        self.dirty: set = set()

    def _live(self, location_id: str, now: float) -> LocationLive:
        if location_id not in self.live:
            self.live[location_id] = LocationLive(location_id, since=now, updated_at=now)
        return self.live[location_id]

    def _rule(self, location_id: str, key: str, default: float) -> float:
        value = self.rules(location_id).get(key)
        return float(value) * 60 if value is not None else default

    # --- input ------------------------------------------------------------------------------------------------------

    def on_event(self, event: Event, now: float) -> List[Dict[str, Any]]:
        """:return: state transitions caused by this event"""
        if event.type in IGNORED_TYPES or event.location_id in self.ignore or event.type.startswith("device_"):
            if event.type.startswith("device_"):
                self.dirty.update(self.live)  # device counts / offline may change everywhere
            return []
        loc = self._live(event.location_id, now)
        loc.updated_at = now
        self.dirty.add(loc.location_id)
        p = event.payload
        key = (event.device_id, p.get("track_id"))
        if event.type == "object_detected" and p.get("class") == "person":
            if p.get("state") == "disappeared":
                loc.people.pop(key, None)
            else:
                loc.people.setdefault(key, {"person_id": None, "name": "?", "action": None})
        elif event.type == "person_identified" and p.get("track_id") is not None:
            person = loc.people.setdefault(key, {"person_id": None, "name": "?", "action": None})
            if p.get("identity") != "not_visible":
                person.update(person_id=p.get("person_id"), name=p.get("identity"))
        elif event.type == "action" and p.get("track_id") is not None:
            loc.people.setdefault(key, {"person_id": None, "name": "?", "action": None})["action"] = p.get("action")
        elif event.type == "speech":
            loc.last_speech = {"name": p.get("speaker_name") or p.get("speaker") or "?",
                               "person_id": p.get("person_id"), "text": p.get("text", ""), "ts": event.timestamp}
        elif event.type == "gas_reading":
            loc.gas = {"main": p.get("gas"), "concentration_pct": p.get("concentration_pct"),
                       "trend": p.get("trend")}
            limit = (self.rules(loc.location_id).get("gas_thresholds") or {}).get(p.get("gas"))
            if limit is not None and float(p.get("concentration_pct") or 0) >= float(limit):
                self._critical(loc, event, now, f"{p.get('gas')} {p.get('concentration_pct')}% ≥ {limit}% "
                                                f"(rule of {self.names(loc.location_id)})")
        if event.type == "llm_decision":
            loc.last_llm = {"risk_level": p.get("risk_level"), "summary": p.get("summary"), "ts": event.timestamp,
                            "at": now, "event_id": event.event_id}
        elif event.severity == "critical" and event.source != "llm":
            self._critical(loc, event, now, describe(event))
        elif event.severity == "warning" and event.source != "llm" and self._counts_as_warning(event):
            loc.warnings.append((now, describe(event), event.event_id))
        return self.evaluate(now, [loc.location_id])

    def _critical(self, loc: LocationLive, event: Event, now: float, reason: str) -> None:
        existing = next((a for a in loc.alerts if not a.acknowledged and a.reason == reason), None)
        if existing is not None:
            existing.ts = now          # the same problem goes on: keep one alert, refresh it
        else:
            loc.alerts.append(Alert(event.event_id, now, reason))
        loc.last_critical = now

    def _counts_as_warning(self, event: Event) -> bool:
        after = self.rules(event.location_id).get("unknown_person_after")
        if event.type == "person_identified" and after:
            return in_evening(self.local_now(), str(after))   # strangers in the daytime are visitors
        return True

    def acknowledge(self, location_id: str, now: float) -> List[Dict[str, Any]]:
        """Clears the location's alerts and its GPT risk (the user has seen it)."""
        loc = self._live(location_id, now)
        for alert in loc.alerts:
            alert.acknowledged = True
        loc.acknowledged_at = now
        loc.warnings.clear()
        self.dirty.add(location_id)
        return self.evaluate(now, [location_id])

    # --- state ------------------------------------------------------------------------------------------------------

    def compute(self, loc: LocationLive, now: float) -> Tuple[str, str, List[str]]:
        """:return: (state, reason, event ids behind it)"""
        cooldown = self._rule(loc.location_id, "cooldown_min", self.cooldown)
        window = self._rule(loc.location_id, "attention_window_min", self.attention_window)
        while loc.warnings and now - loc.warnings[0][0] > window:
            loc.warnings.popleft()
        loc.alerts = [a for a in loc.alerts if not a.acknowledged or now - a.ts < 3600]   # keep an hour of history
        open_alerts = [a for a in loc.alerts if not a.acknowledged]
        llm = loc.last_llm if loc.last_llm and now - loc.last_llm["at"] < self.llm_valid \
            and loc.last_llm["at"] > loc.acknowledged_at else None
        risk = llm["risk_level"] if llm else None
        if open_alerts and now - loc.last_critical < cooldown:
            reasons = "; ".join(dict.fromkeys(a.reason for a in open_alerts[-3:]))
            return DANGER, reasons, [a.event_id for a in open_alerts]
        if risk in ("high", "critical"):
            return DANGER, f"GPT: {llm['summary']}", [llm["event_id"]]
        devices = [d for d in self.devices() if d["location_id"] == loc.location_id and d["enabled"]]
        if devices and not any(d["status"] == "online" for d in devices) and not open_alerts:
            return OFFLINE, "all devices offline", []
        if open_alerts:
            return ATTENTION, "unacknowledged: " + "; ".join(dict.fromkeys(a.reason for a in open_alerts[-3:])), \
                [a.event_id for a in open_alerts]
        if loc.warnings:
            return ATTENTION, "; ".join(dict.fromkeys(w[1] for w in list(loc.warnings)[-3:])), \
                [w[2] for w in loc.warnings]
        if risk == "medium":
            return ATTENTION, f"GPT: {llm['summary']}", [llm["event_id"]]
        return SAFE, "", []

    def evaluate(self, now: float, location_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Recomputes states (time-based de-escalation, offline). :return: transitions"""
        for device in self.devices():  # locations that only have devices (no events yet) exist too
            if device["location_id"] not in self.ignore:
                self._live(device["location_id"], now)
        changes = []
        for location_id in location_ids or list(self.live):
            loc = self.live[location_id]
            state, reason, event_ids = self.compute(loc, now)
            if state != loc.state:
                changes.append({"location_id": location_id, "name": self.names(location_id), "from": loc.state,
                                "to": state, "reason": reason or f"back to {state}", "event_ids": event_ids[-10:]})
                loc.state, loc.since = state, now
                self.dirty.add(location_id)
            if reason != loc.reason:
                loc.reason = reason
                self.dirty.add(location_id)
        return changes

    def summary(self, location_id: str, now: float) -> Dict[str, Any]:
        loc = self._live(location_id, now)
        devices = [d for d in self.devices() if d["location_id"] == location_id]
        people = [{"person_id": p["person_id"], "name": p["name"], "action": p["action"]} for p in loc.people.values()]
        llm = loc.last_llm
        return {"location_id": location_id, "name": self.names(location_id), "state": loc.state, "reason": loc.reason,
                "since": utc_timestamp(datetime.fromtimestamp(loc.since).astimezone()) if loc.since else None,
                "people": people[:20], "people_count": len(people), "last_speech": loc.last_speech, "gas": loc.gas,
                "devices": {"online": sum(d["status"] == "online" for d in devices), "total": len(devices)},
                "last_llm": {k: llm[k] for k in ("risk_level", "summary", "ts")} if llm else None,
                "active_alerts": sum(not a.acknowledged for a in loc.alerts),
                "updated_at": utc_timestamp()}

    def take_dirty(self) -> List[str]:
        dirty, self.dirty = sorted(self.dirty), set()
        return [d for d in dirty if d in self.live]
