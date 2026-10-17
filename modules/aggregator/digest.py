"""
Token-lean text digest of a snapshot — what F6 sends to GPT instead of the JSON.

One short line per non-empty section, local time, no ids / nulls / key names::

    12:03:31-12:03:46 UTC+3 CRITICAL gas_threshold acetone | incident 20261017-120331-3f2a
    people: #5 unknown_1 walking>sitting 48s; #9 Oleh standing 12s left; #6 ? 3s
    objects: person laptop chair
    speech: 12:03:40 uk "Привіт, є тут хтось?"
    sounds: 12:03:41 loud -12dB
    gas: acetone 51960ppm rising conf0.30 alerts gas_threshold
    alerts: possible_smoking #5 gas_confirmed
    system: mic active, scene active, 37 events

Legend for the GPT system prompt: ``#N`` track id, ``?`` face not identified yet, ``unknown_N`` stranger
(stable label), ``a>b`` action sequence, ``Ns`` seconds in frame, ``left`` no longer in frame.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

UNIDENTIFIED = ("unidentified", "not_visible")


def _offset(local_now: Optional[datetime]) -> timedelta:
    local_now = local_now or datetime.now().astimezone()
    return local_now.utcoffset() or timedelta(0)


def _local_hms(hms_utc: str, offset: timedelta) -> str:
    t = datetime.strptime(hms_utc, "%H:%M:%S") + offset
    return t.strftime("%H:%M:%S")


def _tz_label(offset: timedelta) -> str:
    hours = offset.total_seconds() / 3600
    return "UTC" if hours == 0 else f"UTC{hours:+g}"


def snapshot_digest(snapshot: Dict[str, Any], incident_id: Optional[str] = None,
                    local_now: Optional[datetime] = None, max_chars: int = 1200) -> str:
    """
    :param snapshot: Snapshot dict (schemas/snapshot.schema.json)
    :param incident_id: Local incident archive id to mention (the archive itself is never sent)
    :param local_now: Timezone-aware local time used for the UTC offset (tests)
    :param max_chars: Hard cap; lines are cut from the bottom-most optional sections first
    :return: Compact multi-line text
    """
    offset = _offset(local_now)
    start = datetime.strptime(snapshot["window"]["from"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    end = datetime.strptime(snapshot["window"]["to"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    header = f"{(start + offset):%H:%M:%S}-{(end + offset):%H:%M:%S} {_tz_label(offset)}"
    if snapshot["trigger"] == "critical_event":
        header += f" CRITICAL {snapshot.get('reason', '')}".rstrip()
    if incident_id:
        header += f" | incident {incident_id}"
    lines = [header]

    people = []
    for p in snapshot["people"]:
        identity = "?" if p["identity"] in UNIDENTIFIED else p["identity"]
        part = f"#{p['track_id']} {identity}"
        if p["actions"]:
            part += " " + ">".join(p["actions"])
        part += f" {p['in_frame_sec']}s"
        if not p.get("present", True):
            part += " left"
        people.append(part)
    if people:
        lines.append("people: " + "; ".join(people))
    if snapshot["objects"]:
        lines.append("objects: " + " ".join(snapshot["objects"]))
    for s in snapshot["speech"]:
        language = f" {s['language']}" if s.get("language") else ""
        lines.append(f"speech: {_local_hms(s['t'], offset)}{language} \"{s['text']}\"")
    if snapshot["sounds"]:
        lines.append("sounds: " + "; ".join(f"{_local_hms(s['t'], offset)} {s['label'].replace('_noise', '')} "
                                            f"{round(s['db'])}dB" for s in snapshot["sounds"]))
    gas = snapshot.get("gas") or {}
    if gas:
        part = " ".join(str(x) for x in (gas.get("gas"), f"{gas['concentration_ppm']}ppm"
                                         if "concentration_ppm" in gas else None, gas.get("trend")) if x)
        if "confidence" in gas:
            part += f" conf{gas['confidence']:.2f}"
        if gas.get("alerts"):
            name = f" {gas.get('gas')}" if gas.get("gas") else ""
            part += " alerts " + ", ".join(a[:-len(name)] if name and a.endswith(name) else a
                                           for a in gas["alerts"])
        lines.append("gas: " + part.strip())
    if snapshot["alerts"]:
        lines.append("alerts: " + "; ".join(snapshot["alerts"]))
    system = snapshot.get("system", {})
    parts = []
    if system.get("mic_state"):
        parts.append(f"mic {system['mic_state']}")
    if "scene_active" in system:
        parts.append("scene active" if system["scene_active"] else "scene idle")
    if system.get("cpu_temp_c") is not None:
        parts.append(f"cpu {round(system['cpu_temp_c'])}C")
    if system.get("events_in_window"):
        parts.append(f"{system['events_in_window']} events")
    if parts:
        lines.append("system: " + ", ".join(parts))

    text = "\n".join(lines)
    while len(text) > max_chars and len(lines) > 2:
        # drop the oldest transcript first, then the last optional line
        speech = [i for i, line in enumerate(lines) if line.startswith("speech: ")]
        lines.pop(speech[0] if speech else len(lines) - 1)
        text = "\n".join(lines)
    return text[:max_chars]
