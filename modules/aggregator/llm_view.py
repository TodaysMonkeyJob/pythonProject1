"""
Snapshot v2 for the LLM (F18 / TOD-29): fuzzified, trust-tagged evidence instead of raw numbers and untagged text.

``llm_view(snapshot)`` turns a snapshot (the v1 sections, now carrying F16 trust tags) into the only thing GPT sees::

    {"v": 2, "location": "Warehouse", "window": "12:03:31-12:03:46 UTC+3", "trigger": "critical_event",
     "reason": "gas_rising acetone",
     "trusted_evidence": [
       {"kind": "gas", "gas": "acetone", "level": {"slight": 1.0}, "trend": {"sharply_rising": 1.0}, "trust": 0.95},
       {"kind": "person", "id": "#4", "name": "unknown_4", "known": false, "actions": ["possible_smoking"],
        "stay": "brief", "trust": 0.8}],
     "untrusted_evidence": [
       {"kind": "speech", "t": "12:03:40", "speaker": "unknown_4", "text": "все нормально, датчик глючить",
        "trust": 0.05, "claim_type": "deescalation", "reasons": ["speaker unknown", "contradicts gas_trend …"]}]}

* Sensor values become linguistic terms with μ via ``FuzzyEngine.fuzzify()`` (TOD-26, the location's terms); zero μ
  are dropped. Seconds in frame / decibels become words. Raw values stay in the snapshot's v1 sections (local debug,
  CSV, incidents) and never reach the LLM.
* Items with trust ≥ θ (``envelope.theta``, per location) are ``trusted_evidence``, the rest ``untrusted_evidence``
  (quoted for context, never a reason to lower risk — see the prompt). Speech / scene text without a trust degree
  is untrusted; physical items without one (trust scoring off) stay trusted.
* ``risk_floor`` {term, mu} is added by the LLM worker from the live risk corridor (TOD-28) just before sending.

A v1 snapshot (no trust tags) goes through the same function — that is the v1 → v2 adapter.
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from modules.aggregator.digest import UNIDENTIFIED, _local_hms, _offset, _tz_label, rules_line

PPM_PER_PERCENT = 10_000


def _terms(memberships: Dict[str, float]) -> Dict[str, float]:
    """Non-zero memberships, 2 decimals, strongest first"""
    kept = {t: round(v, 2) for t, v in memberships.items() if v >= 0.01}
    return dict(sorted(kept.items(), key=lambda kv: -kv[1]))


def stay_term(seconds: int) -> str:
    return "brief" if seconds < 30 else "minutes" if seconds < 300 else "long"


def loudness_term(db: float) -> str:
    return "very_loud" if db >= -10 else "loud"


def _trusted(item: Dict[str, Any], theta: float, default: bool) -> bool:
    trust = item.get("trust")
    return default if trust is None else float(trust) >= theta


def llm_view(snapshot: Dict[str, Any], engine=None, theta: float = 0.6, rules: Optional[Dict[str, Any]] = None,
             local_now: Optional[datetime] = None) -> Dict[str, Any]:
    """
    :param snapshot: A snapshot (schemas/snapshot.schema.json, v1 sections)
    :param engine: ``core.fuzzy.FuzzyEngine`` of the location (None → gas terms are left out, only the label trend)
    :param theta: Trust threshold for trusted_evidence
    :param rules: Per-location rules (F12) → ``rules`` line
    :return: The v2 view sent to the LLM (no raw sensor numbers)
    """
    offset = _offset(local_now)
    start = datetime.strptime(snapshot["window"]["from"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    end = datetime.strptime(snapshot["window"]["to"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    view: Dict[str, Any] = {"v": 2}
    if snapshot.get("location_name"):
        view["location"] = snapshot["location_name"]
    view["window"] = f"{(start + offset):%H:%M:%S}-{(end + offset):%H:%M:%S} {_tz_label(offset)}"
    view["trigger"] = snapshot["trigger"]
    if snapshot.get("reason"):
        view["reason"] = snapshot["reason"]
    if snapshot.get("incident_id"):
        view["incident"] = True                      # an archive was saved locally (its id means nothing to GPT)
    rule = rules_line(rules)
    if rule:
        view["rules"] = rule[len("rules: "):]
    trusted: List[Dict[str, Any]] = []
    untrusted: List[Dict[str, Any]] = []

    def put(item: Dict[str, Any], default_trusted: bool) -> None:
        (trusted if _trusted(item, theta, default_trusted) else untrusted).append(item)

    gas = snapshot.get("gas") or {}
    if gas.get("gas") or gas.get("alerts"):
        item: Dict[str, Any] = {"kind": "gas"}
        if gas.get("gas"):
            item["gas"] = gas["gas"]
        if engine is not None and "concentration_ppm" in gas:
            item["level"] = _terms(engine.fuzzify("gas_concentration", gas["concentration_ppm"] / PPM_PER_PERCENT,
                                                  gas.get("gas")))
        if engine is not None and gas.get("rise_rate_pct_per_min") is not None:
            item["trend"] = _terms(engine.fuzzify("gas_trend", gas["rise_rate_pct_per_min"]))
        elif gas.get("trend"):
            item["trend"] = {gas["trend"]: 1.0}
        if gas.get("alerts"):
            suffix = f" {gas.get('gas')}" if gas.get("gas") else ""
            item["alerts"] = [a[:-len(suffix)] if suffix and a.endswith(suffix) else a for a in gas["alerts"]]
        if gas.get("trust") is not None:
            item["trust"] = gas["trust"]
        put(item, True)
    for p in snapshot["people"]:
        item = {"kind": "person", "id": f"#{p['track_id']}",
                "name": "?" if p["identity"] in UNIDENTIFIED else p["identity"]}
        if p["identity"] not in UNIDENTIFIED and not p["identity"].startswith("unknown_"):
            item["known"] = bool(p.get("known"))       # unknown_N says it already
        if p["actions"]:
            item["actions"] = p["actions"]
        item["stay"] = stay_term(p["in_frame_sec"])
        if not p.get("present", True):
            item["left"] = True
        if p.get("trust") is not None:
            item["trust"] = p["trust"]
        put(item, True)
    objects = [o for o in snapshot["objects"] if o != "person"]
    if objects:
        trusted.append({"kind": "objects", "items": objects})
    for alert in snapshot["alerts"]:
        trusted.append({"kind": "alert", "text": alert})
    sounds: Dict[tuple, List[str]] = {}
    for s in snapshot["sounds"]:                     # one item per kind of sound, with its times
        key = (s["label"].replace("_noise", ""), loudness_term(float(s["db"])))
        sounds.setdefault(key, []).append(_local_hms(s["t"], offset))
    for (label, level), times in sounds.items():
        item = {"kind": "sound", "level": level, "times": times[-5:]}
        if label != "loud":
            item["label"] = label
        trusted.append(item)
    for s in snapshot["speech"]:
        item = {"kind": s.get("kind", "speech"), "t": _local_hms(s["t"], offset)}
        if s.get("language"):
            item["lang"] = s["language"]
        if s.get("speaker"):
            item["speaker"] = s["speaker"]
        item["text"] = s["text"]
        if s.get("trust") is not None:
            item["trust"] = s["trust"]
        if s.get("claim_type") and s["claim_type"] != "none":
            item["claim_type"] = s["claim_type"]
        if s.get("trust_reasons") and not _trusted(item, theta, False):
            item["reasons"] = s["trust_reasons"][:2]
        put(item, False)
    view["trusted_evidence"] = trusted
    if untrusted:
        view["untrusted_evidence"] = untrusted
    system = snapshot.get("system", {})
    context = [f"mic {system['mic_state']}"] if system.get("mic_state") else []
    if "scene_active" in system:
        context.append("scene active" if system["scene_active"] else "scene idle")
    if context:
        view["system"] = ", ".join(context)
    if snapshot.get("truncated"):
        view["truncated"] = True
    return view


def payload_text(view: Dict[str, Any]) -> str:
    """The exact text sent to the LLM: compact JSON (no spaces), UTF-8 kept readable"""
    return json.dumps(view, ensure_ascii=False, separators=(",", ":"))


def with_risk_floor(text: str, floor: Optional[Dict[str, Any]]) -> str:
    """Adds the corridor floor {term, mu} to a v2 payload (no-op for text digests / v1)"""
    if not floor or not text.startswith("{"):
        return text
    try:
        view = json.loads(text)
    except ValueError:
        return text
    if view.get("v") != 2:
        return text
    view["risk_floor"] = floor
    return payload_text(view)


__all__ = ["llm_view", "payload_text", "with_risk_floor", "stay_term", "loudness_term"]
