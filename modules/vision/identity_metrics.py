"""
Identity metrics for the thesis (F14 / TOD-25): ``reports/identity_metrics.json``, one entry per UTC day.

Registry-derived numbers (unknowns created per hour, auto / review / manual merges, undos, "not the same person"
decisions, junk deletions, gallery sizes) are recomputed from the ``identity_events`` audit, so days before this
feature are filled too — that is the before / after comparison of the duplicate rate. Face counters (good / usable
/ rejected by reason) only exist from the live vision engine and are accumulated across restarts.

    python -m tools.identity metrics        # recompute every day from the audit
"""
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional

from core.identity import IdentityRegistry

MERGE_KINDS = {"auto_merge": "auto_merges", "review_merge": "review_merges", "merge": "manual_merges",
               "link_voice": "voice_links"}


def utc_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def audit_stats(registry: IdentityRegistry, days: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
    """:return: {day: registry-derived numbers} for the given days (default: every day in the audit)"""
    wanted = set(days) if days is not None else None
    out: Dict[str, Dict[str, Any]] = {}
    with registry._lock:
        rows = registry.db.execute("SELECT ts, action, details_json FROM identity_events ORDER BY id").fetchall()
    for ts, action, details in rows:
        day = utc_day(ts)
        if wanted is not None and day not in wanted:
            continue
        entry = out.setdefault(day, {"unknowns_created": 0, "unknowns_created_per_hour": {},
                                     "unknowns_by_modality": {}, "auto_merges": 0,
                                     "review_merges": 0, "manual_merges": 0, "voice_links": 0, "undos": 0,
                                     "not_same": 0, "junk_deleted": 0})
        info = json.loads(details or "{}")
        if action == "create" and info.get("modality") is not None:
            entry["unknowns_created"] += 1
            hour = datetime.fromtimestamp(ts, timezone.utc).strftime("%H")
            entry["unknowns_created_per_hour"][hour] = entry["unknowns_created_per_hour"].get(hour, 0) + 1
            modality = entry["unknowns_by_modality"]
            modality[info["modality"]] = modality.get(info["modality"], 0) + 1
        elif action in MERGE_KINDS:
            entry[MERGE_KINDS[action]] += 1
        elif action == "unmerge":
            entry["undos"] += 1
        elif action == "not_same":
            entry["not_same"] += 1
        elif action == "delete" and info.get("reason") == "junk":
            entry["junk_deleted"] += 1
    return out


def gallery_sizes(registry: IdentityRegistry) -> Dict[str, Any]:
    """Face gallery sizes keyed by a short person_id (no names in a report that may be shared)."""
    people = [p for p in registry.list_persons() if p["face_samples"]]
    sizes = {f"{'known' if p['is_known'] else 'unknown'}:{p['person_id'][:8]}": p["face_samples"] for p in people}
    known = [p["face_samples"] for p in people if p["is_known"]]
    return {"per_person": sizes, "known_people": len(known), "unknown_people": len(people) - len(known),
            "max": max(sizes.values(), default=0), "limit": registry.max_faces}


class MetricsReport:
    def __init__(self, path: str = "reports/identity_metrics.json"):
        self.path = path

    def load(self) -> Dict[str, Any]:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def update(self, registry: IdentityRegistry, face_counters: Optional[Mapping[str, int]] = None,
               now: Optional[float] = None, all_days: bool = False) -> Dict[str, Any]:
        """
        Adds the face counters since the last call to today's entry and refreshes the audit numbers.

        :param face_counters: ``IdentityEngine.stats`` deltas (faces_good / faces_usable / rejected:<reason> …)
        :param all_days: Recompute every day in the audit (CLI)
        """
        now = registry.clock() if now is None else now
        today = utc_day(now)
        data = self.load()
        days = data.setdefault("days", {})
        for day, stats in audit_stats(registry, None if all_days else [today]).items():
            days.setdefault(day, {}).update(stats)
        entry = days.setdefault(today, {})
        faces = Counter(entry.get("faces", {}))
        rejected = Counter(entry.get("rejected_faces", {}))
        for key, value in (face_counters or {}).items():
            if key.startswith("rejected:"):
                rejected[key.split(":", 1)[1]] += value
            elif key.startswith("faces_"):
                faces[key[len("faces_"):]] += value
        entry["faces"], entry["rejected_faces"] = dict(faces), dict(rejected)
        entry["gallery_sizes"] = gallery_sizes(registry)
        data["updated_at"] = datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds")
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, self.path)
        return data
