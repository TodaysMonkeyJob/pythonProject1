"""
Duplicate identities (F14 / TOD-25): finds ``unknown_N`` people that are really someone already registered (or the
same stranger twice) and merges them — deterministic rules and embedding statistics, no model.

Candidate pairs: every unknown vs every known person and unknown vs unknown. Evidence:

* gallery similarity (main score): ``core.gallery.gallery_similarity`` = mean of the top-k pairwise cosine
  similarities between the two galleries, mixed with the centroid cosine;
* track continuity (bonus): the unknown was created right after a track of that person was lost nearby on the same
  camera (``continuity`` hint written by the vision engine into the ``create`` audit row);
* voice link (bonus): both have voice samples and their voice centroids match (``identity.voice_threshold``);
* cannot-link (hard block): seen in one frame, a user's "not the same person", or an undone merge.

Decision: ``score ≥ auto_merge_threshold`` → merge (a known person is the target, else the older unknown);
``review_threshold ≤ score < auto_merge_threshold`` → the review queue (dashboard "Possible duplicates"); below →
ignored. Groups are built with average linkage (never single linkage: A~B and B~C do not pull in C unless C is
close to the group on average) and a group is only merged when its merged centroid still matches every member
centroid ≥ ``review_threshold`` (chain-merge guard). Junk unknowns (< ``min_good_faces`` faces, no voice, not seen
for ``stale_unknown_days``) are deleted; embeddings archived by merges older than ``undo_days`` are purged.

Runs every ``dedup_interval_min`` in the identity process and on demand (dashboard / ``python -m tools.identity
dedup``).
"""
import json
import time
from dataclasses import dataclass, fields
from itertools import combinations
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from core.events import Event, make_event
from core.gallery import gallery_similarity
from core.identity import IdentityRegistry, normalize
from system_setup.logger_setup import init_logger

logger = init_logger()


@dataclass
class DedupConfig:
    """``identity.dedup`` config section."""
    dedup_interval_min: float = 10.0
    auto_merge_threshold: float = 0.62
    review_threshold: float = 0.42
    prefilter_margin: float = 0.20      # pairs with a centroid cosine below review − margin are not scored
    top_k: int = 5
    continuity_bonus: float = 0.05
    voice_bonus: float = 0.05
    stale_unknown_days: float = 3.0
    min_good_faces: int = 3
    max_candidates_per_person: int = 3  # review-queue entries per unknown (best first)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "DedupConfig":
        data = data or {}
        return cls(**{f.name: type(f.default)(data[f.name]) for f in fields(cls) if f.name in data})


Pair = FrozenSet[str]


@dataclass
class Plan:
    merges: List[Tuple[str, str, float, Dict[str, Any]]]     # (source, target, score, evidence)
    reviews: List[Tuple[str, str, float, Dict[str, Any]]]    # (person_a, person_b, score, evidence)
    junk: List[str]


class DedupJob:
    def __init__(self, registry: IdentityRegistry, cfg: Optional[DedupConfig] = None,
                 publish: Optional[Callable[[Event], None]] = None, clock: Callable[[], float] = time.time):
        self.registry = registry
        self.cfg = cfg or DedupConfig()
        self.publish = publish
        self.clock = clock

    # --- evidence -----------------------------------------------------------------------------------------------

    def _continuity(self) -> Dict[str, Set[str]]:
        """:return: unknown person → people whose track was lost right before it was created"""
        out: Dict[str, Set[str]] = {}
        with self.registry._lock:
            rows = self.registry.db.execute("SELECT person_id, details_json FROM identity_events WHERE action = "
                                            "'create' AND details_json LIKE '%continuity%'").fetchall()
        for person_id, details in rows:
            pid = self.registry.resolve(person_id)
            hits = {self.registry.resolve(p) for p in json.loads(details or "{}").get("continuity", [])}
            if pid:
                out.setdefault(pid, set()).update(h for h in hits if h and h != pid)
        return out

    def _voice_match(self, a: str, b: str) -> bool:
        va, vb = self.registry.centroid(a, "voice"), self.registry.centroid(b, "voice")
        return va is not None and vb is not None and float(va @ vb) >= self.registry.voice_threshold

    def score(self, a: str, b: str, continuity: Optional[Dict[str, Set[str]]] = None,
              galleries: Optional[Dict[str, np.ndarray]] = None) -> Tuple[float, Dict[str, Any]]:
        """:return: (score, evidence) of one pair of live people"""
        galleries = galleries or {}
        ga = galleries[a] if a in galleries else self.registry.gallery(a)
        gb = galleries[b] if b in galleries else self.registry.gallery(b)
        similarity, centroid = gallery_similarity(ga, gb, self.cfg.top_k)
        continuity = continuity if continuity is not None else self._continuity()
        evidence = {"similarity": round(similarity, 3), "centroid": round(centroid, 3),
                    "continuity": b in continuity.get(a, ()) or a in continuity.get(b, ()),
                    "voice": self._voice_match(a, b), "samples": [len(ga), len(gb)]}
        score = similarity + self.cfg.continuity_bonus * evidence["continuity"] \
            + self.cfg.voice_bonus * evidence["voice"]
        return min(1.0, score), evidence

    # --- planning -----------------------------------------------------------------------------------------------

    def junk(self) -> List[str]:
        """Unknowns that never gathered enough good faces, have no voice and were not seen for a while."""
        cutoff = self.clock() - self.cfg.stale_unknown_days * 86400
        reg = self.registry
        return [pid for pid, p in reg.persons.items()
                if not p["merged_into"] and not p["is_known"] and (p["last_seen"] or p["created_at"]) < cutoff
                and reg.counts[pid]["face"] < self.cfg.min_good_faces and reg.counts[pid]["voice"] == 0]

    def plan(self) -> Plan:
        reg, cfg = self.registry, self.cfg
        reg.poll()
        junk = set(self.junk())
        live = [pid for pid, p in reg.persons.items() if not p["merged_into"] and pid not in junk
                and reg.counts[pid]["face"]]
        galleries = {pid: reg.gallery(pid) for pid in live}
        centroids = {pid: normalize(g.mean(axis=0)) for pid, g in galleries.items()}
        known = {pid for pid in live if reg.persons[pid]["is_known"]}
        continuity = self._continuity()
        scores: Dict[Pair, float] = {}
        evidence: Dict[Pair, Dict[str, Any]] = {}
        for a, b in combinations(live, 2):
            if a in known and b in known:
                continue                         # two registered people are never merged automatically
            if reg.blocked(a, b) or float(centroids[a] @ centroids[b]) < cfg.review_threshold - cfg.prefilter_margin:
                continue
            scores[frozenset((a, b))], evidence[frozenset((a, b))] = self.score(a, b, continuity, galleries)

        # Average-linkage agglomeration over groups (a group holds at most one known person)
        group_of: Dict[str, int] = {pid: i for i, pid in enumerate(live)}
        members: Dict[int, Set[str]] = {i: {pid} for i, pid in enumerate(live)}

        def linkage(x: Set[str], y: Set[str]) -> float:
            return float(np.mean([scores.get(frozenset((a, b)), 0.0) for a in x for b in y]))

        def mergeable(x: Set[str], y: Set[str]) -> bool:
            if len((x | y) & known) > 1 or any(reg.blocked(a, b) for a in x for b in y):
                return False
            merged = normalize(np.vstack([galleries[p] for p in x | y]).mean(axis=0))
            return all(float(merged @ centroids[p]) >= cfg.review_threshold for p in x | y)   # chain guard

        refused: Set[FrozenSet[FrozenSet[str]]] = set()
        while True:
            best, best_pair = cfg.auto_merge_threshold, None
            for gi, gj in {tuple(sorted((group_of[a], group_of[b]))) for a, b in (tuple(k) for k in scores)}:
                if gi == gj or frozenset((frozenset(members[gi]), frozenset(members[gj]))) in refused:
                    continue
                value = linkage(members[gi], members[gj])
                if value >= best:
                    best, best_pair = value, (gi, gj)
            if best_pair is None:
                break
            gi, gj = best_pair
            if mergeable(members[gi], members[gj]):
                for pid in members[gj]:
                    group_of[pid] = gi
                members[gi] |= members.pop(gj)
            else:
                refused.add(frozenset((frozenset(members[gi]), frozenset(members[gj]))))
        groups = list(members.values())

        merges = []
        targets: Dict[str, str] = {}
        for group in groups:
            if len(group) < 2:
                targets[next(iter(group))] = next(iter(group))
                continue
            members = sorted(group, key=lambda p: (p not in known, reg.persons[p]["created_at"]))
            target = members[0]
            for source in members[1:]:
                pair = frozenset((source, target))
                score = scores.get(pair, linkage({source}, group - {source}))
                merges.append((source, target, round(score, 3), evidence.get(pair, {})))
            for member in group:
                targets[member] = target

        # Review queue: medium-confidence pairs between the resulting groups
        reviews: Dict[Pair, Tuple[float, Dict[str, Any]]] = {}
        for pair, score in scores.items():
            a, b = (targets[p] for p in pair)
            if a == b or score < cfg.review_threshold:      # ≥ auto but not merged (linkage / guard) → a human
                continue
            key = frozenset((a, b))
            if score > reviews.get(key, (0.0, {}))[0]:
                reviews[key] = (round(score, 3), evidence[pair])
        per_person: Dict[str, int] = {}
        queued = []
        for key, (score, ev) in sorted(reviews.items(), key=lambda kv: -kv[1][0]):
            a, b = sorted(key)
            if any(per_person.get(p, 0) >= cfg.max_candidates_per_person for p in (a, b) if p not in known):
                continue
            for p in (a, b):
                per_person[p] = per_person.get(p, 0) + 1
            queued.append((a, b, score, ev))
        return Plan(merges, queued, sorted(junk))

    # --- execution ----------------------------------------------------------------------------------------------

    def run(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        :param dry_run: Only report what would happen
        :return: {"merged", "queued", "deleted", "purged"} (person ids / counts)
        """
        plan = self.plan()
        report: Dict[str, Any] = {"merged": [], "queued": [], "deleted": [], "purged": 0}
        if dry_run:
            report.update(merged=[(s, t, score) for s, t, score, _ in plan.merges],
                          queued=[(a, b, score) for a, b, score, _ in plan.reviews], deleted=plan.junk)
            return report
        reg = self.registry
        for pid in plan.junk:
            try:
                reg.delete(pid, reason="junk")
                report["deleted"].append(pid)
            except KeyError:
                pass
        for source, target, score, evidence in plan.merges:
            try:
                event = reg.merge(source, target, action="auto_merge", score=score, evidence=evidence)
                report["merged"].append((source, target, score, event.payload.get("event_id")))
            except (KeyError, ValueError) as e:      # merged / deleted by another process meanwhile
                logger.warning(f"dedup: skipped merging {source[:8]} → {target[:8]}: {e}")
        existing = {frozenset((c["person_a"], c["person_b"])) for c in reg.candidates()}
        for a, b, score, evidence in plan.reviews:
            try:
                candidate_id = reg.propose(a, b, score, evidence)
            except KeyError:
                continue
            if candidate_id is None:
                continue
            report["queued"].append(candidate_id)
            if frozenset((a, b)) not in existing and self.publish is not None:
                self.publish(make_event("system", "duplicate_candidate", {
                    "candidate_id": candidate_id, "person_a": a, "person_b": b, "score": score,
                    "names": [reg.name(a), reg.name(b)]}))
        report["purged"] = reg.purge_archived()
        if report["merged"] or report["deleted"] or report["queued"]:
            logger.info(f"dedup: merged {len(report['merged'])}, queued {len(report['queued'])}, deleted "
                        f"{len(report['deleted'])} junk unknowns")
        return report
