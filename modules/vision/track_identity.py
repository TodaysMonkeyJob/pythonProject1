"""
Track-level identity decisions (F14 / TOD-25): identity is decided per person track, not per frame.

* A track collects its best ``good`` face embeddings (up to ``track_good_faces``) and the latest ``usable`` ones.
* It is assigned to a person when one face is very similar (≥ ``strong_threshold``) or when ≥ ``vote_ratio`` of its
  good faces match the same person (usable faces vote only while the track has no good face, and need 2 votes).
* Once identified, the track inherits the identity: later fragments are not matched on their own and can never
  spawn an unknown. Good faces still verify it — ByteTrack may hand the id to someone else — and after
  ``switch_confirmations`` consecutive mismatches the track starts collecting again.
* A new ``unknown_N`` is created only from ≥ ``min_good_faces_for_unknown`` good faces seen over ≥
  ``min_unknown_span_sec`` that match nobody; until then the track is ``unidentified`` ("person" on the dashboard).
"""
from collections import Counter, deque
from dataclasses import dataclass, field, fields
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from core.identity import IdentityRegistry


@dataclass
class TrackConfig:
    face_threshold: float = 0.45
    unknown_threshold: float = 0.45
    strong_threshold: float = 0.60
    vote_ratio: float = 0.6
    min_good_faces_for_unknown: int = 3
    min_unknown_span_sec: float = 1.0
    track_good_faces: int = 10
    verify_margin: float = 0.10
    switch_confirmations: int = 2
    enrich_min_similarity: float = 0.30     # inherited good faces this similar to the person enter its gallery

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TrackConfig":
        data = data or {}
        return cls(**{f.name: type(f.default)(data[f.name]) for f in fields(cls) if f.name in data})


@dataclass
class Observation:
    t: float
    embedding: np.ndarray
    quality: Dict[str, Any]                 # face_quality.assess() result
    match: Optional[Tuple[Optional[str], float]] = None
    version: Any = None                     # registry data_version the match was computed for

    @property
    def tier(self) -> str:
        return self.quality["tier"]


@dataclass
class TrackFaces:
    good: List[Observation] = field(default_factory=list)
    usable: Deque[Observation] = field(default_factory=lambda: deque(maxlen=10))
    good_seen: int = 0
    first_good: Optional[float] = None
    last_good: Optional[float] = None
    mismatches: int = 0

    def add(self, obs: Observation, keep: int) -> None:
        if obs.tier == "good":
            self.good_seen += 1
            self.first_good = obs.t if self.first_good is None else self.first_good
            self.last_good = obs.t
            self.good.append(obs)
            self.good.sort(key=lambda o: -o.quality["score"])
            del self.good[keep:]
        else:
            self.usable.append(obs)


@dataclass
class Decision:
    person_id: Optional[str]
    similarity: float
    how: str            # strong / vote / created / verified / switched / reset


class TrackIdentityResolver:
    """
    Per-track state machine over the shared ``IdentityRegistry``. The caller (``IdentityEngine``) keeps the track's
    current ``person_id`` and passes it in; the resolver keeps the collected faces.
    """

    def __init__(self, registry: IdentityRegistry, cfg: Optional[TrackConfig] = None):
        self.registry = registry
        self.cfg = cfg or TrackConfig()
        self.faces: Dict[Any, TrackFaces] = {}

    def drop(self, track_key: Any) -> None:
        self.faces.pop(track_key, None)

    def observe(self, track_key: Any, current: Optional[str], obs: Observation,
                create_details: Optional[Dict[str, Any]] = None) -> Optional[Decision]:
        """
        :param current: The track's person (None = unidentified)
        :param obs: A ``good`` or ``usable`` face of this track
        :param create_details: Audit details for a new unknown (e.g. continuity hints)
        :return: A decision when the track's person changes or was (re)confirmed, else None
        """
        state = self.faces.setdefault(track_key, TrackFaces())
        if current is not None and self.registry.resolve(current) is not None:
            return self._verify(state, self.registry.resolve(current), obs)
        state.add(obs, self.cfg.track_good_faces)
        return self._decide(state, create_details)

    # --- identified tracks: inherit, verify with good faces only -----------------------------------------------------

    def _verify(self, state: TrackFaces, current: str, obs: Observation) -> Optional[Decision]:
        if obs.tier != "good":
            return None                      # inherited: a fragment never changes the label
        registry, cfg = self.registry, self.cfg
        known = registry.is_known(current)
        similarity = registry.similarity(current, "face", obs.embedding)
        other, other_similarity = registry.match_face(obs.embedding, cfg.face_threshold, known=True)
        if not known and other is not None and other != current and other_similarity > similarity:
            state.mismatches = 0             # an unknown turns out to be a registered person (e.g. just enrolled)
            return Decision(other, other_similarity, "switched")
        threshold = cfg.face_threshold if known else cfg.unknown_threshold
        if similarity >= threshold - cfg.verify_margin:
            state.mismatches = 0
            if similarity >= cfg.enrich_min_similarity:
                registry.add_embedding(current, "face", obs.embedding, meta=obs.quality)
            return Decision(current, similarity, "verified")
        state.mismatches += 1
        if state.mismatches < cfg.switch_confirmations:
            return None
        # Another person took over the track id: start collecting from scratch
        fresh = TrackFaces()
        fresh.add(obs, cfg.track_good_faces)
        state.__dict__.update(fresh.__dict__)
        return self._decide(state, None) or Decision(None, 0.0, "reset")

    # --- unidentified tracks: strong match, voting, or a new unknown ------------------------------------------------

    def _match(self, obs: Observation) -> Tuple[Optional[str], float]:
        version = self.registry._data_version
        if obs.match is None or obs.version != version:
            person, similarity = self.registry.match_face(obs.embedding, self.cfg.face_threshold, known=True)
            if person is None:
                unknown, unknown_similarity = self.registry.match_face(obs.embedding, self.cfg.unknown_threshold,
                                                                       known=False)
                if unknown is not None or unknown_similarity > similarity:
                    person, similarity = unknown, unknown_similarity
            obs.match, obs.version = (person, similarity), version
        return obs.match

    def _decide(self, state: TrackFaces, create_details: Optional[Dict[str, Any]]) -> Optional[Decision]:
        cfg = self.cfg
        everything = state.good + list(state.usable)
        matches = [(obs, *self._match(obs)) for obs in everything]
        strong = [(sim, pid) for _obs, pid, sim in matches if pid is not None and sim >= cfg.strong_threshold]
        if strong:
            similarity, person = max(strong)
            return self._assign(state, person, similarity, "strong")
        voters = [m for m in matches if m[0].tier == "good"] or [m for m in matches if m[0].tier == "usable"]
        votes = Counter(pid for _obs, pid, _sim in voters if pid is not None)
        if votes:
            person, count = votes.most_common(1)[0]
            needed = 1 if voters[0][0].tier == "good" else 2
            if count >= needed and count / len(voters) >= cfg.vote_ratio:
                similarity = float(np.mean([sim for _obs, pid, sim in voters if pid == person]))
                return self._assign(state, person, similarity, "vote")
        good = [m for m in matches if m[0].tier == "good"]
        if (state.good_seen >= cfg.min_good_faces_for_unknown and len(good) >= cfg.min_good_faces_for_unknown
                and state.last_good - state.first_good >= cfg.min_unknown_span_sec
                and all(pid is None for _obs, pid, _sim in good)):
            best = state.good[0]
            person = self.registry.create_unknown("face", best.embedding, quality=best.quality["score"],
                                                  meta=best.quality, details=create_details)
            for obs in state.good[1:]:
                self.registry.add_embedding(person, "face", obs.embedding, meta=obs.quality)
            return self._finish(state, Decision(person, 1.0, "created"))
        return None

    def _assign(self, state: TrackFaces, person: str, similarity: float, how: str) -> Decision:
        for obs in state.good:      # the track's good faces now enrich that person (admission keeps it bounded)
            if self.registry.similarity(person, "face", obs.embedding) >= self.cfg.enrich_min_similarity:
                self.registry.add_embedding(person, "face", obs.embedding, meta=obs.quality)
        return self._finish(state, Decision(person, similarity, how))

    @staticmethod
    def _finish(state: TrackFaces, decision: Decision) -> Decision:
        state.good.clear()
        state.usable.clear()
        state.mismatches = 0
        return decision
