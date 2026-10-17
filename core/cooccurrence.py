"""
Cross-modal auto-linking (F9 / TOD-20): a voice-only identity is merged into the face identity of the person who is
alone in view while that voice speaks.

``PresenceTracker`` follows who is visible per location from ``object_detected`` (person tracks appear / disappear)
and ``person_identified`` (track → person_id) events. ``VoiceFaceLinker.on_segment`` counts a co-occurrence when
exactly one person track — with a face identity — is visible for ≥ ``presence_ratio`` of the speech segment. After
``min_cooccurrences`` consistent ones and no conflicting ones (the same voice with another face), the voice identity
is merged into the face identity (``identity_linked``). Known identities are never reassigned automatically.
"""
import collections
import time
from typing import Any, Callable, Counter, Deque, Dict, Optional, Tuple

from core.events import Event
from core.identity import IdentityRegistry

UNIDENTIFIED = ""  # a visible person track whose face has not been identified (yet)


class PresenceTracker:
    def __init__(self, location_id: str = "home", history_sec: float = 300.0,
                 clock: Callable[[], float] = time.time):
        self.location_id = location_id
        self.history_sec = history_sec
        self.clock = clock
        self.tracks: Dict[str, Dict[Any, str]] = collections.defaultdict(dict)     # location → track → person
        self.changes: Dict[str, Deque[Tuple[float, Tuple[str, ...]]]] = collections.defaultdict(collections.deque)

    def on_event(self, event: Event, now: Optional[float] = None) -> None:
        p = event.payload
        location = p.get("location_id") or self.location_id
        tracks = self.tracks[location]
        if event.type == "object_detected" and p.get("class") == "person":
            if p.get("state") == "disappeared":
                tracks.pop(p.get("track_id"), None)
            else:
                tracks.setdefault(p.get("track_id"), UNIDENTIFIED)
        elif event.type == "person_identified" and p.get("track_id") is not None:
            tracks[p["track_id"]] = p.get("person_id") or UNIDENTIFIED
        else:
            return
        self._record(location, self.clock() if now is None else now)

    def _record(self, location: str, now: float) -> None:
        changes = self.changes[location]
        changes.append((now, tuple(sorted(self.tracks[location].values()))))
        while len(changes) > 1 and changes[1][0] < now - self.history_sec:
            changes.popleft()  # keep one change older than the horizon: the state at its start

    def visible(self, location: str, at: float) -> Tuple[str, ...]:
        state: Tuple[str, ...] = ()
        for ts, people in self.changes.get(location, ()):
            if ts > at:
                break
            state = people
        return state

    def sole_person(self, location: str, start: float, end: float, ratio: float = 0.8) -> Optional[str]:
        """
        :return: The person_id visible alone (one person track, identified) for ≥ ratio of [start, end], or None
        """
        if end <= start:
            return None
        points = [start] + [ts for ts, _ in self.changes.get(location, ()) if start < ts < end] + [end]
        time_alone: Counter[str] = collections.Counter()
        for a, b in zip(points, points[1:], strict=False):
            people = self.visible(location, a)
            if len(people) == 1 and people[0] != UNIDENTIFIED:
                time_alone[people[0]] += b - a
        if not time_alone:
            return None
        person, seconds = time_alone.most_common(1)[0]
        return person if seconds >= ratio * (end - start) else None


class VoiceFaceLinker:
    def __init__(self, registry: IdentityRegistry, presence: PresenceTracker, min_cooccurrences: int = 3,
                 presence_ratio: float = 0.8):
        self.registry = registry
        self.presence = presence
        self.min_cooccurrences = min_cooccurrences
        self.presence_ratio = presence_ratio
        self.cooccurrences: Dict[str, Counter[str]] = collections.defaultdict(collections.Counter)

    def on_segment(self, voice_person_id: str, location_id: Optional[str], start: float,
                   end: float) -> Optional[Event]:
        """
        :param voice_person_id: Identity the speaker module gave this speech segment
        :param start: Segment start, wall-clock seconds (same clock as the presence events)
        :return: The ``identity_linked`` event when the voice was linked now, else None
        """
        registry = self.registry
        voice = registry.resolve(voice_person_id)
        if voice is None or registry.is_known(voice) or registry.counts[voice]["face"] > 0:
            return None  # only voice-only strangers are linked automatically
        face = registry.resolve(self.presence.sole_person(location_id or self.presence.location_id, start, end,
                                                          self.presence_ratio))
        if face is None or face == voice or registry.counts[face]["face"] == 0:
            return None
        counts = self.cooccurrences[voice]
        counts[face] += 1
        if len(counts) > 1 or counts[face] < self.min_cooccurrences:
            return None  # conflicting faces for this voice, or not enough evidence yet
        del self.cooccurrences[voice]
        return registry.merge(voice, face, action="link_voice")
