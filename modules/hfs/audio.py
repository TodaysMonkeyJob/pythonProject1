"""
Audio subsystem (A1 / TOD-41) → risk 3. Rule base ``config/fuzzy/audio.yaml``.

Over the last ``window_sec`` of the location:

* ``sound_level`` — loudest ``sound`` event, (dBFS + 60) / 60 clamped to 0..1;
* ``sound_type`` — highest ``maps.sound_type`` danger of a labelled sound (``payload.sound_type``; A4 / TOD-44 adds
  the noise-stage classifier, until then 0);
* ``danger_keywords`` — highest grade of the ``keywords`` of the rule base found in ``speech`` texts;
* ``speaker_unknown`` — the least known speaker: known 0, uncertain 0.5, unknown 1.

Data available = an online microphone in the location, or audio events from it in the last ``stale_sec``.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from core.events import Event
from modules.hfs.base import FuzzySubsystem, window

SPEAKER_SCORE = {"known": 0.0, "uncertain": 0.5, "unknown": 1.0}
DB_FLOOR = -60.0


def keyword_grades(data: Dict[Any, List[str]]) -> List[Tuple[float, str]]:
    """:return: [(grade, lower-case phrase)] strongest first"""
    out = [(float(grade), str(phrase).lower()) for grade, phrases in (data or {}).items() for phrase in phrases]
    return sorted(out, reverse=True)


def keyword_grade(text: str, grades: List[Tuple[float, str]]) -> Tuple[float, Optional[str]]:
    """:return: (highest grade, phrase) found in the text (whole-word start), (0, None) without one"""
    lowered = (text or "").lower()
    for grade, phrase in grades:
        if re.search(r"(?<!\w)" + re.escape(phrase), lowered):
            return grade, phrase
    return 0.0, None


class AudioSubsystem(FuzzySubsystem):
    name = "audio"
    device_kind = "mic"
    event_types = ("sound", "speech")

    def __init__(self, config: Dict[str, Any], path: Optional[str] = None, **kwargs):
        super().__init__(config, path, **kwargs)
        self.window_sec = float(self.cfg.get("window_sec", 30))
        self.grades = keyword_grades(self.data.get("keywords"))
        types = (self.data.get("maps") or {}).get("sound_type") or {}
        self.sound_types = {str(k): float(v) for k, v in types.items()}
        # location → [(ts, device, sound_level, sound_type)] / [(ts, device, keyword grade, speaker score)]
        self.sounds: Dict[str, List[tuple]] = {}
        self.speech: Dict[str, List[tuple]] = {}

    def observe(self, event: Event, now: float) -> None:
        super().observe(event, now)
        p = event.payload
        if event.type == "sound":
            db = p.get("rms_db")
            level = 0.0 if db is None else min(1.0, max(0.0, (float(db) - DB_FLOOR) / -DB_FLOOR))
            kind = self.sound_types.get(str(p.get("sound_type")), 0.0) if p.get("sound_type") else 0.0
            self.sounds.setdefault(event.location_id, []).append((now, event.device_id, level, kind))
        else:
            grade, _ = keyword_grade(p.get("text", ""), self.grades)
            speaker = SPEAKER_SCORE.get(str(p.get("speaker_status")), 0.0)
            self.speech.setdefault(event.location_id, []).append((now, event.device_id, grade, speaker))

    def forget_device(self, device_id: str) -> None:
        for store in (self.sounds, self.speech):
            for location_id, items in store.items():
                store[location_id] = [item for item in items if item[1] != device_id]

    def features(self, location_id: str, now: float, available: bool) -> Optional[Dict[str, Any]]:
        if not (available or self.recent(location_id, now)):
            return None
        sounds = window(self.sounds.get(location_id, []), now, self.window_sec)
        speech = window(self.speech.get(location_id, []), now, self.window_sec)
        return {"sound_level": round(max((s[2] for s in sounds), default=0.0), 4),
                "sound_type": round(max((s[3] for s in sounds), default=0.0), 4),
                "danger_keywords": round(max((s[2] for s in speech), default=0.0), 4),
                "speaker_unknown": round(max((s[3] for s in speech), default=0.0), 4)}
