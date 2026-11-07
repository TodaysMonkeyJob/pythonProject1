"""
Audio subsystem (A1 / TOD-41, A4 / TOD-44) → risk 3. Rule base ``config/fuzzy/audio.yaml``.

Over the last ``window_sec`` of the location:

* ``intensity`` — dB above the microphone's own background (``audio_level`` events of the microphone controller,
  ``modules/audio/intensity.py``; the loudest one of the location); without them the loudest ``sound`` event's
  ``relative_db``;
* ``sound_type_danger`` — highest danger of the classified sound types (``sound_type`` events: the background of
  every speech segment and every non-speech sound burst, ``config/audio_types.yaml``); unclassified (no model,
  ``danger: null``) is not evidence;
* ``keyword_danger`` — highest grade of the danger words in the transcripts (``config/danger_words.yaml``,
  ``modules/audio/keywords.py``; the controller's ``danger_words`` or matched here), the matched words are kept for
  the explanation;
* ``speaker_unknown`` — the least known speaker: known 0, uncertain 0.5, unknown 1.

Data available = an online microphone in the location, or audio events from it in the last ``stale_sec``.
"""
from typing import Any, Dict, List, Optional

from core.events import Event
from modules.audio.keywords import DangerWords
from modules.hfs.base import FuzzySubsystem, window

SPEAKER_SCORE = {"known": 0.0, "uncertain": 0.5, "unknown": 1.0}


class AudioSubsystem(FuzzySubsystem):
    name = "audio"
    device_kind = "mic"
    event_types = ("sound", "speech", "sound_type", "audio_level")

    def __init__(self, config: Dict[str, Any], path: Optional[str] = None, words: Optional[DangerWords] = None,
                 **kwargs):
        super().__init__(config, path, **kwargs)
        self.window_sec = float(self.cfg.get("window_sec", 30))
        self.level_stale = float(self.cfg.get("level_stale_sec", 10))
        self.words = words or DangerWords.load(self.cfg.get("danger_words", "config/danger_words.yaml"))
        # location → [(ts, device, …)]
        self.sounds: Dict[str, List[tuple]] = {}          # (ts, device, relative dB)
        self.types: Dict[str, List[tuple]] = {}           # (ts, device, danger, type)
        self.speech: Dict[str, List[tuple]] = {}          # (ts, device, grade, speaker score, words)
        self.levels: Dict[str, Dict[str, tuple]] = {}     # location → {device: (ts, intensity dB)}

    def observe(self, event: Event, now: float) -> None:
        super().observe(event, now)
        p, loc, device = event.payload, event.location_id, event.device_id
        if event.type == "audio_level":
            self.levels.setdefault(loc, {})[device] = (now, float(p.get("intensity_db") or 0.0))
        elif event.type == "sound":
            self.sounds.setdefault(loc, []).append((now, device, float(p.get("relative_db") or 0.0)))
        elif event.type == "sound_type":
            if p.get("danger") is not None:
                self.types.setdefault(loc, []).append((now, device, float(p["danger"]), p.get("type")))
        else:
            words = p.get("danger_words")
            if words is None:
                grade, matches = self.words.grade(p.get("text", ""))
                words = {"grade": grade, "matches": [m.payload() for m in matches[:3]]}
            speaker = SPEAKER_SCORE.get(str(p.get("speaker_status")), 0.0)
            self.speech.setdefault(loc, []).append((now, device, float(words.get("grade") or 0.0), speaker,
                                                    words.get("matches") or []))

    def forget_device(self, device_id: str) -> None:
        for store in (self.sounds, self.types, self.speech):
            for location_id, items in store.items():
                store[location_id] = [item for item in items if item[1] != device_id]
        for levels in self.levels.values():
            levels.pop(device_id, None)

    def intensity(self, location_id: str, now: float) -> float:
        levels = [v for ts, v in (self.levels.get(location_id) or {}).values() if now - ts <= self.level_stale]
        if levels:
            return round(max(levels), 2)
        sounds = window(self.sounds.get(location_id, []), now, self.window_sec)
        return round(max((s[2] for s in sounds), default=0.0), 2)

    def features(self, location_id: str, now: float, available: bool) -> Optional[Dict[str, Any]]:
        if not (available or self.recent(location_id, now)):
            return None
        types = window(self.types.get(location_id, []), now, self.window_sec)
        speech = window(self.speech.get(location_id, []), now, self.window_sec)
        loudest_type = max(types, key=lambda t: t[2], default=None)
        words = [m for s in speech for m in s[4]]
        return {"intensity": self.intensity(location_id, now),
                "sound_type_danger": round(loudest_type[2], 4) if loudest_type else 0.0,
                "keyword_danger": round(max((s[2] for s in speech), default=0.0), 4),
                "speaker_unknown": round(max((s[3] for s in speech), default=0.0), 4),
                "_danger_words": sorted(words, key=lambda m: -m.get("grade", 0))[:3],
                "_sound_type": loudest_type[3] if loudest_type else None}
