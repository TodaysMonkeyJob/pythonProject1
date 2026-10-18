"""
Speaker-attributed transcripts (F10 / TOD-21): who said what, when, where.

``align()`` gives each Whisper segment (``start`` / ``end`` in seconds from the chunk start) the speaker turn it
overlaps most; a segment that spans two turns (each ≥ ``min_split_sec``) is split at the turn boundary, its words
shared out in proportion to time (Whisper gives no word times here, so this is approximate).

``TranscriptLog`` writes ``logs/YYYY-MM-DD/transcripts.jsonl`` (the ``speech`` events) and a human-readable
``transcripts.txt``::

    [12:30:05] [warehouse · mic-mac] Oleh: "Close the door, please."

The name in the text file is resolved when the line is written; consumers of the events must resolve the current
name from ``person_id`` (renames apply retroactively — the dashboard does).
"""
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.events import Event
from core.logger import JsonlLogger
from modules.audio.speaker import Turn


@dataclass
class Segment:
    start: float
    end: float
    text: str
    cluster: Optional[int] = None
    confidence: Optional[float] = None   # ASR confidence 0..1 (F16 trust): exp(avg_logprob) × (1 − no_speech_prob)
    overlap: bool = False                # another speaker talked during this Whisper segment


def whisper_segments(result: Dict[str, Any], duration: float) -> List[Segment]:
    """Segments from a transcriber result; the whole chunk as one segment when the backend gave none."""
    segments = [Segment(max(0.0, float(s["start"])), min(duration, float(s["end"])), str(s["text"]).strip(),
                        confidence=s.get("confidence"))
                for s in result.get("segments") or [] if str(s.get("text", "")).strip()]
    if not segments and (result.get("text") or "").strip():
        segments = [Segment(0.0, duration, result["text"].strip())]
    return segments


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _split_words(text: str, fractions: Sequence[float]) -> List[str]:
    """Shares the words of `text` out in proportion to `fractions` (each part gets ≥ 1 word when possible)."""
    words = text.split()
    if len(fractions) == 1 or len(words) < len(fractions):
        return [text] + [""] * (len(fractions) - 1)
    total, parts, used = sum(fractions), [], 0
    for i, fraction in enumerate(fractions):
        remaining_parts = len(fractions) - i - 1
        if remaining_parts == 0:
            take = len(words) - used
        else:
            take = round(len(words) * fraction / total)
            take = min(max(1, take), len(words) - used - remaining_parts)
        parts.append(" ".join(words[used:used + take]))
        used += take
    return parts


def align(segments: Sequence[Segment], turns: Sequence[Turn], min_split_sec: float = 0.5) -> List[Segment]:
    """
    :param segments: Whisper segments (seconds from the chunk start)
    :param turns: Speaker turns from the diarization (cluster None = uncertain)
    :return: Segments with ``cluster`` set; split where one segment covers two speakers
    """
    out: List[Segment] = []
    for segment in segments:
        pieces = [(turn, _overlap(segment.start, segment.end, turn.start, turn.end)) for turn in turns]
        pieces = [(turn, seconds) for turn, seconds in pieces if seconds > 0]
        if not pieces:
            out.append(Segment(segment.start, segment.end, segment.text, None, segment.confidence))
            continue
        # Neighbouring turns of the same speaker count as one
        merged: List[List[Any]] = []
        for turn, seconds in pieces:
            if merged and merged[-1][0] == turn.cluster:
                merged[-1][2] = min(segment.end, turn.end)
                merged[-1][3] += seconds
            else:
                merged.append([turn.cluster, max(segment.start, turn.start), min(segment.end, turn.end), seconds])
        long = [m for m in merged if m[3] >= min_split_sec]
        if len(long) < 2:
            best = max(merged, key=lambda m: m[3])
            out.append(Segment(segment.start, segment.end, segment.text, best[0], segment.confidence,
                               overlap=len(merged) > 1))
            continue
        texts = _split_words(segment.text, [m[3] for m in long])
        for (cluster, start, end, _), text in zip(long, texts, strict=True):
            if text:
                out.append(Segment(start, end, text, cluster, segment.confidence, overlap=True))
    return out


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(ts * 1000) % 1000:03d}Z"


class TranscriptLog:
    """logs/YYYY-MM-DD/transcripts.jsonl + transcripts.txt (UTC day folders, like every other log)."""

    def __init__(self, base_dir: str = "logs", max_total_mb: float = 2048,
                 resolve_name: Optional[Callable[[Optional[str], Optional[str]], Optional[str]]] = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.base_dir = base_dir
        self.jsonl = JsonlLogger("transcripts", base_dir, max_total_mb, clock=clock)
        self.resolve_name = resolve_name or (lambda person_id, name: name)
        self.clock = clock
        self._day: Optional[str] = None
        self._text = None

    def write(self, event: Event) -> None:
        self.jsonl.write(event)
        day = self.clock().strftime("%Y-%m-%d")
        if day != self._day:
            if self._text is not None:
                self._text.close()
            os.makedirs(os.path.join(self.base_dir, day), exist_ok=True)
            self._text = open(os.path.join(self.base_dir, day, "transcripts.txt"), "a", encoding="utf-8")
            self._day = day
        self._text.write(text_line(event.payload, self.resolve_name) + "\n")
        self._text.flush()

    def close(self) -> None:
        self.jsonl.close()
        if self._text is not None:
            self._text.close()
            self._text = None


def text_line(payload: Dict[str, Any], resolve_name=None) -> str:
    """``[12:30:05] [warehouse · mic-mac] Oleh: "Close the door, please."`` (local time)"""
    started = payload.get("start")
    when = datetime.strptime(started, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).astimezone() \
        if started else datetime.now().astimezone()
    name = payload.get("speaker_name")
    if resolve_name is not None and payload.get("person_id"):
        name = resolve_name(payload["person_id"], name)
    if payload.get("speaker_status") == "uncertain" or not name:
        name = "?"
    where = " · ".join(str(x) for x in (payload.get("location_id"), payload.get("device_id")) if x)
    return f"[{when:%H:%M:%S}] [{where}] {name}: \"{payload.get('text', '')}\""
