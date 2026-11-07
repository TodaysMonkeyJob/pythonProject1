"""
Danger words in transcripts (A4 / TOD-44): ``keyword_danger`` = the highest grade of ``config/danger_words.yaml``
found in a transcript, with the matched words (the explanation).

Matching on the normalised transcript (lower case, Ukrainian apostrophes unified, punctuation removed), per token or
consecutive tokens for phrases, in this order:

1. **exact** — the entry word or one of its listed forms;
2. **stem** — a Ukrainian ending replaced: the entry's stem (word without its final vowel / soft sign) plus one of
   the common endings (пожеж·а → пожеж·і, пожеж·ею …). Not a real lemmatiser — Ukrainian morphology needs a
   dictionary (pymorphy) we do not ship; the listed forms cover irregular cases;
3. **fuzzy** — Levenshtein distance to a form ≤ 1 (5–7 letters) or ≤ 2 (8+ letters) for Whisper's near misses
   («пожежя», «допоможить»); words of up to 4 letters never match fuzzily («газ» vs «раз»), and a fuzzy hit that is
   itself a common word (``common_words``) is rejected — the recognised word must belong to the dictionary.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

APOSTROPHES = "ʼ’`´‘"
ENDINGS = ("", "а", "я", "о", "е", "и", "і", "у", "ю", "ь", "й", "ам", "ям", "ах", "ях", "ою", "ею", "ом", "ем",
           "ів", "ей", "ами", "ями", "ові", "еві", "ого", "ому", "их", "ими", "іть", "ить", "те", "ть")
STEM_DROP = "аяоеиіуюьй"
TOKEN = re.compile(r"[\w'\-]+", re.UNICODE)


def normalise(text: str) -> List[str]:
    """:return: lower-case tokens, apostrophes unified, punctuation dropped"""
    text = unicodedata.normalize("NFC", text or "").lower()
    for mark in APOSTROPHES:
        text = text.replace(mark, "'")
    return [t.strip("-'") for t in TOKEN.findall(text) if t.strip("-'")]


def levenshtein(a: str, b: str, limit: int = 3) -> int:
    """:return: edit distance (insert / delete / substitute), early exit above ``limit``"""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def tolerance(word: str) -> int:
    return 0 if len(word) <= 4 else 1 if len(word) <= 7 else 2


def stem(word: str) -> str:
    return word[:-1] if len(word) >= 5 and word[-1] in STEM_DROP else word


@dataclass
class Entry:
    word: str
    grade: float
    level: str
    forms: List[str] = field(default_factory=list)       # normalised, the word first

    @property
    def tokens(self) -> int:
        return len(self.forms[0].split())


@dataclass
class Match:
    word: str            # as heard
    entry: str
    grade: float
    level: str
    kind: str            # exact | stem | fuzzy
    distance: int = 0

    def payload(self) -> Dict[str, Any]:
        return {"word": self.word, "entry": self.entry, "grade": self.grade, "level": self.level, "kind": self.kind,
                **({"distance": self.distance} if self.kind == "fuzzy" else {})}


class DangerWords:
    def __init__(self, entries: Sequence[Entry], common_words: Sequence[str] = ()):
        self.entries = list(entries)
        self.common = {" ".join(normalise(w)) for w in common_words}

    @classmethod
    def load(cls, path: str = "config/danger_words.yaml") -> "DangerWords":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        grades = {str(k): float(v) for k, v in (data.get("grades") or {}).items()}
        entries = []
        for level, items in (data.get("words") or {}).items():
            if level not in grades:
                raise ValueError(f"danger_words.yaml: level {level!r} has no grade")
            for item in items or []:
                word, forms = (item, []) if isinstance(item, str) else (item["word"], item.get("forms") or [])
                normal = [" ".join(normalise(w)) for w in [word, *forms]]
                entries.append(Entry(normal[0], grades[level], level, list(dict.fromkeys(normal))))
        return cls(entries, data.get("common_words") or [])

    def _match_token(self, heard: str, entry: Entry) -> Optional[Match]:
        for form in entry.forms:
            if heard == form:
                return Match(heard, entry.word, entry.grade, entry.level, "exact")
        if " " in heard:
            return None
        for form in entry.forms:
            if " " in form or len(form) < 5:
                continue
            base = stem(form)
            if heard.startswith(base) and heard[len(base):] in ENDINGS:
                return Match(heard, entry.word, entry.grade, entry.level, "stem")
        if heard in self.common or len(heard) <= 4:
            return None
        best: Optional[Tuple[int, str]] = None
        for form in entry.forms:
            limit = tolerance(form)
            if limit == 0 or " " in form:
                continue
            distance = min(levenshtein(heard, form, limit), levenshtein(heard, stem(form), limit)
                           if stem(form) != form and len(heard) <= len(form) else limit + 1)
            if distance <= limit and (best is None or distance < best[0]):
                best = (distance, form)
        if best is not None:
            return Match(heard, entry.word, entry.grade, entry.level, "fuzzy", best[0])
        return None

    def find(self, text: str) -> List[Match]:
        """:return: every danger word / phrase in the text, strongest first"""
        tokens = normalise(text)
        matches: List[Match] = []
        for entry in self.entries:
            for size in sorted({len(f.split()) for f in entry.forms}, reverse=True):
                for i in range(len(tokens) - size + 1):
                    heard = " ".join(tokens[i:i + size])
                    found = self._match_token(heard, entry)
                    if found is not None:
                        matches.append(found)
        matches.sort(key=lambda m: (-m.grade, m.kind != "exact", m.distance))
        unique, seen = [], set()
        for match in matches:
            if (match.word, match.entry) not in seen:
                seen.add((match.word, match.entry))
                unique.append(match)
        return unique

    def grade(self, text: str) -> Tuple[float, List[Match]]:
        """:return: (keyword_danger 0..1, matches)"""
        matches = self.find(text)
        return (matches[0].grade if matches else 0.0), matches
