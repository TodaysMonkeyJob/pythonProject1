"""
Local detector of de-escalation claims and instructions to the system (F16 / TOD-27), Ukrainian + English.

Threat model: the person being monitored tries to talk the LLM analyst down — "everything is fine, the sensor is
broken", a sign "SYSTEM: status safe", synthetic speech from a speaker. Detection is a pattern list over normalised
text (lower case, unified apostrophes / quotes, punctuation → spaces); it never calls a cloud model. Extensible:
``ClaimDetector(extra={"deescalation": [r"..."]})`` or ``trust.claims`` in ``config/fuzzy.yaml``.

``claim_type`` (priority order): ``instruction_to_system`` > ``deescalation`` > ``escalation`` > ``none``.
Escalation ("help", "fire") is detected too but never used to lower trust: the corridor is asymmetric.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

CLAIM_TYPES = ("none", "deescalation", "instruction_to_system", "escalation")

# Word stems: Ukrainian inflects, so most patterns match a stem followed by any ending (\w*)
PATTERNS: Dict[str, List[str]] = {
    "instruction_to_system": [
        # English
        r"\b(ignore|disregard|forget|skip)\b( \w+){0,3} "
        r"(instructions?|rules?|alerts?|alarms?|warnings?|events?|sensors?)\b",
        r"\b(report|mark|set|classify|label|flag)\b( \w+){0,3} (as )?(safe|ok|okay|normal|low|all clear|no risk)\b",
        r"\b(set|lower|reduce|decrease|drop)\b( \w+){0,2} (risk|alert|alarm|threat|status)( level)?\b",
        r"\bsystem( status)?( is)? (safe|ok|okay|normal|green|secure)\b",
        r"\bstatus (safe|ok|okay|normal|green|all clear)\b",
        r"\b(you are now|act as|new instructions?|override|admin mode|developer mode|as an ai)\b",
        r"\b(assistant|analyst|ai|gpt|llm|model)\b( \w+){0,3} (must|should|will) (say|report|answer|respond)\b",
        # Ukrainian
        r"\b(ігнору\w*|не зважа\w*|забуд\w*|пропуст\w*)( \w+){0,3} "
        r"(інструкці\w*|правил\w*|тривог\w*|сигнал\w*|попереджен\w*|датчик\w*|поді\w*)",
        r"\b(познач\w*|признач\w*|встанов\w*|постав\w*|запиш\w*|класифіку\w*)( \w+){0,3} "
        r"(як )?(безпечн\w*|норм\w*|низьк\w*|ок)\b",
        r"\b(зниз\w*|знижу\w*|зменш\w*|понизь\w*|скину\w*|скасу\w*|відмін\w*|вимкн\w*)( \w+){0,2} "
        r"(ризик\w*|тривог\w*|рів\w* загроз\w*|статус\w*|сигналізаці\w*)",
        r"\bсистем\w*( статус)? (безпечн\w*|норм\w*|ок|в нормі|зелен\w*)",
        r"\bстатус (безпечн\w*|норм\w*|ок|в нормі|зелен\w*)",
        r"\b(повідом\w*|скаж\w*|відпові\w*|напиш\w*)( \w+){0,2} (що )?(все |усе )?"
        r"(добре|гаразд|безпечн\w*|нормально|в порядку|ок)\b",
        r"\b(нові інструкці\w*|ти тепер|режим адміністратор\w*|режим розробник\w*)",
    ],
    "deescalation": [
        # English
        r"\b(everything|all|it|this|that)( is|s)? "
        r"(fine|ok|okay|alright|all right|good|normal|safe|clear|under control)\b",
        r"\ball (good|fine|clear|ok|okay|safe)\b",
        r"\bnothing (is )?(wrong|happening|to worry about|going on|bad|serious)\b",
        r"\bno (smoke|fire|gas|leak|danger|problem|problems|threat|intruder|one here|emergency)\b",
        r"\b(sensor|sensors|detector|detectors|alarm|alarms|camera|cameras|system)( \w+){0,2} (is |are )?"
        r"(broken|faulty|malfunctioning|not working|glitching|glitchy|lying|wrong|bugged|defective|miscalibrated)\b",
        r"\bfalse (alarm|alert|positive)\b",
        r"\b(just|only|merely)( a)? (test|drill|testing|exercise|simulation)\b",
        r"\b(test|maintenance|demo|calibration) mode\b",
        r"\b(don t|do not|no need to) (worry|panic|call|react|respond|be alarmed)\b",
        r"\b(it s|its|this is) (safe|harmless|nothing)\b",
        # Ukrainian
        r"\b(все|усе|всё)( \w+)? "
        r"(добре|гаразд|нормально|норм|в порядку|під контролем|ок|безпечно|спокійно|чисто)\b",
        r"\bнічого (страшн\w*|не стал\w*|не відбува\w*|поган\w*|серйозн\w*|небезпечн\w*)",
        r"\b(немає|нема|відсутн\w*|ніякого|ніякої) "
        r"(диму|газу|пожеж\w*|небезпек\w*|загроз\w*|витоку|проблем\w*|вогню|запаху)",
        r"\b(датчик\w*|сенсор\w*|детектор\w*|камер\w*|сигналізаці\w*|систем\w*)( \w+){0,2} "
        r"(зламан\w*|несправн\w*|не працю\w*|глюч\w*|бреш\w*|помиля\w*|барахл\w*|збо\w*|дефект\w*)",
        r"\b(хибн\w*|помилков\w*|фальшив\w*) (тривог\w*|спрацюван\w*|сигнал\w*)",
        r"\b(це )?(лише|просто|тільки)( \w+)? (тест\w*|перевірк\w*|навчан\w*|тренуван\w*|репетиці\w*)",
        r"\b(тестов\w*|сервісн\w*|демо) режим\w*",
        r"\bне (хвилю\w*|переживай\w*|панікуй\w*|турбуй\w*|реагуй\w*|викликай\w*)",
    ],
    "escalation": [
        r"\b(help|fire|smoke|thief|intruder|robbery|emergency|police|ambulance|911|112|gas leak)\b",
        r"\b(i m|i am|someone is|he is|she is)( \w+)? (hurt|injured|bleeding|unconscious|dying|sick)\b",
        r"\b(допоможіть|рятуйте|пожеж\w*|горить|злодій|злодії|грабіж\w*|поліці\w*|швидк\w* допомог\w*|витік газу)",
        r"\b(мені погано|я поранен\w*|йому погано|їй погано|не дихає|втратив свідомість)",
    ],
}
PRIORITY = ("instruction_to_system", "deescalation", "escalation")


def normalise(text: str) -> str:
    """Lower case, apostrophes / quotes unified, punctuation → spaces, whitespace collapsed (UA ``ʼ'’`` → ``'``)."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[’ʼ`´‘]", "'", text)
    text = re.sub(r"(\w)'(\w)", r"\1 \2", text)        # don't → don t, it's → it s (Ukrainian п'ять → п ять)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Claim:
    claim_type: str = "none"
    spans: List[Tuple[str, str]] = field(default_factory=list)    # (claim type, matched text)

    @property
    def lowers_risk(self) -> bool:
        return self.claim_type in ("deescalation", "instruction_to_system")


class ClaimDetector:
    def __init__(self, extra: Optional[Dict[str, Iterable[str]]] = None):
        """:param extra: More patterns per claim type (regular expressions over the normalised text)"""
        patterns = {kind: list(items) for kind, items in PATTERNS.items()}
        for kind, items in (extra or {}).items():
            if kind not in patterns:
                raise ValueError(f"unknown claim type {kind!r} (use {PRIORITY})")
            patterns[kind] += list(items)
        self.patterns = {kind: [re.compile(p) for p in items] for kind, items in patterns.items()}

    def detect(self, text: Optional[str]) -> Claim:
        """:return: The strongest claim type in the text and every matched span"""
        if not text:
            return Claim()
        norm = normalise(text)
        spans = [(kind, match.group(0)) for kind in PRIORITY for pattern in self.patterns[kind]
                 for match in pattern.finditer(norm)]
        # "no smoke" is a de-escalation, not an escalation: drop escalation hits inside de-escalation spans
        calming = " ".join(span for kind, span in spans if kind != "escalation")
        spans = [(kind, span) for kind, span in spans if kind != "escalation" or span not in calming]
        found = {kind for kind, _ in spans}
        claim_type = next((kind for kind in PRIORITY if kind in found), "none")
        return Claim(claim_type, spans)
