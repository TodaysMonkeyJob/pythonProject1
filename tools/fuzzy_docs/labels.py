"""Ukrainian names of the fuzzy variables / terms and a plain-language rendering of rules."""
import re
from typing import Dict

VAR_UA: Dict[str, str] = {
    "gas_concentration": "концентрація газу", "gas_trend": "динаміка газу", "motion_level": "рівень руху",
    "people_count": "кількість людей", "unknown_share": "частка незнайомців", "action_risk": "ризик дії",
    "time_of_day_risk": "час доби", "risk": "ризик", "known_present": "присутність знайомої людини",
    "confirmation": "підтвердження безпеки", "speaker": "впізнаваність голосу", "face_voice": "голос ↔ обличчя",
    "playback": "підозра на відтворення запису", "overlap": "накладання мовців", "source": "довіра до джерела мови",
    "asr_confidence": "впевненість розпізнавання мови", "asr": "довіра до транскрипту",
    "face_match": "схожість обличчя", "identity": "довіра до впізнання", "novelty": "новизна сцени",
    "width": "ширина коридору", "agreement": "згода GPT з коридором", "untrusted_claims": "недовірені заяви",
    "time_since_last_llm": "час від останнього виклику GPT", "severity": "серйозність (нижня межа)",
    "call_llm": "потреба викликати GPT", "trust": "довіра",
}

TERM_UA: Dict[str, Dict[str, str]] = {
    "gas_concentration": {"clean": "чисте повітря", "slight": "незначна", "elevated": "підвищена",
                          "dangerous": "небезпечна"},
    "gas_trend": {"falling": "спадає", "stable": "стабільна", "rising": "зростає", "sharply_rising": "різко зростає"},
    "motion_level": {"none": "немає", "low": "слабкий", "high": "сильний"},
    "people_count": {"none": "нікого", "one": "одна людина", "several": "кілька", "crowd": "натовп"},
    "unknown_share": {"none": "немає", "some": "частина", "all": "усі"},
    "action_risk": {"low": "низький", "medium": "середній", "high": "високий"},
    "time_of_day_risk": {"normal": "робочий час", "off_hours": "неробочий час", "night": "ніч"},
    "risk": {"low": "низький", "medium": "середній", "high": "високий", "critical": "критичний"},
    "known_present": {"absent": "немає", "present": "є"},
    "confirmation": {"none": "немає", "partial": "часткове", "full": "повне"},
    "speaker": {"unknown": "невідомий", "uncertain": "непевний", "known": "відомий"},
    "face_voice": {"mismatch": "не збігаються", "off_camera": "поза камерою", "possible": "можливо збігаються",
                   "match": "збігаються"},
    "playback": {"not_suspected": "немає підозри", "suspected": "є підозра"},
    "overlap": {"absent": "немає", "present": "є"},
    "source": {"very_low": "дуже низька", "low": "низька", "medium": "середня", "high": "висока"},
    "asr_confidence": {"poor": "погана", "fair": "середня", "good": "добра"},
    "asr": {"low": "низька", "medium": "середня", "high": "висока"},
    "face_match": {"weak": "слабка", "fair": "середня", "strong": "сильна"},
    "identity": {"low": "низька", "medium": "середня", "high": "висока"},
    "novelty": {"low": "низька", "medium": "середня", "high": "висока"},
    "width": {"low": "вузький", "medium": "середній", "high": "широкий"},
    "agreement": {"low": "низька", "medium": "середня", "high": "висока"},
    "untrusted_claims": {"none": "немає", "some": "є інші недовірені дані", "present": "є заява на зниження"},
    "time_since_last_llm": {"recent": "щойно", "while": "кілька хвилин", "long": "давно"},
    "severity": {"low": "низька", "medium": "середня", "high": "висока", "critical": "критична"},
    "call_llm": {"skip": "не викликати", "maybe": "можливо", "call": "викликати"},
    "trust": {"low": "низька", "medium": "середня", "high": "висока"},
}

ACTION_UA = {"lying": "лежить", "falling": "падає", "fighting": "бійка", "distress": "лихо / паніка",
             "possible_smoking": "можливе куріння", "staggering": "хитається", "climbing": "лізе", "throwing": "кидає",
             "running": "біжить", "walking": "іде", "hugging": "обіймається", "sitting": "сидить", "standing": "стоїть",
             "picking_up": "піднімає", "sitting_down": "сідає", "standing_up": "встає", "drinking": "п'є",
             "eating": "їсть", "phone_call": "говорить телефоном", "waving": "махає"}

SHAPE_UA = {"trap": "трапеція", "tri": "трикутник", "gauss": "гаусіана"}


def var_ua(name: str) -> str:
    return VAR_UA.get(name, name)


def term_ua(var: str, term: str) -> str:
    return TERM_UA.get(var, {}).get(term, term)


def rule_ua(text: str) -> str:
    """'IF gas_concentration IS dangerous AND … THEN risk IS critical' → 'ЯКЩО концентрація газу = небезпечна і …'"""
    def clause(m: re.Match) -> str:
        var, neg, term = m.group(1), m.group(2), m.group(3)
        return f"{var_ua(var)} {'≠' if neg else '='} {term_ua(var, term)}"

    out = re.sub(r"(\w+) IS (NOT )?(\w+)", clause, text)
    out = re.sub(r"\bIF\b", "ЯКЩО", out)
    out = re.sub(r"\bTHEN\b", "→", out)
    out = re.sub(r"\bAND\b", "і", out)
    out = re.sub(r"\bOR\b", "або", out)
    out = re.sub(r"\bNOT\b", "НЕ", out)
    return out


def params_text(parts) -> str:
    """Parsed term parts → 'трапеція (0; 0; 0,2; 0,5)' (several parts joined by ' ∪ ')"""
    def num(v: float) -> str:
        return f"{v:g}".replace(".", ",")
    return " ∪ ".join(f"{SHAPE_UA.get(shape, shape)} ({'; '.join(num(p) for p in params)})" for shape, params in parts)


def core_support(parts):
    """:return: (core [from, to] where μ = 1, support [from, to] where μ > 0)"""
    cores, supports = [], []
    for shape, p in parts:
        if shape == "trap":
            cores.append((p[1], p[2]))
            supports.append((p[0], p[3]))
        elif shape == "tri":
            cores.append((p[1], p[1]))
            supports.append((p[0], p[2]))
        else:
            cores.append((p[0], p[0]))
            supports.append((p[0] - 3 * p[1], p[0] + 3 * p[1]))
    return cores, supports
