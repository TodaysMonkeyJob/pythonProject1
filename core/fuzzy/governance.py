"""
Rule governance (F26 / TOD-37): every rule base of ``config/fuzzy.yaml`` in one place — metadata checks, a normalised
rule set and its version hash.

Rule bases: the risk rules (``rules``), per-location overrides (``locations.<id>.rules``), the trust bases
(``trust.source`` / ``asr`` / ``identity``), the safety confirmation (``envelope.confirmation``) and LLM gating
(``gating``). Every rule entry carries ``meta: {source, author, rationale, added}`` — where it comes from (norm,
location requirement, incident, experiment, suggestion id), who wrote it, why, when.

The version = sha256 of the normalised rule set (section, id, canonical rule text, weight — metadata excluded: it does
not change any decision), written into experiment manifests, ``llm_decision`` events and the changelog of
``tools/rules.py``.
"""
import hashlib
import json
from typing import Any, Dict, Iterator, List, Tuple

from core.fuzzy.rules import parse_rule, render

META_KEYS = ("source", "author", "rationale", "added")


def rule_sections(data: Dict[str, Any]) -> Iterator[Tuple[str, List[Any]]]:
    """:return: (section name, its list of rule entries) for every rule base in the parsed fuzzy.yaml"""
    yield "risk", data.get("rules") or []
    for location, block in sorted((data.get("locations") or {}).items()):
        yield f"locations.{location}", (block or {}).get("rules") or []
    for name in ("source", "asr", "identity"):
        yield f"trust.{name}", ((data.get("trust") or {}).get(name) or {}).get("rules") or []
    yield "envelope.confirmation", ((data.get("envelope") or {}).get("confirmation") or {}).get("rules") or []
    yield "gating", (data.get("gating") or {}).get("rules") or []


def iter_rules(data: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    for section, rules in rule_sections(data):
        for spec in rules:
            yield section, spec if isinstance(spec, dict) else {"rule": spec}


def metadata_errors(data: Dict[str, Any]) -> List[str]:
    """:return: one message per rule without complete metadata (source, author, rationale, added)"""
    errors = []
    for section, spec in iter_rules(data):
        meta = spec.get("meta") or {}
        missing = [k for k in META_KEYS if not str(meta.get(k) or "").strip()]
        if missing:
            errors.append(f"{section}/{spec.get('id', '?')}: metadata missing {', '.join(missing)}")
    return errors


def normalized(data: Dict[str, Any]) -> List[List[Any]]:
    """:return: [[section, id, canonical rule, weight]] sorted — the content the version hash covers"""
    out = []
    for section, spec in iter_rules(data):
        rule = parse_rule(spec, 0)
        out.append([section, rule.id, f"IF {render(rule.antecedent)} THEN {rule.output} IS {rule.term}",
                    round(float(rule.weight), 6)])
    return sorted(out)


def version_hash(data: Dict[str, Any]) -> str:
    payload = json.dumps(normalized(data), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def short(version: str) -> str:
    return version[:12]


def rules_table(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """:return: every rule with its section and metadata (dashboard / reports)"""
    rows = []
    for section, spec in iter_rules(data):
        meta = spec.get("meta") or {}
        rows.append({"section": section, "id": spec.get("id"), "rule": spec.get("rule"),
                     "weight": spec.get("weight", 1.0), **{k: str(meta.get(k, "")) for k in META_KEYS}})
    return rows
