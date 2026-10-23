"""
Human review of GPT rule suggestions (F27 / TOD-38), shared by ``tools/suggestions.py`` and the dashboard page
«Пропозиції правил»: reject, edit (re-validated), send to the TOD-37 approval flow.

Sending never touches ``config/fuzzy.yaml``: it writes a candidate copy with the rule (as text, so the comments of the
file survive an approval) plus ``meta {source: "suggestion <id>", author, rationale, added}`` and runs
``tools.rules.RuleStore.propose`` (checks + regression gate). Only ``python -m tools.rules approve`` deploys it.
"""
import json
import os
import re
import tempfile
from datetime import date
from typing import Any, Dict, List, Optional

from modules.rules.suggester import validate_rule
from modules.rules.suggestions import RuleSuggestionStore, now_iso


class ReviewError(ValueError):
    """A review action that is not allowed in the suggestion's state"""


def _require(store: RuleSuggestionStore, suggestion_id: int) -> Dict[str, Any]:
    item = store.get(int(suggestion_id))
    if item is None:
        raise ReviewError(f"no suggestion {suggestion_id}")
    return item


def _reviewer(by: str) -> str:
    if not str(by or "").strip():
        raise ReviewError("a reviewer name (--by) is required")
    return str(by).strip()


def reject(store: RuleSuggestionStore, suggestion_id: int, by: str, reason: str = "") -> Dict[str, Any]:
    item = _require(store, suggestion_id)
    if item["status"] in ("sent_to_approval", "approved"):
        raise ReviewError(f"suggestion {suggestion_id} is already {item['status']}")
    store.update(item["id"], status="rejected", reviewer=_reviewer(by), reviewed_at=now_iso(),
                 review_reason=str(reason or "")[:500])
    return store.get(item["id"])


def edit(store: RuleSuggestionStore, suggestion_id: int, rule: str, by: str,
         fuzzy_path: str = "config/fuzzy.yaml", location: Optional[str] = None) -> Dict[str, Any]:
    """Replaces the rule text (GPT's original stays in ``original_rule``) and re-runs the validator."""
    item = _require(store, suggestion_id)
    if item["status"] in ("rejected", "sent_to_approval", "approved"):
        raise ReviewError(f"suggestion {suggestion_id} is already {item['status']}")
    location = location or item["location"]
    report = validate_rule(rule, location, fuzzy_path, item.get("new_variable_needed"))
    store.update(item["id"], rule=rule, location=location, validator_report=report, status="edited",
                 reviewer=_reviewer(by), reviewed_at=now_iso())
    return store.get(item["id"])


# --- candidate rule base ------------------------------------------------------------------------------------------

def _quote(text: str) -> str:
    return json.dumps(str(text), ensure_ascii=False)          # a JSON string is a valid YAML double-quoted scalar


def rule_block(rule_id: str, rule: str, meta: Dict[str, str], indent: int) -> List[str]:
    pad = " " * indent
    return [f"{pad}- id: {rule_id}", f"{pad}  rule: {_quote(rule)}",
            f"{pad}  meta: {{source: {_quote(meta['source'])}, author: {_quote(meta['author'])}, "
            f"added: {meta['added']},", f"{pad}         rationale: {_quote(meta['rationale'])}}}"]


def _block_end(lines: List[str], start: int, indent: int) -> int:
    """Index after the last content line of the block opened at ``start`` (lines deeper than ``indent``)"""
    end = start + 1
    for i in range(start + 1, len(lines)):
        text = lines[i]
        if not text.strip() or text.lstrip().startswith("#"):
            continue
        if len(text) - len(text.lstrip()) <= indent:
            break
        end = i + 1
    return end


def _find(lines: List[str], pattern: str, start: int = 0, stop: Optional[int] = None) -> Optional[int]:
    regex = re.compile(pattern)
    for i in range(start, len(lines) if stop is None else stop):
        if regex.match(lines[i]):
            return i
    return None


def insert_rule(text: str, location: str, block_for) -> str:
    """
    :param block_for: indent → the rule's YAML lines
    :return: the fuzzy.yaml text with the rule appended to ``rules:`` or ``locations.<location>.rules``
    """
    lines = text.splitlines()
    if not location or location == "*":
        head = _find(lines, r"^rules:\s*(#.*)?$")
        if head is None:
            raise ReviewError("no top-level rules: list in the rule base")
        end = _block_end(lines, head, 0)
        lines[end:end] = block_for(2)
        return "\n".join(lines) + "\n"
    head = _find(lines, r"^locations:\s*(#.*)?$")
    if head is None:
        lines += ["", "locations:"] + [f"  {location}:", "    rules:"] + block_for(6)
        return "\n".join(lines) + "\n"
    end = _block_end(lines, head, 0)
    own = _find(lines, rf"^  {re.escape(location)}:\s*(#.*)?$", head + 1, end)
    if own is None:
        lines[end:end] = [f"  {location}:", "    rules:"] + block_for(6)
        return "\n".join(lines) + "\n"
    own_end = _block_end(lines, own, 2)
    rules = _find(lines, r"^    rules:\s*(#.*)?$", own + 1, own_end)
    if rules is None:
        lines[own_end:own_end] = ["    rules:"] + block_for(6)
    else:
        rules_end = _block_end(lines, rules, 4)
        lines[rules_end:rules_end] = block_for(6)
    return "\n".join(lines) + "\n"


def candidate_text(item: Dict[str, Any], by: str, fuzzy_path: str) -> str:
    """The current rule base + the suggestion as rule ``GS<id>`` (checked to land where intended; S… = trust.source)"""
    import yaml

    from core.fuzzy.governance import iter_rules

    with open(fuzzy_path, encoding="utf-8") as f:
        text = f.read()
    taken = {str(spec.get("id")) for _, spec in iter_rules(yaml.safe_load(text) or {})}
    rule_id, n = f"GS{item['id']}", 1
    while rule_id in taken:
        n += 1
        rule_id = f"GS{item['id']}_{n}"
    meta = {"source": f"suggestion {item['id']}", "author": by, "added": date.today().isoformat(),
            "rationale": item.get("rationale") or "GPT rule suggestion (no rationale given)"}
    result = insert_rule(text, item["location"], lambda indent: rule_block(rule_id, item["rule"], meta, indent))
    data = yaml.safe_load(result)
    rules = data["rules"] if item["location"] in ("", "*") else data["locations"][item["location"]]["rules"]
    if not any(isinstance(r, dict) and r.get("id") == rule_id and r.get("rule") == item["rule"] for r in rules):
        raise ReviewError("could not place the rule in the rule base")
    return result


def send_to_approval(store: RuleSuggestionStore, suggestion_id: int, by: str, fuzzy_path: str = "config/fuzzy.yaml",
                     rules_dir: str = "data/rules", dataset: str = "data/eval/scripted",
                     tolerances: Optional[Dict[str, float]] = None, llm=None) -> Dict[str, Any]:
    """
    A valid (possibly edited) suggestion → a TOD-37 proposal with the gate report.

    :return: the proposal report of ``RuleStore.propose``; the suggestion gets ``sent_to_approval`` + ``proposal_id``
    """
    from tools.rules import RuleStore

    by = _reviewer(by)
    item = _require(store, suggestion_id)
    if item["status"] in ("rejected", "sent_to_approval", "approved"):
        raise ReviewError(f"suggestion {suggestion_id} is {item['status']}")
    report = validate_rule(item["rule"], item["location"], fuzzy_path, item.get("new_variable_needed"))
    if not report["valid"]:
        store.update(item["id"], validator_report=report)
        raise ReviewError("the rule does not validate — edit it first: " + "; ".join(report["errors"]))
    text = candidate_text(item, by, fuzzy_path)
    handle, path = tempfile.mkstemp(prefix="suggestion_", suffix=".yaml")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            f.write(text)
        proposal = RuleStore(rules_dir, fuzzy_path).propose(path, dataset, tolerances, llm)
    finally:
        os.unlink(path)
    gate = proposal.get("gate") or {}
    summary = {"proposal": proposal["proposal"], "passed": proposal["passed"], "errors": proposal["checks"]["errors"],
               "warnings": proposal["checks"]["warnings"],
               **{k: gate.get(k) for k in ("passed", "reasons", "baseline", "candidate", "failing_scenarios")
                  if k in gate}}
    status = "sent_to_approval" if proposal["passed"] else item["status"]
    store.update(item["id"], gate_report=summary, proposal_id=proposal["proposal"], status=status, reviewer=by,
                 reviewed_at=now_iso())
    return proposal


def mark_approved(store: RuleSuggestionStore, rules_dir: str = "data/rules") -> int:
    """Suggestions whose proposal was approved (``changelog.jsonl``) → ``approved``. :return: how many changed"""
    path = os.path.join(rules_dir, "changelog.jsonl")
    if not os.path.isfile(path):
        return 0
    with open(path, encoding="utf-8") as f:
        approved = {json.loads(line).get("proposal") for line in f if '"approve"' in line}
    changed = 0
    for item in store.list(status="sent_to_approval", limit=1000):
        if item.get("proposal_id") in approved:
            store.update(item["id"], status="approved")
            changed += 1
    return changed
