"""
Governed rule changes (F26 / TOD-37): propose → automatic checks + regression gate → approval → versioned deploy,
rollback. Nothing changes ``config/fuzzy.yaml`` without an approval record.

    python -m tools.rules status
    python -m tools.rules propose candidate.yaml               # validation + gate report, nothing applied
    python -m tools.rules approve <proposal_id> --by "Name" --role security_owner --comment "…"
    python -m tools.rules rollback <version> --by "Name" --role supervisor --comment "…"

Storage (``rules.dir``, default ``data/rules``): ``proposals/<id>/`` (candidate.yaml, gate.json, gate.md),
``versions/<sha256>.yaml`` (every deployed rule base, for rollback), ``changelog.jsonl`` (append-only: baseline /
approve / rollback with approver, role, comment, the from / to version hashes and the gate result).

Checks of a proposal: the engine's validation of every rule base (syntax, unknown terms, coverage, duplicates /
conflicts in any clause order), complete metadata of every rule (source, author, rationale, added), subsumed rules
(warning), then ``tools.rules_gate`` (ASR / macro-F1 / ECE vs the current base on the evaluation scenarios, rules that
never fire). An approval needs a passed gate.
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):  # allow `python tools/rules.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROLES = ("security_owner", "supervisor", "author")
RULES_DIR = "data/rules"
CONFIG = "config/fuzzy.yaml"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RuleStore:
    def __init__(self, rules_dir: str = RULES_DIR, config: str = CONFIG):
        self.dir, self.config = rules_dir, config
        self.versions = os.path.join(rules_dir, "versions")
        self.proposals = os.path.join(rules_dir, "proposals")
        self.changelog = os.path.join(rules_dir, "changelog.jsonl")

    # --- versions / changelog ------------------------------------------------------------------------------------

    def version_of(self, path: str) -> str:
        """The rule-set hash of a file (rules never contain @refs: the file is read as is, wherever it lies)"""
        import yaml

        from core.fuzzy.governance import version_hash

        with open(path, encoding="utf-8") as f:
            return version_hash(yaml.safe_load(f) or {})

    def current(self) -> str:
        return self.version_of(self.config)

    def keep(self, path: str) -> str:
        """Stores a rule base file under versions/<hash>.yaml (idempotent). :return: its hash"""
        version = self.version_of(path)
        os.makedirs(self.versions, exist_ok=True)
        target = os.path.join(self.versions, f"{version}.yaml")
        if not os.path.isfile(target):
            shutil.copy(path, target)
        return version

    def log(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        os.makedirs(self.dir, exist_ok=True)
        entry = {"at": now_iso(), **entry}
        with open(self.changelog, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def history(self) -> List[Dict[str, Any]]:
        if not os.path.isfile(self.changelog):
            return []
        with open(self.changelog, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def resolve(self, prefix: str) -> str:
        matches = [n[:-5] for n in os.listdir(self.versions)] if os.path.isdir(self.versions) else []
        matches = [m for m in matches if m.startswith(prefix)]
        if len(matches) != 1:
            raise SystemExit(f"version {prefix!r}: {'not found' if not matches else 'ambiguous'} in {self.versions}")
        return matches[0]

    def register_baseline(self, by: str, role: str, comment: str) -> Optional[Dict[str, Any]]:
        """Records the current rule base as the first version (no change applied). :return: the entry or None"""
        version = self.keep(self.config)
        if any(e.get("to") == version for e in self.history()):
            return None
        return self.log({"action": "baseline", "to": version, "by": by, "role": role, "comment": comment})

    # --- workflow ------------------------------------------------------------------------------------------------

    def check(self, candidate: str) -> Dict[str, Any]:
        """Engine validation of every rule base + metadata + subsumption. :return: {errors, warnings}"""
        from core.fuzzy import FuzzyConfigError, FuzzyEngine, load_yaml
        from core.fuzzy.governance import metadata_errors
        from core.fuzzy.validate import subsumed
        from tools.rules_gate import staged

        errors, warnings = [], []
        with staged(candidate, os.path.dirname(self.config) or ".") as path:
            data = load_yaml(path)
            try:
                engines = [FuzzyEngine.from_dict(data)]
                for location in data.get("locations") or {}:
                    engines.append(FuzzyEngine.from_config(path, location, None))
                for block in ((data.get("trust") or {}).get(k) for k in ("source", "asr", "identity")):
                    if block:
                        engines.append(FuzzyEngine.from_dict(block))
                if data.get("gating"):
                    engines.append(FuzzyEngine.from_dict(data["gating"]))
                from modules.envelope.confirmation import Confirmation

                engines.append(Confirmation(data, data["envelope"]["confirmation"]).engine)
                for engine in engines:
                    warnings += subsumed(engine.rules)
            except (FuzzyConfigError, KeyError, ValueError) as e:
                errors.append(str(e))
            errors += metadata_errors(data)
        return {"errors": errors, "warnings": sorted(set(warnings))}

    def propose(self, candidate: str, dataset: str = "data/eval/scripted",
                tolerances: Optional[Dict[str, float]] = None, llm=None) -> Dict[str, Any]:
        from core.fuzzy.governance import short
        from tools.rules_gate import gate, markdown

        checks = self.check(candidate)
        version = None if checks["errors"] else self.version_of(candidate)
        proposal_id = short(version) if version else "invalid"
        folder = os.path.join(self.proposals, proposal_id)
        os.makedirs(folder, exist_ok=True)
        shutil.copy(candidate, os.path.join(folder, "candidate.yaml"))
        report: Dict[str, Any] = {"proposal": proposal_id, "created": now_iso(), "checks": checks,
                                  "base": self.current()}
        if not checks["errors"]:
            report["gate"] = gate(candidate, self.config, dataset, tolerances, llm,
                                  os.path.dirname(self.config) or ".")
            with open(os.path.join(folder, "gate.md"), "w", encoding="utf-8") as f:
                f.write(markdown(report["gate"]))
        report["passed"] = not checks["errors"] and report.get("gate", {}).get("passed", False)
        with open(os.path.join(folder, "gate.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in report.items()}, f, indent=1, ensure_ascii=False, default=str)
        return report

    def approve(self, proposal_id: str, by: str, role: str, comment: str) -> Dict[str, Any]:
        folder = os.path.join(self.proposals, proposal_id)
        path = os.path.join(folder, "gate.json")
        if not os.path.isfile(path):
            raise SystemExit(f"no proposal {proposal_id!r} (run: python -m tools.rules propose <file>)")
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        if not report.get("passed"):
            raise SystemExit(f"proposal {proposal_id} did not pass the checks / gate — it cannot be approved")
        if report.get("base") != self.current():
            raise SystemExit("the rule base changed since the proposal was gated — propose it again")
        if role not in ROLES or not by.strip() or not comment.strip():
            raise SystemExit(f"an approval needs --by, --role ({' / '.join(ROLES)}) and --comment")
        previous = self.keep(self.config)
        candidate = os.path.join(folder, "candidate.yaml")
        version = self.keep(candidate)
        shutil.copy(candidate, self.config)
        gate = report.get("gate") or {}
        return self.log({"action": "approve", "proposal": proposal_id, "from": previous, "to": version, "by": by,
                         "role": role, "comment": comment,
                         "gate": {k: gate.get(k) for k in ("passed", "baseline", "candidate", "never_fired")}})

    def rollback(self, prefix: str, by: str, role: str, comment: str) -> Dict[str, Any]:
        if role not in ROLES or not by.strip() or not comment.strip():
            raise SystemExit(f"a rollback needs --by, --role ({' / '.join(ROLES)}) and --comment")
        version = self.resolve(prefix)
        previous = self.keep(self.config)
        shutil.copy(os.path.join(self.versions, f"{version}.yaml"), self.config)
        return self.log({"action": "rollback", "from": previous, "to": version, "by": by, "role": role,
                         "comment": comment})


def main(argv: Optional[List[str]] = None) -> int:
    from core.config import load_config
    from core.fuzzy.governance import short

    parser = argparse.ArgumentParser(description="Governed rule changes (propose / approve / rollback)")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--rules-dir")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("propose")
    p.add_argument("file")
    p.add_argument("--dataset", default="data/eval/scripted")
    for name in ("approve", "rollback"):
        p = sub.add_parser(name)
        p.add_argument("proposal" if name == "approve" else "version")
        p.add_argument("--by", required=True)
        p.add_argument("--role", required=True, choices=ROLES)
        p.add_argument("--comment", required=True)
    args = parser.parse_args(argv)
    settings = load_config("config.yaml").get("rules") or {}
    store = RuleStore(args.rules_dir or settings.get("dir", RULES_DIR), args.config)
    if args.cmd == "status":
        current = store.current()
        known = os.path.isfile(os.path.join(store.versions, f"{current}.yaml"))
        print(f"rule base {short(current)} ({'registered' if known else 'NOT registered — unapproved change?'})")
        for entry in store.history()[-5:]:
            print(f"  {entry['at']} {entry['action']:<8} {short(entry.get('from', '') or '-'):<12} → "
                  f"{short(entry['to'])}  {entry['by']} ({entry['role']}): {entry.get('comment', '')}")
        return 0
    if args.cmd == "propose":
        report = store.propose(args.file, args.dataset, settings.get("gate"))
        for error in report["checks"]["errors"]:
            print(f"error: {error}")
        for warning in report["checks"]["warnings"]:
            print(f"warning: {warning}")
        gate = report.get("gate")
        if gate:
            verdict = "PASSED" if gate["passed"] else "BLOCKED"
            print(f"gate {verdict}: " + ("; ".join(gate["reasons"]) or "no regression")
                  + f"; never fired: {len(gate['never_fired'])}")
        print(f"proposal {report['proposal']}: {'ready for approval' if report['passed'] else 'cannot be approved'}"
              f" → {os.path.join(store.proposals, report['proposal'])}/")
        return 0 if report["passed"] else 2
    if args.cmd == "approve":
        entry = store.approve(args.proposal, args.by, args.role, args.comment)
    else:
        entry = store.rollback(args.version, args.by, args.role, args.comment)
    print(f"{entry['action']}: {short(entry['from'])} → {short(entry['to'])} ({entry['by']}, {entry['role']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
