"""
Review of GPT rule suggestions (F27 / TOD-38). The suggestions live only in ``rules.suggestions_db``
(default ``data/rule_suggestions.db``) — never in the logs or on the event bus.

    python -m tools.suggestions list [--status new] [--location lab] [--confidence high] [--min-support 2]
    python -m tools.suggestions show <id>
    python -m tools.suggestions reject <id> --by "Name" --reason "…"
    python -m tools.suggestions edit <id> --rule "IF … THEN risk IS high" --by "Name"     # re-validated
    python -m tools.suggestions send <id> --by "Name"        # → TOD-37 proposal (checks + gate), then
                                                             #   python -m tools.rules approve <proposal> …
    python -m tools.suggestions export [--status …] > suggestions.json
"""
import argparse
import json
import os
import sys
from typing import List, Optional

if __package__ in (None, ""):  # allow `python tools/suggestions.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv: Optional[List[str]] = None) -> int:
    from core.config import load_config
    from modules.rules import review
    from modules.rules.suggester import store_from_config
    from modules.rules.suggestions import STATUSES

    parser = argparse.ArgumentParser(description="Review GPT rule suggestions")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--db", help="suggestion store (default rules.suggestions_db)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("list", "export"):
        p = sub.add_parser(name)
        p.add_argument("--status", choices=STATUSES)
        p.add_argument("--location")
        p.add_argument("--confidence", choices=("low", "medium", "high"))
        p.add_argument("--min-support", type=int, default=1)
    sub.add_parser("show").add_argument("id", type=int)
    p = sub.add_parser("reject")
    p.add_argument("id", type=int)
    p.add_argument("--by", required=True)
    p.add_argument("--reason", default="")
    p = sub.add_parser("edit")
    p.add_argument("id", type=int)
    p.add_argument("--rule", required=True)
    p.add_argument("--location")
    p.add_argument("--by", required=True)
    p = sub.add_parser("send")
    p.add_argument("id", type=int)
    p.add_argument("--by", required=True)
    p.add_argument("--dataset", default="data/eval/scripted")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    rules_cfg = config.get("rules") or {}
    store = review.RuleSuggestionStore(args.db) if args.db else store_from_config(config)
    fuzzy = (config.get("trust") or {}).get("fuzzy_config", "config/fuzzy.yaml")
    rules_dir = rules_cfg.get("dir", "data/rules")
    review.mark_approved(store, rules_dir)
    try:
        if args.cmd in ("list", "export"):
            items = store.list(args.status, args.location, args.confidence, args.min_support)
            if args.cmd == "export":
                print(json.dumps(items, ensure_ascii=False, indent=1, default=str))
                return 0
            for s in items:
                valid = "valid" if s["validator_report"].get("valid") else "INVALID"
                print(f"#{s['id']:<4} {s['status']:<16} ×{s['support_count']:<3} {s['confidence'] or '-':<6} "
                      f"{s['location']:<12} {valid:<7} {s['rule']}")
            print(f"{len(items)} suggestion(s)")
            return 0
        if args.cmd == "show":
            item = store.get(args.id)
            if item is None:
                raise review.ReviewError(f"no suggestion {args.id}")
            item["cases"] = store.outcomes(ids=item["supporting_cases"] or [-1], limit=50)
            print(json.dumps(item, ensure_ascii=False, indent=1, default=str))
            return 0
        if args.cmd == "reject":
            item = review.reject(store, args.id, args.by, args.reason)
        elif args.cmd == "edit":
            item = review.edit(store, args.id, args.rule, args.by, fuzzy, args.location)
            report = item["validator_report"]
            for line in report["errors"] + report["warnings"]:
                print(line)
        else:
            proposal = review.send_to_approval(store, args.id, args.by, fuzzy, rules_dir, args.dataset,
                                               rules_cfg.get("gate"))
            gate = proposal.get("gate") or {}
            for line in proposal["checks"]["errors"] + list(gate.get("reasons") or []):
                print(line)
            print(f"proposal {proposal['proposal']}: "
                  + ("ready — python -m tools.rules approve " + proposal["proposal"]
                     + " --by … --role … --comment …" if proposal["passed"] else "BLOCKED by the checks / gate"))
            return 0 if proposal["passed"] else 2
        print(f"#{item['id']}: {item['status']}"
              + ("" if item["validator_report"].get("valid") else " (invalid — edit again before sending)"))
        return 0
    except review.ReviewError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
