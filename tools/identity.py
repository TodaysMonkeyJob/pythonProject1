"""
The identity registry from the command line (F9 / TOD-20) — same operations as the dashboard People page.

    python -m tools.identity list
    python -m tools.identity rename unknown_3 "Oleh"      # by display name, person_id or an id prefix (≥ 8 chars)
    python -m tools.identity merge unknown_5 Oleh         # unknown_5 was Oleh: faces + voices move to Oleh
    python -m tools.identity delete unknown_7             # privacy / GDPR request: embeddings + face crops removed
    python -m tools.identity history [who]                # audit trail (identity_events)

Running modules pick the change up within identity.sync_sec (the CLI is not on the event bus).
"""
import argparse
import os
import sys
from datetime import datetime
from typing import List, Optional

if __package__ in (None, ""):  # allow `python tools/identity.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config  # noqa: E402
from core.identity import IdentityRegistry, registry_from_config  # noqa: E402


def when(ts: Optional[float]) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "—"


def person(registry: IdentityRegistry, ref: str) -> str:
    pid = registry.find(ref)
    if pid is None:
        raise SystemExit(f"No person {ref!r} (python -m tools.identity list)")
    return pid


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="List / rename / merge / delete people in the identity registry")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--db", help="Overrides identity.registry_path")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    rename = commands.add_parser("rename")
    rename.add_argument("who")
    rename.add_argument("name")
    merge = commands.add_parser("merge", help="merge SOURCE into TARGET (same person seen twice)")
    merge.add_argument("source")
    merge.add_argument("target")
    delete = commands.add_parser("delete")
    delete.add_argument("who")
    delete.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    history = commands.add_parser("history")
    history.add_argument("who", nargs="?")
    history.add_argument("--limit", type=int, default=30)
    args = parser.parse_args(argv)

    cfg = dict(load_config(args.config).get("identity", {}))
    if args.db:
        cfg["registry_path"] = args.db
    registry = registry_from_config(cfg)
    try:
        if args.command == "list":
            people = registry.list_persons()
            for p in people:
                print(f"{p['person_id'][:8]}  {p['display_name']:<20} {'known' if p['is_known'] else 'unknown':<8} "
                      f"faces {p['face_samples']:>2}  voices {p['voice_samples']:>2}  last seen {when(p['last_seen'])} "
                      f"{p['last_location'] or ''}")
            print(f"{len(people)} people in {registry.path}")
        elif args.command == "rename":
            pid = person(registry, args.who)
            event = registry.rename(pid, args.name)
            print(f"{event.payload['previous']} → {event.payload['display_name']} ({pid[:8]})")
        elif args.command == "merge":
            source, target = person(registry, args.source), person(registry, args.target)
            event = registry.merge(source, target)
            print(f"{event.payload['source_name']} merged into {event.payload['display_name']} ({target[:8]}): "
                  f"faces {registry.counts[target]['face']}, voices {registry.counts[target]['voice']}")
        elif args.command == "delete":
            pid = person(registry, args.who)
            name = registry.name(pid)
            if not args.yes and input(f"Delete {name} with all face and voice samples? [y/N] ").lower() != "y":
                raise SystemExit("Cancelled")
            registry.delete(pid)
            print(f"Deleted {name} ({pid[:8]})")
        elif args.command == "history":
            pid = person(registry, args.who) if args.who else None
            for row in registry.history(pid, args.limit):
                details = {k: v for k, v in row.items() if k not in ("ts", "action", "person_id")}
                print(f"{when(row['ts'])}  {row['action']:<10} {(row['person_id'] or '')[:8]}  {details}")
    except ValueError as e:  # NameTaken, invalid name, merge into itself
        raise SystemExit(str(e)) from e
    finally:
        registry.close()


if __name__ == "__main__":
    main()
