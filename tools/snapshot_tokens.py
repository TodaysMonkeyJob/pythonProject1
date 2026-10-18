"""
Token cost of what GPT gets per snapshot (F18 / TOD-29): v1 JSON snapshot, v1 text digest and the v2 view.

    python -m tools.snapshot_tokens                         # exports/snapshots/*.json (read only)
    python -m tools.snapshot_tokens "path/*.json" --encoding o200k_base

Old snapshots without an ``llm_view`` go through the v1 → v2 adapter (``modules.aggregator.llm_view``) with the
location's fuzzy terms. Counts use tiktoken (``o200k_base`` = the gpt-4o family).
"""
import argparse
import glob
import json
import os
import sys
from statistics import mean, median

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.aggregator.digest import snapshot_digest  # noqa: E402
from modules.aggregator.llm_view import llm_view, payload_text  # noqa: E402


def counts(snapshot, encoder, engine_for) -> dict:
    record = {k: v for k, v in snapshot.items() if k not in ("llm_view", "schema_version")}
    view = snapshot.get("llm_view") or llm_view(record, engine_for(record.get("location_id")))
    return {"v1_json": len(encoder.encode(json.dumps(record, ensure_ascii=False, separators=(",", ":")))),
            "v1_digest": len(encoder.encode(snapshot_digest(record, record.get("incident_id")))),
            "v2_view": len(encoder.encode(payload_text(view)))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Average LLM input tokens per snapshot, v1 vs v2")
    parser.add_argument("pattern", nargs="?", default="exports/snapshots/*.json")
    parser.add_argument("--encoding", default="o200k_base")
    parser.add_argument("--fuzzy", default="config/fuzzy.yaml")
    args = parser.parse_args(argv)
    import tiktoken

    from core.fuzzy import FuzzyEngine

    encoder = tiktoken.get_encoding(args.encoding)
    engines = {}

    def engine_for(location_id):
        if location_id not in engines:
            engines[location_id] = FuzzyEngine.from_config(args.fuzzy, location_id) if os.path.isfile(args.fuzzy) \
                else None
        return engines[location_id]

    rows = []
    for path in sorted(glob.glob(args.pattern)):
        with open(path, encoding="utf-8") as f:
            snapshot = json.load(f)
        if snapshot.get("trigger") == "global" or "people" not in snapshot:
            continue
        rows.append(counts(snapshot, encoder, engine_for))
    if not rows:
        print(f"No location snapshots in {args.pattern}", file=sys.stderr)
        return 1
    print(f"{len(rows)} snapshots, tokens per snapshot ({args.encoding}):")
    print(f"{'':12}{'mean':>8}{'median':>8}{'max':>8}")
    for key in ("v1_json", "v1_digest", "v2_view"):
        values = [r[key] for r in rows]
        print(f"{key:12}{mean(values):>8.0f}{median(values):>8.0f}{max(values):>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
