"""
On-demand GPT rule suggestions (F27 / TOD-38): one suggester job over the stored cases of a period, then every
pending job (one GPT call each). Results go only to the suggestion store — review them with ``tools.suggestions``.

    python -m tools.suggest_rules --since 7d [--location lab]    # needs the Azure keys of .env
    python -m tools.suggest_rules --pending-only                 # jobs queued by the live triggers
"""
import argparse
import os
import sys
from typing import List, Optional

if __package__ in (None, ""):  # allow `python tools/suggest_rules.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def duration(text: str) -> float:
    """``90``, ``30m``, ``12h``, ``7d`` → seconds"""
    text = text.strip().lower()
    if text and text[-1] in UNITS:
        return float(text[:-1]) * UNITS[text[-1]]
    return float(text)


def main(argv: Optional[List[str]] = None) -> int:
    from core.config import load_config
    from modules.rules.suggester import RuleSuggester, manual_job, store_from_config

    parser = argparse.ArgumentParser(description="Ask GPT for rule suggestions from stored cases")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--since", default="7d", type=duration)
    parser.add_argument("--location")
    parser.add_argument("--pending-only", action="store_true")
    parser.add_argument("--limit", type=int, default=10, help="max GPT calls")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    store = store_from_config(config)
    try:
        if not args.pending_only:
            job = manual_job(store, args.since, args.location)
            print("no cases in the period" if job is None else f"job {job} queued")
        if not store.pending_jobs(1):
            return 0
        from modules.llm.gpt_client import GPTClient

        try:
            gpt = GPTClient(config)
        except Exception as e:  # missing keys / endpoint
            print(f"GPT unavailable ({type(e).__name__}): jobs stay pending for the LLM worker", file=sys.stderr)
            return 1
        suggester = RuleSuggester(store, gpt, (config.get("trust") or {}).get("fuzzy_config", "config/fuzzy.yaml"))
        print(f"{suggester.process_pending(args.limit)} job(s) handled → python -m tools.suggestions list")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
