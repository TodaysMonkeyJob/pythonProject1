"""
Sends 10 prepared snapshot digests (normal, gas, fall, smoking, unknown person) to GPT and prints the decisions.

    python -m tools.test_llm             # live: needs AZURE_OPENAI_* in .env and llm.gpt_deployment
    python -m tools.test_llm --dry-run   # no network: digests, token counts and the expected cost per request

Each case has the risk levels the prompt rules allow; the summary line counts valid JSON answers and sensible
decisions (acceptance for F6 / TOD-17).
"""
import argparse
import json
import os
import sys
import time
from typing import List, Optional

if __package__ in (None, ""):  # allow `python tools/test_llm.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config  # noqa: E402

CASES = [
    ("normal: office work", {"low"}, "\n".join([
        "10:15:00-10:16:00 UTC+3",
        "people: #3 Oleh sitting 58s; #5 Anna standing 40s",
        "objects: person laptop chair",
        'speech: 10:15:20 uk "Надішли мені звіт до обіду"',
        "gas: clean_air 0ppm stable conf0.97",
        "system: mic active, scene active, 14 events",
    ])),
    ("normal: empty room", {"low"}, "\n".join([
        "03:00:00-03:01:00 UTC+3",
        "gas: clean_air 0ppm stable conf0.98",
        "system: mic standby, scene idle, 2 events",
    ])),
    ("gas: threshold", {"high", "critical"}, "\n".join([
        "14:02:10-14:02:20 UTC+3 CRITICAL alert gas_threshold | incident 20261017-140220-1a2b",
        "people: #8 Oleh standing 12s",
        "objects: person",
        "gas: acetone 62000ppm rising conf0.91 alerts gas_threshold",
        "system: mic active, scene active, 9 events",
    ])),
    ("gas: rising, empty room", {"high", "critical"}, "\n".join([
        "23:40:00-23:40:05 UTC+3 CRITICAL alert gas_rising | incident 20261017-234005-9f00",
        "gas: ammonia 21000ppm rising conf0.84 alerts gas_rising",
        "system: mic standby, scene idle, 4 events",
    ])),
    ("fall: lying 45 s", {"critical"}, "\n".join([
        "16:20:00-16:20:50 UTC+3 CRITICAL lying #4 | incident 20261017-162005-77aa",
        "people: #4 Anna walking>lying 50s",
        "objects: person chair",
        "sounds: 16:20:03 loud -8dB",
        "alerts: lying #4",
        "system: mic active, scene active, 11 events",
    ])),
    ("fall: lying, stranger", {"critical"}, "\n".join([
        "02:10:00-02:10:40 UTC+3 CRITICAL lying #9 | incident 20261018-021005-c3c3",
        "people: #9 unknown_4 standing>lying 40s",
        "alerts: lying #9",
        "system: mic active, scene active, 6 events",
    ])),
    ("smoking: gas confirmed", {"medium", "high", "critical"}, "\n".join([
        "12:30:00-12:30:30 UTC+3 CRITICAL possible_smoking #7 | incident 20261017-123030-5e5e",
        "people: #7 unknown_2 walking>standing 30s",
        "objects: person",
        "gas: toluol 9000ppm rising conf0.62",
        "alerts: possible_smoking #7 gas_confirmed",
        "system: mic active, scene active, 12 events",
    ])),
    ("smoking: no gas", {"low", "medium"}, "\n".join([
        "12:45:00-12:45:30 UTC+3 CRITICAL possible_smoking #3 | incident 20261017-124530-2d2d",
        "people: #3 Oleh sitting 30s",
        "objects: person cup",
        "gas: clean_air 0ppm stable conf0.95",
        "alerts: possible_smoking #3",
        "system: mic active, scene active, 8 events",
    ])),
    ("unknown: daytime visitor", {"low", "medium"}, "\n".join([
        "11:05:00-11:06:00 UTC+3",
        "people: #12 unknown_5 standing>walking 35s; #3 Oleh sitting 60s",
        "objects: person backpack",
        'speech: 11:05:10 uk "Добрий день, я до Олега на зустріч"',
        "system: mic active, scene active, 10 events",
    ])),
    ("unknown: at night, alone", {"medium", "high"}, "\n".join([
        "02:47:00-02:47:30 UTC+3 CRITICAL unknown person unknown_6 at 02:47 | incident 20261018-024730-8b8b",
        "people: #15 unknown_6 walking 28s",
        "objects: person backpack",
        "sounds: 02:47:05 loud -15dB",
        "system: mic active, scene idle, 7 events",
    ])),
]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Send example digests to GPT")
    parser.add_argument("--dry-run", action="store_true", help="no network: tokens and expected cost only")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    llm = config["llm"]

    if args.dry_run:
        from modules.llm.gpt_client import load_prompt

        prompt = load_prompt(llm.get("prompt", "security_analyst"), llm.get("prompts_dir", "prompts"))
        try:
            import tiktoken

            count = len(tiktoken.get_encoding("o200k_base").encode(prompt + "\n"))
            tokens = lambda text: len(tiktoken.get_encoding("o200k_base").encode(text))  # noqa: E731
        except ImportError:
            count, tokens = len(prompt) // 4, lambda text: len(text) // 4  # noqa: E731
        print(f"System prompt: {count} tokens")
        for name, expected, digest in CASES:
            prompt_tokens = count + tokens(digest)
            cost = (prompt_tokens * llm.get("price_per_1k_input", 0) + 150 * llm.get("price_per_1k_output", 0)) / 1000
            print(f"\n=== {name} (expected {'/'.join(sorted(expected))}) — ~{prompt_tokens} in + ~150 out, "
                  f"≈ ${cost:.5f}\n{digest}")
        return

    from modules.llm.azure_client import MissingCredentials
    from modules.llm.gpt_client import GPTClient, InvalidResponse

    try:
        gpt = GPTClient(config)
    except MissingCredentials as e:
        sys.exit(f"{e}. Add them to .env, or run with --dry-run.")
    valid = sensible = 0
    for name, expected, digest in CASES:
        started = time.monotonic()
        try:
            decision, usage = gpt.analyze(digest)
        except InvalidResponse as e:
            print(f"\n=== {name}: INVALID — {e}")
            continue
        valid += 1
        ok = decision.risk_level in expected
        sensible += ok
        print(f"\n=== {name}: {'OK' if ok else 'CHECK'} risk={decision.risk_level} (expected "
              f"{'/'.join(sorted(expected))}) action={decision.recommended_action} — {usage.prompt_tokens}+"
              f"{usage.completion_tokens} tokens, ${usage.cost_usd:.5f}, {time.monotonic() - started:.1f} s")
        print(json.dumps(decision.model_dump(), ensure_ascii=False, indent=1))
    print(f"\n{valid}/{len(CASES)} valid JSON, {sensible}/{len(CASES)} within the expected risk levels; "
          f"total {gpt.total.prompt_tokens}+{gpt.total.completion_tokens} tokens, ${gpt.total.cost_usd:.4f}")


if __name__ == "__main__":
    main()
