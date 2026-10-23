"""
Explain one fuzzy inference from the command line (F15 / TOD-26)::

    python -m core.fuzzy gas_concentration=4.2 gas=acetone time_of_day_risk=23.5 [--location warehouse]
    python -m core.fuzzy --check            # validate config/fuzzy.yaml + every location override
"""
import argparse
import os
import sys
import time

from core.fuzzy.engine import FuzzyEngine, load_yaml
from core.fuzzy.governance import metadata_errors, normalized, short, version_hash
from core.fuzzy.validate import FuzzyConfigError, subsumed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fuzzy risk inference with explanations")
    parser.add_argument("inputs", nargs="*", help="variable=value (selectors like gas=acetone too)")
    parser.add_argument("--config", default="config/fuzzy.yaml")
    parser.add_argument("--locations", default="config/locations.yaml")
    parser.add_argument("--location")
    parser.add_argument("--check", action="store_true", help="validate the base rule base and every location")
    args = parser.parse_args(argv)
    try:
        if args.check:
            data = load_yaml(args.config)
            ids = list(data.get("locations") or {})
            if args.locations and os.path.isfile(args.locations):
                ids += [str(e["id"]) for e in load_yaml(args.locations).get("locations") or []
                        if e.get("fuzzy") and str(e.get("id")) not in ids]
            for location in [None, *ids]:
                engine = FuzzyEngine.from_config(args.config, location, args.locations)
                print(f"ok  {location or 'base':<12} {len(engine.rules)} rules, {len(engine.variables)} variables")
                for warning in subsumed(engine.rules):
                    print(f"    warning: {warning}")
            for section in (("trust", "source"), ("trust", "asr"), ("trust", "identity"), ("gating",)):
                block = data
                for key in section:
                    block = (block or {}).get(key) or {}
                for warning in subsumed(FuzzyEngine.from_dict(block).rules) if block.get("rules") else []:
                    print(f"    warning ({'.'.join(section)}): {warning}")
            missing = metadata_errors(data)                     # F26: every rule needs source / author / rationale
            if missing:
                raise FuzzyConfigError(missing, "rule metadata")
            print(f"ok  metadata     {len(normalized(data))} rules; rule base version {short(version_hash(data))}")
            return 0
        engine = FuzzyEngine.from_config(args.config, args.location, args.locations)
    except FuzzyConfigError as e:
        print(e, file=sys.stderr)
        return 1
    inputs = {}
    for item in args.inputs:
        key, _, value = item.partition("=")
        try:
            inputs[key] = float(value)
        except ValueError:
            inputs[key] = value
    start = time.perf_counter()
    result = engine.infer(inputs)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"risk = {result['crisp']:.3f}   ({elapsed:.2f} ms, location {args.location or 'base'})")
    print("output terms: " + ", ".join(f"{t} {v:.2f}" for t, v in result["memberships"].items()))
    for rule in result["fired_rules"]:
        print(f"  {rule['id']:<4} {rule['explanation']}")
    if not result["fired_rules"]:
        print("  no rule fired → default_output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
