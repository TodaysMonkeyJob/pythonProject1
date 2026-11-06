"""
Explain one inference of an HFS rule base (A1 / TOD-41)::

    python -m modules.hfs gas max_norm=0.6 mixture_index=1.2 dynamics=0.1
    python -m modules.hfs top risk_gas=0.88 risk_camera=0.1        # risk_audio missing → partial
    python -m modules.hfs --check                                   # validate + metadata of config/fuzzy/*.yaml
"""
import argparse
import os
import sys
import time

from core.fuzzy import FuzzyConfigError, FuzzyEngine, load_yaml
from core.fuzzy.governance import metadata_errors, short, version_hash
from core.fuzzy.validate import subsumed
from modules.hfs.base import CONFIG_DIR, level_of
from modules.hfs.top import SUBSYSTEMS

LEVELS = (*SUBSYSTEMS, "top")


def check(config_dir: str = CONFIG_DIR) -> int:
    failed = 0
    for name in LEVELS:
        path = os.path.join(config_dir, f"{name}.yaml")
        try:
            data = load_yaml(path)
            engine = FuzzyEngine.from_dict(data)
            missing = metadata_errors({"rules": data.get("rules")})
            if missing:
                raise FuzzyConfigError(missing, f"{name} rule metadata")
            shapes = {part[0] for var in engine.variables.values() for parts in var.terms.values() for part in parts}
            print(f"ok  {name:<7} {len(engine.rules)} rules, {len(engine.variables)} variables, "
                  f"shapes {'/'.join(sorted(shapes))}, version {short(version_hash({'rules': data.get('rules')}))}")
            for warning in subsumed(engine.rules):
                print(f"    warning: {warning}")
        except (FuzzyConfigError, OSError) as e:
            print(f"FAIL {name}: {e}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Autonomous hierarchical fuzzy system: explain one inference")
    parser.add_argument("level", nargs="?", choices=LEVELS)
    parser.add_argument("inputs", nargs="*", help="variable=value")
    parser.add_argument("--dir", default=CONFIG_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check or not args.level:
        return check(args.dir)
    engine = FuzzyEngine.from_dict(load_yaml(os.path.join(args.dir, f"{args.level}.yaml")))
    inputs = {}
    for item in args.inputs:
        key, _, value = item.partition("=")
        inputs[key] = float(value)
    start = time.perf_counter()
    result = engine.infer(inputs)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"{args.level}: risk = {result['crisp']:.3f} ({level_of(engine, result['crisp'])}), "
          f"α-cut {result['cut']}, {elapsed:.2f} ms")
    for rule in result["fired_rules"]:
        print(f"  {rule['id']:<4} {rule['explanation']}")
    if not result["fired_rules"]:
        print("  no rule fired → default_output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
