"""
Imports the pre-registry face data (``faces.npz`` + ``data/faces/unknown/unknown_N``) into the identity registry.

    python -m tools.migrate_faces_npz                 # identity.legacy_db_path → identity.registry_path
    python -m tools.migrate_faces_npz --npz other/faces.npz

People that already exist in the registry (same name / unknown_N) are skipped, so it is safe to run twice. The
identity module runs it automatically when the registry is empty. Legacy files are not deleted.
"""
import argparse
import os
import sys
from typing import List, Optional

if __package__ in (None, ""):  # allow `python tools/migrate_faces_npz.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config  # noqa: E402
from core.identity import registry_from_config  # noqa: E402
from modules.vision.legacy_faces import migrate  # noqa: E402


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Import faces.npz and unknown folders into the identity registry")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--npz", help="Overrides identity.legacy_db_path")
    parser.add_argument("--unknown-dir", help="Legacy unknown folders (default: <npz folder>/unknown)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config).get("identity", {})
    npz = args.npz or cfg.get("legacy_db_path", "data/faces/faces.npz")
    if not os.path.isfile(npz):
        raise SystemExit(f"{npz} not found")
    registry = registry_from_config(cfg)
    try:
        report = migrate(registry, npz, args.unknown_dir)
        print(f"Imported {report['known']} known and {report['unknown']} unknown people, {report['images']} face "
              f"images ({report['skipped']} skipped) into {registry.path}")
    finally:
        registry.close()


if __name__ == "__main__":
    main()
