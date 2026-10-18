"""
Building the recorded evaluation set (F20 / TOD-31) from your own runs of the system.

1. Record: run ``python main.py`` and act the scenario out (normal / dangerous / adversarial — ask 2–3 people who do
   not know the defence to invent the manipulation attempts). Raw camera / microphone recordings, if you keep any,
   go to ``data/eval/raw/`` (git-ignored — never committed).
2. Cut the scenario's events out of the day's log into the dataset (the event log is the replay file)::

       python -m experiments.record cut --log logs/2026-10-20/events.jsonl --start 14:31:00 --end 14:34:00 \\
           --id adv_07 --category adversarial --attack sign --onset 14:31:20 --description "smoke + sign"

   (times are local; ``--onset`` = when the danger starts, the latency reference)
3. Each annotator labels independently in a CSV ``scenario_id,level,risk`` (level low|medium|high|critical, risk
   0..1), then::

       python -m experiments.record label --annotator annotator_a --csv labels_a.csv
       python -m experiments.record status          # counts per category, labelled by whom, Cohen's κ

The dataset lives in ``data/eval/recorded/`` (manifest.json + scenarios/*.jsonl) — only event logs and labels.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, time as dtime
from typing import Any, Dict

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.dataset import CATEGORIES, LEVELS, Dataset, annotator_kappa  # noqa: E402

ATTACKS = ("speech_claim", "sign", "tts_playback", "voice_mismatch", "other")


def _manifest(dataset: str) -> Dict[str, Any]:
    path = os.path.join(dataset, "manifest.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"name": os.path.basename(os.path.abspath(dataset)), "synthetic": False, "scenarios": []}


def _save(dataset: str, manifest: Dict[str, Any]) -> None:
    os.makedirs(dataset, exist_ok=True)
    with open(os.path.join(dataset, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)


def _local(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone()


def cut(args) -> int:
    if args.category not in CATEGORIES:
        raise SystemExit(f"category must be one of {CATEGORIES}")
    if args.category == "adversarial" and args.attack not in ATTACKS:
        raise SystemExit(f"adversarial scenarios need --attack {ATTACKS}")
    start, end = dtime.fromisoformat(args.start), dtime.fromisoformat(args.end)
    lines, onset = [], None
    with open(args.log, encoding="utf-8") as f:
        for line in f:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            moment = _local(event["timestamp"]).replace(microsecond=0)        # whole seconds, like --start / --end
            if start <= moment.time() <= end and event.get("type") not in ("heartbeat", "location_summary"):
                event.get("payload", {}).pop("trust", None)          # re-scored on replay
                lines.append(json.dumps(event, ensure_ascii=False))
                if args.onset and onset is None and moment.time() >= dtime.fromisoformat(args.onset):
                    onset = event["timestamp"]
    if not lines:
        raise SystemExit(f"No events between {args.start} and {args.end} in {args.log}")
    manifest = _manifest(args.dataset)
    if any(s["scenario_id"] == args.id for s in manifest["scenarios"]):
        raise SystemExit(f"{args.id} already exists in the manifest")
    os.makedirs(os.path.join(args.dataset, "scenarios"), exist_ok=True)
    path = os.path.join("scenarios", f"{args.id}.jsonl")
    with open(os.path.join(args.dataset, path), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    manifest["scenarios"].append({"scenario_id": args.id, "category": args.category,
                                  "attack_type": args.attack if args.category == "adversarial" else None,
                                  "event_log": path, "description": args.description or "", "labels": {},
                                  "danger_onset": onset})
    _save(args.dataset, manifest)
    print(f"{args.id}: {len(lines)} events → {os.path.join(args.dataset, path)}")
    return 0


def label(args) -> int:
    manifest = _manifest(args.dataset)
    by_id = {s["scenario_id"]: s for s in manifest["scenarios"]}
    count = 0
    with open(args.csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scenario = by_id.get(row["scenario_id"])
            if scenario is None:
                print(f"  unknown scenario {row['scenario_id']} — skipped", file=sys.stderr)
                continue
            level, risk = row["level"].strip(), float(row["risk"])
            if level not in LEVELS or not 0 <= risk <= 1:
                raise SystemExit(f"{row['scenario_id']}: level must be {LEVELS}, risk 0..1")
            scenario.setdefault("labels", {})[args.annotator] = {"level": level, "risk": risk}
            count += 1
    _save(args.dataset, manifest)
    print(f"{count} labels by {args.annotator}")
    return 0


def status(args) -> int:
    if not os.path.isfile(os.path.join(args.dataset, "manifest.json")):
        print(f"No dataset at {args.dataset} yet")
        return 1
    dataset = Dataset.load(args.dataset)
    print(f"{dataset.name}: {len(dataset.scenarios)} scenarios — " + ", ".join(
        f"{c} {n}/40" for c, n in Counter(s.category for s in dataset.scenarios).items()))
    annotators = Counter(name for s in dataset.scenarios for name in s.labels)
    print("labels: " + (", ".join(f"{a} {n}" for a, n in annotators.items()) or "none"))
    print(f"agreement: {annotator_kappa(dataset.labelled())}")
    print(f"attacks: {dict(Counter(s.attack_type for s in dataset.scenarios if s.category == 'adversarial'))}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cut, label and check the recorded evaluation set")
    parser.add_argument("--dataset", default="data/eval/recorded")
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("cut", help="copy a time range of a day's events into a scenario")
    c.add_argument("--log", required=True)
    c.add_argument("--start", required=True, help="local HH:MM:SS")
    c.add_argument("--end", required=True, help="local HH:MM:SS")
    c.add_argument("--id", required=True)
    c.add_argument("--category", required=True, choices=CATEGORIES)
    c.add_argument("--attack", choices=ATTACKS)
    c.add_argument("--onset", help="local HH:MM:SS when the danger starts")
    c.add_argument("--description")
    lab = commands.add_parser("label", help="merge one annotator's CSV (scenario_id,level,risk)")
    lab.add_argument("--annotator", required=True)
    lab.add_argument("--csv", required=True)
    commands.add_parser("status")
    args = parser.parse_args(argv)
    return {"cut": cut, "label": label, "status": status}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
