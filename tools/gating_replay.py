"""
Replays events through trust scoring → aggregator → risk corridor → LLM gate and compares the number of GPT calls
with "GPT on every snapshot" (F19 / TOD-30 acceptance).

    python -m tools.gating_replay --synthetic-routine 60     # an hour of people working normally (generated)
    python -m tools.gating_replay "logs/2026-10-18/events.jsonl"

Everything runs in memory / a temporary folder (the logs are only read). GPT is a stub that answers ``low`` (what it
answers for routine scenes); a call happens when the gate says so and ``min_llm_interval_sec`` allows it (overrides
bypass the interval). Reported: snapshots, baseline calls, gated calls (overrides, deferred by the interval),
reduction, final risk levels gated vs baseline, estimated tokens saved.
"""
import argparse
import glob
import json
import os
import random
import sys
import tempfile
from collections import Counter
from typing import Iterator, List

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.events import Event, make_event  # noqa: E402
from modules.audio.attribution import iso  # noqa: E402


def synthetic_routine(minutes: int, seed: int = 7, start: float = 1_792_311_200.0) -> Iterator[Event]:
    """Three colleagues working in an office (known faces, ordinary actions, clean air, everyday speech)."""
    rng = random.Random(seed)
    people = [("oleh", "Oleh"), ("anna", "Anna"), ("ivan", "Ivan")]
    actions = ["sitting", "standing", "walking"]
    phrases = ["Надішли мені звіт", "Де лежить степлер?", "Я на дзвінок", "Каву будеш?", "Добре, зроблю"]

    def ev(t, source, type, payload, device):
        return make_event(source, type, payload, device_id=device, location_id="office").model_copy(
            update={"timestamp": iso(t)})

    for track, (pid, name) in enumerate(people, start=1):
        yield ev(start, "camera", "object_detected", {"track_id": track, "class": "person", "state": "appeared"},
                 "cam-1")
        yield ev(start + 1, "camera", "person_identified", {"track_id": track, "person_id": pid, "identity": name,
                                                            "known": True, "similarity": 0.72}, "cam-1")
    for o, cls in enumerate(("laptop", "chair", "cup"), start=10):
        yield ev(start + 2, "camera", "object_detected", {"track_id": o, "class": cls, "state": "appeared"}, "cam-1")
    t = start + 5
    while t < start + minutes * 60:
        roll = rng.random()
        track = rng.randint(1, 3)
        if roll < 0.08:
            yield ev(t, "camera", "action", {"track_id": track, "action": rng.choice(actions)}, "cam-1")
        elif roll < 0.12:
            pid, name = people[track - 1]
            yield ev(t, "audio", "speech", {"text": rng.choice(phrases), "language": "uk", "person_id": pid,
                                            "speaker_name": name, "speaker_status": "known", "speaker_score": 0.6,
                                            "asr_confidence": 0.85, "start": iso(t - 2), "end": iso(t)}, "mic-1")
        elif roll < 0.2:
            yield ev(t, "camera", "motion", {"state": "start", "area_ratio": round(rng.uniform(0.02, 0.1), 3)},
                     "cam-1")
        yield ev(t, "gas", "gas_reading", {"gas": "clean_air", "concentration_pct": round(rng.uniform(0, 0.1), 3),
                                           "trend": "stable", "gas_confidence": 0.97,
                                           "rise_rate_pct_per_min": round(rng.uniform(-0.05, 0.05), 3)}, "gas-1")
        t += 5


def from_logs(pattern: str) -> Iterator[Event]:
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    event = Event.from_json(line)
                except ValueError:
                    continue
                event.payload.pop("trust", None)          # re-scored with the current rules
                yield event


def replay(events: Iterator[Event], fuzzy: str, min_interval: float) -> dict:
    from modules.aggregator.aggregator import Aggregator
    from modules.envelope import RiskEnvelope
    from modules.envelope.projection import level_of
    from modules.llm.gating import LLMGate, estimate_tokens
    from modules.trust import TrustScorer
    from modules.trust.channels import epoch

    scorer = TrustScorer.from_config(fuzzy)
    envelope = RiskEnvelope.from_config(fuzzy)
    gate = LLMGate.from_config(fuzzy, min_interval)
    clock = [0.0]
    with tempfile.TemporaryDirectory() as tmp:
        aggregator = Aggregator({"aggregator": {"export_dir": tmp, "incidents": {"enabled": False},
                                                "write_snapshot_files": False, "export_csv": False,
                                                "global_snapshot_min": 0},
                                 "trust": {"fuzzy_config": fuzzy}}, clock=lambda: clock[0])
        stats = Counter()
        mismatches: List[str] = []
        last_call, last_regular = {}, {}
        prompt = estimate_tokens(open(os.path.join("prompts", "security_analyst.md"), encoding="utf-8").read())

        def handle(snapshot, row):
            now = clock[0]
            location = snapshot.get("location_id") or ""
            _, facts, corridor = envelope.assess(location, now)
            ignored = sum(1 for c in facts["ignored"] if location not in last_call or c["at"] > last_call[location])
            decision = gate.evaluate(row, corridor, ignored, last_call.get(location), now)
            stats["snapshots"] += 1
            baseline = envelope.decide(location, "low", now=now)["risk_level_final"]
            if decision.call and (decision.override or now - last_regular.get(location, -1e18) >= min_interval
                                  or row["trigger"] == "critical_event"):
                stats["gated_calls"] += 1
                stats[f"override:{decision.override}"] += bool(decision.override)
                last_call[location] = now
                if row["trigger"] != "critical_event":
                    last_regular[location] = now
                level = envelope.decide(location, "low", now=now)["risk_level_final"]
            elif decision.call:
                stats["deferred"] += 1            # waits for the interval (a newer snapshot supersedes it)
                return
            else:
                stats["fuzzy_only"] += 1
                stats["tokens_saved"] += prompt + estimate_tokens(row["digest"]) + 150
                level = level_of(corridor["floor"], envelope.thresholds)
            if level != baseline:
                mismatches.append(f"{iso(now)} {location}: gated {level} vs baseline {baseline}")

        for event in events:
            clock[0] = epoch(event.timestamp) or clock[0]
            scorer.apply(event)
            envelope.observe(event)
            produced = aggregator.on_event(event) or aggregator.tick()
            if produced and produced.get("trigger") != "global":
                row = next(r for r in aggregator.outbox.pending(limit=1000)
                           if r["payload"]["snapshot_id"] == produced["snapshot_id"])
                handle(produced, row)
        aggregator.close()
    stats["baseline_calls"] = stats["snapshots"]
    return {"stats": dict(stats), "mismatches": mismatches}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="How many GPT calls the fuzzy gate saves on a recording")
    parser.add_argument("pattern", nargs="?", help="events.jsonl glob (read only)")
    parser.add_argument("--synthetic-routine", type=int, metavar="MIN", help="generated routine recording, minutes")
    parser.add_argument("--fuzzy", default="config/fuzzy.yaml")
    parser.add_argument("--min-interval", type=float, default=300.0, help="llm.min_llm_interval_sec")
    args = parser.parse_args(argv)
    if not args.pattern and not args.synthetic_routine:
        parser.error("give an events.jsonl glob or --synthetic-routine MINUTES")
    events = synthetic_routine(args.synthetic_routine) if args.synthetic_routine else from_logs(args.pattern)
    result = replay(events, args.fuzzy, args.min_interval)
    s = result["stats"]
    if not s.get("snapshots"):
        print("No location snapshots produced", file=sys.stderr)
        return 1
    reduction = 100 * (1 - s.get("gated_calls", 0) / s["snapshots"])
    print(f"snapshots {s['snapshots']} → GPT calls: every snapshot {s['baseline_calls']}, "
          f"gated {s.get('gated_calls', 0)} ({reduction:.0f} % fewer); fuzzy-only {s.get('fuzzy_only', 0)}, "
          f"deferred by the interval "
          f"{s.get('deferred', 0)}")
    overrides = {k.split(':', 1)[1]: v for k, v in s.items() if k.startswith("override:") and v}
    print(f"overrides {overrides or {}}; estimated tokens saved ≈ {s.get('tokens_saved', 0)}")
    print(f"final risk level differs from the baseline in {len(result['mismatches'])} snapshot(s)")
    for line in result["mismatches"][:10]:
        print("  " + line)
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
