"""
Measured weights of the agents (TOD-47): writes ``agents/<name>/eval.yaml`` from evaluation reports — the weight of
an agent's claims in validation is its held-out accuracy (Wilson 95 % lower bound), never a typed value.

* video — ``reports/action_classifier/metrics.json``: accuracy of the skeleton action classifier on its held-out
  test windows (HMDB51 test1 + NTU RGB+D 60 xsub_val). Public datasets, not our cameras: the domain gap is unmeasured;
* audio — ``config/uncertainty.yaml`` ``audio.keyword_danger``: danger-word recognition on noisy synthetic speech
  (``tools/fit_uncertainty``, TOD-45) — the evidence its opinions rest on. Intent patterns are not measured separately.

``python -m tools.agent_eval [--check]`` (``--check``: exit 1 when a written file differs from the reports).
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from typing import Any, Dict

import yaml

from core.calibration import wilson_lower

AGENTS = "agents"


def sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def video(path: str = "reports/action_classifier/metrics.json") -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    n = int(sum(data["test_counts"].values()))
    correct = round(float(data["test"]["accuracy"]) * n)
    return {"accuracy": round(wilson_lower(correct, n), 4), "n": n, "point_accuracy": data["test"]["accuracy"],
            "macro_f1": data["test"]["macro_f1"], "metric": "Wilson 95 % lower bound of the window accuracy",
            "source": f"{path} (sha {sha(path)}): skeleton classifier on held-out "
                      f"{' + '.join(data['datasets'])} test windows; domain gap to the deployed cameras unmeasured",
            "date": str(data.get("trained_at", ""))[:10]}


def audio(path: str = "config/uncertainty.yaml") -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        entry = (yaml.safe_load(f)["audio"])["keyword_danger"]
    return {"accuracy": round(float(entry["accuracy_lower"]), 4), "n": int(entry["n"]),
            "point_accuracy": entry["accuracy"], "metric": "Wilson 95 % lower bound of danger-word recognition",
            "source": f"{path} audio.keyword_danger (sha {sha(path)}): {entry['source']}",
            "date": str(entry.get("date", ""))}


SOURCES = {"video": video, "audio": audio}


def render(name: str, entry: Dict[str, Any]) -> str:
    head = (f"# Measured weight of the {name} agent — written by python -m tools.agent_eval; do not edit.\n"
            f"# generated {dt.date.today().isoformat()}\n")
    return head + yaml.safe_dump(entry, allow_unicode=True, sort_keys=False, width=100)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--check", action="store_true", help="fail when an eval file is out of date")
    args = parser.parse_args(argv)
    stale = []
    for name, source in SOURCES.items():
        entry = source()
        path = os.path.join(AGENTS, name, "eval.yaml")
        if args.check:
            with open(path, encoding="utf-8") as f:
                if yaml.safe_load(f) != entry:
                    stale.append(path)
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(render(name, entry))
        print(f"{path}: accuracy {entry['accuracy']} (n {entry['n']})")
    if stale:
        print("out of date: " + ", ".join(stale) + " (python -m tools.agent_eval)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
