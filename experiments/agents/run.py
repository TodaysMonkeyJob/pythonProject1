"""
Article 2 comparison (TOD-47): Article 1 (fuzzy only) vs the multi-agent system on the same scenarios — one command::

    python -m experiments.agents.run --seed 42 [--out reports/article2_<date>]

Scenarios: the 46 Article 1 scenarios (``experiments/hfs/scenarios.yaml``) + the multi-agent ones
(``experiments/agents/scenarios.yaml``: actions, spoken attacks). Each is replayed through the real
``HierarchicalFuzzySystem`` (→ ``hfs_risk``) and then through ``agents.framework.orchestrator.AgentHost`` with the
shipped agents and their measured weights, in these configurations:

| configuration | agents on |
| -- | -- |
| article1 | fuzzy (the corridor decides) |
| multi_agent | fuzzy, video, audio, validation |
| no_video / no_audio | multi_agent without that agent |
| no_physical_guard | multi_agent, but any estimate may lower the risk below the corridor (the protocol's guard off) |

Metrics against the labels: macro-F1 / accuracy of the decided level and of the level after the operator answers
an ``agent_operator_request`` (assumed to accept the agents' level — an upper bound); attack success rate (attack
scenarios decided below their label); operator load (requests per scenario); decision latency on this machine (HFS
evaluation + agents round, inline). The end-to-end scenario m01 (smoke + "все гаразд" + a stranger) is written as a
full message trace (``trace_m01.jsonl``).
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from agents.framework.orchestrator import AgentHost
from agents.framework.registry import AgentRegistry
from core.events import Event, make_event
from experiments.hfs import metrics as M
from experiments.hfs.run import LOCATION, WEDNESDAY, events_of, sha
from modules.hfs.worker import HierarchicalFuzzySystem

RANK = {level: i for i, level in enumerate(M.LEVELS)}
ALL = {"validation": True, "video": True, "audio": True, "echo": False}
CONFIGS = {                                    # name → (agents on, agents.settings)
    "article1": ({"validation": False, "video": False, "audio": False, "echo": False}, {}),
    "multi_agent": (ALL, {}),
    "no_video": ({**ALL, "video": False}, {}),
    "no_audio": ({**ALL, "audio": False}, {}),
    "no_physical_guard": (ALL, {"validation": {"require_physical": False}}),   # speech may lower the risk
}
TRACE = "m01"
START = 1_000_000.0


def scenario_events(scenario: Dict[str, Any], seed: int) -> Tuple[List[Tuple[float, Event]], List[Dict]]:
    """:return: the Article 1 evidence events + the action events only the agents read, sorted by time"""
    events, devices = events_of(scenario, np.random.default_rng(seed))
    for a in scenario.get("actions") or []:
        events.append((40.0, make_event("camera", "action", {"track_id": a["track"], "action": a["action"],
                                                             "source": a.get("source", "rules"),
                                                             "probability": a.get("probability")},
                                        confidence=a.get("probability"), location_id=LOCATION, device_id="cam-1")))
    events.sort(key=lambda item: item[0])
    return events, devices


def article1_payload(scenario: Dict[str, Any], config: Dict[str, Any], events, devices) -> Dict[str, Any]:
    """The hfs_risk payload of the scenario after 60 s (the real HFS; ``action`` events are offered and ignored)."""
    hh, mm = (int(x) for x in str((scenario.get("camera") or {}).get("time", "11:00")).split(":"))
    moment = dt.datetime.combine(WEDNESDAY, dt.time(hh, mm))
    now = [START]
    system = HierarchicalFuzzySystem(config, devices=lambda: devices, clock=lambda: now[0],
                                     local_now=lambda: moment + dt.timedelta(seconds=now[0] - START))
    for t, event in events:
        now[0] = START + t
        system.observe(event)
    now[0] = START + 60.0
    return system.evaluate(LOCATION, now[0])          # incl. the cross-subsystem escalation context (TOD-51)


class Recorder:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def __call__(self, type_, payload, severity, location_id):
        self.events.append({"type": type_, "severity": severity, "location_id": location_id, "payload": payload})

    def of(self, type_):
        return [e["payload"] for e in self.events if e["type"] == type_]


def host_for(enabled: Dict[str, bool], settings: Optional[Dict[str, Any]] = None) -> Tuple[AgentHost, Recorder]:
    recorder = Recorder()
    registry = AgentRegistry({"mode": "hybrid", "agents": {"enabled_agents": enabled, "settings": settings or {}}},
                             entry_points=False, state_path=None)
    host = AgentHost(registry, recorder, decides=True)
    host.start()
    return host, recorder


def decide(enabled: Dict[str, bool], settings: Dict[str, Any], events, hfs_event: Event) -> Dict[str, Any]:
    host, recorder = host_for(enabled, settings)
    try:
        for _, event in events:
            host.observe(event)
        began = time.perf_counter()
        host.observe(hfs_event)
        ms = (time.perf_counter() - began) * 1000
    finally:
        host.stop()
    decision = recorder.of("agent_decision")[-1]
    request = (recorder.of("agent_operator_request") or [None])[-1]
    return {"level": decision["level"], "risk": decision["risk"], "branch": decision["branch"],
            "operator": request is not None, "level_operator": request["level_agents"] if request else
            decision["level"], "ms": ms, "agents": decision["agents"], "reasons": decision["reasons"],
            "trace": recorder.events}


def summarise(scenarios: List[Dict[str, Any]], results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    truth = [s["label"] for s in scenarios]
    auto = [results[s["id"]]["level"] for s in scenarios]
    with_op = [results[s["id"]]["level_operator"] for s in scenarios]
    attacks = [s for s in scenarios if s.get("attack")]
    ms = sorted(results[s["id"]]["ms"] for s in scenarios)
    groups = {}
    for group in sorted({s.get("group", "") for s in scenarios}):
        items = [s for s in scenarios if s.get("group") == group]
        groups[group] = M.macro_f1([s["label"] for s in items], [results[s["id"]]["level"] for s in items])
    return {"macro_f1": M.macro_f1(truth, auto), "accuracy": round(float(np.mean([a == t for a, t in
                                                                                  zip(auto, truth, strict=True)])), 4),
            "macro_f1_operator": M.macro_f1(truth, with_op),
            "accuracy_operator": round(float(np.mean([a == t for a, t in zip(with_op, truth, strict=True)])), 4),
            "attack_success_rate": round(float(np.mean([RANK[results[s["id"]]["level"]] < RANK[s["label"]]
                                                        for s in attacks])), 4) if attacks else None,
            "attacks": len(attacks),
            "operator_requests": int(sum(results[s["id"]]["operator"] for s in scenarios)),
            "operator_load": round(float(np.mean([results[s["id"]]["operator"] for s in scenarios])), 4),
            "latency_ms": {"mean": round(float(np.mean(ms)), 3), "p95": round(ms[int(0.95 * (len(ms) - 1))], 3)},
            "macro_f1_by_group": groups,
            "misses": [{"id": s["id"], "label": s["label"], "got": results[s["id"]]["level"],
                        "branch": results[s["id"]]["branch"]} for s in scenarios
                       if results[s["id"]]["level"] != s["label"]]}


def run(seed: int, hfs_path: str, agents_path: str, configs: Optional[Dict[str, tuple]] = None):
    """:return: (scenarios, results config → id → result, hfs ms per scenario)"""
    with open("config.yaml", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    config = {"hfs": base["hfs"], "identity": base.get("identity", {}), "gas": base.get("gas", {})}
    scenarios = []
    for path in (hfs_path, agents_path):
        with open(path, encoding="utf-8") as f:
            scenarios += yaml.safe_load(f)["scenarios"]
    results: Dict[str, Dict[str, Any]] = {name: {} for name in configs or CONFIGS}
    hfs_ms = {}
    for i, scenario in enumerate(scenarios):
        events, devices = scenario_events(scenario, seed + i)
        began = time.perf_counter()
        payload = article1_payload(scenario, config, events, devices)
        hfs_ms[scenario["id"]] = (time.perf_counter() - began) * 1000
        hfs_event = make_event("system", "hfs_risk", payload, location_id=LOCATION, device_id="hub:hfs")
        for name, (enabled, settings) in (configs or CONFIGS).items():
            results[name][scenario["id"]] = {**decide(enabled, settings, events, hfs_event), "hfs": payload,
                                             "evidence": events}
    return scenarios, results, hfs_ms


def write_trace(path: str, result: Dict[str, Any]) -> None:
    """Every message of one scenario: evidence events, the hfs_risk, every claim, the decision."""
    with open(path, "w", encoding="utf-8") as f:
        for t, event in result["evidence"]:
            f.write(json.dumps({"t": t, "kind": "evidence", "type": event.type, "source": event.source,
                                "event_id": event.event_id, "payload": event.payload}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"t": 60.0, "kind": "hfs_risk", "payload": {k: result["hfs"][k] for k in (
            "overall", "level", "interval", "risk1", "risk2", "risk3", "missing")}}, ensure_ascii=False) + "\n")
        for message in result["trace"]:
            if message["type"] == "agent_status":
                continue
            f.write(json.dumps({"t": 60.0, "kind": message["type"], "payload": message["payload"]},
                               ensure_ascii=False, default=str) + "\n")


def reading(scenarios, results, summary) -> List[str]:
    """The interpretation, with the numbers of this run."""
    a1, ma, guard = summary["article1"], summary["multi_agent"], summary["no_physical_guard"]
    hfs_ids = [s["id"] for s in scenarios if not s["id"].startswith("m")]
    same = sum(results["multi_agent"][i]["level"] == results["article1"][i]["level"] for i in hfs_ids)
    actions = [s["id"] for s in scenarios if s.get("actions") and s["label"] in ("high", "critical")]
    asked = [i for i in actions if results["multi_agent"][i]["operator"]]
    return ["", "### Reading the results", "",
            f"* **The agents do not change Article 1.** On the {len(hfs_ids)} Article 1 scenarios the multi-agent "
            f"system decides the same level {same} times out of {len(hfs_ids)}: an agent that observed nothing risky "
            "says nothing, and danger words the HFS already reads only raise when they exceed the corridor.",
            f"* **The guard is what stops spoken attacks.** With the validation protocol, attack success is "
            f"{ma['attack_success_rate']:.0%} of {ma['attacks']}; switching only the physical-confirmation rule off "
            f"lets every \"everything is fine\" through ({guard['attack_success_rate']:.0%}, macro-F1 "
            f"{guard['macro_f1']:.3f} vs {ma['macro_f1']:.3f}). Article 1 is immune for a different reason: it never "
            f"reads the meaning of speech ({a1['attack_success_rate']:.0%}).",
            f"* **Actions reach the decision through the operator.** Of the {len(actions)} risky-action scenarios, "
            f"{len(asked)} raise an operator request ({', '.join(asked) or 'none'}); the decided level stays at the "
            "corridor top until the operator answers, because a fall or a fight is outside what the measured "
            f"evidence of the fuzzy system supports. Operator load: {ma['operator_requests']} requests in "
            f"{len(scenarios)} scenarios ({ma['operator_load']:.1%}).",
            "* **Latency** of a round is under a millisecond here; the HFS evaluation dominates (table above)."]


def write_report(out: str, manifest: Dict[str, Any], scenarios, results, summary, hfs_ms) -> None:
    agent_ids = [s["id"] for s in scenarios if s["id"].startswith("m")]
    lines = [f"# Article 2 — Article 1 vs the multi-agent system ({manifest['date'][:10]})", "",
             f"`python -m experiments.agents.run --seed {manifest['seed']}` at commit `{manifest['git']}`: "
             f"{len(scenarios)} scenarios ({len(scenarios) - len(agent_ids)} of Article 1 + {len(agent_ids)} "
             "multi-agent), each replayed through the real HFS and the agents host with the shipped agents.", "",
             f"**Labels: {manifest['labels']}.** Agent weights = measured accuracy (Wilson lower bound): "
             + ", ".join(f"{k} {v}" for k, v in manifest["weights"].items()) + ".", "",
             "## Results", "",
             "| Configuration | Macro-F1 | Accuracy | Macro-F1 after the operator | Attack success rate | "
             "Operator requests | Decision latency mean / p95, ms |", "| -- | -- | -- | -- | -- | -- | -- |"]
    for name, s in summary.items():
        asr = "—" if s["attack_success_rate"] is None else f"{s['attack_success_rate']:.0%} of {s['attacks']}"
        lines.append(f"| `{name}` | {s['macro_f1']:.3f} | {s['accuracy']:.1%} | {s['macro_f1_operator']:.3f} | {asr} | "
                     f"{s['operator_requests']} ({s['operator_load']:.1%}) | {s['latency_ms']['mean']} / "
                     f"{s['latency_ms']['p95']} |")
    ms = sorted(hfs_ms.values())
    lines += ["", f"HFS evaluation (all subsystems + top, once per scenario): mean {np.mean(ms):.2f} ms, p95 "
              f"{ms[int(0.95 * (len(ms) - 1))]:.2f} ms. Latency of the agents round is inline on this machine, "
              "not the Raspberry Pi.", "",
              "“After the operator” assumes the operator accepts the agents' level whenever the validation agent asks "
              "(an upper bound of what the operator can add); the decided level never goes above the corridor "
              "without that answer.", "",
              "## Multi-agent scenarios", "",
              "| Scenario | Label | " + " | ".join(summary) + " | Why (label) |", "| -- " * (len(summary) + 3) + "|"]
    why = {s["id"]: s["why"] for s in scenarios}
    label = {s["id"]: s["label"] for s in scenarios}
    for sid in agent_ids:
        cells = []
        for name in summary:
            r = results[name][sid]
            op = f" → operator: {r['level_operator']}" if r["operator"] else ""
            cells.append(f"{r['level']} {r['risk']:.2f} ({r['branch']}){op}")
        lines.append(f"| {sid} | {label[sid]} | " + " | ".join(cells) + f" | {why[sid]} |")
    lines += reading(scenarios, results, summary)
    lines += ["", "## Disagreements with the labels (multi_agent)", ""]
    misses = summary["multi_agent"]["misses"]
    lines += [f"* {m['id']}: label {m['label']}, decided {m['got']} ({m['branch']})" for m in misses] or ["None."]
    r = results["multi_agent"][TRACE]
    lines += ["", f"## End-to-end trace: {TRACE} — {why[TRACE]}", "",
              f"`trace_{TRACE}.jsonl` holds every message: evidence events → `hfs_risk` → every agent claim → "
              "the decision. Summary:", "",
              f"* fuzzy corridor: {r['hfs']['level']} {r['hfs']['overall']:.2f}, interval {r['hfs']['interval']};",
              f"* agents in the round: {r['agents']};",
              f"* decision: **{r['level']} {r['risk']:.2f}**, branch `{r['branch']}` — " + "; ".join(r["reasons"]),
              "", "Files: metrics.json, scenarios.csv, trace_m01.jsonl."]
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Article 2: Article 1 vs the multi-agent system")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=os.path.join("reports", f"article2_{dt.date.today().isoformat()}"))
    parser.add_argument("--hfs-scenarios", default=os.path.join("experiments", "hfs", "scenarios.yaml"))
    parser.add_argument("--scenarios", default=os.path.join("experiments", "agents", "scenarios.yaml"))
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    scenarios, results, hfs_ms = run(args.seed, args.hfs_scenarios, args.scenarios)
    summary = {name: summarise(scenarios, results[name]) for name in results}
    weights = AgentRegistry({"agents": {}}, entry_points=False, state_path=None).weights()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(args.scenarios, encoding="utf-8") as f:
        status = yaml.safe_load(f).get("status")
    manifest = {"seed": args.seed, "git": git, "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "scenarios": len(scenarios), "labels": status,
                "weights": {k: v for k, v in weights.items() if v > 0},
                "files": {p: sha(p) for p in [args.hfs_scenarios, args.scenarios, "agents/video/eval.yaml",
                                              "agents/audio/eval.yaml", "config/fuzzy/top.yaml"]}}
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"manifest": manifest, "summary": summary}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "scenarios.csv"), "w", encoding="utf-8") as f:
        f.write("id,group,label," + ",".join(f"{n}_level,{n}_risk,{n}_branch,{n}_operator" for n in results) + "\n")
        for s in scenarios:
            row = [s["id"], s.get("group", ""), s["label"]]
            for name in results:
                r = results[name][s["id"]]
                row += [r["level"], f"{r['risk']:.4f}", r["branch"], r["level_operator"] if r["operator"] else ""]
            f.write(",".join(row) + "\n")
    write_trace(os.path.join(args.out, f"trace_{TRACE}.jsonl"), results["multi_agent"][TRACE])
    write_report(args.out, manifest, scenarios, results, summary, hfs_ms)
    print(f"→ {args.out}/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
