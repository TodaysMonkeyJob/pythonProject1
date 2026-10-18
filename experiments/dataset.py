"""
Evaluation dataset (F20 / TOD-31): a manifest of scenarios, each an event-log replay file plus expert labels.

``<dataset>/manifest.json``::

    {"name": "fixit-eval-2026-10", "synthetic": false, "scenarios": [
      {"scenario_id": "adv_07", "category": "adversarial", "attack_type": "sign", "event_log": "scenarios/adv_07.jsonl",
       "description": "smoke + sign SYSTEM: status safe", "labels": {"annotator_a": {"level": "high", "risk": 0.75},
                                                                       "annotator_b": {"level": "high", "risk": 0.7}}}]}

* category ∈ normal | dangerous | adversarial; attack_type for adversarial (speech_claim, sign, tts_playback,
  voice_mismatch); ``danger_onset`` (ISO time, dangerous / adversarial) = when the danger starts — the latency
  reference; event logs are events.jsonl lines (raw recordings stay outside the repo, see ``record.py``);
* expert label = consensus of the annotators: risk = mean, level = the level of that mean (``thresholds``) unless
  all annotators agree on a level; Cohen's κ over the two first annotators is reported;
* splits are deterministic by seed (stratified by category); ``dataset_hash`` covers the manifest and every log.
"""
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

LEVELS = ("low", "medium", "high", "critical")
CATEGORIES = ("normal", "dangerous", "adversarial")
THRESHOLDS = {"medium": 0.25, "high": 0.5, "critical": 0.8}


def level_of(risk: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    thresholds = thresholds or THRESHOLDS
    level = "low"
    for name in LEVELS[1:]:
        if risk >= thresholds[name] - 1e-9:
            level = name
    return level


@dataclass
class Scenario:
    scenario_id: str
    category: str
    event_log: str                         # absolute path
    labels: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    attack_type: Optional[str] = None
    description: str = ""
    danger_onset: Optional[str] = None     # ISO time the danger starts (latency reference; annotators mark it)

    @property
    def expert_risk(self) -> float:
        risks = [float(v["risk"]) for v in self.labels.values() if v.get("risk") is not None]
        if not risks:
            raise ValueError(f"{self.scenario_id}: no expert label")
        return round(sum(risks) / len(risks), 4)

    @property
    def expert_level(self) -> str:
        levels = {v.get("level") for v in self.labels.values() if v.get("level")}
        return levels.pop() if len(levels) == 1 else level_of(self.expert_risk)

    def events(self) -> List[str]:
        with open(self.event_log, encoding="utf-8") as f:
            return [line for line in f if line.strip()]


@dataclass
class Dataset:
    name: str
    root: str
    scenarios: List[Scenario]
    synthetic: bool = False

    @classmethod
    def load(cls, path: str) -> "Dataset":
        """:param path: manifest.json or the folder containing it"""
        manifest = os.path.join(path, "manifest.json") if os.path.isdir(path) else path
        root = os.path.dirname(os.path.abspath(manifest))
        with open(manifest, encoding="utf-8") as f:
            data = json.load(f)
        scenarios = []
        for s in data["scenarios"]:
            if s["category"] not in CATEGORIES:
                raise ValueError(f"{s['scenario_id']}: unknown category {s['category']!r}")
            scenarios.append(Scenario(s["scenario_id"], s["category"], os.path.join(root, s["event_log"]),
                                      s.get("labels") or {}, s.get("attack_type"), s.get("description", ""),
                                      s.get("danger_onset")))
        ids = [s.scenario_id for s in scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate scenario ids in the manifest")
        return cls(data.get("name", os.path.basename(root)), root, scenarios, bool(data.get("synthetic")))

    def labelled(self) -> List[Scenario]:
        return [s for s in self.scenarios if s.labels]

    def hash(self) -> str:
        digest = hashlib.sha256()
        for scenario in sorted(self.scenarios, key=lambda s: s.scenario_id):
            digest.update(json.dumps([scenario.scenario_id, scenario.category, scenario.attack_type,
                                      scenario.labels], sort_keys=True).encode())
            with open(scenario.event_log, "rb") as f:
                digest.update(hashlib.sha256(f.read()).digest())
        return digest.hexdigest()

    def split(self, seed: int = 42, calibration_share: float = 0.3) -> Tuple[List[Scenario], List[Scenario]]:
        """:return: (calibration, evaluation) — stratified by category, fixed by seed"""
        calibration, evaluation = [], []
        rng = random.Random(seed)
        for category in CATEGORIES:
            group = sorted((s for s in self.labelled() if s.category == category), key=lambda s: s.scenario_id)
            rng.shuffle(group)
            cut = int(round(len(group) * calibration_share))
            calibration += group[:cut]
            evaluation += group[cut:]
        return calibration, evaluation


def cohen_kappa(a: List[str], b: List[str], labels=LEVELS) -> Optional[float]:
    """Cohen's κ of two annotators' levels (None when undefined)"""
    if not a or len(a) != len(b):
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum((a.count(label) / n) * (b.count(label) / n) for label in labels)
    return None if expected >= 1 else round((observed - expected) / (1 - expected), 4)


def annotator_kappa(scenarios: List[Scenario]) -> Optional[Dict[str, Any]]:
    """κ over the two most frequent annotators on the scenarios both labelled"""
    counts: Dict[str, int] = {}
    for s in scenarios:
        for name in s.labels:
            counts[name] = counts.get(name, 0) + 1
    names = sorted(counts, key=lambda k: -counts[k])[:2]
    if len(names) < 2:
        return None
    both = [s for s in scenarios if all(n in s.labels and s.labels[n].get("level") for n in names)]
    kappa = cohen_kappa([s.labels[names[0]]["level"] for s in both], [s.labels[names[1]]["level"] for s in both])
    return {"annotators": names, "n": len(both), "kappa": kappa}
