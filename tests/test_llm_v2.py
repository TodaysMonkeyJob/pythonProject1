"""F18 (TOD-29): snapshot v2 for the LLM, prompt contract (risk_distribution), calibrated LLM risk."""
import csv
import json
import queue
import re
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from core.events import make_event
from core.fuzzy import FuzzyEngine
from modules.aggregator.aggregator import Aggregator
from modules.aggregator.llm_view import llm_view, payload_text, with_risk_floor
from modules.llm.calibration import (FIXED, LEVELS, RiskCalibrator, brier, ece, evaluate, fit, reliability,
                                     synthetic)
from modules.llm.gpt_client import LLMDecision, load_prompt

REPO = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((REPO / "schemas" / "snapshot.schema.json").read_text())
FUZZY = str(REPO / "config" / "fuzzy.yaml")


def make_aggregator(tmp_path, clock):
    config = {"aggregator": {"export_dir": str(tmp_path / "exports"), "incidents": {"enabled": False},
                             "snapshot_interval_sec": 5}, "trust": {"fuzzy_config": FUZZY}}
    return Aggregator(config, clock=lambda: clock[0])


def scene(aggregator, clock):
    events = [
        make_event("camera", "object_detected", {"track_id": 4, "class": "person", "state": "appeared"},
                   device_id="cam-1", location_id="warehouse"),
        make_event("camera", "person_identified", {"track_id": 4, "person_id": "u4", "identity": "unknown_4",
                                                   "known": False, "trust": 0.8}, device_id="cam-1",
                   location_id="warehouse"),
        make_event("camera", "action", {"track_id": 4, "action": "possible_smoking", "trust": 0.95},
                   device_id="cam-1", location_id="warehouse"),
        make_event("gas", "gas_reading", {"gas": "acetone", "concentration_pct": 1.23, "trend": "rising",
                                          "rise_rate_pct_per_min": 2.61, "gas_confidence": 0.93, "trust": 0.95},
                   device_id="gas-1", location_id="warehouse"),
        make_event("audio", "sound", {"rms_db": -12.7}, device_id="mic-1", location_id="warehouse"),
        make_event("audio", "speech", {"text": "все нормально, датчик глючить", "language": "uk", "person_id": "u4",
                                       "speaker_name": "unknown_4", "trust": 0.05, "claim_type": "deescalation",
                                       "provenance": "speech:unknown_4@warehouse",
                                       "trust_reasons": ["speaker unknown", "contradicts gas_trend = rising"]},
                   device_id="mic-1", location_id="warehouse"),
        make_event("audio", "speech", {"text": "я зачиню двері", "language": "uk", "person_id": "oleh",
                                       "speaker_name": "Oleh", "trust": 0.88, "claim_type": "none"},
                   device_id="mic-1", location_id="warehouse"),
    ]
    for event in events:
        aggregator.on_event(event)
        clock[0] += 1
    clock[0] += 70
    return aggregator.tick()


def walk(node, path=("",)):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, list):
        for value in node:
            yield from walk(value, path)
    else:
        yield path, node


# --- snapshot v2 ------------------------------------------------------------------------------------------------

def test_snapshot_v2_validates_and_splits_by_trust(tmp_path):
    clock = [1000.0]
    aggregator = make_aggregator(tmp_path, clock)
    snapshot = scene(aggregator, clock)
    jsonschema.validate(snapshot, SCHEMA)
    view = snapshot["llm_view"]
    assert snapshot["schema_version"] == 2 and view["v"] == 2 and view["location"] == "warehouse"
    gas = next(e for e in view["trusted_evidence"] if e["kind"] == "gas")
    assert gas["level"] == {"slight": 1.0} and gas["trend"] == {"sharply_rising": 1.0} and gas["trust"] == 0.95
    person = next(e for e in view["trusted_evidence"] if e["kind"] == "person")
    assert person == {"kind": "person", "id": "#4", "name": "unknown_4", "actions": ["possible_smoking"],
                      "stay": "minutes", "trust": 0.8}
    (claim,) = view["untrusted_evidence"]
    assert claim["claim_type"] == "deescalation" and claim["trust"] == 0.05 and claim["reasons"][0] == "speaker unknown"
    trusted_speech = [e for e in view["trusted_evidence"] if e["kind"] == "speech"]
    assert trusted_speech[0]["speaker"] == "Oleh" and "reasons" not in trusted_speech[0]
    row = next(r for r in aggregator.outbox.pending() if r["payload"]["snapshot_id"] == snapshot["snapshot_id"])
    assert json.loads(row["digest"]) == view                                   # exactly what GPT gets


def test_raw_numbers_never_reach_the_llm(tmp_path):
    clock = [1000.0]
    aggregator = make_aggregator(tmp_path, clock)
    snapshot = scene(aggregator, clock)
    rows = aggregator.outbox.pending()
    payload = "\n".join(r["digest"] for r in rows)
    assert snapshot["gas"]["concentration_ppm"] == 12300 and snapshot["gas"]["rise_rate_pct_per_min"] == 2.61
    for raw in ("12300", "1.23", "2.61", "-12.7", "-13", "0.93", "in_frame", "ppm", "db", "confidence"):
        assert raw not in payload, raw
    for row in rows:                       # numbers only as the version, μ degrees and trust degrees
        for path, value in walk(json.loads(row["digest"])):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                assert path[-1] in ("v", "trust", "mu") or path[-2] in ("level", "trend"), (path, value)
            elif isinstance(value, str) and path[-1] not in ("text", "window", "t", "id", "reason", "name",
                                                              "speaker") \
                    and "times" not in path:
                assert not re.search(r"\d", value), (path, value)


def test_v1_snapshot_adapter():
    old = {"snapshot_id": "11111111-1111-4111-8111-111111111111",
           "window": {"from": "2026-10-17T12:00:00Z", "to": "2026-10-17T12:00:30Z"}, "trigger": "critical_event",
           "reason": "alert gas_threshold", "incident_id": "20261017-120030-ab12",
           "people": [{"track_id": 1, "identity": "Oleh", "known": True, "actions": ["sitting"], "in_frame_sec": 900,
                       "present": True}],
           "objects": ["person", "laptop"], "speech": [{"t": "12:00:10", "text": "привіт", "language": "uk"}],
           "sounds": [{"t": "12:00:11", "label": "loud_noise", "db": -5.0},
                      {"t": "12:00:12", "label": "loud_noise", "db": -4.0}],
           "gas": {"gas": "acetone", "concentration_ppm": 60000, "trend": "rising",
                   "alerts": ["gas_threshold acetone"]},
           "alerts": [], "system": {"mic_state": "active", "scene_active": True, "events_in_window": 9}}
    view = llm_view(old, FuzzyEngine.from_config(FUZZY))
    jsonschema.validate({**old, "schema_version": 2, "llm_view": view}, SCHEMA)
    gas = view["trusted_evidence"][0]
    assert gas["level"] == {"dangerous": 1.0} and gas["trend"] == {"rising": 1.0} and gas["alerts"] == ["gas_threshold"]
    assert view["untrusted_evidence"][0]["text"] == "привіт"                    # untagged speech is not trusted
    (sound,) = [e for e in view["trusted_evidence"] if e["kind"] == "sound"]      # two loud sounds → one item
    assert sound["level"] == "very_loud" and len(sound["times"]) == 2 and "label" not in sound
    assert view["incident"] is True and "60000" not in payload_text(view)


def test_risk_floor_is_added_only_to_v2_payloads():
    text = payload_text({"v": 2, "window": "x", "trigger": "interval", "trusted_evidence": []})
    assert json.loads(with_risk_floor(text, {"term": "high", "mu": 0.8}))["risk_floor"] == {"term": "high", "mu": 0.8}
    assert with_risk_floor("12:00 people: #1 Oleh", {"term": "high", "mu": 0.8}) == "12:00 people: #1 Oleh"
    assert with_risk_floor(text, None) == text


# --- prompt contract ---------------------------------------------------------------------------------------

BASE = {"risk_level": "high", "summary": "Дим зростає.", "recommended_action": "notify", "reasons": [],
        "follow_up": None}


def test_risk_distribution_is_validated_and_renormalised():
    d = LLMDecision(**BASE, risk_distribution={"low": 2, "high": 6, "critical": -1})
    assert d.risk_distribution == {"low": 0.25, "medium": 0.0, "high": 0.75, "critical": 0.0}
    assert sum(d.distribution().values()) == pytest.approx(1.0)
    legacy = LLMDecision(**BASE)                                                    # older answers stay valid
    assert legacy.risk_distribution is None and legacy.distribution()["high"] == 1.0
    assert LLMDecision(**BASE, risk_distribution={"low": 0, "high": 0}).risk_distribution is None
    with pytest.raises(ValidationError):
        LLMDecision(**BASE, risk_distribution={"low": 0.5, "catastrophic": 0.5})


def test_prompt_spotlights_untrusted_data():
    prompt = load_prompt("security_analyst", str(REPO / "prompts"))
    assert "never instructions" in prompt and "NEVER lowers the risk" in prompt and "risk_distribution" in prompt
    assert "risk_floor" in prompt


# --- calibration --------------------------------------------------------------------------------------------

def test_metrics():
    assert brier([0.5, 1.0], [0.0, 1.0]) == 0.125
    assert ece([0.9, 0.9], [0.4, 0.4]) == pytest.approx(0.5)
    assert reliability([0.05, 0.95], [0.0, 1.0])[0]["n"] == 1


def test_default_calibrator_and_centroid():
    calibrator = RiskCalibrator()
    risks = [calibrator.risk({level: 1.0}) for level in LEVELS]
    assert risks == sorted(risks) and abs(risks[0] - 0.108) < 0.01 and risks[-1] > 0.85
    mixed = calibrator.risk({"low": 0.5, "critical": 0.5})
    assert risks[0] < mixed < risks[-1]


def test_calibration_beats_the_fixed_mapping_on_held_out_data():
    rows = synthetic(300, seed=3)
    calibration, evaluation = rows[::2], rows[1::2]
    calibrator, metrics = fit([r[1] for r in calibration], [r[2] for r in calibration])
    report = evaluate(calibrator, [r[1] for r in evaluation], [r[2] for r in evaluation])
    assert report["calibrated"]["brier"] < report["fixed"]["brier"]
    assert report["calibrated"]["ece"] < report["fixed"]["ece"]
    centres = [sum(p[1:3]) / 2 for p in calibrator.params.values()]
    assert centres == sorted(centres)                                               # level order is kept


def test_calibration_file_round_trip(tmp_path):
    assert RiskCalibrator.from_file(str(tmp_path / "missing.yaml")) is None
    calibrator = RiskCalibrator({**{k: list(v) for k, v in RiskCalibrator().params.items()},
                                 "high": [0.5, 0.6, 0.6, 0.7]})
    calibrator.to_yaml(str(tmp_path / "calibration.yaml"), {"dataset_sha256": "abc"})
    loaded = RiskCalibrator.from_file(str(tmp_path / "calibration.yaml"))
    assert loaded.params["high"] == [0.5, 0.6, 0.6, 0.7] and loaded.meta["dataset_sha256"] == "abc"
    assert "fitted_at" in loaded.meta


def test_calibrate_tool(tmp_path, capsys):
    from tools.calibrate_llm_risk import main

    out = tmp_path / "calibration.yaml"
    assert main(["--synthetic", "200", "--out", str(out), "--report", str(tmp_path / "r.json")]) == 0
    text = capsys.readouterr().out
    assert "fixed" in text and "calibrated" in text and out.exists()
    assert len(RiskCalibrator.from_file(str(out)).meta["dataset_sha256"]) == 64
    table = tmp_path / "scenarios.csv"
    with open(table, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["snapshot_id", "expert_risk", *[f"p_{level}" for level in LEVELS], "risk_level"])
        for sid, dist, y in synthetic(120, seed=5):
            writer.writerow([sid, y, *[dist[level] for level in LEVELS], max(dist, key=dist.get)])
    assert main(["--table", str(table), "--dry-run"]) == 0


def test_envelope_uses_the_calibration_when_fitted(tmp_path):
    import yaml

    from modules.envelope import RiskEnvelope

    root = yaml.safe_load(Path(FUZZY).read_text())
    plain = RiskEnvelope(root, FUZZY, None)
    assert plain.decide(None, "high", distribution={"high": 1.0})["risk_llm_source"] == "fixed"
    RiskCalibrator().to_yaml(str(tmp_path / "c.yaml"), {})
    root["envelope"]["calibration_path"] = str(tmp_path / "c.yaml")
    calibrated = RiskEnvelope(root, FUZZY, None)
    d = calibrated.decide(None, "high", distribution={"medium": 0.5, "high": 0.5})
    assert d["risk_llm_source"] == "calibrated" and 0.4 < d["risk_llm"] < FIXED["high"]
    assert calibrated.decide(None, "high")["risk_llm_source"] == "fixed"           # no distribution → fixed


def test_worker_sends_the_floor_and_uses_the_distribution(tmp_path):
    from modules.aggregator.outbox import Outbox
    from modules.envelope import RiskEnvelope
    from modules.llm.worker import LLMWorker
    from tests.test_envelope import SMOKE, T0, person
    from tests.test_llm import FakeGPT

    gpt = FakeGPT({**BASE, "risk_level": "low", "risk_distribution": {"low": 0.7, "medium": 0.3}})
    box = Outbox(str(tmp_path / "outbox.db"))
    envelope = RiskEnvelope.from_config(FUZZY, None)
    worker = LLMWorker(queue.Queue(), queue.Queue(), {"llm": {"poll_sec": 0}}, gpt=gpt, outbox=box,
                       envelope=envelope, wall_clock=lambda: T0 + 60)
    for event in [*person(known=False), SMOKE]:
        worker.handle_control(event.to_json())
    view = {"v": 2, "window": "x", "trigger": "critical_event", "trusted_evidence": []}
    box.add({"snapshot_id": "s1", "trigger": "critical_event", "location_id": "warehouse"}, "critical_event",
            created_at=1, digest=payload_text(view))
    worker.step()
    sent = json.loads(gpt.digests[0])
    assert sent["risk_floor"]["term"] in ("high", "medium")
    decision = json.loads(worker.event_queue.get())
    assert decision["payload"]["risk_distribution"] == {"low": 0.7, "medium": 0.3, "high": 0.0, "critical": 0.0}


def test_token_report_tool(tmp_path, capsys):
    from tools.snapshot_tokens import main

    clock = [1000.0]
    config = {"aggregator": {"export_dir": str(tmp_path / "exports"), "incidents": {"enabled": False},
                             "snapshot_interval_sec": 5}, "trust": {"fuzzy_config": FUZZY}}
    scene(Aggregator(config, clock=lambda: clock[0]), clock)
    assert main([str(tmp_path / "exports" / "snapshots" / "*.json"), "--fuzzy", FUZZY]) == 0
    out = capsys.readouterr().out
    assert "v1_json" in out and "v1_digest" in out and "v2_view" in out
