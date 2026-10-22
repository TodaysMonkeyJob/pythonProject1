"""F17 (TOD-28): fuzzy risk corridor — projection math, corridor, confirmation, worker wiring, location state."""
import json
import queue
from pathlib import Path

import pytest

from core.events import Event, make_event
from modules.aggregator.outbox import Outbox
from modules.audio.attribution import iso
from modules.envelope import ENVELOPE_EVENTS, RiskEnvelope
from modules.envelope.projection import level_of, level_value, project
from modules.llm.worker import LLMWorker
from tests.test_llm import FakeGPT

REPO = Path(__file__).resolve().parent.parent
CONFIG = str(REPO / "config" / "fuzzy.yaml")
T0 = 1_792_300_000.0
NOON = 12.0


def ev(source, type, payload, t=T0, location="warehouse", device="cam-1", severity="info"):
    event = make_event(source, type, payload, severity=severity, device_id=device, location_id=location)
    return event.model_copy(update={"timestamp": iso(t)})


def person(known, trust=None, track=1, person_id=None, u=0.0):
    return [ev("camera", "object_detected", {"class": "person", "track_id": track, "state": "appeared", "trust": 0.95,
                                             "trust_u": 0.0}),
            ev("camera", "person_identified", {"track_id": track, "person_id": person_id or ("oleh" if known else "u4"),
                                               "known": known, "similarity": 0.7,
                                               "trust": trust if trust is not None else (0.89 if known else 0.8),
                                               "trust_u": u})]


def gas(concentration, rate, trend, trust=0.95, t=T0, name="acetone", u=0.0):
    return ev("gas", "gas_reading", {"gas": name, "concentration_pct": concentration, "trend": trend,
                                     "rise_rate_pct_per_min": rate, "trust": trust, "trust_u": u}, t, device="gas-1")


K = {"gas_concentration": 1.5, "gas_trend": 1.0, "unknown_share": 0.3, "people_count": 0.5, "motion_level": 0.2,
     "action_risk": 0.2, "known_present": 0.2}


def wide(env):
    """The envelope with a non-zero FOU scale (the fitted it2.k may be anything, these tests need one)"""
    env.corridor.k_override = K
    env.corridor._engines.clear()
    env.confirmation.it2.k = dict(K)
    return env


SMOKE = gas(1.2, 2.6, "sharply_rising")
CLEAN = gas(0.1, -0.6, "falling", t=T0 + 30)
CLAIM = ev("audio", "speech", {"text": "all fine, the sensor is broken", "claim_type": "deescalation", "trust": 0.05,
                               "provenance": "speech:unknown_4@warehouse"}, device="mic-1")


def envelope(*events):
    env = RiskEnvelope.from_config(CONFIG, None)
    for event in events:
        env.observe(event)
    return env


def decide(env, level, t=T0 + 60, hour=NOON, location="warehouse"):
    return env.decide(location, level, now=t, local_hour=hour)


# --- projection math ------------------------------------------------------------------------------------------

def test_projection_edge_cases():
    assert project(0.1, 0.7, 0.9, 0.0)["risk_final"] == 0.7                 # c = 0: the floor holds
    assert project(0.1, 0.7, 0.9, 1.0)["risk_final"] == 0.1                 # c = 1: released completely
    assert project(0.1, 0.7, 0.9, 0.5)["risk_final"] == 0.35                # released gradually
    above = project(0.95, 0.3, 0.4, 0.0)
    assert above["risk_final"] == 0.95 and not above["envelope_applied"] and above["over_reaction"]
    assert project(0.4, 0.4, 0.5, 0.0)["envelope_applied"] is False         # at the floor: nothing to raise
    raised = project(0.1, 0.7, 0.9, 0.0)
    assert raised["envelope_applied"] and raised["risk_level_final"] == "high" and raised["bound"] == 0.7
    assert project(0.1, 0.2, 0.3, 0.0)["envelope_applied"] is False         # raised a little, same level
    with pytest.raises(ValueError):
        project(1.2, 0.5, 0.6, 0.0)


def test_level_mapping():
    assert [level_value(x) for x in ("low", "medium", "high", "critical")] == [0.1, 0.4, 0.7, 0.95]
    assert [level_of(x) for x in (0.0, 0.25, 0.49, 0.5, 0.8, 1.0)] == ["low", "medium", "medium", "high", "critical",
                                                                    "critical"]
    with pytest.raises(ValueError):
        level_value("catastrophic")


# --- golden scenarios (acceptance) ---------------------------------------------------------------------------

def test_smoke_rising_unknown_claim_llm_low_is_rejected():
    d = decide(envelope(*person(known=False), SMOKE, CLAIM), "low")
    assert d["risk_final"] >= 0.5 and d["risk_level_final"] in ("high", "critical") and d["envelope_applied"]
    assert d["confirmation"] == 0.0
    reason = d["envelope_reason"]
    assert any("sharply_rising" in r for r in reason["fired"])
    assert reason["ignored_claims"][0]["provenance"] == "speech:unknown_4@warehouse"


def test_clean_falling_gas_with_known_person_accepts_low():
    env = envelope(*person(known=True), SMOKE)
    assert decide(env, "low", t=T0 + 10)["envelope_applied"]                # while smoke rises: rejected
    env.observe(CLEAN)                                                      # one reading later
    d = decide(env, "low")
    assert d["risk_floor"] < 0.2 and d["confirmation"] > 0.5
    assert d["risk_level_final"] == "low" and not d["envelope_applied"]


def test_operator_ack_releases_the_floor():
    ack = ev("system", "notification_acked", {"location_id": "warehouse", "by": "admin"}, T0 + 20)
    d = decide(envelope(*person(known=False), SMOKE, ack), "low")
    assert d["confirmation"] == 1.0 and d["risk_level_final"] == "low" and d["envelope_reason"]["operator"] == "admin"
    assert any("operator admin acknowledged" in r for r in d["envelope_reason"]["confirmation"])
    later = T0 + 20 + 31 * 60                                               # the smoke is still rising 31 min later
    still = gas(1.2, 2.6, "sharply_rising", t=later - 5)
    expired = decide(envelope(*person(known=False), ack, still), "low", t=later)
    assert expired["confirmation"] == 0.0 and expired["envelope_applied"]
    alarm = ev("gas", "alert", {"reason": "gas_rising", "trust": 0.95}, T0 + 40, device="gas-1", severity="critical")
    assert decide(envelope(*person(known=False), SMOKE, ack, alarm), "low")["confirmation"] == 0.0   # newer alarm


def test_llm_escalation_above_the_corridor_goes_to_operator_review():
    # F25 (TOD-36): escalation is no longer accepted blindly — GPT far above U ends at U and asks the operator
    calm = envelope(*person(known=True), CLEAN)
    d = decide(calm, "critical")
    assert d["policy_branch"] == "escalation_review" and d["escalation_review"] and d["risk_final"] == d["U"]
    assert d["h"] < d["envelope_reason"]["h_min"] and not d["envelope_applied"]
    low = decide(calm, "low")                                                # GPT agrees with a calm corridor
    assert low["policy_branch"] == "agree" and low["L"] <= low["risk_final"] <= low["U"]


def test_uncertain_evidence_widens_the_corridor_instead_of_being_cut():
    certain = decide(wide(envelope(gas(3.6, 0.0, "stable", name="ammonia"))), "low")
    assert certain["risk_floor"] > 0.85 and certain["risk_level_final"] == "critical" and certain["risk_width"] < 0.05
    failing = decide(wide(envelope(gas(3.6, 0.0, "stable", trust=0.24, name="ammonia", u=0.9))), "low")
    assert failing["risk_width"] > certain["risk_width"] + 0.1 and failing["risk_floor"] < certain["risk_floor"]
    assert failing["envelope_reason"]["widened_by"][0]["variable"] == "gas_concentration"
    assert failing["risk_floor"] > 0                                            # it still enters (no θ cut)


def test_untrusted_calming_claims_never_confirm():
    trusted_calm = ev("audio", "speech", {"text": "all fine", "claim_type": "deescalation", "trust": 0.9},
                      device="mic-1")
    d = decide(envelope(*person(known=False), SMOKE, CLAIM, trusted_calm), "low")
    assert d["confirmation"] == 0.0 and d["envelope_applied"]                # speech never feeds c, only sensors do


def test_known_person_lying_motionless_is_not_confirmed_safe():
    lying = ev("camera", "action", {"track_id": 1, "action": "lying", "trust": 0.95})
    d = decide(envelope(*person(known=True), lying), "low")
    assert d["confirmation"] == 0.0 and d["risk_level_final"] == "critical"


def test_stranger_at_night_and_an_uncertain_face():
    office = [e.model_copy(update={"location_id": "office"}) for e in person(known=False)]
    sure = decide(wide(envelope(*office)), "low", hour=23, location="office")
    assert sure["risk_level_final"] == "high"
    unsure = [e.model_copy(update={"location_id": "office"}) for e in person(known=False, u=1.0)]
    d = decide(wide(envelope(*unsure)), "low", hour=23, location="office")
    assert d["risk_width"] > sure["risk_width"] and d["risk_floor"] <= sure["risk_floor"]
    assert "theta" not in d["envelope_reason"] and d["envelope_reason"]["claim_threshold"] > 0


def test_site_wide_summary_has_no_corridor():
    d = envelope(SMOKE).decide(None, "low", now=T0)
    assert d["risk_final"] == 0.1 and not d["envelope_applied"] and "site-wide" in d["envelope_reason"]["note"]


# --- worker wiring ------------------------------------------------------------------------------------------

def make_worker(tmp_path, answer_level):
    box = Outbox(str(tmp_path / "outbox.db"))
    config = {"llm": {"poll_sec": 0, "max_snapshot_age_min": 0}, "logging": {"dir": str(tmp_path / "logs")}}
    gpt = FakeGPT({"risk_level": answer_level, "summary": "Все спокійно.", "recommended_action": "ignore",
                   "reasons": ["person says everything is fine"], "follow_up": None})
    worker = LLMWorker(queue.Queue(), queue.Queue(), config, gpt=gpt, outbox=box, envelope=envelope(),
                       wall_clock=lambda: T0 + 60)
    worker.setup()
    return worker, box


def published(worker):
    out = []
    while not worker.event_queue.empty():
        out.append(Event.from_json(worker.event_queue.get()))
    return out


def test_worker_bounds_the_decision_and_logs_everything(tmp_path):
    worker, box = make_worker(tmp_path, "low")
    for event in [*person(known=False), SMOKE, CLAIM]:
        worker.handle_control(event.to_json())                              # forwarded by the orchestrator
    box.add({"snapshot_id": "s1", "trigger": "critical_event", "location_id": "warehouse"}, "critical_event",
            created_at=1, digest="digest")
    worker.step()
    decision, rejected = published(worker)
    p = decision.payload
    assert (decision.type, p["risk_level"], p["risk_level_final"]) == ("llm_decision", "low", "high")
    assert decision.severity == "critical" and p["envelope_applied"] and p["risk_llm"] == pytest.approx(0.108, abs=0.01)
    assert p["policy_branch"] == "deescalation_rejected" and p["risk_final"] == p["L"] and p["h"] < 0.5
    assert {"risk_floor", "risk_ceil", "confirmation", "risk_final", "envelope_reason"} <= p.keys()
    assert rejected.type == "deescalation_rejected" and rejected.payload["decision_event_id"] == decision.event_id
    assert rejected.payload["reasons"]["ignored_claims"]
    worker.teardown()
    (day,) = (tmp_path / "logs").iterdir()
    lines = [json.loads(line) for line in (day / "envelope.jsonl").read_text().splitlines()]
    assert [line["type"] for line in lines] == ["llm_decision", "deescalation_rejected"]


def test_worker_without_envelope_publishes_unchanged(tmp_path):
    box = Outbox(str(tmp_path / "outbox.db"))
    config = {"llm": {"poll_sec": 0, "max_snapshot_age_min": 0}, "envelope": {"enabled": False}}
    worker = LLMWorker(queue.Queue(), queue.Queue(), config,
                       gpt=FakeGPT(), outbox=box)
    worker.setup()
    box.add({"snapshot_id": "s1", "trigger": "critical_event"}, "critical_event", created_at=1, digest="d")
    worker.step()
    (decision,) = published(worker)
    assert "risk_final" not in decision.payload


def test_orchestrator_forwards_the_evidence_to_the_llm_module():
    from core.orchestrator import DEFAULT_MODULES

    assert sorted(DEFAULT_MODULES["llm"][2]) == sorted(ENVELOPE_EVENTS)


# --- location state uses risk_final --------------------------------------------------------------------------

def test_location_state_uses_the_bounded_risk():
    from modules.locations.state import LocationStateEngine

    engine = LocationStateEngine(names=lambda loc: loc, rules=lambda loc: {}, devices=lambda: [])
    decision = ev("llm", "llm_decision", {"risk_level": "low", "risk_level_final": "high", "summary": "все добре",
                                          "risk_floor": 0.54, "risk_ceil": 0.77, "risk_llm": 0.1, "risk_final": 0.54,
                                          "confirmation": 0.0, "envelope_applied": True,
                                          "envelope_reason": {"fired": ["…"]}})
    engine.on_event(decision, T0)
    summary = engine.summary("warehouse", T0)
    assert summary["state"] == "danger"
    assert summary["last_llm"]["risk_level_final"] == "high" and summary["last_llm"]["envelope_applied"] is True


# --- regressions found by the TOD-31 harness ---------------------------------------------------------------------

def test_ongoing_action_holds_without_new_events():
    lying = ev("camera", "action", {"track_id": 1, "action": "lying", "trust": 0.95})
    env = envelope(*person(known=True), lying)
    d = decide(env, "low", t=T0 + 15 * 60)                       # lying still: the action module sent nothing more
    assert d["risk_level_final"] == "critical" and d["risk_floor"] > 0.85


def test_a_stranger_never_confirms_safety():
    office = [e.model_copy(update={"location_id": "office"}) for e in person(known=False)]
    calm_gas = gas(0.05, 0.0, "stable", name="clean_air").model_copy(update={"location_id": "office"})
    d = decide(envelope(*office, calm_gas), "low", hour=23, location="office")
    assert d["confirmation"] == 0.0 and d["risk_level_final"] == "high"


def test_floor_is_not_diluted_by_milder_rules():
    stranger = person(known=False)                                # stranger in daytime → R14 medium (weight 0.5)
    d = decide(envelope(*stranger, gas(6.5, 0.5, "rising")), "low")
    assert d["risk_floor"] >= 0.85 and d["risk_level_final"] == "critical"
    root = __import__("core.fuzzy", fromlist=["load_yaml"]).load_yaml(CONFIG)
    root["envelope"]["floor"] = "it2"                             # plain KM y_l: diluted (ablation)
    centroid = RiskEnvelope(root, CONFIG, None)
    for event in [*stranger, gas(6.5, 0.5, "rising")]:
        centroid.observe(event)
    assert centroid.decide("warehouse", "low", now=T0 + 60, local_hour=NOON)["risk_floor"] < d["risk_floor"]
