"""F29 (TOD-40): experiment v2 — variant flags, the sensor-failure block, corridor metrics, the v2 report."""
from pathlib import Path

import pytest

from core.events import Event
from core.fuzzy import load_yaml
from experiments.configs import BY_NAME, FAILURE, V2
from experiments.metrics import corridor_stats, failure_stats, review_correct
from experiments.pipeline import Decision, ScenarioResult
from experiments.variants import STATIC_PHYSICAL, StaticTrustRegistry, gas_failure_lines, with_gas_failure

REPO = Path(__file__).resolve().parent.parent
FUZZY = str(REPO / "config" / "fuzzy.yaml")


def result(name, category, expert, final, decisions, duration=3600.0, sid="s1"):
    r = ScenarioResult(sid, name, category, None, expert, 0.5, final_level=final)
    r.decisions, r.duration_s = decisions, duration
    return r


# --- configurations ---------------------------------------------------------------------------------------------

def test_v2_suite_flags():
    names = [c.name for c in V2]
    assert names[:4] == ["rules_only", "llm_only", "llm_pattern", "proposed_v2"]
    ref = BY_NAME["proposed_v2"]
    assert ref.flags() == {**BY_NAME["proposed"].flags()}
    for name, flag in (("proposed_v2_type1", "type1"), ("proposed_v2_static_trust", "static_trust"),
                       ("proposed_v2_floor_only", "floor_only"), ("proposed_v2_local", "local")):
        cfg = BY_NAME[name]
        assert getattr(cfg, flag) and {k: v for k, v in cfg.flags().items() if k not in (flag, "gating")} == \
            {k: v for k, v in ref.flags().items() if k not in (flag, "gating")}
    assert not BY_NAME["proposed_v2_no_B"].trust and not BY_NAME["proposed_v2_no_C"].view
    assert not BY_NAME["proposed_v2_no_E"].gating and FAILURE == ["proposed_v2", "proposed_v2_static_trust"]


def test_floor_only_policy_is_the_max_projection():
    from modules.envelope import RiskEnvelope

    envelope = RiskEnvelope.from_config(FUZZY, None)
    two_sided = envelope.decide("lab", "low", now=1.7e9, local_hour=12)
    envelope.policy = "floor_only"
    floor = envelope.decide("lab", "low", now=1.7e9, local_hour=12)
    assert floor["policy_branch"] in ("floor_kept", "floor_raised") and floor["h"] is None and floor["U"] == 1.0
    assert floor["risk_final"] == pytest.approx(max(floor["risk_llm"], floor["L"]), abs=1e-3)
    assert two_sided["policy_branch"] not in ("floor_kept", "floor_raised")
    root = load_yaml(FUZZY)
    with pytest.raises(ValueError, match="envelope.policy"):
        RiskEnvelope({**root, "envelope": {**root["envelope"], "policy": "one_sided"}}, FUZZY, None)


def test_static_trust_registry_never_moves():
    from modules.trust import TrustScorer
    from experiments.variants import static_registry

    scorer = TrustScorer.from_config(FUZZY, registry_factory=static_registry)
    registry = scorer.registry
    assert isinstance(registry, StaticTrustRegistry)
    registry.ensure_device("gas-1", "gas", "lab", 0.0)
    cid = "gas-1/gas_concentration"
    assert registry.tau(cid, 0.0) == STATIC_PHYSICAL and registry.uncertainty(cid, 0.0) == 0.0
    registry.refute(cid, 5, "test", 10.0)
    assert registry.set_health("gas-1", {"baseline_drift": {"factor": 0.3}}, 10.0) == []
    assert registry.tau(cid, 20.0) == STATIC_PHYSICAL and registry.get(cid, 20.0)["h"] == 1.0
    registry.ensure_device("cam-1", "camera", "lab", 0.0)
    assert registry.tau("cam-1/scene_text", 0.0) == 0.1


# --- sensor-failure block ----------------------------------------------------------------------------------------

def scenario_lines():
    from experiments.dataset import Dataset
    from experiments.scripted import build
    import tempfile

    tmp = tempfile.mkdtemp()
    build(tmp, 42)
    return next(s for s in Dataset.load(tmp).scenarios if s.category == "dangerous")


def test_failure_block_drifts_readings_and_reports_health():
    scenario = scenario_lines()
    healthy = [Event.from_json(line) for line in scenario.events()]
    failed = [Event.from_json(line) for line in with_gas_failure(scenario).events()]
    assert len(failed) == len(healthy) + 1 and with_gas_failure(scenario).scenario_id == scenario.scenario_id
    (health,) = [e for e in failed if e.type == "device_health"]
    assert health.payload["failing"]["baseline_drift"]["factor"] == 0.3 and health.device_id == "gas-1"
    readings = [(a.payload["concentration_pct"], b.payload["concentration_pct"])
                for a, b in zip([e for e in healthy if e.type == "gas_reading"],
                                [e for e in failed if e.type == "gas_reading"], strict=True)]
    assert all(abs(f - h - 1.5) < 1e-6 for h, f in readings)
    assert gas_failure_lines(scenario.events()) == with_gas_failure(scenario).events()      # deterministic


def test_failure_lowers_registry_trust_but_not_static_trust():
    from modules.trust import TrustScorer
    from experiments.variants import static_registry

    scenario = scenario_lines()
    out = {}
    for name, factory in (("registry", None), ("static", static_registry)):
        for variant, lines in (("healthy", scenario.events()), ("failed", with_gas_failure(scenario).events())):
            scorer = TrustScorer.from_config(FUZZY, registry_factory=factory)
            trusts = []
            for line in lines:
                event = Event.from_json(line)
                scorer.apply(event)
                if event.type == "gas_reading" and "trust" in event.payload:
                    trusts.append((event.payload["trust"], event.payload.get("trust_u")))
            out[(name, variant)] = trusts[-1]
    assert out[("registry", "failed")][0] < 0.5 * out[("registry", "healthy")][0]
    assert out[("registry", "failed")][1] > out[("registry", "healthy")][1]          # u_eff = 1 − h(1 − u)
    assert out[("static", "failed")][0] == pytest.approx(out[("static", "healthy")][0])


# --- metrics -----------------------------------------------------------------------------------------------------

def test_corridor_stats_on_synthetic_decisions():
    ds = [Decision(0, "interval", "llm", "high", 0.7, "high", width=0.1, u=0.1, h=0.9, branch="agree", truth="high"),
          Decision(1, "interval", "llm", "high", 0.7, "low", width=0.3, u=0.3, h=0.0,
                   branch="deescalation_rejected", truth="high"),
          Decision(2, "interval", "llm", "high", 0.7, "critical", width=0.5, u=0.5, h=0.1,
                   branch="escalation_review", truth="high"),
          Decision(3, "interval", "llm", "high", 0.7, "high", width=0.6, u=0.6, h=0.2,
                   branch="escalation_review", truth="high"),
          Decision(4, "interval", "fuzzy_only", "low", 0.1, width=0.0, u=0.0, branch="fuzzy_only", truth="low")]
    stats = corridor_stats([result("x", "dangerous", "high", "high", ds, duration=1800)])
    assert stats["mean_width"] == pytest.approx(0.3) and stats["width_u_corr"] == pytest.approx(1.0)
    assert stats["h"]["n"] == 4 and stats["h"]["median"] == 0.2 and sum(stats["h"]["histogram"]) == 4
    assert stats["branches"] == {"agree": 0.2, "deescalation_rejected": 0.2, "escalation_review": 0.4,
                                 "fuzzy_only": 0.2}
    assert stats["reviews"] == 2 and stats["reviews_per_hour"] == 4.0 and stats["reviews_correct"] == 0.5
    assert review_correct(ds[3]) and not review_correct(ds[2])


def test_failure_stats():
    rs = [result("x", "adversarial", "high", "medium", [], sid="a"), result("x", "adversarial", "high", "high", [],
                                                                             sid="b"),
          result("x", "normal", "low", "medium", [], sid="c"), result("x", "dangerous", "critical", "high", [],
                                                                       sid="d")]
    stats = failure_stats(rs)
    assert stats["asr"] == 0.5 and stats["false_alarm_rate"] == 1.0
    assert stats["dangerous_correct"] == 0.0 and stats["dangerous_under"] == 1.0


def test_questions_table_answers_every_question():
    from experiments.report import QUESTIONS, questions_table

    base = {"macro_f1": 1.0, "asr": 0.0, "ece": 0.03, "llm_calls_per_hour": 17.0, "false_alarm_rate": 0.0}
    summaries = {"proposed_v2": base, **{name: {**base, "asr": 0.1} for _, name in QUESTIONS}}
    tests = {name: {"a_only_correct": 3, "b_only_correct": 0, "p": 0.25, "n": 84} for _, name in QUESTIONS}
    corridor = {name: {"mean_width": 0.1, "width_u_corr": 0.8, "reviews_per_hour": 1.0, "reviews_correct": 0.5}
                for name in summaries}
    failure = {"proposed_v2": {"asr": 0.0, "false_alarm_rate": 0.1}, "proposed_v2_static_trust": {
        "asr": 0.0, "false_alarm_rate": 0.4}}
    rows = questions_table(summaries, tests, corridor, failure)
    assert len(rows) == 2 + len(QUESTIONS)
    assert all("proposed_v2 better on 3 vs 0 scenarios (not significant, p = 0.25)" in r for r in rows[2:])
    assert "sensor failure: ASR 0.0 vs 0.0, false alarms 0.1 vs 0.4" in rows[3]
