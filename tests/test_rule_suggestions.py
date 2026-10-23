"""F27 (TOD-38): GPT rule suggestions — triggers, strict answers, validation, dedup, isolation, review, approval."""
import json
import queue
import time
from types import SimpleNamespace

import pytest
import yaml

from core.events import make_event
from modules.rules import review
from modules.rules.suggester import (RuleSuggester, SuggestionTriggers, manual_job, parse_suggestion,
                                     validate_rule)
from modules.rules.suggestions import RuleSuggestionStore
from tests.test_rules_governance import CONFIG, config_copy, sim_llm, toy_dataset  # noqa: F401

T0 = 1.7e9
GOOD_RULE = "IF action_risk IS high AND time_of_day_risk IS night AND people_count IS crowd THEN risk IS critical"


def ev(type, payload, location="lab", device="llm"):
    return make_event("system", type, payload, device_id=device, location_id=location)


def answer(rule=GOOD_RULE, **extra):
    return json.dumps({"rule": rule, "location": "lab", "rationale": "Натовп із ризиковими діями вночі.",
                       "supporting_cases": [], "new_variable_needed": None, "confidence": "medium", **extra},
                      ensure_ascii=False)


class FakeGPT:
    """The ``client.chat.completions.create`` / ``deployment`` part of GPTClient"""
    deployment = "gpt-test"

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.answers.pop(0)
        if isinstance(content, Exception):
            raise content
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def triggers(store, **cfg):
    return SuggestionTriggers(store, {"repeat_n": 3, "repeat_window_h": 24, **cfg}, str(CONFIG))


# --- triggers -----------------------------------------------------------------------------------------------------

def test_accepted_escalation_queues_a_job_rejected_does_not():
    store = RuleSuggestionStore()
    hooks = triggers(store)
    hooks.record_outcome("r1", False, T0, "olena", "lab", "critical", "medium", "false alarm", {"reasons": ["x"]})
    assert store.pending_jobs() == []
    case = hooks.record_outcome("r2", True, T0, "olena", "lab", "critical", "medium", None, {"reasons": ["smoke"]})
    jobs = store.pending_jobs()
    assert [(j["trigger"], j["location_id"], j["case_ids"]) for j in jobs] == [("accepted_escalation", "lab", [case])]


def test_repeated_escalations_queue_one_job_per_window():
    store = RuleSuggestionStore()
    hooks = triggers(store)
    review_event = lambda i: ev("escalation_review", {"review_id": f"r{i}", "risk_level_llm": "high",  # noqa: E731
                                                      "risk_level_final": "medium", "summary": "s", "U": 0.5})
    for i in range(5):
        hooks.observe(review_event(i), T0 + i * 60)
    jobs = store.pending_jobs()
    assert len(jobs) == 1 and jobs[0]["trigger"] == "repeated_escalation" and len(jobs[0]["case_ids"]) == 3
    assert {c["kind"] for c in store.outcomes("lab")} == {"repeated_escalation"}
    hooks.observe(review_event(9), T0 + 25 * 3600)                   # a new window, but only one review in it
    assert len(store.pending_jobs()) == 1


def test_confirmed_alarm_without_a_fired_risk_rule_is_a_coverage_gap():
    store = RuleSuggestionStore()
    hooks = triggers(store)
    confirm = ev("trust_feedback", {"channel_id": "cam-1/smoking", "verdict": "confirm", "by": "olena",
                                    "location_id": "lab"}, device="dash")
    hooks.observe(ev("llm_decision", {"fired_ids": ["R1"], "location_id": "lab"}), T0)      # R1 → critical
    hooks.observe(confirm, T0 + 1)
    assert store.pending_jobs() == []
    hooks.observe(ev("llm_decision", {"fired_ids": ["R19"], "location_id": "lab"}), T0 + 2)  # R19 → low only
    hooks.observe(confirm, T0 + 3)
    hooks.observe(ev("trust_feedback", {"channel_id": "llm/risk_assessment", "verdict": "confirm",
                                        "location_id": "lab"}, device="dash"), T0 + 4)       # not a physical channel
    jobs = store.pending_jobs()
    assert [j["trigger"] for j in jobs] == ["coverage_gap"]
    assert store.outcomes(ids=jobs[0]["case_ids"])[0]["context"]["fired_rules"] == ["R19"]


def test_manual_job_collects_cases_of_the_period():
    store = RuleSuggestionStore()
    now = time.time()
    store.record_outcome("old", True, now - 30 * 86400, location_id="lab")
    store.record_outcome("rejected", False, now - 60, location_id="lab")
    recent = store.record_outcome("new", True, now - 60, location_id="lab")
    job = manual_job(store, 7 * 86400)
    assert store.pending_jobs()[0]["id"] == job and store.pending_jobs()[0]["case_ids"] == [recent]
    assert manual_job(store, 7 * 86400, "hall") is None


# --- answers, validation, dedup -------------------------------------------------------------------------------

def test_strict_answer_schema():
    assert parse_suggestion(answer()).confidence == "medium"
    for bad in ("not json", "[1, 2]", answer(confidence="sure"), answer(supporting_cases="all")):
        with pytest.raises(ValueError):
            parse_suggestion(bad)


def test_suggester_stores_validated_answers_and_dedups():
    store = RuleSuggestionStore()
    case = store.record_outcome("r1", True, T0, location_id="lab")
    gpt = FakeGPT(answer(), answer(rule=GOOD_RULE.replace("  ", " ").replace("IS crowd", "IS  crowd")),
                  answer(rule="IF speech_intent IS threat THEN risk IS high"), "garbage")
    suggester = RuleSuggester(store, gpt, str(CONFIG))
    for _ in range(4):
        store.add_job("accepted_escalation", "lab", [case], T0)
    assert suggester.process_pending(10) == 4
    prompt = gpt.calls[0]["messages"][0]["content"]
    assert "gas_concentration" in prompt and "{grammar}" not in prompt and "THEN risk IS" in prompt
    assert json.loads(gpt.calls[0]["messages"][1]["content"])["cases"][0]["case_id"] == case
    good, invalid = store.list()
    assert good["support_count"] == 2 and good["validator_report"]["valid"] and good["supporting_cases"] == [case]
    assert not invalid["validator_report"]["valid"] and "speech_intent" in invalid["validator_report"]["errors"][0]
    assert invalid["rule"] == "IF speech_intent IS threat THEN risk IS high"            # stored as is, never fixed
    statuses = [r[0] for r in store.db.execute("SELECT status FROM jobs ORDER BY id")]
    assert statuses == ["done", "done", "done", "failed"]


def test_validator_catches_duplicates_and_needs_a_rule():
    assert validate_rule(GOOD_RULE, "*", str(CONFIG))["valid"]
    assert validate_rule(GOOD_RULE, "warehouse", str(CONFIG))["valid"]
    assert "duplicates R1" in validate_rule("IF gas_concentration IS dangerous THEN risk IS critical", "*",
                                            str(CONFIG))["errors"][0]
    assert "new variable" in validate_rule("", "*", str(CONFIG), "speech_intent = threat")["errors"][0]


# --- isolation: no events, no JSONL logs --------------------------------------------------------------------------

def test_worker_runs_jobs_when_idle_without_events_or_logs(tmp_path):
    from modules.aggregator.outbox import Outbox
    from modules.llm.worker import LLMWorker

    store = RuleSuggestionStore(str(tmp_path / "suggestions.db"))
    store.add_job("accepted_escalation", "lab", [store.record_outcome("r1", True, T0, location_id="lab")], T0)
    events, gpt = queue.Queue(), FakeGPT(answer())
    config = {"llm": {"poll_sec": 0, "max_snapshot_age_min": 0}, "logging": {"dir": str(tmp_path / "logs")},
              "rules": {"suggestions": {"every_sec": 60}}}
    worker = LLMWorker(events, queue.Queue(), config, gpt=gpt, outbox=Outbox(str(tmp_path / "outbox.db")),
                       clock=lambda: 1000.0)
    worker.suggester = RuleSuggester(store, gpt, str(CONFIG))
    worker.step()
    worker.step()                                                       # within every_sec: no second call
    assert len(gpt.calls) == 1 and store.list()[0]["rule"] == GOOD_RULE
    assert events.empty() and not (tmp_path / "logs").exists()


# --- review ---------------------------------------------------------------------------------------------------

def stored(store, rule=GOOD_RULE, location="lab", fuzzy=CONFIG):
    report = validate_rule(rule, location, str(fuzzy))
    return store.add_suggestion({"rule": rule, "location": location, "rationale": "Натовп уночі.",
                                 "supporting_cases": [1], "confidence": "high"}, report, "manual")["id"]


def test_invalid_suggestion_cannot_be_sent_until_edited(tmp_path):
    store = RuleSuggestionStore()
    bad = stored(store, "IF speech_intent IS threat THEN risk IS high")
    with pytest.raises(review.ReviewError, match="edit it first"):
        review.send_to_approval(store, bad, "olena", str(CONFIG), str(tmp_path / "rules"))
    with pytest.raises(review.ReviewError, match="--by"):
        review.edit(store, bad, GOOD_RULE, "")
    item = review.edit(store, bad, GOOD_RULE, "olena", str(CONFIG))
    assert item["status"] == "edited" and item["validator_report"]["valid"]
    assert item["original_rule"] == "IF speech_intent IS threat THEN risk IS high"
    rejected = review.reject(store, stored(store, GOOD_RULE.replace("crowd", "several")), "olena", "too broad")
    assert rejected["status"] == "rejected" and rejected["review_reason"] == "too broad"
    with pytest.raises(review.ReviewError, match="rejected"):
        review.send_to_approval(store, rejected["id"], "olena", str(CONFIG), str(tmp_path / "rules"))


def test_candidate_keeps_comments_and_places_rules(tmp_path):
    path = config_copy(tmp_path)
    store = RuleSuggestionStore()
    for location in ("*", "warehouse", "lab"):
        item = store.get(stored(store, GOOD_RULE, location, path))
        text = review.candidate_text(item, "olena", str(path))
        assert "# Per-location overrides" in text and text.count("\n#") == path.read_text().count("\n#")
        data = yaml.safe_load(text)
        rules = data["rules"] if location == "*" else data["locations"][location]["rules"]
        new = [r for r in rules if r["id"] == f"GS{item['id']}"]
        assert new and new[0]["meta"]["source"] == f"suggestion {item['id']}" and new[0]["meta"]["author"] == "olena"
    assert [r["id"] for r in data["locations"]["warehouse"]["rules"]] == ["W1", "W2"]


def test_send_to_approval_creates_a_proposal_without_changing_the_rule_base(tmp_path, toy_dataset):  # noqa: F811
    from tools.rules import RuleStore

    path = config_copy(tmp_path)
    before = path.read_text()
    store = RuleSuggestionStore()
    sid = stored(store, GOOD_RULE, "*", path)
    rules_dir = tmp_path / "rules"
    proposal = review.send_to_approval(store, sid, "olena", str(path), str(rules_dir), toy_dataset, llm=sim_llm())
    assert path.read_text() == before
    item = store.get(sid)
    assert proposal["passed"], proposal.get("gate", {}).get("reasons")
    assert item["status"] == "sent_to_approval" and item["proposal_id"] == proposal["proposal"]
    assert item["gate_report"]["passed"] and (rules_dir / "proposals" / proposal["proposal"] / "gate.md").is_file()
    candidate = yaml.safe_load((rules_dir / "proposals" / proposal["proposal"] / "candidate.yaml").read_text())
    assert any(r["id"] == f"GS{sid}" for r in candidate["rules"])
    RuleStore(str(rules_dir), str(path)).approve(proposal["proposal"], "Supervisor", "supervisor", "covers crowds")
    assert review.mark_approved(store, str(rules_dir)) == 1 and store.get(sid)["status"] == "approved"


# --- dashboard ------------------------------------------------------------------------------------------------

def test_dashboard_review_endpoints(tmp_path, monkeypatch):
    from modules.rules import suggester
    from tests.test_dashboard import auth_header, client

    store = RuleSuggestionStore()
    monkeypatch.setattr(suggester, "store_from_config", lambda config: store)      # the page's configured store
    app, state = client(tmp_path)
    good = stored(store)
    bad = stored(store, "IF speech_intent IS threat THEN risk IS high")
    headers = auth_header()
    assert app.get("/api/suggestions").status_code == 401
    items = app.get("/api/suggestions", headers=headers).json()["suggestions"]
    assert {i["id"] for i in items} == {good, bad}
    assert app.get("/api/suggestions?confidence=low", headers=headers).json()["suggestions"] == []
    detail = app.get(f"/api/suggestions/{bad}", headers=headers).json()
    assert detail["validator_report"]["valid"] is False and detail["sending"] is False
    live = app.post(f"/api/suggestions/{bad}/validate", json={"rule": GOOD_RULE}, headers=headers).json()
    assert live["valid"] and not store.get(bad)["validator_report"]["valid"]           # validate saves nothing
    assert app.post(f"/api/suggestions/{bad}/send", headers=headers).status_code == 409
    edited = app.post(f"/api/suggestions/{bad}/edit", json={"rule": GOOD_RULE.replace("crowd", "several")},
                      headers=headers).json()
    assert edited["status"] == "edited" and edited["reviewer"] == "admin" and edited["validator_report"]["valid"]
    rejected = app.post(f"/api/suggestions/{good}/reject", json={"reason": "no"}, headers=headers).json()
    assert rejected["status"] == "rejected"
    assert app.post(f"/api/suggestions/{good}/edit", json={"rule": GOOD_RULE}, headers=headers).status_code == 409
    assert not any("suggestion" in line for line in state.recent)                    # nothing in the live feed


def test_accepted_review_through_the_router_ends_as_a_suggestion(tmp_path):
    from modules.trust.scorer import TrustScorer

    store = RuleSuggestionStore(str(tmp_path / "suggestions.db"))
    scorer = TrustScorer.from_config(str(CONFIG))
    scorer.evidence.suggestions = triggers(store)
    review_event = ev("escalation_review", {"review_id": "d1", "location_id": "lab", "risk_level_llm": "critical",
                                            "risk_level_final": "medium", "summary": "дим", "reasons": ["smoke"]})
    scorer.apply(review_event)
    scorer.apply(ev("escalation_resolved", {"review_id": "d1", "accepted": True, "by": "olena",
                                            "location_id": "lab"}, device="dash"))
    accepted = [j for j in store.pending_jobs() if j["trigger"] == "accepted_escalation"]
    assert len(accepted) == 1
    RuleSuggester(store, FakeGPT(answer()), str(CONFIG)).process_pending(10)
    suggestion = store.list()[0]
    assert suggestion["supporting_cases"] == accepted[0]["case_ids"] and suggestion["trigger"] == "accepted_escalation"
