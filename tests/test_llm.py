import json
import queue
from types import SimpleNamespace

import httpx
import openai
import pytest

from core.events import Event
from modules.aggregator.outbox import Outbox
from modules.llm.gpt_client import GPTClient, InvalidResponse, LLMDecision, load_prompt, parse_decision
from modules.llm.worker import LLMWorker

GOOD = {"risk_level": "high", "summary": "Ацетон перевищив поріг.", "recommended_action": "notify",
        "reasons": ["gas: alerts gas_threshold"], "follow_up": "Перевірити через 60 с."}
REQUEST = httpx.Request("POST", "https://example.openai.azure.com/openai/v1/chat/completions")
CONFIG = {"llm": {"gpt_deployment": "gpt-4o-mini", "price_per_1k_input": 0.15, "price_per_1k_output": 0.6}}


def completion(content, prompt_tokens=400, completion_tokens=80):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                           usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))


class FakeOpenAI:
    """client.chat.completions.create replays scripted answers (str / exception)."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return completion(answer)


def status_error(cls, code, message="error"):
    return cls(message, response=httpx.Response(code, request=REQUEST), body=None)


# --- client -----------------------------------------------------------------------------------------------------

def test_load_prompt_and_parse():
    prompt = load_prompt("security_analyst")
    assert "Reply ONLY with JSON" in prompt and "lying" in prompt
    assert parse_decision(json.dumps(GOOD)).risk_level == "high"
    assert parse_decision("```json\n" + json.dumps(GOOD) + "\n```").recommended_action == "notify"
    with pytest.raises(ValueError):
        parse_decision("not json")


def test_analyze_sends_digest_with_system_prompt_and_counts_cost():
    fake = FakeOpenAI(json.dumps(GOOD))
    gpt = GPTClient(CONFIG, client=fake)
    decision, usage = gpt.analyze("12:00:00-12:01:00 UTC+3\ngas: acetone 62000ppm rising alerts gas_threshold")
    assert decision == LLMDecision(**GOOD)
    call = fake.calls[0]
    assert call["model"] == "gpt-4o-mini" and call["temperature"] == 0.2
    assert call["response_format"] == {"type": "json_object"}
    assert call["messages"][0]["role"] == "system" and "indoor security" in call["messages"][0]["content"]
    assert call["messages"][1]["content"].startswith("12:00:00")
    assert (usage.prompt_tokens, usage.completion_tokens) == (400, 80)
    assert usage.cost_usd == pytest.approx((400 * 0.15 + 80 * 0.6) / 1000)
    assert gpt.total.requests == 1


def test_invalid_answer_gets_one_clarification_retry():
    fake = FakeOpenAI('{"risk_level": "extreme"}', json.dumps(GOOD))
    decision, usage = GPTClient(CONFIG, client=fake).analyze("digest")
    assert decision.risk_level == "high" and usage.requests == 2
    retry_messages = fake.calls[1]["messages"]
    assert retry_messages[-2]["role"] == "assistant" and "valid JSON" in retry_messages[-1]["content"]


def test_invalid_twice_raises():
    with pytest.raises(InvalidResponse):
        GPTClient(CONFIG, client=FakeOpenAI("nope", "still nope")).analyze("digest")


def test_json_mode_fallback_when_deployment_rejects_it():
    unsupported = status_error(openai.BadRequestError, 400, "response_format json_object is not supported")
    fake = FakeOpenAI(unsupported, json.dumps(GOOD), json.dumps(GOOD))
    gpt = GPTClient(CONFIG, client=fake)
    gpt.analyze("digest")
    gpt.analyze("digest")
    assert "response_format" in fake.calls[0] and "response_format" not in fake.calls[1]
    assert "response_format" not in fake.calls[2] and gpt.json_mode is False      # remembered


def test_missing_credentials():
    from modules.llm.azure_client import MissingCredentials, create_client

    with pytest.raises(MissingCredentials):
        create_client({})
    assert type(create_client({"azure_api_key": "k", "azure_endpoint": "https://x.openai.azure.com/openai/v1/"})
                ).__name__ == "OpenAI"
    assert type(create_client({"azure_api_key": "k", "azure_endpoint": "https://x.openai.azure.com/",
                               "azure_api_version": "2024-06-01"})).__name__ == "AzureOpenAI"


# --- worker -----------------------------------------------------------------------------------------------------

class FakeGPT:
    deployment = "gpt-4o-mini"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.digests = []

    def analyze(self, digest):
        self.digests.append(digest)
        answer = self.answers.pop(0) if self.answers else GOOD
        if isinstance(answer, Exception):
            raise answer
        from modules.llm.gpt_client import Usage

        return LLMDecision(**answer), Usage(400, 80, 0.0001, 900, 1)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make_worker(tmp_path, gpt, min_interval=300):
    box = Outbox(str(tmp_path / "outbox.db"))
    clock = Clock()
    config = {"llm": {"min_llm_interval_sec": min_interval, "poll_sec": 0, "max_backoff_sec": 300}}
    worker = LLMWorker(queue.Queue(), queue.Queue(), config, gpt=gpt, outbox=box, clock=clock)
    return worker, box, clock


def add(box, n, trigger="interval", created=None, incident=None):
    snapshot = {"snapshot_id": f"s{n}", "trigger": trigger}
    if incident:
        snapshot["incident_id"] = incident
    box.add(snapshot, trigger, created_at=created or n, digest=f"digest {n}")


def decisions(worker):
    events = []
    while not worker.event_queue.empty():
        events.append(Event.from_json(worker.event_queue.get()))
    return events


def test_critical_sent_immediately_with_decision_event(tmp_path):
    gpt = FakeGPT()
    worker, box, clock = make_worker(tmp_path, gpt)
    add(box, 1, "critical_event", incident="20261017-120000-ab12")
    worker.step()
    (event,) = decisions(worker)
    assert (event.source, event.type, event.severity) == ("llm", "llm_decision", "critical")
    assert event.payload["risk_level"] == "high" and event.payload["snapshot_id"] == "s1"
    assert event.payload["incident_id"] == "20261017-120000-ab12"
    assert event.payload["prompt_tokens"] == 400 and event.payload["model"] == "gpt-4o-mini"
    assert gpt.digests == ["digest 1"] and box.counts() == {"sent": 1}


def test_regular_snapshots_throttled_and_stale_ones_skipped(tmp_path):
    gpt = FakeGPT()
    worker, box, clock = make_worker(tmp_path, gpt, min_interval=300)
    add(box, 1)
    worker.step()
    assert gpt.digests == ["digest 1"]
    for n in (2, 3, 4):                                                  # three more within the interval
        add(box, n)
        clock.now += 60
        worker.step()
    assert gpt.digests == ["digest 1"]                                    # nothing sent inside the interval
    add(box, 5, "critical_event")
    worker.step()
    assert gpt.digests[-1] == "digest 5"                                  # critical ignores the throttle
    clock.now += 120                                                      # interval over
    worker.step()
    assert gpt.digests == ["digest 1", "digest 5", "digest 4"]            # only the newest regular one
    assert box.counts() == {"sent": 3, "failed": 2}
    assert box.pending() == []                                            # skipped rows are never retried


def test_offline_keeps_rows_pending_and_backs_off(tmp_path):
    offline = openai.APIConnectionError(request=REQUEST)
    gpt = FakeGPT(offline, offline)
    worker, box, clock = make_worker(tmp_path, gpt)
    add(box, 1, "critical_event")
    worker.step()
    assert box.counts() == {"pending": 1} and worker._blocked_until == clock.now + 2
    clock.now += 1
    worker.step()
    assert len(gpt.digests) == 1                                          # still waiting
    clock.now += 1
    worker.step()
    assert worker._blocked_until == clock.now + 4                         # 2 → 4 s
    clock.now += 4
    worker.step()                                                         # network back
    assert box.counts() == {"sent": 1} and decisions(worker)[0].type == "llm_decision"


@pytest.mark.parametrize("error", [
    status_error(openai.RateLimitError, 429),
    status_error(openai.InternalServerError, 503),
    openai.APITimeoutError(request=REQUEST),
])
def test_transient_errors_back_off(tmp_path, error):
    worker, box, clock = make_worker(tmp_path, FakeGPT(error))
    add(box, 1, "critical_event")
    worker.step()
    assert box.counts() == {"pending": 1} and worker._blocked_until > clock.now


def test_auth_error_pauses_without_losing_rows(tmp_path):
    worker, box, clock = make_worker(tmp_path, FakeGPT(status_error(openai.AuthenticationError, 401)))
    add(box, 1, "critical_event")
    worker.step()
    assert box.counts() == {"pending": 1} and worker._blocked_until == clock.now + 300


def test_invalid_decision_marks_failed_and_continues(tmp_path):
    worker, box, clock = make_worker(tmp_path, FakeGPT(InvalidResponse("bad json"), GOOD))
    add(box, 1, "critical_event", created=1)
    add(box, 2, "critical_event", created=2)
    worker.step()
    assert box.counts() == {"failed": 1, "sent": 1}


def test_no_credentials_module_idles_without_crashing(tmp_path, monkeypatch):
    for name in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    config = {"llm": {}, "aggregator": {"outbox_path": str(tmp_path / "outbox.db")}}
    worker = LLMWorker(queue.Queue(), queue.Queue(), config)
    worker.setup()
    assert worker.gpt is None and "AZURE_OPENAI_API_KEY" in worker.unavailable
    add(worker.outbox, 1, "critical_event")
    worker.step()
    assert worker.outbox.counts() == {"pending": 1}
    worker.teardown()


def test_aggregator_ignores_llm_decisions_as_triggers(tmp_path):
    from core.events import make_event
    from tests.test_aggregator import make

    aggregator, _ = make(tmp_path)
    decision = make_event("llm", "llm_decision", {**GOOD, "risk_level": "critical"}, severity="critical")
    assert aggregator.on_event(decision) is None
