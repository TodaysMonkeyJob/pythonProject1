"""Whisper accuracy settings: language restriction, hallucination filters, backlog skipping (no models / network)."""
import queue
from types import SimpleNamespace

import numpy as np

from core.events import Event
from modules.audio.mic_controller import MicController
from modules.llm.whisper_client import WhisperOptions, _segments, azure_transcriber, local_transcriber

AUDIO = np.zeros(16000, np.int16)


def seg(text, start=0.0, end=1.0, no_speech=0.1, logprob=-0.3, compression=1.2):
    return SimpleNamespace(text=text, start=start, end=end, no_speech_prob=no_speech, avg_logprob=logprob,
                           compression_ratio=compression)


def test_hallucinations_are_dropped():
    kept = _segments([seg("Закрий двері."), seg("Ммм", no_speech=0.9, logprob=-1.4),
                      seg("I don't know, I don't know, I don't know", compression=3.1),
                      seg("Дякую за перегляд!"), seg("Субтитры сделал DimaTorzok"),
                      seg("Тихо, але впевнено", no_speech=0.9, logprob=-0.2)])
    assert [s["text"] for s in kept] == ["Закрий двері.", "Тихо, але впевнено"]


def test_options_from_config():
    opts = WhisperOptions.from_mic({"languages": ["UK", "en"], "whisper_beam_size": 3, "whisper_prompt": "Oleh",
                                    "hallucination_phrases": None})
    assert opts.languages == ("uk", "en") and opts.beam_size == 3 and opts.initial_prompt == "Oleh"
    assert opts.condition_on_previous_text is False and "дякую за перегляд" in opts.hallucination_phrases
    assert opts.allowed("uk") and not opts.allowed("ru") and WhisperOptions().allowed("ru")


class FakeWhisper:
    def __init__(self, probabilities):
        self.probabilities, self.calls = probabilities, []

    def detect_language(self, audio):
        return self.probabilities[0][0], self.probabilities[0][1], self.probabilities

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return iter([seg("привіт"), seg("дякую за перегляд")]), SimpleNamespace(language=kwargs["language"])


def test_local_detection_only_among_allowed_languages():
    model = FakeWhisper([("ru", 0.55), ("uk", 0.35), ("en", 0.05)])     # Ukrainian that sounds Russian-ish
    transcribe = local_transcriber({"audio": {"mic": {"languages": ["uk", "en"]}}}, whisper_model=model)
    result = transcribe(AUDIO, 16000)
    assert result["language"] == "uk" and result["text"] == "привіт" and result["backend"] == "local"
    call = model.calls[0]
    assert call["language"] == "uk" and call["condition_on_previous_text"] is False and call["beam_size"] == 5
    fixed = FakeWhisper([("ru", 0.9)])
    local_transcriber({"audio": {"mic": {"language": "en"}}}, whisper_model=fixed)(AUDIO, 16000)
    assert fixed.calls[0]["language"] == "en"


class FakeAzure:
    def __init__(self, *languages):
        self.languages, self.calls = list(languages), []
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        language = kwargs.get("language") or self.languages.pop(0)
        return SimpleNamespace(text="x", language=language, segments=[seg("Закрий двері.")])


def test_azure_retries_in_the_main_language_when_detection_is_off_list():
    client = FakeAzure("russian")
    config = {"audio": {"mic": {"languages": ["uk", "en"], "whisper_prompt": "Oleh"}}, "llm": {}}
    result = azure_transcriber(config, client=client)(AUDIO, 16000)
    assert result["language"] == "uk" and result["text"] == "Закрий двері."
    assert "language" not in client.calls[0] and client.calls[1]["language"] == "uk"
    assert client.calls[0]["prompt"] == "Oleh" and client.calls[0]["temperature"] == 0
    english = FakeAzure("english")
    assert azure_transcriber(config, client=english)(AUDIO, 16000)["language"] == "en" and len(english.calls) == 1


def test_stale_segments_are_skipped_when_whisper_falls_behind():
    texts = []
    module = MicController(queue.Queue(), queue.Queue(),
                           {"audio": {"mic": {"whisper_max_backlog": 2, "transcript_log": False}}},
                           frame_source=object(), transcriber=lambda a, sr: texts.append(1) or {"text": "hi"})
    module.transcriptions = 5                      # 5 segments submitted so far
    module._transcribe(AUDIO, 1.0, 100.0, number=3)    # 2 newer ones wait → skipped
    module._transcribe(AUDIO, 1.0, 100.0, number=4)    # 1 newer one → transcribed
    assert len(texts) == 1
    event = Event.from_json(module.event_queue.get_nowait())
    assert event.payload["text"] == "hi"
