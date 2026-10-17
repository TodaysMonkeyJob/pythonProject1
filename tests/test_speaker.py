"""F10 speaker attribution: diarization, identification, alignment, transcript logs (synthetic voices, no model)."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from core.events import make_event
from core.identity import IdentityRegistry, normalize
from modules.audio.attribution import Segment, TranscriptLog, align, text_line, whisper_segments
from modules.audio.speaker import (SAMPLE_RATE, UNCERTAIN, SpeakerAnalyzer, SpeakerConfig, Turn, cluster_embeddings,
                                   preprocess)

# A "voice" is a tone; the fake ECAPA maps each tone's energy to that speaker's embedding, so mixed speech
# (two tones at once) lands between the two speakers, like real overlapping speech
VOICE_A, VOICE_B, VOICE_C = (normalize(np.random.default_rng(seed).normal(size=192)) for seed in (11, 12, 13))
TONES = {220: VOICE_A, 1200: VOICE_B, 2500: VOICE_C}


def tone(freq, seconds, amplitude=0.3):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), np.float32)


def fake_embed(windows):
    rng = np.random.default_rng(len(windows))
    freqs = np.fft.rfftfreq(windows.shape[1], 1 / SAMPLE_RATE)
    out = []
    for window in windows:
        spectrum = np.abs(np.fft.rfft(window))
        e = sum(spectrum[(freqs > f - 40) & (freqs < f + 40)].sum() * v for f, v in TONES.items())
        out.append(normalize(normalize(e) + 0.15 * rng.normal(size=192) / np.sqrt(192)))
    return np.array(out)


def fake_mask(audio):
    frame = int(0.03 * SAMPLE_RATE)
    return np.array([np.sqrt(np.mean(audio[i:i + frame] ** 2)) > 0.02 for i in range(0, len(audio) - frame + 1, frame)])


@pytest.fixture
def registry(tmp_path):
    reg = IdentityRegistry(str(tmp_path / "identity.db"))
    yield reg
    reg.close()


def analyzer(registry, **cfg):
    return SpeakerAnalyzer(registry, fake_embed, fake_mask, SpeakerConfig(**cfg))


def speakers_by_time(d):
    return [(round(t.start, 2), round(t.end, 2), d.speaker(t.cluster).name) for t in d.turns]


# --- building blocks ----------------------------------------------------------------------------------------------

def test_cluster_embeddings_separates_two_groups():
    rng = np.random.default_rng(0)
    group = lambda base, n: [normalize(base + 0.3 * rng.normal(size=192) / np.sqrt(192)) for _ in range(n)]  # noqa
    embeddings = np.array(group(VOICE_A, 5) + group(VOICE_B, 4) + group(VOICE_A, 2))
    assert cluster_embeddings(embeddings, 0.75).tolist() == [0] * 5 + [1] * 4 + [0] * 2
    assert cluster_embeddings(embeddings[:1], 0.75).tolist() == [0]


def test_preprocess_resamples_filters_and_normalises():
    hum = tone(30, 1.0, 0.5)                                       # below the 80 Hz high-pass
    voice = tone(440, 1.0, 0.01)                                   # very quiet
    out = preprocess((hum + voice) * 32767 // 1, SAMPLE_RATE)
    assert out.dtype == np.float32 and len(out) == SAMPLE_RATE
    assert abs(20 * np.log10(np.sqrt(np.mean(out ** 2))) + 20) < 0.5   # −20 dBFS
    resampled = preprocess(np.zeros((48000, 2), np.float32), 48000)    # stereo 48 kHz → mono 16 kHz
    assert len(resampled) == SAMPLE_RATE


# --- diarization + identification ---------------------------------------------------------------------------------

def test_two_speakers_in_one_chunk(registry):
    d = analyzer(registry).analyze(np.concatenate([tone(220, 3), tone(1200, 3)]), SAMPLE_RATE)
    names = speakers_by_time(d)
    assert len(names) == 2 and {n for _, _, n in names} == {"unknown_1", "unknown_2"}         # two people
    assert abs(names[0][1] - 3.0) <= 0.75                          # turn boundary near the change
    assert {s.status for s in d.speakers.values()} == {"unknown"}


def test_unknown_speaker_keeps_one_label_across_chunks(registry):
    a = analyzer(registry)
    first = a.analyze(tone(220, 4), SAMPLE_RATE).speakers[0]
    for _ in range(5):
        again = a.analyze(np.concatenate([silence(0.5), tone(220, 2.5)]), SAMPLE_RATE).speakers[0]
        assert again.person_id == first.person_id and again.name == "unknown_1"
    assert len(registry.list_persons()) == 1


def test_known_speaker_and_rename_applies(registry):
    pid = registry.create_unknown("voice", VOICE_B)
    registry.rename(pid, "Anna")
    who = analyzer(registry).analyze(tone(1200, 3), SAMPLE_RATE).speakers[0]
    assert (who.person_id, who.name, who.status) == (pid, "Anna", "known") and who.score > 0.9


def test_short_speech_is_uncertain_and_not_stored(registry):
    d = analyzer(registry).analyze(np.concatenate([silence(0.3), tone(220, 0.9)]), SAMPLE_RATE)
    assert {d.speaker(t.cluster).status for t in d.turns} == {UNCERTAIN}
    assert registry.list_persons() == []


def test_overlapping_speech_is_never_given_to_a_known_person(registry):
    for voice, name in ((VOICE_A, "Oleh"), (VOICE_B, "Anna")):
        registry.rename(registry.create_unknown("voice", voice), name)
    both = tone(220, 3) + tone(1200, 3)                            # talking at the same time
    d = analyzer(registry).analyze(both, SAMPLE_RATE)
    assert {d.speaker(t.cluster).status for t in d.turns} == {UNCERTAIN}


def test_enrichment_only_from_clean_single_speaker_chunks(registry):
    pid = registry.create_unknown("voice", VOICE_A)
    a = analyzer(registry)
    a.analyze(np.concatenate([tone(220, 1.5), silence(0.5), tone(220, 1.5)]), SAMPLE_RATE)   # 3 s, quiet gap
    assert registry.counts[pid]["voice"] == 2
    a.analyze(np.concatenate([tone(220, 3), tone(1200, 3)]), SAMPLE_RATE)                    # two speakers
    a.analyze(tone(220, 1.6), SAMPLE_RATE)                                                    # < 2 s
    assert registry.counts[pid]["voice"] == 2


# --- alignment ----------------------------------------------------------------------------------------------------

TURNS = [Turn(0.0, 3.0, 0), Turn(3.0, 6.0, 1), Turn(6.0, 7.0, None)]


def test_align_by_largest_overlap():
    out = align([Segment(0.2, 2.8, "close the door"), Segment(2.7, 5.5, "ok one second"),
                 Segment(6.1, 6.9, "hm"), Segment(8.0, 9.0, "late")], TURNS)
    assert [(s.text, s.cluster) for s in out] == [("close the door", 0), ("ok one second", 1), ("hm", None),
                                                  ("late", None)]


def test_segment_spanning_two_turns_is_split_by_time():
    (first, second) = align([Segment(1.0, 5.0, "one two three four five six seven eight")], TURNS)
    assert (first.cluster, first.start, first.end) == (0, 1.0, 3.0)
    assert (second.cluster, second.start, second.end) == (1, 3.0, 5.0)
    assert first.text == "one two three four" and second.text == "five six seven eight"


def test_same_speaker_turns_merge_and_tiny_overlap_does_not_split():
    turns = [Turn(0.0, 2.0, 0), Turn(2.0, 2.3, 0), Turn(2.3, 6.0, 1)]
    assert [(s.cluster, s.text) for s in align([Segment(0.0, 2.6, "a b c")], turns)] == [(0, "a b c")]


def test_whisper_segments_from_results():
    result = {"text": "hello there", "segments": [{"start": 0.0, "end": 1.2, "text": " hello"},
                                                  {"start": 1.2, "end": 9.9, "text": "there "},
                                                  {"start": 2.0, "end": 2.5, "text": "  "}]}
    assert [(s.start, s.end, s.text) for s in whisper_segments(result, 2.0)] == [(0.0, 1.2, "hello"),
                                                                                 (1.2, 2.0, "there")]
    assert [(s.start, s.end) for s in whisper_segments({"text": "hi"}, 1.5)] == [(0.0, 1.5)]
    from modules.llm.whisper_client import _segments

    azure_like = [SimpleNamespace(start=0.0, end=1.0, text=" a"), SimpleNamespace(start=1.0, end=2.0, text="")]
    assert _segments(azure_like) == [{"start": 0.0, "end": 1.0, "text": "a"}]


# --- logs ---------------------------------------------------------------------------------------------------------

PAYLOAD = {"person_id": "p1", "speaker_name": "unknown_4", "speaker_score": 0.62, "speaker_status": "unknown",
           "text": "OK, one second.", "language": "en", "start": "2026-10-17T12:30:09.120Z",
           "end": "2026-10-17T12:30:10.000Z", "device_id": "mic-mac", "location_id": "warehouse"}


def test_text_line_format_and_name_resolution():
    local = datetime(2026, 10, 17, 12, 30, 9, tzinfo=timezone.utc).astimezone().strftime("%H:%M:%S")
    assert text_line(PAYLOAD) == f'[{local}] [warehouse · mic-mac] unknown_4: "OK, one second."'
    assert "] Oleh: " in text_line(PAYLOAD, lambda pid, name: "Oleh")
    assert "] ?: " in text_line({**PAYLOAD, "speaker_status": "uncertain"})


def test_transcript_log_writes_jsonl_and_txt(tmp_path):
    clock = lambda: datetime(2026, 10, 17, 12, 31, tzinfo=timezone.utc)  # noqa: E731
    log = TranscriptLog(str(tmp_path), resolve_name=lambda pid, name: "Oleh", clock=clock)
    log.write(make_event("audio", "speech", PAYLOAD))
    log.close()
    day = tmp_path / "2026-10-17"
    assert json.loads((day / "transcripts.jsonl").read_text())["payload"]["person_id"] == "p1"
    assert (day / "transcripts.txt").read_text().strip().endswith('Oleh: "OK, one second."')


def test_calibration_eer_and_threshold():
    from tools.calibrate_voice import equal_error_rate, report

    rng = np.random.default_rng(1)
    people = {name: np.array([normalize(v + 0.6 * rng.normal(size=192) / np.sqrt(192)) for _ in range(6)])
              for name, v in (("a", VOICE_A), ("b", VOICE_B), ("c", VOICE_C))}
    result = report(people)
    assert result["eer"] == 0.0 and 0.1 < result["suggested_voice_threshold"] < 0.9
    assert result["same_speaker"]["mean"] > result["different_speaker"]["mean"]
    eer, threshold = equal_error_rate(np.array([0.5, 0.6, 0.2]), np.array([0.1, 0.3, 0.05]))
    assert eer == pytest.approx(1 / 3) and 0.2 <= threshold <= 0.3
