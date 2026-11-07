"""A4 (TOD-44): intensity above the background, noise stage, sound types, danger words, the controller's events."""
import json
import os

import numpy as np
import pytest
import soundfile as sf

from modules.audio.intensity import BackgroundLevel
from modules.audio.keywords import DangerWords, levenshtein, normalise
from modules.audio.noise_stage import NoiseProfile, SoundBurst, wiener_denoise
from modules.audio.sound_type import SoundTypeClassifier, clips, features, load_types
from modules.hfs.audio import AudioSubsystem
from tests.test_mic_controller import SILENCE, SPEECH, drain, feed, make
from tools.audio_bench import synthetic_model, synthetic_sounds


# --- danger words -----------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def words():
    return DangerWords.load()


def test_critical_word_gives_critical_risk(words):
    grade, matches = words.grade("Пожежа!")
    assert grade == 1.0 and matches[0].level == "critical" and matches[0].kind == "exact"
    audio = AudioSubsystem({}, words=words)
    assert audio.infer({"keyword_danger": grade})["level"] == "critical"


@pytest.mark.parametrize("text, entry, kind", [
    ("тут пожежя", "пожежа", "stem"),                    # Whisper's near miss of «пожежа»
    ("пожежею", "пожежа", "stem"),
    ("допоможить", "допоможіть", "stem"),
    ("допаможіть", "допоможіть", "fuzzy"),
    ("рятуйтее", "рятуйте", "fuzzy"),
    ("Небезпека!", "небезпека", "exact"),
    ("вибухнуло", "вибух", "exact"),
    ("витік газу", "газ", "exact"),
    ("Help me, please", "help", "exact"),
    ("ОЙ", "ой", "exact"),
])
def test_close_forms_match(words, text, entry, kind):
    matches = words.find(text)
    assert matches and matches[0].entry == entry and matches[0].kind == kind, [m.payload() for m in matches]


@pytest.mark.parametrize("text", ["газета на столі", "ще раз", "the file is fine", "добрий день", "вибори",
                                  "fire" + "place"])
def test_ordinary_words_do_not_match(words, text):
    assert words.grade(text) == (0.0, [])


def test_normalise_and_levenshtein():
    assert normalise("Обʼєкт — ГОРИТЬ!!") == ["об'єкт", "горить"]
    assert levenshtein("пожежя", "пожежа") == 1 and levenshtein("abc", "xyzuvw", limit=2) == 3


# --- intensity ---------------------------------------------------------------------------------------------------

def test_intensity_is_relative_to_the_background():
    quiet, noisy = BackgroundLevel(warmup_sec=3), BackgroundLevel(warmup_sec=3)
    t = 0.0
    for _ in range(400):                                # 12 s of background: −60 dBFS vs −35 dBFS (a fan)
        quiet.update(-60, t), noisy.update(-35, t)
        t += 0.03
    assert quiet.background == pytest.approx(-60) and noisy.background == pytest.approx(-35)
    assert quiet.relative(-30) == 30 and noisy.relative(-30) == 5          # the same −30 dBFS: a shout vs a word
    for _ in range(200):                                # 6 s at −30 dBFS: smoothed (τ 2 s) towards +30 dB
        quiet.update(-30, t)
        t += 0.03
    assert 25 < quiet.intensity <= 30
    assert BackgroundLevel(background_db=-50).relative(-40) == 10            # calibrated once per microphone


# --- noise stage --------------------------------------------------------------------------------------------------

def test_noise_profile_wiener_and_bursts():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.02, 16000).astype(np.float32)
    profile = NoiseProfile(keep_sec=1.0)
    for start in range(0, 16000, 480):
        profile.update((noise[start:start + 480] * 32768).astype(np.int16), is_speech=False)
    profile.update(np.ones(480, np.int16), is_speech=True)                   # speech frames are not noise
    assert profile.seconds == pytest.approx(1.0, abs=0.04) and profile.psd is not None
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(8000) / 16000)).astype(np.float32)
    clean = wiener_denoise(tone + noise[:8000], profile.psd)
    residual_before = np.std((tone + noise[:8000]) - tone)
    assert np.std(clean - tone) < residual_before and len(clean) == 8000
    burst = SoundBurst(min_db=10, min_sec=0.3, gap_sec=0.3)
    out = [burst.push(np.full(480, 2000, np.int16), 25, False) for _ in range(20)]     # 0.6 s loud
    out += [burst.push(np.zeros(480, np.int16), 0, False) for _ in range(12)]
    finished = [o for o in out if o is not None]
    assert len(finished) == 1 and len(finished[0]) >= 0.6 * 16000 and burst.take_peak() == 25


# --- sound types ---------------------------------------------------------------------------------------------------

def test_without_a_model_every_sound_is_unknown_and_no_evidence():
    classifier = SoundTypeClassifier.from_config({"sound_type_model": "models/does-not-exist.joblib"})
    result = classifier.classify(synthetic_sounds("scream"))
    assert result.payload() == {"type": "unknown", "confidence": 0.0, "danger": None, "model": None}
    assert features(synthetic_sounds("fan")).shape == (96,)


def test_classifier_with_a_model_reports_type_and_danger():
    classifier = SoundTypeClassifier(load_types(), synthetic_model(), "synthetic.joblib")
    result = classifier.classify(synthetic_sounds("scream", seed=123))
    assert result.type == "scream" and result.danger == 0.8 and result.confidence >= 0.5
    assert classifier.classify(synthetic_sounds("keyboard", seed=124)).danger == 0.0
    assert len(clips(np.zeros(5 * 16000, np.float32))) == 3                   # 2 s clips, a half-long tail kept


def test_training_tool_reports_per_type_accuracy(tmp_path):
    from tools.train_sound_types import main

    for kind in ("fan", "keyboard", "walking", "scream"):
        folder = tmp_path / "data" / kind
        folder.mkdir(parents=True)
        for i in range(4):
            sf.write(folder / f"{kind}_{i}.wav", synthetic_sounds(kind, 6.0, seed=i), 16000)
    (tmp_path / "data" / "birdsong").mkdir()                                  # not a configured type: skipped
    assert main(["--data", str(tmp_path / "data"), "--out", str(tmp_path / "m.joblib"),
                 "--report", str(tmp_path / "report")]) == 0
    metrics = json.loads((tmp_path / "report" / "metrics.json").read_text())
    assert set(metrics["per_type"]) == {"fan", "keyboard", "walking", "scream"} and metrics["split"] == "by file"
    assert metrics["accuracy"] >= 0.9 and os.path.isfile(tmp_path / "m.joblib")
    (tmp_path / "empty").mkdir()
    assert main(["--data", str(tmp_path / "empty"), "--out", str(tmp_path / "x"), "--report",
                 str(tmp_path / "r2")]) == 1                                  # no types: nothing to train


# --- the microphone controller --------------------------------------------------------------------------------------

def test_controller_publishes_sound_type_level_and_danger_words(tmp_path):
    controller, mic, clock, _ = make(sound_type_model=str(tmp_path / "none.joblib"), level_event_sec=0.5)
    controller.transcriber = lambda audio, sr: {"text": "Пожежа, допоможіть!", "language": "uk"}
    feed(controller, mic, clock, [SILENCE] * 100 + [SPEECH] * 20 + [SILENCE] * 40)
    controller.executor.shutdown(wait=True)
    controller.sound_executor.shutdown(wait=True)
    events = drain(controller)
    (speech,) = [e for e in events if e.type == "speech"]
    assert speech.payload["danger_words"]["grade"] == 1.0
    assert speech.payload["danger_words"]["matches"][0]["entry"] == "пожежа"
    types = [e for e in events if e.type == "sound_type"]
    assert types and types[0].payload["source"] == "background" and types[0].payload["type"] == "unknown"
    assert types[0].payload["danger"] is None                                 # no model: not evidence
    levels = [e for e in events if e.type == "audio_level"]
    assert levels and max(e.payload["intensity_db"] for e in levels) > 10      # speech well above the silence
    sounds = [e for e in events if e.type == "sound"]
    assert all("relative_db" in e.payload for e in sounds)
