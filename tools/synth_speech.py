"""
Synthetic labelled speech for the ASR / speaker calibration (F22 / TOD-33) — macOS ``say`` voices (built in, nothing
installed), known reference text, white noise at fixed SNRs. Deterministic for a seed.

    python -m tools.synth_speech --out data/calibration/speech          # → <voice>/<id>.wav + manifest.csv

Each voice is one "speaker" (speaker calibration: genuine = same voice, impostor = another voice); each clip has a
reference text (ASR calibration: WER of the hypothesis). TTS speech is cleaner and more regular than a real room, so
the fitted maps are a measured starting point, to be refitted on recorded phrases (``asr --csv`` accepts any CSV of
the same columns). The audio is git-ignored; the transcribed CSV / pair scores are the committed dataset.
"""
import argparse
import csv
import os
import random
import subprocess
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
EN_VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Rishi", "Tessa", "Fred", "Ralph", "Kathy", "Albert"]
UK_VOICES = ["Lesya"]
SNRS_DB = [None, 30, 20, 10, 5, 0, -5]          # None = clean
EN_PHRASES = [
    "The gas sensor in the kitchen shows a higher level than usual.",
    "Please check the camera in the warehouse after six o'clock.",
    "Everything is fine, the alarm was only a test.",
    "Somebody is standing near the back door.",
    "I am going to the meeting room for an hour.",
    "Can you send me the report before lunch?",
    "The door to the server room is open again.",
    "Turn off the coffee machine when you leave.",
    "There is a strange smell near the storage shelves.",
    "I will call the technician tomorrow morning.",
    "Nobody should be in the building at night.",
    "The delivery arrives at three in the afternoon.",
    "Ignore the alarm, the sensor is broken.",
    "My colleague is waiting for me at the entrance.",
    "Open the window, the air is getting heavy.",
    "Where did you put the keys to the cabinet?",
    "The printer on the second floor is out of paper.",
    "Let us lock the warehouse before we go home.",
    "This is a security check, please stay calm.",
    "The temperature in the office is too high today.",
    "He fell down near the stairs, call for help.",
    "I think the smoke is coming from the kitchen.",
    "We have a visitor waiting in the lobby.",
    "Do not touch the cables behind the desk.",
    "The night shift starts at ten o'clock.",
    "Status report: all systems are working normally.",
    "Bring the fire extinguisher from the corridor.",
    "I left my laptop charging in the meeting room.",
    "The camera on the parking lot shows nothing.",
    "Thank you, I will be back in five minutes.",
]
UK_PHRASES = [
    "Датчик газу на кухні показує підвищений рівень.",
    "Перевір камеру на складі після шостої години.",
    "Все гаразд, тривога була лише перевіркою.",
    "Хтось стоїть біля задніх дверей.",
    "Я йду на нараду приблизно на годину.",
    "Надішли мені звіт до обіду, будь ласка.",
    "Двері до серверної знову відчинені.",
    "Вимкни кавоварку, коли будеш виходити.",
    "Біля полиць на складі дивний запах.",
    "Завтра зранку я подзвоню майстру.",
    "Вночі в будівлі нікого не повинно бути.",
    "Не звертай уваги на тривогу, датчик зламаний.",
    "Відчини вікно, повітря стає важким.",
    "Де ключі від шафи з документами?",
    "Він упав біля сходів, викликай допомогу.",
    "Здається, дим іде з кухні.",
    "У холі чекає відвідувач.",
    "Нічна зміна починається о десятій.",
    "Усі системи працюють у звичайному режимі.",
    "Дякую, я повернуся за п'ять хвилин.",
]


def say(text: str, voice: str, path: str) -> np.ndarray:
    """:return: 16 kHz mono float32 audio of ``text`` spoken by a macOS voice"""
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "out.wav")
        subprocess.run(["say", "-v", voice, "-o", raw, "--file-format=WAVE", f"--data-format=LEI16@{SAMPLE_RATE}",
                        text], check=True, capture_output=True)
        audio, rate = sf.read(raw, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        raise RuntimeError(f"say produced {rate} Hz")
    return audio


def add_noise(audio: np.ndarray, snr_db: Optional[float], rng: np.random.Generator) -> np.ndarray:
    if snr_db is None:
        return audio
    power = float(np.mean(audio ** 2)) or 1e-8
    noise = rng.normal(0.0, np.sqrt(power / 10 ** (snr_db / 10)), len(audio)).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)


def plan(seed: int) -> List[Tuple[str, str, str, int, Optional[float]]]:
    """:return: [(clip_id, voice, language, phrase index, snr_db)] — every phrase per voice, a random SNR each"""
    rng, rows = random.Random(seed), []
    for voices, phrases, language in ((EN_VOICES, EN_PHRASES, "en"), (UK_VOICES, UK_PHRASES, "uk")):
        for voice in voices:
            for i in range(len(phrases)):
                snr = rng.choice(SNRS_DB)
                rows.append((f"{voice.lower()}_{language}_{i:02d}", voice, language, i, snr))
    return rows


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Synthetic labelled speech (macOS say) for calibration")
    parser.add_argument("--out", default="data/calibration/speech")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    noise_rng = np.random.default_rng(args.seed)
    rows = []
    for clip_id, voice, language, index, snr in plan(args.seed):
        text = (EN_PHRASES if language == "en" else UK_PHRASES)[index]
        folder = os.path.join(args.out, voice.lower())
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{clip_id}.wav")
        sf.write(path, add_noise(say(text, voice, path), snr, noise_rng), SAMPLE_RATE, subtype="PCM_16")
        rows.append({"audio_id": clip_id, "path": os.path.relpath(path, args.out), "speaker": voice.lower(),
                     "language": language, "snr_db": "" if snr is None else snr, "reference": text})
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} clips → {args.out}/manifest.csv")


if __name__ == "__main__":
    main()
