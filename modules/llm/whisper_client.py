"""
Speech-to-text for speech segments from the microphone controller.

Backends (``audio.mic.whisper_backend``):
  * ``azure`` — Azure OpenAI Whisper (client from ``modules.llm.azure_client``, shared with GPT),
    model ``llm.whisper_deployment``.
    Falls back to the local model automatically: without keys it is local from the start; when an Azure request
    fails (offline, auth, timeout) that segment is transcribed locally and Azure is retried after
    ``audio.mic.azure_retry_sec``;
  * ``azure_only`` — only the cloud: no local fallback (a failed request leaves that segment untranscribed; without
    keys the module logs an error and transcribes nothing);
  * ``local`` — only faster-whisper on the device (``audio.mic.local_model``, e.g. ``base``), loaded on first use;
  * ``none`` — no transcription (segments are only counted), for tests.

Results carry ``backend`` ("azure" / "local") so the events show which model produced the text, and ``segments``
``[{"start", "end", "text"}]`` (seconds from the chunk start; Azure ``verbose_json`` / faster-whisper) for the
speaker attribution (F10).

Only the transcript text goes further; audio never leaves the device except to the configured Whisper backend.
"""
import io
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from system_setup.logger_setup import init_logger

logger = init_logger()

Transcriber = Callable[[np.ndarray, int], Dict[str, Any]]
# Azure returns language names ("english"), faster-whisper ISO codes ("en"): events use the codes
LANGUAGE_CODES = {"english": "en", "ukrainian": "uk", "russian": "ru", "polish": "pl", "german": "de",
                  "french": "fr", "spanish": "es", "italian": "it", "portuguese": "pt", "czech": "cs",
                  "slovak": "sk", "romanian": "ro", "hungarian": "hu", "turkish": "tr", "chinese": "zh",
                  "japanese": "ja", "korean": "ko", "arabic": "ar"}


def language_code(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    return LANGUAGE_CODES.get(language.lower(), language.lower())


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@dataclass
class WhisperOptions:
    """Accuracy settings shared by both backends (``audio.mic`` config section)."""
    language: Optional[str] = None              # fixed language, e.g. "uk"
    languages: Tuple[str, ...] = ()             # otherwise: detect, but only among these (e.g. uk, en)
    beam_size: int = 5
    initial_prompt: Optional[str] = None        # names / domain words, e.g. "Oleh, датчик газу"
    condition_on_previous_text: bool = False    # True makes Whisper repeat itself after a mistake
    no_speech_max: float = 0.6                  # segment dropped if no_speech_prob above this …
    logprob_min: float = -1.0                   # … and avg_logprob below this (Whisper's silence rule)
    compression_max: float = 2.4                # repetitive text ("I don't know, I don't know, …") is dropped
    hallucination_phrases: Tuple[str, ...] = ()
    cpu_threads: int = 4                        # Raspberry Pi: 4 cores
    compute_type: str = "int8"

    @classmethod
    def from_mic(cls, mic: Dict[str, Any]) -> "WhisperOptions":
        names = {"whisper_beam_size": "beam_size", "whisper_prompt": "initial_prompt",
                 "whisper_condition_on_previous_text": "condition_on_previous_text",
                 "whisper_no_speech_max": "no_speech_max", "whisper_logprob_min": "logprob_min",
                 "whisper_compression_max": "compression_max", "whisper_cpu_threads": "cpu_threads",
                 "whisper_compute_type": "compute_type", "language": "language", "languages": "languages",
                 "hallucination_phrases": "hallucination_phrases"}
        opts = cls(**{field: mic[key] for key, field in names.items() if mic.get(key) is not None})
        opts.languages = tuple(code.lower() for code in opts.languages or ())
        opts.hallucination_phrases = tuple(opts.hallucination_phrases or DEFAULT_HALLUCINATIONS)
        return opts

    def allowed(self, code: Optional[str]) -> bool:
        return not self.languages or code in self.languages


# Phrases Whisper invents on silence / noise (subtitle credits from its training data)
DEFAULT_HALLUCINATIONS = (
    "дякую за перегляд", "дякуємо за перегляд", "субтитри зроблені", "підписуйтесь на канал",
    "спасибо за просмотр", "продолжение следует", "субтитры сделал", "субтитры создавал", "редактор субтитров",
    "thanks for watching", "thank you for watching", "please subscribe", "subtitles by",
)


def _normalized(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _segments(raw, opts: Optional[WhisperOptions] = None) -> List[Dict[str, Any]]:
    """
    Azure segment objects / dicts or faster-whisper segments → [{"start", "end", "text"}], without hallucinations:
    silence (high no_speech_prob with low avg_logprob), repetitive text (high compression ratio), known phrases.
    """
    opts = opts or WhisperOptions(hallucination_phrases=DEFAULT_HALLUCINATIONS)
    out = []
    for s in raw or []:
        get = s.get if isinstance(s, dict) else lambda key, s=s: getattr(s, key, None)
        text = (get("text") or "").strip()
        if not text or get("start") is None or get("end") is None:
            continue
        no_speech, logprob, compression = get("no_speech_prob"), get("avg_logprob"), get("compression_ratio")
        if no_speech is not None and logprob is not None and no_speech > opts.no_speech_max \
                and logprob < opts.logprob_min:
            continue
        if compression is not None and compression > opts.compression_max:
            continue
        if any(phrase in _normalized(text) for phrase in opts.hallucination_phrases):
            continue
        out.append({"start": round(float(get("start")), 3), "end": round(float(get("end")), 3), "text": text})
    return out


def _result(segments: List[Dict[str, Any]], language: Optional[str], backend: str) -> Dict[str, Any]:
    return {"text": " ".join(s["text"] for s in segments).strip(), "language": language, "backend": backend,
            "segments": segments}


def azure_transcriber(config: Dict[str, Any], client=None) -> Transcriber:
    """:param client: OpenAI / AzureOpenAI client (tests); created from ``llm`` by default"""
    from modules.llm.azure_client import create_client

    llm = config.get("llm", {})
    client = client or create_client(llm)
    model = llm.get("whisper_deployment", "whisper-1")
    opts = WhisperOptions.from_mic(config.get("audio", {}).get("mic", {}))
    fixed = opts.language or (opts.languages[0] if len(opts.languages) == 1 else None)

    def request(wav: bytes, language: Optional[str]):
        kwargs = {"language": language} if language else {}
        if opts.initial_prompt:
            kwargs["prompt"] = opts.initial_prompt
        return client.audio.transcriptions.create(model=model, file=("speech.wav", wav), response_format="verbose_json",
                                                  temperature=0, **kwargs)

    def transcribe(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        wav = _wav_bytes(audio, sample_rate)
        result = request(wav, fixed)
        language = language_code(getattr(result, "language", None) or fixed)
        if not fixed and not opts.allowed(language):  # e.g. Ukrainian heard as Russian: ask again in the main one
            result, language = request(wav, opts.languages[0]), opts.languages[0]
        return _result(_segments(getattr(result, "segments", None), opts), language, "azure")

    return transcribe


def local_transcriber(config: Dict[str, Any], whisper_model=None) -> Transcriber:
    """:param whisper_model: faster_whisper.WhisperModel-like object (tests); loaded on first use by default"""
    mic = config.get("audio", {}).get("mic", {})
    opts = WhisperOptions.from_mic(mic)
    state: Dict[str, Any] = {"model": whisper_model} if whisper_model is not None else {}
    lock = threading.Lock()

    def model():
        with lock:  # warm_up() and the first segment may race
            if "model" not in state:
                from faster_whisper import WhisperModel

                name = mic.get("local_model", "base")
                started = time.monotonic()
                state["model"] = WhisperModel(name, device="cpu", compute_type=opts.compute_type,
                                              cpu_threads=opts.cpu_threads)
                logger.info(f"Local Whisper '{name}' loaded in {time.monotonic() - started:.1f} s")
        return state["model"]

    def pick_language(samples: np.ndarray) -> Optional[str]:
        if opts.language:
            return opts.language
        if len(opts.languages) == 1:
            return opts.languages[0]
        if not opts.languages:
            return None
        _, _, probabilities = model().detect_language(samples)
        allowed = [(p, code) for code, p in probabilities if code in opts.languages]
        return max(allowed)[1] if allowed else opts.languages[0]

    def transcribe(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        samples = audio.astype(np.float32) / 32768.0
        language = pick_language(samples)
        segments, info = model().transcribe(
            samples, language=language, beam_size=opts.beam_size, vad_filter=True,
            condition_on_previous_text=opts.condition_on_previous_text, initial_prompt=opts.initial_prompt,
            no_speech_threshold=opts.no_speech_max, log_prob_threshold=opts.logprob_min,
            compression_ratio_threshold=opts.compression_max)
        return _result(_segments(list(segments), opts), info.language, "local")

    transcribe.warm_up = model
    return transcribe


class AzureWithLocalFallback:
    """Azure first; local faster-whisper when Azure is not configured or a request fails."""

    def __init__(self, config: Dict[str, Any], azure_factory=None, local_factory=None,
                 clock: Callable[[], float] = time.monotonic):
        self.retry_sec = float(config.get("audio", {}).get("mic", {}).get("azure_retry_sec", 300))
        self.clock = clock
        self.local = (local_factory or local_transcriber)(config)
        self.azure: Optional[Transcriber] = None
        self.azure_failed_at: Optional[float] = None
        try:
            self.azure = (azure_factory or azure_transcriber)(config)
        except Exception as e:
            logger.warning(f"Azure Whisper not available ({e}); using the local model")

    @property
    def using_azure(self) -> bool:
        return self.azure is not None and (self.azure_failed_at is None
                                           or self.clock() - self.azure_failed_at >= self.retry_sec)

    def warm_up(self) -> None:
        """Loads the local model in advance when it is going to be used."""
        if not self.using_azure and hasattr(self.local, "warm_up"):
            self.local.warm_up()

    def __call__(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        if self.using_azure:
            try:
                result = self.azure(audio, sample_rate)
                if self.azure_failed_at is not None:
                    logger.warning("Azure Whisper is reachable again")
                self.azure_failed_at = None
                return {**result, "backend": "azure"}
            except Exception as e:
                self.azure_failed_at = self.clock()
                logger.warning(f"Azure Whisper failed ({type(e).__name__}: {e}); using the local model, "
                               f"retrying Azure in {self.retry_sec:.0f} s")
        return {**self.local(audio, sample_rate), "backend": "local"}


def create_transcriber(config: Dict[str, Any]) -> Transcriber:
    """
    :param config: Full merged config
    :return: transcribe(audio_int16, sample_rate) -> {"text", "language"}
    """
    backend = config.get("audio", {}).get("mic", {}).get("whisper_backend", "azure")
    if backend == "azure":
        return AzureWithLocalFallback(config)
    if backend == "azure_only":
        azure = azure_transcriber(config)  # raises MissingCredentials without keys
        return lambda audio, sample_rate: {**azure(audio, sample_rate), "backend": "azure"}
    if backend == "local":
        return local_transcriber(config)
    if backend == "none":
        return lambda audio, sample_rate: {"text": "", "language": None, "backend": "none"}
    raise ValueError(f"Unknown audio.mic.whisper_backend {backend!r} (azure | azure_only | local | none)")
