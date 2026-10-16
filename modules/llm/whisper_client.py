"""
Speech-to-text for speech segments from the microphone controller.

Backends (``audio.mic.whisper_backend``):
  * ``azure`` — Azure OpenAI Whisper (same client as ``speech_to_text_whisper.transcription_worker``):
    ``OpenAI(api_key=AZURE_OPENAI_API_KEY, base_url=AZURE_OPENAI_ENDPOINT)``, model ``llm.whisper_deployment``.
    Falls back to the local model automatically: without keys it is local from the start; when an Azure request
    fails (offline, auth, timeout) that segment is transcribed locally and Azure is retried after
    ``audio.mic.azure_retry_sec``;
  * ``local`` — only faster-whisper on the device (``audio.mic.local_model``, e.g. ``base``), loaded on first use;
  * ``none`` — no transcription (segments are only counted), for tests.

Results carry ``backend`` ("azure" / "local") so the events show which model produced the text.

Only the transcript text goes further; audio never leaves the device except to the configured Whisper backend.
"""
import io
import threading
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import soundfile as sf

from system_setup.logger_setup import init_logger

logger = init_logger()

Transcriber = Callable[[np.ndarray, int], Dict[str, Any]]


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def azure_transcriber(config: Dict[str, Any]) -> Transcriber:
    from openai import OpenAI

    llm = config.get("llm", {})
    if not llm.get("azure_api_key") or not llm.get("azure_endpoint"):
        raise RuntimeError("Azure Whisper needs AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT (.env)")
    # Few retries / a short timeout: on failure the local model takes over instead of waiting
    client = OpenAI(api_key=llm["azure_api_key"], base_url=llm["azure_endpoint"],
                    timeout=float(llm.get("request_timeout", 30)), max_retries=int(llm.get("max_retries", 1)))
    model = llm.get("whisper_deployment", "whisper-1")
    language = config.get("audio", {}).get("mic", {}).get("language")

    def transcribe(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        kwargs = {"language": language} if language else {}
        result = client.audio.transcriptions.create(model=model, file=("speech.wav", _wav_bytes(audio, sample_rate)),
                                                    response_format="verbose_json", **kwargs)
        return {"text": (getattr(result, "text", "") or "").strip(),
                "language": getattr(result, "language", None) or language, "backend": "azure"}

    return transcribe


def local_transcriber(config: Dict[str, Any]) -> Transcriber:
    mic = config.get("audio", {}).get("mic", {})
    state: Dict[str, Any] = {}
    lock = threading.Lock()

    def model():
        with lock:  # warm_up() and the first segment may race
            if "model" not in state:
                from faster_whisper import WhisperModel

                name = mic.get("local_model", "base")
                started = time.monotonic()
                state["model"] = WhisperModel(name, device="cpu", compute_type="int8")
                logger.info(f"Local Whisper '{name}' loaded in {time.monotonic() - started:.1f} s")
        return state["model"]

    def transcribe(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        samples = audio.astype(np.float32) / 32768.0
        segments, info = model().transcribe(samples, language=mic.get("language") or None, vad_filter=True)
        return {"text": " ".join(s.text.strip() for s in segments).strip(), "language": info.language,
                "backend": "local"}

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
    if backend == "local":
        return local_transcriber(config)
    if backend == "none":
        return lambda audio, sample_rate: {"text": "", "language": None, "backend": "none"}
    raise ValueError(f"Unknown audio.mic.whisper_backend {backend!r} (azure | local | none)")
