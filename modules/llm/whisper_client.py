"""
Speech-to-text for speech segments from the microphone controller.

Backends (``audio.mic.whisper_backend``):
  * ``azure`` — Azure OpenAI Whisper (same client as ``speech_to_text_whisper.transcription_worker``):
    ``OpenAI(api_key=AZURE_OPENAI_API_KEY, base_url=AZURE_OPENAI_ENDPOINT)``, model ``llm.whisper_deployment``;
  * ``local`` — faster-whisper on the device (``audio.mic.local_model``, e.g. ``base``), loaded on first use;
  * ``none`` — no transcription (segments are only counted), for tests / offline setups.

Only the transcript text goes further; audio never leaves the device except to the configured Whisper backend.
"""
import io
from typing import Any, Callable, Dict

import numpy as np
import soundfile as sf

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
    client = OpenAI(api_key=llm["azure_api_key"], base_url=llm["azure_endpoint"],
                    timeout=float(llm.get("request_timeout", 30)))
    model = llm.get("whisper_deployment", "whisper-1")
    language = config.get("audio", {}).get("mic", {}).get("language")

    def transcribe(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        kwargs = {"language": language} if language else {}
        result = client.audio.transcriptions.create(model=model, file=("speech.wav", _wav_bytes(audio, sample_rate)),
                                                    response_format="verbose_json", **kwargs)
        return {"text": (getattr(result, "text", "") or "").strip(),
                "language": getattr(result, "language", None) or language}

    return transcribe


def local_transcriber(config: Dict[str, Any]) -> Transcriber:
    mic = config.get("audio", {}).get("mic", {})
    state: Dict[str, Any] = {}

    def transcribe(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        if "model" not in state:
            from faster_whisper import WhisperModel

            state["model"] = WhisperModel(mic.get("local_model", "base"), device="cpu", compute_type="int8")
        samples = audio.astype(np.float32) / 32768.0
        segments, info = state["model"].transcribe(samples, language=mic.get("language") or None, vad_filter=True)
        return {"text": " ".join(s.text.strip() for s in segments).strip(), "language": info.language}

    return transcribe


def create_transcriber(config: Dict[str, Any]) -> Transcriber:
    """
    :param config: Full merged config
    :return: transcribe(audio_int16, sample_rate) -> {"text", "language"}
    """
    backend = config.get("audio", {}).get("mic", {}).get("whisper_backend", "azure")
    if backend == "azure":
        return azure_transcriber(config)
    if backend == "local":
        return local_transcriber(config)
    if backend == "none":
        return lambda audio, sample_rate: {"text": "", "language": None}
    raise ValueError(f"Unknown audio.mic.whisper_backend {backend!r} (azure | local | none)")
