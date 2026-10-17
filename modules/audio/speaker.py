"""
Speaker identification for speech segments (F9 / TOD-20): SpeechBrain ECAPA (192-d) voice embeddings matched
against the unified identity registry, so a voice and a face can be the same person.

An unmatched voice becomes a voice-only ``unknown_N`` person; ``core.cooccurrence.VoiceFaceLinker`` later merges it
into the face identity of the person who is alone in view while speaking. Voice embeddings stay on the device.
"""
from typing import Any, Callable, Dict, Optional

import numpy as np

from core.identity import IdentityRegistry
from system_setup.logger_setup import init_logger

logger = init_logger()

Embedder = Callable[[np.ndarray, int], np.ndarray]
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


def ecapa_embedder(savedir: str = "model/speechbrain") -> Embedder:
    """
    :param savedir: SpeechBrain model folder (downloaded on first use, ~80 MB)
    :return: (float32 mono audio in [-1, 1], sample rate) -> 192-d embedding
    """
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    model = EncoderClassifier.from_hparams(source=ECAPA_SOURCE, savedir=savedir, run_opts={"device": "cpu"})

    def embed(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        signal = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        if sample_rate != 16000:
            import torchaudio

            signal = torchaudio.functional.resample(signal, sample_rate, 16000)
        with torch.no_grad():
            return model.encode_batch(signal.unsqueeze(0)).reshape(-1).numpy()
    return embed


def to_float(audio: np.ndarray) -> np.ndarray:
    """int16 samples → float32 in [-1, 1] (float input is returned as float32)."""
    if np.issubdtype(audio.dtype, np.integer):
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


class SpeakerIdentifier:
    def __init__(self, registry: IdentityRegistry, embedder: Embedder, min_duration_sec: float = 1.0,
                 learn_margin: float = 0.10):
        """
        :param min_duration_sec: Shorter segments carry too little voice to identify (no speaker then)
        :param learn_margin: A matched voice adds the sample to its person only when score ≥ threshold + margin
        """
        self.registry = registry
        self.embedder = embedder
        self.min_duration_sec = min_duration_sec
        self.learn_margin = learn_margin

    def identify(self, audio: np.ndarray, sample_rate: int) -> Optional[Dict[str, Any]]:
        """
        :param audio: int16 or float mono samples of one speech segment
        :return: {"person_id", "speaker", "speaker_score", "new_voice"} or None (segment too short)
        """
        duration = len(audio) / sample_rate
        if duration < self.min_duration_sec:
            return None
        registry = self.registry
        registry.poll()  # renames / merges / links done by other processes
        vector = self.embedder(to_float(audio), sample_rate)
        person_id, score = registry.match_voice(vector)
        new = person_id is None
        if new:
            person_id = registry.create_unknown("voice", vector, duration_sec=duration)
        elif score >= registry.voice_threshold + self.learn_margin:
            registry.add_embedding(person_id, "voice", vector, duration_sec=duration)
        registry.touch(person_id)
        return {"person_id": person_id, "speaker": registry.name(person_id), "speaker_score": round(score, 3),
                "new_voice": new}
