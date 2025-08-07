import torch
import numpy as np
from scipy.signal import resample_poly
from .speech_config import INT16_MAX_ABS_VALUE, SAMPLE_RATE, INIT_SILERO_SENSITIVITY


class SileroVAD:
    """
    Silero-based Voice Activity Detection (VAD).
    """

    def __init__(self, sensitivity: float = INIT_SILERO_SENSITIVITY,
                 model_repo: str = "snakers4/silero-vad",
                 silero_use_onnx: bool = False,
                 state_callback=None):
        """
        Initialize Silero VAD.

        Args:
            sensitivity (float): VAD sensitivity (higher values are less aggressive).
            model_repo (str): Silero model repository for loading the VAD model.
        """
        self.silero_sensitivity = sensitivity
        self.silero_vad_model, _ = torch.hub.load(repo_or_dir=model_repo, model="silero_vad",
                                                  verbose=False, onnx=silero_use_onnx)
        self.state_callback = state_callback

    def is_voice_active(self, audio_chunk: bytes, sample_rate: int = SAMPLE_RATE) -> bool:
        """
        Check if speech is present in the audio chunk.

        Args:
            audio_chunk (bytes): Raw PCM audio chunk.
            sample_rate (int): Sample rate of the incoming audio.

        Returns:
            bool: True if speech is detected, False otherwise.
        """
        if self.state_callback:
            self.state_callback(working=True, is_speech_active=False)

        if sample_rate != 16000:  # Silero requires 16kHz
            audio_chunk = self._resample_audio(audio_chunk, sample_rate, 16000)

        audio_float = self._to_float(audio_chunk)
        vad_probability = self.silero_vad_model(torch.from_numpy(audio_float), sample_rate).item()
        is_active = vad_probability > (1 - self.silero_sensitivity)

        # Update `is_silero_speech_active` state in parent class
        if self.state_callback:
            self.state_callback(working=False, is_speech_active=is_active)

        return is_active

    @staticmethod
    def _resample_audio(audio_chunk: bytes, original_rate: int, target_rate: int) -> bytes:
        """
        Resample audio to the target sample rate.

        Args:
            audio_chunk (bytes): Raw PCM audio data.
            original_rate (int): Original sample rate.
            target_rate (int): Target sample rate.

        Returns:
            bytes: Resampled audio chunk.
        """
        pcm_data = np.frombuffer(audio_chunk, dtype=np.int16)
        resampled_data = resample_poly(pcm_data, target_rate, original_rate)
        return resampled_data.astype(np.int16).tobytes()

    @staticmethod
    def _to_float(audio_chunk: bytes) -> np.ndarray:
        """
        Convert raw PCM audio to normalized float32.

        Args:
            audio_chunk (bytes): Raw PCM audio chunk.

        Returns:
            np.ndarray: Float32 audio data normalized to [-1, 1].
        """
        pcm_data = np.frombuffer(audio_chunk, dtype=np.int16)
        return pcm_data.astype(np.float32) / INT16_MAX_ABS_VALUE

    def reset_states(self):
        self.silero_vad_model.reset_states()