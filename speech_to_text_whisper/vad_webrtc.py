import webrtcvad
import numpy as np
from scipy.signal import resample_poly
from .speech_config import SAMPLE_RATE, INIT_WEBRTC_SENSITIVITY


class WebRTCVAD:
    """
    Handles voice activity detection using the WebRTC VAD model. It processes audio chunks
    to determine if they contain speech.
    """

    def __init__(self, sensitivity: int = INIT_WEBRTC_SENSITIVITY, state_callback=None):
        """
        Args:
            webrtc_sensitivity (int): WebRTC VAD mode sensitivity (0-3).
        """

        self.webrtc_vad_model = webrtcvad.Vad()
        self.webrtc_vad_model.set_mode(sensitivity)
        self.state_callback = state_callback

    def is_voice_active(self, audio_chunk: bytes, all_frames_must_be_true: bool = False, sample_rate: int = SAMPLE_RATE) -> bool:
        """
        Checks if the given audio chunk contains speech.

        Args:
            audio_chunk (bytes): Audio data in raw PCM format.
            sample_rate (int): Sample rate of the audio data.
        """
        if sample_rate != 16000:
            audio_chunk = self._resample_audio(audio_chunk, sample_rate, 16000)

        # Number of audio frames per millisecond
        frame_length = int(16000 * 0.01)  # for 10ms frame
        num_frames = int(len(audio_chunk) / (2 * frame_length))
        speech_frames = 0

        for i in range(num_frames):
            start_byte = i * frame_length * 2
            end_byte = start_byte + frame_length * 2
            frame = audio_chunk[start_byte:end_byte]
            if self.webrtc_vad_model.is_speech(frame, 16000):
                speech_frames += 1
                if not all_frames_must_be_true:
                    if self.state_callback:
                        self.state_callback(is_speech_active=True)
                    return True
        if all_frames_must_be_true:
            speech_detected = speech_frames == num_frames
            self.is_webrtc_speech_active = speech_detected
            if self.state_callback:
                self.state_callback(is_speech_active=speech_detected)
            return speech_detected
        else:
            if self.state_callback:
                self.state_callback(is_speech_active=False)
            return False


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
