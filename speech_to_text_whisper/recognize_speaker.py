import torch
import torchaudio
import os
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from speechbrain.inference.speaker import SpeakerRecognition
from .speech_config import COSINE_SIMILARITY_THRESHOLD


class SpeakerRecognizer:
    def __init__(self, model_type="speechbrain", speaker_db_path="speech_to_text_whisper/voices"):
        """
        Initializes the SpeakerRecognizer class.

        :param model_type: "wav2vec2" or "speechbrain"
        :param speaker_db_path: Path to folder containing known speaker .wav files
        """
        self.model_type = model_type.lower()
        self.speaker_db_path = speaker_db_path
        self.speaker_embeddings = {}

        # Load the selected model
        if self.model_type == "wav2vec2":
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        elif self.model_type == "speechbrain":
            self.model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="model/speechbrain"
            )
        else:
            raise ValueError("Invalid model type. Choose 'wav2vec2' or 'speechbrain'.")

        # Load all known speaker embeddings
        self.load_speaker_embeddings()

    def extract_embedding(self, audio_array, sampling_rate=16000):
        """
        Extracts speaker embedding from a numpy audio array.

        :param audio_array: NumPy array containing audio samples
        :param sampling_rate: Sampling rate of the audio
        :return: Speaker embedding tensor
        """
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)

        if self.model_type == "wav2vec2":
            # Process input for Wav2Vec2
            inputs = self.processor(audio_tensor, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)  # Mean pooling for embedding

        elif self.model_type == "speechbrain":
            return self.model.encode_batch(audio_tensor.unsqueeze(0)).squeeze(0)  # Add batch, extract embedding

    def load_speaker_embeddings(self):
        """Loads all speaker .wav files and computes embeddings."""
        self.speaker_embeddings.clear()

        for file in os.listdir(self.speaker_db_path):
            if file.endswith(".wav"):
                speaker_name = os.path.splitext(file)[0]  # Get filename without extension
                file_path = os.path.join(self.speaker_db_path, file)

                # Load .wav file
                waveform, sr = torchaudio.load(file_path)
                waveform = waveform.mean(dim=0).numpy()  # Convert to 1D NumPy

                # Extract embedding
                embedding = self.extract_embedding(waveform, sampling_rate=sr)
                self.speaker_embeddings[speaker_name] = embedding

        print(f"🔹 Loaded {len(self.speaker_embeddings)} speaker embeddings.")

    def identify_speaker(self, real_time_audio_array):
        """
        Compares real-time audio against stored speaker embeddings.

        :param real_time_audio_array: NumPy array of real-time audio
        :return: Identified speaker name or "Unknown Speaker"
        """
        real_time_embedding = self.extract_embedding(real_time_audio_array)

        results = []
        for speaker, embedding in self.speaker_embeddings.items():
            similarity = torch.nn.functional.cosine_similarity(real_time_embedding, embedding).item()
            results.append((speaker, similarity))

        # Sort results by highest similarity
        results.sort(key=lambda x: x[1], reverse=True)

        # Select the best match
        best_match, best_score = results[0]

        # Check against threshold
        if best_score > COSINE_SIMILARITY_THRESHOLD:
            return f"Speaker Identified: {best_match} (Score: {best_score:.4f})"
        else:
            return f"Unknown Speaker (Best Match: {best_match}, Score: {best_score:.4f})"

