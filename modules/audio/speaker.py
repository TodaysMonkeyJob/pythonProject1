"""
Speaker identification for speech chunks (F9 / F10, TOD-20 / TOD-21): who spoke when inside one chunk, matched against
the unified identity registry so a voice and a face can be the same person.

Pipeline for one VAD chunk (≤ 30 s):

1. preprocess — mono, resample to 16 kHz, 80 Hz high-pass (room rumble), loudness normalisation;
2. sliding windows (``speaker_window_sec`` 1.5 s, hop 0.75 s); windows with < ``speaker_min_speech_sec`` (0.8 s) of
   VAD speech are skipped;
3. one batched SpeechBrain ECAPA call (192-d embeddings; the encoder is loaded once per process, MPS on a Mac if it
   works, else CPU);
4. local diarization — agglomerative clustering (cosine, ``speaker_cluster_distance``) → speaker turns; a window that
   is about as close to two clusters (overlapping speech) or far from its own is marked ambiguous;
5. identification per cluster: centroid → ``registry.rank()``; score ≥ ``voice_threshold`` → that person, else a new
   voice-only ``unknown_N``; clusters with < ``min_speaker_sec`` of clean speech, or a near tie between two people,
   are ``uncertain`` and never written to the registry;
6. enrichment — a single-speaker chunk with ≥ ``enrich_min_sec`` of speech and SNR ≥ ``enrich_min_snr_db`` adds its
   centroid to the person, so recognition improves over time.

Voice embeddings never leave the device.
"""
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from core.identity import IdentityRegistry, normalize
from system_setup.logger_setup import init_logger

logger = init_logger()

SAMPLE_RATE = 16000
FRAME_SEC = 0.03
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
KNOWN, UNKNOWN, UNCERTAIN = "known", "unknown", "uncertain"

BatchEmbedder = Callable[[np.ndarray], np.ndarray]     # (n, samples) float32 at 16 kHz -> (n, 192)
SpeechMask = Callable[[np.ndarray], np.ndarray]        # float32 16 kHz -> bool per 30 ms frame

_ENCODERS: Dict[str, Any] = {}


def pick_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    try:
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return "cpu"


def ecapa_batch_embedder(savedir: str = "model/speechbrain", device: str = "auto",
                         batch_size: int = 8) -> BatchEmbedder:
    """
    :param savedir: SpeechBrain model folder (downloaded on first use, ~80 MB)
    :param device: auto (MPS if it works, else CPU) / cpu / mps / cuda
    :param batch_size: windows per forward pass — large CPU batches are disproportionately slow (Mac CPU: 8 windows
                       0.2 s, 39 windows at once 19 s)
    :return: windows (n, samples) float32 at 16 kHz -> (n, 192) embeddings; the model is cached per process
    """
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    def load(dev: str):
        key = f"{savedir}|{dev}"
        if key not in _ENCODERS:
            _ENCODERS[key] = EncoderClassifier.from_hparams(source=ECAPA_SOURCE, savedir=savedir,
                                                            run_opts={"device": dev})
        return _ENCODERS[key]

    dev = pick_device(device)
    model = load(dev)
    if dev != "cpu":
        try:  # some ops are not implemented on MPS in every torch version
            with torch.no_grad():
                model.encode_batch(torch.zeros(1, SAMPLE_RATE))
        except Exception as e:
            logger.warning(f"ECAPA on {dev} failed ({e}); using the CPU")
            dev, model = "cpu", load("cpu")

    def embed(windows: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = torch.from_numpy(np.ascontiguousarray(windows[i:i + batch_size], dtype=np.float32)).to(dev)
                out.append(model.encode_batch(batch).reshape(len(batch), -1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 192), np.float32)
    embed.device = dev
    return embed


def ecapa_embedder(savedir: str = "model/speechbrain", device: str = "auto") -> Callable[[np.ndarray, int], np.ndarray]:
    """(audio, sample rate) -> one 192-d embedding of the whole (preprocessed) clip — enrollment / calibration."""
    batch = ecapa_batch_embedder(savedir, device)
    return lambda audio, sample_rate: batch(preprocess(audio, sample_rate)[None, :])[0]


def to_float(audio: np.ndarray) -> np.ndarray:
    """int16 samples → float32 in [-1, 1] (float input is returned as float32); stereo → mono."""
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1 if audio.shape[1] <= 8 else 0)
    if np.issubdtype(audio.dtype, np.integer):
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


def preprocess(audio: np.ndarray, sample_rate: int, highpass_hz: float = 80.0,
               target_dbfs: float = -20.0) -> np.ndarray:
    """
    :return: float32 mono 16 kHz, high-passed and RMS-normalised to target_dbfs (peaks limited to ±1)
    """
    import torch
    import torchaudio.functional as AF

    signal = torch.from_numpy(to_float(audio))
    if sample_rate != SAMPLE_RATE:
        signal = AF.resample(signal, sample_rate, SAMPLE_RATE)
    if highpass_hz and signal.numel() > 32:
        signal = AF.highpass_biquad(signal, SAMPLE_RATE, highpass_hz)
    out = signal.numpy().astype(np.float32)
    rms = float(np.sqrt(np.mean(out ** 2))) if out.size else 0.0
    if rms > 1e-6:
        out = out * (10 ** (target_dbfs / 20) / rms)
        peak = float(np.max(np.abs(out)))
        if peak > 1.0:
            out = out / peak
    return out


def webrtc_speech_mask(aggressiveness: int = 2) -> SpeechMask:
    import webrtcvad

    vad = webrtcvad.Vad(aggressiveness)
    frame = int(SAMPLE_RATE * FRAME_SEC)

    def mask(audio: np.ndarray) -> np.ndarray:
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        return np.array([vad.is_speech(pcm[i:i + frame].tobytes(), SAMPLE_RATE)
                         for i in range(0, len(pcm) - frame + 1, frame)], bool)
    return mask


def snr_db(audio: np.ndarray, speech: np.ndarray) -> float:
    """Speech vs non-speech frame energy (the quietest 10 % of frames when everything is speech)."""
    frame = int(SAMPLE_RATE * FRAME_SEC)
    count = min(len(speech), len(audio) // frame)
    if count == 0:
        return 0.0
    energy = np.array([np.mean(audio[i * frame:(i + 1) * frame] ** 2) for i in range(count)]) + 1e-12
    mask = speech[:count]
    if mask.all() or not mask.any():
        noise = np.percentile(energy, 10)
        signal = np.percentile(energy, 90)
    else:
        noise, signal = np.mean(energy[~mask]), np.mean(energy[mask])
    return float(10 * math.log10(signal / noise))


def cluster_embeddings(embeddings: np.ndarray, distance_threshold: float,
                       split_similarity: Optional[float] = 0.35) -> np.ndarray:
    """
    Agglomerative clustering, cosine distance, average linkage. A window spanning a speaker change is close to both
    speakers and can chain them into one cluster, so every cluster of ≥ 4 windows is then split in two when the two
    halves' centroids are less similar than ``split_similarity`` (one speaker's windows stay ≥ ~0.5 apart in ECAPA).

    :return: Cluster label per embedding (0, 1, …, in order of first appearance)
    """
    if len(embeddings) <= 1:
        return np.zeros(len(embeddings), int)
    from sklearn.cluster import AgglomerativeClustering

    raw = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                  distance_threshold=distance_threshold).fit_predict(embeddings)
    if split_similarity is not None:
        next_label = int(raw.max()) + 1
        for label in sorted(set(raw.tolist())):
            members = np.flatnonzero(raw == label)
            if len(members) < 4:
                continue
            halves = AgglomerativeClustering(n_clusters=2, metric="cosine",
                                             linkage="average").fit_predict(embeddings[members])
            sizes = np.bincount(halves, minlength=2)
            a, b = (normalize(embeddings[members[halves == h]].mean(axis=0)) for h in (0, 1))
            if sizes.min() >= 2 and float(a @ b) < split_similarity:
                raw[members[halves == 1]] = next_label
                next_label += 1
    order = {label: i for i, label in enumerate(dict.fromkeys(raw.tolist()))}
    return np.array([order[label] for label in raw], int)


@dataclass
class Turn:
    start: float                # seconds from the chunk start
    end: float
    cluster: Optional[int]      # None = ambiguous (overlapping speech) / no usable window


@dataclass
class Speaker:
    person_id: Optional[str]
    name: Optional[str]
    score: float
    status: str                 # known | unknown | uncertain
    duration_sec: float


@dataclass
class Diarization:
    turns: List[Turn] = field(default_factory=list)
    speakers: Dict[int, Speaker] = field(default_factory=dict)
    duration_sec: float = 0.0
    snr_db: float = 0.0

    def speaker(self, cluster: Optional[int]) -> Speaker:
        if cluster is None or cluster not in self.speakers:
            return Speaker(None, None, 0.0, UNCERTAIN, 0.0)
        return self.speakers[cluster]


@dataclass
class SpeakerConfig:
    window_sec: float = 1.5
    hop_sec: float = 0.75
    min_speech_sec: float = 0.8         # per window
    cluster_distance: float = 0.75      # cosine distance; ECAPA 1.5 s windows: same speaker 0.2–0.8, different ≥ 0.85
    split_similarity: float = 0.35      # a cluster whose two halves are less similar is two speakers
    min_speaker_sec: float = 1.0        # shorter clusters → uncertain
    ambiguous_margin: float = 0.08      # a window as close to two clusters (±margin) → overlapping speech
    match_margin: float = 0.05          # two registry people this close → uncertain
    enrich_min_sec: float = 2.0
    enrich_min_snr_db: float = 10.0
    highpass_hz: float = 80.0

    @classmethod
    def from_mic(cls, mic: Dict[str, Any]) -> "SpeakerConfig":
        keys = {f"speaker_{name}": name for name in cls.__dataclass_fields__}
        keys["min_speaker_sec"] = "min_speaker_sec"
        return cls(**{name: mic[key] for key, name in keys.items() if key in mic})


class SpeakerAnalyzer:
    def __init__(self, registry: IdentityRegistry, embed: BatchEmbedder, speech_mask: SpeechMask,
                 cfg: Optional[SpeakerConfig] = None):
        self.registry = registry
        self.embed = embed
        self.speech_mask = speech_mask
        self.cfg = cfg or SpeakerConfig()

    def windows(self, audio: np.ndarray, speech: np.ndarray) -> List[tuple]:
        """:return: [(start_sec, end_sec)] of the windows with enough speech"""
        cfg, duration = self.cfg, len(audio) / SAMPLE_RATE
        starts = [0.0] if duration <= cfg.window_sec else \
            [i * cfg.hop_sec for i in range(int((duration - cfg.window_sec) / cfg.hop_sec) + 1)]
        if duration > cfg.window_sec and starts[-1] + cfg.window_sec < duration - cfg.hop_sec / 2:
            starts.append(duration - cfg.window_sec)          # cover the tail
        result = []
        for start in starts:
            end = min(duration, start + cfg.window_sec)
            frames = speech[int(start / FRAME_SEC):int(end / FRAME_SEC)]
            if frames.sum() * FRAME_SEC >= cfg.min_speech_sec:
                result.append((start, end))
        return result

    def analyze(self, audio: np.ndarray, sample_rate: int) -> Diarization:
        """
        :param audio: One speech chunk (int16 or float, any rate)
        :return: Speaker turns (seconds from the chunk start) and who each cluster is
        """
        cfg = self.cfg
        clean = preprocess(audio, sample_rate, cfg.highpass_hz)
        speech = self.speech_mask(clean)
        result = Diarization(duration_sec=len(clean) / SAMPLE_RATE, snr_db=snr_db(clean, speech))
        spans = self.windows(clean, speech)
        if not spans:
            result.turns = [Turn(0.0, result.duration_sec, None)]
            return result
        size = int(cfg.window_sec * SAMPLE_RATE)
        pieces = [clean[int(a * SAMPLE_RATE):int(a * SAMPLE_RATE) + size] for a, _ in spans]
        batch = np.stack([np.pad(piece, (0, size - len(piece))) for piece in pieces])
        embeddings = np.array([normalize(e) for e in self.embed(batch)])
        labels = cluster_embeddings(embeddings, cfg.cluster_distance, cfg.split_similarity)
        centroids = {c: normalize(embeddings[labels == c].mean(axis=0)) for c in set(labels.tolist())}

        # Overlapping speech: about as close to another cluster as to its own
        ambiguous = np.zeros(len(spans), bool)
        if len(centroids) > 1:
            for i, e in enumerate(embeddings):
                own = float(centroids[labels[i]] @ e)
                other = max(float(centroids[c] @ e) for c in centroids if c != labels[i])
                ambiguous[i] = own - other < cfg.ambiguous_margin
        result.turns = self._turns(spans, labels, ambiguous, result.duration_sec)

        speech_sec = {c: 0.0 for c in centroids}  # VAD speech inside the cluster's turns, not the turn length
        for turn in result.turns:
            if turn.cluster is not None:
                frames = speech[int(turn.start / FRAME_SEC):int(turn.end / FRAME_SEC)]
                speech_sec[turn.cluster] += float(frames.sum()) * FRAME_SEC
        taken: set = set()  # two clusters of one chunk are two different people
        for c in sorted(centroids, key=lambda c: -speech_sec[c]):
            centroid, clean_windows = centroids[c], embeddings[(labels == c) & ~ambiguous]
            if len(clean_windows):
                centroid = normalize(clean_windows.mean(axis=0))
            result.speakers[c] = self._identify(centroid, speech_sec[c], single=len(centroids) == 1,
                                                snr=result.snr_db, taken=taken)
            if result.speakers[c].person_id:
                taken.add(result.speakers[c].person_id)
        return result

    def _turns(self, spans, labels, ambiguous, duration: float) -> List[Turn]:
        """Each window owns the stretch around its centre; neighbours of the same cluster merge into one turn."""
        centres = [(a + b) / 2 for a, b in spans]
        turns: List[Turn] = []
        for i, (label, unsure) in enumerate(zip(labels, ambiguous, strict=True)):
            start = 0.0 if i == 0 else (centres[i - 1] + centres[i]) / 2
            end = duration if i == len(spans) - 1 else (centres[i] + centres[i + 1]) / 2
            cluster = None if unsure else int(label)
            if turns and turns[-1].cluster == cluster:
                turns[-1].end = end
            else:
                turns.append(Turn(start, end, cluster))
        return turns

    def _identify(self, centroid: np.ndarray, seconds: float, single: bool, snr: float,
                  taken: Optional[set] = None) -> Speaker:
        cfg, registry = self.cfg, self.registry
        if seconds < cfg.min_speaker_sec:
            return Speaker(None, None, 0.0, UNCERTAIN, round(seconds, 2))
        registry.poll()  # renames / merges / links done by other processes
        ranked = [r for r in registry.rank("voice", centroid, k=2 + len(taken or ())) if r[0] not in (taken or ())][:2]
        best_id, best = ranked[0] if ranked else (None, 0.0)
        if best_id is not None and best >= registry.voice_threshold:
            if len(ranked) > 1 and ranked[1][1] >= registry.voice_threshold and best - ranked[1][1] < cfg.match_margin:
                return Speaker(None, None, round(best, 3), UNCERTAIN, round(seconds, 2))   # two people this close
            if single and seconds >= cfg.enrich_min_sec and snr >= cfg.enrich_min_snr_db:
                registry.add_embedding(best_id, "voice", centroid, duration_sec=seconds)
            person_id = best_id
        else:
            person_id = registry.create_unknown("voice", centroid, duration_sec=seconds)
        registry.touch(person_id)
        status = KNOWN if registry.is_known(person_id) else UNKNOWN
        return Speaker(person_id, registry.name(person_id), round(best, 3), status, round(seconds, 2))


def analyzer_from_config(registry: IdentityRegistry, mic: Dict[str, Any],
                         embed: Optional[BatchEmbedder] = None,
                         speech_mask: Optional[SpeechMask] = None) -> SpeakerAnalyzer:
    """:param mic: The ``audio.mic`` config section"""
    embed = embed or ecapa_batch_embedder(mic.get("speaker_model_dir", "model/speechbrain"),
                                          mic.get("speaker_device", "auto"), int(mic.get("speaker_batch_size", 8)))
    speech_mask = speech_mask or webrtc_speech_mask(int(mic.get("vad_aggressiveness", 2)))
    return SpeakerAnalyzer(registry, embed, speech_mask, SpeakerConfig.from_mic(mic))


def describe(diarization: Diarization) -> Sequence[str]:
    """Debug lines: turns with speakers."""
    return [f"{t.start:5.2f}-{t.end:5.2f} {diarization.speaker(t.cluster).name or '?'}" for t in diarization.turns]
