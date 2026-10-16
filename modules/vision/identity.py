"""
Person identification (F3): OpenCV YuNet face detection + SFace embeddings, cosine similarity against the
local known-people database (``faces.npz``, built by ``tools/enroll_face.py``).

Each person track is identified once; while it is unknown or its face is not visible, recognition is retried
every ``retry_sec``. Unknown people get a session label ``unknown_N`` that sticks to the same face (their
embeddings are cached in memory only). A ``person_identified`` event is published when a track's label
changes::

    {"track_id": 7, "identity": "Oleh", "known": true, "similarity": 0.71, "previous": "not_visible"}

Unknown people are kept in ``identity.unknown_dir`` (face crops + embeddings, see ``identity_store``); renaming
an ``unknown_N`` folder to a name enrolls that person automatically.

Privacy: face images and embeddings never leave the device. Only unknown people's crops are stored (for naming
them later, deleted after ``unknown_retention_days``; ``save_unknowns: false`` disables it), plus debug crops when
``identity.debug_dir`` is set. Only the label goes to the bus.
"""
import os
import time
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.events import Event
from modules.vision.consumer import SharedFrameConsumer
from modules.vision.identity_store import AliasRegistry, UnknownStore
from modules.vision.shared_frame import SharedFrameBuffer
from system_setup.logger_setup import init_logger

logger = init_logger()

NOT_VISIBLE = "not_visible"
EMBEDDING_SIZE = 128


@dataclass
class IdentityConfig:
    """Thresholds; every field can be overridden from the ``identity`` config section."""
    face_threshold: float = 0.45        # cosine ≥ this → known person (SFace reference: 0.363)
    unknown_threshold: float = 0.45     # cosine ≥ this → same unknown person as before
    retry_sec: float = 2.0              # retry not_visible tracks this often
    verify_sec: float = 5.0             # re-check already identified tracks (another person may take the place)
    verify_margin: float = 0.10         # face still "the same person" down to threshold - margin
    switch_confirmations: int = 2       # consecutive different-person checks before a track's label changes
    detection_score: float = 0.8        # YuNet score
    min_face_size: int = 40             # px, smaller faces are not recognised reliably
    face_region: float = 0.6            # face centre must lie in the top part of the person box
    box_padding: float = 0.1            # face centre may lie this much (box fraction) outside the person box
    max_face_offset: float = 0.35       # |face centre − box centre| / box width; larger = someone else's face
    track_lost_sec: float = 10.0
    max_unknowns: int = 200
    debug_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "IdentityConfig":
        data = data or {}
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


class FaceEngine:
    """YuNet detector + SFace recognizer."""

    def __init__(self, detector_model: str, recognizer_model: str, score_threshold: float = 0.8):
        for path in (detector_model, recognizer_model):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Face model not found: {path} (python -m modules.vision.face_models)")
        self.detector = cv2.FaceDetectorYN.create(detector_model, "", (320, 320), score_threshold, 0.3, 50)
        self.recognizer = cv2.FaceRecognizerSF.create(recognizer_model, "")

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        :param image: BGR image
        :return: (n, 15) YuNet rows: x, y, w, h, 5 landmarks (x, y), score
        """
        self.detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = self.detector.detect(image)
        return np.zeros((0, 15), np.float32) if faces is None else faces

    def embed(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        """
        :param image: Image the face row refers to
        :param face: One YuNet row
        :return: L2-normalised 128-d embedding
        """
        aligned = self.recognizer.alignCrop(image, face)
        return normalize(self.recognizer.feature(aligned))

    def aligned_crop(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        return self.recognizer.alignCrop(image, face)


def largest_face(faces: np.ndarray, min_size: int = 0) -> Optional[np.ndarray]:
    """
    :return: The YuNet row with the largest box (at least min_size px wide and high), or None
    """
    candidates = [f for f in faces if f[2] >= min_size and f[3] >= min_size]
    return max(candidates, key=lambda f: f[2] * f[3]) if candidates else None


def face_fit(face: np.ndarray, bbox: Sequence[float], cfg: "IdentityConfig") -> Optional[float]:
    """
    How well a face belongs to a person box (lower = better).

    :return: Cost, or None when the face can't be this person's (outside the head region of the box)
    """
    x1, y1, x2, y2 = bbox
    width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    fx, fy = face[0] + face[2] / 2, face[1] + face[3] / 2
    dx = abs(fx - (x1 + x2) / 2) / width
    dy = (fy - y1) / height
    if dx > cfg.max_face_offset or dy < -cfg.box_padding or dy > cfg.face_region:
        return None
    # Centred, high in the box, and a face size that suits the box (≈ 30 % of an upper-body box width)
    return dx + 0.5 * max(0.0, dy) + 0.2 * abs(float(np.log(max(face[2], 1.0) / width / 0.3)))


def assign_faces(persons: Sequence[Dict[str, Any]], faces: np.ndarray,
                 cfg: "IdentityConfig") -> Dict[int, np.ndarray]:
    """
    One-to-one assignment of detected faces to person boxes (greedy by face_fit cost).

    :param persons: [{"track_id", "bbox"}]
    :param faces: YuNet rows in frame coordinates
    :return: {track_id: face row}
    """
    pairs = []
    for person in persons:
        for index, face in enumerate(faces):
            cost = face_fit(face, person["bbox"], cfg)
            if cost is not None:
                pairs.append((cost, int(person["track_id"]), index))
    assigned, used_faces = {}, set()
    for _cost, track_id, index in sorted(pairs):
        if track_id in assigned or index in used_faces:
            continue
        assigned[track_id] = faces[index]
        used_faces.add(index)
    return assigned


class FaceDatabase:
    """
    Known people: one averaged embedding per name, stored in an .npz file (names, embeddings, counts).
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.names: List[str] = []
        self.embeddings = np.zeros((0, EMBEDDING_SIZE), np.float32)
        self.counts: List[int] = []
        if path and os.path.isfile(path):
            self.load(path)

    def __len__(self) -> int:
        return len(self.names)

    def load(self, path: str) -> None:
        data = np.load(path, allow_pickle=False)
        self.names = [str(n) for n in data["names"]]
        self.embeddings = data["embeddings"].astype(np.float32).reshape(-1, EMBEDDING_SIZE)
        self.counts = [int(c) for c in data["counts"]]

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, names=np.array(self.names), embeddings=self.embeddings, counts=np.array(self.counts))

    def add(self, name: str, embeddings: Sequence[np.ndarray]) -> None:
        """
        Adds embeddings for a person; an existing entry is updated with a count-weighted average.
        """
        if not len(embeddings):
            raise ValueError("No embeddings to add")
        new = np.mean([normalize(e) for e in embeddings], axis=0)
        if name in self.names:
            i = self.names.index(name)
            total = self.counts[i] + len(embeddings)
            self.embeddings[i] = normalize((self.embeddings[i] * self.counts[i] + new * len(embeddings)) / total)
            self.counts[i] = total
        else:
            self.names.append(name)
            self.embeddings = np.vstack([self.embeddings, normalize(new)])
            self.counts.append(len(embeddings))

    def remove(self, name: str) -> bool:
        if name not in self.names:
            return False
        i = self.names.index(name)
        del self.names[i]
        del self.counts[i]
        self.embeddings = np.delete(self.embeddings, i, axis=0)
        return True

    def match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        :return: (best name, cosine similarity) or (None, 0.0) for an empty database
        """
        if not self.names:
            return None, 0.0
        similarities = self.embeddings @ normalize(embedding)
        best = int(np.argmax(similarities))
        return self.names[best], float(similarities[best])


@dataclass
class TrackIdentity:
    identity: Optional[str] = None
    known: bool = False
    similarity: float = 0.0
    last_attempt: float = float("-inf")
    last_seen: float = 0.0
    attempts: int = 0
    different_checks: int = 0           # consecutive verifications that saw another face


@dataclass
class UnknownFace:
    label: str
    embedding: np.ndarray
    count: int = 1


@dataclass
class IdentityEngine:
    """
    Identification logic: feed frames + person boxes, get event payloads back.
    ``faces`` is anything with detect(image) and embed(image, face) (FaceEngine in production).
    With a ``store`` unknown people are kept on disk (``identity_store.UnknownStore``) and renamed folders are
    promoted to known people by ``sync()``; ``aliases`` maps old unknown labels to names.
    """
    faces: Any
    database: FaceDatabase
    cfg: IdentityConfig = field(default_factory=IdentityConfig)
    store: Optional[UnknownStore] = None
    aliases: Optional[AliasRegistry] = None
    tracks: Dict[int, TrackIdentity] = field(default_factory=dict)
    unknowns: List[UnknownFace] = field(default_factory=list)
    unknown_counter: int = 0

    def __post_init__(self) -> None:
        if self.store is not None:
            self.unknowns = [UnknownFace(label, embedding, count) for label, embedding, count in self.store.load()]

    def drop(self, track_id: int) -> None:
        self.tracks.pop(track_id, None)

    def update(self, now: float, frame: np.ndarray,
               persons: Sequence[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], str, Optional[float]]]:
        """
        :param now: Frame time, seconds
        :param frame: BGR frame
        :param persons: [{"track_id", "bbox"}] from the vision module
        :return: [(payload, severity, confidence)] for the events to publish
        """
        out, due = [], []
        for person in persons:
            track = self.tracks.setdefault(int(person["track_id"]), TrackIdentity())
            track.last_seen = now
            interval = self.cfg.verify_sec if track.known else self.cfg.retry_sec
            if now - track.last_attempt >= interval:
                due.append(person)
        if due:
            # Faces are detected once per frame and shared out one-to-one over ALL persons in the frame, so a face
            # inside an overlapping (or huge) box of someone else is never taken for that person
            assigned = assign_faces(persons, self._detect_faces(frame), self.cfg)
            for person in due:
                track_id = int(person["track_id"])
                track = self.tracks[track_id]
                track.last_attempt = now
                track.attempts += 1
                face = assigned.get(track_id)
                found = None if face is None else (self.faces.embed(frame, face), frame, face)
                settled = track.identity not in (None, NOT_VISIBLE)
                event = self._verify(track_id, track, found) if settled \
                    else self._first_identification(track_id, track, found)
                if event:
                    out.append(event)
        for track_id in [t for t, s in self.tracks.items() if now - s.last_seen > self.cfg.track_lost_sec]:
            del self.tracks[track_id]
        return out

    def _first_identification(self, track_id: int, track: TrackIdentity, found):
        if found is None:
            if track.identity is None:
                return self._set(track_id, track, NOT_VISIBLE, False, 0.0)
            return None
        return self._set(track_id, track, *self._label(*found))

    def _verify(self, track_id: int, track: TrackIdentity, found):
        """
        Re-checks an identified track: the tracker may keep the id when another person takes the same place.
        The label changes only after ``switch_confirmations`` consecutive checks see a different face; a check
        that sees the same person has no side effects (nothing is stored for it).

        :param found: (embedding, image, face row) of the face assigned to this person, or None
        """
        if found is None:
            return None  # face turned away: keep the label
        embedding = found[0]
        name, similarity = self.database.match(embedding)
        if name == track.identity and similarity >= self.cfg.face_threshold:
            track.different_checks = 0
            track.similarity = max(track.similarity, similarity)
            return None
        if not track.known and name is not None and similarity >= self.cfg.face_threshold:
            track.different_checks = 0
            return self._set(track_id, track, name, True, similarity)  # unknown upgraded (e.g. just enrolled)
        reference, threshold = self._reference(track)
        if reference is not None and float(reference @ embedding) >= threshold - self.cfg.verify_margin:
            track.different_checks = 0
            if not track.known:  # same stranger: keep collecting samples for naming them later
                unknown = next(u for u in self.unknowns if u.label == track.identity)
                unknown.embedding = normalize(unknown.embedding * unknown.count + embedding)
                unknown.count += 1
                if self.store is not None:
                    crop, face = found[1], found[2]
                    aligned = self.faces.aligned_crop(crop, face) if hasattr(self.faces, "aligned_crop") else None
                    self.store.save(track.identity, embedding, aligned)
            return None
        track.different_checks += 1
        if track.different_checks < self.cfg.switch_confirmations:
            return None
        track.different_checks = 0
        return self._set(track_id, track, *self._label(*found))

    def _reference(self, track: TrackIdentity) -> Tuple[Optional[np.ndarray], float]:
        """
        :return: (embedding of the track's current identity, its matching threshold) or (None, 0)
        """
        if track.known and track.identity in self.database.names:
            return self.database.embeddings[self.database.names.index(track.identity)], self.cfg.face_threshold
        unknown = next((u for u in self.unknowns if u.label == track.identity), None)
        if unknown is not None:
            return unknown.embedding, self.cfg.unknown_threshold
        return None, 0.0

    def _set(self, track_id: int, track: TrackIdentity, label: str, known: bool, similarity: float):
        if label == track.identity:
            track.similarity = max(track.similarity, similarity)
            return None
        previous = track.identity
        track.identity, track.known, track.similarity = label, known, similarity
        severity = "info" if known or label == NOT_VISIBLE else "warning"
        payload = {"track_id": track_id, "identity": label, "known": known,
                   "similarity": round(similarity, 3), "previous": previous}
        return payload, severity, round(similarity, 3) if label != NOT_VISIBLE else None

    def sync(self) -> List[Tuple[Dict[str, Any], str, Optional[float]]]:
        """
        Promotes renamed unknown folders to known people and applies aliases added elsewhere (e.g. by
        ``tools.enroll_face --from-unknown``). Call periodically.

        :return: Events for the renames (one notice + one per affected live track)
        """
        out = []
        if self.store is None:
            return out
        self.store.purge()
        for old_label, name, folder in self.store.find_renamed():
            try:
                promotion = self.store.promote(old_label, name, folder)
            except (OSError, ValueError) as e:
                logger.warning(f"Cannot promote {old_label} → {name}: {e}")
                continue
            out += self.apply_rename(old_label, name, promotion.embeddings)
            logger.info(f"{old_label} renamed to {name}: {len(promotion.embeddings)} embeddings enrolled, "
                        f"images in {promotion.folder}")
        if self.aliases is not None:
            for old_label, name in list(self.aliases.aliases.items()):
                if any(u.label == old_label for u in self.unknowns):
                    out += self.apply_rename(old_label, name, np.zeros((0, EMBEDDING_SIZE), np.float32))
        return out

    def apply_rename(self, old_label: str, name: str,
                     embeddings: np.ndarray) -> List[Tuple[Dict[str, Any], str, Optional[float]]]:
        """
        Makes ``old_label`` a known person everywhere in this process: database, unknown cache, live tracks,
        alias registry.

        :param embeddings: Stored embeddings of the unknown (the cached mean is used when empty)
        """
        cached = next((u for u in self.unknowns if u.label == old_label), None)
        if len(embeddings) == 0 and cached is not None and name not in self.database.names:
            embeddings = cached.embedding.reshape(1, -1)
        if len(embeddings):
            self.database.add(name, list(embeddings))
            if self.database.path:
                self.database.save()
        self.unknowns = [u for u in self.unknowns if u.label != old_label]
        if self.aliases is not None and self.aliases.aliases.get(old_label) != name:
            self.aliases.add(old_label, name)

        out = [({"track_id": None, "identity": name, "known": True, "similarity": None, "previous": old_label,
                 "renamed_from": old_label}, "info", None)]
        for track_id, track in self.tracks.items():
            if track.identity == old_label:
                track.identity, track.known = name, True
                out.append(({"track_id": track_id, "identity": name, "known": True,
                             "similarity": round(track.similarity, 3), "previous": old_label,
                             "renamed_from": old_label}, "info", round(track.similarity, 3)))
        return out

    def _detect_faces(self, frame: np.ndarray) -> np.ndarray:
        """
        :return: YuNet rows (frame coordinates) passing the score and size limits
        """
        faces = [f for f in self.faces.detect(frame)
                 if f[-1] >= self.cfg.detection_score and min(f[2], f[3]) >= self.cfg.min_face_size]
        return np.array(faces, np.float32).reshape(-1, 15)

    def _label(self, embedding: np.ndarray, crop: np.ndarray, face: np.ndarray) -> Tuple[str, bool, float]:
        """Known name or unknown_N for a face; unknown faces are stored."""
        name, similarity = self.database.match(embedding)
        if name is not None and similarity >= self.cfg.face_threshold:
            label, known = name, True
        else:
            label, known = self._unknown_label(embedding), False
            if self.store is not None:
                aligned = self.faces.aligned_crop(crop, face) if hasattr(self.faces, "aligned_crop") else None
                self.store.save(label, embedding, aligned)
        self._debug_save(crop, face, label)
        return label, known, similarity

    def _unknown_label(self, embedding: np.ndarray) -> str:
        best, best_similarity = None, self.cfg.unknown_threshold
        for unknown in self.unknowns:
            similarity = float(unknown.embedding @ embedding)
            if similarity >= best_similarity:
                best, best_similarity = unknown, similarity
        if best is not None:
            best.embedding = normalize(best.embedding * best.count + embedding)
            best.count += 1
            return best.label
        if self.store is not None:
            label = self.store.next_label()
        else:
            self.unknown_counter += 1
            label = f"unknown_{self.unknown_counter}"
        self.unknowns.append(UnknownFace(label, embedding))
        if len(self.unknowns) > self.cfg.max_unknowns:
            self.unknowns.pop(0)
        return label

    def _debug_save(self, crop: np.ndarray, face: np.ndarray, label: str) -> None:
        if not self.cfg.debug_dir or not hasattr(self.faces, "aligned_crop"):
            return
        os.makedirs(self.cfg.debug_dir, exist_ok=True)
        path = os.path.join(self.cfg.debug_dir, f"{time.strftime('%Y%m%d-%H%M%S')}_{label}.jpg")
        cv2.imwrite(path, self.faces.aligned_crop(crop, face))


class IdentityRecognizer(SharedFrameConsumer):
    """
    Reads frames and person boxes from the vision shared buffer and publishes ``person_identified`` events.
    Every ``sync_sec`` it applies renamed unknown folders and reloads ``faces.npz`` / ``aliases.json`` when
    another process (``tools.enroll_face``) changed them.
    """
    name = "identity"
    source = "camera"

    def __init__(self, event_queue, control_queue, config: Dict[str, Any], engine: Optional[IdentityEngine] = None,
                 frame_buffer: Optional[SharedFrameBuffer] = None):
        self.cfg = config.get("identity", {})
        super().__init__(event_queue, control_queue, config, self.cfg.get("identity_fps", 2), frame_buffer)
        self.engine = engine
        self.sync_sec = float(self.cfg.get("sync_sec", 5.0))
        self._next_sync = 0.0
        self._mtimes: Dict[str, float] = {}

    def setup(self) -> None:
        if self.engine is None:
            cfg = IdentityConfig.from_dict(self.cfg)
            faces = FaceEngine(self.cfg["detector_model"], self.cfg["recognizer_model"], cfg.detection_score)
            database = FaceDatabase(self.cfg.get("db_path"))
            if not len(database):
                logger.warning(f"Face database {self.cfg.get('db_path')} is empty: everyone will be unknown "
                               f"(python -m tools.enroll_face)")
            store = aliases = None
            if self.cfg.get("save_unknowns", True):
                store = UnknownStore(self.cfg.get("unknown_dir", "data/faces/unknown"),
                                     self.cfg.get("faces_dir", "data/faces"),
                                     self.cfg.get("unknown_max_images", 10), self.cfg.get("unknown_retention_days", 30))
                aliases = AliasRegistry(self.cfg.get("aliases_path", "data/faces/aliases.json"))
            self.engine = IdentityEngine(faces, database, cfg, store, aliases)
        self._remember_mtimes()

    def step(self) -> None:
        now = time.monotonic()
        if now >= self._next_sync:
            self._next_sync = now + self.sync_sec
            self.sync()
        super().step()

    def sync(self) -> None:
        """Reloads externally changed files, applies renames and publishes their events."""
        engine = self.engine
        if self._changed(engine.database.path):
            engine.database.load(engine.database.path)
            logger.info(f"Face database reloaded: {len(engine.database)} people")
        if engine.aliases is not None and self._changed(engine.aliases.path):
            engine.aliases = AliasRegistry(engine.aliases.path)
        for payload, severity, confidence in engine.sync():
            self.emit("person_identified", payload, severity=severity, confidence=confidence)
        self._remember_mtimes()

    def on_event(self, event: Event, now: float) -> None:
        if event.type == "object_detected" and event.payload.get("state") == "disappeared":
            self.engine.drop(event.payload.get("track_id"))

    def process(self, frame_id: int, timestamp: float, frame: np.ndarray, meta: Dict[str, Any]) -> None:
        persons = meta.get("persons", [])
        if not persons:
            return
        for payload, severity, confidence in self.engine.update(timestamp, frame, persons):
            self.emit("person_identified", payload, severity=severity, confidence=confidence)

    def _paths(self) -> List[str]:
        paths = [self.engine.database.path]
        if self.engine.aliases is not None:
            paths.append(self.engine.aliases.path)
        return [p for p in paths if p]

    def _changed(self, path: Optional[str]) -> bool:
        if not path or not os.path.isfile(path):
            return False
        return os.path.getmtime(path) != self._mtimes.get(path)

    def _remember_mtimes(self) -> None:
        self._mtimes = {p: os.path.getmtime(p) for p in self._paths() if os.path.isfile(p)}


def run(event_queue, control_queue, config):
    """
    Module entry point for the orchestrator (own process).

    :param event_queue: Bus queue to publish events to
    :param control_queue: Control messages ("stop") and forwarded object_detected events
    :param config: Full merged config
    """
    IdentityRecognizer(event_queue, control_queue, config).run()
