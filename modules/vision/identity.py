"""
Person identification (F3): OpenCV YuNet face detection + SFace embeddings, matched against the unified identity
registry (``core.identity.IdentityRegistry``, F9 / TOD-20) that also holds voices.

Each person track is identified once; while it is unknown or its face is not visible, recognition is retried
every ``retry_sec``, identified tracks are re-verified every ``verify_sec``. A stranger becomes an ``unknown_N``
person in the registry (stable across restarts, one counter for faces and voices). A ``person_identified`` event
is published when a track's label changes::

    {"track_id": 7, "person_id": "3f2a…", "identity": "Oleh", "known": true, "similarity": 0.71,
     "previous": "not_visible"}

Renaming / merging / deleting happens in one place — the dashboard People page, ``python -m tools.identity`` or the
registry API — and live tracks are relabelled within ``sync_sec`` (immediately on forwarded ``identity_*`` events).

Privacy: face images and embeddings never leave the device. Face crops (thumbnails for the People page) are stored
in ``identity.faces_dir`` only when ``store_faces`` is on; unknown people not seen for ``unknown_retention_days`` are
deleted. Only person_id + display name go to the bus.
"""
import os
import time
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.events import Event
from core.identity import IdentityRegistry, registry_from_config
from modules.vision.consumer import SharedFrameConsumer
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
    face_images: int = 10               # face crops kept per person (People page thumbnails)
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
    In-memory gallery: one averaged embedding per name, optionally an .npz file (names, embeddings, counts).
    Used by ``tools/eval_faces.py`` and to import the legacy pre-registry ``faces.npz`` (``legacy_faces``).
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
    person_id: Optional[str] = None     # registry id; None while not identified / face not visible
    identity: Optional[str] = None      # display name when the event was published (NOT_VISIBLE = no face yet)
    known: bool = False
    similarity: float = 0.0
    last_attempt: float = float("-inf")
    last_seen: float = 0.0
    attempts: int = 0
    different_checks: int = 0           # consecutive verifications that saw another face


Result = Tuple[Dict[str, Any], str, Optional[float]]


@dataclass
class IdentityEngine:
    """
    Identification logic: feed frames + person boxes, get event payloads back.
    ``faces`` is anything with detect(image) and embed(image, face) (FaceEngine in production). People — known and
    ``unknown_N`` — live in the shared ``IdentityRegistry``; face crops are stored when it has a ``faces_dir``.
    """
    faces: Any
    registry: IdentityRegistry
    cfg: IdentityConfig = field(default_factory=IdentityConfig)
    tracks: Dict[int, TrackIdentity] = field(default_factory=dict)

    def drop(self, track_id: int) -> None:
        self.tracks.pop(track_id, None)

    def update(self, now: float, frame: np.ndarray, persons: Sequence[Dict[str, Any]]) -> List[Result]:
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
                settled = track.person_id is not None
                event = self._verify(track_id, track, found) if settled \
                    else self._first_identification(track_id, track, found)
                if event:
                    out.append(event)
        for track_id in [t for t, s in self.tracks.items() if now - s.last_seen > self.cfg.track_lost_sec]:
            del self.tracks[track_id]
        return out

    def _first_identification(self, track_id: int, track: TrackIdentity, found) -> Optional[Result]:
        if found is None:
            if track.identity is None:
                return self._set(track_id, track, None, NOT_VISIBLE, False, 0.0)
            return None
        return self._set(track_id, track, *self._label(*found))

    def _verify(self, track_id: int, track: TrackIdentity, found) -> Optional[Result]:
        """
        Re-checks an identified track: the tracker may keep the id when another person takes the same place.
        The label changes only after ``switch_confirmations`` consecutive checks see a different face.

        :param found: (embedding, image, face row) of the face assigned to this person, or None
        """
        if found is None:
            return None  # face turned away: keep the label
        embedding, image, face = found
        registry = self.registry
        current = registry.resolve(track.person_id)
        known_id, similarity = registry.match_face(embedding, self.cfg.face_threshold, known=True)
        if known_id is not None and known_id == current:
            track.different_checks = 0
            track.similarity = max(track.similarity, similarity)
            registry.touch(current)
            return None
        if not track.known and known_id is not None:
            track.different_checks = 0
            return self._set(track_id, track, known_id, registry.name(known_id), True, similarity)  # just named
        reference = registry.centroid(current, "face") if current else None
        threshold = self.cfg.face_threshold if track.known else self.cfg.unknown_threshold
        if reference is not None and float(reference @ embedding) >= threshold - self.cfg.verify_margin:
            track.different_checks = 0
            if not track.known:  # same stranger: keep collecting samples for naming them later
                registry.add_embedding(current, "face", embedding, quality=face_quality(face))
                self._save_crop(current, image, face)
            registry.touch(current)
            return None
        track.different_checks += 1
        if track.different_checks < self.cfg.switch_confirmations:
            return None
        track.different_checks = 0
        return self._set(track_id, track, *self._label(*found))

    def _set(self, track_id: int, track: TrackIdentity, person_id: Optional[str], label: str, known: bool,
             similarity: float, renamed: bool = False) -> Optional[Result]:
        if person_id == track.person_id and label == track.identity and known == track.known:
            track.similarity = max(track.similarity, similarity)
            return None
        previous = track.identity
        track.person_id, track.identity, track.known = person_id, label, known
        track.similarity = similarity if not renamed else track.similarity
        severity = "info" if known or label == NOT_VISIBLE else "warning"
        payload = {"track_id": track_id, "person_id": person_id, "identity": label, "known": known,
                   "similarity": round(track.similarity, 3), "previous": previous}
        if renamed:
            payload["renamed_from"] = previous
            severity = "info"
        return payload, severity, round(track.similarity, 3) if label != NOT_VISIBLE else None

    def sync(self, force: bool = False) -> List[Result]:
        """
        Applies registry changes made by other processes (rename / merge / delete from the dashboard or CLI) to the
        live tracks. Call periodically and on ``identity_*`` bus events.

        :param force: Re-check the tracks even if the database did not change (e.g. after an own purge)
        :return: Events for the relabelled tracks
        """
        if not (self.registry.poll() or force):
            return []
        out = []
        for track_id, track in self.tracks.items():
            if track.person_id is None:
                continue
            pid = self.registry.resolve(track.person_id)
            if pid is None:  # deleted: identify again from scratch
                self.tracks[track_id] = TrackIdentity(last_seen=track.last_seen)
                continue
            event = self._set(track_id, track, pid, self.registry.name(pid), self.registry.is_known(pid),
                              track.similarity, renamed=True)
            if event:
                out.append(event)
        return out

    def _detect_faces(self, frame: np.ndarray) -> np.ndarray:
        """
        :return: YuNet rows (frame coordinates) passing the score and size limits
        """
        faces = [f for f in self.faces.detect(frame)
                 if f[-1] >= self.cfg.detection_score and min(f[2], f[3]) >= self.cfg.min_face_size]
        return np.array(faces, np.float32).reshape(-1, 15)

    def _label(self, embedding: np.ndarray, image: np.ndarray,
               face: np.ndarray) -> Tuple[str, str, bool, float]:
        """
        Known person, an earlier stranger, or a new ``unknown_N`` (created in the registry).

        :return: (person_id, display name, known, similarity)
        """
        registry = self.registry
        person_id, similarity = registry.match_face(embedding, self.cfg.face_threshold, known=True)
        known = person_id is not None
        if not known:
            person_id, similarity = registry.match_face(embedding, self.cfg.unknown_threshold, known=False)
            if person_id is not None:
                registry.add_embedding(person_id, "face", embedding, quality=face_quality(face))
            else:
                person_id = registry.create_unknown("face", embedding, quality=face_quality(face))
            self._save_crop(person_id, image, face)
        registry.touch(person_id)
        label = registry.name(person_id)
        self._debug_save(image, face, label)
        return person_id, label, known, similarity

    def _save_crop(self, person_id: str, image: np.ndarray, face: np.ndarray) -> None:
        if self.registry.faces_dir and hasattr(self.faces, "aligned_crop"):
            self.registry.save_face_crop(person_id, self.faces.aligned_crop(image, face), self.cfg.face_images)

    def _debug_save(self, crop: np.ndarray, face: np.ndarray, label: str) -> None:
        if not self.cfg.debug_dir or not hasattr(self.faces, "aligned_crop"):
            return
        os.makedirs(self.cfg.debug_dir, exist_ok=True)
        path = os.path.join(self.cfg.debug_dir, f"{time.strftime('%Y%m%d-%H%M%S')}_{label}.jpg")
        cv2.imwrite(path, self.faces.aligned_crop(crop, face))


def face_quality(face: np.ndarray) -> float:
    """Detector score × size factor (a 112 px face — SFace input — or larger counts fully)."""
    return round(float(face[-1]) * min(1.0, float(min(face[2], face[3])) / 112), 4)


def open_registry(cfg: Dict[str, Any], publish=None) -> IdentityRegistry:
    """
    The identity registry configured in the ``identity`` section; imports a legacy ``faces.npz`` / unknown folders
    the first time (empty registry).
    """
    registry = registry_from_config(cfg, publish)
    if not registry.persons and cfg.get("legacy_db_path") and os.path.isfile(cfg["legacy_db_path"]):
        from modules.vision.legacy_faces import migrate

        report = migrate(registry, cfg["legacy_db_path"], cfg.get("legacy_unknown_dir"))
        logger.warning(f"Imported the legacy face database into {registry.path}: {report}")
    return registry


class IdentityRecognizer(SharedFrameConsumer):
    """
    Reads frames and person boxes from the vision shared buffer and publishes ``person_identified`` events
    (``person_id`` + ``identity`` = display name). Registry changes from other processes (forwarded ``identity_*``
    events, or every ``sync_sec`` for CLI edits) relabel the live tracks at once.
    """
    name = "identity"
    source = "camera"

    def __init__(self, event_queue, control_queue, config: Dict[str, Any], engine: Optional[IdentityEngine] = None,
                 frame_buffer: Optional[SharedFrameBuffer] = None):
        self.cfg = config.get("identity", {})
        super().__init__(event_queue, control_queue, config, self.cfg.get("identity_fps", 2), frame_buffer)
        self.engine = engine                        # the first camera's engine (single camera: the only one)
        self.engines: Dict[str, IdentityEngine] = {}  # device_id → engine (tracks are per camera, registry shared)
        self.sync_sec = float(self.cfg.get("sync_sec", 5.0))
        self.retention_days = float(self.cfg.get("unknown_retention_days", 30))
        self._next_sync = 0.0
        self._next_purge = 0.0

    def engine_for(self, device_id: str) -> IdentityEngine:
        if device_id not in self.engines:
            self.engines[device_id] = self.engine if not self.engines else \
                IdentityEngine(self.engine.faces, self.engine.registry, self.engine.cfg)
        return self.engines[device_id]

    def on_camera_removed(self, device_id: str) -> None:
        engine = self.engines.pop(device_id, None)
        if engine is self.engine and self.engines:  # keep a live engine as the template
            self.engine = next(iter(self.engines.values()))

    def setup(self) -> None:
        super().setup()
        if self.engine is None:
            cfg = IdentityConfig.from_dict(self.cfg)
            faces = FaceEngine(self.cfg["detector_model"], self.cfg["recognizer_model"], cfg.detection_score)
            registry = open_registry(self.cfg, publish=self.publish)
            if not any(p["is_known"] for p in registry.persons.values()):
                logger.warning("No known people in the identity registry: everyone will be unknown_N (rename them on "
                               "the dashboard People page, python -m tools.identity, or python -m tools.enroll)")
            self.engine = IdentityEngine(faces, registry, cfg)

    def step(self) -> None:
        now = time.monotonic()
        if now >= self._next_sync:
            self._next_sync = now + self.sync_sec
            self.sync(purge=now >= self._next_purge)
            if now >= self._next_purge:
                self._next_purge = now + 3600
        super().step()

    def sync(self, force: bool = False, purge: bool = False) -> None:
        """Applies registry changes (and the unknown-people retention) to live tracks and publishes the events."""
        if purge and self.retention_days > 0:
            force = bool(self.engine.registry.purge_unknowns(self.retention_days)) or force
        if not (self.engine.registry.poll() or force):
            return
        for device_id, engine in (self.engines or {None: self.engine}).items():
            slot = self.scheduler.slots.get(device_id) if device_id else None
            for payload, severity, confidence in engine.sync(force=True):
                self.emit("person_identified", payload, severity=severity, confidence=confidence,
                          device_id=device_id, location_id=slot.location_id if slot else None)

    def on_event(self, event: Event, now: float) -> None:
        if event.type == "object_detected" and event.payload.get("state") == "disappeared":
            engine = self.engines.get(event.device_id) or (self.engine if not self.engines else None)
            if engine is not None:
                engine.drop(event.payload.get("track_id"))
        elif event.type.startswith("identity_"):
            self.sync(force=True)

    def process(self, frame_id: int, timestamp: float, frame: np.ndarray, meta: Dict[str, Any]) -> None:
        persons = meta.get("persons", [])
        if not persons:
            return
        engine = self.engine_for(self.current.device_id) if self.current else self.engine
        for payload, severity, confidence in engine.update(timestamp, frame, persons):
            self.emit("person_identified", payload, severity=severity, confidence=confidence)

    def teardown(self) -> None:
        super().teardown()
        if self.engine is not None:
            self.engine.registry.close()


def run(event_queue, control_queue, config):
    """
    Module entry point for the orchestrator (own process).

    :param event_queue: Bus queue to publish events to
    :param control_queue: Control messages ("stop"), forwarded object_detected and identity_* events
    :param config: Full merged config
    """
    IdentityRecognizer(event_queue, control_queue, config).run()
