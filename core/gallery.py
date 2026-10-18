"""
Multi-prototype face gallery (F14 / TOD-25): the pure numpy rules behind ``IdentityRegistry``'s face embeddings.

A person is a set of up to ``limit`` prototypes (L2-normalised SFace vectors) with metadata (tier, yaw / pitch /
brightness bucket). Matching is the mean of the top-3 cosine similarities between a query and the prototypes, so a
person in glasses matches their "glasses" prototypes instead of one blurred centroid.

Admission (``admit``): only ``good`` / ``enrolled`` faces (and pre-TOD-25 rows without a tier) enter; a vector
≥ ``redundancy`` similar to an existing prototype adds nothing and is skipped; a full gallery drops its most
redundant prototype (highest mean similarity to the others), never the last ones of an under-represented bucket.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ADMISSIBLE_TIERS = (None, "good", "enrolled")    # None: rows from before TOD-25 (no quality recorded)
YAW_BUCKETS = ("left", "frontal", "right")


def yaw_bucket(yaw: Optional[float], frontal_deg: float = 15.0) -> Optional[str]:
    if yaw is None:
        return None
    return "frontal" if abs(yaw) <= frontal_deg else ("left" if yaw < 0 else "right")


def pitch_bucket(pitch: Optional[float], level_deg: float = 10.0) -> Optional[str]:
    if pitch is None:
        return None
    return "level" if abs(pitch) <= level_deg else ("up" if pitch < 0 else "down")


def brightness_bucket(brightness: Optional[float]) -> Optional[str]:
    if brightness is None:
        return None
    return "dark" if brightness < 80 else ("bright" if brightness > 170 else "normal")


def top_k_mean(similarities: np.ndarray, k: int = 3) -> float:
    """:return: Mean of the k largest values (all of them when there are fewer)"""
    if similarities.size == 0:
        return 0.0
    k = min(k, similarities.size)
    return float(np.mean(np.partition(similarities.reshape(-1), -k)[-k:]))


@dataclass
class Quotas:
    """Per-bucket limits so variations stay represented in a full gallery."""
    min_frontal: int = 3        # never drop below this many frontal prototypes
    max_per_yaw: int = 8        # cap on left / right prototypes (frontal is capped by the limit only)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Quotas":
        data = data or {}
        return cls(int(data.get("min_frontal", 3)), int(data.get("max_per_yaw", 8)))


@dataclass
class Prototype:
    id: Any
    vector: np.ndarray
    yaw_bucket: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Admission:
    accepted: bool
    reason: str                       # added / evicted / redundant / tier / bucket_full
    evict: List[Any] = field(default_factory=list)   # prototype ids to remove


def redundancy_scores(vectors: np.ndarray) -> np.ndarray:
    """:return: For every prototype the mean similarity to the others (high = adds little)"""
    n = len(vectors)
    if n < 2:
        return np.zeros(n, np.float32)
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, 0.0)
    return sims.sum(axis=1) / (n - 1)


def admit(gallery: Sequence[Prototype], vector: np.ndarray, tier: Optional[str], bucket: Optional[str],
          limit: int = 30, redundancy: float = 0.9, quotas: Optional[Quotas] = None) -> Admission:
    """
    Decides whether a new face embedding enters a person's gallery.

    :param gallery: The person's current prototypes
    :param vector: L2-normalised embedding
    :param tier: Quality tier of the face (``good`` / ``usable`` / ``enrolled`` / None = legacy)
    :param bucket: Its yaw bucket (left / frontal / right) or None
    :return: Admission (accepted + prototype ids to evict)
    """
    quotas = quotas or Quotas()
    if tier not in ADMISSIBLE_TIERS:
        return Admission(False, "tier")
    if gallery:
        matrix = np.stack([p.vector for p in gallery])
        if float(np.max(matrix @ vector)) >= redundancy:
            return Admission(False, "redundant")
    counts: Dict[Optional[str], int] = {}
    for p in gallery:
        counts[p.yaw_bucket] = counts.get(p.yaw_bucket, 0) + 1
    side_full = bucket in ("left", "right") and counts.get(bucket, 0) >= quotas.max_per_yaw
    if len(gallery) < limit and not side_full:
        return Admission(True, "added")
    # Full (or this side bucket is full): drop the most redundant prototype the quotas allow to drop
    candidates = list(gallery) + [Prototype("__new__", vector, bucket)]
    scores = redundancy_scores(np.stack([p.vector for p in candidates]))
    for index in np.argsort(-scores):
        victim = candidates[index]
        if side_full and victim.yaw_bucket != bucket:
            continue        # make room inside the full side bucket only
        if victim.yaw_bucket == "frontal" and bucket != "frontal" and counts.get("frontal", 0) <= quotas.min_frontal:
            continue
        if victim.id == "__new__":
            return Admission(False, "bucket_full" if side_full else "redundant")
        return Admission(True, "evicted", [victim.id])
    return Admission(False, "bucket_full")


def gallery_similarity(a: np.ndarray, b: np.ndarray, k: int = 5, centroid_weight: float = 0.5) -> Tuple[float, float]:
    """
    How alike two galleries are (dedup score).

    :param a: (n, d) prototypes of one person
    :param b: (m, d) prototypes of the other
    :return: (score = weighted top-k mean of the pairwise similarities + centroid cosine, centroid cosine)
    """
    if not len(a) or not len(b):
        return 0.0, 0.0
    pairwise = top_k_mean(a @ b.T, k)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    centroid = float(ca @ cb / max(1e-9, float(np.linalg.norm(ca) * np.linalg.norm(cb))))
    return (1 - centroid_weight) * pairwise + centroid_weight * centroid, centroid
