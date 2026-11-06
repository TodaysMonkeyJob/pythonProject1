"""
Decision policies of the hierarchical fuzzy system (A1 / TOD-41): the level of every cell of the four rule tables
``config/fuzzy/{gas,camera,audio,top}.yaml`` (tests/test_hfs.py checks every cell against these functions).

Why tables: a Mamdani system with centroid defuzzification is monotonic only for a complete rule base that is
monotone and smooth (neighbouring cells differ by at most one level) on a Ruspini partition (the input terms sum to
1 — triangles / trapezoids whose edges meet the neighbours' peaks). With product AND / implication, sum aggregation
and output terms of equal area the centroid is Σ w_r · c_r, a multilinear interpolation of the table — exactly
monotonic (tools/plot_surfaces.py → reports/surfaces/). Every policy below is a sum / max of steps of at most one
level per input term, clamped to 0..3, hence monotone and smooth by construction.

Each function takes the term index of every input (0 = the first, least risky term of that variable in the YAML) and
returns 0 = low, 1 = medium, 2 = high, 3 = critical.
"""
from typing import Sequence

LEVELS = ("low", "medium", "high", "critical")


def clamp(level: int) -> int:
    return max(0, min(3, level))


def gas(level: int, dynamics: int, mixture: int) -> int:
    """
    gas_level clean / slight / elevated / critical → 0 / 1 / 2 / 3 (a critical concentration is critical on its own;
    the limit is the critical one, so a quarter of it is already worth attention); gas_dynamics sharply_rising → +1
    (an early leak reaches the next level within minutes; falling never lowers a level: the gas is still there);
    mixture_index high → +1 (several gases at or above the combined limit Σ C/L ≥ 1).
    """
    return clamp(level + (dynamics == 3) + (mixture == 2))


def camera(unknown: int, motion: int, off_hours: int) -> int:
    """
    unknown_share none / some / all + off_hours within / near / outside + (motion_intensity high) − 2: strangers only
    matter outside activity hours — only strangers long after hours is high, critical when they move a lot; visitors
    among known people by day are normal; an empty room is low at any time, sustained motion long after hours with
    nobody identified is medium (someone the camera cannot see well).
    """
    return clamp(unknown + off_hours + (motion == 2) - 2)


def audio(sound_level: int, sound_type: int, keywords: int, speaker: int) -> int:
    """
    max(danger_keywords none / mild / severe, sound_type benign / suspicious / dangerous) + (sound_level loud) +
    speaker_unknown known / uncertain / unknown → 0 / 0 / 1: a shouted call for help or a loud dangerous sound is
    critical, a quiet one high; distressed words are medium from a known voice, high from an unknown one; an unknown
    voice talking normally or a loud sound alone is medium.
    """
    return clamp(max(keywords, sound_type) + (sound_level == 2) + (speaker == 2))


def top(gas_risk: int, camera_risk: int, audio_risk: int) -> int:
    """
    Supervisor's combination: any subsystem critical → critical; two high → critical; one high (others low / medium)
    → high; two medium → high; one medium (others low) → medium; all low → low.
    """
    ranks = [gas_risk, camera_risk, audio_risk]
    if 3 in ranks or ranks.count(2) >= 2:
        return 3
    if 2 in ranks or ranks.count(1) >= 2:
        return 2
    return 1 if 1 in ranks else 0


POLICIES = {"gas": gas, "camera": camera, "audio": audio, "top": top}


def policy_level(levels: Sequence[str]) -> str:
    """:return: the top-level policy over subsystem level names (low / medium / high / critical)"""
    return LEVELS[top(*(LEVELS.index(level) for level in levels))]
