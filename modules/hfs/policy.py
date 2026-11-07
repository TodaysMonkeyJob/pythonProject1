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
    A2 (TOD-42), all substances as shares of their critical concentration: max_norm safe / moderate / dangerous /
    critical → low / medium / high / critical (any single substance at its critical level is critical on its own);
    dynamics sharp_rise → one level up (the next level within minutes; falling never lowers a level — the gas is still
    there); mixture_index high (Σ c/c_crit ≥ 1) → one level up: two gases at 0.6 are worse than one.
    """
    return clamp(level + (dynamics == 3) + (mixture == 2))


def camera(unknown: int, motion: int, activity: int) -> int:
    """
    A3 (TOD-43). unknown_share none / part / most / all, motion_intensity none / low / medium / high, activity_time
    working / transition / off. Base by unknown share and time::

                 working  transition  off
        none        0         0        0
        part        0         0        0      visitors among known people: low
        most        1         1        1      mostly strangers: medium
        all         1         2        2      only strangers: medium by day, high around / after hours

    high motion → one level up when anybody unknown is in view (most strangers + high motion → high at any time,
    only strangers after hours moving a lot → critical), or with nobody unknown in view after hours (sustained motion
    the camera cannot attribute: medium). Steps of at most one level per term: smooth, monotone.
    """
    base = ((0, 0, 0), (0, 0, 0), (1, 1, 1), (1, 2, 2))[unknown][activity]
    return clamp(base + (motion == 3 and (unknown >= 1 or activity == 2)))


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
