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


def gas(level: int, second: int, group: int, third: int, dynamics: int) -> int:
    """
    A2 (TOD-42) + A8 (TOD-50) multi-gas escalation, every substance as a share of its critical concentration:

    * ``max_norm`` safe / moderate / dangerous / critical → low / medium / high / critical (any single substance at
      its critical level is critical on its own);
    * ``second_norm`` (the second-highest *active* gas: lower bound above the calm band) none / low / medium / high:
      a second active gas → one level above the maximum;
    * … but critical only with evidence — a gas at critical, ``group_index`` reached (Σ c / c_crit ≥ 1 inside one
      effect group, the additive mixture rule) or ``third_norm`` high (three gases at high); otherwise capped at high;
    * ``dynamics`` sharp_rise → one level up (as before; falling never lowers a level).

    Matrix (second × max, group below, third below, dynamics stable)::

                 low   medium  high  critical
        none      0      1      2       3
        low       1      2      2       3      ← +1, high + any → high (critical only with evidence)
        medium    1      2      2       3
        high      1      2      2       3      (second ≤ max in practice; the table is complete)

    Every input changes the level by at most one step: smooth and monotone.
    """
    escalated = level + (second >= 1)
    cap = 3 if (level == 3 or group == 1 or third == 1) else 2
    return clamp(min(escalated, cap) + (dynamics == 3))


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


def audio(intensity: int, sound_type: int, keywords: int, speaker: int) -> int:
    """
    A4 (TOD-44). intensity low / medium / high (dB above the microphone's background), sound_type_danger safe /
    suspicious / dangerous, keyword_danger none / alarming / dangerous / critical, speaker_unknown known / uncertain /
    unknown. level = max(speech, sound):

    * speech = keyword grade (none 0 … critical 3), +1 for an unknown voice that is at least medium loud and says
      nothing worse than alarming words (an unknown voice → medium, alarming words from it → high);
    * sound = sound danger (0 / 1 / 2) + loud − 1: a dangerous sound is high when loud and medium otherwise, a loud
      suspicious one medium.

    Each is a sum of one-level steps, max keeps that: smooth and monotone. The issue's examples hold: critical words →
    critical; dangerous sound + loud → high; dangerous words + unknown voice → high; loud + suspicious → medium; quiet
    and no words → low (unless the quiet sound itself is dangerous: medium).
    """
    speech = keywords + (speaker == 2 and keywords <= 1 and intensity >= 1)
    sound = max(0, sound_type + (intensity == 2) - 1)
    return clamp(max(speech, sound))


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
