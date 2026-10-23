"""
Window features of a pose sequence (learned action classifier) — numpy only, the same code for training
(``tools/train_actions.py`` on public COCO-17 skeleton datasets) and live recognition (``modules/vision/actions.py``).

A window = the samples of one person over ``window_sec`` at ``action_fps`` (live: 2.5 s at 5 fps ≈ 12 samples), each
sample ``(t, keypoints (17, 2) pixels, scores (17,))`` plus optional other persons in the same frame. Every frame is
normalised by the person's own body (origin = mid-hip, else the keypoint box centre; scale = torso length, else a
third of the keypoint box height), so camera distance and resolution drop out; keypoints below ``min_score`` are
missing (NaN) and only feed the visibility features. Features (fixed length, ``FEATURE_NAMES``):

* posture: mean / std of every normalised joint, visibility per joint, torso angle (mean, min, max), knee angles,
  box aspect ratio;
* motion: mean speed per joint, hip vertical speed (mean / max down / max up), centre speed, scale change rate,
  overall motion energy;
* hands: wrist ↔ nose distance (min / mean per wrist), share of samples with a wrist at the mouth / above the
  shoulders, number of separate hand-to-mouth approaches (smoking, drinking, eating, phone);
* interaction: distance to the nearest other person (min / mean), the closest wrist / ankle to another person's
  torso (fighting, hugging, pushing).
"""
import math
import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np

NOSE, L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = \
    0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
JOINTS = 17
MOUTH_DIST = 0.45                     # wrist within this many torso lengths of the nose = at the mouth

Sample = Tuple[float, np.ndarray, np.ndarray]          # (t seconds, (17, 2) pixels, (17,) scores)


def _mid(kp: np.ndarray, a: int, b: int) -> Optional[np.ndarray]:
    pa, pb = kp[a], kp[b]
    ok_a, ok_b = not np.isnan(pa).any(), not np.isnan(pb).any()
    if ok_a and ok_b:
        return (pa + pb) / 2
    return pa if ok_a else pb if ok_b else None


def _angle_from_vertical(top: Optional[np.ndarray], bottom: Optional[np.ndarray]) -> float:
    if top is None or bottom is None:
        return math.nan
    dx, dy = top[0] - bottom[0], bottom[1] - top[1]
    return math.degrees(math.atan2(abs(dx), dy if dy != 0 else 1e-9)) if dy > -1e9 else math.nan


def _knee(kp: np.ndarray, hip: int, knee: int, ankle: int) -> float:
    a, b, c = kp[hip], kp[knee], kp[ankle]
    if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
        return math.nan
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return math.degrees(math.acos(float(np.clip(v1 @ v2 / n, -1, 1)))) if n > 0 else math.nan


def normalise(kp: np.ndarray, scores: np.ndarray, min_score: float) -> Tuple[Optional[np.ndarray], dict]:
    """
    :return: ((17, 2) body-normalised joints with NaN for missing ones, frame info {origin, scale, box_h, box_w})
             — None when fewer than 4 joints are visible
    """
    kp = np.asarray(kp, float).copy()
    kp[np.asarray(scores) < min_score] = np.nan
    visible = ~np.isnan(kp).any(axis=1)
    if visible.sum() < 4:
        return None, {}
    pts = kp[visible]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    box_w, box_h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    hip, shoulder = _mid(kp, L_HIP, R_HIP), _mid(kp, L_SHO, R_SHO)
    origin = hip if hip is not None else np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    torso = float(np.linalg.norm(shoulder - hip)) if hip is not None and shoulder is not None else 0.0
    scale = torso if torso > 0.15 * box_h else box_h / 3
    return (kp - origin) / scale, {"origin": origin, "scale": scale, "box_h": box_h, "box_w": box_w, "raw": kp}


def _nanstats(values: Sequence[float]) -> Tuple[float, float, float, float]:
    arr = np.asarray([v for v in values if not math.isnan(v)], float)
    if not len(arr):
        return 0.0, 0.0, 0.0, 0.0
    return float(arr.mean()), float(arr.min()), float(arr.max()), float(arr.std())


def window_features(samples: Sequence[Sample], others: Optional[Sequence[Sequence[np.ndarray]]] = None,
                    min_score: float = 0.4) -> Optional[np.ndarray]:
    with warnings.catch_warnings():                          # all-NaN joints are expected (hidden body parts)
        warnings.simplefilter("ignore", RuntimeWarning)
        return _window_features(samples, others, min_score)


def _window_features(samples: Sequence[Sample], others: Optional[Sequence[Sequence[np.ndarray]]],
                     min_score: float) -> Optional[np.ndarray]:
    """
    :param samples: the person's samples, oldest first
    :param others: per sample, the (17, 2) keypoints (NaN = missing) of the other persons in that frame
    :return: the feature vector (len(FEATURE_NAMES)), or None when fewer than 3 usable samples
    """
    frames, infos, times, kept = [], [], [], []
    for i, (t, kp, sc) in enumerate(samples):
        norm, info = normalise(kp, sc, min_score)
        if norm is None:
            continue
        frames.append(norm)
        infos.append(info)
        times.append(float(t))
        kept.append(i)
    if len(frames) < 3:
        return None
    seq = np.stack(frames)                                   # (n, 17, 2)
    vis = ~np.isnan(seq).any(axis=2)                         # (n, 17)
    with np.errstate(invalid="ignore"):
        mean = np.nan_to_num(np.nanmean(seq, axis=0)).ravel()
        std = np.nan_to_num(np.nanstd(seq, axis=0)).ravel()
    duration = max(1e-3, times[-1] - times[0])
    dt = np.diff(times)
    dt[dt <= 0] = 1e-3
    vel = np.linalg.norm(np.diff(seq, axis=0), axis=2) / dt[:, None]          # (n-1, 17) torso lengths / s
    with np.errstate(invalid="ignore"):
        joint_speed = np.nan_to_num(np.nanmean(vel, axis=0))
    energy = float(np.nan_to_num(np.nanmean(vel))) if np.isfinite(vel).any() else 0.0

    torso_angles, knees, aspects, wrist_nose, mouth, above, hips_y = [], [], [], [[], []], [], [], []
    for norm, info in zip(frames, infos, strict=True):
        raw = info["raw"]
        torso_angles.append(_angle_from_vertical(_mid(raw, L_SHO, R_SHO), _mid(raw, L_HIP, R_HIP)))
        knees.append(np.nanmean([_knee(raw, L_HIP, L_KNE, L_ANK), _knee(raw, R_HIP, R_KNE, R_ANK)])
                     if not (np.isnan(raw[[L_KNE, R_KNE]]).all()) else math.nan)
        aspects.append(info["box_h"] / info["box_w"])
        at_mouth, up = False, False
        for side, wrist in enumerate((L_WRI, R_WRI)):
            if np.isnan(norm[wrist]).any() or np.isnan(norm[NOSE]).any():
                wrist_nose[side].append(math.nan)
                continue
            d = float(np.linalg.norm(norm[wrist] - norm[NOSE]))
            wrist_nose[side].append(d)
            at_mouth |= d < MOUTH_DIST
            shoulder = norm[L_SHO] if side == 0 else norm[R_SHO]
            up |= not np.isnan(shoulder).any() and norm[wrist][1] < shoulder[1]
        mouth.append(at_mouth)
        above.append(up)
        hips_y.append(info["origin"][1] / info["scale"])
    approaches = sum(1 for a, b in zip(mouth, mouth[1:], strict=False) if b and not a) + int(bool(mouth and mouth[0]))
    hip_v = np.diff(hips_y) / dt                                                # + = moving down (image y)
    centres = np.array([info["origin"] for info in infos])
    scales = np.array([info["scale"] for info in infos])
    centre_speed = float(np.mean(np.linalg.norm(np.diff(centres, axis=0), axis=1) / scales[1:] / dt))
    scale_rate = float(np.mean(np.abs(np.diff(np.log(scales))) / dt))

    near, limb_near = [], []
    if others is not None:
        for idx, info in zip(kept, infos, strict=True):
            raw, scale = info["raw"], info["scale"]
            best, best_limb = math.inf, math.inf
            for other in others[idx] if idx < len(others) else []:
                other = np.asarray(other, float)
                centre = _mid(other, L_HIP, R_HIP)
                top = _mid(other, L_SHO, R_SHO)
                if centre is None and top is None:
                    continue
                target = (centre + top) / 2 if centre is not None and top is not None else (centre if centre is not
                                                                                             None else top)
                best = min(best, float(np.linalg.norm(info["origin"] - target)) / scale)
                for limb in (L_WRI, R_WRI, L_ANK, R_ANK):
                    if not np.isnan(raw[limb]).any():
                        best_limb = min(best_limb, float(np.linalg.norm(raw[limb] - target)) / scale)
            near.append(best)
            limb_near.append(best_limb)
    near = [min(v, 10.0) for v in near] or [10.0]
    limb_near = [min(v, 10.0) for v in limb_near] or [10.0]

    t_mean, t_min, t_max, t_std = _nanstats(torso_angles)
    k_mean, k_min, _, k_std = _nanstats(knees)
    a_mean, _, _, a_std = _nanstats(aspects)
    wl, wr = _nanstats(wrist_nose[0]), _nanstats(wrist_nose[1])
    hv = hip_v[np.isfinite(hip_v)]
    extra = [t_mean / 90, t_min / 90, t_max / 90, t_std / 90, k_mean / 180, k_min / 180, k_std / 180,
             min(a_mean, 6.0), min(a_std, 6.0),
             float(hv.mean()) if len(hv) else 0.0, float(hv.max()) if len(hv) else 0.0,
             float(hv.min()) if len(hv) else 0.0, min(centre_speed, 20.0), min(scale_rate, 20.0), min(energy, 20.0),
             min(wl[1], 10.0) if wrist_nose[0] else 10.0, min(wl[0], 10.0) if wrist_nose[0] else 10.0,
             min(wr[1], 10.0) if wrist_nose[1] else 10.0, min(wr[0], 10.0) if wrist_nose[1] else 10.0,
             float(np.mean(mouth)), float(np.mean(above)), approaches / max(1.0, duration),
             min(near), float(np.mean(near)), min(limb_near), float(vis[:, [L_ANK, R_ANK]].mean()), duration]
    return np.concatenate([mean, std, vis.mean(axis=0), np.minimum(joint_speed, 20.0), extra]).astype(np.float32)


FEATURE_NAMES: List[str] = (
    [f"mean_{j}_{c}" for j in range(JOINTS) for c in "xy"] + [f"std_{j}_{c}" for j in range(JOINTS) for c in "xy"]
    + [f"vis_{j}" for j in range(JOINTS)] + [f"speed_{j}" for j in range(JOINTS)]
    + ["torso_mean", "torso_min", "torso_max", "torso_std", "knee_mean", "knee_min", "knee_std", "aspect_mean",
       "aspect_std", "hip_vy_mean", "hip_vy_down", "hip_vy_up", "centre_speed", "scale_rate", "energy",
       "wrist_l_nose_min", "wrist_l_nose_mean", "wrist_r_nose_min", "wrist_r_nose_mean", "mouth_share",
       "hands_up_share", "mouth_approaches_per_s", "other_min", "other_mean", "limb_to_other_min", "ankles_visible",
       "duration"])
