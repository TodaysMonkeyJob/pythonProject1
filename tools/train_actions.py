"""
Learned action classifier on public COCO-17 skeleton datasets → ``models/action_classifier.joblib``.

Data (pre-extracted 2D skeletons, OpenMMLab / MMAction2 — HRNet COCO-17 keypoints, the layout yolo11n-pose outputs):

* HMDB51 (``hmdb51_2d.pkl``, 6 371 clips, movies / web video; CC BY-NC-SA — research use) — split 1;
* NTU RGB+D 60 (``ntu60_2d.pkl``, 56 578 clips, indoor Kinect studio; ROSE Lab terms, non-commercial research) —
  cross-subject split.

    curl -L -o data/action_datasets/hmdb51_2d.pkl https://download.openmmlab.com/mmaction/v1.0/skeleton/data/hmdb51_2d.pkl
    curl -L -o data/action_datasets/ntu60_2d.pkl https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl
    python -m tools.train_actions                       # → models/action_classifier.joblib + reports/action_classifier/

Clips are resampled to the live ``actions.action_fps`` (5 fps) and cut into ``window_sec`` windows (2.5 s, stride 1 s)
— exactly what the action module sees — and turned into ``modules.vision.action_features.window_features``. The
dataset classes are mapped to the security-relevant actions below (everything else = ``other``); training windows are
also mirrored left ↔ right, and a third of the single-person training clips get a synthetic bystander (another
clip's person placed beside them, label unchanged — proximity alone must not mean fighting / hugging).
Model selection: a few sklearn models compared by macro-F1 on a validation part of the
training clips (grouped by clip), the winner refitted on all training windows, reported on the official test split.
"""
import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

if __package__ in (None, ""):  # allow `python tools/train_actions.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.vision.action_features import FEATURE_NAMES, window_features  # noqa: E402

DATA_DIR = "data/action_datasets"
MODEL_PATH = "models/action_classifier.joblib"
REPORT_DIR = "reports/action_classifier"
SOURCE_FPS = 30.0

HMDB = {"walk": "walking", "run": "running", "sit": "sitting_down", "stand": "standing_up", "fall_floor": "falling",
        "punch": "fighting", "hit": "fighting", "kick": "fighting", "smoke": "smoking", "drink": "drinking",
        "eat": "eating", "wave": "waving", "pick": "picking_up", "throw": "throwing", "climb": "climbing",
        "climb_stairs": "climbing", "hug": "hugging"}
HMDB_NAMES = ["brush_hair", "cartwheel", "catch", "chew", "clap", "climb", "climb_stairs", "dive", "draw_sword",
              "dribble", "drink", "eat", "fall_floor", "fencing", "flic_flac", "golf", "handstand", "hit", "hug",
              "jump", "kick_ball", "kick", "kiss", "laugh", "pick", "pour", "pullup", "punch", "push", "pushup",
              "ride_bike", "ride_horse", "run", "shake_hands", "shoot_ball", "shoot_bow", "shoot_gun", "sit", "situp",
              "smile", "smoke", "somersault", "stand", "swing_baseball", "sword_exercise", "sword", "talk", "throw",
              "turn", "walk", "wave"]
# NTU RGB+D 60 action numbers (A001 = 1)
NTU = {1: "drinking", 2: "eating", 6: "picking_up", 7: "throwing", 8: "sitting_down", 9: "standing_up",
       23: "waving", 28: "phone_call", 42: "staggering", 43: "falling", 44: "distress", 45: "distress",
       46: "distress", 47: "distress", 48: "distress", 50: "fighting", 51: "fighting", 52: "fighting",
       55: "hugging", 59: "walking", 60: "walking"}
NTU_MUTUAL = set(range(50, 61))
# Peaceful two-person NTU actions (pat on back, point finger, giving object, touch pocket, shaking hands): all kept as
# "other", so another person standing close is not by itself evidence of fighting / hugging
NTU_BENIGN_MUTUAL = {53, 54, 56, 57, 58}
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]   # COCO-17 left ↔ right


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def persons(ann: Dict[str, Any], mutual: bool) -> List[int]:
    """:return: person indices to learn from — the most confident one, both in a two-person NTU action"""
    scores = ann["keypoint_score"].reshape(ann["keypoint_score"].shape[0], -1).mean(axis=1)
    order = list(np.argsort(-scores))
    order = [i for i in order if scores[i] > 0.05]
    return order[:2] if mutual else order[:1]


def _origin(points: np.ndarray, scores: np.ndarray) -> Optional[np.ndarray]:
    good = scores >= 0.4
    hips = [i for i in (11, 12) if good[i]]
    if hips:
        return points[hips].mean(axis=0)
    return points[good].mean(axis=0) if good.any() else None


def partner_track(ann: Dict[str, Any], partner: Dict[str, Any], person: int, rng: random.Random) -> Optional[list]:
    """
    A synthetic bystander (augmentation): another clip's person placed 0.6–2 body heights beside this one, frame by
    frame (looped). Labels do not change — another person nearby alone is no evidence of fighting / hugging.
    :return: per source frame, the bystander's (17, 2) keypoints (NaN = missing), or None
    """
    kp, sc = ann["keypoint"][person], ann["keypoint_score"][person]
    pkp, psc = partner["keypoint"][0], partner["keypoint_score"][0]
    first = next((f for f in range(kp.shape[0]) if _origin(kp[f], sc[f]) is not None), None)
    pfirst = next((f for f in range(pkp.shape[0]) if _origin(pkp[f], psc[f]) is not None), None)
    if first is None or pfirst is None:
        return None
    good = sc[first] >= 0.4
    height = float(np.ptp(kp[first][good][:, 1])) if good.sum() >= 4 else 100.0
    pgood = psc[pfirst] >= 0.4
    pheight = float(np.ptp(pkp[pfirst][pgood][:, 1])) if pgood.sum() >= 4 else 100.0
    zoom = height / max(pheight, 1.0)
    side = rng.choice((-1, 1))
    target = _origin(kp[first], sc[first]) + np.array([side * rng.uniform(0.6, 2.0) * height, 0.0])
    porigin = _origin(pkp[pfirst], psc[pfirst])
    out = []
    for f in range(kp.shape[0]):
        pf = f % pkp.shape[0]
        moved = (pkp[pf] - porigin) * zoom + target
        out.append(np.where(psc[pf][:, None] >= 0.4, moved, np.nan))
    return out


def windows(ann: Dict[str, Any], person: int, fps: float, window_sec: float, stride_sec: float, min_samples: int,
            flip: bool = False, bystander: Optional[list] = None) -> List[np.ndarray]:
    kp, sc = ann["keypoint"], ann["keypoint_score"]
    step = max(1, int(round(SOURCE_FPS / fps)))
    frames = list(range(0, kp.shape[1], step))
    per_window, stride = int(round(window_sec * fps)) + 1, max(1, int(round(stride_sec * fps)))
    width = ann["img_shape"][1]
    out = []
    starts = list(range(0, max(1, len(frames) - per_window + 1), stride))
    for start in starts:
        idx = frames[start:start + per_window]
        if len(idx) < min_samples:
            continue
        samples, others = [], []
        for f in idx:
            main, score = kp[person, f].copy(), sc[person, f].copy()
            rest = [np.where(sc[o, f][:, None] >= 0.4, kp[o, f], np.nan)          # other people in the frame
                    for o in range(kp.shape[0]) if o != person and sc[o, f].mean() > 0.2]
            if bystander is not None:
                rest.append(bystander[f].copy())
            if flip:
                main, score = main[FLIP], score[FLIP]
                main[:, 0] = width - main[:, 0]
                rest = [np.column_stack([width - r[:, 0], r[:, 1]]) for r in rest]
            samples.append((f / SOURCE_FPS, main, score))
            others.append(rest)
        feature = window_features(samples, others)
        if feature is not None:
            out.append(feature)
    return out


def build(data_dir: str, fps: float, window_sec: float, stride_sec: float, min_samples: int,
          other_per_class: int, seed: int, bystander_share: float = 0.35) -> Dict[str, Any]:
    """:return: {"train" / "test": (X, y, groups)} + dataset info"""
    rng = random.Random(seed)
    sets: Dict[str, Dict[str, list]] = {"train": defaultdict(list), "test": defaultdict(list)}
    info = {}
    sources = []
    hmdb_path = os.path.join(data_dir, "hmdb51_2d.pkl")
    if os.path.isfile(hmdb_path):
        with open(hmdb_path, "rb") as f:
            data = pickle.load(f)
        train, test = set(data["split"]["train1"]), set(data["split"]["test1"])
        sources.append(("hmdb51", data, train, test, lambda a: HMDB.get(HMDB_NAMES[a["label"]], "other"),
                        lambda a: False))
        info["hmdb51"] = {"path": hmdb_path, "sha256": sha256(hmdb_path), "split": "train1 / test1",
                          "license": "CC BY-NC-SA 3.0 (research use)"}
    ntu_path = os.path.join(data_dir, "ntu60_2d.pkl")
    if os.path.isfile(ntu_path):
        with open(ntu_path, "rb") as f:
            data = pickle.load(f)
        train, test = set(data["split"]["xsub_train"]), set(data["split"]["xsub_val"])
        sources.append(("ntu60", data, train, test, lambda a: NTU.get(a["label"] + 1, "other"),
                        lambda a: a["label"] + 1 in NTU_MUTUAL))
        always_other = {id(a) for a in data["annotations"] if a["label"] + 1 in NTU_BENIGN_MUTUAL}
        info["ntu60"] = {"path": ntu_path, "sha256": sha256(ntu_path), "split": "xsub_train / xsub_val",
                         "license": "NTU RGB+D terms (non-commercial research)"}
    else:
        always_other = set()
    if not sources:
        raise SystemExit(f"no datasets in {data_dir} (see the module docstring for the download commands)")
    for name, data, train, test, label_of, mutual_of in sources:
        anns = data["annotations"]
        others = [a for a in anns if label_of(a) == "other"]
        rng.shuffle(others)
        keep_other = {id(a) for a in others[:other_per_class]} | always_other   # capped "other" + benign pairs
        singles = [a for a in anns if not mutual_of(a) and a["frame_dir"] in train]
        for ann in anns:
            label = label_of(ann)
            if label == "other" and id(ann) not in keep_other:
                continue
            part = "train" if ann["frame_dir"] in train else "test" if ann["frame_dir"] in test else None
            if part is None:
                continue
            mutual = mutual_of(ann)
            for person in persons(ann, mutual):
                bystander = None
                if part == "train" and not mutual and rng.random() < bystander_share:
                    bystander = partner_track(ann, rng.choice(singles), person, rng)
                for flip in ((False, True) if part == "train" else (False,)):
                    for feature in windows(ann, person, fps, window_sec, stride_sec, min_samples, flip,
                                           bystander if not flip else None):
                        sets[part]["X"].append(feature)
                        sets[part]["y"].append(label)
                        sets[part]["group"].append(f"{name}:{ann['frame_dir']}")
        del data
    out = {}
    for part, cols in sets.items():
        out[part] = (np.stack(cols["X"]), np.array(cols["y"]), np.array(cols["group"]))
    out["info"] = info
    return out


def candidates(seed: int) -> Dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "hgb": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                              l2_regularization=1.0, class_weight="balanced", random_state=seed),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3,
                                                             max_iter=200, early_stopping=True, random_state=seed)),
    }


def evaluate(model, X, y, classes: List[str]) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    pred = model.predict(X)
    return {"macro_f1": round(float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "report": classification_report(y, pred, labels=classes, output_dict=True, zero_division=0),
            "confusion": confusion_matrix(y, pred, labels=classes).tolist()}


def save_confusion(matrix: List[List[int]], classes: List[str], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.asarray(matrix, float)
    m = m / np.maximum(1, m.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if m[i, j] >= 0.05:
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if m[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> None:
    import joblib
    import sklearn
    from sklearn.model_selection import GroupShuffleSplit

    parser = argparse.ArgumentParser(description="Train the skeleton action classifier on public datasets")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--fps", type=float, default=5.0, help="actions.action_fps of the live system")
    parser.add_argument("--window-sec", type=float, default=2.5, help="actions.window_sec")
    parser.add_argument("--stride-sec", type=float, default=1.0)
    parser.add_argument("--min-samples", type=int, default=6)
    parser.add_argument("--other-per-class", type=int, default=1500, help="clips per dataset kept as 'other'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--report-dir", default=REPORT_DIR)
    args = parser.parse_args(argv)
    started = time.time()
    data = build(args.data_dir, args.fps, args.window_sec, args.stride_sec, args.min_samples, args.other_per_class,
                 args.seed)
    X, y, groups = data["train"]
    Xt, yt, _ = data["test"]
    classes = sorted(set(y))
    print(f"train {len(y)} windows ({len(set(groups))} clips), test {len(yt)}; {len(classes)} classes "
          f"({time.time() - started:.0f} s)")
    print(" ", dict(sorted(Counter(y).items())))
    split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    fit_idx, val_idx = next(split.split(X, y, groups))
    scores = {}
    for name, model in candidates(args.seed).items():
        t = time.time()
        model.fit(X[fit_idx], y[fit_idx])
        scores[name] = evaluate(model, X[val_idx], y[val_idx], classes)["macro_f1"]
        print(f"  {name}: validation macro-F1 {scores[name]:.3f} ({time.time() - t:.0f} s)")
    best = max(scores, key=scores.get)
    model = candidates(args.seed)[best]
    model.fit(X, y)
    test = evaluate(model, Xt, yt, classes)
    print(f"→ {best}: test macro-F1 {test['macro_f1']:.3f}, accuracy {test['accuracy']:.3f}")
    os.makedirs(os.path.dirname(args.model) or ".", exist_ok=True)
    meta = {"model": model, "classes": classes, "feature_names": FEATURE_NAMES, "fps": args.fps,
            "window_sec": args.window_sec, "min_samples": args.min_samples, "min_score": 0.4,
            "sklearn_version": sklearn.__version__,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "datasets": data["info"], "label_map": {"hmdb51": HMDB, "ntu60": {str(k): v for k, v in NTU.items()}}}
    joblib.dump(meta, args.model, compress=3)
    os.makedirs(args.report_dir, exist_ok=True)
    save_confusion(test["confusion"], classes, os.path.join(args.report_dir, "confusion_test.png"))
    report = {"selected": best, "validation_macro_f1": scores, "test": test, "classes": classes,
              "windows": {"train": int(len(y)), "test": int(len(yt))}, "train_counts": dict(Counter(y.tolist())),
              "test_counts": dict(Counter(yt.tolist())), **{k: v for k, v in meta.items() if k not in ("model",)},
              "model_size_kb": round(os.path.getsize(args.model) / 1024)}
    with open(os.path.join(args.report_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False, default=str)
    print(f"→ {args.model} ({report['model_size_kb']} KB), {args.report_dir}/metrics.json "
          f"({time.time() - started:.0f} s total)")


if __name__ == "__main__":
    main()
