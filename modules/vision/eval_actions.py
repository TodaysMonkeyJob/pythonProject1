"""
Offline evaluation of action recognition on labelled videos.

Manifest CSV, two forms:

* per video — ``video,expected`` (acceptance: 10 videos, 2 per action);
* per segment — ``video,start,end,expected[,note]`` for edited clips with several scenes
  (e.g. ``data/human_behaviour_video/labels.csv``).

``expected`` may list alternatives (``lying|sitting``). Each video runs once through the live pipeline
(YOLO + ByteTrack persons → pose → ActionEngine) sampled at ``actions.action_fps``; letterbox bars are cropped.
In every sample the largest person with a pose is the "main" person.

* posture labels: score = share of samples with a confirmed action whose action is expected
  (coverage = share of samples with any confirmed action); passes at ≥ 0.8;
* ``possible_smoking`` (per video): passes when a possible_smoking event was raised;
* ``hand_to_mouth`` (per segment): passes when the main person's wrist was near the mouth at least once —
  the building block of the smoking rule, for clips too short / too edited to reach the 3-in-30 s rule;
* videos without any smoking label: counted hand-to-mouth episodes per minute and possible_smoking events
  (false alarms).

    python -m modules.vision.eval_actions data/human_behaviour_video/labels.csv
    python -m modules.vision.eval_actions data/human_behaviour_video/labels.csv \
        --classifier models/action_classifier.joblib --cigarette models/cigarette_yolo11n.pt   # + the learned layer

With the learned layer, hand-to-mouth segments also report whether ``possible_smoking`` was raised there (and the
classifier's highest smoking probability), every segment lists the classifier's actions, and the false alarms of the
non-smoking videos count the combined smoking decision.
"""
import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.config import load_config
from modules.vision.actions import SMOKING, ActionConfig, ActionEngine, match_poses
from modules.vision.pose import PoseEstimator, wrist_to_mouth
from modules.vision.yolo import ObjectDetector

PASS_SCORE = 0.8
HAND_TO_MOUTH = "hand_to_mouth"


def detect_letterbox(path: str, samples: int = 20, threshold: int = 20) -> Tuple[int, int]:
    """
    :return: (top, bottom) rows of the picture area, excluding black bars present in every sampled frame
    """
    cap = cv2.VideoCapture(path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    row_max = None
    for i in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * count / samples))
        ok, frame = cap.read()
        if not ok:
            continue
        rows = frame.max(axis=(1, 2))
        row_max = rows if row_max is None else np.maximum(row_max, rows)
    cap.release()
    if row_max is None:
        return 0, 0
    bright = np.where(row_max > threshold)[0]
    if len(bright) == 0:
        return 0, len(row_max)
    return int(bright[0]), int(bright[-1]) + 1


def analyse_video(path: str, detector, estimator, action_cfg: ActionConfig, action_fps: float,
                  crop_letterbox: bool = True, classifier=None, cigarette=None) -> List[Dict[str, Any]]:
    """
    Runs the pipeline over the whole video.

    :return: One record per sampled frame: t, main person (track id, action, near_mouth) and events
    """
    top, bottom = detect_letterbox(path) if crop_letterbox else (0, 0)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    stride = max(1, round(video_fps / action_fps))
    engine = ActionEngine(action_cfg, classifier, cigarette)
    records, frame_index = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % stride == 0:
            if bottom > top:
                frame = frame[top:bottom]
            t = frame_index / video_fps
            persons = [{"track_id": d.track_id, "bbox": d.bbox} for d in detector.track(frame)
                       if d.cls == "person" and d.track_id is not None]
            poses = estimator.estimate(frame) if persons else []
            episodes_before = {tid: len(tr.mouth_events) for tid, tr in engine.tracks.items()}
            events = engine.update(t, persons, poses, frame_size=(frame.shape[1], frame.shape[0]), frame=frame)
            episodes = sum(max(0, len(tr.mouth_events) - episodes_before.get(tid, 0))
                           for tid, tr in engine.tracks.items())
            main = None
            matched = match_poses(persons, poses, action_cfg.match_iou)
            if matched:
                track_id, pose = max(matched, key=lambda m: (m[1].bbox[2] - m[1].bbox[0])
                                     * (m[1].bbox[3] - m[1].bbox[1]))
                track = engine.tracks.get(track_id)
                distance = wrist_to_mouth(pose.keypoints, pose.bbox)
                main = {"track_id": track_id, "action": track.action if track else None,
                        "near_mouth": not math.isnan(distance) and distance <= action_cfg.mouth_radius,
                        "activity": track.activity if track else None,
                        "smoking": bool(track.smoking) if track else False,
                        "smoking_prob": round(track.proba.get("smoking", 0.0), 3) if track and track.proba else None,
                        "cigarette": round(track.cigarette_score, 3) if track else 0.0}
            records.append({"t": round(t, 3), "main": main, "events": [p for p, _, _ in events],
                            "mouth_episodes": episodes})
        frame_index += 1
    cap.release()
    return records


def score_segment(records: List[Dict[str, Any]], expected: str, start: float = 0.0,
                  end: float = math.inf) -> Dict[str, Any]:
    """
    :return: Score of one labelled time range
    """
    window = [r for r in records if start <= r["t"] < end]
    allowed = set(expected.split("|"))
    if expected == SMOKING:
        raised = any(e["action"] == SMOKING for r in window for e in r["events"])
        return {"score": float(raised), "passed": raised, "coverage": 1.0, "actions": {}}
    if expected == HAND_TO_MOUTH:
        near = sum(bool(r["main"] and r["main"]["near_mouth"]) for r in window)
        smoking = [e for r in window for e in r["events"] if e["action"] == SMOKING]
        probs = [r["main"]["smoking_prob"] for r in window if r["main"] and r["main"].get("smoking_prob") is not None]
        cigarettes = [r["main"].get("cigarette", 0.0) for r in window if r["main"]]
        return {"score": float(near > 0), "passed": near > 0, "coverage": 1.0,
                "actions": {"near_mouth_samples": near, "samples": len(window)},
                "smoking_raised": bool(smoking) or any(r["main"] and r["main"].get("smoking") for r in window),
                "smoking_evidence": sorted({x for e in smoking
                                                                            for x in e.get("evidence", [])}),
                "smoking_prob_max": max(probs) if probs else None,
                "cigarette_max": max(cigarettes) if cigarettes else 0.0}
    decisions = [r["main"]["action"] for r in window if r["main"] and r["main"]["action"]]
    score = sum(a in allowed for a in decisions) / len(decisions) if decisions else 0.0
    activities = [r["main"]["activity"] for r in window if r["main"] and r["main"].get("activity")]
    return {"score": round(score, 3), "passed": score >= PASS_SCORE,
            "coverage": round(len(decisions) / len(window), 3) if window else 0.0, "actions": dict(Counter(decisions)),
            "activities": dict(Counter(activities))}


def smoking_alarm_evidence(records: List[Dict[str, Any]]) -> List[List[str]]:
    """:return: the evidence of every possible_smoking event (false alarms when the video has no smoking)"""
    return [e.get("evidence", ["hand_to_mouth"]) for r in records for e in r["events"] if e["action"] == SMOKING]


def smoking_false_alarms(records: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    """
    For videos without smoking: counted hand-to-mouth episodes (after hysteresis) and possible_smoking events.

    :return: (episodes, possible_smoking events, minutes analysed)
    """
    episodes = sum(r["mouth_episodes"] for r in records)
    alarms = sum(e["action"] == SMOKING for r in records for e in r["events"])
    span = (records[1]["t"] - records[0]["t"]) if len(records) > 1 else 0.2
    return episodes, alarms, len(records) * span / 60


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate action recognition on labelled videos")
    parser.add_argument("manifest")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", help="Overrides vision.model (detector)")
    parser.add_argument("--pose-model", help="Overrides actions.pose_model")
    parser.add_argument("--imgsz", type=int, help="Overrides the inference size of both models")
    parser.add_argument("--no-letterbox-crop", action="store_true")
    parser.add_argument("--classifier", help="models/action_classifier.joblib (learned actions + smoking evidence)")
    parser.add_argument("--cigarette", help="models/cigarette_yolo11n.pt (cigarette in the head / hands crop)")
    parser.add_argument("--out", default="reports/actions_eval.json")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    from modules.vision.action_classifier import ActionClassifier, CigaretteDetector

    classifier = ActionClassifier.from_path(args.classifier)
    cigarette = CigaretteDetector.from_config({**config.get("actions", {}), "cigarette_model": args.cigarette}) \
        if args.cigarette else None
    vision, actions = config["vision"], config["actions"]
    action_cfg = ActionConfig.from_dict(actions)
    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    base = os.path.dirname(os.path.abspath(args.manifest))
    by_video: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_video[row["video"]].append(row)

    results, mouth_episodes, smoking_alarms, mouth_minutes, alarm_evidence = [], 0, 0, 0.0, Counter()
    for video, labels in by_video.items():
        path = video if os.path.isabs(video) or os.path.exists(video) else os.path.join(base, video)
        # Fresh models per video so ByteTrack ids do not leak between clips
        detector = ObjectDetector(args.model or vision["model"], args.imgsz or vision.get("imgsz", 320),
                                  vision.get("confidence_threshold", 0.4), [0], vision.get("tracker", "bytetrack.yaml"))
        estimator = PoseEstimator(args.pose_model or actions["pose_model"], args.imgsz or actions.get("imgsz", 320),
                                  actions.get("pose_conf", 0.4), actions.get("keypoint_conf", 0.4))
        records = analyse_video(path, detector, estimator, action_cfg, float(actions.get("action_fps", 5)),
                                crop_letterbox=not args.no_letterbox_crop, classifier=classifier,
                                cigarette=cigarette)
        for row in labels:
            start = float(row.get("start") or 0)
            end = float(row["end"]) if row.get("end") else math.inf
            result = {"video": video, "start": start, "end": end if end != math.inf else None,
                      "expected": row["expected"], "note": row.get("note", ""),
                      **score_segment(records, row["expected"], start, end)}
            results.append(result)
            print(f"{'PASS' if result['passed'] else 'FAIL'}  {result['score']:.2f} cov={result['coverage']:.2f}  "
                  f"{row['expected']:16} {os.path.basename(video)[:38]:38} {start:6.1f}-{row.get('end') or 'end':>6}  "
                  f"{result['actions']}")
        smoking_ranges = [(float(r.get("start") or 0), float(r["end"]) if r.get("end") else math.inf)
                          for r in labels if r["expected"] in (SMOKING, HAND_TO_MOUTH)]
        if not smoking_ranges:
            episodes, alarms, minutes = smoking_false_alarms(records)
            mouth_episodes += episodes
            smoking_alarms += alarms
            mouth_minutes += minutes
            alarm_evidence.update("+".join(e) for e in smoking_alarm_evidence(records))

    posture = [r for r in results if r["expected"] not in (SMOKING, HAND_TO_MOUTH)]
    mouth = [r for r in results if r["expected"] == HAND_TO_MOUTH]
    summary = {
        "segments": len(results),
        "passed": sum(r["passed"] for r in results),
        "posture": {"segments": len(posture), "passed": sum(r["passed"] for r in posture),
                    "mean_score": round(float(np.mean([r["score"] for r in posture])), 3) if posture else None,
                    "mean_coverage": round(float(np.mean([r["coverage"] for r in posture])), 3) if posture else None},
        "hand_to_mouth": {"segments": len(mouth), "detected": sum(r["passed"] for r in mouth),
                          "smoking_raised": sum(bool(r.get("smoking_raised")) for r in mouth)},
        "learned_layer": {"classifier": args.classifier if classifier else None,
                          "cigarette": args.cigarette if cigarette else None},
        "non_smoking_videos": {"minutes": round(mouth_minutes, 1),
                               "counted_mouth_episodes_per_min": round(mouth_episodes / mouth_minutes, 2)
                               if mouth_minutes else None,
                               "possible_smoking_false_alarms": smoking_alarms,
                               "false_alarm_evidence": dict(alarm_evidence)},
        "target": PASS_SCORE,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
