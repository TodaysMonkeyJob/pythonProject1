"""
Cigarette detector: yolo11n fine-tuned on CigDet → ``models/cigarette_yolo11n.pt`` (+ ``reports/cigarette_detector/``).

CigDet (Mendeley Data, DOI 10.17632/6hyrr8typ7.1, CC BY 4.0): 557 smoking photos, one class ``cigarette``, YOLO
labels next to the images (446 train / 111 test). The live action module runs the detector on a crop around a
person's head and hands (from the pose keypoints), the same close-up framing as these photos.

    curl -L -o data/action_datasets/CigDet_dataset.zip \\
      https://data.mendeley.com/public-files/datasets/6hyrr8typ7/files/24e3327e-155e-4565-ab44-8b45dc6445ee/file_downloaded
    (cd data/action_datasets && mkdir -p cigdet && cd cigdet && unzip ../CigDet_dataset.zip)
    python -m tools.train_cigarette --epochs 50          # CPU (Ultralytics on MPS did not converge)
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from typing import List, Optional

if __package__ in (None, ""):  # allow `python tools/train_cigarette.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = "data/action_datasets/cigdet/CigDet_dataset"
OUT = "models/cigarette_yolo11n.pt"
REPORT_DIR = "reports/cigarette_detector"


def main(argv: Optional[List[str]] = None) -> None:
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description="Fine-tune yolo11n on CigDet (cigarette detection)")
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--base", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--runs", default="/private/tmp/claude-501/actions/runs", help="Ultralytics run folder")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--report-dir", default=REPORT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="cpu (default) | mps | cuda index")
    parser.add_argument("--live-imgsz", type=int, default=320, help="actions.cigarette_imgsz (best-F1 confidence)")
    args = parser.parse_args(argv)
    root = os.path.abspath(args.root)
    data_yaml = os.path.join(os.path.dirname(root), "cigdet.yaml")
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {root}\ntrain: train\nval: test\nnames:\n  0: cigarette\n")
    # CPU on purpose: Ultralytics training on Apple MPS did not converge here (mAP50 0.06 after 30 epochs)
    device = args.device or "cpu"
    model = YOLO(args.base)
    model.train(data=data_yaml, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
                project=args.runs, name="cigdet", exist_ok=True, seed=args.seed, deterministic=True, patience=25,
                plots=True, verbose=False, workers=2)
    best = os.path.join(args.runs, "cigdet", "weights", "best.pt")
    metrics = YOLO(best).val(data=data_yaml, imgsz=args.imgsz, device=device, project=args.runs, name="cigdet_val",
                             exist_ok=True, verbose=False)
    import numpy as np

    live = YOLO(best).val(data=data_yaml, imgsz=args.live_imgsz, device=device, project=args.runs, name="cigdet_live",
                          exist_ok=True, verbose=False, plots=False)          # the live crop size (actions config)
    f1, px = np.asarray(live.box.f1_curve)[0], np.asarray(live.box.px)
    best_conf = round(float(px[int(np.argmax(f1))]), 3)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    shutil.copy(best, args.out)
    os.makedirs(args.report_dir, exist_ok=True)
    for plot in ("PR_curve.png", "confusion_matrix.png", "results.png"):
        for folder in ("cigdet_val", "cigdet"):
            src = os.path.join(args.runs, folder, plot)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(args.report_dir, plot))
                break
    def images(part):
        return len([p for p in os.listdir(os.path.join(root, part)) if p.endswith(".jpg")])

    report = {"dataset": {"name": "CigDet", "doi": "10.17632/6hyrr8typ7.1", "license": "CC BY 4.0",
                          "train_images": images("train"), "test_images": images("test")},
              "base": args.base, "epochs": args.epochs, "imgsz": args.imgsz, "device": device,
              "test": {"precision": round(float(metrics.box.mp), 4), "recall": round(float(metrics.box.mr), 4),
                       "mAP50": round(float(metrics.box.map50), 4), "mAP50_95": round(float(metrics.box.map), 4)},
              "live": {"imgsz": args.live_imgsz, "mAP50": round(float(live.box.map50), 4),
                       "best_f1": round(float(f1.max()), 4), "best_f1_conf": best_conf,
                       "note": "actions.cigarette_conf = best_f1_conf (the CigDet test split, not the eval videos)"},
              "model": args.out, "model_size_kb": round(os.path.getsize(args.out) / 1024),
              "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(os.path.join(args.report_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report["test"]), f"→ {args.out}")


if __name__ == "__main__":
    main()
