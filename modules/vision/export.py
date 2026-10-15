"""
Exports a YOLO .pt model to NCNN (fast on ARM CPUs) into models/.

``python -m modules.vision.export --model yolo11n.pt --imgsz 320`` → ``models/yolo11n_ncnn_model``
(the .pt file is downloaded by ultralytics if it is not present).
"""
import argparse
import os
import shutil
from typing import List, Optional


def export_ncnn(model: str = "yolo11n.pt", imgsz: int = 320, out_dir: str = "models") -> str:
    """
    :param model: .pt model name or path
    :param imgsz: Inference size baked into the export
    :param out_dir: Directory for the exported model
    :return: Path of the NCNN model directory
    """
    from ultralytics import YOLO

    exported = YOLO(model).export(format="ncnn", imgsz=imgsz)
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, os.path.basename(str(exported).rstrip("/")))
    if os.path.abspath(exported) != os.path.abspath(target):
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(exported), target)
    return target


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Export a YOLO model to NCNN")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args(argv)
    print(export_ncnn(args.model, args.imgsz, args.out_dir))


if __name__ == "__main__":
    main()
