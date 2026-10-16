"""
Downloads the OpenCV Zoo face models (YuNet detector, SFace recognizer) into models/ and checks their SHA-256.

    python -m modules.vision.face_models
"""
import argparse
import hashlib
import os
import urllib.request
from typing import List, Optional

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "face_recognition_sface_2021dec.onnx": (
        f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(out_dir: str = "models") -> List[str]:
    """
    :return: Paths of the verified model files
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, (url, expected) in MODELS.items():
        path = os.path.join(out_dir, name)
        if not os.path.isfile(path) or sha256(path) != expected:
            tmp = path + ".part"
            urllib.request.urlretrieve(url, tmp)
            if sha256(tmp) != expected:
                os.remove(tmp)
                raise RuntimeError(f"Checksum mismatch for {name}")
            os.replace(tmp, path)
        paths.append(path)
    return paths


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Download YuNet + SFace face models")
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args(argv)
    for path in download(args.out_dir):
        print(path)


if __name__ == "__main__":
    main()
