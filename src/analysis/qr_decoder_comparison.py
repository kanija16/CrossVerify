"""
Compare M3's QR decoding (OpenCV QRCodeDetector, with the same corner-crop/upscale
fallback used in src/m3_crossmodal/live_qr.py) against pyzbar (used by the dataset
generator) on the same images.

Usage:
    DATASET_ROOT=/path/to/dataset python -m src.analysis.qr_decoder_comparison [--split test]

Without --split, every PNG under DATASET_ROOT/images is checked.
Requires: opencv-python, pyzbar (+ system zbar library), pillow, pandas.
"""
import argparse
import os
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image
from pyzbar.pyzbar import decode as zbar_decode

from src.m3_crossmodal.live_qr import QR_REGION_HINTS, _CROP_UPSCALE_ATTEMPTS, _crop_corner

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_detector = cv2.QRCodeDetector()


def opencv_readable(path: Path, family: str) -> bool:
    bgr = cv2.imread(str(path))
    data, _, _ = _detector.detectAndDecode(bgr)
    if data:
        return True
    crop = _crop_corner(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), QR_REGION_HINTS[family]["corner"])
    for factor in _CROP_UPSCALE_ATTEMPTS:
        up = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        data, _, _ = _detector.detectAndDecode(cv2.cvtColor(up, cv2.COLOR_RGB2BGR))
        if data:
            return True
    return False


def pyzbar_readable(path: Path) -> bool:
    return bool(zbar_decode(Image.open(path).convert("RGB")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], default=None)
    args = ap.parse_args()
    root = Path(os.environ.get("DATASET_ROOT", PROJECT_ROOT / "dataset"))

    if args.split:
        df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / f"{args.split}.csv")
        paths = [root / p for p in df["image_path"]]
    else:
        paths = sorted(root.glob("images/*/*.png"))

    rows = []
    for p in paths:
        family = "family_a" if p.name.startswith("family_a") else "family_b"
        tamper = p.stem.split("_", 3)[-1]
        rows.append({"id": p.stem, "tamper_type": tamper,
                     "opencv": opencv_readable(p, family), "pyzbar": pyzbar_readable(p)})
    res = pd.DataFrame(rows)
    n = len(res)
    print(f"images checked: {n}")
    print(f"OpenCV (M3) readable: {res.opencv.sum()} ({res.opencv.mean():.1%})")
    print(f"pyzbar readable:      {res.pyzbar.sum()} ({res.pyzbar.mean():.1%})")
    print(f"OpenCV failures that pyzbar decodes: {(~res.opencv & res.pyzbar).sum()}")
    print(res.groupby("tamper_type")[["opencv", "pyzbar"]].mean().round(3))
    out = PROJECT_ROOT / "outputs" / "revision_analysis" / "qr_decoder_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"saved {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
