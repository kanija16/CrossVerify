"""
Can the image-only CNNs read the document at their 224x224 input?

Applies the exact CNN preprocessing resize (src.baseline.transforms.AspectPreservingResizeAndPad)
to the released Family A example images and reports:
  * the QR module size and printed-text height after resizing, measured on the genuine card;
  * whether OpenCV and pyzbar decode the QR code, at full resolution and at 224 px
    (also after 4x nearest-neighbour upscaling of the 224 px input);
  * whether Tesseract reads the printed fields at full resolution and at 224 px (upscaled).

Run from the repository root:  python -m src.analysis.resolution_check
Needs opencv-python, pyzbar (+ zbar), pytesseract (+ tesseract).
"""
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pyzbar.pyzbar import decode as zbar_decode

from src.baseline.transforms import AspectPreservingResizeAndPad

ROOT = Path(__file__).resolve().parents[2]
EX = ROOT / "outputs/figures/example_images"
QR_BOX = (628, 268, 946, 586)        # QR region on the Family A genuine card (full resolution)
NAME_BOX = (280, 150, 440, 190)      # printed name line
FIELDS_BOX = (270, 140, 640, 400)    # printed fields block


def qr_ok(img):
    bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(bgr)
    return bool(text), len(zbar_decode(img)) > 0


def main():
    resize = AspectPreservingResizeAndPad(target_size=(224, 224), fill=0)
    g = Image.open(EX / "genuine.png").convert("RGB")
    scale = 224 / g.width

    q = np.asarray(g.convert("L").crop(QR_BOX)) < 128
    row = q[3]
    first_dark_run = np.argmax(~row)                    # finder pattern = 7 modules
    module = first_dark_run / 7
    t = np.asarray(g.convert("L").crop(NAME_BOX)) < 128
    rows = np.flatnonzero(t.any(1))
    text_h = rows.max() - rows.min() + 1
    print(f"Family A card {g.size} -> scale {scale:.3f} at the 224 px input")
    print(f"QR: {module:.1f} px/module, {round(q.shape[1] / module)} modules -> {module * scale:.2f} px/module at 224")
    print(f"printed text height: {text_h} px -> {text_h * scale:.1f} px at 224")

    for key in ["genuine", "text_qr_mismatch", "checksum_invalid"]:
        im = Image.open(EX / f"{key}.png").convert("RGB")
        small = resize(im)
        up = small.resize((896, 896), Image.NEAREST)
        ocr_full = " ".join(pytesseract.image_to_string(im.crop(FIELDS_BOX)).split())
        box = tuple(int(round(c * scale)) + (0, 40, 0, 40)[i] for i, c in enumerate(FIELDS_BOX))
        crop = small.crop(box)
        ocr_small = " ".join(pytesseract.image_to_string(crop.resize((crop.width * 4, crop.height * 4), Image.LANCZOS)).split())
        print(f"\n{key}")
        print(f"  QR decoded (OpenCV, pyzbar): full-res {qr_ok(im)}  224 px {qr_ok(small)}  224 px upscaled x4 {qr_ok(up)}")
        print(f"  OCR full-res: {ocr_full[:80]!r}")
        print(f"  OCR 224 px (upscaled x4): {ocr_small[:80]!r}")


if __name__ == "__main__":
    main()
