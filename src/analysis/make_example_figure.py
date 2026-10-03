"""
Example-documents figure: one synthetic identity in its genuine form and under each
forgery mechanism. Boxes mark what was changed.

The images are dataset images of one synthetic identity, used unchanged (only scaled for
display); copies are in outputs/figures/example_images/, named <tamper_type>.png.

    python -m src.analysis.make_example_figure

Writes outputs/figures/fig_examples.pdf and .png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/figures"
EDIT = "#d62728"   # box colour for the changed region

# Regions in Family A pixel coordinates (1000 x 640): (x0, y0, x1, y1)
NAME, DOB, GENDER, IDNUM = (280, 149, 442, 186), (280, 217, 412, 254), (280, 285, 382, 322), (280, 353, 450, 390)
QR, FACE = (616, 256, 958, 598), (95, 195, 200, 300)

PANELS = [
    ("genuine", "Genuine", "print, QR, checksum and\nformat all agree", []),
    ("text_qr_mismatch", "Text\u2013QR mismatch", "printed name changed;\nQR keeps the original", [NAME]),
    ("qr_only_mismatch", "QR-only mismatch", "QR payload changed (gender);\nprint unchanged", [QR]),
    ("checksum_invalid", "Checksum invalid", "last ID digit changed;\nVerhoeff check fails", [IDNUM]),
    ("format_invalid", "Format invalid", "ID leading digit not allowed\nby the format", [IDNUM]),
    ("field_missing", "Field missing", "date of birth removed\nfrom the print", [DOB]),
    ("fine_grained_edit", "Fine-grained edit", "one character changed\n(Female \u2192 Wemale)", [GENDER]),
    ("visual_splice_3", "Visual splice", "copy\u2013move patch over\nthe face (record unchanged)", [FACE]),
    ("coordinated_full_forgery", "Coordinated full forgery", "whole record replaced;\nprint, QR and ID all agree", []),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "outputs/figures/example_images"),
                    help="folder with <tamper_type>.png files (default: the released example images)")
    args = ap.parse_args()
    folder = Path(args.images)

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 5.9))
    for ax, (key, title, note, boxes) in zip(axes.flat, PANELS):
        img = Image.open(folder / f"{key}.png").convert("RGB")
        ax.imshow(img, interpolation="lanczos")
        for (x0, y0, x1, y1) in boxes:
            ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                           edgecolor=EDIT, linewidth=1.3))
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=3)
        ax.text(0.5, -0.03, note, transform=ax.transAxes, ha="center", va="top", fontsize=6.5, linespacing=1.15,
                color="#333333")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#999999"); s.set_linewidth(0.5)
    fig.subplots_adjust(wspace=0.06, hspace=0.42)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_examples.pdf")
    fig.savefig(OUT / "fig_examples.png", dpi=300)
    print(f"written {OUT / 'fig_examples.pdf'}")


if __name__ == "__main__":
    main()
