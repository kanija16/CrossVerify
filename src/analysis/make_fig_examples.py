"""Rebuild Figure 1 (fig_examples.pdf) from the full-resolution 1000x640 cards.

Images are embedded unresampled (interpolation='none'), so each card keeps
its native 1000x640 pixels (~520 ppi at the printed size).
Red boxes mark the altered region; coordinates are in card pixels and were
located from OCR word boxes / pixel differences against the genuine card.
"""
import sys
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

SRC = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "fig_examples.pdf"

plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})

PAD = 6
panels = [
    ("Genuine", "000000_genuine",
     "print, QR, checksum and\nformat all agree", None),
    ("Text–QR mismatch", "000001_text_qr_mismatch",
     "printed name changed;\nQR keeps the original", (291, 158, 418, 178)),
    ("QR-only mismatch", "000002_qr_only_mismatch",
     "QR payload changed (gender);\nprint unchanged", (628, 268, 945, 585)),
    ("Checksum invalid", "000003_checksum_invalid",
     "last ID digit changed;\nVerhoeff check fails", (290, 362, 433, 378)),
    ("Format invalid", "000004_format_invalid",
     "ID leading digit not allowed\nby the format", (290, 362, 433, 378)),
    ("Field missing", "000005_field_missing",
     "date of birth removed\nfrom the print", (291, 226, 397, 242)),
    ("Fine-grained edit", "000006_fine_grained_edit",
     "one character changed\n(Female → Wemale)", (290, 294, 369, 310)),
    ("Visual splice", "000008_visual_splice",
     "copy–move patch over\nthe face (record unchanged)", (116, 228, 192, 280)),
    ("Coordinated full forgery", "000007_coordinated_full_forgery",
     "all fields replaced, portrait kept;\nprint, QR and ID all agree", None),
]

fig, axes = plt.subplots(3, 3, figsize=(6.6, 6.1))
fig.subplots_adjust(left=0.01, right=0.99, top=0.955, bottom=0.075,
                    wspace=0.06, hspace=0.55)

for ax, (title, stem, caption, box) in zip(axes.flat, panels):
    img = Image.open(f"{SRC}/family_a_{stem}.png").convert("RGB")
    assert img.size == (1000, 640), (stem, img.size)
    ax.imshow(img, interpolation="none")
    if box:
        x0, y0, x1, y1 = box
        ax.add_patch(Rectangle((x0 - PAD, y0 - PAD), x1 - x0 + 2 * PAD,
                               y1 - y0 + 2 * PAD, fill=False,
                               edgecolor="#d40000", linewidth=1.4))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#888888"); s.set_linewidth(0.6)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=3)
    ax.text(0.5, -0.04, caption, transform=ax.transAxes, ha="center",
            va="top", fontsize=7, color="#222222", linespacing=1.15)

fig.savefig(OUT)
print("wrote", OUT)
