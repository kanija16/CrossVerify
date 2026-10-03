"""
Render a fictional document image from a record, dispatched by the family's
configured layout name. Each layout is an ORIGINAL design (not modeled on
any real government document's layout) and every layout has two visually
distinct template_variants (used for the unseen-template generalization
split).

Family A ("identity_v1"): portrait-style card, photo box, blue header.
Family B ("registry_v1"): landscape ledger-style document, no photo box,
table-style field rows, green/grey header — deliberately different visual
grammar from Family A so a CNN can't trivially use "photo box present" as a
proxy label between families.
"""
import hashlib
import textwrap

import qrcode
from PIL import Image, ImageDraw, ImageFont

LAYOUT_REGISTRY = {}

# Regions containing document content, in the coordinate system of each
# layout. Keeping these next to the layout registry makes visual tampering
# configurable by layout rather than tied to family names.
INFORMATIVE_REGION_REGISTRY = {
    "identity_v1": [
        (30, 120, 260, 380),    # photo box
        (290, 120, 550, 315),   # name, date, and gender fields
        (290, 395, 550, 480),   # address field; excludes identifier line
    ],
    "registry_v1": [
        (30, 236, 700, 410),    # non-identifier table content
    ],
}


def register_layout(name):
    def deco(fn):
        LAYOUT_REGISTRY[name] = fn
        return fn
    return deco


def get_informative_regions(config: dict):
    """Return content-bearing regions for the configured document layout."""
    return INFORMATIVE_REGION_REGISTRY[config["layout"]]


def _font(size, bold=False):
    """Look up a system font across platforms. Falls back to PIL's basic
    default font if none of these paths exist (still functional, just
    plainer-looking cards -- doesn't affect OCR/checksum/QR logic)."""
    candidates = [
        # macOS
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        # Windows
        "C:\\Windows\\Fonts\\arialbd.ttf" if bold else "C:\\Windows\\Fonts\\arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_wrapped_field(draw, x, y, value, font, fill, wrap_width, line_height):
    """Draw the full field value, wrapping instead of truncating.

    Truncating (``value[:N]``) silently drops characters that the QR payload
    still contains in full, which manufactures a spurious text/QR mismatch
    on genuine, untampered documents whenever a field runs long. Wrapping
    guarantees the printed text always contains the complete value, so OCR
    can recover it in full and text/QR consistency reflects reality rather
    than a rendering artifact. Returns the number of lines drawn so the
    caller can reserve enough vertical space for the next field.
    """
    lines = textwrap.wrap(value, width=wrap_width) or [""]
    for line_index, line in enumerate(lines):
        draw.text((x, y + line_index * line_height), line, font=font, fill=fill)
    return len(lines)


def render_qr(payload: str, module_px: int = 6) -> Image.Image:
    """Render a crisp QR without post-render resampling.

    Dense payloads need a large enough module pitch and a proper quiet zone
    to survive the shared image-quality augmentation. Keeping the native
    module grid avoids uneven modules introduced by resizing.
    """
    qr = qrcode.QRCode(
        border=4,
        box_size=module_px,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGB")


def _synthetic_portrait(record_id: str, size=(224, 254)) -> Image.Image:
    """Create a deterministic, illustrated non-identifying avatar.

    A SHA-256 digest of the synthetic record ID selects simple palette and
    feature variations.  It never uses photographs, external assets, or a
    random generator, so every variant of one record receives the same avatar.
    """
    digest = hashlib.sha256(record_id.encode("utf-8")).digest()
    palettes = [
        ((229, 214, 194), (65, 46, 36), (57, 105, 160)),
        ((198, 145, 108), (38, 29, 25), (165, 78, 76)),
        ((151, 97, 66), (25, 22, 20), (76, 128, 96)),
        ((239, 193, 155), (105, 70, 42), (132, 91, 158)),
    ]
    skin, hair, shirt = palettes[digest[0] % len(palettes)]
    img = Image.new("RGB", size, (226, 234, 242))
    draw = ImageDraw.Draw(img)
    w, h = size
    # Soft background blocks and shoulders make the portrait useful content
    # for copy-move splicing while remaining unmistakably illustrative.
    draw.rectangle([0, int(h * .62), w, h], fill=(205, 216, 229))
    draw.ellipse([int(w * .10), int(h * .67), int(w * .90), int(h * 1.25)], fill=shirt)
    draw.ellipse([int(w * .25), int(h * .16), int(w * .75), int(h * .76)], fill=skin)
    hair_top = int(h * (.10 + (digest[1] % 8) / 100))
    draw.ellipse([int(w * .22), hair_top, int(w * .78), int(h * .49)], fill=hair)
    draw.ellipse([int(w * .27), int(h * .24), int(w * .73), int(h * .74)], fill=skin)
    eye_y = int(h * .43)
    eye_dx = int(w * .14)
    eye_r = 4 + digest[2] % 3
    for cx in (w // 2 - eye_dx, w // 2 + eye_dx):
        draw.ellipse([cx - eye_r, eye_y - eye_r, cx + eye_r, eye_y + eye_r], fill=(35, 35, 35))
    nose_x = w // 2
    draw.line([(nose_x, eye_y + 8), (nose_x - 4, eye_y + 28), (nose_x + 5, eye_y + 28)], fill=(115, 76, 58), width=2)
    mouth_y = int(h * .62)
    smile = 4 + digest[3] % 7
    draw.arc([int(w * .39), mouth_y - smile, int(w * .61), mouth_y + smile], 0, 180, fill=(120, 54, 58), width=3)
    return img


# ---------------------------------------------------------------------------
# Family A layout
# ---------------------------------------------------------------------------
@register_layout("identity_v1")
def _render_identity_v1(fields: dict, qr_img: Image.Image, config: dict, template_variant: str):
    W, H = 1000, 640
    header_color = (30, 60, 120) if template_variant == "v1" else (90, 40, 130)
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, W, 90], fill=header_color)
    draw.text((30, 25), config["display_name"].upper(), font=_font(30, bold=True), fill="white")

    draw.rectangle([30, 120, 260, 380], outline=(120, 120, 120), width=3)
    portrait = _synthetic_portrait(str(fields.get("record_id", "synthetic")))
    img.paste(portrait, (33, 123))

    labels = config["field_labels"]
    order = ["name", "dob", "gender", "id_number", "address"]
    y = 130
    value_font = _font(22)
    for field in order:
        label = labels[field]
        value = str(fields.get(field, ""))
        draw.text((300, y), f"{label}:", font=_font(22, bold=True), fill=(20, 20, 20))
        # Every field wraps rather than truncates -- see _draw_wrapped_field.
        line_count = _draw_wrapped_field(
            draw, 290, y + 24, value, value_font, (0, 0, 0),
            wrap_width=27, line_height=24,
        )
        y += max(68, 24 + line_count * 24 + 8)

    # Reserve a right-hand column for the native QR rather than shrinking it.
    img.paste(qr_img, (W - 30 - qr_img.width, H - 30 - qr_img.height))
    return img


# ---------------------------------------------------------------------------
# Family B layout
# ---------------------------------------------------------------------------
@register_layout("registry_v1")
def _render_registry_v1(fields: dict, qr_img: Image.Image, config: dict, template_variant: str):
    W, H = 1150, 520
    header_color = (40, 90, 70) if template_variant == "v1" else (100, 90, 30)
    img = Image.new("RGB", (W, H), (250, 250, 247))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, W, 80], fill=header_color)
    draw.text((30, 20), config["display_name"].upper(), font=_font(26, bold=True), fill="white")
    draw.text((30, 52), "CERTIFICATE OF REGISTRATION (FICTIONAL)", font=_font(14), fill=(230, 230, 230))

    # QR block top-right instead of bottom-right — different visual grammar.
    # Paste the native module grid with its quiet zone intact.
    img.paste(qr_img, (W - 30 - qr_img.width, 95))

    labels = config["field_labels"]
    order = ["full_name", "registry_id", "entity_type", "jurisdiction_code", "registration_date"]
    y = 120
    row_h = 58
    value_font = _font(19)
    table_left, table_right = 30, 700
    for field in order:
        label = labels[field]
        value = str(fields.get(field, ""))
        # Every field wraps rather than truncates -- see _draw_wrapped_field.
        wrapped_lines = textwrap.wrap(value, width=45) or [""]
        row_height = max(row_h, 34 + len(wrapped_lines) * 22)
        draw.rectangle([table_left, y, table_right, y + row_height - 8], outline=(200, 200, 195), width=1)
        draw.text((table_left + 15, y + 6), label, font=_font(15, bold=True), fill=(80, 80, 80))
        _draw_wrapped_field(
            draw, table_left + 15, y + 28, value, value_font, (15, 15, 15),
            wrap_width=45, line_height=22,
        )
        y += row_height

    draw.text((30, H - 30), "This is a fictional document generated for research purposes only.",
              font=_font(12), fill=(140, 140, 140))
    return img


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def render_card(record: dict, config: dict, field_overrides: dict = None,
                 qr_payload_override: str = None, template_variant: str = "v1") -> Image.Image:
    fields = dict(record)
    if field_overrides:
        fields.update(field_overrides)

    payload = qr_payload_override if qr_payload_override is not None else record["qr_payload"]
    qr_img = render_qr(payload)

    layout_fn = LAYOUT_REGISTRY[config["layout"]]
    return layout_fn(fields, qr_img, config, template_variant)