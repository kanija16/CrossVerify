"""
Visual-only augmentations: copy-move splice (for the visual_splice tamper
family) and post-processing noise (blur/jpeg/scale) applied to ALL images so
the CNN doesn't learn to key off compression artifacts as a shortcut.
"""
import io
import random
import hashlib

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# Mild per-splice intermediate recompression.  ``copy_move_patch`` accepts
# an explicit quality for controlled experiments; otherwise it samples this
# inclusive range from its deterministic local splice RNG.
INTERMEDIATE_JPEG_QUALITY_RANGE = (90, 94)


def _sample_patch_in_region(region, patch_w, patch_h, rng):
    """Choose a patch fully contained in a content-bearing region."""
    left, top, right, bottom = region
    if right - left < patch_w or bottom - top < patch_h:
        raise ValueError(f"Informative region {region} is smaller than splice patch")
    return rng.randint(left, right - patch_w), rng.randint(top, bottom - patch_h)


def _irregular_mask(size, rng):
    """Return a seeded, softly feathered organic mask within ``size``.

    The mask is deliberately generated per splice rather than being a fixed
    shape.  Its boundary remains well inside the sampled content rectangle,
    so only an irregular local area is replaced; the rectangle is only a
    sampling window, not the visible operation footprint.
    """
    width, height = size
    center_x = width * rng.uniform(0.42, 0.58)
    center_y = height * rng.uniform(0.42, 0.58)
    radius_x = width * rng.uniform(0.33, 0.44)
    radius_y = height * rng.uniform(0.33, 0.44)
    points = []
    point_count = rng.randint(9, 15)
    for index in range(point_count):
        angle = 2.0 * np.pi * index / point_count
        # Independent radial variation avoids a repeated ellipse or polygon.
        radial = rng.uniform(0.76, 1.05)
        points.append((
            center_x + np.cos(angle) * radius_x * radial,
            center_y + np.sin(angle) * radius_y * radial,
        ))

    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).polygon(points, fill=255)
    feather_radius = max(1.25, min(width, height) * rng.uniform(0.04, 0.07))
    return mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))


def _seeded_splice_mask(size, mask_seed, opacity):
    """Recreate the exact local alpha mask from persisted audit metadata."""
    mask_rng = random.Random(mask_seed)
    mask = _irregular_mask(size, mask_rng)
    return mask.point(lambda value: round(value * opacity))


def reconstruct_splice_mask(image_size, splice_metadata):
    """Return a full-image ground-truth alpha mask from splice metadata.

    This is an audit/localization helper only.  It is not used by feature
    extraction or classification.
    """
    parameters = splice_metadata["splice_mask_parameters"]
    patch_w, patch_h = parameters["patch_size"]
    dx, dy = parameters["destination"]
    local_mask = _seeded_splice_mask(
        (patch_w, patch_h), splice_metadata["splice_mask_seed"], parameters["opacity"]
    )
    full_mask = Image.new("L", image_size, 0)
    full_mask.paste(local_mask, (dx, dy))
    return full_mask


def validate_splice_metadata(original, spliced, metadata, allowed_region):
    """Validate reconstructability and placement before shared augmentation."""
    if original.size != spliced.size or original.mode != spliced.mode:
        raise RuntimeError("Splice changed image dimensions or mode")
    full_mask = reconstruct_splice_mask(original.size, metadata)
    mask_array = np.asarray(full_mask)
    if not mask_array.any():
        raise RuntimeError("Splice mask is empty")
    left, top, right, bottom = allowed_region
    mask_box = full_mask.getbbox()
    if (mask_box is None or mask_box[0] < left or mask_box[1] < top
            or mask_box[2] > right or mask_box[3] > bottom):
        raise RuntimeError("Splice mask is outside its allowed destination region")
    if hashlib.sha256(full_mask.tobytes()).hexdigest() != metadata["splice_mask_sha256"]:
        raise RuntimeError("Splice mask metadata does not reconstruct the generated mask")
    changed = np.any(np.asarray(original) != np.asarray(spliced), axis=2)
    if not np.any(changed & (mask_array > 0)):
        raise RuntimeError("Splice did not change pixels inside its expected mask")
    return full_mask


def _match_local_appearance(patch, destination, rng):
    """Apply a restrained channel-wise tone adjustment toward destination.

    This is intentionally partial: it reduces an implausible exposure jump
    without erasing source texture, colour, or the compositing residual that
    makes the manipulation forensically meaningful.
    """
    source = np.asarray(patch).astype(np.float32)
    target = np.asarray(destination).astype(np.float32)
    source_mean = source.mean(axis=(0, 1), keepdims=True)
    target_mean = target.mean(axis=(0, 1), keepdims=True)
    source_std = source.std(axis=(0, 1), keepdims=True)
    target_std = target.std(axis=(0, 1), keepdims=True)
    gain = np.clip(target_std / np.maximum(source_std, 1.0), 0.88, 1.12)
    matched = (source - source_mean) * gain + target_mean
    strength = rng.uniform(0.28, 0.48)
    adjusted = source * (1.0 - strength) + matched * strength
    return Image.fromarray(np.clip(adjusted, 0, 255).astype(np.uint8))


def _intermediate_jpeg_roundtrip(patch, quality):
    """Apply a mild, in-memory JPEG history to a copied source patch only.

    This models a localized copy-move compression-history mismatch.  The
    result remains confined by the existing irregular alpha mask; no JPEG
    block is pasted as a rectangle and no temporary file is created.
    """
    if not 1 <= quality <= 95:
        raise ValueError("intermediate_jpeg_quality must be between 1 and 95")
    encoded = io.BytesIO()
    patch.convert("RGB").save(encoded, format="JPEG", quality=quality)
    encoded.seek(0)
    with Image.open(encoded) as decoded:
        return decoded.convert("RGB").copy()


def _composite_copy_move(img, patch, destination, destination_patch, rng,
                         intermediate_jpeg_quality, mask_seed):
    """Insert a copied patch with one subtle, general compositing process.

    The small resample round-trip reflects ordinary raster insertion rather
    than a family- or region-specific synthetic artifact.  A mask feather
    proportional to patch size blends only the insertion transition; it is
    not a separately drawn border or noise ring.
    """
    resampling = getattr(Image, "Resampling", Image)
    patch_w, patch_h = patch.size
    scale = rng.uniform(0.985, 1.015)
    intermediate_size = (
        max(1, round(patch_w * scale)),
        max(1, round(patch_h * scale)),
    )
    composited_patch = _intermediate_jpeg_roundtrip(patch, intermediate_jpeg_quality)
    composited_patch = composited_patch.resize(intermediate_size, resampling.BICUBIC)
    composited_patch = composited_patch.resize((patch_w, patch_h), resampling.BICUBIC)
    composited_patch = _match_local_appearance(composited_patch, destination_patch, rng)

    # The alpha footprint is irregular and feathered.  No border, halo, or
    # separately generated edge artifact is drawn around the insertion.
    opacity = rng.uniform(0.58, 0.78)
    mask = _seeded_splice_mask((patch_w, patch_h), mask_seed, opacity)
    result = img.copy()
    result.paste(composited_patch, destination, mask)
    return result, mask, opacity


def copy_move_patch(img: Image.Image, informative_regions, rng=random, patch_frac=0.12,
                    excluded_signatures=None, max_attempts=100,
                    intermediate_jpeg_quality=None):
    """Copy content into one informative region through soft compositing.

    Both source and destination are fully inside configured content regions.
    Each copied patch receives a mild intermediate JPEG round trip before the
    existing resample, tone-match, and irregular feathered compositing steps.
    The JPEG quality is either an explicit experimental value or a
    deterministic local draw from ``INTERMEDIATE_JPEG_QUALITY_RANGE``.
    Candidates that produce no pixel change or duplicate a sibling splice are
    rejected.
    """
    img = img.copy()
    w, h = img.size
    excluded_signatures = excluded_signatures or set()
    # Use a local stream for the expanded mask/tone sampling.  The shared
    # pipeline stream is advanced below by its former splice sequence.
    splice_rng = random.Random()
    splice_rng.setstate(rng.getstate())
    if intermediate_jpeg_quality is None:
        low, high = INTERMEDIATE_JPEG_QUALITY_RANGE
        intermediate_jpeg_quality = splice_rng.randint(low, high)

    for _ in range(max_attempts):
        # Vary both axes so each sampling window and its resulting irregular
        # footprint differs across seeded sibling splice variants.
        pw = max(8, round(w * patch_frac * splice_rng.uniform(0.50, 0.80)))
        ph = max(8, round(h * patch_frac * splice_rng.uniform(0.50, 0.80)))
        # Sampling both locations from one content region makes the local
        # replacement compatible with the surrounding document grammar while
        # preserving a general copy-move operation across layouts.
        region = splice_rng.choice(informative_regions)
        sx, sy = _sample_patch_in_region(region, pw, ph, splice_rng)
        dx, dy = _sample_patch_in_region(region, pw, ph, splice_rng)
        signature = (sx, sy, dx, dy, pw, ph)
        if signature in excluded_signatures or (sx, sy) == (dx, dy):
            continue
        patch = img.crop((sx, sy, sx + pw, sy + ph))
        destination = img.crop((dx, dy, dx + pw, dy + ph))
        if patch.tobytes() == destination.tobytes():
            continue
        mask_seed = splice_rng.getrandbits(64)
        result, local_mask, opacity = _composite_copy_move(
            img, patch, (dx, dy), destination, splice_rng, intermediate_jpeg_quality, mask_seed
        )
        if result.crop((dx, dy, dx + pw, dy + ph)).tobytes() == destination.tobytes():
            continue
        metadata = {
            "splice_region": list(region),
            "splice_mask_seed": mask_seed,
            "splice_mask_parameters": {
                "source": [sx, sy],
                "destination": [dx, dy],
                "patch_size": [pw, ph],
                "opacity": opacity,
                "intermediate_jpeg_quality": intermediate_jpeg_quality,
            },
            "splice_mask_sha256": hashlib.sha256(
                reconstruct_splice_mask(img.size, {
                    "splice_mask_seed": mask_seed,
                    "splice_mask_parameters": {
                        "destination": [dx, dy], "patch_size": [pw, ph], "opacity": opacity,
                    },
                }).tobytes()
            ).hexdigest(),
        }
        validate_splice_metadata(img, result, metadata, region)
        return result, signature, metadata

    raise RuntimeError("Unable to generate a distinct informative copy-move splice")


def realistic_noise(img: Image.Image, rng=random) -> Image.Image:
    """Apply mild blur/jpeg-recompression/scale jitter so genuine and
    forged images share the same nuisance variation (avoids the model
    learning to detect 'this image was edited in PIL' instead of the
    actual forgery)."""
    img = img.copy()

    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.5)))

    if rng.random() < 0.6:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(80, 95))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    if rng.random() < 0.3:
        scale = rng.uniform(0.95, 1.0)
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale))).resize((w, h))

    if rng.random() < 0.3:
        arr = np.array(img).astype(np.int16)
        # NumPy's module-global RNG is not seeded by build_dataset.  Derive a
        # local generator from the pipeline RNG so a fixed build seed fully
        # reproduces this augmentation too.
        noise_rng = np.random.default_rng(rng.getrandbits(64))
        noise = noise_rng.normal(0, rng.uniform(2, 8), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img
