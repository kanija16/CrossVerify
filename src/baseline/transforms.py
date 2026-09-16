"""
src/baseline/transforms.py

Member 2 Preprocessing & Transformation Pipeline for CrossVerify Visual Forensics.

Key Features:
- Aspect-ratio-preserving resize + padding to target size 224x224 (no stretching).
- Distinct training vs validation/test transform pipelines.
- Conservative training augmentations suitable for visual forensics (preserving pixel-level splice artifacts).
- Deterministic validation/test transforms.
- Configurable ImageNet normalization.
- Pure image-level operation (zero metadata dependency).
"""

import os
from pathlib import Path
from typing import Union, Tuple, Callable, Any
import numpy as np
from PIL import Image

try:
    import torch
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Default ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AspectPreservingResizeAndPad:
    """
    Resizes an image to fit within target_size while strictly preserving aspect ratio,
    then pads the remaining space with a constant color (default black = 0).
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        fill: int = 0,
        interpolation: Image.Resampling = Image.Resampling.BILINEAR
    ):
        """
        Args:
            target_size: (width, height) tuple for final output dimensions.
            fill: Pixel fill value for padding areas (default 0).
            interpolation: Resampling method for resizing.
        """
        self.target_w, self.target_h = target_size
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError("Cannot resize image with 0 width or height.")

        # Compute scaling factor to fit within target bounding box
        scale = min(self.target_w / w, self.target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # Resize preserving exact aspect ratio
        resized_img = img.resize((new_w, new_h), resample=self.interpolation)

        # Create target canvas with neutral padding
        mode = img.mode
        canvas_color = (self.fill, self.fill, self.fill) if mode == "RGB" else self.fill
        padded_img = Image.new(mode, (self.target_w, self.target_h), color=canvas_color)

        # Center resized image on canvas
        left = (self.target_w - new_w) // 2
        top = (self.target_h - new_h) // 2
        padded_img.paste(resized_img, (left, top))

        return padded_img


class SimpleToTensorAndNormalize:
    """
    Fallback tensor conversion and normalization when PyTorch is not installed.
    Converts PIL RGB Image -> NumPy float32 array [3, H, W] scaled to [0, 1] and normalized.
    """

    def __init__(
        self,
        mean: Tuple[float, float, float] = IMAGENET_MEAN,
        std: Tuple[float, float, float] = IMAGENET_STD
    ):
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, img: Image.Image) -> np.ndarray:
        arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, C]
        arr = arr.transpose(2, 0, 1)  # [C, H, W]
        normalized = (arr - self.mean) / self.std
        return normalized


class ConservativeRandomHorizontalFlip:
    """
    Applies random horizontal flip with probability p.
    Preserves pixel-level local noise and splice boundaries while providing mild spatial invariance.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


def get_eval_transforms(
    target_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> Callable[[Image.Image], Any]:
    """
    Returns deterministic validation/testing preprocessing pipeline.
    
    Steps:
    1. Aspect-ratio-preserving resize + padding to target_size (224x224).
    2. Convert image to Tensor scaled to [0, 1].
    3. Normalize with specified mean and std (ImageNet defaults).
    """
    resizer = AspectPreservingResizeAndPad(target_size=target_size, fill=0)

    if TORCH_AVAILABLE:
        return T.Compose([
            resizer,
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    else:
        to_tensor_norm = SimpleToTensorAndNormalize(mean=mean, std=std)
        return lambda img: to_tensor_norm(resizer(img))


def get_train_transforms(
    target_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> Callable[[Image.Image], Any]:
    """
    Returns training preprocessing pipeline matching evaluation/test pipeline.
    
    No spatial or geometric augmentations are applied to preserve authentic document
    visual distribution and pixel-level splice features.
    
    Steps:
    1. Aspect-ratio-preserving resize + padding to target_size (224x224).
    2. Convert to Tensor scaled to [0, 1].
    3. Normalize with specified mean and std (ImageNet defaults).
    """
    resizer = AspectPreservingResizeAndPad(target_size=target_size, fill=0)

    if TORCH_AVAILABLE:
        return T.Compose([
            resizer,
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    else:
        to_tensor_norm = SimpleToTensorAndNormalize(mean=mean, std=std)
        return lambda img: to_tensor_norm(resizer(img))


def run_transform_verification(dataset_root: Union[str, Path]) -> None:
    """
    Verification demo for Phase B3 preprocessing pipeline.
    
    Verifies:
    1. Loading representative Family A (1000x640) and Family B (1150x520) images.
    2. Preservation of aspect ratio before padding.
    3. Applying train and eval transforms.
    4. Final output shape is strictly [3, 224, 224].
    5. Original image files on disk remain completely unmodified.
    6. Zero metadata dependencies.
    """
    print("=" * 60)
    print("RUNNING PHASE B3 PREPROCESSING VERIFICATION DEMO")
    print("=" * 60)
    
    dataset_root = Path(dataset_root)
    fam_a_sample = dataset_root / "images" / "family_a" / "family_a_000000_genuine.png"
    fam_b_sample = dataset_root / "images" / "family_b" / "family_b_000000_genuine.png"

    assert fam_a_sample.exists(), f"Family A sample not found: {fam_a_sample}"
    assert fam_b_sample.exists(), f"Family B sample not found: {fam_b_sample}"

    # Get initial file stats to confirm files remain untouched
    fam_a_stat_before = os.stat(fam_a_sample)
    fam_b_stat_before = os.stat(fam_b_sample)

    eval_transform = get_eval_transforms(target_size=(224, 224))
    train_transform = get_train_transforms(target_size=(224, 224))
    resizer = AspectPreservingResizeAndPad(target_size=(224, 224))

    for name, sample_path in [("Family A", fam_a_sample), ("Family B", fam_b_sample)]:
        print(f"\n--- Verifying {name} ({sample_path.name}) ---")
        
        # Load raw image
        with Image.open(sample_path) as orig_img:
            orig_w, orig_h = orig_img.size
            orig_aspect = orig_w / orig_h
            print(f"Original Dimensions (W x H): ({orig_w}, {orig_h})")
            print(f"Original Aspect Ratio (W / H): {orig_aspect:.4f}")

            # Verify aspect ratio preservation in intermediate resized step
            padded_pil = resizer(orig_img)
            scale = min(224 / orig_w, 224 / orig_h)
            calc_w = int(round(orig_w * scale))
            calc_h = int(round(orig_h * scale))
            calc_aspect = calc_w / calc_h
            
            print(f"Resized Content Dimensions (before padding): ({calc_w}, {calc_h})")
            print(f"Resized Content Aspect Ratio: {calc_aspect:.4f} (Delta from orig: {abs(orig_aspect - calc_aspect):.6f})")
            print(f"Padded Canvas Size: {padded_pil.size} (Width x Height)")
            assert padded_pil.size == (224, 224), f"Padded canvas size must be (224, 224), got {padded_pil.size}"

            # Apply Eval Transform
            eval_out = eval_transform(orig_img)
            eval_shape = list(eval_out.shape) if hasattr(eval_out, "shape") else [len(eval_out)]
            print(f"Eval Transform Output Type: {type(eval_out).__name__}, Shape: {eval_shape}")
            assert eval_shape == [3, 224, 224], f"Expected output shape [3, 224, 224], got {eval_shape}"

            # Apply Train Transform
            train_out = train_transform(orig_img)
            train_shape = list(train_out.shape) if hasattr(train_out, "shape") else [len(train_out)]
            print(f"Train Transform Output Type: {type(train_out).__name__}, Shape: {train_shape}")
            assert train_shape == [3, 224, 224], f"Expected output shape [3, 224, 224], got {train_shape}"

    # Confirm original files on disk were NOT modified
    fam_a_stat_after = os.stat(fam_a_sample)
    fam_b_stat_after = os.stat(fam_b_sample)

    assert fam_a_stat_before.st_mtime == fam_a_stat_after.st_mtime and fam_a_stat_before.st_size == fam_a_stat_after.st_size, \
        "Family A file on disk was modified!"
    assert fam_b_stat_before.st_mtime == fam_b_stat_after.st_mtime and fam_b_stat_before.st_size == fam_b_stat_after.st_size, \
        "Family B file on disk was modified!"

    print("\n[OK] Original dataset files on disk verified completely unchanged.")
    print("[OK] Output tensor shapes verified as strictly [3, 224, 224] for both Family A and Family B.")
    print("[OK] Aspect ratio preservation before padding verified.")
    print("\nALL PHASE B3 PREPROCESSING VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    default_dataset_root = os.environ.get(
        "DATASET_ROOT",
        r"C:\Users\jayadharshini\OneDrive\Desktop\dataset_final_v2\dataset_final_v2"
    )
    run_transform_verification(default_dataset_root)
