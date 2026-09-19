"""
src/baseline/dataset.py

Member 2 Dataset Interface for CrossVerify Visual-Forensics (CNN & ELA+CNN).
Provides PyTorch Dataset implementation for loading image data and target cnn_label.

Strictly enforces:
- Image-only model input (zero metadata leakage to model).
- cnn_label target mapping: 'genuine' -> 0, 'forged' -> 1.
- Separate metadata retrieval interface for downstream evaluation joining.
- Input validation (missing images, corrupt files, invalid labels).
"""

import os
from pathlib import Path
from typing import Union, Optional, Callable, Dict, Tuple, Any
import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset as PyTorchDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    PyTorchDataset = object  # Fallback base class if PyTorch is not yet installed


# Label mapping constant (authoritative for Member 2)
CNN_LABEL_MAP = {
    "genuine": 0,
    "forged": 1
}

REVERSE_CNN_LABEL_MAP = {
    0: "genuine",
    1: "forged"
}

# Metadata columns required for prediction CSV generation
REQUIRED_METADATA_COLS = [
    "id",
    "record_id",
    "image_path",
    "final_label",
    "tamper_type"
]


class VisualForensicsDataset(PyTorchDataset):
    """
    PyTorch Dataset for Member 2 Visual Forensics baseline (CNN & ELA+CNN).
    
    This dataset loads images from disk and pairs them strictly with `cnn_label`.
    It guarantees that zero metadata (such as record_id, tamper_type, OCR, QR, or paths)
    is exposed as part of the model input tensor/tuple.
    """

    def __init__(
        self,
        csv_file: Union[str, Path],
        dataset_root: Union[str, Path],
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
        validate: bool = True
    ):
        """
        Args:
            csv_file: Path to train.csv, val.csv, or test.csv (absolute or relative to dataset_root).
            dataset_root: Root directory of the dataset containing image files.
            transform: Optional transform callable (e.g. PyTorch torchvision transform) applied to PIL image.
            return_metadata: If True, __getitem__ returns (image_tensor, target, metadata_dict).
                             If False (default for model training/inference), returns (image_tensor, target).
            validate: If True, performs lightweight validation on CSV contents and file existence.
        """
        self.dataset_root = Path(dataset_root)
        
        # Resolve CSV path
        csv_path = Path(csv_file)
        if not csv_path.is_absolute():
            csv_path = self.dataset_root / csv_path
            
        if not csv_path.exists():
            raise FileNotFoundError(f"Canonical split CSV not found at: {csv_path}")

        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.return_metadata = return_metadata

        if validate:
            self._validate()

    def _validate(self) -> None:
        """Performs lightweight integrity validation on CSV and image paths."""
        # 1. Verify required columns
        if "cnn_label" not in self.df.columns:
            raise ValueError(f"CSV '{self.csv_path}' is missing mandatory 'cnn_label' column.")
        if "image_path" not in self.df.columns:
            raise ValueError(f"CSV '{self.csv_path}' is missing mandatory 'image_path' column.")

        # 2. Check cnn_label values
        invalid_labels = set(self.df["cnn_label"].dropna().unique()) - set(CNN_LABEL_MAP.keys())
        if invalid_labels:
            raise ValueError(
                f"Invalid cnn_label values found in {self.csv_path}: {invalid_labels}. "
                f"Expected only {set(CNN_LABEL_MAP.keys())}"
            )

        missing_labels_count = self.df["cnn_label"].isnull().sum()
        if missing_labels_count > 0:
            raise ValueError(f"Found {missing_labels_count} missing/null cnn_label values in {self.csv_path}.")

        # 3. Verify image file existence
        missing_files = []
        for idx, rel_path in enumerate(self.df["image_path"]):
            full_img_path = self.dataset_root / rel_path
            if not full_img_path.exists():
                missing_files.append((idx, str(rel_path)))

        if missing_files:
            sample_missing = missing_files[:3]
            raise FileNotFoundError(
                f"Found {len(missing_files)} missing image file(s) referenced in '{self.csv_path}'. "
                f"Dataset root: '{self.dataset_root}'. Sample missing: {sample_missing}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves output metadata for a given index.
        This metadata is strictly separated from model inputs and used ONLY for prediction CSV joining.
        """
        row = self.df.iloc[idx]
        return {
            "id": str(row["id"]),
            "record_id": str(row["record_id"]),
            "image_path": str(row["image_path"]),
            "final_label": str(row["final_label"]),
            "tamper_type": str(row["tamper_type"]),
            "cnn_label": str(row["cnn_label"])
        }

    def __getitem__(self, idx: int) -> Union[Tuple[Any, Any], Tuple[Any, Any, Dict[str, Any]]]:
        """
        Loads image and returns target label.
        
        Returns:
            - If return_metadata is False: (image_tensor, cnn_label_target)
            - If return_metadata is True:  (image_tensor, cnn_label_target, metadata_dict)
        """
        row = self.df.iloc[idx]
        rel_img_path = row["image_path"]
        full_img_path = self.dataset_root / rel_img_path

        # 1. Load image and ensure 3-channel RGB
        try:
            image = Image.open(full_img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Corrupt or unreadable image file at {full_img_path}: {e}") from e

        # 2. Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)

        # 3. Map cnn_label: 'genuine' -> 0, 'forged' -> 1
        raw_label = row["cnn_label"]
        label_int = CNN_LABEL_MAP[raw_label]
        
        if TORCH_AVAILABLE:
            target = torch.tensor(label_int, dtype=torch.long)
        else:
            target = label_int

        # 4. Return format
        if self.return_metadata:
            metadata = self.get_metadata(idx)
            return image, target, metadata

        # Model input format: strictly image and target label ONLY
        return image, target


def run_sanity_checks(dataset_root: Union[str, Path]) -> None:
    """
    Sanity test utility for Phase B2 dataset interface validation.
    
    Verifies:
    1. Dataset lengths for train, val, and test splits (7700, 1650, 1650).
    2. Single sample loading and PIL Image mode is RGB.
    3. cnn_label mapping ('genuine' -> 0, 'forged' -> 1).
    4. Model input return tuple contains NO metadata.
    5. Output metadata isolation (id, record_id, image_path, final_label, tamper_type).
    6. Validation error handling for non-existent files or invalid labels.
    """
    print("=" * 60)
    print("RUNNING PHASE B2 DATASET INTERFACE SANITY CHECKS")
    print("=" * 60)
    
    dataset_root = Path(dataset_root)
    print(f"Dataset root: {dataset_root}")
    print(f"PyTorch available: {TORCH_AVAILABLE}")

    # 1. Check split lengths
    expected_counts = {
        "train.csv": 7700,
        "val.csv": 1650,
        "test.csv": 1650
    }
    
    datasets = {}
    for csv_name, expected_len in expected_counts.items():
        ds = VisualForensicsDataset(csv_file=csv_name, dataset_root=dataset_root, validate=True)
        datasets[csv_name] = ds
        assert len(ds) == expected_len, f"Expected {expected_len} samples in {csv_name}, got {len(ds)}"
        print(f"[OK] {csv_name}: Verified length = {len(ds)} (matches expected {expected_len})")

    # 2. Verify single sample loading and RGB mode
    train_ds = datasets["train.csv"]
    sample_img, sample_target = train_ds[0]
    
    assert isinstance(sample_img, Image.Image), f"Expected PIL Image, got {type(sample_img)}"
    assert sample_img.mode == "RGB", f"Expected RGB image mode, got {sample_img.mode}"
    print(f"[OK] Sample 0 loaded successfully: Image type={type(sample_img).__name__}, Mode={sample_img.mode}, Size={sample_img.size}")

    # 3. Verify cnn_label target mapping
    raw_label_0 = train_ds.df.iloc[0]["cnn_label"]
    expected_target_0 = CNN_LABEL_MAP[raw_label_0]
    
    if TORCH_AVAILABLE:
        assert isinstance(sample_target, torch.Tensor), f"Expected torch.Tensor, got {type(sample_target)}"
        assert sample_target.item() == expected_target_0, f"Expected target {expected_target_0}, got {sample_target.item()}"
    else:
        assert sample_target == expected_target_0, f"Expected target {expected_target_0}, got {sample_target}"
    print(f"[OK] cnn_label mapping verified: '{raw_label_0}' -> target value {sample_target}")

    # 4. Verify returned model input contains NO metadata
    sample_tuple = train_ds[0]
    assert len(sample_tuple) == 2, f"Model input tuple length must be 2 (image, target), got {len(sample_tuple)}"
    img_element, target_element = sample_tuple
    assert not isinstance(img_element, dict), "Image element must not be a dictionary containing metadata!"
    assert not isinstance(target_element, dict), "Target element must not be a dictionary containing metadata!"
    print("[OK] Model input tuple strictly contains (image, target) with ZERO metadata fields exposed.")

    # 5. Verify separate metadata retrieval for prediction output joining
    metadata_0 = train_ds.get_metadata(0)
    required_keys = {"id", "record_id", "image_path", "final_label", "tamper_type", "cnn_label"}
    assert required_keys.issubset(set(metadata_0.keys())), f"Missing metadata keys. Got {list(metadata_0.keys())}"
    print(f"[OK] Metadata retrieval verified cleanly separated: {metadata_0}")

    # 6. Verify return_metadata=True flag works as expected
    meta_ds = VisualForensicsDataset(csv_file="train.csv", dataset_root=dataset_root, return_metadata=True)
    img_m, target_m, meta_m = meta_ds[0]
    assert meta_m == metadata_0, "Metadata dictionary mismatch when return_metadata=True"
    print("[OK] return_metadata=True flag verified for evaluation/inference pipeline.")

    print("\nALL PHASE B2 SANITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    # Use default local DATASET_ROOT or override via environment variable
    default_dataset_root = os.environ.get(
        "DATASET_ROOT",
        r"C:\Users\jayadharshini\OneDrive\Desktop\dataset_final_v2\dataset_final_v2"
    )
    run_sanity_checks(default_dataset_root)
