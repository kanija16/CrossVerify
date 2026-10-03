"""
data_interface.py
-----------------
M3's read-only data access layer for M1's dataset and split CSVs.

Responsibilities:
1. Load train/val/test CSVs exactly as produced by M1 (never modifying them).
2. Resolve each row's label_path against DATASET_ROOT.
3. Load the corresponding label JSON and extract the three M3-owned blocks:
     - ocr_extracted_fields
     - qr_decoded_fields
     - consistency_vector
4. Convert final_label to binary target (0 = genuine, 1 = forged).
5. Retain record_id, document_family, template_variant, and tamper_type
   strictly as metadata (for identity grouping and post-hoc attack analysis).
6. Guarantee that no leaky feature fields are injected into feature vectors.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import pandas as pd

from . import config


@dataclass
class M3Record:
    """
    In-memory representation of a single document sample for M3.
    """
    id: str
    record_id: str
    split: str                               # "train" | "val" | "test"
    document_family: Optional[str]           # "family_a" | "family_b"
    template_variant: Optional[str]          # "v1" | "v2"
    tamper_type: Optional[str]               # evaluation slice only — never a feature
    image_path: Path                         # bookkeeping / disk locator
    label_path: Path                         # bookkeeping / disk locator
    target: int                              # 0 = genuine, 1 = forged
    consistency_vector: Dict[str, Any]       # primary feature input for Part 1
    ocr_extracted_fields: Dict[str, Any]     # preserved for later parts
    qr_decoded_fields: Dict[str, Any]        # preserved for later parts

    @property
    def family(self) -> Optional[str]:
        """Convenience alias for document_family."""
        return self.document_family


def resolve_label_path(raw_path: str, root: Optional[Path] = None) -> Path:
    """
    Resolve a label_path string into an absolute Path on disk.
    Handles relative paths (e.g. 'labels/family_a_...json') and absolute paths.
    """
    p = Path(raw_path)
    if p.is_absolute():
        return p
    dataset_root = root if root is not None else config.DATASET_ROOT
    return dataset_root / raw_path


def resolve_image_path(raw_path: str, root: Optional[Path] = None) -> Path:
    """
    Resolve an image_path string into an absolute Path on disk.
    """
    p = Path(raw_path)
    if p.is_absolute():
        return p
    dataset_root = root if root is not None else config.DATASET_ROOT
    return dataset_root / raw_path


def binary_target(final_label: Any) -> int:
    """
    Map final_label value to binary integer:
    0 = genuine, 1 = forged.
    """
    try:
        return config.FINAL_LABEL_MAP[final_label]
    except KeyError as e:
        raise ValueError(
            f"Unrecognized final_label value: {final_label!r}. "
            f"Allowed values are in config.FINAL_LABEL_MAP. "
            f"Do not modify M1's CSV to work around this."
        ) from e


def load_label_json(path: Path) -> Dict[str, Any]:
    """
    Load a label JSON file from disk.
    """
    if not path.exists():
        raise FileNotFoundError(f"Label JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_split_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load one of M1's split CSV files into a DataFrame.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_split(
    csv_path: Path,
    split_name: str,
    dataset_root: Optional[Path] = None
) -> List[M3Record]:
    """
    Load an M1 split CSV and join each row with its corresponding label JSON.
    Optimized to load 7,700+ rows efficiently without pandas iterrows overhead.
    """
    df = load_split_csv(csv_path)

    # Validate essential columns
    required_cols = {
        config.COL_RECORD_ID,
        config.COL_LABEL_PATH,
        config.COL_FINAL_LABEL
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing expected columns: {missing}. "
            f"Actual columns found: {list(df.columns)}."
        )

    records: List[M3Record] = []
    # Using to_dict('records') for high-throughput row iteration
    rows = df.to_dict(orient="records")

    for row in rows:
        label_p = resolve_label_path(str(row[config.COL_LABEL_PATH]), root=dataset_root)
        raw_json = load_label_json(label_p)

        # Validate that M3-owned keys exist in the JSON
        for required_key in (config.KEY_OCR_FIELDS, config.KEY_QR_FIELDS, config.KEY_CONSISTENCY):
            if required_key not in raw_json:
                raise ValueError(
                    f"Label JSON {label_p} is missing required key '{required_key}'."
                )

        image_p = (
            resolve_image_path(str(row[config.COL_IMAGE_PATH]), root=dataset_root)
            if config.COL_IMAGE_PATH in row
            else Path("")
        )

        doc_fam = row.get(config.COL_DOCUMENT_FAMILY) or row.get("family")
        tmpl_var = row.get(config.COL_TEMPLATE_VARIANT)
        tamper_t = row.get(config.COL_TAMPER_TYPE)
        sample_id = str(row.get(config.COL_ID, ""))

        record = M3Record(
            id=sample_id,
            record_id=str(row[config.COL_RECORD_ID]),
            split=split_name,
            document_family=str(doc_fam) if doc_fam is not None else None,
            template_variant=str(tmpl_var) if tmpl_var is not None else None,
            tamper_type=str(tamper_t) if tamper_t is not None else None,
            image_path=image_p,
            label_path=label_p,
            target=binary_target(row[config.COL_FINAL_LABEL]),
            consistency_vector=raw_json[config.KEY_CONSISTENCY],
            ocr_extracted_fields=raw_json[config.KEY_OCR_FIELDS],
            qr_decoded_fields=raw_json[config.KEY_QR_FIELDS],
        )
        records.append(record)

    return records


def load_all_splits(
    train_csv: Optional[Path] = None,
    val_csv: Optional[Path] = None,
    test_csv: Optional[Path] = None,
    dataset_root: Optional[Path] = None
) -> Dict[str, List[M3Record]]:
    """
    Load all three standard splits: train, val, test.
    """
    tr_p = train_csv if train_csv is not None else config.TRAIN_CSV
    va_p = val_csv if val_csv is not None else config.VAL_CSV
    te_p = test_csv if test_csv is not None else config.TEST_CSV

    return {
        "train": load_split(tr_p, "train", dataset_root=dataset_root),
        "val": load_split(va_p, "val", dataset_root=dataset_root),
        "test": load_split(te_p, "test", dataset_root=dataset_root),
    }


def records_to_dataframe(records: List[M3Record]) -> pd.DataFrame:
    """
    Convert a list of M3Record instances into a diagnostics DataFrame.
    """
    return pd.DataFrame([{
        "id": r.id,
        "record_id": r.record_id,
        "split": r.split,
        "document_family": r.document_family,
        "template_variant": r.template_variant,
        "tamper_type": r.tamper_type,
        "target": r.target,
        "image_path": str(r.image_path),
        "label_path": str(r.label_path),
        "consistency_vector": r.consistency_vector,
        "ocr_extracted_fields": r.ocr_extracted_fields,
        "qr_decoded_fields": r.qr_decoded_fields,
    } for r in records])
