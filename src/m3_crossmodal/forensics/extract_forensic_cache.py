"""
extract_forensic_cache.py
-------------------------
Batch extraction and caching of Document-Region Anomaly Forensic Features
for CrossVerify M3 on TRAIN and VALIDATION splits ONLY.

Guarantees:
- Zero access to test split (test.csv is strictly rejected).
- Zero forbidden metadata/labels in feature columns.
- Multiprocessing with incremental batch writing and deterministic sorting on unique sample 'id'.
- Produces:
    outputs/forensics/train_forensic_features.csv (7,700 rows)
    outputs/forensics/val_forensic_features.csv (1,650 rows)
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from PIL import Image

from m3_crossmodal import config
from m3_crossmodal.data_interface import load_split_csv, resolve_image_path
from m3_crossmodal.forensics.region_features import (
    FORENSIC_FEATURE_NAMES,
    assert_clean_forensic_features,
    extract_region_anomaly_features,
)

FORENSIC_OUTPUT_DIR = Path("outputs/forensics")


def _worker_extract_forensic(row_tuple: Tuple[str, str, str, str, str, str, int]) -> Dict[str, Any]:
    """
    Worker function for multiprocessing.
    row_tuple: (sample_id, record_id, split_name, image_path_str, document_family, tamper_type, final_label)
    """
    sample_id, record_id, split_name, image_path_str, document_family, tamper_type, final_label = row_tuple

    # 1. Load image pixels
    with Image.open(image_path_str) as im:
        img_arr = np.array(im.convert("RGB"))

    # 2. Extract 16 regional anomaly features
    feats = extract_region_anomaly_features(img_arr, document_family)

    # 3. Assemble row record (metadata strictly separated from feature extraction)
    record: Dict[str, Any] = {
        "id": sample_id,
        "record_id": record_id,
        "document_family": document_family,
        "split": split_name,
        "tamper_type": tamper_type,
        "final_label": final_label,
    }
    record.update(feats)
    return record


def build_forensic_feature_cache(
    split_name: str,
    max_workers: int = 8,
    batch_write_size: int = 250,
) -> Path:
    """
    Extracts forensic features for train or val split and saves CSV cache.
    Raises ValueError if split_name == 'test'.
    """
    if split_name == "test":
        raise ValueError("PROTOCOL ERROR: Test split extraction is strictly forbidden during this phase.")

    if split_name not in ["train", "val"]:
        raise ValueError(f"Unknown or unauthorized split: {split_name!r}")

    FORENSIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = FORENSIC_OUTPUT_DIR / f"{split_name}_forensic_features.csv"

    csv_path = config.TRAIN_CSV if split_name == "train" else config.VAL_CSV
    df_meta = load_split_csv(csv_path)
    total_target = len(df_meta)

    # Check for existing completed cache
    if out_csv.exists():
        df_existing = pd.read_csv(out_csv)
        if len(df_existing) == total_target:
            print(f"[{split_name.upper()}] Cache already complete with {len(df_existing)} records at {out_csv}")
            return out_csv

    tasks = []
    for _, row in df_meta.iterrows():
        sample_id = str(row[config.COL_ID])
        rec_id = str(row[config.COL_RECORD_ID])
        img_p = str(resolve_image_path(row[config.COL_IMAGE_PATH]))
        fam = str(row[config.COL_DOCUMENT_FAMILY])
        tt = str(row[config.COL_TAMPER_TYPE])
        lbl = int(config.FINAL_LABEL_MAP[row[config.COL_FINAL_LABEL]])
        tasks.append((sample_id, rec_id, split_name, img_p, fam, tt, lbl))

    print(f"[{split_name.upper()}] Extracting forensic features for {len(tasks)} images (workers={max_workers})...")
    t0 = time.time()
    batch_buffer: List[Dict[str, Any]] = []
    done_count = 0

    if out_csv.exists():
        out_csv.unlink()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker_extract_forensic, t) for t in tasks]
        for f in as_completed(futures):
            res = f.result()
            batch_buffer.append(res)
            done_count += 1

            if len(batch_buffer) >= batch_write_size:
                df_batch = pd.DataFrame(batch_buffer)
                write_header = not out_csv.exists()
                df_batch.to_csv(out_csv, mode="a", index=False, header=write_header)
                batch_buffer.clear()
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - done_count) / rate if rate > 0 else 0
                print(f"[{split_name.upper()}] {done_count}/{total_target} ({rate:.1f} imgs/s, ETA {eta:.1f}s)")

    if batch_buffer:
        df_batch = pd.DataFrame(batch_buffer)
        write_header = not out_csv.exists()
        df_batch.to_csv(out_csv, mode="a", index=False, header=write_header)
        batch_buffer.clear()

    # Final cleanup & deterministic sort
    df_final = pd.read_csv(out_csv)
    df_final.drop_duplicates(subset=["id"], keep="last", inplace=True)
    df_final.sort_values(by="id", inplace=True)
    df_final.to_csv(out_csv, index=False)

    # Verification
    assert len(df_final) == total_target, f"Expected {total_target} rows, got {len(df_final)}"
    assert_clean_forensic_features(FORENSIC_FEATURE_NAMES)
    for col in FORENSIC_FEATURE_NAMES:
        assert col in df_final.columns, f"Missing feature {col}"
        assert not df_final[col].isnull().any(), f"Found nulls in feature {col}"

    print(f"[{split_name.upper()}] Finished in {time.time() - t0:.1f}s. Saved {len(df_final)} rows to {out_csv}")
    return out_csv


if __name__ == "__main__":
    print("Building TRAIN forensic cache...")
    build_forensic_feature_cache("train")
    print("Building VAL forensic cache...")
    build_forensic_feature_cache("val")
