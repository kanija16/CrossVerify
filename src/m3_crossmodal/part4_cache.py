"""
part4_cache.py
--------------
Batch feature extraction and caching for M3 Part 4.

Runs the live Part 3 cross-modal pipeline (live OCR + live QR + Part 2 consistency)
once per image across the fixed train/val/test splits, computing the expanded 16-feature set.
Caches tables in outputs/features/ to avoid re-extracting live features during model fitting.
Includes incremental checkpointing and automatic resume capability.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from . import config
from .data_interface import load_split_csv, resolve_image_path
from .live_pipeline import run_live_crossmodal
from .part4_features import (
    PART4_FEATURE_ORDER_NO_FAMILY,
    PART4_FEATURE_ORDER_WITH_FAMILY,
    assert_no_forbidden_features,
    extract_part4_features,
)

FEATURES_CACHE_DIR = Path("outputs/features")


def _process_single_image(row_tuple: Tuple[str, str, str, str, str, str, int]) -> Dict[str, Any]:
    """
    Top-level worker function for multiprocessing.
    row_tuple: (sample_id, record_id, split_name, image_path_str, document_family, tamper_type, final_label)
    """
    sample_id, record_id, split_name, image_path_str, document_family, tamper_type, final_label = row_tuple

    # 1. Run live cross-modal extraction on image pixels
    live_res = run_live_crossmodal(image_path_str, document_family)

    # 2. Extract 16 tabular features (preserving NaNs)
    feat_dict = extract_part4_features(
        ocr_fields=live_res["ocr_extracted_fields"],
        qr_fields=live_res["qr_decoded_fields"],
        consistency_vector=live_res["consistency_vector"],
        document_family=document_family,
    )

    # 3. Assemble row record (metadata strictly separated from feature extraction)
    row_record: Dict[str, Any] = {
        "id": sample_id,
        "record_id": record_id,
        "split": split_name,
        "tamper_type": tamper_type,
        "final_label": final_label,
    }
    row_record.update(feat_dict)
    return row_record


def build_split_feature_cache(
    split_name: str,
    max_workers: int = 8,
    limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
    batch_write_size: int = 100,
) -> Path:
    """
    Extracts live features for one M1 split and caches the resulting table.
    Supports incremental checkpointing and resume by sample 'id'.
    """
    out_dir = output_dir or FEATURES_CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_csv = out_dir / f"{split_name}_features.csv"

    csv_path_map = {
        "train": config.TRAIN_CSV,
        "val": config.VAL_CSV,
        "test": config.TEST_CSV,
    }
    if split_name not in csv_path_map:
        raise ValueError(f"Unknown split name: {split_name!r}")

    df_meta = load_split_csv(csv_path_map[split_name])
    if limit is not None:
        df_meta = df_meta.iloc[:limit]

    total_target = len(df_meta)

    # Check for existing checkpoint using unique 'id'
    completed_ids: Set[str] = set()
    if cache_csv.exists():
        try:
            df_existing = pd.read_csv(cache_csv)
            if "id" in df_existing.columns:
                completed_ids = set(df_existing["id"].astype(str))
                if len(completed_ids) >= total_target:
                    print(f"[{split_name.upper()}] Cache already complete with {len(completed_ids)} records at {cache_csv}")
                    return cache_csv
                print(f"[{split_name.upper()}] Resuming: {len(completed_ids)}/{total_target} already cached.")
            elif "record_id" in df_existing.columns and len(df_existing) < total_target:
                # Previous flawed deduplication on record_id, reset cache
                print(f"[{split_name.upper()}] Resetting previous cache to use unique sample 'id'...")
                cache_csv.unlink(missing_ok=True)
                completed_ids = set()
        except Exception as e:
            print(f"[{split_name.upper()}] Could not read existing cache ({e}), starting fresh.")
            completed_ids = set()

    tasks = []
    for _, row in df_meta.iterrows():
        sample_id = str(row[config.COL_ID])
        if sample_id in completed_ids:
            continue
        rec_id = str(row[config.COL_RECORD_ID])
        img_p = str(resolve_image_path(row[config.COL_IMAGE_PATH]))
        fam = str(row[config.COL_DOCUMENT_FAMILY])
        tt = str(row[config.COL_TAMPER_TYPE])
        lbl = int(config.FINAL_LABEL_MAP[row[config.COL_FINAL_LABEL]])
        tasks.append((sample_id, rec_id, split_name, img_p, fam, tt, lbl))

    pending_count = len(tasks)
    print(f"[{split_name.upper()}] Processing {pending_count} pending images (workers={max_workers})...")

    if pending_count == 0:
        return cache_csv

    t0 = time.time()
    batch_buffer: List[Dict[str, Any]] = []
    done_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_single_image, t) for t in tasks]
        for f in as_completed(futures):
            res = f.result()
            batch_buffer.append(res)
            done_count += 1

            if len(batch_buffer) >= batch_write_size:
                df_batch = pd.DataFrame(batch_buffer)
                write_header = not cache_csv.exists()
                df_batch.to_csv(cache_csv, mode="a", index=False, header=write_header)
                batch_buffer.clear()
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (pending_count - done_count) / rate if rate > 0 else 0
                print(f"[{split_name.upper()}] {len(completed_ids) + done_count}/{total_target} ({rate:.1f} imgs/s, elapsed {elapsed:.1f}s, ETA {eta:.1f}s)")

    # Flush remaining buffer
    if batch_buffer:
        df_batch = pd.DataFrame(batch_buffer)
        write_header = not cache_csv.exists()
        df_batch.to_csv(cache_csv, mode="a", index=False, header=write_header)
        batch_buffer.clear()

    # Final cleanup & deterministic row sorting on unique sample id
    df_final = pd.read_csv(cache_csv)
    df_final.drop_duplicates(subset=["id"], keep="last", inplace=True)
    df_final.sort_values(by="id", inplace=True)
    df_final.to_csv(cache_csv, index=False)

    total_time = time.time() - t0
    print(f"[{split_name.upper()}] Finished in {total_time:.1f}s. Saved {len(df_final)} rows to {cache_csv}")
    return cache_csv


def load_cached_features(
    split_name: str,
    include_family: bool = False,
    cache_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Loads cached features and cleanly separates:
      - X: DataFrame with exactly the model features (15 or 16), zero forbidden columns
      - y: 1D numpy array of binary targets (0=genuine, 1=forged)
      - metadata: DataFrame with id, record_id, split, tamper_type, final_label for post-hoc evaluation

    Guarantees:
      - assert_no_forbidden_features(X.columns) is run on X.
    """
    c_dir = cache_dir or FEATURES_CACHE_DIR
    cache_csv = c_dir / f"{split_name}_features.csv"
    if not cache_csv.exists():
        raise FileNotFoundError(f"Feature cache not found: {cache_csv}. Run build_split_feature_cache first.")

    df = pd.read_csv(cache_csv)

    feature_cols = PART4_FEATURE_ORDER_WITH_FAMILY if include_family else PART4_FEATURE_ORDER_NO_FAMILY

    # Strictly verify that forbidden columns never enter X
    assert_no_forbidden_features(feature_cols)

    X = df[feature_cols].copy()
    y = df["final_label"].to_numpy(dtype=np.int64)
    metadata = df[["id", "record_id", "split", "tamper_type", "final_label"]].copy()

    return X, y, metadata
