"""
Independent Verification Audit for CrossVerify Phase 1B-C.

Audits criteria A through R:
A. canonical test row count = 1,650
B. canonical test unique record IDs = 150
C. 11 rows per record_id
D. no duplicate IDs
E. CNN prediction alignment = 1:1
F. M3 feature alignment = 1:1
G. zero null CNN probabilities
H. zero null M3 probabilities
I. probability ranges valid in [0, 1]
J. corrected fusion SHA matches expected hash (73931e2fa90b84f83b8d61348aeed9729a802bf52c0e80704f62f5f13a5281e2)
K. M3 SHA matches expected hash (a0d3dd8a4c544ecd307e0926addda7b0b4d3a187138c13307cd582d6500d55f3)
L. threshold exactly 0.82
M. no fit/refit calls in evaluation runner
N. no threshold search in evaluation runner
O. historical artifacts unchanged
P. train/test record_id overlap = 0
Q. validation/test record_id overlap = 0
R. final prediction count = 1,650

Generates: outputs/fusion/corrected_fusion_test_audit.json
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

TEST_SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "test.csv"
TRAIN_SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
VAL_SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "val.csv"

M2_TEST_PREDS_PATH = PROJECT_ROOT / "models" / "cnn" / "test_predictions.csv"
M3_MODEL_PATH = PROJECT_ROOT / "outputs" / "final_m3_model_freeze" / "final_m3_model.joblib"
M3_FEATURES_PATH = PROJECT_ROOT / "outputs" / "features" / "test_features.csv"
FUSION_MODEL_PATH = PROJECT_ROOT / "models" / "fusion" / "corrected_fusion_model.joblib"
RUNNER_SCRIPT_PATH = PROJECT_ROOT / "src" / "m3_crossmodal" / "fusion" / "run_corrected_final_test_evaluation.py"

OUT_DIR = PROJECT_ROOT / "outputs" / "fusion"
OUT_PREDS_PATH = OUT_DIR / "corrected_fusion_test_predictions.csv"
OUT_METRICS_PATH = OUT_DIR / "corrected_fusion_test_metrics.json"
OUT_ATTACK_PATH = OUT_DIR / "corrected_fusion_test_attack_wise.csv"
OUT_FAMILY_PATH = OUT_DIR / "corrected_fusion_test_family_wise.csv"
OUT_CNN_PROB_PATH = OUT_DIR / "cnn_probability_test.csv"
OUT_AUDIT_PATH = OUT_DIR / "corrected_fusion_test_audit.json"

HISTORICAL_HASHES = {
    "outputs/fusion/train_oof_alignment.csv": "132b5ebf93eb2136c40adccd5fc12d9852c2d43dfda6d7e455c8883a52bfe9b9",
    "outputs/fusion/final_fusion_freeze/final_fusion_model.joblib": "4e8391922ec267865abecb2348c256ed6f6c981986876772bfa6f17ab81110d2",
    "outputs/fusion/final_fusion_freeze/final_fusion_config.json": "18007b4aa12e70c2bc47f655dd4423bb836ee003567b1cf906755b8dff9637c7",
    "outputs/fusion/oof/train_oof_predictions.csv": "6cee59da0633bf90773cbf49719a8293a51fb6e1a3f73b99b2f8ecc7ebcc2673",
    "outputs/fusion/oof/cnn_oof_predictions.csv": "d602c271009bff4dedf19c8c4009a66ea225d4c02ef371f9d438bf8a97cbc415",
    "models/cnn/test_predictions.csv": "fdaafab5e7b2477164b2b64709af2d44b17d0f23ff33c9f8c8e4877338b5e77b",
    "models/cnn/val_predictions.csv": "43742f26b02cf1108d318f01da7235fa40c2cc5208e12ff0d42dec2945d35f0a",
    "data/splits/train.csv": "12177bb673263f8de2913544aaffe38260446ae016a75acf6788130eb3403eae",
    "data/splits/val.csv": "e2117afadc6d7f62a74845f1a03ef250a06531fd4e3a47da12a75c3f1289d52f",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_audit() -> bool:
    print("=" * 80)
    print("RUNNING CROSSVERIFY PHASE 1B-C FINAL TEST EVALUATION AUDIT")
    print("=" * 80)

    audit_results: Dict[str, Any] = {}
    all_passed = True

    test_df = pd.read_csv(TEST_SPLIT_PATH)
    preds_df = pd.read_csv(OUT_PREDS_PATH)

    # Check A: canonical test row count = 1,650
    try:
        assert len(test_df) == 1650, f"Expected 1,650 rows, got {len(test_df)}"
        print(f"[PASS] A. Canonical test row count = {len(test_df)}")
        audit_results["check_a_test_row_count"] = {"status": "PASS", "count": len(test_df)}
    except Exception as e:
        print(f"[FAIL] Check A: {e}")
        audit_results["check_a_test_row_count"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check B: canonical test unique record IDs = 150
    try:
        n_rec = int(test_df["record_id"].nunique())
        assert n_rec == 150, f"Expected 150 unique record IDs, got {n_rec}"
        print(f"[PASS] B. Canonical test unique record IDs = {n_rec}")
        audit_results["check_b_unique_record_ids"] = {"status": "PASS", "count": n_rec}
    except Exception as e:
        print(f"[FAIL] Check B: {e}")
        audit_results["check_b_unique_record_ids"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check C: 11 rows per record_id
    try:
        counts = test_df["record_id"].value_counts()
        assert (counts == 11).all(), "Not all record IDs have exactly 11 rows"
        print("[PASS] C. Exactly 11 rows per record_id across all 150 records.")
        audit_results["check_c_rows_per_record"] = {"status": "PASS", "rows_per_record": 11}
    except Exception as e:
        print(f"[FAIL] Check C: {e}")
        audit_results["check_c_rows_per_record"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check D: no duplicate IDs
    try:
        n_ids = int(test_df["id"].nunique())
        assert n_ids == len(test_df), f"Duplicate sample IDs: {len(test_df) - n_ids}"
        print(f"[PASS] D. Zero duplicate IDs in test set ({n_ids} unique).")
        audit_results["check_d_no_duplicate_ids"] = {"status": "PASS", "unique_ids": n_ids}
    except Exception as e:
        print(f"[FAIL] Check D: {e}")
        audit_results["check_d_no_duplicate_ids"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check E: CNN prediction alignment = 1:1
    try:
        cnn_df = pd.read_csv(M2_TEST_PREDS_PATH)
        assert len(cnn_df) == 1650
        assert (cnn_df["id"] == test_df["id"]).all()
        assert (cnn_df["record_id"] == test_df["record_id"]).all()
        assert (preds_df["cnn_probability"] == cnn_df["cnn_probability"].round(8)).all()
        print("[PASS] E. CNN prediction alignment is 1:1 with test.csv.")
        audit_results["check_e_cnn_alignment"] = {"status": "PASS"}
    except Exception as e:
        print(f"[FAIL] Check E: {e}")
        audit_results["check_e_cnn_alignment"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check F: M3 feature alignment = 1:1
    try:
        m3_feat = pd.read_csv(M3_FEATURES_PATH)
        assert len(m3_feat) == 1650
        assert (m3_feat["id"] == test_df["id"]).all()
        assert (m3_feat["record_id"] == test_df["record_id"]).all()
        print("[PASS] F. M3 feature alignment is 1:1 with test.csv.")
        audit_results["check_f_m3_alignment"] = {"status": "PASS"}
    except Exception as e:
        print(f"[FAIL] Check F: {e}")
        audit_results["check_f_m3_alignment"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check G: zero null CNN probabilities
    try:
        nulls_cnn = int(preds_df["cnn_probability"].isnull().sum())
        assert nulls_cnn == 0, f"Null CNN probabilities: {nulls_cnn}"
        print(f"[PASS] G. Zero null CNN probabilities ({nulls_cnn} nulls).")
        audit_results["check_g_zero_null_cnn"] = {"status": "PASS", "null_count": 0}
    except Exception as e:
        print(f"[FAIL] Check G: {e}")
        audit_results["check_g_zero_null_cnn"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check H: zero null M3 probabilities
    try:
        nulls_m3 = int(preds_df["m3_probability"].isnull().sum())
        assert nulls_m3 == 0, f"Null M3 probabilities: {nulls_m3}"
        print(f"[PASS] H. Zero null M3 probabilities ({nulls_m3} nulls).")
        audit_results["check_h_zero_null_m3"] = {"status": "PASS", "null_count": 0}
    except Exception as e:
        print(f"[FAIL] Check H: {e}")
        audit_results["check_h_zero_null_m3"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check I: probability ranges valid
    try:
        assert (preds_df["cnn_probability"] >= 0.0).all() and (preds_df["cnn_probability"] <= 1.0).all()
        assert (preds_df["m3_probability"] >= 0.0).all() and (preds_df["m3_probability"] <= 1.0).all()
        assert (preds_df["fusion_probability"] >= 0.0).all() and (preds_df["fusion_probability"] <= 1.0).all()
        print(f"[PASS] I. Probability ranges valid in [0, 1] (CNN: [{preds_df['cnn_probability'].min():.4f}, {preds_df['cnn_probability'].max():.4f}], M3: [{preds_df['m3_probability'].min():.4f}, {preds_df['m3_probability'].max():.4f}], Fusion: [{preds_df['fusion_probability'].min():.4f}, {preds_df['fusion_probability'].max():.4f}]).")
        audit_results["check_i_probability_ranges"] = {"status": "PASS"}
    except Exception as e:
        print(f"[FAIL] Check I: {e}")
        audit_results["check_i_probability_ranges"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check J: corrected fusion SHA matches expected hash
    try:
        f_hash = sha256_file(FUSION_MODEL_PATH)
        exp_f = "73931e2fa90b84f83b8d61348aeed9729a802bf52c0e80704f62f5f13a5281e2"
        assert f_hash == exp_f, f"Fusion hash mismatch: {f_hash}"
        print(f"[PASS] J. Corrected fusion model SHA-256 matches expected: {f_hash}")
        audit_results["check_j_fusion_model_hash"] = {"status": "PASS", "sha256": f_hash}
    except Exception as e:
        print(f"[FAIL] Check J: {e}")
        audit_results["check_j_fusion_model_hash"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check K: M3 SHA matches expected hash
    try:
        m_hash = sha256_file(M3_MODEL_PATH)
        exp_m = "a0d3dd8a4c544ecd307e0926addda7b0b4d3a187138c13307cd582d6500d55f3"
        assert m_hash == exp_m, f"M3 hash mismatch: {m_hash}"
        print(f"[PASS] K. M3 model SHA-256 matches expected: {m_hash}")
        audit_results["check_k_m3_model_hash"] = {"status": "PASS", "sha256": m_hash}
    except Exception as e:
        print(f"[FAIL] Check K: {e}")
        audit_results["check_k_m3_model_hash"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check L: threshold exactly 0.82
    try:
        with open(OUT_METRICS_PATH) as f:
            metrics_json = json.load(f)
        assert metrics_json["operating_threshold"] == 0.82
        assert (preds_df["threshold"] == 0.82).all()
        # Verify predictions match threshold rule
        expected_pred = (preds_df["fusion_probability"] >= 0.82).astype(int)
        assert (preds_df["final_prediction"] == expected_pred).all()
        print("[PASS] L. Operating threshold is strictly 0.82 across all samples and metrics.")
        audit_results["check_l_threshold_exact"] = {"status": "PASS", "threshold": 0.82}
    except Exception as e:
        print(f"[FAIL] Check L: {e}")
        audit_results["check_l_threshold_exact"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check M: no fit/refit calls
    try:
        with open(RUNNER_SCRIPT_PATH, "r") as f:
            script_text = f.read()
        assert ".fit(" not in script_text
        assert ".fit_transform(" not in script_text
        assert ".partial_fit(" not in script_text
        print("[PASS] M. Zero fit/refit/fit_transform calls in evaluation runner script.")
        audit_results["check_m_no_fit_calls"] = {"status": "PASS"}
    except Exception as e:
        print(f"[FAIL] Check M: {e}")
        audit_results["check_m_no_fit_calls"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check N: no threshold search
    try:
        assert "threshold_grid" not in script_text
        assert "np.linspace" not in script_text
        assert "best_f1" not in script_text
        print("[PASS] N. Zero threshold search loops or optimization in evaluation runner script.")
        audit_results["check_n_no_threshold_search"] = {"status": "PASS"}
    except Exception as e:
        print(f"[FAIL] Check N: {e}")
        audit_results["check_n_no_threshold_search"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check O: historical artifacts unchanged
    try:
        for rel_path, exp_h in HISTORICAL_HASHES.items():
            full_p = PROJECT_ROOT / rel_path
            assert full_p.exists(), f"Missing historical file: {rel_path}"
            act_h = sha256_file(full_p)
            assert act_h == exp_h, f"Hash changed for {rel_path}: expected {exp_h}, got {act_h}"
        print("[PASS] O. All historical artifacts verified byte-identical.")
        audit_results["check_o_historical_artifacts_preserved"] = {"status": "PASS"}
    except Exception as e:
        print(f"[FAIL] Check O: {e}")
        audit_results["check_o_historical_artifacts_preserved"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check P: train/test record_id overlap = 0
    try:
        tr_df = pd.read_csv(TRAIN_SPLIT_PATH)
        s_tr = set(tr_df["record_id"])
        s_te = set(test_df["record_id"])
        ov_tr = s_tr.intersection(s_te)
        assert len(ov_tr) == 0, f"Overlap between train and test: {len(ov_tr)}"
        print(f"[PASS] P. Zero train/test record_id overlap ({len(ov_tr)} overlapping).")
        audit_results["check_p_train_test_disjointness"] = {"status": "PASS", "overlap": 0}
    except Exception as e:
        print(f"[FAIL] Check P: {e}")
        audit_results["check_p_train_test_disjointness"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check Q: validation/test record_id overlap = 0
    try:
        va_df = pd.read_csv(VAL_SPLIT_PATH)
        s_va = set(va_df["record_id"])
        ov_va = s_va.intersection(s_te)
        assert len(ov_va) == 0, f"Overlap between val and test: {len(ov_va)}"
        print(f"[PASS] Q. Zero validation/test record_id overlap ({len(ov_va)} overlapping).")
        audit_results["check_q_val_test_disjointness"] = {"status": "PASS", "overlap": 0}
    except Exception as e:
        print(f"[FAIL] Check Q: {e}")
        audit_results["check_q_val_test_disjointness"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check R: final prediction count = 1,650
    try:
        assert len(preds_df) == 1650, f"Expected 1,650 predictions, got {len(preds_df)}"
        assert (preds_df["id"] == test_df["id"]).all()
        print(f"[PASS] R. Final prediction count = {len(preds_df)} (exact 1:1 match with test.csv).")
        audit_results["check_r_prediction_count"] = {"status": "PASS", "count": len(preds_df)}
    except Exception as e:
        print(f"[FAIL] Check R: {e}")
        audit_results["check_r_prediction_count"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    status_str = "PASS" if all_passed else "FAIL"
    print("=" * 80)
    print(f"PHASE 1B-C TEST EVALUATION AUDIT STATUS: {status_str}")
    print("=" * 80)

    # Generated artifact hashes
    generated_hashes = {
        "corrected_fusion_test_predictions_csv": sha256_file(OUT_PREDS_PATH) if OUT_PREDS_PATH.exists() else None,
        "corrected_fusion_test_metrics_json": sha256_file(OUT_METRICS_PATH) if OUT_METRICS_PATH.exists() else None,
        "corrected_fusion_test_attack_wise_csv": sha256_file(OUT_ATTACK_PATH) if OUT_ATTACK_PATH.exists() else None,
        "corrected_fusion_test_family_wise_csv": sha256_file(OUT_FAMILY_PATH) if OUT_FAMILY_PATH.exists() else None,
        "cnn_probability_test_csv": sha256_file(OUT_CNN_PROB_PATH) if OUT_CNN_PROB_PATH.exists() else None,
    }

    audit_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": status_str,
        "checks": audit_results,
        "generated_artifact_hashes": generated_hashes,
    }

    with open(OUT_AUDIT_PATH, "w") as f:
        json.dump(audit_payload, f, indent=2)
    print(f"Saved audit report to: {OUT_AUDIT_PATH}")

    return all_passed


if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)
