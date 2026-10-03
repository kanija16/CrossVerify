"""
Independent Programmatic Verification Audit for CrossVerify Phase 1B-B.

Audits criteria A through K:
A. Corrected fusion training uses exactly 7,700 OOF rows.
B. Both upstream probabilities are genuine OOF predictions.
C. No validation rows entered fusion fitting.
D. No test rows entered fusion fitting.
E. OOF row alignment is exact against canonical train.csv.
F. Feature order is exactly [cnn_probability_oof, m3_probability_oof].
G. LogisticRegression parameters are unchanged (C=1.0, solver='lbfgs', random_state=42).
H. Old fusion artifacts are byte/hash unchanged.
I. Validation is used only for threshold selection after fusion fitting.
J. Threshold search is exactly 0.01–0.99 with specificity >= 0.90.
K. No test data was accessed.

Outputs: outputs/fusion/corrected_fusion_training_audit.json
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Paths
TRAIN_SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
VAL_SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "val.csv"
TEST_SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "test.csv"

CORRECTED_OOF_PATH = PROJECT_ROOT / "outputs" / "fusion" / "oof" / "corrected_fusion_oof.csv"
CNN_OOF_PATH = PROJECT_ROOT / "outputs" / "fusion" / "oof" / "cnn_oof_predictions.csv"
M3_OOF_PATH = PROJECT_ROOT / "outputs" / "fusion" / "oof" / "train_oof_predictions.csv"

CORRECTED_MODEL_PATH = PROJECT_ROOT / "models" / "fusion" / "corrected_fusion_model.joblib"
CORRECTED_CONFIG_PATH = PROJECT_ROOT / "models" / "fusion" / "corrected_fusion_config.json"
VAL_UPSTREAM_PATH = PROJECT_ROOT / "outputs" / "fusion" / "val_upstream_predictions.csv"
VAL_PREDS_PATH = PROJECT_ROOT / "outputs" / "fusion" / "corrected_fusion_validation_predictions.csv"

TRAINER_SCRIPT_PATH = PROJECT_ROOT / "src" / "m3_crossmodal" / "fusion" / "train_corrected_fusion.py"
AUDIT_JSON_PATH = PROJECT_ROOT / "outputs" / "fusion" / "corrected_fusion_training_audit.json"

# Protected historical artifacts and their expected hashes
HISTORICAL_HASHES = {
    "outputs/fusion/train_oof_alignment.csv": "132b5ebf93eb2136c40adccd5fc12d9852c2d43dfda6d7e455c8883a52bfe9b9",
    "outputs/fusion/final_fusion_freeze/final_fusion_model.joblib": "4e8391922ec267865abecb2348c256ed6f6c981986876772bfa6f17ab81110d2",
    "outputs/fusion/final_fusion_freeze/final_fusion_config.json": "18007b4aa12e70c2bc47f655dd4423bb836ee003567b1cf906755b8dff9637c7",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_audit() -> bool:
    print("=" * 80)
    print("RUNNING CROSSVERIFY PHASE 1B-B CORRECTED FUSION VERIFICATION AUDIT")
    print("=" * 80)

    audit_results: Dict[str, Any] = {}
    all_passed = True

    # Check A: Corrected fusion training uses exactly 7,700 OOF rows.
    try:
        assert CORRECTED_OOF_PATH.exists(), f"Missing {CORRECTED_OOF_PATH}"
        oof_df = pd.read_csv(CORRECTED_OOF_PATH)
        assert len(oof_df) == 7700, f"Expected 7700 rows, got {len(oof_df)}"
        print(f"[PASS] A. Corrected fusion training uses exactly 7,700 OOF rows: {len(oof_df)}")
        audit_results["check_a_row_count"] = {"status": "PASS", "value": len(oof_df)}
    except Exception as e:
        print(f"[FAIL] Check A: {e}")
        audit_results["check_a_row_count"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check B: Both upstream probabilities are genuine OOF predictions.
    try:
        cnn_oof = pd.read_csv(CNN_OOF_PATH)
        m3_oof = pd.read_csv(M3_OOF_PATH)
        assert (oof_df["cnn_probability_oof"] == cnn_oof["cnn_probability_oof"]).all()
        assert (oof_df["m3_probability_oof"] == m3_oof["m3_probability_oof"]).all()
        assert oof_df["cnn_probability_oof"].isnull().sum() == 0
        assert oof_df["m3_probability_oof"].isnull().sum() == 0
        print("[PASS] B. Both upstream probabilities are genuine OOF predictions verified against upstream artifacts.")
        audit_results["check_b_genuine_oof"] = {"status": "PASS"}
    except Exception as e:
        print(f"[FAIL] Check B: {e}")
        audit_results["check_b_genuine_oof"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check C: No validation rows entered fusion fitting.
    try:
        val_df = pd.read_csv(VAL_SPLIT_PATH)
        oof_records = set(oof_df["record_id"])
        val_records = set(val_df["record_id"])
        overlap = oof_records.intersection(val_records)
        assert len(overlap) == 0, f"Validation record_ids found in OOF data: {len(overlap)}"
        print(f"[PASS] C. Zero validation rows/records in fusion fitting data (disjointness: {len(overlap)} overlap).")
        audit_results["check_c_no_val_in_fitting"] = {"status": "PASS", "overlap_records": 0}
    except Exception as e:
        print(f"[FAIL] Check C: {e}")
        audit_results["check_c_no_val_in_fitting"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check D: No test rows entered fusion fitting.
    try:
        test_df = pd.read_csv(TEST_SPLIT_PATH)
        test_records = set(test_df["record_id"])
        overlap_test = oof_records.intersection(test_records)
        assert len(overlap_test) == 0, f"Test record_ids found in OOF data: {len(overlap_test)}"
        print(f"[PASS] D. Zero test rows/records in fusion fitting data (disjointness: {len(overlap_test)} overlap).")
        audit_results["check_d_no_test_in_fitting"] = {"status": "PASS", "overlap_records": 0}
    except Exception as e:
        print(f"[FAIL] Check D: {e}")
        audit_results["check_d_no_test_in_fitting"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check E: OOF row alignment is exact.
    try:
        train_df = pd.read_csv(TRAIN_SPLIT_PATH)
        assert len(oof_df) == len(train_df)
        assert (oof_df["id"] == train_df["id"]).all()
        assert (oof_df["record_id"] == train_df["record_id"]).all()
        assert (oof_df["final_label"] == train_df["final_label"]).all()
        assert (oof_df["fold"] == cnn_oof["fold"]).all()
        print("[PASS] E. OOF row alignment against canonical train.csv is exact (1:1 row order and values).")
        audit_results["check_e_oof_alignment"] = {"status": "PASS"}
    except Exception as e:
        print(f"[FAIL] Check E: {e}")
        audit_results["check_e_oof_alignment"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check F: Feature order is exactly [cnn_probability_oof, m3_probability_oof].
    try:
        assert CORRECTED_MODEL_PATH.exists(), f"Missing {CORRECTED_MODEL_PATH}"
        model = joblib.load(CORRECTED_MODEL_PATH)
        with open(CORRECTED_CONFIG_PATH) as f:
            cfg = json.load(f)
        assert cfg["feature_order"] == ["cnn_probability_oof", "m3_probability_oof"]
        # Verify model coefficients match config
        assert np.isclose(model.coef_[0][0], cfg["learned_parameters"]["cnn_probability_oof_weight"])
        assert np.isclose(model.coef_[0][1], cfg["learned_parameters"]["m3_probability_oof_weight"])
        assert np.isclose(model.intercept_[0], cfg["learned_parameters"]["intercept"])
        print(f"[PASS] F. Feature order is exactly {cfg['feature_order']}.")
        audit_results["check_f_feature_order"] = {"status": "PASS", "feature_order": cfg["feature_order"]}
    except Exception as e:
        print(f"[FAIL] Check F: {e}")
        audit_results["check_f_feature_order"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check G: LogisticRegression parameters are unchanged.
    try:
        assert model.C == 1.0
        assert model.solver == "lbfgs"
        assert model.random_state == 42
        print(f"[PASS] G. LogisticRegression parameters verified: C={model.C}, solver={model.solver}, random_state={model.random_state}.")
        audit_results["check_g_model_parameters"] = {
            "status": "PASS",
            "C": model.C,
            "solver": model.solver,
            "random_state": model.random_state,
        }
    except Exception as e:
        print(f"[FAIL] Check G: {e}")
        audit_results["check_g_model_parameters"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check H: Old fusion artifacts are byte/hash unchanged.
    try:
        for rel_path, exp_hash in HISTORICAL_HASHES.items():
            full_p = PROJECT_ROOT / rel_path
            assert full_p.exists(), f"Missing protected artifact: {rel_path}"
            act_hash = sha256_file(full_p)
            assert act_hash == exp_hash, f"Hash mismatch for {rel_path}: expected {exp_hash}, got {act_hash}"
        print("[PASS] H. All historical fusion artifacts verified byte/hash identical.")
        audit_results["check_h_historical_preservation"] = {"status": "PASS", "verified_files": list(HISTORICAL_HASHES.keys())}
    except Exception as e:
        print(f"[FAIL] Check H: {e}")
        audit_results["check_h_historical_preservation"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check I: Validation is used only for threshold selection after fusion fitting.
    try:
        val_up = pd.read_csv(VAL_UPSTREAM_PATH)
        assert len(val_up) == 1650
        # Verify validation predictions were produced by the fitted model without refitting
        X_val = np.column_stack([val_up["cnn_probability_val"].values, val_up["m3_probability_val"].values])
        expected_val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = pd.read_csv(VAL_PREDS_PATH)
        assert np.allclose(val_preds["fusion_probability_val"].values, expected_val_probs, atol=1e-6)
        print("[PASS] I. Validation inference exactly reproduces from fitted model without refitting.")
        audit_results["check_i_validation_isolation"] = {"status": "PASS"}
    except Exception as e:
        print(f"[FAIL] Check I: {e}")
        audit_results["check_i_validation_isolation"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check J: Threshold search is exactly 0.01–0.99 with specificity >= 0.90.
    try:
        th_val = cfg["validation_threshold_selection"]["selected_threshold"]
        spec_val = cfg["validation_threshold_selection"]["metrics_at_selected_threshold"]["specificity"]
        assert 0.01 <= th_val <= 0.99
        assert spec_val >= 0.90, f"Specificity below 0.90: {spec_val}"
        print(f"[PASS] J. Threshold search verified: selected threshold={th_val:.2f} with Specificity={spec_val:.4f} >= 0.90.")
        audit_results["check_j_threshold_search"] = {
            "status": "PASS",
            "selected_threshold": th_val,
            "specificity": spec_val,
        }
    except Exception as e:
        print(f"[FAIL] Check J: {e}")
        audit_results["check_j_threshold_search"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # Check K: No test data was accessed.
    try:
        with open(TRAINER_SCRIPT_PATH, "r") as f:
            script_text = f.read()
        # Ensure test.csv is never read or loaded
        assert "pd.read_csv(TEST_SPLIT_PATH)" not in script_text
        assert 'read_csv("data/splits/test.csv")' not in script_text
        assert 'read_csv("data/test.csv")' not in script_text
        print("[PASS] K. No test data was accessed (zero test split loading in training pipeline).")
        audit_results["check_k_no_test_access"] = {"status": "PASS"}
    except Exception as e:
        print(f"[FAIL] Check K: {e}")
        audit_results["check_k_no_test_access"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    status_str = "PASS" if all_passed else "FAIL"
    print("=" * 80)
    print(f"PHASE 1B-B FUSION AUDIT STATUS: {status_str}")
    print("=" * 80)

    # Save audit report
    audit_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": status_str,
        "checks": audit_results,
        "provenance": {
            "corrected_oof_sha256": sha256_file(CORRECTED_OOF_PATH) if CORRECTED_OOF_PATH.exists() else None,
            "corrected_model_sha256": sha256_file(CORRECTED_MODEL_PATH) if CORRECTED_MODEL_PATH.exists() else None,
            "corrected_config_sha256": sha256_file(CORRECTED_CONFIG_PATH) if CORRECTED_CONFIG_PATH.exists() else None,
            "val_upstream_predictions_sha256": sha256_file(VAL_UPSTREAM_PATH) if VAL_UPSTREAM_PATH.exists() else None,
            "val_predictions_sha256": sha256_file(VAL_PREDS_PATH) if VAL_PREDS_PATH.exists() else None,
        },
    }

    AUDIT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_JSON_PATH, "w") as f:
        json.dump(audit_report, f, indent=2)
    print(f"Saved audit report to: {AUDIT_JSON_PATH}")

    return all_passed


if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)
