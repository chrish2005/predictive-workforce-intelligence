"""
Unit tests for model predictive power, calibration, and discrimination criteria:
- ROC-AUC >= 0.82
- PR-AUC >= 0.58
- Calibrated Brier Score <= 0.12
- Precision@Top10% >= 0.60
"""

import pytest
import numpy as np
from src.data.loader import load_ibm_dataset
from src.preprocessing.pipeline import prepare_train_test_data
from src.models.train import load_model_artifacts
from src.models.evaluate import evaluate_attrition_model


@pytest.fixture(scope="module")
def model_and_test_data():
    calibrated_model, raw_model, xgb_model, preprocessor, metadata = load_model_artifacts()
    raw_df = load_ibm_dataset()
    X_train_df, X_test_df, y_train, y_test, _, _ = prepare_train_test_data(raw_df, test_size=0.20, random_state=42)

    X_test = preprocessor.transform(X_test_df)
    y_test_arr = np.array(y_test)

    return calibrated_model, preprocessor, X_test, y_test_arr, metadata


def test_roc_auc_benchmark(model_and_test_data):
    calibrated_model, _, X_test, y_test, metadata = model_and_test_data
    results = evaluate_attrition_model(calibrated_model, X_test, y_test)

    # Test set ROC-AUC must achieve >= 0.82
    assert results["roc_auc"] >= 0.82, (
        f"ROC-AUC {results['roc_auc']} is below target threshold of 0.82."
    )


def test_brier_score_calibration_benchmark(model_and_test_data):
    calibrated_model, _, X_test, y_test, metadata = model_and_test_data
    results = evaluate_attrition_model(calibrated_model, X_test, y_test)

    # Calibrated Brier score must be <= 0.12
    assert results["brier_score"] <= 0.12, (
        f"Brier score {results['brier_score']} exceeds maximum calibrated threshold of 0.12."
    )


def test_pr_auc_and_precision_top10(model_and_test_data):
    calibrated_model, _, X_test, y_test, metadata = model_and_test_data
    results = evaluate_attrition_model(calibrated_model, X_test, y_test)

    # PR-AUC reflects class-imbalance capability
    assert results["pr_auc"] >= 0.58, (
        f"PR-AUC {results['pr_auc']} is below target threshold of 0.58."
    )

    # Precision in top 10% reflects actionable HR triage capacity
    assert results["precision_at_top10"] >= 0.60, (
        f"Precision@Top10% {results['precision_at_top10']} is below target of 0.60."
    )
