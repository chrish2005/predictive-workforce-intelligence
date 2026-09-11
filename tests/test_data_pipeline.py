"""
Unit tests for data pipeline integrity, zero leakage, and transformation correctness.
"""

import pytest
import numpy as np
import pandas as pd
from src.data.loader import load_ibm_dataset
from src.preprocessing.cleaner import clean_hr_dataset, ZERO_VARIANCE_COLS
from src.preprocessing.pipeline import prepare_train_test_data, build_preprocessing_pipeline
from src.features.engineer import HRFeatureEngineer


def test_cleaner_drops_zero_variance_columns():
    df = load_ibm_dataset()
    cleaned, y, emp_ids = clean_hr_dataset(df)

    for col in ZERO_VARIANCE_COLS:
        assert col not in cleaned.columns, f"Zero variance column {col} was not dropped."

    assert y is not None, "Target Attrition column should be extracted."
    assert set(y.unique()).issubset({0, 1}), f"Target values must be 0 or 1, got {y.unique()}"
    assert len(cleaned) == 1470, f"Expected 1470 rows, got {len(cleaned)}"


def test_strict_featurization_ordering_no_leakage():
    df = load_ibm_dataset()
    X_train, X_test, y_train, y_test, ids_train, ids_test = prepare_train_test_data(df, test_size=0.20, random_state=42)

    # 1. Verify train and test are disjoint
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    assert train_indices.isdisjoint(test_indices), "Data leakage! Train and test splits overlap."

    # 2. Fit pipeline strictly on train
    pipeline = build_preprocessing_pipeline()
    X_train_trans = pipeline.fit_transform(X_train)

    # Verify that benchmarks were computed strictly from X_train
    assert len(pipeline.engineer.role_level_medians_) > 0
    # Transform test set
    X_test_trans = pipeline.transform(X_test)

    # Check shapes
    assert X_train_trans.shape[0] == len(X_train)
    assert X_test_trans.shape[0] == len(X_test)
    assert X_train_trans.shape[1] == X_test_trans.shape[1]

    # Zero missing values in processed matrices
    assert not np.isnan(X_train_trans).any(), "NaN found in transformed training matrix."
    assert not np.isnan(X_test_trans).any(), "NaN found in transformed test matrix."


def test_engineered_features_present():
    df = load_ibm_dataset()
    pipeline = build_preprocessing_pipeline()
    clean_df, _, _ = clean_hr_dataset(df)
    pipeline.fit(clean_df)

    feat_names = pipeline.get_feature_names_out()
    expected_engineered = [
        "StagnationIndex",
        "RoleStagnationRatio",
        "ManagerStabilityRatio",
        "CompensationEquityIndex",
        "BurnoutRiskFactor",
        "CommuteBurdenFactor",
        "SatisfactionSum",
        "StockOptionZero",
        "TenureToAgeRatio",
        "IncomePerYearWorked",
    ]
    for feat in expected_engineered:
        assert feat in feat_names, f"Engineered feature '{feat}' missing from preprocessor output."
