"""
Unit tests for Universal HR Ingestion, Preprocessing & Auto-Training Engine.
Verifies dynamic target discovery, robust zero-leakage transformation,
and automated model calibration across arbitrary workforce schemas.
"""

import pytest
import pandas as pd
import numpy as np

from src.preprocessing.universal import (
    auto_detect_target_column,
    auto_detect_id_column,
    clean_and_encode_universal,
    UniversalHRPreprocessor,
)
from src.models.auto_trainer import train_universal_workforce_bundle, WorkforceCohortBundle
from src.data.loader import get_data_filepath


@pytest.fixture
def mock_workforce_dataframe():
    """Generates a synthetic mock workforce dataset with mixed schemas."""
    np.random.seed(42)
    n = 250
    return pd.DataFrame({
        "staff_id": [f"EMP_{i:04d}" for i in range(n)],
        "department_name": np.random.choice(["Engineering", "Sales", "HR", "Product"], size=n),
        "monthly_salary": np.random.uniform(3000, 15000, size=n),
        "weekly_hours": np.random.randint(35, 70, size=n),
        "satisfaction_score": np.random.uniform(0.1, 1.0, size=n),
        "projects_count": np.random.randint(2, 8, size=n),
        "constant_col": ["FixedCompanyPolicy"] * n,
        "is_left": np.random.choice([0, 1], size=n, p=[0.75, 0.25]),
    })


def test_target_and_id_detection(mock_workforce_dataframe):
    """Verify regex heuristic target and ID column auto-discovery."""
    detected_target = auto_detect_target_column(mock_workforce_dataframe)
    detected_id = auto_detect_id_column(mock_workforce_dataframe)

    assert detected_target == "is_left"
    assert detected_id == "staff_id"


def test_clean_and_encode_universal(mock_workforce_dataframe):
    """Verify cleaning, zero-variance feature elimination, and target binarization."""
    X_clean, y, emp_ids, target_col = clean_and_encode_universal(
        mock_workforce_dataframe,
        target_col="is_left",
        id_col="staff_id",
    )

    assert target_col == "is_left"
    assert "constant_col" not in X_clean.columns  # Dropped zero-variance
    assert "staff_id" not in X_clean.columns      # Dropped ID from features
    assert "is_left" not in X_clean.columns       # Dropped target from features
    assert len(emp_ids) == len(mock_workforce_dataframe)
    assert set(y.unique()).issubset({0, 1})


def test_universal_preprocessor_resilience(mock_workforce_dataframe):
    """Verify UniversalHRPreprocessor fits correctly and tolerates missing inference columns."""
    X_clean, y, _, _ = clean_and_encode_universal(mock_workforce_dataframe)

    preprocessor = UniversalHRPreprocessor()
    X_trans = preprocessor.fit_transform(X_clean)

    assert X_trans.shape[0] == len(X_clean)
    assert X_trans.shape[1] > 0
    assert not np.isnan(X_trans).any()

    # Test inference on a row with missing column (defensive handling)
    row_missing = X_clean.iloc[[0]].drop(columns=["monthly_salary"])
    X_trans_row = preprocessor.transform(row_missing)
    assert X_trans_row.shape == (1, X_trans.shape[1])
    assert not np.isnan(X_trans_row).any()


def test_train_universal_workforce_bundle_mock(mock_workforce_dataframe):
    """Verify end-to-end auto-trainer on mock workforce data."""
    bundle = train_universal_workforce_bundle(
        mock_workforce_dataframe,
        cohort_name="Mock Unit Test Cohort",
        target_col="is_left",
        id_col="staff_id",
    )

    assert isinstance(bundle, WorkforceCohortBundle)
    assert bundle.cohort_name == "Mock Unit Test Cohort"
    assert "turnover_risk" in bundle.scored_df.columns
    assert "RiskTier" in bundle.scored_df.columns
    assert "expected_financial_exposure" in bundle.scored_df.columns

    # Probability bounds
    assert bundle.scored_df["turnover_risk"].between(0.0, 1.0).all()

    # Model evaluation metrics presence
    metrics = bundle.metadata["test_metrics"]
    assert "roc_auc" in metrics
    assert "brier_score" in metrics
    assert "precision_at_top10" in metrics
    assert metrics["brier_score"] <= 0.35

    # Verify SHAP explainer runs on this bundle
    top_driver = bundle.explainer.explain(bundle.X_trans[:10])
    assert top_driver.shape[0] == 10
    assert top_driver.shape[1] == bundle.X_trans.shape[1]


def test_train_universal_workforce_bundle_tech15k():
    """Verify bundle generation on real-world Kaggle Tech 15k dataset."""
    tech_path = get_data_filepath("tech_hr_15k.csv", folder="raw")
    if not tech_path.exists():
        pytest.skip("tech_hr_15k.csv not available")

    df_tech = pd.read_csv(tech_path).sample(n=1000, random_state=42)  # Representative 1,000 row sample
    bundle = train_universal_workforce_bundle(
        df_tech,
        cohort_name="Tech 1k Slice",
        target_col="left",
    )

    assert len(bundle.scored_df) == 1000
    assert bundle.metadata["test_metrics"]["roc_auc"] >= 0.85
