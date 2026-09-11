"""
Unit tests for SHAP Explainability Engine and additivity property.
"""

import pytest
import numpy as np
from src.data.loader import load_ibm_dataset
from src.preprocessing.pipeline import clean_hr_dataset
from src.models.train import load_model_artifacts
from src.explainability.explainer import HRExplainer, get_global_feature_importance, get_employee_waterfall_data


@pytest.fixture(scope="module")
def explainer_and_data():
    _, _, xgb_model, preprocessor, metadata = load_model_artifacts()
    df = load_ibm_dataset()
    clean_df, _, _ = clean_hr_dataset(df)
    X_trans = preprocessor.transform(clean_df)

    explainer = HRExplainer(xgb_model, metadata["feature_names"])
    return explainer, xgb_model, X_trans


def test_shap_additivity_property(explainer_and_data):
    """
    Verify the fundamental SHAP efficiency axiom (additivity):
    The sum of SHAP feature attributions plus the expected base value
    must equal the model's uncalibrated margin / log-odds output.
    """
    explainer, model, X_trans = explainer_and_data

    # Test across sample cohort
    sample_X = X_trans[:20]
    shap_vals = explainer.explain(sample_X)
    base_val = explainer.get_base_value()

    # Raw model output in margin space
    if hasattr(model, "predict"):
        raw_margin = model.predict(sample_X, output_margin=True)
    else:
        raw_margin = model.decision_function(sample_X)

    for i in range(len(sample_X)):
        calculated_sum = float(base_val + np.sum(shap_vals[i]))
        expected_output = float(raw_margin[i])
        np.testing.assert_allclose(
            calculated_sum,
            expected_output,
            atol=1e-3,
            err_msg=f"SHAP additivity violated for row {i}: sum={calculated_sum}, raw={expected_output}"
        )


def test_global_importance_structure(explainer_and_data):
    explainer, _, X_trans = explainer_and_data
    imp_df = get_global_feature_importance(explainer, X_trans[:100], top_n=10)

    assert len(imp_df) == 10
    assert "feature" in imp_df.columns
    assert "mean_abs_shap" in imp_df.columns
    assert "direction" in imp_df.columns
    assert imp_df["mean_abs_shap"].is_monotonic_decreasing


def test_employee_waterfall_structure(explainer_and_data):
    explainer, _, X_trans = explainer_and_data
    wf = get_employee_waterfall_data(explainer, X_trans[0], top_n=5)

    assert "base_value" in wf
    assert "top_drivers" in wf
    assert "other_impact" in wf
    assert "final_raw_margin" in wf
    assert len(wf["top_drivers"]) <= 5
