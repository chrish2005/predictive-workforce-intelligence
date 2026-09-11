"""
Unit tests for Prescriptive Retention Simulator and counterfactual engine.
"""

import pytest
from src.data.loader import load_ibm_dataset
from src.models.train import load_model_artifacts
from src.explainability.simulator import simulate_retention_action, calculate_retention_roi


@pytest.fixture(scope="module")
def models_and_preprocessor():
    calibrated_model, raw_model, xgb_model, preprocessor, metadata = load_model_artifacts()
    return calibrated_model, preprocessor


def test_retention_action_reduces_flight_risk(models_and_preprocessor):
    calibrated_model, preprocessor = models_and_preprocessor
    df = load_ibm_dataset()

    # Select an employee with overtime and moderate income
    ot_employees = df[df["OverTime"] == "Yes"]
    emp = ot_employees.iloc[0].to_dict()

    interventions = {
        "salary_hike_pct": 15.0,
        "eliminate_overtime": True,
        "work_life_balance": 4,
        "stock_option_level": 2,
    }

    sim = simulate_retention_action(emp, interventions, calibrated_model, preprocessor)

    # Simulated risk should be strictly less than baseline risk
    assert sim["simulated_risk"] < sim["baseline_risk"], (
        f"Simulated risk {sim['simulated_risk']} should be lower than baseline {sim['baseline_risk']}"
    )
    assert sim["risk_delta"] < 0, "Risk delta should be negative (reduction in turnover risk)"
    assert sim["risk_reduction_pct"] > 0, "Risk reduction percentage should be positive"


def test_calculate_retention_roi_math():
    roi = calculate_retention_roi(
        monthly_income=6000.0,
        baseline_risk=0.75,
        simulated_risk=0.25,
        salary_hike_pct=10.0,
        eliminate_overtime=True,
    )

    annual_salary = 6000.0 * 12.0  # 72,000
    expected_replacement = 1.5 * annual_salary  # 108,000
    expected_saved = (0.75 - 0.25) * expected_replacement  # 54,000

    salary_cost = annual_salary * 0.10  # 7,200
    overtime_cost = 2500.0
    total_cost = salary_cost + overtime_cost  # 9,700

    assert roi["expected_replacement_saved"] == pytest.approx(expected_saved, rel=1e-2)
    assert roi["total_intervention_cost"] == pytest.approx(total_cost, rel=1e-2)
    assert roi["net_financial_savings"] == pytest.approx(expected_saved - total_cost, rel=1e-2)
    assert roi["net_financial_savings"] > 0
