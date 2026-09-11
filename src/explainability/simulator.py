"""
Prescriptive Retention Simulator and Counterfactual Decision Engine.
Allows HR leaders to simulate interventions (salary hikes, overtime removal,
promotion, stock grant, work-life balance) and evaluate real-time risk reduction and financial ROI.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from src.preprocessing.pipeline import HRPreprocessor
import logging

logger = logging.getLogger(__name__)


def calculate_retention_roi(
    monthly_income: float,
    baseline_risk: float,
    simulated_risk: float,
    salary_hike_pct: float = 0.0,
    eliminate_overtime: bool = False,
    promote_role: bool = False,
    stock_boost: int = 0,
    wlb_boost: bool = False,
) -> Dict[str, float]:
    """
    Computes business financial return on investment (ROI) for retention actions.
    SHRM benchmark: Turnover replacement cost is ~1.5x annual base salary.
    """
    annual_salary = float(monthly_income) * 12.0
    replacement_cost = 1.5 * annual_salary

    risk_delta = max(0.0, float(baseline_risk - simulated_risk))
    expected_replacement_saved = risk_delta * replacement_cost

    # Intervention Direct Costs
    salary_increase_cost = annual_salary * (max(0.0, salary_hike_pct) / 100.0)
    promotion_cost = 7500.0 if promote_role else 0.0
    stock_cost = max(0, stock_boost) * 4000.0
    wlb_cost = 1200.0 if wlb_boost else 0.0
    overtime_reallocation_cost = 2500.0 if eliminate_overtime else 0.0

    total_intervention_cost = (
        salary_increase_cost
        + promotion_cost
        + stock_cost
        + wlb_cost
        + overtime_reallocation_cost
    )

    net_savings = expected_replacement_saved - total_intervention_cost
    roi_pct = (net_savings / total_intervention_cost * 100.0) if total_intervention_cost > 0 else 0.0

    return {
        "annual_salary": round(annual_salary, 2),
        "turnover_replacement_cost": round(replacement_cost, 2),
        "expected_replacement_saved": round(expected_replacement_saved, 2),
        "total_intervention_cost": round(total_intervention_cost, 2),
        "net_financial_savings": round(net_savings, 2),
        "roi_percentage": round(roi_pct, 1),
    }


def simulate_retention_action(
    employee_data: Dict[str, Any],
    interventions: Dict[str, Any],
    model: Any,
    preprocessor: HRPreprocessor,
) -> Dict[str, Any]:
    """
    Simulates counterfactual retention scenarios for a single employee record.

    Supported interventions:
    - salary_hike_pct (float): e.g. 10.0 for 10% increase
    - eliminate_overtime (bool): True to set OverTime to 'No'
    - work_life_balance (int): New rating 1 to 4
    - job_satisfaction (int): New rating 1 to 4
    - stock_option_level (int): New level 0 to 3
    - promote_role (bool): True to increase JobLevel by 1 and reset YearsSinceLastPromotion to 0

    Returns:
        Dict with baseline_risk, simulated_risk, risk_delta, roi_metrics, and updated_features.
    """
    baseline_df = pd.DataFrame([employee_data])
    modified_df = baseline_df.copy()

    # Apply Interventions
    salary_pct = float(interventions.get("salary_hike_pct", 0.0))
    if salary_pct > 0:
        orig_inc = float(modified_df["MonthlyIncome"].iloc[0])
        modified_df["MonthlyIncome"] = orig_inc * (1.0 + salary_pct / 100.0)
        orig_hike = float(modified_df["PercentSalaryHike"].iloc[0]) if "PercentSalaryHike" in modified_df.columns else 12.0
        modified_df["PercentSalaryHike"] = orig_hike + (salary_pct * 0.5)

    if interventions.get("eliminate_overtime", False):
        modified_df["OverTime"] = "No"

    if "work_life_balance" in interventions and interventions["work_life_balance"] is not None:
        modified_df["WorkLifeBalance"] = int(interventions["work_life_balance"])

    if "job_satisfaction" in interventions and interventions["job_satisfaction"] is not None:
        modified_df["JobSatisfaction"] = int(interventions["job_satisfaction"])

    if "stock_option_level" in interventions and interventions["stock_option_level"] is not None:
        modified_df["StockOptionLevel"] = int(interventions["stock_option_level"])

    promote = bool(interventions.get("promote_role", False))
    if promote:
        if "JobLevel" in modified_df.columns:
            modified_df["JobLevel"] = min(5, int(modified_df["JobLevel"].iloc[0]) + 1)
        if "YearsSinceLastPromotion" in modified_df.columns:
            modified_df["YearsSinceLastPromotion"] = 0

    # Transform through fitted preprocessor
    X_base = preprocessor.transform(baseline_df)
    X_mod = preprocessor.transform(modified_df)

    # Predict calibrated probability
    base_prob = float(model.predict_proba(X_base)[0, 1])
    sim_prob = float(model.predict_proba(X_mod)[0, 1])
    risk_delta = sim_prob - base_prob

    # Financial ROI
    monthly_inc = float(employee_data.get("MonthlyIncome", 5000.0))
    orig_stock = int(employee_data.get("StockOptionLevel", 0))
    new_stock = int(modified_df["StockOptionLevel"].iloc[0]) if "StockOptionLevel" in modified_df.columns else orig_stock
    stock_boost = max(0, new_stock - orig_stock)

    wlb_boost = (
        int(modified_df["WorkLifeBalance"].iloc[0]) > int(employee_data.get("WorkLifeBalance", 2))
        if "WorkLifeBalance" in modified_df.columns else False
    )

    roi = calculate_retention_roi(
        monthly_income=monthly_inc,
        baseline_risk=base_prob,
        simulated_risk=sim_prob,
        salary_hike_pct=salary_pct,
        eliminate_overtime=interventions.get("eliminate_overtime", False),
        promote_role=promote,
        stock_boost=stock_boost,
        wlb_boost=wlb_boost,
    )

    return {
        "baseline_risk": round(base_prob, 4),
        "simulated_risk": round(sim_prob, 4),
        "risk_delta": round(risk_delta, 4),
        "risk_reduction_pct": round(max(0.0, -risk_delta / max(0.001, base_prob) * 100.0), 1),
        "roi": roi,
        "modified_attributes": modified_df.iloc[0].to_dict(),
    }
