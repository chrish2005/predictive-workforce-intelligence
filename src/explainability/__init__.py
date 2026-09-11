"""Explainable AI (XAI) and prescriptive retention simulation package."""
from .explainer import HRExplainer, get_global_feature_importance, get_employee_waterfall_data
from .simulator import simulate_retention_action, calculate_retention_roi

__all__ = [
    "HRExplainer",
    "get_global_feature_importance",
    "get_employee_waterfall_data",
    "simulate_retention_action",
    "calculate_retention_roi",
]
