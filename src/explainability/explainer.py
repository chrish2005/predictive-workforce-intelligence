"""
Unified SHAP Explainability Engine for HR Attrition.
Calculates global feature attribution rankings, directional risk drivers,
and personalized employee waterfall charts satisfying SHAP additivity.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import shap
import logging

logger = logging.getLogger(__name__)


class HRExplainer:
    """
    Unified SHAP explainer providing both global cohort drivers
    and local per-employee root-cause waterfall attributions.
    """

    def __init__(self, model: Any, feature_names: List[str], background_data: Optional[np.ndarray] = None):
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        """Initialize appropriate SHAP explainer based on model architecture."""
        try:
            # Check if tree-based (XGBoost, LightGBM, Random Forest)
            if hasattr(self.model, "get_booster") or hasattr(self.model, "estimators_") or hasattr(self.model, "booster_"):
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("Initialized shap.TreeExplainer.")
            elif hasattr(self.model, "coef_"):
                # Linear/Logistic model
                if self.background_data is not None:
                    # Use sample of background data for linear explainer
                    bg = self.background_data[:100] if len(self.background_data) > 100 else self.background_data
                    self.explainer = shap.LinearExplainer(self.model, bg)
                else:
                    self.explainer = shap.LinearExplainer(self.model, np.zeros((1, len(self.feature_names))))
                logger.info("Initialized shap.LinearExplainer.")
            else:
                # Fallback to general Explainer
                bg = self.background_data[:50] if self.background_data is not None else np.zeros((1, len(self.feature_names)))
                self.explainer = shap.Explainer(self.model, bg)
                logger.info("Initialized general shap.Explainer.")
        except Exception as err:
            logger.warning(f"Default explainer init failed ({err}), falling back to KernelExplainer.")
            bg = self.background_data[:30] if self.background_data is not None else np.zeros((1, len(self.feature_names)))
            self.explainer = shap.KernelExplainer(self.model.predict_proba, bg)

    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for feature matrix X.
        Returns 2D array of shape (n_samples, n_features).
        """
        raw_shap = self.explainer.shap_values(X)

        # Handle various output shapes from different SHAP versions
        if isinstance(raw_shap, list):
            # Binary classification list: [class_0_shap, class_1_shap] -> take class 1
            return np.array(raw_shap[1])
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            # Shape (samples, features, classes) -> take class 1
            return raw_shap[:, :, 1]
        else:
            return np.array(raw_shap)

    def get_base_value(self) -> float:
        """Return expected base value (in margin / log-odds space)."""
        expected = self.explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            if len(expected) > 1:
                return float(expected[1])
            return float(expected[0])
        return float(expected)


def get_global_feature_importance(
    explainer: HRExplainer,
    X_samples: np.ndarray,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Computes global feature importance based on mean absolute SHAP values,
    including directional impact on attrition risk.
    """
    shap_vals = explainer.explain(X_samples)
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

    # Directionality: correlation between feature value and SHAP attribution
    directionality = []
    for i in range(X_samples.shape[1]):
        feat_col = X_samples[:, i]
        shap_col = shap_vals[:, i]
        std_feat = np.std(feat_col)
        std_shap = np.std(shap_col)
        if std_feat > 1e-6 and std_shap > 1e-6:
            corr = float(np.corrcoef(feat_col, shap_col)[0, 1])
            direction = "Increases Risk" if corr > 0.15 else ("Reduces Risk" if corr < -0.15 else "Mixed/Contextual")
        else:
            direction = "Neutral"
        directionality.append(direction)

    df_importance = pd.DataFrame({
        "feature": explainer.feature_names,
        "mean_abs_shap": mean_abs_shap,
        "direction": directionality,
    }).sort_values(by="mean_abs_shap", ascending=False)

    return df_importance.head(top_n).reset_index(drop=True)


def get_employee_waterfall_data(
    explainer: HRExplainer,
    employee_vector: np.ndarray,
    top_n: int = 8,
) -> Dict[str, Any]:
    """
    Generates structured waterfall decomposition data for an individual employee.
    Separates factors pushing risk upward vs downward relative to the company baseline.

    Args:
        explainer: HRExplainer instance.
        employee_vector: 1D array of length n_features for single employee.
        top_n: Max number of individual drivers to display before grouping remainder into 'Other'.

    Returns:
        Dict with keys: base_value, top_drivers, other_impact, final_value, raw_features.
    """
    vec_2d = employee_vector.reshape(1, -1)
    shap_row = explainer.explain(vec_2d)[0]
    base_val = explainer.get_base_value()

    indices = np.argsort(np.abs(shap_row))[::-1]
    top_indices = indices[:top_n]
    other_indices = indices[top_n:]

    top_drivers = []
    for idx in top_indices:
        top_drivers.append({
            "feature": explainer.feature_names[idx],
            "attribution": round(float(shap_row[idx]), 4),
            "feature_val": round(float(employee_vector[idx]), 4),
            "effect": "Risk Increasing" if shap_row[idx] > 0 else "Risk Decreasing",
        })

    other_impact = float(np.sum(shap_row[other_indices])) if len(other_indices) > 0 else 0.0
    final_output = base_val + np.sum(shap_row)

    return {
        "base_value": round(base_val, 4),
        "top_drivers": top_drivers,
        "other_impact": round(other_impact, 4),
        "final_raw_margin": round(float(final_output), 4),
        "total_shap_sum": round(float(np.sum(shap_row)), 4),
    }
