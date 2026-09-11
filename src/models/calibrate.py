"""
Probability Calibration for Attrition Prediction.
Ensures that predicted turnover risk probabilities strictly reflect empirical likelihoods,
essential for executive credibility and risk tier thresholds.
"""

from typing import Any, Tuple
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import logging

logger = logging.getLogger(__name__)


def calibrate_model(
    base_estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "sigmoid",
    cv: int = 5,
) -> CalibratedClassifierCV:
    """
    Fits a cross-validated probability calibration wrapper around the base estimator.

    Args:
        base_estimator: Candidate classifier (e.g. XGBoost, LightGBM, Random Forest).
        X_train: Training feature matrix.
        y_train: Training labels.
        method: 'sigmoid' (Platt scaling) or 'isotonic'.
        cv: Cross-validation fold count for calibration.

    Returns:
        Fitted CalibratedClassifierCV object.
    """
    logger.info(f"Fitting CalibratedClassifierCV using method='{method}' with {cv}-fold CV...")
    calibrator = CalibratedClassifierCV(
        estimator=base_estimator,
        method=method,
        cv=cv
    )
    calibrator.fit(X_train, y_train)
    logger.info("Probability calibration complete.")
    return calibrator


def compute_calibration_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates empirical fraction of positives and mean predicted probabilities
    for reliability plots.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return prob_true, prob_pred
