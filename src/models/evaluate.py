"""
Model Evaluation and Cost-Sensitive Metrics for Attrition Prediction.
Implements ROC-AUC, PR-AUC, Precision@K (simulating HR triage capacity),
Brier score calibration error, and turnover replacement financial loss.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
)


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.10) -> float:
    """
    Compute Precision within the top K fraction of highest predicted risk employees.
    Simulates real-world HR capacity constraints (e.g. HRBP only has time to intervene
    on top 10% or 20% of at-risk staff).

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted turnover probabilities.
        k: Fraction of cohort to inspect (0.0 to 1.0).
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    k_count = max(1, int(np.ceil(k * n)))
    top_indices = np.argsort(y_prob)[::-1][:k_count]
    return float(np.mean(y_true[top_indices]))


def compute_cost_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = 75000.0,  # Cost of lost employee (~1.5x average annual salary)
    cost_fp: float = 1500.0,   # Cost of unwarranted HR review/perk
    cost_tp: float = 4000.0,   # Cost of targeted retention intervention for true flight risk
) -> Dict[str, float]:
    """
    Calculate business cost impact based on turnover replacement economics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    total_cost = (fn * cost_fn) + (fp * cost_fp) + (tp * cost_tp)
    baseline_unmanaged_cost = np.sum(y_true) * cost_fn
    net_savings = max(0.0, baseline_unmanaged_cost - total_cost)

    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_financial_loss": float(total_cost),
        "baseline_unmanaged_cost": float(baseline_unmanaged_cost),
        "net_retention_savings": float(net_savings),
    }


def evaluate_attrition_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.35,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of an attrition classification model.

    Returns:
        Dict of metrics: roc_auc, pr_auc, brier_score, precision_top10,
        precision_top20, f1, recall, precision, confusion_matrix, financial_loss.
    """
    # Obtain predicted probabilities for positive class
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df_vals = model.decision_function(X_test)
        y_prob = 1.0 / (1.0 + np.exp(-df_vals))
    else:
        y_prob = model.predict(X_test)

    y_test_arr = np.array(y_test).astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_test_arr, y_prob))
    pr_auc = float(average_precision_score(y_test_arr, y_prob))
    brier = float(brier_score_loss(y_test_arr, y_prob))
    p_top10 = precision_at_k(y_test_arr, y_prob, k=0.10)
    p_top20 = precision_at_k(y_test_arr, y_prob, k=0.20)

    f1 = float(f1_score(y_test_arr, y_pred, zero_division=0))
    recall = float(recall_score(y_test_arr, y_pred, zero_division=0))
    precision = float(precision_score(y_test_arr, y_pred, zero_division=0))

    financial = compute_cost_matrix(y_test_arr, y_pred)

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "precision_at_top10": round(p_top10, 4),
        "precision_at_top20": round(p_top20, 4),
        "f1": round(f1, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "decision_threshold": threshold,
        "financial_analysis": financial,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }
