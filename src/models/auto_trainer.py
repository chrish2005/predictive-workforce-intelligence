"""
Dynamic On-The-Fly Auto-Trainer and Explainer Engine.
Enables instant training, probability calibration, and SHAP explainability
for any arbitrary uploaded or benchmark HR dataset in 2-3 seconds.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import logging

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.preprocessing.universal import clean_and_encode_universal, UniversalHRPreprocessor
from src.models.evaluate import evaluate_attrition_model
from src.explainability.explainer import HRExplainer

logger = logging.getLogger(__name__)


@dataclass
class WorkforceCohortBundle:
    """Encapsulates all models, explainers, and scored data for an active workforce cohort."""
    cohort_name: str
    target_col: str
    raw_df: pd.DataFrame
    scored_df: pd.DataFrame
    calibrated_model: Any
    tree_model: Any
    explainer: HRExplainer
    preprocessor: UniversalHRPreprocessor
    X_trans: np.ndarray
    metadata: Dict[str, Any]


def train_universal_workforce_bundle(
    raw_df: pd.DataFrame,
    cohort_name: str = "Custom Uploaded Cohort",
    target_col: Optional[str] = None,
    id_col: Optional[str] = None,
) -> WorkforceCohortBundle:
    """
    Cleans, preprocesses, trains, calibrates, explains, and scores an arbitrary HR dataset on the fly.
    """
    logger.info(f"Initiating dynamic universal training for: {cohort_name} ({len(raw_df)} records)")

    # 1. Clean & Encode
    X_clean, y, emp_ids, detected_target = clean_and_encode_universal(raw_df, target_col=target_col, id_col=id_col)

    # 2. Stratified Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Fit Universal Preprocessor strictly on Train Split
    preprocessor = UniversalHRPreprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    # 4. Class balance weight
    pos_count = max(1, int(np.sum(y_train == 1)))
    neg_count = max(1, int(np.sum(y_train == 0)))
    pos_scale = float(neg_count / pos_count)

    # 5. Train Tree Ensemble for SHAP TreeExplainer
    # Use LightGBM for speed and high accuracy across any arbitrary dimension
    tree_model = LGBMClassifier(
        n_estimators=160,
        max_depth=4,
        num_leaves=12,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.80,
        scale_pos_weight=min(pos_scale, 10.0),
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    tree_model.fit(X_train_trans, np.array(y_train))

    # Also train regularized linear model
    lr_model = LogisticRegression(C=0.08, class_weight="balanced", max_iter=2000, random_state=42)
    lr_model.fit(X_train_trans, np.array(y_train))

    # Evaluate candidates on test set to pick champion
    lr_metrics = evaluate_attrition_model(lr_model, X_test_trans, np.array(y_test))
    tree_metrics = evaluate_attrition_model(tree_model, X_test_trans, np.array(y_test))

    champion_raw = lr_model if lr_metrics["roc_auc"] >= tree_metrics["roc_auc"] else tree_model
    champion_name = "Logistic_Regression" if champion_raw is lr_model else "LightGBM_Tree"

    # 6. Fit Platt Probability Calibration
    calibrated_model = CalibratedClassifierCV(estimator=champion_raw, method="sigmoid", cv=5)
    calibrated_model.fit(X_train_trans, np.array(y_train))

    # Test set metrics on calibrated model
    test_metrics = evaluate_attrition_model(calibrated_model, X_test_trans, np.array(y_test))

    # 7. Initialize SHAP Explainer on Tree Model
    explainer = HRExplainer(tree_model, feature_names)

    # 8. Score Full Dataset
    X_full_trans = preprocessor.transform(X_clean)
    probs = calibrated_model.predict_proba(X_full_trans)[:, 1]

    scored = raw_df.copy()
    scored["EmployeeID"] = emp_ids.values
    scored["turnover_risk"] = probs
    scored["Risk Score (%)"] = (probs * 100.0).round(1)
    scored["RiskTier"] = np.where(probs >= 0.70, "High", np.where(probs >= 0.40, "Medium", "Low"))

    # Estimate Financial Exposure (if salary column exists or proxy)
    salary_col = None
    for c in scored.columns:
        if "income" in c.lower() or "salary" in c.lower() or "rate" in c.lower():
            if pd.api.types.is_numeric_dtype(scored[c]):
                salary_col = c
                break

    if salary_col:
        ann_salary = scored[salary_col] * (12.0 if scored[salary_col].mean() < 25000 else 1.0)
    else:
        ann_salary = 60000.0

    scored["expected_financial_exposure"] = (probs * 1.5 * ann_salary).round(2)

    metadata = {
        "cohort_name": cohort_name,
        "target_col": detected_target,
        "champion_name": champion_name,
        "test_metrics": {k: v for k, v in test_metrics.items() if k not in ["y_prob", "y_pred"]},
        "lr_test_metrics": {k: v for k, v in lr_metrics.items() if k not in ["y_prob", "y_pred"]},
        "tree_test_metrics": {k: v for k, v in tree_metrics.items() if k not in ["y_prob", "y_pred"]},
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "n_samples": len(raw_df),
        "turnover_rate": float(y.mean()),
    }

    logger.info(f"Universal bundle for '{cohort_name}' ready! Test ROC-AUC={test_metrics['roc_auc']}, Brier={test_metrics['brier_score']}")

    return WorkforceCohortBundle(
        cohort_name=cohort_name,
        target_col=detected_target,
        raw_df=raw_df,
        scored_df=scored,
        calibrated_model=calibrated_model,
        tree_model=tree_model,
        explainer=explainer,
        preprocessor=preprocessor,
        X_trans=X_full_trans,
        metadata=metadata,
    )
