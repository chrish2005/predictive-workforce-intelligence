"""
Model Training, Cross-Validation, and Artifact Persistence.
Trains 4 model families (Logistic Regression, Random Forest, XGBoost, LightGBM),
performs 5-Fold Stratified Cross-Validation, selects the optimal architecture,
fits Platt probability calibration, and persists production artifacts.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.data.loader import load_ibm_dataset, get_project_root
from src.preprocessing.pipeline import prepare_train_test_data, build_preprocessing_pipeline, HRPreprocessor
from src.models.evaluate import evaluate_attrition_model, precision_at_k
from src.models.calibrate import calibrate_model

logger = logging.getLogger(__name__)


def get_candidate_models(pos_scale_weight: float = 5.0) -> Dict[str, Any]:
    """
    Returns initialized candidate models with tuned class-imbalance weights.
    """
    return {
        "Logistic_Regression_Penalized": LogisticRegression(
            C=0.04,
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.75,
            scale_pos_weight=pos_scale_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=180,
            max_depth=3,
            num_leaves=8,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.75,
            scale_pos_weight=pos_scale_weight,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        ),
    }


def cross_validate_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Executes 5-Fold Stratified Cross-Validation across candidate models.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_records = []

    for name, model in models.items():
        roc_aucs, pr_aucs, briers, p_top10s = [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model.fit(X_tr, y_tr)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_val)[:, 1]
            else:
                probs = model.decision_function(X_val)
                probs = 1.0 / (1.0 + np.exp(-probs))

            roc_aucs.append(roc_auc_score(y_val, probs))
            pr_aucs.append(average_precision_score(y_val, probs))
            briers.append(brier_score_loss(y_val, probs))
            p_top10s.append(precision_at_k(y_val, probs, k=0.10))

        cv_records.append({
            "model_name": name,
            "cv_roc_auc_mean": round(float(np.mean(roc_aucs)), 4),
            "cv_roc_auc_std": round(float(np.std(roc_aucs)), 4),
            "cv_pr_auc_mean": round(float(np.mean(pr_aucs)), 4),
            "cv_pr_auc_std": round(float(np.std(pr_aucs)), 4),
            "cv_brier_mean": round(float(np.mean(briers)), 4),
            "cv_p_top10_mean": round(float(np.mean(p_top10s)), 4),
        })
        logger.info(f"CV for {name}: ROC-AUC={np.mean(roc_aucs):.4f}, PR-AUC={np.mean(pr_aucs):.4f}")

    return pd.DataFrame(cv_records).sort_values(by="cv_pr_auc_mean", ascending=False)


def save_model_artifacts(
    calibrated_model: Any,
    raw_model: Any,
    xgb_model: Any,
    preprocessor: HRPreprocessor,
    metadata: Dict[str, Any],
    artifacts_dir: Optional[Path] = None,
):
    """
    Saves trained models, preprocessor, and metadata JSON to disk.
    """
    if artifacts_dir is None:
        artifacts_dir = get_project_root() / "models"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(calibrated_model, artifacts_dir / "best_model.joblib")
    joblib.dump(raw_model, artifacts_dir / "raw_model.joblib")
    joblib.dump(xgb_model, artifacts_dir / "xgb_model.joblib")
    joblib.dump(preprocessor, artifacts_dir / "preprocessor.joblib")

    meta_path = artifacts_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model artifacts successfully written to: {artifacts_dir}")


def load_model_artifacts(artifacts_dir: Optional[Path] = None) -> Tuple[Any, Any, Any, HRPreprocessor, Dict[str, Any]]:
    """
    Loads persisted production artifacts.
    """
    if artifacts_dir is None:
        artifacts_dir = get_project_root() / "models"

    calibrated_model = joblib.load(artifacts_dir / "best_model.joblib")
    raw_model = joblib.load(artifacts_dir / "raw_model.joblib")
    xgb_model = joblib.load(artifacts_dir / "xgb_model.joblib")
    preprocessor = joblib.load(artifacts_dir / "preprocessor.joblib")

    with open(artifacts_dir / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return calibrated_model, raw_model, xgb_model, preprocessor, metadata


def train_and_evaluate_all_models(save_artifacts: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    End-to-end training and evaluation pipeline:
    1. Loads dataset and splits train/test stratified
    2. Fits HRPreprocessor on train
    3. Runs 5-Fold Stratified CV on candidate models
    4. Selects top model, fits on train and calibrates
    5. Evaluates on test set
    6. Persists artifacts
    """
    raw_df = load_ibm_dataset()
    X_train_df, X_test_df, y_train, y_test, ids_train, ids_test = prepare_train_test_data(raw_df)

    preprocessor = build_preprocessing_pipeline()
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)
    feature_names = preprocessor.get_feature_names_out()

    # Calculate class balance weight (negative / positive ratio)
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    pos_scale = float(neg_count / max(1, pos_count))
    logger.info(f"Class imbalance scale_pos_weight: {pos_scale:.2f}")

    # Cross-validation
    candidates = get_candidate_models(pos_scale_weight=pos_scale)
    cv_summary = cross_validate_models(candidates, X_train, np.array(y_train))

    best_model_name = cv_summary.iloc[0]["model_name"]
    logger.info(f"Selected champion architecture based on CV PR-AUC: {best_model_name}")

    # Re-instantiate raw best model and fit on full training set
    raw_model = get_candidate_models(pos_scale_weight=pos_scale)[best_model_name]
    raw_model.fit(X_train, np.array(y_train))

    # Also fit champion XGBoost model on full training set for tree explainability
    xgb_model = get_candidate_models(pos_scale_weight=pos_scale)["XGBoost"]
    xgb_model.fit(X_train, np.array(y_train))

    # Fit Platt calibrated model on training set
    calibrated_model = calibrate_model(raw_model, X_train, np.array(y_train), method="sigmoid", cv=5)

    # Evaluate calibrated model on held-out test set
    test_metrics = evaluate_attrition_model(calibrated_model, X_test, np.array(y_test))

    # Evaluate raw model on held-out test set
    raw_test_metrics = evaluate_attrition_model(raw_model, X_test, np.array(y_test))

    logger.info("=== HELD-OUT TEST PERFORMANCE (Calibrated Model) ===")
    logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']}")
    logger.info(f"Test PR-AUC:  {test_metrics['pr_auc']}")
    logger.info(f"Test Brier:   {test_metrics['brier_score']}")
    logger.info(f"Test Precision@Top10%: {test_metrics['precision_at_top10']}")
    logger.info(f"Test Precision@Top20%: {test_metrics['precision_at_top20']}")

    # Prepare metadata
    metadata = {
        "best_model_name": best_model_name,
        "class_imbalance_weight": pos_scale,
        "cv_summary": cv_summary.to_dict(orient="records"),
        "test_metrics": {k: v for k, v in test_metrics.items() if k not in ["y_prob", "y_pred"]},
        "raw_test_metrics": {k: v for k, v in raw_test_metrics.items() if k not in ["y_prob", "y_pred"]},
        "feature_names": feature_names,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_features": len(feature_names),
    }

    if save_artifacts:
        save_model_artifacts(calibrated_model, raw_model, xgb_model, preprocessor, metadata)

    return cv_summary, metadata


if __name__ == "__main__":
    cv_df, meta = train_and_evaluate_all_models()
    print("\n--- Cross-Validation Leaderboard ---")
    print(cv_df.to_string(index=False))
    print("\n--- Test Set Metrics ---")
    print(json.dumps(meta["test_metrics"], indent=2))
