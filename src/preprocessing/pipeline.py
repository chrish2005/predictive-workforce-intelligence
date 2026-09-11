"""
Preprocessing and Featurization Pipeline.
Enforces strict featurization ordering (train/test split strictly prior to fitting),
integrates domain feature engineering, and performs one-hot encoding and standard scaling.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.engineer import HRFeatureEngineer
from src.preprocessing.cleaner import clean_hr_dataset
import logging

logger = logging.getLogger(__name__)


def prepare_train_test_data(
    raw_df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    """
    Cleans raw dataset and splits into Train and Test splits STRATIFIED by Attrition.
    Guarantees no data leakage before fitting transformers.

    Returns:
        (X_train, X_test, y_train, y_test, ids_train, ids_test)
    """
    X_clean, y, emp_ids = clean_hr_dataset(raw_df)

    if y is None:
        raise ValueError("Target 'Attrition' column not found in dataset.")

    if emp_ids is not None:
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X_clean, y, emp_ids,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        ids_train, ids_test = None, None

    logger.info(
        f"Prepared splits: Train shape={X_train.shape}, Test shape={X_test.shape}. "
        f"Train turnover rate={y_train.mean():.3f}, Test turnover rate={y_test.mean():.3f}"
    )
    return X_train, X_test, y_train, y_test, ids_train, ids_test


class HRPreprocessor:
    """
    Complete Preprocessor integrating domain feature engineering,
    categorical one-hot encoding, and numerical scaling.
    Maintains clean feature naming for model explainability.
    """

    def __init__(self):
        self.engineer = HRFeatureEngineer()
        self.column_transformer: Optional[ColumnTransformer] = None
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        # 1. Fit & transform domain feature engineer
        X_eng = self.engineer.fit_transform(X)

        # 2. Identify numerical and categorical columns
        self.numeric_cols_ = [
            c for c in X_eng.columns
            if pd.api.types.is_numeric_dtype(X_eng[c])
        ]
        self.categorical_cols_ = [
            c for c in X_eng.columns
            if c not in self.numeric_cols_
        ]

        # 3. Build ColumnTransformer
        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_cols_),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_cols_),
            ],
            remainder="drop"
        )

        self.column_transformer.fit(X_eng)

        # 4. Extract feature names
        cat_encoder = self.column_transformer.named_transformers_["cat"]
        encoded_cat_names = list(cat_encoder.get_feature_names_out(self.categorical_cols_))
        self.feature_names_out_ = self.numeric_cols_ + encoded_cat_names

        logger.info(
            f"Fitted HRPreprocessor: {len(self.numeric_cols_)} numerical + "
            f"{len(encoded_cat_names)} one-hot encoded = {len(self.feature_names_out_)} total features."
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.column_transformer is None:
            raise RuntimeError("HRPreprocessor is not fitted yet.")
        X_eng = self.engineer.transform(X)
        return self.column_transformer.transform(X_eng)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        return self.feature_names_out_


def build_preprocessing_pipeline() -> HRPreprocessor:
    """Factory function returning a new un-fitted HRPreprocessor instance."""
    return HRPreprocessor()
