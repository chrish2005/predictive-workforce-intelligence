"""
Universal Auto-Preprocessor for Arbitrary & Uploaded HR Datasets.
Automatically detects target turnover columns, separates employee identifiers,
handles missing values, and builds zero-leakage scikit-learn transformation pipelines.
"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import logging

logger = logging.getLogger(__name__)

POTENTIAL_TARGET_NAMES = [
    "attrition", "left", "churn", "turnover", "is_leaving", "status", "exit", "resigned", "target"
]
POTENTIAL_ID_NAMES = [
    "id", "empid", "emp_id", "employeeid", "employee_id", "employeenumber", "employee_number"
]


def auto_detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detects the most probable binary turnover/attrition column."""
    cols_lower = {col.lower().strip().replace(" ", "_"): col for col in df.columns}

    # 1. Exact match against candidate keywords
    for candidate in POTENTIAL_TARGET_NAMES:
        if candidate in cols_lower:
            return cols_lower[candidate]

    # 2. Fuzzy match on columns containing keywords
    for cand in POTENTIAL_TARGET_NAMES:
        for clean_name, orig_name in cols_lower.items():
            if cand in clean_name and df[orig_name].nunique() == 2:
                return orig_name

    # 3. Fallback: inspect any binary column
    for col in df.columns:
        if df[col].nunique() == 2:
            val_set = set(df[col].dropna().astype(str).str.lower().unique())
            if val_set.issubset({"yes", "no", "1", "0", "true", "false", "left", "stayed"}):
                return col

    return None


def auto_detect_id_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detects employee identifier column."""
    cols_lower = {col.lower().strip().replace(" ", "_"): col for col in df.columns}
    for candidate in POTENTIAL_ID_NAMES:
        if candidate in cols_lower:
            return cols_lower[candidate]

    # Check for near-unique string/int column
    n = len(df)
    for col in df.columns:
        if df[col].nunique() >= n * 0.98 and col.lower() not in POTENTIAL_TARGET_NAMES:
            return col

    return None


def clean_and_encode_universal(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    id_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.Series, str]:
    """
    Cleans an arbitrary HR DataFrame, extracts target and IDs.

    Returns:
        (X_features, y_target, employee_ids, target_col_name)
    """
    data = df.copy()

    # Determine target column
    if target_col is None or target_col not in data.columns:
        target_col = auto_detect_target_column(data)

    if target_col is None:
        raise ValueError("Could not automatically detect the target turnover/attrition column. Please specify target_col.")

    # Determine ID column
    if id_col is None or id_col not in data.columns:
        id_col = auto_detect_id_column(data)

    # Extract ID Series
    if id_col and id_col in data.columns:
        emp_ids = data[id_col].astype(str)
        data = data.drop(columns=[id_col])
    else:
        emp_ids = pd.Series([f"EMP-{i+1:05d}" for i in range(len(data))], index=data.index)

    # Encode target to binary integer
    target_raw = data[target_col].copy()
    data = data.drop(columns=[target_col])

    if target_raw.dtype == object or isinstance(target_raw.iloc[0], str):
        mapping = {
            "yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0,
            "left": 1, "stayed": 0, "churn": 1, "retained": 0, "y": 1, "n": 0
        }
        y = target_raw.astype(str).str.lower().str.strip().map(mapping).fillna(0).astype(int)
    else:
        y = target_raw.astype(int)

    # Drop zero variance columns
    nunique = data.nunique()
    zero_var_cols = nunique[nunique <= 1].index.tolist()
    if zero_var_cols:
        data = data.drop(columns=zero_var_cols)

    # Handle missing values
    for col in data.columns:
        if data[col].isnull().any():
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = data[col].fillna(data[col].mode()[0])

    return data, y, emp_ids, target_col


class UniversalHRPreprocessor:
    """
    Scikit-learn pipeline for arbitrary HR datasets.
    Scales numerical features and one-hot encodes categorical features with zero leakage.
    """

    def __init__(self):
        self.column_transformer: Optional[ColumnTransformer] = None
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.medians_: Dict[str, float] = {}
        self.modes_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y=None):
        self.numeric_cols_ = [
            c for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c])
        ]
        self.categorical_cols_ = [
            c for c in X.columns
            if c not in self.numeric_cols_
        ]

        # Store medians and modes for inference safety
        for c in self.numeric_cols_:
            s = X[c].dropna()
            self.medians_[c] = float(s.median()) if not s.empty else 0.0

        for c in self.categorical_cols_:
            s = X[c].dropna()
            self.modes_[c] = s.mode().iloc[0] if not s.empty else "Unknown"

        transformers = []
        if self.numeric_cols_:
            transformers.append(("num", StandardScaler(), self.numeric_cols_))
        if self.categorical_cols_:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_cols_))

        self.column_transformer = ColumnTransformer(transformers=transformers, remainder="drop")
        self.column_transformer.fit(X)

        feature_names = []
        if self.numeric_cols_:
            feature_names.extend(self.numeric_cols_)
        if self.categorical_cols_:
            cat_enc = self.column_transformer.named_transformers_["cat"]
            feature_names.extend(list(cat_enc.get_feature_names_out(self.categorical_cols_)))

        self.feature_names_out_ = feature_names
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.column_transformer is None:
            raise RuntimeError("UniversalHRPreprocessor is not fitted yet.")

        X_copy = X.copy()
        for c in self.numeric_cols_:
            if c not in X_copy.columns:
                X_copy[c] = self.medians_.get(c, 0.0)
            else:
                X_copy[c] = X_copy[c].fillna(self.medians_.get(c, 0.0))

        for c in self.categorical_cols_:
            if c not in X_copy.columns:
                X_copy[c] = self.modes_.get(c, "Unknown")
            else:
                X_copy[c] = X_copy[c].fillna(self.modes_.get(c, "Unknown"))

        return self.column_transformer.transform(X_copy)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        return self.feature_names_out_
