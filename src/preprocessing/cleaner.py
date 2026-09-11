"""
Data cleaner and schema validator for HR analytics data.
Removes zero-variance and administrative ID columns, validates types,
and encodes the target attrition variable.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

ZERO_VARIANCE_COLS = ["StandardHours", "Over18", "EmployeeCount"]
IDENTIFIER_COLS = ["EmployeeNumber"]


def clean_hr_dataset(
    df: pd.DataFrame,
    target_col: str = "Attrition",
    drop_identifiers: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """
    Clean the HR dataset:
    - Drop zero-variance columns (StandardHours, Over18, EmployeeCount)
    - Optionally extract/drop employee identifier column (EmployeeNumber)
    - Encode binary target Attrition ('Yes' -> 1, 'No' -> 0)
    - Handle missing values if any present via median/mode imputation

    Args:
        df: Input raw DataFrame.
        target_col: Target column name.
        drop_identifiers: Whether to drop identifier columns from feature matrix.

    Returns:
        (X, y, employee_ids): Cleaned features DataFrame, binary target Series (or None),
                              and employee IDs Series (or None).
    """
    data = df.copy()

    # Extract employee identifier if present
    employee_ids = None
    if "EmployeeNumber" in data.columns:
        employee_ids = data["EmployeeNumber"].copy()
        if drop_identifiers:
            data = data.drop(columns=["EmployeeNumber"])

    # Drop zero-variance columns
    cols_to_drop = [c for c in ZERO_VARIANCE_COLS if c in data.columns]
    if cols_to_drop:
        logger.info(f"Dropping zero-variance columns: {cols_to_drop}")
        data = data.drop(columns=cols_to_drop)

    # Separate target variable if present
    y = None
    if target_col in data.columns:
        target_series = data[target_col]
        if target_series.dtype == object or isinstance(target_series.iloc[0], str):
            y = target_series.map({"Yes": 1, "No": 0, "1": 1, "0": 0}).astype(int)
        else:
            y = target_series.astype(int)
        data = data.drop(columns=[target_col])

    # Check for missing values and impute if necessary
    null_counts = data.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        logger.warning(f"Detected missing values in columns: {null_cols.to_dict()}")
        for col in null_cols.index:
            if np.issubdtype(data[col].dtype, np.number):
                median_val = data[col].median()
                data[col] = data[col].fillna(median_val)
                logger.info(f"Imputed numerical column {col} with median {median_val}")
            else:
                mode_val = data[col].mode()[0]
                data[col] = data[col].fillna(mode_val)
                logger.info(f"Imputed categorical column {col} with mode '{mode_val}'")

    return data, y, employee_ids
