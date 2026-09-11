"""
Helper utilities for dynamic schema adaptation across heterogeneous HR datasets.
"""

from typing import List, Optional, Any
import pandas as pd


def get_column_matching(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Finds the first column in df that matches any of the candidate keywords (case-insensitive)."""
    cols_lower = {col.lower().strip().replace(" ", "_"): col for col in df.columns}
    for cand in candidates:
        cand_clean = cand.lower().strip().replace(" ", "_")
        if cand_clean in cols_lower:
            return cols_lower[cand_clean]
        for c_clean, orig in cols_lower.items():
            if cand_clean in c_clean:
                return orig
    return None


def get_employee_display_role(row: pd.Series) -> str:
    """Returns a displayable title/role for an employee row across different datasets."""
    role_col = get_column_matching(row.to_frame().T, ["jobrole", "role", "title", "position", "sales", "department", "dept"])
    if role_col and pd.notna(row.get(role_col)):
        return str(row[role_col])
    return "Staff Member"


def get_employee_display_department(row: pd.Series) -> str:
    """Returns a displayable department for an employee row across different datasets."""
    dept_col = get_column_matching(row.to_frame().T, ["department", "dept", "division", "sales"])
    if dept_col and pd.notna(row.get(dept_col)):
        return str(row[dept_col])
    return "General Operations"


def get_employee_compensation(row: pd.Series) -> float:
    """Extracts annual compensation or proxy across different datasets."""
    comp_col = get_column_matching(row.to_frame().T, ["monthlyincome", "income", "salary", "compensation", "rate"])
    if comp_col and pd.notna(row.get(comp_col)):
        val = row[comp_col]
        try:
            val = float(val)
            return val * 12.0 if val < 25000 else val
        except (ValueError, TypeError):
            # Categorical salary (e.g. 'low', 'medium', 'high' in Tech 15k)
            str_val = str(val).lower()
            if "high" in str_val:
                return 110000.0
            elif "medium" in str_val:
                return 65000.0
            else:
                return 40000.0
    return 60000.0
