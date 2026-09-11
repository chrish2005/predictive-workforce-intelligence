"""
Domain-Specific HR Feature Engineering.
Implements composite indicators for burnout, stagnation, and compensation fairness
as a scikit-learn compatible transformer adhering strictly to zero-leakage standards.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HRFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms HR dataset features with composite domain indicators:
    - StagnationIndex: YearsSinceLastPromotion / (YearsAtCompany + 1)
    - RoleStagnationRatio: YearsInCurrentRole / (YearsAtCompany + 1)
    - ManagerStabilityRatio: YearsWithCurrManager / (YearsAtCompany + 1)
    - CompensationEquityIndex: MonthlyIncome / (Median MonthlyIncome for JobRole + JobLevel)
    - BurnoutRiskFactor: (OverTime == 'Yes') * (BusinessTravel == 'Travel_Frequently') * (WorkLifeBalance <= 2)
    - CommuteBurdenFactor: DistanceFromHome / (JobSatisfaction + 0.5)
    - TenureToAgeRatio: TotalWorkingYears / max(1, Age - 17)
    - IncomePerYearWorked: MonthlyIncome / max(1, TotalWorkingYears)
    """

    def __init__(self):
        self.role_level_medians_: Dict[Tuple[str, int], float] = {}
        self.role_medians_: Dict[str, float] = {}
        self.global_income_median_: float = 4919.0

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit benchmark medians strictly on the training set to prevent data leakage.
        """
        df = X.copy()
        if "MonthlyIncome" in df.columns:
            self.global_income_median_ = float(df["MonthlyIncome"].median())

            if "JobRole" in df.columns and "JobLevel" in df.columns:
                grouped = df.groupby(["JobRole", "JobLevel"])["MonthlyIncome"].median()
                self.role_level_medians_ = grouped.to_dict()

            if "JobRole" in df.columns:
                self.role_medians_ = df.groupby("JobRole")["MonthlyIncome"].median().to_dict()

        logger.info(f"Fitted HRFeatureEngineer with {len(self.role_level_medians_)} role-level benchmarks.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute engineered features and return enriched DataFrame.
        """
        df = X.copy()

        # 1. Career Stagnation Indicators
        years_company = df["YearsAtCompany"].astype(float) if "YearsAtCompany" in df.columns else 0.0
        if "YearsSinceLastPromotion" in df.columns:
            df["StagnationIndex"] = df["YearsSinceLastPromotion"].astype(float) / (years_company + 1.0)
        else:
            df["StagnationIndex"] = 0.0

        if "YearsInCurrentRole" in df.columns:
            df["RoleStagnationRatio"] = df["YearsInCurrentRole"].astype(float) / (years_company + 1.0)
        else:
            df["RoleStagnationRatio"] = 0.0

        if "YearsWithCurrManager" in df.columns:
            df["ManagerStabilityRatio"] = df["YearsWithCurrManager"].astype(float) / (years_company + 1.0)
        else:
            df["ManagerStabilityRatio"] = 0.0

        # 2. Compensation Equity Index (Relative to peers)
        if "MonthlyIncome" in df.columns:
            incomes = df["MonthlyIncome"].astype(float)
            equity_ratios = []

            for idx, row in df.iterrows():
                role = row.get("JobRole", None)
                level = row.get("JobLevel", None)
                income = row.get("MonthlyIncome", self.global_income_median_)

                # Fallback hierarchy: (role, level) -> role -> global
                benchmark = self.role_level_medians_.get(
                    (role, level),
                    self.role_medians_.get(role, self.global_income_median_)
                )
                if benchmark <= 0:
                    benchmark = self.global_income_median_
                equity_ratios.append(float(income) / float(benchmark))

            df["CompensationEquityIndex"] = np.array(equity_ratios)
        else:
            df["CompensationEquityIndex"] = 1.0

        # 3. Burnout Risk Factor
        is_overtime = (
            (df["OverTime"] == "Yes") | (df["OverTime"] == 1)
        ).astype(int) if "OverTime" in df.columns else 0
        is_frequent_travel = (
            df["BusinessTravel"] == "Travel_Frequently"
        ).astype(int) if "BusinessTravel" in df.columns else 0
        is_low_wlb = (
            df["WorkLifeBalance"].astype(float) <= 2
        ).astype(int) if "WorkLifeBalance" in df.columns else 0

        # Composite interaction
        df["BurnoutRiskFactor"] = is_overtime * (1 + is_frequent_travel + is_low_wlb)

        # 4. Commute Burden Factor
        if "DistanceFromHome" in df.columns and "JobSatisfaction" in df.columns:
            df["CommuteBurdenFactor"] = df["DistanceFromHome"].astype(float) / (
                df["JobSatisfaction"].astype(float) + 0.5
            )
        else:
            df["CommuteBurdenFactor"] = 0.0

        # 5. Composite Engagement Score (4 to 16)
        sat_cols = [c for c in ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance"] if c in df.columns]
        if sat_cols:
            df["SatisfactionSum"] = df[sat_cols].sum(axis=1).astype(float)
        else:
            df["SatisfactionSum"] = 10.0

        # 6. High-Risk Vulnerability Flags
        if "StockOptionLevel" in df.columns:
            df["StockOptionZero"] = (df["StockOptionLevel"].astype(float) == 0).astype(int)
        else:
            df["StockOptionZero"] = 0

        if "MaritalStatus" in df.columns and "BusinessTravel" in df.columns:
            df["SingleAndFrequentTravel"] = (
                (df["MaritalStatus"] == "Single") & (df["BusinessTravel"] == "Travel_Frequently")
            ).astype(int)
        else:
            df["SingleAndFrequentTravel"] = 0

        # 7. Career Velocity & Age Ratios
        if "TotalWorkingYears" in df.columns and "Age" in df.columns:
            denom_age = np.maximum(1.0, df["Age"].astype(float) - 17.0)
            df["TenureToAgeRatio"] = df["TotalWorkingYears"].astype(float) / denom_age
        else:
            df["TenureToAgeRatio"] = 0.0

        if "MonthlyIncome" in df.columns and "TotalWorkingYears" in df.columns:
            denom_tenure = np.maximum(1.0, df["TotalWorkingYears"].astype(float))
            df["IncomePerYearWorked"] = df["MonthlyIncome"].astype(float) / denom_tenure
        else:
            df["IncomePerYearWorked"] = 0.0

        return df
