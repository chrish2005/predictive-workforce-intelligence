"""
Enterprise HR Cohort Synthetic Generator.
Generates realistic multi-thousand employee datasets respecting real-world HR
covariances, career tenure hierarchies, and attrition driver interactions.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)

DEPARTMENTS_ROLES = {
    "Research & Development": [
        "Research Scientist",
        "Laboratory Technician",
        "Manufacturing Director",
        "Healthcare Representative",
        "Research Director",
    ],
    "Sales": [
        "Sales Executive",
        "Sales Representative",
        "Manager",
    ],
    "Human Resources": [
        "Human Resources",
        "Manager",
    ],
}

EDUCATION_FIELDS = [
    "Life Sciences",
    "Medical",
    "Marketing",
    "Technical Degree",
    "Human Resources",
    "Other",
]


def generate_synthetic_hr_data(
    n_records: int = 5000,
    random_seed: int = 42,
    target_attrition_rate: float = 0.16,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate an enterprise cohort of synthetic employee records matching the IBM schema
    with mathematically consistent tenure hierarchies and turnover risk factors.

    Args:
        n_records: Number of employee records to generate.
        random_seed: Random seed for reproducibility.
        target_attrition_rate: Approximate positive turnover proportion (~16%).
        output_path: Optional path to save parquet/csv.

    Returns:
        pd.DataFrame containing synthetic HR data.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Demographics
    ages = rng.integers(18, 61, size=n_records)
    genders = rng.choice(["Male", "Female"], p=[0.60, 0.40], size=n_records)
    marital_statuses = rng.choice(["Married", "Single", "Divorced"], p=[0.46, 0.32, 0.22], size=n_records)
    distance_from_home = np.clip(rng.exponential(scale=7.0, size=n_records) + 1, 1, 30).astype(int)
    education_levels = rng.choice([1, 2, 3, 4, 5], p=[0.11, 0.19, 0.39, 0.27, 0.04], size=n_records)
    education_fields = rng.choice(EDUCATION_FIELDS, p=[0.41, 0.31, 0.11, 0.09, 0.03, 0.05], size=n_records)

    # 2. Organizational Alignment
    dept_choices = list(DEPARTMENTS_ROLES.keys())
    departments = rng.choice(dept_choices, p=[0.65, 0.30, 0.05], size=n_records)
    job_roles = []
    for dept in departments:
        roles = DEPARTMENTS_ROLES[dept]
        job_roles.append(rng.choice(roles))
    job_roles = np.array(job_roles)

    business_travels = rng.choice(
        ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
        p=[0.71, 0.19, 0.10],
        size=n_records,
    )

    # 3. Career Velocity & Hierarchy (Enforce temporal bounds)
    # TotalWorkingYears <= Age - 18
    max_total_work = np.maximum(0, ages - 18)
    total_working_years = np.clip(
        rng.beta(2, 3, size=n_records) * max_total_work,
        0,
        max_total_work
    ).astype(int)

    # YearsAtCompany <= TotalWorkingYears
    years_at_company = np.clip(
        rng.beta(1.5, 2.5, size=n_records) * total_working_years,
        0,
        total_working_years
    ).astype(int)

    # YearsInCurrentRole <= YearsAtCompany
    years_in_current_role = np.clip(
        rng.beta(2, 2, size=n_records) * years_at_company,
        0,
        years_at_company
    ).astype(int)

    # YearsWithCurrManager <= YearsAtCompany
    years_with_curr_manager = np.clip(
        rng.beta(2, 2, size=n_records) * years_at_company,
        0,
        years_at_company
    ).astype(int)

    # YearsSinceLastPromotion <= YearsAtCompany
    years_since_last_promotion = np.clip(
        rng.beta(1.2, 3.0, size=n_records) * years_at_company,
        0,
        years_at_company
    ).astype(int)

    num_companies_worked = np.clip(rng.poisson(lam=2.5, size=n_records), 0, 9)

    # Job Level (1 to 5) determined by seniority & total tenure
    job_levels = np.ones(n_records, dtype=int)
    for i in range(n_records):
        tw = total_working_years[i]
        if tw <= 3:
            lvl = 1
        elif tw <= 7:
            lvl = rng.choice([1, 2], p=[0.3, 0.7])
        elif tw <= 13:
            lvl = rng.choice([2, 3], p=[0.4, 0.6])
        elif tw <= 20:
            lvl = rng.choice([3, 4], p=[0.4, 0.6])
        else:
            lvl = rng.choice([4, 5], p=[0.3, 0.7])
        # Specific executive roles force high level
        if job_roles[i] in ["Manager", "Research Director"]:
            lvl = max(lvl, rng.choice([4, 5], p=[0.5, 0.5]))
        job_levels[i] = lvl

    # 4. Compensation & Equity (Realistic bands by JobLevel)
    base_incomes = {
        1: (2200, 3400),
        2: (4200, 6500),
        3: (7200, 10500),
        4: (11500, 15500),
        5: (16000, 20000),
    }
    monthly_incomes = np.zeros(n_records, dtype=int)
    for i in range(n_records):
        low, high = base_incomes[job_levels[i]]
        monthly_incomes[i] = int(rng.uniform(low, high))

    daily_rates = rng.integers(100, 1500, size=n_records)
    hourly_rates = rng.integers(30, 101, size=n_records)
    monthly_rates = rng.integers(2000, 27000, size=n_records)
    percent_salary_hikes = rng.integers(11, 26, size=n_records)
    stock_option_levels = rng.choice([0, 1, 2, 3], p=[0.43, 0.40, 0.11, 0.06], size=n_records)

    # 5. Satisfaction, Work-Life & Engagement (1 to 4 scales)
    job_satisfaction = rng.choice([1, 2, 3, 4], p=[0.20, 0.20, 0.30, 0.30], size=n_records)
    env_satisfaction = rng.choice([1, 2, 3, 4], p=[0.20, 0.20, 0.30, 0.30], size=n_records)
    rel_satisfaction = rng.choice([1, 2, 3, 4], p=[0.19, 0.21, 0.31, 0.29], size=n_records)
    job_involvement = rng.choice([1, 2, 3, 4], p=[0.06, 0.25, 0.59, 0.10], size=n_records)
    work_life_balance = rng.choice([1, 2, 3, 4], p=[0.06, 0.23, 0.61, 0.10], size=n_records)
    perf_ratings = np.where(percent_salary_hikes > 20, 4, 3)
    trainings_last_year = rng.integers(0, 7, size=n_records)
    overtimes = rng.choice(["Yes", "No"], p=[0.28, 0.72], size=n_records)

    # 6. Latent Attrition Propensity (Logistic function based on real HR drivers)
    # Stagnation penalty
    stagnation = years_since_last_promotion / (years_at_company + 1.0)
    # Undercompensation penalty relative to median level income
    level_medians = {lvl: np.mean(monthly_incomes[job_levels == lvl]) for lvl in range(1, 6)}
    equity_ratios = np.array([monthly_incomes[i] / level_medians[job_levels[i]] for i in range(n_records)])

    # Log-odds calculation
    log_odds = (
        -2.4  # baseline constant
        + 1.6 * (overtimes == "Yes")
        + 0.9 * (business_travels == "Travel_Frequently")
        - 0.5 * (job_satisfaction - 2.5)
        - 0.4 * (env_satisfaction - 2.5)
        - 0.4 * (work_life_balance - 2.5)
        + 0.6 * stagnation
        - 1.1 * (equity_ratios - 1.0)
        + 0.04 * (distance_from_home - 9)
        - 0.4 * stock_option_levels
        - 0.05 * years_at_company
        + 0.5 * (marital_statuses == "Single")
    )
    probabilities = 1.0 / (1.0 + np.exp(-log_odds))

    # Calibrate to target attrition rate
    current_rate = np.mean(probabilities)
    shift = np.log(target_attrition_rate / (1 - target_attrition_rate)) - np.log(current_rate / (1 - current_rate))
    adjusted_probs = 1.0 / (1.0 + np.exp(-(log_odds + shift)))

    attritions = np.where(rng.random(size=n_records) < adjusted_probs, "Yes", "No")

    # Assemble canonical 35-column DataFrame
    df = pd.DataFrame({
        "Age": ages,
        "Attrition": attritions,
        "BusinessTravel": business_travels,
        "DailyRate": daily_rates,
        "Department": departments,
        "DistanceFromHome": distance_from_home,
        "Education": education_levels,
        "EducationField": education_fields,
        "EmployeeCount": 1,
        "EmployeeNumber": np.arange(10001, 10001 + n_records),
        "EnvironmentSatisfaction": env_satisfaction,
        "Gender": genders,
        "HourlyRate": hourly_rates,
        "JobInvolvement": job_involvement,
        "JobLevel": job_levels,
        "JobRole": job_roles,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_statuses,
        "MonthlyIncome": monthly_incomes,
        "MonthlyRate": monthly_rates,
        "NumCompaniesWorked": num_companies_worked,
        "Over18": "Y",
        "OverTime": overtimes,
        "PercentSalaryHike": percent_salary_hikes,
        "PerformanceRating": perf_ratings,
        "RelationshipSatisfaction": rel_satisfaction,
        "StandardHours": 80,
        "StockOptionLevel": stock_option_levels,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": trainings_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
    })

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".parquet":
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        logger.info(f"Saved {n_records} synthetic records to: {output_path}")

    return df


if __name__ == "__main__":
    test_df = generate_synthetic_hr_data(n_records=2000)
    print(f"Generated {len(test_df)} synthetic records. Columns: {len(test_df.columns)}")
    print("Attrition distribution:\n", test_df["Attrition"].value_counts(normalize=True))
