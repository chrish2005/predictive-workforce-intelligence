"""
IBM HR Analytics Employee Attrition & Performance Dataset Loader.
Provides automated retrieval, caching, and verification of the gold-standard 1,470-record dataset.
"""

from pathlib import Path
import pandas as pd
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Primary and mirror URLs for the canonical IBM HR dataset
DATASET_URLS = [
    "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv",
    "https://raw.githubusercontent.com/datasets/awesome-data/master/data/hr/ibm-hr-analytics-employee-attrition-performance.csv",
]

# Benchmark 2: Tech & IT Industry Churn Dataset (14,999 records)
TECH_DATASET_URLS = [
    "https://raw.githubusercontent.com/aiplanethub/Datasets/refs/heads/master/HR_comma_sep.csv",
    "https://raw.githubusercontent.com/sundarstyles89/1df6b5c777e4fb3cb2f9c40212f71dc6/raw/HR_comma_sep.csv",
]

EXPECTED_COLUMNS = [
    "Age", "Attrition", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
    "Education", "EducationField", "EmployeeCount", "EmployeeNumber",
    "EnvironmentSatisfaction", "Gender", "HourlyRate", "JobInvolvement", "JobLevel",
    "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "Over18", "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StandardHours", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager"
]


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_data_filepath(filename: str = "ibm_hr_attrition.csv", folder: str = "raw") -> Path:
    """Return the absolute path for a data file, ensuring target directory exists."""
    target_dir = get_project_root() / "data" / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / filename


def load_ibm_dataset(force_download: bool = False) -> pd.DataFrame:
    """
    Load the IBM HR dataset. If not found locally, downloads and caches it.

    Args:
        force_download: If True, bypasses cache and re-downloads the dataset.

    Returns:
        pd.DataFrame with 1,470 rows and standard 35 HR features.
    """
    local_path = get_data_filepath("ibm_hr_attrition.csv", folder="raw")

    if local_path.exists() and not force_download:
        logger.info(f"Loading cached IBM HR dataset from: {local_path}")
        df = pd.read_csv(local_path)
        if len(df) == 1470:
            return df
        logger.warning(f"Cached dataset has unexpected row count ({len(df)}). Re-downloading...")

    logger.info("Downloading canonical IBM HR Analytics dataset...")
    downloaded_df = None

    for url in DATASET_URLS:
        try:
            logger.info(f"Attempting download from: {url}")
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 10000:
                from io import StringIO
                candidate_df = pd.read_csv(StringIO(resp.text))
                if "Attrition" in candidate_df.columns and len(candidate_df) >= 1400:
                    downloaded_df = candidate_df
                    logger.info(f"Successfully retrieved {len(candidate_df)} records from {url}")
                    break
        except Exception as err:
            logger.warning(f"Failed download from {url}: {err}")

    if downloaded_df is None:
        raise RuntimeError("Unable to download IBM HR Attrition dataset from available mirrors.")

    # Cache dataset locally
    downloaded_df.to_csv(local_path, index=False)
    logger.info(f"Saved canonical dataset to: {local_path}")
    return downloaded_df


def load_tech_dataset(force_download: bool = False) -> pd.DataFrame:
    """
    Load the Tech & IT Industry Churn benchmark dataset (14,999 records).
    If not cached, downloads and persists to data/raw/tech_hr_15k.csv.
    """
    local_path = get_data_filepath("tech_hr_15k.csv", folder="raw")

    if local_path.exists() and not force_download:
        logger.info(f"Loading cached Tech HR dataset from: {local_path}")
        df = pd.read_csv(local_path)
        if len(df) >= 14000:
            return df

    logger.info("Downloading Tech & IT Industry HR benchmark dataset (14,999 records)...")
    downloaded_df = None

    for url in TECH_DATASET_URLS:
        try:
            logger.info(f"Attempting download from: {url}")
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 100000:
                from io import StringIO
                candidate_df = pd.read_csv(StringIO(resp.text))
                if "left" in candidate_df.columns:
                    downloaded_df = candidate_df
                    logger.info(f"Successfully retrieved {len(candidate_df)} records from {url}")
                    break
        except Exception as err:
            logger.warning(f"Failed download from {url}: {err}")

    if downloaded_df is None:
        raise RuntimeError("Unable to download Tech HR dataset from available mirrors.")

    downloaded_df.to_csv(local_path, index=False)
    logger.info(f"Saved Tech HR dataset to: {local_path}")
    return downloaded_df


if __name__ == "__main__":
    df = load_ibm_dataset()
    print(f"Dataset successfully loaded. Shape: {df.shape}")
    print("Columns:", list(df.columns))
    print(df["Attrition"].value_counts(normalize=True))
