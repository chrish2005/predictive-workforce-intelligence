"""
View 5: Universal Dataset Ingestion Hub & Batch Scoring.
Allows uploading any corporate HR CSV or Excel file, auto-detecting the target column,
and either dynamically retraining the entire platform or batch-scoring new records.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict

from src.preprocessing.pipeline import HRPreprocessor
from src.preprocessing.cleaner import clean_hr_dataset
from src.preprocessing.universal import auto_detect_target_column
from src.models.auto_trainer import train_universal_workforce_bundle
from src.data.loader import get_data_filepath


def score_batch_dataframe(
    raw_df: pd.DataFrame,
    calibrated_model: Any,
    preprocessor: Any,
) -> pd.DataFrame:
    """
    Cleans, transforms, and scores a batch of employee records.
    """
    cleaned_df, y_truth, emp_ids = clean_hr_dataset(raw_df, drop_identifiers=False)

    if "EmployeeNumber" in cleaned_df.columns:
        emp_ids = cleaned_df["EmployeeNumber"]
        features_df = cleaned_df.drop(columns=["EmployeeNumber"])
    elif emp_ids is not None:
        features_df = cleaned_df
    else:
        emp_ids = pd.Series(np.arange(1, len(cleaned_df) + 1), index=cleaned_df.index)
        features_df = cleaned_df

    X_trans = preprocessor.transform(features_df)
    probs = calibrated_model.predict_proba(X_trans)[:, 1]

    scored = raw_df.copy()
    scored["EmployeeID"] = emp_ids.values
    scored["turnover_risk"] = probs
    scored["Risk Score (%)"] = (probs * 100.0).round(1)
    scored["RiskTier"] = np.where(probs >= 0.70, "High", np.where(probs >= 0.40, "Medium", "Low"))

    monthly_incomes = scored["MonthlyIncome"] if "MonthlyIncome" in scored.columns else 5000.0
    annual_salaries = monthly_incomes * (12.0 if monthly_incomes.mean() < 25000 else 1.0)
    scored["expected_financial_exposure"] = (probs * 1.5 * annual_salaries).round(2)

    return scored


def render_batch_scoring_view(
    current_calibrated_model: Any,
    current_preprocessor: Any,
):
    st.markdown("### 📤 Universal Dataset Ingestion & Auto-Training Hub")
    st.markdown("Upload any enterprise HR dataset (CSV or Excel). The platform will auto-detect the target turnover column, train and calibrate machine learning models, compute SHAP attributions, and **re-populate the entire application** with the uploaded data.")

    col_upload, col_demo = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Enterprise HR Dataset (CSV / Excel)",
            type=["csv", "xlsx"],
            help="Upload any company employee dataset with features and a turnover/attrition column.",
        )

    with col_demo:
        st.markdown("<div style='padding-top: 1.8rem;'></div>", unsafe_allow_html=True)
        use_synth = st.button("🚀 Load 5,000-Record Enterprise Cohort", use_container_width=True)

    df_loaded = None
    file_label = ""

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".xlsx"):
                df_loaded = pd.read_excel(uploaded_file)
            else:
                df_loaded = pd.read_csv(uploaded_file)
            file_label = uploaded_file.name
            st.success(f"Successfully loaded `{uploaded_file.name}` ({len(df_loaded):,} records, {len(df_loaded.columns)} columns)")
        except Exception as e:
            st.error(f"Error parsing uploaded file: {e}")

    elif use_synth:
        synth_path = get_data_filepath("enterprise_cohort_5000.parquet", folder="synthetic")
        if synth_path.exists():
            df_loaded = pd.read_parquet(synth_path)
        else:
            from src.data.generator import generate_synthetic_hr_data
            df_loaded = generate_synthetic_hr_data(n_records=5000)
        file_label = "Synthetic Enterprise Cohort (5,000 Employees)"
        st.info(f"Loaded **{len(df_loaded):,}** synthetic enterprise records.")

    if df_loaded is not None:
        st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)
        st.markdown("#### 🔍 Schema & Target Configuration")

        # Auto-detect target column
        detected_target = auto_detect_target_column(df_loaded)
        col_target, col_action = st.columns([1.5, 1.5])

        with col_target:
            all_cols = list(df_loaded.columns)
            target_idx = all_cols.index(detected_target) if detected_target in all_cols else 0
            selected_target = st.selectbox(
                "Target Turnover Column (Binary Attrition / Churn)",
                options=all_cols,
                index=target_idx,
                help="Select the column indicating whether an employee left or stayed.",
            )

        with col_action:
            st.markdown("<div style='padding-top: 1.8rem;'></div>", unsafe_allow_html=True)
            train_btn = st.button(
                "⚡ Train, Calibrate & Activate Platform with This Dataset",
                type="primary",
                use_container_width=True,
            )

        if train_btn:
            with st.spinner("Executing Zero-Leakage Preprocessing, Model Training, Platt Calibration & SHAP Engine..."):
                new_bundle = train_universal_workforce_bundle(
                    df_loaded,
                    cohort_name=f"Custom: {file_label}",
                    target_col=selected_target,
                )
                # Store in Streamlit session state
                st.session_state["custom_bundle"] = new_bundle
                st.session_state["active_cohort_key"] = "Custom Uploaded Dataset"
                st.success(f"🎉 **Platform Successfully Populated with '{file_label}'!**")
                st.balloons()
                st.rerun()

        # Data Preview Table
        st.markdown("#### Dataset Preview (First 5 Rows)")
        st.dataframe(df_loaded.head(5), use_container_width=True)

    else:
        st.info("Upload a CSV/Excel file or click 'Load 5,000-Record Enterprise Cohort' to begin.")
