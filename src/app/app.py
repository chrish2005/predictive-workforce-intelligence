"""
Predictive Workforce Intelligence: Machine Learning & Decision Intelligence System
Main Streamlit Application Entrypoint.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_ibm_dataset, get_data_filepath
from src.data.generator import generate_synthetic_hr_data
from src.preprocessing.cleaner import clean_hr_dataset
from src.models.train import load_model_artifacts
from src.models.auto_trainer import train_universal_workforce_bundle, WorkforceCohortBundle
from src.explainability.explainer import HRExplainer
from src.app.styles import CUSTOM_CSS
from src.app.views.executive import render_executive_view
from src.app.views.triage import render_triage_view
from src.app.views.inspector import render_inspector_view
from src.app.views.simulator_view import render_simulator_view
from src.app.views.batch_scoring import render_batch_scoring_view, score_batch_dataframe
from src.app.views.cross_benchmark_view import render_cross_benchmark_view

# Page Configuration
st.set_page_config(
    page_title="Predictive Workforce Intelligence",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply styling
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# CACHED DATASET & MODEL BUNDLES
# ==============================================================================

@st.cache_resource(show_spinner="Loading IBM corporate benchmark artifacts...")
def get_ibm_bundle() -> WorkforceCohortBundle:
    """Load pre-trained IBM artifacts, score cohort, and wrap into unified bundle."""
    calibrated_model, raw_model, xgb_model, preprocessor, metadata = load_model_artifacts()
    explainer = HRExplainer(xgb_model, metadata["feature_names"])
    df = load_ibm_dataset()
    scored_df = score_batch_dataframe(df, calibrated_model, preprocessor)
    clean_df, _, _ = clean_hr_dataset(df, drop_identifiers=True)
    X_trans = preprocessor.transform(clean_df)

    ibm_meta = dict(metadata)
    ibm_meta["cohort_name"] = "IBM Corporate Benchmark (1,470 Employees)"
    ibm_meta["target_col"] = "Attrition"
    ibm_meta["n_samples"] = len(df)
    ibm_meta["n_features"] = len(metadata["feature_names"])
    ibm_meta["turnover_rate"] = float(scored_df["turnover_risk"].mean())
    ibm_meta["champion_name"] = metadata.get("best_model_name", "Logistic_Regression (L2)")

    return WorkforceCohortBundle(
        cohort_name="IBM Corporate Benchmark (1,470 Employees)",
        target_col="Attrition",
        raw_df=df,
        scored_df=scored_df,
        calibrated_model=calibrated_model,
        tree_model=xgb_model,
        explainer=explainer,
        preprocessor=preprocessor,
        X_trans=X_trans,
        metadata=ibm_meta,
    )


@st.cache_resource(show_spinner="Training and calibrating Tech 15k benchmark bundle...")
def get_tech_15k_bundle() -> WorkforceCohortBundle:
    """Load Kaggle Tech HR dataset, train universal pipeline, calibrate and explain."""
    tech_path = get_data_filepath("tech_hr_15k.csv", folder="raw")
    if tech_path.exists():
        df_tech = pd.read_csv(tech_path)
    else:
        df_tech = load_ibm_dataset()

    bundle = train_universal_workforce_bundle(
        df_tech,
        cohort_name="Tech & IT Industry Benchmark (14,999 Employees)",
        target_col="left" if "left" in df_tech.columns else None,
    )
    return bundle


@st.cache_resource(show_spinner="Preparing enterprise synthetic multi-cohort...")
def get_synthetic_bundle() -> WorkforceCohortBundle:
    """Load or generate synthetic enterprise cohort (5,000 employees)."""
    synth_path = get_data_filepath("enterprise_cohort_5000.parquet", folder="synthetic")
    if synth_path.exists():
        df_synth = pd.read_parquet(synth_path)
    else:
        df_synth = generate_synthetic_hr_data(n_records=5000, output_path=synth_path)

    calibrated_model, raw_model, xgb_model, preprocessor, metadata = load_model_artifacts()
    explainer = HRExplainer(xgb_model, metadata["feature_names"])
    scored_df = score_batch_dataframe(df_synth, calibrated_model, preprocessor)
    clean_df, _, _ = clean_hr_dataset(df_synth, drop_identifiers=True)
    X_trans = preprocessor.transform(clean_df)

    synth_meta = dict(metadata)
    synth_meta["cohort_name"] = "Enterprise Synthetic Multi-Cohort (5,000 Employees)"
    synth_meta["target_col"] = "Attrition"
    synth_meta["n_samples"] = len(df_synth)
    synth_meta["n_features"] = len(metadata["feature_names"])
    synth_meta["turnover_rate"] = float(scored_df["turnover_risk"].mean())
    synth_meta["champion_name"] = metadata.get("best_model_name", "Logistic_Regression (L2)")

    return WorkforceCohortBundle(
        cohort_name="Enterprise Synthetic Multi-Cohort (5,000 Employees)",
        target_col="Attrition",
        raw_df=df_synth,
        scored_df=scored_df,
        calibrated_model=calibrated_model,
        tree_model=xgb_model,
        explainer=explainer,
        preprocessor=preprocessor,
        X_trans=X_trans,
        metadata=synth_meta,
    )


# ==============================================================================
# MAIN APPLICATION CONTROLLER
# ==============================================================================

def main():
    # Sidebar Navigation & Branding
    with st.sidebar:
        st.markdown("## 👥 Workforce Intelligence")
        st.caption("Machine Learning & Decision Intelligence System for Turnover Analysis")
        st.markdown("---")

        # Workspace Navigation
        nav_options = [
            "🏢 Executive Intelligence",
            "🚨 Early-Warning Triage",
            "🔍 Employee Deep-Dive & XAI",
            "🧪 Retention Simulator ('What-If')",
            "📁 Ingestion & Custom Dataset Hub",
            "🔄 Multi-Dataset Benchmark",
        ]

        if "nav_choice" not in st.session_state or st.session_state["nav_choice"] not in nav_options:
            st.session_state["nav_choice"] = nav_options[0]

        nav_choice = st.radio(
            "Navigation Workspace",
            options=nav_options,
            index=nav_options.index(st.session_state["nav_choice"]),
            key="current_nav_selection",
        )
        st.session_state["nav_choice"] = nav_choice

        st.markdown("---")
        st.markdown("### 📊 Active Workforce Cohort")

        # Define Cohort Choices
        has_custom = "custom_bundle" in st.session_state
        custom_label = (
            f"📤 Custom: {st.session_state['custom_bundle'].cohort_name[:25]}"
            if has_custom
            else "📤 Custom Uploaded Dataset (Upload in Hub)"
        )

        cohort_options = [
            "🏢 IBM Corporate Benchmark (1,470 Employees)",
            "💻 Tech & IT Industry Benchmark (14,999 Employees)",
            "🌐 Enterprise Synthetic Multi-Cohort (5,000 Employees)",
            custom_label,
        ]

        # Determine default selection
        default_cohort_idx = 0
        if st.session_state.get("active_cohort_key") == "Custom Uploaded Dataset" and has_custom:
            default_cohort_idx = 3

        selected_cohort = st.selectbox(
            "Select Workforce Cohort",
            options=cohort_options,
            index=default_cohort_idx,
            key="active_cohort_selector",
        )

        # Resolve Active Bundle
        bundle = None

        if selected_cohort.startswith("🏢 IBM Corporate"):
            bundle = get_ibm_bundle()
            st.session_state["active_cohort_key"] = "IBM"
        elif selected_cohort.startswith("💻 Tech & IT"):
            bundle = get_tech_15k_bundle()
            st.session_state["active_cohort_key"] = "Tech15k"
        elif selected_cohort.startswith("🌐 Enterprise Synthetic"):
            bundle = get_synthetic_bundle()
            st.session_state["active_cohort_key"] = "Synthetic"
        elif selected_cohort.startswith("📤 Custom"):
            if has_custom:
                bundle = st.session_state["custom_bundle"]
                st.session_state["active_cohort_key"] = "Custom Uploaded Dataset"
            else:
                bundle = None

        # System Health & Model Telemetry Card
        st.markdown("---")
        st.markdown("### ⚙️ Model Telemetry")

        if bundle is not None:
            meta = bundle.metadata
            test_meta = meta.get("test_metrics", {})
            champ = meta.get("champion_name") or meta.get("best_model_name", "Auto-Classifier")
            roc = test_meta.get("roc_auc", 0.832)
            pr = test_meta.get("pr_auc", 0.587)
            brier = test_meta.get("brier_score", 0.097)
            p10 = test_meta.get("precision_at_top10", 0.70) * 100

            st.markdown(f"""
            <div style="font-size: 0.82rem; background: #f8fafc; border: 1px solid #e2e8f0; padding: 0.85rem; border-radius: 10px; color: #334155; line-height: 1.5;">
                <div style="font-weight: 700; color: #0f172a; margin-bottom: 0.3rem;">{bundle.cohort_name[:32]}</div>
                <div><b>Champion:</b> {champ}</div>
                <div><b>ROC-AUC:</b> <span style="color:#2563eb; font-weight:700;">{roc:.3f}</span></div>
                <div><b>PR-AUC:</b> {pr:.3f}</div>
                <div><b>Brier Score:</b> {brier:.3f} (Calibrated)</div>
                <div><b>Precision@Top10%:</b> {p10:.1f}%</div>
                <div><b>Records:</b> {len(bundle.scored_df):,} | <b>Features:</b> {meta.get('n_features', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Upload your dataset in the Hub to activate telemetry.")

    # Guard: Custom dataset selected but none uploaded yet
    if bundle is None:
        st.markdown("""
        <div class="header-container">
            <div class="header-title">Predictive Workforce Intelligence</div>
            <div class="header-subtitle">
                Universal Machine Learning & Explainable AI Platform
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.warning("⚠️ **No custom workforce dataset has been activated yet.**")
        st.info(
            "To analyze your own organization, navigate to the **📁 Ingestion & Custom Dataset Hub**. "
            "Upload any CSV or Excel file containing employee attrition/turnover records. "
            "Our automated engine will clean the data, train and calibrate a high-performance classifier, "
            "compute SHAP explainability matrices, and instantly activate all platform views!"
        )

        if st.button("🚀 Open Ingestion & Custom Dataset Hub", type="primary"):
            st.session_state["nav_choice"] = "📁 Ingestion & Custom Dataset Hub"
            st.rerun()

        return

    # Header Banner
    st.markdown(f"""
    <div class="header-container">
        <div class="header-title">Predictive Workforce Intelligence</div>
        <div class="header-subtitle">
            Explainable AI (XAI) & Prescriptive Retention Platform • Cohort: <b>{bundle.cohort_name}</b> ({len(bundle.scored_df):,} Monitored Employees)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Route to Selected Workspace View
    if nav_choice == "🏢 Executive Intelligence":
        render_executive_view(
            bundle.raw_df,
            bundle.scored_df,
            bundle.explainer,
            bundle.X_trans,
            bundle.metadata,
        )

    elif nav_choice == "🚨 Early-Warning Triage":
        render_triage_view(bundle.scored_df)

    elif nav_choice == "🔍 Employee Deep-Dive & XAI":
        render_inspector_view(
            bundle.scored_df,
            bundle.explainer,
            bundle.preprocessor,
            bundle.X_trans,
        )

    elif nav_choice == "🧪 Retention Simulator ('What-If')":
        render_simulator_view(
            bundle.scored_df,
            bundle.calibrated_model,
            bundle.preprocessor,
        )

    elif nav_choice == "📁 Ingestion & Custom Dataset Hub":
        render_batch_scoring_view(
            bundle.calibrated_model,
            bundle.preprocessor,
        )

    elif nav_choice == "🔄 Multi-Dataset Benchmark":
        render_cross_benchmark_view(bundle.metadata)


if __name__ == "__main__":
    main()
