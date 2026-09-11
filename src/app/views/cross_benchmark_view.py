"""
View 6: Multi-Dataset Benchmark & Cross-Domain Generalization.
Compares organizational dynamics, turnover rates, and predictive performance
across Corporate (IBM 1.4k), Tech Industry (Kaggle 15k), and Custom Uploaded Cohorts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

from src.app.styles import PLOTLY_LAYOUT_TEMPLATE, ACCENT_BLUE, ACCENT_INDIGO, DANGER_RED, SUCCESS_GREEN, WARNING_AMBER


def render_cross_benchmark_view(
    active_bundle_meta: Dict[str, Any],
):
    st.markdown("### 🔄 Multi-Dataset Benchmark & Generalization Analysis")
    st.markdown("Compare workforce dynamics, model discrimination, and systemic drivers across corporate, technology, and custom cohorts.")

    # 1. Macro Comparison Cards
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="metric-card accent-blue">
            <div class="metric-label">🏢 Benchmark 1: IBM Corporate</div>
            <div class="metric-value">1,470</div>
            <div class="metric-subtitle" style="color:#64748b;">
                Attrition Rate: <b>16.1%</b> | 35 Attributes<br>
                Champion: <b>Penalized Logistic Reg.</b> (ROC: 0.832)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card accent-indigo">
            <div class="metric-label">💻 Benchmark 2: Tech & IT Churn</div>
            <div class="metric-value">14,999</div>
            <div class="metric-subtitle" style="color:#64748b;">
                Attrition Rate: <b>23.8%</b> | 10 Attributes<br>
                Champion: <b>LightGBM Tree</b> (ROC: 0.990)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        active_name = active_bundle_meta.get("cohort_name", "Active Cohort")
        active_samples = active_bundle_meta.get("n_samples", 0)
        active_rate = active_bundle_meta.get("turnover_rate", 0.16) * 100
        active_roc = active_bundle_meta.get("test_metrics", {}).get("roc_auc", 0.832)
        st.markdown(f"""
        <div class="metric-card accent-green">
            <div class="metric-label">⚡ Active Dataset: {active_name}</div>
            <div class="metric-value">{active_samples:,}</div>
            <div class="metric-subtitle" style="color:#64748b;">
                Turnover Rate: <b>{active_rate:.1f}%</b> | Features: <b>{active_bundle_meta.get('n_features', 0)}</b><br>
                Test ROC-AUC: <b>{active_roc:.3f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Side-by-Side Architectural Comparison Table
    st.markdown("#### 📊 Comparative Industry Telemetry Matrix")

    comp_data = [
        {
            "Domain Dimension": "Industry Sector",
            "IBM Corporate Benchmark": "Enterprise Corporate / R&D",
            "Tech & IT Industry Benchmark": "Software, Technical & Startups",
            "Active / Custom Cohort": active_name,
        },
        {
            "Domain Dimension": "Cohort Scale (Records)",
            "IBM Corporate Benchmark": "1,470 Employees",
            "Tech & IT Industry Benchmark": "14,999 Employees",
            "Active / Custom Cohort": f"{active_samples:,} Employees",
        },
        {
            "Domain Dimension": "Empirical Turnover Rate",
            "IBM Corporate Benchmark": "16.1% (Moderate Imbalance)",
            "Tech & IT Industry Benchmark": "23.8% (High Velocity Churn)",
            "Active / Custom Cohort": f"{active_rate:.1f}%",
        },
        {
            "Domain Dimension": "Primary Turnover Catalyst",
            "IBM Corporate Benchmark": "OverTime + Stagnation + Comp Equity",
            "Tech & IT Industry Benchmark": "Extreme Work Hours (>250/mo) + Evaluation",
            "Active / Custom Cohort": "Computed dynamically via SHAP",
        },
        {
            "Domain Dimension": "Optimal Model Architecture",
            "IBM Corporate Benchmark": "L2-Penalized Logistic Reg. (C=0.04)",
            "Tech & IT Industry Benchmark": "LightGBM Gradient Boosted Trees",
            "Active / Custom Cohort": active_bundle_meta.get("champion_name", "Auto-Selected"),
        },
        {
            "Domain Dimension": "Test Set ROC-AUC",
            "IBM Corporate Benchmark": "0.8320 (High Discrimination)",
            "Tech & IT Industry Benchmark": "0.9901 (Near-Deterministic Churn)",
            "Active / Custom Cohort": f"{active_roc:.4f}",
        },
        {
            "Domain Dimension": "Calibrated Brier Score",
            "IBM Corporate Benchmark": "0.0967 (Superior Calibration)",
            "Tech & IT Industry Benchmark": "0.0210 (High Probability Precision)",
            "Active / Custom Cohort": f"{active_bundle_meta.get('test_metrics', {}).get('brier_score', 0.09):.4f}",
        },
        {
            "Domain Dimension": "Precision @ Top 10%",
            "IBM Corporate Benchmark": "66.7% - 75.0%",
            "Tech & IT Industry Benchmark": "100.0%",
            "Active / Custom Cohort": f"{active_bundle_meta.get('test_metrics', {}).get('precision_at_top10', 0.70) * 100:.1f}%",
        },
    ]

    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Chart: Industry Sector Attrition Dynamics
    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        st.markdown("#### Industry Attrition Rates")
        bar_df = pd.DataFrame({
            "Cohort": ["IBM Corporate", "Tech & IT Industry", active_name],
            "Attrition Rate (%)": [16.1, 23.8, active_rate],
        })
        fig_bar = px.bar(
            bar_df,
            x="Cohort",
            y="Attrition Rate (%)",
            color="Attrition Rate (%)",
            text="Attrition Rate (%)",
            color_continuous_scale=[[0, "#3b82f6"], [0.5, "#f59e0b"], [1, "#ef4444"]],
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar.update_layout(PLOTLY_LAYOUT_TEMPLATE, height=320, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with c_chart2:
        st.markdown("#### Cross-Domain Generalization & Transfer Insights")
        st.markdown("""
        <div class="card-panel" style="font-size:0.88rem; color:#334155; line-height:1.6;">
            <b>Why Cross-Dataset Benchmarking Matters in Applied ML:</b>
            <ul>
                <li><b>Domain Shift:</b> In tech startups (Kaggle 15k), employee churn is heavily driven by <i>work overload</i> (extreme monthly hours >250) and <i>under-evaluation</i>. A tree model identifies sharp threshold cliffs.</li>
                <li><b>Corporate Shift:</b> In mature corporate enterprises (IBM 1.4k), turnover is driven by <i>peer compensation disparity</i> and <i>promotion lag</i> across structured seniority ladders.</li>
                <li><b>Platform Adaptability:</b> Rather than overfitting a single dataset, our platform dynamically discovers the target, normalizes features, calibrates probabilities, and generates XAI explanations for <b>any</b> organization.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
