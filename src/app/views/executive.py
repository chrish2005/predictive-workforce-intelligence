"""
View 1: Executive Workforce Intelligence.
Displays macro organizational attrition risk metrics, financial exposure,
departmental breakdowns, and top enterprise-wide root cause drivers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

from src.app.styles import PLOTLY_LAYOUT_TEMPLATE, ACCENT_BLUE, ACCENT_INDIGO, DANGER_RED, WARNING_AMBER, SUCCESS_GREEN
from src.app.helpers import get_column_matching
from src.explainability.explainer import HRExplainer, get_global_feature_importance


def render_executive_view(
    df: pd.DataFrame,
    scored_df: pd.DataFrame,
    explainer: HRExplainer,
    X_trans: np.ndarray,
    metadata: Dict[str, Any],
):
    cohort_name = metadata.get("cohort_name", "Monitored Workforce")
    st.markdown(f"### 🏢 Executive Workforce Intelligence: {cohort_name}")
    st.markdown("Macro enterprise turnover telemetry, organizational risk concentrations, and primary systemic drivers.")

    # 1. Macro KPI Banner
    total_headcount = len(scored_df)
    avg_turnover_prob = float(scored_df["turnover_risk"].mean())
    high_risk_count = int((scored_df["turnover_risk"] >= 0.70).sum())
    medium_risk_count = int(((scored_df["turnover_risk"] >= 0.40) & (scored_df["turnover_risk"] < 0.70)).sum())
    total_financial_exposure = float(scored_df["expected_financial_exposure"].sum())

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(f"""
        <div class="metric-card accent-blue">
            <div class="metric-label">Total Monitored Headcount</div>
            <div class="metric-value">{total_headcount:,}</div>
            <div class="metric-subtitle" style="color:#64748b;">Active employee records</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="metric-card accent-amber">
            <div class="metric-label">Predicted Turnover Rate</div>
            <div class="metric-value">{avg_turnover_prob * 100:.1f}%</div>
            <div class="metric-subtitle" style="color:{DANGER_RED if avg_turnover_prob > 0.20 else WARNING_AMBER}; font-weight:600;">
                {high_risk_count + medium_risk_count:,} employees at moderate/high risk
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-card accent-red">
            <div class="metric-label">Critical Flight Risk</div>
            <div class="metric-value" style="color:{DANGER_RED};">{high_risk_count:,}</div>
            <div class="metric-subtitle" style="color:#64748b;">Turnover probability > 70%</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="metric-card accent-green">
            <div class="metric-label">Financial Turnover Exposure</div>
            <div class="metric-value">${total_financial_exposure / 1e6:.2f}M</div>
            <div class="metric-subtitle" style="color:#64748b;">SHRM 1.5x salary replacement cost</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Charts Row: Department / Category Breakdown & Workload / Seniority
    col_dept, col_seniority = st.columns(2)

    dept_col = get_column_matching(scored_df, ["department", "dept", "sales", "division"])

    with col_dept:
        if dept_col:
            st.markdown(f"#### Turnover Risk by {dept_col}")
            dept_summary = scored_df.groupby(dept_col).agg(
                headcount=("turnover_risk", "count"),
                avg_risk=("turnover_risk", "mean"),
            ).reset_index()
            dept_summary["avg_risk_pct"] = dept_summary["avg_risk"] * 100.0
            dept_summary = dept_summary.sort_values(by="avg_risk_pct", ascending=False)

            fig_dept = px.bar(
                dept_summary,
                x=dept_col,
                y="avg_risk_pct",
                text="avg_risk_pct",
                color="avg_risk_pct",
                color_continuous_scale=[[0, "#93c5fd"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                labels={"avg_risk_pct": "Avg Risk (%)", dept_col: "Department"},
            )
            fig_dept.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_dept.update_layout(PLOTLY_LAYOUT_TEMPLATE, height=340, showlegend=False)
            st.plotly_chart(fig_dept, use_container_width=True)
        else:
            # Fallback to Risk Tier Distribution
            st.markdown("#### Workforce Risk Tier Distribution")
            tier_counts = scored_df["RiskTier"].value_counts().reset_index()
            tier_counts.columns = ["Risk Tier", "Headcount"]
            fig_tier = px.pie(tier_counts, names="Risk Tier", values="Headcount", color="Risk Tier",
                              color_discrete_map={"High": DANGER_RED, "Medium": WARNING_AMBER, "Low": SUCCESS_GREEN})
            fig_tier.update_layout(PLOTLY_LAYOUT_TEMPLATE, height=340)
            st.plotly_chart(fig_tier, use_container_width=True)

    with col_seniority:
        level_col = get_column_matching(scored_df, ["joblevel", "seniority", "number_project", "time_spend_company", "yearsatcompany"])
        if level_col:
            st.markdown(f"#### Risk Distribution across {level_col}")
            lvl_summary = scored_df.groupby(level_col)["turnover_risk"].mean().reset_index()
            lvl_summary["avg_risk_pct"] = lvl_summary["turnover_risk"] * 100.0

            fig_lvl = px.line(
                lvl_summary,
                x=level_col,
                y="avg_risk_pct",
                markers=True,
                labels={"avg_risk_pct": "Avg Risk (%)"},
            )
            fig_lvl.update_traces(line_color=ACCENT_BLUE, line_width=3, marker=dict(size=10, color=DANGER_RED))
            fig_lvl.update_layout(PLOTLY_LAYOUT_TEMPLATE, height=340)
            st.plotly_chart(fig_lvl, use_container_width=True)
        else:
            st.markdown("#### Risk Distribution by Attrition Score")
            fig_hist = px.histogram(scored_df, x="turnover_risk", nbins=25, color="RiskTier",
                                    color_discrete_map={"High": DANGER_RED, "Medium": WARNING_AMBER, "Low": SUCCESS_GREEN})
            fig_hist.update_layout(PLOTLY_LAYOUT_TEMPLATE, height=340)
            st.plotly_chart(fig_hist, use_container_width=True)

    # 3. Charts Row: Role Breakdown & Global SHAP Drivers
    col_roles, col_shap = st.columns(2)

    role_col = get_column_matching(scored_df, ["jobrole", "role", "title", "position", "sales", "salary"])

    with col_roles:
        if role_col and scored_df[role_col].nunique() <= 20:
            st.markdown(f"#### Turnover Risk by {role_col}")
            role_summary = scored_df.groupby(role_col)["turnover_risk"].mean().reset_index()
            role_summary["turnover_risk_pct"] = role_summary["turnover_risk"] * 100.0
            role_summary = role_summary.sort_values(by="turnover_risk_pct", ascending=True)

            fig_role = px.bar(
                role_summary,
                x="turnover_risk_pct",
                y=role_col,
                orientation="h",
                text="turnover_risk_pct",
                color="turnover_risk_pct",
                color_continuous_scale=[[0, "#60a5fa"], [1, "#dc2626"]],
                labels={"turnover_risk_pct": "Avg Risk (%)"},
            )
            fig_role.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_role.update_layout(PLOTLY_LAYOUT_TEMPLATE, height=400, showlegend=False)
            st.plotly_chart(fig_role, use_container_width=True)
        else:
            st.markdown("#### Turnover Probability Density")
            fig_dens = px.box(scored_df, y="turnover_risk", points="outliers", color="RiskTier",
                              color_discrete_map={"High": DANGER_RED, "Medium": WARNING_AMBER, "Low": SUCCESS_GREEN})
            fig_dens.update_layout(PLOTLY_LAYOUT_TEMPLATE, height=400)
            st.plotly_chart(fig_dens, use_container_width=True)

    with col_shap:
        st.markdown("#### Top Systemic Attrition Drivers (SHAP Global)")
        n_samples_shap = min(len(X_trans), 250)
        imp_df = get_global_feature_importance(explainer, X_trans[:n_samples_shap], top_n=10)
        imp_df = imp_df.sort_values(by="mean_abs_shap", ascending=True)

        fig_shap = px.bar(
            imp_df,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={
                "Increases Risk": DANGER_RED,
                "Reduces Risk": SUCCESS_GREEN,
                "Neutral": "#94a3b8",
                "Mixed/Contextual": WARNING_AMBER,
            },
            labels={"mean_abs_shap": "Mean |SHAP Value|", "feature": "Feature"},
        )
        fig_shap.update_layout(PLOTLY_LAYOUT_TEMPLATE, height=400, legend=dict(orientation="h", y=1.1, x=0))
        st.plotly_chart(fig_shap, use_container_width=True)
