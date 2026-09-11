"""
View 3: Employee Deep-Dive & Root-Cause Inspector.
Provides individual employee profile analytics, interactive SHAP waterfall attribution,
peer compensation benchmarking, and targeted flight-risk diagnostic insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any

from src.app.styles import PLOTLY_LAYOUT_TEMPLATE, DANGER_RED, SUCCESS_GREEN, ACCENT_BLUE, WARNING_AMBER
from src.app.helpers import get_employee_display_role, get_employee_display_department, get_employee_compensation, get_column_matching
from src.explainability.explainer import HRExplainer, get_employee_waterfall_data


def render_inspector_view(
    scored_df: pd.DataFrame,
    explainer: HRExplainer,
    preprocessor: Any,
    X_trans: np.ndarray,
):
    st.markdown("### 🔍 Employee Deep-Dive & Root-Cause Inspector")
    st.markdown("Individualized Explainable AI (XAI) diagnostics. Understand the precise positive and negative drivers behind any employee's flight risk.")

    # 1. Employee Selector
    col_sel, col_quick = st.columns([2, 1])

    with col_quick:
        filter_at_risk = st.checkbox("Show only high-risk employees (> 50%)", value=True)

    candidates_df = scored_df[scored_df["turnover_risk"] >= 0.50] if filter_at_risk else scored_df
    if candidates_df.empty:
        candidates_df = scored_df

    employee_options = [
        f"ID {row['EmployeeID']} - {get_employee_display_role(row)} ({row['turnover_risk'] * 100:.1f}% Risk)"
        for _, row in candidates_df.iterrows()
    ]

    with col_sel:
        selected_option = st.selectbox("Select Employee to Inspect", options=employee_options, index=0)

    selected_emp_id = selected_option.split(" - ")[0].replace("ID ", "")
    emp_matches = scored_df[scored_df["EmployeeID"].astype(str) == str(selected_emp_id)]
    if emp_matches.empty:
        emp_row = scored_df.iloc[0]
        emp_idx = 0
    else:
        emp_row = emp_matches.iloc[0]
        emp_idx = emp_matches.index[0]

    st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)

    # 2. Employee Profile & Risk Card
    p1, p2, p3, p4 = st.columns([1.2, 1, 1, 1])

    with p1:
        risk_pct = emp_row["turnover_risk"] * 100.0
        risk_color = DANGER_RED if risk_pct >= 70 else (WARNING_AMBER if risk_pct >= 40 else SUCCESS_GREEN)
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {risk_color};">
            <div class="metric-label">Employee ID #{selected_emp_id}</div>
            <div class="metric-value" style="color:{risk_color}; font-size: 2.2rem;">{risk_pct:.1f}%</div>
            <div class="metric-subtitle">
                <span class="badge-{emp_row.get('RiskTier', 'Medium').lower()}">{emp_row.get('RiskTier', 'Medium')} Risk Tier</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Organizational Role</div>
            <div style="font-weight:700; color:#0f172a; margin-top:0.3rem;">{get_employee_display_role(emp_row)}</div>
            <div style="font-size:0.85rem; color:#64748b;">Dept: {get_employee_display_department(emp_row)}</div>
        </div>
        """, unsafe_allow_html=True)

    with p3:
        annual_comp = get_employee_compensation(emp_row)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Compensation Benchmark</div>
            <div style="font-weight:700; color:#0f172a; margin-top:0.3rem;">${annual_comp:,.0f}/yr</div>
            <div style="font-size:0.85rem; color:#64748b;">Turnover Impact: ${emp_row['expected_financial_exposure']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with p4:
        # Dynamic workload info depending on dataset
        if "average_montly_hours" in emp_row:
            work_info = f"Monthly Hours: {emp_row.get('average_montly_hours', 'N/A')} hrs"
            ten_info = f"Tenure: {emp_row.get('time_spend_company', 'N/A')} yrs"
        else:
            work_info = f"OverTime: {emp_row.get('OverTime', 'N/A')}"
            ten_info = f"Tenure: {emp_row.get('YearsAtCompany', 'N/A')} yrs"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Workload & Tenure</div>
            <div style="font-weight:700; color:#0f172a; margin-top:0.3rem;">{work_info}</div>
            <div style="font-size:0.85rem; color:#64748b;">{ten_info}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Interactive SHAP Waterfall Decomposition Chart
    st.markdown("#### 🌊 Root-Cause Feature Attribution (SHAP Waterfall)")
    st.markdown("Visualizes individual pushes (increasing risk in red) and pulls (mitigating risk in green) relative to the cohort baseline.")

    emp_vector = X_trans[emp_idx]
    wf_data = get_employee_waterfall_data(explainer, emp_vector, top_n=8)

    # Prepare Plotly Waterfall data
    base_val = wf_data["base_value"]
    measures = ["absolute"]
    x_labels = ["Cohort Baseline"]
    y_values = [base_val]
    text_values = [f"{base_val:+.2f}"]

    for driver in wf_data["top_drivers"]:
        measures.append("relative")
        clean_name = driver["feature"].replace("cat__", "").replace("num__", "").replace("_", " ")
        x_labels.append(clean_name)
        attr = driver["attribution"]
        y_values.append(attr)
        text_values.append(f"{attr:+.2f}")

    if abs(wf_data["other_impact"]) > 1e-4:
        measures.append("relative")
        x_labels.append("Other Factors")
        y_values.append(wf_data["other_impact"])
        text_values.append(f"{wf_data['other_impact']:+.2f}")

    measures.append("total")
    x_labels.append("Final Score")
    y_values.append(wf_data["final_raw_margin"])
    text_values.append(f"{wf_data['final_raw_margin']:+.2f}")

    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=x_labels,
        textposition="outside",
        text=text_values,
        y=y_values,
        connector={"line": {"color": "#cbd5e1"}},
        increasing={"marker": {"color": DANGER_RED}},
        decreasing={"marker": {"color": SUCCESS_GREEN}},
        totals={"marker": {"color": ACCENT_BLUE}},
    ))

    fig_wf.update_layout(
        PLOTLY_LAYOUT_TEMPLATE,
        height=450,
        yaxis=dict(title="Contribution to Risk Log-Odds"),
        xaxis=dict(tickangle=-25),
    )
    st.plotly_chart(fig_wf, use_container_width=True)
