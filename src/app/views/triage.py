"""
View 2: Early-Warning Triage Matrix.
Provides HR Business Partners with an interactive, filterable roster of at-risk employees,
diagnostic flight-risk badges, and one-click export for HRBP operational reviews.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from src.app.styles import DANGER_RED, WARNING_AMBER, SUCCESS_GREEN
from src.app.helpers import get_column_matching, get_employee_display_role, get_employee_display_department


def assign_diagnostic_tags(row: pd.Series) -> str:
    """Assigns human-readable root-cause diagnostic tags based on employee attributes."""
    tags = []
    # Check OverTime / extreme hours
    if row.get("OverTime") == "Yes" or row.get("average_montly_hours", 0) >= 240:
        tags.append("⚡ Overworked")

    # Check Underpayment
    if row.get("CompensationEquityIndex", 1.0) < 0.90 or str(row.get("salary", "")).lower() == "low":
        tags.append("📉 Underpaid")

    # Check Stagnation
    if row.get("StagnationIndex", 0.0) > 0.40 or (row.get("promotion_last_5years", 1) == 0 and row.get("time_spend_company", 0) >= 4):
        tags.append("⏳ Stagnated")

    # Check Satisfaction
    if row.get("SatisfactionSum", 10.0) <= 8 or row.get("satisfaction_level", 1.0) < 0.35:
        tags.append("💔 Low Engagement")

    # Check Zero Stock
    if row.get("StockOptionLevel", 1) == 0:
        tags.append("0️⃣ Zero Stock")

    return " | ".join(tags) if tags else "Standard Profile"


def render_triage_view(scored_df: pd.DataFrame):
    st.markdown("### 🚨 Early-Warning Triage Matrix")
    st.markdown("Operational retention prioritization queue. Filter employees by risk threshold, flight-risk tags, or department.")

    # 1. Filter Controls Bar
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        risk_tier_filter = st.selectbox(
            "Risk Severity Tier",
            options=["All Employees", "Critical Risk (> 70%)", "Moderate Risk (40% - 70%)", "Stable (< 40%)"],
            index=1,
        )

    dept_col = get_column_matching(scored_df, ["department", "dept", "sales", "division"])
    with f_col2:
        if dept_col and dept_col in scored_df.columns:
            dept_options = ["All Departments"] + sorted(scored_df[dept_col].dropna().unique().tolist())
            dept_filter = st.selectbox("Department / Division", options=dept_options, index=0)
        else:
            dept_filter = "All Departments"
            st.selectbox("Department", options=["All Departments"], index=0, disabled=True)

    role_col = get_column_matching(scored_df, ["jobrole", "role", "title", "position", "sales", "salary"])
    with f_col3:
        if role_col and role_col in scored_df.columns:
            role_options = ["All Roles / Tiers"] + sorted(scored_df[role_col].dropna().unique().tolist())
            role_filter = st.selectbox("Role / Category", options=role_options, index=0)
        else:
            role_filter = "All Roles / Tiers"
            st.selectbox("Role / Category", options=["All Roles / Tiers"], index=0, disabled=True)

    with f_col4:
        id_search = st.text_input("Search Employee ID", placeholder="e.g. 1024 or EMP-105")

    # Apply Filters
    filtered = scored_df.copy()

    if risk_tier_filter == "Critical Risk (> 70%)":
        filtered = filtered[filtered["turnover_risk"] >= 0.70]
    elif risk_tier_filter == "Moderate Risk (40% - 70%)":
        filtered = filtered[(filtered["turnover_risk"] >= 0.40) & (filtered["turnover_risk"] < 0.70)]
    elif risk_tier_filter == "Stable (< 40%)":
        filtered = filtered[filtered["turnover_risk"] < 0.40]

    if dept_filter != "All Departments" and dept_col:
        filtered = filtered[filtered[dept_col] == dept_filter]

    if role_filter != "All Roles / Tiers" and role_col:
        filtered = filtered[filtered[role_col] == role_filter]

    if id_search.strip():
        search_term = id_search.strip().lower()
        filtered = filtered[filtered["EmployeeID"].astype(str).str.lower().str.contains(search_term)]

    # 2. Roster Statistics Banner
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Prioritized Cohort Size", f"{len(filtered):,} employees")
    with c2:
        avg_risk = filtered["turnover_risk"].mean() if len(filtered) > 0 else 0.0
        st.metric("Cohort Mean Flight Risk", f"{avg_risk * 100:.1f}%")
    with c3:
        exposure = filtered["expected_financial_exposure"].sum() if len(filtered) > 0 else 0.0
        st.metric("Estimated Financial Exposure", f"${exposure:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    if filtered.empty:
        st.info("No employees match the specified triage filters.")
        return

    # Sort descending by turnover risk
    display_df = filtered.sort_values(by="turnover_risk", ascending=False).copy()
    display_df["Diagnostic Flight Tags"] = display_df.apply(assign_diagnostic_tags, axis=1)
    display_df["Turnover Replacement Cost ($)"] = display_df["expected_financial_exposure"].map("${:,.0f}".format)

    # Standardize display columns
    display_df["Role / Title"] = display_df.apply(get_employee_display_role, axis=1)
    display_df["Department"] = display_df.apply(get_employee_display_department, axis=1)

    preferred_cols = ["EmployeeID", "Risk Score (%)", "RiskTier", "Diagnostic Flight Tags", "Department", "Role / Title", "Turnover Replacement Cost ($)"]
    st.dataframe(
        display_df[preferred_cols],
        use_container_width=True,
        height=450,
        column_config={
            "Risk Score (%)": st.column_config.ProgressColumn(
                "Turnover Risk",
                help="Calibrated empirical probability of attrition",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "RiskTier": st.column_config.TextColumn("Risk Tier"),
        },
    )

    # Export Button
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        label="📥 Export Triage List to CSV (for HRBP Review)",
        data=csv_bytes,
        file_name=f"workforce_triage_{timestamp}.csv",
        mime="text/csv",
    )
