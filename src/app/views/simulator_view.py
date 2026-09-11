"""
View 4: Prescriptive Retention Simulator ('What-If' Simulation Lab).
Allows HR leaders to simulate specific retention levers (salary increases, overtime elimination,
work hour reduction, promotion, equity grants, work-life balance) and evaluate real-time risk reduction and financial ROI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any

from src.app.styles import PLOTLY_LAYOUT_TEMPLATE, DANGER_RED, SUCCESS_GREEN, ACCENT_BLUE, WARNING_AMBER
from src.app.helpers import get_employee_display_role, get_employee_compensation, get_column_matching
from src.explainability.simulator import calculate_retention_roi


def render_simulator_view(
    scored_df: pd.DataFrame,
    calibrated_model: Any,
    preprocessor: Any,
):
    st.markdown("### 🧪 Prescriptive Retention Simulator ('What-If' Lab)")
    st.markdown("Design and stress-test customized retention packages. Observe real-time predicted flight-risk reduction and business financial ROI.")

    # 1. Target Employee Selection
    high_risk_cohort = scored_df[scored_df["turnover_risk"] >= 0.40]
    if high_risk_cohort.empty:
        high_risk_cohort = scored_df

    emp_labels = [
        f"ID {row['EmployeeID']} - {get_employee_display_role(row)} ({row['turnover_risk'] * 100:.1f}% Risk)"
        for _, row in high_risk_cohort.iterrows()
    ]

    sel_col, _ = st.columns([2, 1])
    with sel_col:
        selected_label = st.selectbox("Select Target Employee for Retention Package", options=emp_labels, index=0)

    selected_emp_id = selected_label.split(" - ")[0].replace("ID ", "")
    emp_matches = scored_df[scored_df["EmployeeID"].astype(str) == str(selected_emp_id)]
    if emp_matches.empty:
        emp_record = scored_df.iloc[0].to_dict()
    else:
        emp_record = emp_matches.iloc[0].to_dict()

    st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)

    # 2. Simulator Interface: Left Levers Panel, Right Live Impact Panel
    col_levers, col_results = st.columns([1.1, 1.3])

    mod_record = emp_record.copy()
    base_prob = float(emp_record.get("turnover_risk", 0.50))
    ann_salary = get_employee_compensation(pd.Series(emp_record))

    salary_hike_pct = 0.0
    eliminate_ot = False
    promote_role = False
    stock_boost = 0
    wlb_boost = False

    with col_levers:
        st.markdown("#### 🎛️ Dynamic Retention Levers")

        # Lever A: Compensation
        if "MonthlyIncome" in emp_record:
            curr_inc = float(emp_record["MonthlyIncome"])
            st.caption(f"Current Monthly Base: **${curr_inc:,.0f}** (${curr_inc * 12:,.0f}/yr)")
            salary_hike = st.slider("Salary Increase (%)", 0, 35, 10, format="%d%%")
            salary_hike_pct = float(salary_hike)
            mod_record["MonthlyIncome"] = curr_inc * (1.0 + salary_hike / 100.0)
            if "PercentSalaryHike" in mod_record:
                mod_record["PercentSalaryHike"] = float(mod_record["PercentSalaryHike"]) + salary_hike * 0.5
        elif "salary" in emp_record:
            curr_sal = str(emp_record["salary"])
            st.caption(f"Current Salary Tier: **{curr_sal.capitalize()}**")
            new_sal = st.selectbox("Upgrade Salary Tier", ["Current", "Medium", "High"])
            if new_sal != "Current":
                mod_record["salary"] = new_sal.lower()
                salary_hike_pct = 15.0

        # Lever B: Workload / Overtime
        if "OverTime" in emp_record:
            if emp_record["OverTime"] == "Yes":
                eliminate_ot = st.checkbox("🚫 Eliminate Mandatory OverTime", value=True)
                if eliminate_ot:
                    mod_record["OverTime"] = "No"
        elif "average_montly_hours" in emp_record:
            curr_hrs = float(emp_record["average_montly_hours"])
            st.caption(f"Current Workload: **{curr_hrs:.0f} hours/month**")
            reduce_hrs = st.slider("Target Monthly Hours", min_value=120, max_value=int(curr_hrs), value=min(int(curr_hrs), 170), step=10)
            mod_record["average_montly_hours"] = float(reduce_hrs)
            if reduce_hrs < curr_hrs - 30:
                eliminate_ot = True

        # Lever C: Work-Life Balance / Satisfaction
        if "WorkLifeBalance" in emp_record:
            curr_wlb = int(emp_record.get("WorkLifeBalance", 2))
            new_wlb = st.select_slider("Work-Life Balance Program", [1, 2, 3, 4], value=max(curr_wlb, 3))
            mod_record["WorkLifeBalance"] = new_wlb
            wlb_boost = new_wlb > curr_wlb
        elif "satisfaction_level" in emp_record:
            curr_sat = float(emp_record.get("satisfaction_level", 0.3))
            st.caption(f"Current Engagement Score: **{curr_sat * 100:.0f}%**")
            new_sat = st.slider("Target Engagement / Pulse Score", min_value=float(curr_sat), max_value=1.0, value=min(1.0, curr_sat + 0.35), step=0.05)
            mod_record["satisfaction_level"] = float(new_sat)
            wlb_boost = True

        # Lever D: Equity & Stock
        if "StockOptionLevel" in emp_record:
            curr_stock = int(emp_record.get("StockOptionLevel", 0))
            new_stock = st.slider("Stock Option Grant Level", curr_stock, 3, min(3, curr_stock + 1))
            mod_record["StockOptionLevel"] = new_stock
            stock_boost = max(0, new_stock - curr_stock)

        # Lever E: Promotion
        if "JobLevel" in emp_record:
            curr_lvl = int(emp_record.get("JobLevel", 1))
            if curr_lvl < 5 and st.checkbox(f"🎖️ Fast-Track Promotion to Seniority Level {curr_lvl + 1}", value=False):
                mod_record["JobLevel"] = curr_lvl + 1
                if "YearsSinceLastPromotion" in mod_record:
                    mod_record["YearsSinceLastPromotion"] = 0
                promote_role = True
        elif "promotion_last_5years" in emp_record:
            if st.checkbox("🎖️ Authorize Career Promotion", value=True):
                mod_record["promotion_last_5years"] = 1
                promote_role = True

    # Compute live inference
    clean_mod = pd.DataFrame([mod_record])
    for col in ["EmployeeID", "turnover_risk", "Risk Score (%)", "RiskTier", "expected_financial_exposure"]:
        if col in clean_mod.columns:
            clean_mod = clean_mod.drop(columns=[col])

    X_sim = preprocessor.transform(clean_mod)
    sim_prob = float(calibrated_model.predict_proba(X_sim)[0, 1])
    risk_delta = sim_prob - base_prob

    roi = calculate_retention_roi(
        monthly_income=ann_salary / 12.0,
        baseline_risk=base_prob,
        simulated_risk=sim_prob,
        salary_hike_pct=salary_hike_pct,
        eliminate_overtime=eliminate_ot,
        promote_role=promote_role,
        stock_boost=stock_boost,
        wlb_boost=wlb_boost,
    )

    baseline_pct = base_prob * 100.0
    sim_pct = sim_prob * 100.0

    with col_results:
        st.markdown("#### 📊 Real-Time Predicted Retention Impact")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sim_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            delta={'reference': baseline_pct, 'increasing': {'color': DANGER_RED}, 'decreasing': {'color': SUCCESS_GREEN}},
            gauge={
                'axis': {'range': [0, 100], 'ticksuffix': "%"},
                'bar': {'color': ACCENT_BLUE},
                'steps': [
                    {'range': [0, 40], 'color': "#d1fae5"},
                    {'range': [40, 70], 'color': "#fef3c7"},
                    {'range': [70, 100], 'color': "#fee2e2"},
                ],
                'threshold': {
                    'line': {'color': DANGER_RED, 'width': 4},
                    'thickness': 0.75,
                    'value': baseline_pct
                }
            },
            number={'suffix': "%"}
        ))
        fig_gauge.update_layout(PLOTLY_LAYOUT_TEMPLATE, height=270, margin=dict(t=30, b=10, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)

        rel_reduction = max(0.0, -risk_delta / max(0.001, base_prob) * 100.0)
        st.markdown(f"""
        <div style="display:flex; justify-content:space-around; background:#f8fafc; padding:0.8rem; border-radius:10px; margin-bottom:1rem; border:1px solid #e2e8f0;">
            <div style="text-align:center;">
                <div style="font-size:0.75rem; color:#64748b; font-weight:700;">BASELINE RISK</div>
                <div style="font-size:1.3rem; font-weight:800; color:{DANGER_RED};">{baseline_pct:.1f}%</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:0.75rem; color:#64748b; font-weight:700;">SIMULATED RISK</div>
                <div style="font-size:1.3rem; font-weight:800; color:{SUCCESS_GREEN};">{sim_pct:.1f}%</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:0.75rem; color:#64748b; font-weight:700;">RISK ATTENUATION</div>
                <div style="font-size:1.3rem; font-weight:800; color:{ACCENT_BLUE};">-{rel_reduction:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 💼 Business ROI Analysis (SHRM Benchmark)")
        r1, r2 = st.columns(2)
        with r1:
            st.metric("Turnover Cost Preserved", f"${roi['expected_replacement_saved']:,.0f}")
            st.metric("Total Intervention Investment", f"${roi['total_intervention_cost']:,.0f}")
        with r2:
            st.metric("Net Financial Benefit", f"${roi['net_financial_savings']:,.0f}")
            st.metric("Prescriptive ROI Ratio", f"{roi['roi_percentage']:.0f}%")

        if roi["net_financial_savings"] > 0:
            st.success(f"💡 **Positive Business Case**: Every \$1 invested yields **\${roi['roi_percentage'] / 100 + 1:.2f}** in preserved organizational productivity.")
        else:
            st.info("Balanced business scenario. Prioritizing schedule rebalancing over direct cash incentives.")
