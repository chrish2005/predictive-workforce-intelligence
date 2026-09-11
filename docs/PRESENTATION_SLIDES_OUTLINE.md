# 📽️ Viva & Seminar Presentation Slides Outline
## Predictive Workforce Intelligence: A Machine Learning Approach for Employee Attrition Analysis

*Estimated Presentation Time: 12–15 Minutes | Target Audience: Examination Committee, Faculty Reviewers, HR Analytics Experts*

---

### Slide 1: Title & Candidate Information
- **Title:** Predictive Workforce Intelligence: A Machine Learning Approach for Employee Attrition Analysis
- **Subtitle:** An End-to-End Decision Intelligence System with Platt Calibration, Game-Theoretic XAI (SHAP), and Prescriptive Retention Simulation
- **Presented by:** [Student Name / Roll Number]
- **Department:** Computer Science & Engineering / Data Science
- **Academic Year:** 2026

---

### Slide 2: The Enterprise Problem: The High Cost of Unmanaged Attrition
- **The Core Challenge:** Voluntary employee turnover costs organizations 50%–200% of annual salary per departed knowledge worker (SHRM Benchmark).
- **The Operational Trap:** Traditional HR relies on exit interviews—documenting failure *after* talent has left.
- **Project Vision:** Transform HR from retrospective post-mortems into **proactive, calibrated, and prescriptive retention intelligence**.

---

### Slide 3: System Architecture & Workflow
- *Visual:* Mermaid flowchart showing raw data ingestion $\rightarrow$ zero-leakage pipeline $\rightarrow$ 5-Fold Stratified CV $\rightarrow$ Platt probability calibration $\rightarrow$ SHAP explainability $\rightarrow$ Prescriptive simulation $\rightarrow$ Streamlit dashboard.
- **Key Architectural Principles:**
  - Strict featurization ordering (zero leakage).
  - Algorithmic class imbalance handling.
  - End-to-end model persistence and automated testing.

---

### Slide 4: Dataset Profiling & Strict Featurization Ordering
- **Canonical Dataset:** IBM HR Analytics (1,470 employees, 35 attributes).
- **Class Imbalance:** 16.1% voluntary turnover (237 positive cases vs. 1,233 retained).
- **Zero-Variance Cleaning:** Dropped `StandardHours`, `Over18`, and `EmployeeCount`.
- **Leakage Prevention Rule:** Split dataset *strictly before* fitting encoders, scalers, or peer compensation medians.
- **Synthetic Scalability:** Parametric generator capable of synthesizing 5,000–10,000+ realistic records preserving empirical covariances.

---

### Slide 5: Domain-Driven Feature Engineering
- **Why Composite Indicators?** Raw HR features in isolation miss compound organizational friction.
- **High-Signal Formulations:**
  - **`BurnoutRiskFactor`:** OverTime $\times$ Frequent Travel $\times$ Low Work-Life Balance.
  - **`CompensationEquityIndex`:** $\frac{\text{MonthlyIncome}}{\text{Peer Median for (JobRole, JobLevel)}}$ (identifies relative underpayment).
  - **`StagnationIndex`:** $\frac{\text{YearsSinceLastPromotion}}{\text{YearsAtCompany} + 1}$ (measures career plateau).
  - **`SatisfactionSum`:** Unified composite engagement score across 4 survey dimensions.
  - **`StockOptionZero`:** Identifies lack of equity-based golden handcuffs.

---

### Slide 6: Model Training & 5-Fold Stratified Cross-Validation
- **Candidate Architectures:**
  1. Penalized Logistic Regression (L2 / ElasticNet baseline).
  2. Random Forest with `balanced_subsample`.
  3. XGBoost with `scale_pos_weight = 5.19`.
  4. LightGBM with `scale_pos_weight = 5.19`.
- **5-Fold CV Leaderboard:**
  - Logistic Regression: **0.8351 ROC-AUC** | **0.6609 PR-AUC** | **75.0% Precision@Top10%**
  - XGBoost: **0.8090 ROC-AUC** | **0.6044 PR-AUC** | **65.0% Precision@Top10%**
  - LightGBM: **0.8048 ROC-AUC** | **0.5884 PR-AUC** | **64.2% Precision@Top10%**
  - Random Forest: **0.7997 ROC-AUC** | **0.5687 PR-AUC** | **66.7% Precision@Top10%**

---

### Slide 7: Why Regularized Linear Models Outperformed Trees (Occam's Razor)
- **Sample Complexity:** 1,470 rows with 59 encoded features.
- **Collinearity & Smoothness:** Career tenure and compensation exhibit monotonic, smooth trends with attrition log-odds.
- **Variance Control:** Deep tree splits risk minor over-partitioning in sparse categorical combinations. L2 regularization ($C=0.04$) creates an optimal generalized hyperplane.
- **Dual Role:** We preserve XGBoost in production specifically to power **SHAP TreeExplainer** for non-linear interaction insights.

---

### Slide 8: Platt Probability Calibration & Brier Score
- **The Problem:** Raw machine learning outputs are arbitrary ranking scores, not true probabilities.
- **The Solution:** Platt Scaling via `CalibratedClassifierCV(method='sigmoid', cv=5)`:
  $$P(y=1 \mid f(x)) = \frac{1}{1 + \exp(A \cdot f(x) + B)}$$
- **Verification Metric:** Brier Score dropped from 0.1511 (raw) to **0.0967 (calibrated)**, well below the $\le 0.12$ quality ceiling.

---

### Slide 9: Evaluation Under Class Imbalance: ROC-AUC vs. PR-AUC
- **Why Accuracy Fails:** Majority class prediction yields 83.9% accuracy with zero utility.
- **Held-Out Test Set Performance:**
  - **ROC-AUC:** `0.8320` (Target $\ge 0.82$ passed)
  - **PR-AUC:** `0.5867` (Cross-validation mean: `0.6609`)
  - **Precision @ Top 10%:** `66.67%` (2 of every 3 top-flagged employees truly depart)
  - **Financial Savings:** **\$1,751,000** net replacement value preserved.

---

### Slide 10: Explainable AI (SHAP) & Game-Theoretic Attributions
- **Theoretical Basis:** Lloyd Shapley (1953 Nobel Prize in Economics).
- **Four Guarantees:** Efficiency/Additivity ($\sum \phi_i + \phi_0 = f(x)$), Symmetry, Dummy Player, Linearity.
- **Top Systemic Drivers Across Organization:**
  1. `BurnoutRiskFactor` (+ Risk)
  2. `Age` (- Risk)
  3. `SatisfactionSum` (- Risk)
  4. `StockOptionLevel` (- Risk)
  5. `MonthlyIncome` (- Risk)
  6. `NumCompaniesWorked` (+ Risk)

---

### Slide 11: Individual Employee Diagnostic: SHAP Waterfall
- *Visual:* Screenshot or diagram of individual employee SHAP waterfall chart.
- **Diagnostic Power:**
  - Decomposes exact push factors (Overtime, low peer compensation equity, promotion lag).
  - Decomposes protective pull factors (high total tenure, good relationship satisfaction).
  - Explains *why* Employee #1042 is at 74% flight risk.

---

### Slide 12: Prescriptive Analytics: The Counterfactual Retention Simulator
- **Moving from "Who Will Leave?" to "How Do We Retain Them?"**
- **Actionable Interventions:**
  - Salary Increase % (Merit hike)
  - OverTime Elimination (Workload rebalancing)
  - Stock Option Grant (Equity lock-in)
  - Role Promotion (Career velocity acceleration)
  - Work-Life Balance Initiatives
- **Real-Time Dynamic Feedback:** Immediate probability gauge update + Financial ROI card.

---

### Slide 13: Financial ROI Model (SHRM Benchmark)
- **Replacement Cost:** $1.5 \times \text{Annual Base Salary}$.
- **Turnover Risk Avoided:** $\Delta \text{Risk} \times \text{Replacement Cost}$.
- **Intervention Expenses:** Salary increase + stock grant + promotion bonus + wellness perk.
- **Empirical Demonstration:** On typical at-risk technical talent, targeted interventions yield **200%–350% net financial ROI**.

---

### Slide 14: System Implementation & Automated Testing Suite
- **Streamlit Multi-Workspace Application:**
  - 🏢 Executive Intelligence
  - 🚨 Early-Warning Triage Matrix
  - 🔍 Employee Deep-Dive & Root-Cause Inspector
  - 🧪 Prescriptive Retention Simulator ('What-If' Lab)
  - 📁 Batch CSV Scoring Module
- **Automated Verification:** 11 unit tests running under `pytest` with 100% pass rate.

---

### Slide 15: Conclusion, Ethical Guardrails & Future Research
- **Summary of Achievements:** High-discrimination model (0.832 ROC-AUC), calibrated probability (0.097 Brier), exact XAI, and prescriptive ROI simulation.
- **Ethical Safeguards:** Human-in-the-loop decision making; interventions restricted strictly to job/compensation levers, avoiding demographic bias.
- **Future Directions:** Survival analysis (time-to-event modeling), NLP sentiment analysis on employee pulse surveys, organizational graph networks.
- **Thank You:** Questions & Discussion.
