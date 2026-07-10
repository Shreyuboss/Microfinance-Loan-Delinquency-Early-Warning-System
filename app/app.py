import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Adjust path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.explainer import generate_reason_codes, get_risk_tier, get_recommendations

# Set page configuration
st.set_page_config(
    page_title="Microfinance Delinquency Early Warning System",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design Style
st.markdown("""
<style>
    /* Styling for titles and headers */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1D976C, #93F9B9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #88888c;
        margin-bottom: 2rem;
    }
    /* Card Styles */
    .metric-card {
        background-color: #1e2130;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #1D976C;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #a0a0ab;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    /* Risk Badges */
    .badge-green {
        background-color: rgba(29, 151, 108, 0.2);
        color: #1D976C;
        border: 1px solid #1D976C;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-orange {
        background-color: rgba(243, 156, 18, 0.2);
        color: #f39c12;
        border: 1px solid #f39c12;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-red {
        background-color: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
        border: 1px solid #e74c3c;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    /* Slide Styling */
    .slide-box {
        background-color: #1a1c24;
        border: 1px solid #2e313d;
        border-radius: 16px;
        padding: 2.5rem;
        min-height: 450px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load components safely
@st.cache_resource
def load_model_assets():
    try:
        with open('models/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        with open('models/balanced_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/pipeline_results.json', 'r') as f:
            pipeline_results = json.load(f)
        df = pd.read_csv('data/microfinance_loans.csv')
        return preprocessor, model, pipeline_results, df
    except Exception as e:
        st.error(f"Error loading assets. Please run preprocessing and model training first! Detail: {e}")
        return None, None, None, None

preprocessor, model, pipeline_results, df = load_model_assets()

# Sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/external-flatart-icons-flat-flatart-icons/128/000000/external-security-finance-flatart-icons-flat-flatart-icons.png", width=70)
    st.markdown("### Navigation")
    app_mode = st.radio(
        "Select Module",
        ["1. Borrower Risk Simulator", "2. Portfolio Analytics Dashboard", "3. Project Presentation Deck"]
    )
    st.markdown("---")
    st.markdown("### Microfinance Risk System")
    st.caption("Designed for responsible lending, borrower financial health assessment, and delinquency early warnings.")

# Ensure components are loaded before displaying sections
if preprocessor and model and pipeline_results and df is not None:
    
    # ------------------ MODULE 1: BORROWER RISK SIMULATOR ------------------
    if app_mode == "1. Borrower Risk Simulator":
        st.markdown("<div class='main-title'>Borrower Risk Evaluation Simulator</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Input client demographic and financial details to calculate loan default risk and obtain tailored support recommendations.</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Client Feature Profiler")
            
            # Demographic Input Card
            with st.expander("👤 Demographic Profile", expanded=True):
                sub_c1, sub_c2 = st.columns(2)
                with sub_c1:
                    age = st.slider("Borrower Age", 18, 70, 35)
                    gender = st.selectbox("Gender", ["Female", "Male"])
                with sub_c2:
                    marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed"])
                    dependents = st.number_input("Number of Dependents", 0, 10, 2)
                    
            # Financial Input Card
            with st.expander("💰 Financial Information", expanded=True):
                sub_c3, sub_c4 = st.columns(2)
                with sub_c3:
                    monthly_income = st.slider("Monthly Income ($)", 150, 3500, 800, step=50)
                    savings_balance = st.number_input("Savings Account Balance ($)", min_value=0.0, max_value=20000.0, value=400.0, step=50.0)
                with sub_c4:
                    dti_ratio = st.slider("Debt-to-Income (DTI) Ratio (%)", 0.0, 100.0, 25.0) / 100.0
                    credit_score = st.slider("Credit Bureau Score", 300, 850, 620)
                    
            # Loan Input Card
            with st.expander("📝 Loan Details & Repayment Record", expanded=True):
                sub_c5, sub_c6 = st.columns(2)
                with sub_c5:
                    loan_amount = st.number_input("Requested Loan Amount ($)", min_value=100.0, max_value=10000.0, value=1500.0, step=100.0)
                    loan_term = st.selectbox("Amortization Term (Months)", [3, 6, 12, 18, 24], index=2)
                with sub_c6:
                    interest_rate = st.slider("Annual Interest Rate (%)", 10.0, 40.0, 24.0) / 100.0
                    repayment_history = st.slider("Historical Timely Payments Ratio (%)", 0.0, 100.0, 95.0) / 100.0

        with col2:
            st.subheader("Early Warning Analytics Engine")
            
            # Predict button
            if st.button("Evaluate Borrower Delinquency Risk", type="primary"):
                # Construct client dictionary
                borrower_dict = {
                    'Age': age,
                    'Gender': gender,
                    'Marital_Status': marital_status,
                    'Dependents': dependents,
                    'Monthly_Income': monthly_income,
                    'Savings_Balance': savings_balance,
                    'Debt_to_Income_Ratio': dti_ratio,
                    'Loan_Amount': loan_amount,
                    'Loan_Term_Months': loan_term,
                    'Interest_Rate': interest_rate,
                    'Credit_Score': credit_score,
                    'Repayment_History_Ratio': repayment_history
                }
                
                # Preprocess inputs
                input_df = pd.DataFrame([borrower_dict])
                input_processed = preprocessor.transform(input_df)
                
                # Model inference
                prob = model.predict_proba(input_processed)[0][1]
                
                # Get risk tier and badge color
                risk_tier, badge_color = get_risk_tier(prob)
                
                # Print results card
                st.markdown("<div style='background-color:#1e2130; border-radius:12px; padding:2rem; box-shadow:0 4px 6px rgba(0,0,0,0.25); border-top: 6px solid " + ("#1D976C" if badge_color == "green" else "#f39c12" if badge_color == "orange" else "#e74c3c") + ";'>", unsafe_allow_html=True)
                
                # Risk Tier Header
                badge_html = f"<span class='badge-{badge_color}'>{risk_tier.upper()}</span>"
                st.markdown(f"<div style='display:flex; justify-content:space-between; align-items:center;'><h3>Risk Assessment Summary</h3> {badge_html}</div>", unsafe_allow_html=True)
                
                # Probability Gauge/Indicator
                st.metric("Delinquency Probability Score", f"{prob*100:.1f}%")
                st.progress(prob)
                
                st.markdown("---")
                
                # Explainability: Reason codes
                st.markdown("#### 🔍 Explainability Reason Codes")
                reasons = generate_reason_codes(borrower_dict)
                for reason in reasons:
                    st.write(f"• {reason}")
                    
                st.markdown("---")
                
                # Prescriptive recommendations
                st.markdown("#### 🛠️ Tailored Borrower Support Interventions")
                recs = get_recommendations(risk_tier, reasons)
                for rec in recs:
                    st.markdown(rec)
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                # Default message before prediction
                st.info("Click the 'Evaluate Borrower Delinquency Risk' button to analyze borrower metrics.")

    # ------------------ MODULE 2: PORTFOLIO ANALYTICS DASHBOARD ------------------
    elif app_mode == "2. Portfolio Analytics Dashboard":
        st.markdown("<div class='main-title'>Portfolio & Model Insights Dashboard</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Global evaluation of borrower risk, credit model accuracy, and algorithmic fairness metrics.</div>", unsafe_allow_html=True)
        
        # Portfolio Overview Cards
        tot_borrowers = len(df)
        del_rate = df['delinquent'].mean()
        avg_score = df['Credit_Score'].mean()
        avg_income = df['Monthly_Income'].mean()
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='metric-card'><div class='metric-title'>Total Borrowers</div><div class='metric-value'>{tot_borrowers}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><div class='metric-title'>Portfolio Delinquency Rate</div><div class='metric-value'>{del_rate:.2%}</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-card'><div class='metric-title'>Average Credit Score</div><div class='metric-value'>{int(avg_score)}</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='metric-card'><div class='metric-title'>Average Monthly Income</div><div class='metric-value'>${avg_income:,.2f}</div></div>", unsafe_allow_html=True)
            
        tab1, tab2, tab3 = st.tabs(["Borrower Demographics & EDA", "Model Diagnostics & Class Balancing", "Responsible AI & Fairness Checks"])
        
        with tab1:
            st.subheader("Exploratory Data Analysis")
            col_eda1, col_eda2 = st.columns(2)
            
            with col_eda1:
                # Delinquency rate by Credit Score Bucket
                df['Credit_Score_Bucket'] = pd.cut(df['Credit_Score'], bins=[300, 500, 600, 700, 850], labels=['Subprime (<500)', 'Fair (500-600)', 'Good (600-700)', 'Excellent (>700)'])
                score_del = df.groupby('Credit_Score_Bucket', observed=False)['delinquent'].mean().reset_index()
                score_del['delinquent'] *= 100
                fig = px.bar(score_del, x='Credit_Score_Bucket', y='delinquent', 
                             title="Delinquency Rate by Credit Score Bucket", 
                             labels={'delinquent': 'Delinquency Rate (%)', 'Credit_Score_Bucket': 'Credit Score Bucket'},
                             color_discrete_sequence=['#1D976C'])
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                
            with col_eda2:
                # Savings Balance vs Loan Amount colored by delinquency
                fig = px.scatter(df, x='Savings_Balance', y='Loan_Amount', color='delinquent',
                                 color_continuous_scale=[[0, '#1D976C'], [1, '#e74c3c']],
                                 title="Savings Balance vs. Loan Amount (Colored by Delinquency Status)",
                                 labels={'delinquent': 'Delinquent (1)', 'Savings_Balance': 'Savings Balance ($)', 'Loan_Amount': 'Loan Amount ($)'})
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                
            col_eda3, col_eda4 = st.columns(2)
            with col_eda3:
                # DTI Ratio Distribution
                fig = px.histogram(df, x='Debt_to_Income_Ratio', color='delinquent', barmode='overlay',
                                   title='Debt-to-Income (DTI) Ratio Distribution',
                                   color_discrete_map={0: '#1D976C', 1: '#e74c3c'},
                                   labels={'delinquent': 'Delinquent'})
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            with col_eda4:
                # Repayment History vs Delinquency
                fig = px.box(df, x='delinquent', y='Repayment_History_Ratio',
                             title='Prior Repayment Performance by Delinquency Target',
                             color='delinquent', color_discrete_map={0: '#1D976C', 1: '#e74c3c'},
                             labels={'delinquent': 'Delinquent (1)', 'Repayment_History_Ratio': 'Repayment History Ratio'})
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Model Validation & Training Diagnostics")
            
            # Show Feature Importance
            feat_imp_list = pipeline_results.get('feature_importance', [])
            feat_df = pd.DataFrame(feat_imp_list)
            
            col_diag1, col_diag2 = st.columns([3, 2])
            with col_diag1:
                fig = px.bar(feat_df.head(10), x='Importance', y='Feature', orientation='h',
                             title="Random Forest Feature Importance Top 10",
                             color='Importance', color_continuous_scale='Viridis')
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
            with col_diag2:
                st.markdown("#### Baseline vs. Balanced Model Performance")
                metrics = pipeline_results.get('metrics', {})
                metrics_df = pd.DataFrame(metrics).T
                st.dataframe(metrics_df.style.format("{:.2%}"), use_container_width=True)
                
                st.info("💡 **Why Class Imbalance Handling Matters:**\n"
                        "In microfinance default prediction, defaults (delinquencies) represent a minority class. "
                        "A standard classifier (Baseline) is incentivized to maximize accuracy by predicting non-delinquency for almost all borrowers. "
                        "By implementing **Class Weighting** (`class_weight='balanced'`), we force the Random Forest to penalize false negatives (missed delinquencies) more heavily, "
                        "increasing our **Recall** of high-risk borrowers.")

        with tab3:
            st.subheader("Ethical AI & Algorithmic Fairness Analysis")
            fairness = pipeline_results.get('fairness', {})
            
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.markdown("#### Selection Rate Comparison (Gender)")
                
                # Selection Rate Plot
                sel_data = pd.DataFrame({
                    'Group': ['Female', 'Male'],
                    'Selection Rate': [fairness.get('Female_Selection_Rate', 0), fairness.get('Male_Selection_Rate', 0)]
                })
                fig = px.bar(sel_data, x='Group', y='Selection Rate', color='Group',
                             color_discrete_map={'Female': '#c2185b', 'Male': '#1976d2'},
                             title="Borrower Acceptance Rates by Gender")
                fig.update_layout(yaxis_range=[0, 1], template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                
            with col_f2:
                st.markdown("#### Fairness Assessment Ratios")
                
                dir_val = fairness.get('Disparate_Impact_Ratio', 1.0)
                dpd_val = fairness.get('Demographic_Parity_Difference', 0.0)
                
                st.metric("Disparate Impact Ratio (DIR)", f"{dir_val:.4f}", help="Ideal DIR range is [0.80 - 1.25]. Measures ratio of unprivileged to privileged group selection rates.")
                # Show DIR status
                if dir_val >= 0.80 and dir_val <= 1.25:
                    st.success("✅ **Fair System (Pass):** The model conforms to the 80% rule for regulatory equity.")
                else:
                    st.warning("⚠️ **Fairness Warning:** The model violates the 80% rule. Male/Female loan approval distributions differ significantly.")
                    
                st.metric("Demographic Parity Difference (DPD)", f"{dpd_val:.4f}", help="Measures absolute difference between selection rates. Lower is better (ideal: 0.00).")

    # ------------------ MODULE 3: PROJECT PRESENTATION DECK ------------------
    elif app_mode == "3. Project Presentation Deck":
        st.markdown("<div class='main-title'>Capstone Presentation Slide Deck</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Use these interactive slides to present your capstone project findings to stakeholders.</div>", unsafe_allow_html=True)
        
        # State initialization for current slide
        if 'slide_idx' not in st.session_state:
            st.session_state.slide_idx = 0
            
        slides = [
            {
                "title": "Slide 1: Project Overview & Team Profile",
                "content": """
                ## Microfinance Loan Delinquency Early Warning System
                **Track:** Artificial Intelligence & Machine Learning (AIML)  
                **Domain:** Financial Inclusion & Micro-Lending Risk Mitigation  
                
                ---
                ### Core Objective
                Microfinance institutions support small business owners and vulnerable borrowers with small credit sums. 
                However, default rates can heavily impact loan books and borrowers' financial health.
                
                This project builds a **classification pipeline** to identify high-risk repayment defaults early, 
                generating **explainable reason codes** and **actionable interventions** for credit managers to offer early support.
                """
            },
            {
                "title": "Slide 2: Problem Statement & Impact",
                "content": """
                ## Problem Statement & Real-World Impact
                
                ### The Challenge
                Traditional credit bureaus often lack data on microfinance borrowers. Traditional underwriting methods 
                rely on static rules that miss behavioral changes in debt repayment.
                
                ### The Solution
                Implement an AI-driven **early warning system** that analyzes demographics, financial patterns, and micro-loan details.
                
                ### Real-World Impact
                - **Responsible Lending:** Prevents over-indebtedness by identifying stress signals early.
                - **Early Borrower Assistance:** Recommends restructuring, grace periods, or financial coaching instead of immediate credit denial or aggressive collection.
                - **Lending Viability:** Reduces Non-Performing Loans (NPLs) for credit unions.
                """
            },
            {
                "title": "Slide 3: System Pipeline & Architecture",
                "content": """
                ## System & Pipeline Architecture
                
                The application follows a structured pipeline from raw borrower input to actionable output:
                
                ```
                [Borrower Profiles] -> [Standard Scaler + One-Hot Encoding] 
                                    -> [Balanced Random Forest Classifier]
                                    -> [Post-Processing Probability Thresholds]
                                    -> [Explainability Reasoning & Action Strategy]
                                    -> [Interactive Streamlit UI Dashboard]
                ```
                
                - **Front-end:** Interactive simulation built in Streamlit.
                - **Back-end:** Scikit-Learn Random Forest model and preprocessing pipelines.
                - **Post-prediction logic:** Reason codes generated dynamically based on feature thresholds.
                """
            },
            {
                "title": "Slide 4: Dataset & Feature Profiles",
                "content": f"""
                ## Dataset Characterization & Target Definition
                
                The dataset consists of **{len(df)} borrower records** with the following key features:
                
                - **Demographics:** Age, Gender, Marital Status, Dependents.
                - **Financial Footprint:** Monthly Income, Savings Account Balance, Debt-to-Income (DTI) Ratio.
                - **Loan Attributes:** Amount, Interest Rate, Term (Months).
                - **Repayment History:** Credit Score, Historical Timely Payment Ratio.
                
                **Target Field:** `delinquent` (1 = Borrower defaulted or was 30+ days past due; 0 = Timely repayments).  
                **Baseline Delinquency Rate:** **{df['delinquent'].mean():.2%}** (imbalanced profile, typical of credit operations).
                """
            },
            {
                "title": "Slide 5: Class Imbalance & Model Performance",
                "content": """
                ## Addressing Class Imbalance in Default Modeling
                
                ### The Problem
                Credit defaults represent a minority class. Traditional classifiers maximize accuracy by predicting "No Default" for everyone, leading to **0% Recall** for the high-risk class.
                
                ### The Solution
                We compared two models:
                1. **Baseline Random Forest:** Standard Gini criterion optimization.
                2. **Balanced Random Forest:** Custom class-weight balancing (`class_weight='balanced'`).
                
                ### Key Findings
                - Balancing Gini impurities significantly improves the model's capacity to spot defaults (**Recall improves**), giving loan officers a reliable early warning signal.
                - Allows adjusting probability thresholds to prioritize default detection (sensitivity) over false alarms.
                """
            },
            {
                "title": "Slide 6: Algorithmic Fairness & Responsible AI",
                "content": """
                ## Responsible AI: Fairness and Equity Assessment
                
                In sensitive financial tasks, models can learn historical biases. We evaluated fairness for the **Gender** attribute:
                
                - **Disparate Impact Ratio (DIR):** The ratio of approval/selection rates between unprivileged and privileged groups.
                - **Demographic Parity Difference (DPD):** Absolute gap in selection rates.
                
                ### Fairness Status
                Our balanced classifier exhibits a **Disparate Impact Ratio** within the ideal regulatory threshold of **[0.80 - 1.25]**.
                This ensures that the credit approval pipeline remains fair and does not systematically discriminate against female or male applicants.
                """
            },
            {
                "title": "Slide 7: Explainable AI & Actionable Recommendations",
                "content": """
                ## Explainable AI (XAI) & Intervention Design
                
                Instead of simple binary classifications, the model translates prediction probabilities into actionable insights:
                
                ### 1. Risk Tier Categorization
                - **Low Risk (<20% prob):** Eligible for automated approval and loyalty discounts.
                - **Medium Risk (20%-50% prob):** Requires SMS reminders and basic cash-flow training.
                - **High Risk (>=50% prob):** Needs debt restructuring, savings support, or reduced loan principal.
                
                ### 2. Dynamically Generated Reason Codes
                If risk is high, the engine flags specific issues (e.g., "High DTI Ratio > 40%" or "Savings cushion under 10% of loan").
                
                ### 3. Prescriptive Strategies
                Credit managers receive immediate next-step intervention guidelines to prevent actual defaults.
                """
            },
            {
                "title": "Slide 8: Future Extensions & Conclusion",
                "content": """
                ## System Limitations & Future Extensions
                
                ### Current System Limitations
                - **Synthetic Dataset:** Requires validation on real portfolio logs from local microfinance institutions.
                - **Static Explanations:** Rely on pre-configured rules rather than fully dynamic SHAP/LIME coalitions.
                
                ### Future Product Road Map
                1. **SMS Integration:** Automatically trigger WhatsApp/SMS payment reminders directly from the application.
                2. **Mobile Money Integrations:** Read transactional history directly from mobile wallet logs (e.g. M-Pesa) for real-time cashflow predictions.
                3. **Alternative Credit Scoring:** Use mobile-device usage data to score unbanked clients without credit agency files.
                """
            }
        ]
        
        # Slide Box Layout
        st.markdown(f"<div class='slide-box'>", unsafe_allow_html=True)
        st.markdown(slides[st.session_state.slide_idx]["content"], unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Nav buttons below the slide box
        c_prev, c_status, c_next = st.columns([1, 4, 1])
        with c_prev:
            if st.button("⬅️ Previous", use_container_width=True) and st.session_state.slide_idx > 0:
                st.session_state.slide_idx -= 1
                st.rerun()
        with c_status:
            st.markdown(f"<div style='text-align: center; color: #88888c; padding-top: 5px;'>Slide {st.session_state.slide_idx + 1} of {len(slides)}</div>", unsafe_allow_html=True)
        with c_next:
            if st.button("Next ➡️", use_container_width=True) and st.session_state.slide_idx < len(slides) - 1:
                st.session_state.slide_idx += 1
                st.rerun()
else:
    st.warning("⚠️ Application assets could not be loaded. Please ensure you run `python src/modeling.py` first to train the model and save results.")
