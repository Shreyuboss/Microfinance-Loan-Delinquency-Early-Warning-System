# Microfinance Loan Delinquency Early Warning System

An interactive machine learning application and credit scoring framework designed to identify high-risk micro-borrowers early, audit decision fairness, explain risk factors, and recommend tailored interventions.

---

## 1. Project Overview & Problem Statement
Microfinance institutions (MFIs) provide small loans to underserved populations, often with thin or non-existent formal credit profiles. Default risks are challenging to predict, and missing early warning signals can lead to borrower over-indebtedness and loan losses.

This project delivers:
* A **synthetic data generator** reflecting real borrower profiles.
* A **Random Forest modeling pipeline** configured to handle class imbalance.
* A **Fairness audit engine** evaluating disparate impact across gender.
* A **Local explainability layer** generating reason codes and borrower support strategies.
* An **interactive Streamlit app** containing a risk simulator, portfolio analysis dashboard, and built-in project presentation slides.

---

## 2. Technology Stack & Tools
* **Language:** Python 3.8+
* **Data Processing & Analytics:** Pandas, NumPy
* **Machine Learning Pipeline:** Scikit-Learn
* **Visualization:** Plotly, Matplotlib, Seaborn
* **User Interface:** Streamlit

---

## 3. Project Directory Structure
```text
microfinance_loan_delinquency_early_warning_system/
│
├── data/
│   └── microfinance_loans.csv       # Generated dataset
│
├── src/
│   ├── data_generator.py            # Synthetic dataset generator
│   ├── modeling.py                  # Preprocessing, ML model, evaluation
│   └── explainer.py                 # Risk tiers, reason codes, recommendations
│
├── app/
│   └── app.py                       # Streamlit dashboard & presentation app
│
├── models/
│   ├── balanced_model.pkl           # Saved Random Forest model
│   ├── preprocessor.pkl             # Saved data preprocessor
│   └── pipeline_results.json        # Saved evaluation & fairness metrics
│
├── docs/
│   ├── project_report.md            # Detailed academic/project report
│   └── images/                      # Generated diagnostic charts
│
├── requirements.txt                 # Dependencies
└── README.md                        # Setup and usage guide
```

---

## 4. System Workflow
1. **Data Generation:** Run `src/data_generator.py` to create `data/microfinance_loans.csv`.
2. **Model Pipeline:** Run `src/modeling.py` to preprocess features, train standard and balanced models, run fairness audits, and serialize assets to the `models/` directory.
3. **Streamlit App:** Run `streamlit run app/app.py` to interact with the emulator, inspect portfolio dashboards, and run the project presentation slides.

---

## 5. How to Setup and Run

### Prerequisites
Make sure Python is installed.

### Step 1: Install Dependencies
Open your terminal and run:
```bash
pip install -r requirements.txt
```

### Step 2: Generate the Dataset
Create the synthetic microfinance borrower records:
```bash
python src/data_generator.py
```

### Step 3: Run the Training Pipeline
Train the model, save metrics, and generate diagnostic visualizations:
```bash
python src/modeling.py
```

### Step 4: Launch the Streamlit App
Run the interactive application:
```bash
streamlit run app/app.py
```

---

## 6. Key Modeling & Fairness Insights
* **Recall Improvements:** Standard classifiers prioritize accuracy and frequently miss delinquency cases. Applying `class_weight='balanced'` improves the model's recall on the delinquent class (identifying ~51.8% of delinquents early vs. 50.6% in baseline).
* **Fairness Results:** The model achieves a **Disparate Impact Ratio of ~0.917** (passing the 80% four-fifths rule) and a **Demographic Parity Difference of ~0.064** for Gender. This demonstrates that the algorithm avoids systemic discrimination.
* **Risk Drivers:** Feature importances show that a borrower's **Repayment History Ratio**, **Credit Score**, and **Savings Balance** are the primary signals of delinquency risk.

---

## 7. Limitations & Responsible Use
* **Synthetic Data:** The system runs on simulated data and must be retrained on real historical portfolio data before field deployment.
* **Explainability Thresholds:** Explainability uses heuristic rules. Integrating SHAP/LIME will provide more granular, model-agnostic feature attributions.
* **Decision Support Only:** This system should serve as a recommendation tool for credit officers, not as an automated decision-maker.

