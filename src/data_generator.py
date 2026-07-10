import os
import numpy as np
import pandas as pd

def generate_synthetic_data(num_records=1200, seed=42):
    np.random.seed(seed)
    
    # 1. Demographics
    gender = np.random.choice(['Female', 'Male'], size=num_records, p=[0.60, 0.40]) # 60% female in microfinance
    age = np.random.randint(18, 70, size=num_records)
    marital_status = np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], size=num_records, p=[0.3, 0.5, 0.15, 0.05])
    dependents = np.random.randint(0, 6, size=num_records)
    
    # 2. Financial Details
    # Monthly income for microfinance clients
    monthly_income = np.random.lognormal(mean=6.5, sigma=0.5, size=num_records) # median ~665, range ~150 to ~3000
    monthly_income = np.clip(monthly_income, 150, 3500).round(2)
    
    # Savings balance: highly correlated with income but with some zeros
    has_savings = np.random.choice([0, 1], size=num_records, p=[0.25, 0.75])
    savings_balance = has_savings * np.random.exponential(scale=monthly_income * 0.8, size=num_records)
    savings_balance = savings_balance.round(2)
    
    # Debt-to-income (DTI) ratio
    dti_ratio = np.random.beta(a=2, b=5, size=num_records) * 0.8
    dti_ratio = dti_ratio.round(4)
    
    # 3. Loan Details
    loan_amount = monthly_income * np.random.uniform(1.5, 4.0, size=num_records)
    loan_amount = np.clip(loan_amount, 100, 8000).round(0)
    
    loan_term_months = np.random.choice([3, 6, 12, 18, 24], size=num_records, p=[0.1, 0.3, 0.4, 0.1, 0.1])
    
    # Interest rate
    interest_rate = (0.36 - 0.12 * (monthly_income / 3500) + np.random.normal(0, 0.02, size=num_records))
    interest_rate = np.clip(interest_rate, 0.10, 0.40).round(4)
    
    # 4. Credit History
    credit_score = np.random.normal(loc=600, scale=80, size=num_records).astype(int)
    credit_score = np.clip(credit_score, 300, 850)
    
    # Repayment history ratio: fraction of past payments made on time (0.0 to 1.0)
    repayment_history_ratio = np.random.beta(a=8, b=1.5, size=num_records)
    repayment_history_ratio = repayment_history_ratio.round(4)
    
    # 5. Target Variable Generation: delinquent (0 or 1)
    # Define a risk score with stronger, centered correlations to make the ML model perform well (ROC-AUC ~0.80+)
    # We add a subtle historical bias related to Gender (Males have +0.4 risk score) to showcase fairness checking
    risk_score = (
        -0.015 * (credit_score - 600) 
        - 0.001 * (savings_balance - 500) 
        + 6.0 * (dti_ratio - 0.3) 
        + 0.0005 * (loan_amount - 1500) 
        - 10.0 * (repayment_history_ratio - 0.85) 
        + 0.40 * (gender == 'Male') 
        + 0.30 * (age < 25)
        + 0.20 * dependents
        + np.random.normal(loc=-1.8, scale=0.8, size=num_records) # centered to yield ~15-20% default rate
    )
    
    # Sigmoid function to convert risk_score to probability
    delinquency_probability = 1 / (1 + np.exp(-risk_score))
    
    # Binary class assignment
    delinquent = (np.random.rand(num_records) < delinquency_probability).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Borrower_ID': [f'BRW_{i:04d}' for i in range(1, num_records + 1)],
        'Age': age,
        'Gender': gender,
        'Marital_Status': marital_status,
        'Dependents': dependents,
        'Monthly_Income': monthly_income,
        'Savings_Balance': savings_balance,
        'Debt_to_Income_Ratio': dti_ratio,
        'Loan_Amount': loan_amount,
        'Loan_Term_Months': loan_term_months,
        'Interest_Rate': interest_rate,
        'Credit_Score': credit_score,
        'Repayment_History_Ratio': repayment_history_ratio,
        'delinquent': delinquent
    })
    
    return df

if __name__ == '__main__':
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Generate data
    df = generate_synthetic_data(num_records=1200, seed=42)
    
    # Save to CSV
    csv_path = os.path.join('data', 'microfinance_loans.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Dataset generated successfully at: {csv_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Delinquency rate: {df['delinquent'].mean():.2%}")
    print(f"Class count: \n{df['delinquent'].value_counts()}")
