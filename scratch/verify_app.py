import os
import sys
import pickle
import pandas as pd

# Adjust path to find src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.explainer import generate_reason_codes, get_risk_tier, get_recommendations

def test_inference_pipeline():
    print("Starting pipeline verification test...")
    
    # 1. Check if files exist
    files = [
        'models/preprocessor.pkl',
        'models/balanced_model.pkl',
        'models/pipeline_results.json',
        'data/microfinance_loans.csv'
    ]
    for file in files:
        if os.path.exists(file):
            print(f"OK File found: {file}")
        else:
            print(f"ERROR File missing: {file}")
            sys.exit(1)
            
    # 2. Load model assets
    print("Loading preprocessor and model...")
    with open('models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open('models/balanced_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("OK Model assets loaded successfully!")
    
    # 3. Create dummy client data (High risk case)
    high_risk_client = {
        'Age': 22,
        'Gender': 'Male',
        'Marital_Status': 'Single',
        'Dependents': 3,
        'Monthly_Income': 400.0,
        'Savings_Balance': 20.0,
        'Debt_to_Income_Ratio': 0.65,
        'Loan_Amount': 2000.0,
        'Loan_Term_Months': 24,
        'Interest_Rate': 0.32,
        'Credit_Score': 420,
        'Repayment_History_Ratio': 0.72
    }
    
    # 4. Preprocess inputs
    df_client = pd.DataFrame([high_risk_client])
    client_processed = preprocessor.transform(df_client)
    
    # 5. Predict default probability
    prob = model.predict_proba(client_processed)[0][1]
    risk_tier, badge_color = get_risk_tier(prob)
    
    print(f"\n--- Model Prediction Test ---")
    print(f"Client Profile: Age=22, Income=$400, Savings=$20, DTI=65%, Loan=$2000, Credit Score=420")
    print(f"Result Probability: {prob:.4f}")
    print(f"Risk Tier: {risk_tier} (Color: {badge_color})")
    
    # 6. Generate explanations
    reasons = generate_reason_codes(high_risk_client)
    print(f"\n--- Reason Codes ---")
    for reason in reasons:
        print(f"- {reason}")
        
    # 7. Generate recommendations
    recs = get_recommendations(risk_tier, reasons)
    print(f"\n--- Support Recommendations ---")
    for rec in recs:
        print(f"- {rec}")
        
    print("\nOK Inference pipeline verification test completed successfully!")

if __name__ == '__main__':
    test_inference_pipeline()
