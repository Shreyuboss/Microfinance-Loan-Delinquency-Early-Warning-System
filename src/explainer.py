import numpy as np

def generate_reason_codes(borrower_data):
    """
    Generates explanation codes highlighting the key factors driving high risk.
    Arguments:
        borrower_data: dict containing feature names and values.
    Returns:
        reasons: list of strings detailing high-risk drivers.
    """
    reasons = []
    
    # 1. DTI Ratio
    dti = borrower_data.get('Debt_to_Income_Ratio', 0.0)
    if dti > 0.40:
        reasons.append(f"High Debt-to-Income Ratio ({dti*100:.1f}%) exceeds the recommended 40% threshold.")
        
    # 2. Savings Balance vs. Income / Loan Amount
    savings = borrower_data.get('Savings_Balance', 0.0)
    loan_amount = borrower_data.get('Loan_Amount', 1.0)
    monthly_income = borrower_data.get('Monthly_Income', 1.0)
    
    if savings < (loan_amount * 0.10):
        reasons.append(f"Low Savings Cushion: Current savings (${savings:,.2f}) are less than 10% of the loan amount (${loan_amount:,.2f}).")
    elif savings < 100.0:
        reasons.append(f"Critical Savings Shortfall: Savings balance of ${savings:,.2f} leaves little margin for emergencies.")
        
    # 3. Credit Score
    credit_score = borrower_data.get('Credit_Score', 600)
    if credit_score < 580:
        reasons.append(f"Subprime Credit Score ({credit_score}): Represents elevated risk based on external historical credit files.")
        
    # 4. Repayment History
    repayment_ratio = borrower_data.get('Repayment_History_Ratio', 1.0)
    if repayment_ratio < 0.90:
        reasons.append(f"Prior Repayment Inconsistencies: Historical timely payment rate of {repayment_ratio*100:.1f}% indicates past defaults or delays.")
        
    # 5. Loan Amount to Income
    if loan_amount > (monthly_income * 3.5):
        reasons.append(f"High Leverage: Loan amount (${loan_amount:,.2f}) is over 3.5 times the monthly income (${monthly_income:,.2f}).")
        
    # 6. Age
    age = borrower_data.get('Age', 35)
    if age < 25:
        reasons.append(f"Young Borrower Profile: Age {age} is associated with potentially shorter employment/credit history.")

    if not reasons:
        reasons.append("No critical risk drivers identified; healthy financial indicators across all features.")
        
    return reasons

def get_risk_tier(probability):
    """
    Classifies default probability into Low, Medium, or High Risk tier.
    """
    if probability < 0.20:
        return 'Low Risk', 'green'
    elif probability < 0.50:
        return 'Medium Risk', 'orange'
    else:
        return 'High Risk', 'red'

def get_recommendations(risk_tier, reasons):
    """
    Generates actionable support and risk-mitigation recommendations.
    """
    recommendations = []
    
    if risk_tier == 'Low Risk':
        recommendations = [
            "**Fast-Track Processing:** Eligible for automated credit approval and immediate disbursement.",
            "**Loyalty Perks:** Qualify for a 1.5% interest rate discount on subsequent loan applications.",
            "**Financial Inclusion:** Encourage participation in advanced business development training."
        ]
    elif risk_tier == 'Medium Risk':
        recommendations = [
            "**SMS Reminders:** Set up automated text alerts 3 days prior to each weekly repayment date.",
            "**Financial Literacy:** Require completion of a 15-minute mobile micro-course on cash-flow budgeting.",
            "**Amortization Adjustment:** Consider reducing the loan term or principal by 15% to buffer debt service capacity."
        ]
    else:  # High Risk
        recommendations = [
            "**Principal Restructuring:** Recommend scaling down the loan size by 30-50% to align with income constraints.",
            "**Early Officer Engagement:** Assign a local credit officer for pre-disbursement orientation and check-ins within 5 days of repayment start.",
            "**Support Structures:** Explore group lending structures (joint liability) or adding a family guarantor.",
            "**Budget Counseling:** Offer free access to an emergency savings coach and micro-savings planning program."
        ]
        
        # Reason-specific tailored recommendations
        for reason in reasons:
            if "Savings" in reason:
                recommendations.append("**Micro-Savings Program:** Enroll borrower in a matched-savings program to rebuild savings habits.")
            if "Debt-to-Income" in reason:
                recommendations.append("**Debt Consolidation:** Verify other active microfinance liabilities and offer restructuring support.")
                
    return recommendations
