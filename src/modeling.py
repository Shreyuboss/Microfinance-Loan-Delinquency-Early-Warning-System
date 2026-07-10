import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def load_data(filepath='data/microfinance_loans.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Run src/data_generator.py first.")
    return pd.read_csv(filepath)

def preprocess_and_split(df):
    # Features and Target
    X = df.drop(columns=['Borrower_ID', 'delinquent'])
    y = df['delinquent']
    
    # Identify feature types
    numeric_features = ['Age', 'Dependents', 'Monthly_Income', 'Savings_Balance', 
                        'Debt_to_Income_Ratio', 'Loan_Amount', 'Loan_Term_Months', 
                        'Interest_Rate', 'Credit_Score', 'Repayment_History_Ratio']
    categorical_features = ['Gender', 'Marital_Status']
    
    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])
    
    # Split train-test (80/20) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, preprocessor, numeric_features, categorical_features

def compute_fairness_metrics(y_true, y_pred, df_test):
    """
    Computes fairness metrics for the 'Gender' attribute on the test set.
    For loan applications:
    - Selection Rate is the rate at which borrowers are flagged as 'Low/Medium Risk' (delinquent = 0 predicted).
    - Disparate Impact Ratio (DIR): Selection Rate of unprivileged group / Selection Rate of privileged group.
    - Demographic Parity Difference (DPD): |Selection Rate (Female) - Selection Rate (Male)|.
    """
    df_fairness = df_test.copy()
    # Predicted 'selection' is 1 if pred is 0 (approved / low risk), 0 if pred is 1 (delinquent / high risk)
    df_fairness['approved'] = (y_pred == 0).astype(int)
    
    groups = df_fairness.groupby('Gender')['approved'].agg(['count', 'sum', 'mean'])
    
    female_selection_rate = groups.loc['Female', 'mean']
    male_selection_rate = groups.loc['Male', 'mean']
    
    # Let's define the group with lower selection rate as unprivileged
    if female_selection_rate < male_selection_rate:
        unprivileged_group = 'Female'
        privileged_group = 'Male'
        dir_value = female_selection_rate / male_selection_rate
    else:
        unprivileged_group = 'Male'
        privileged_group = 'Female'
        dir_value = male_selection_rate / female_selection_rate
        
    dpd_value = abs(female_selection_rate - male_selection_rate)
    
    return {
        'Female_Selection_Rate': float(female_selection_rate),
        'Male_Selection_Rate': float(male_selection_rate),
        'Unprivileged_Group': unprivileged_group,
        'Privileged_Group': privileged_group,
        'Disparate_Impact_Ratio': float(dir_value),
        'Demographic_Parity_Difference': float(dpd_value)
    }

def train_and_evaluate():
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    os.makedirs('docs/images', exist_ok=True)
    
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor, numeric_features, categorical_features = preprocess_and_split(df)
    
    # Fit preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 1. Train Baseline Model (No Class Imbalance Handling)
    baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline_model.fit(X_train_processed, y_train)
    y_pred_baseline = baseline_model.predict(X_test_processed)
    
    # 2. Train Balanced Model (With Class Imbalance Handling using Class Weighting)
    balanced_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    balanced_model.fit(X_train_processed, y_train)
    y_pred_balanced = balanced_model.predict(X_test_processed)
    y_prob_balanced = balanced_model.predict_proba(X_test_processed)[:, 1]
    
    # Compare Metrics
    metrics = {
        'baseline': {
            'accuracy': float(accuracy_score(y_test, y_pred_baseline)),
            'precision': float(precision_score(y_test, y_pred_baseline)),
            'recall': float(recall_score(y_test, y_pred_baseline)),
            'f1': float(f1_score(y_test, y_pred_baseline)),
            'roc_auc': float(roc_auc_score(y_test, baseline_model.predict_proba(X_test_processed)[:, 1]))
        },
        'balanced': {
            'accuracy': float(accuracy_score(y_test, y_pred_balanced)),
            'precision': float(precision_score(y_test, y_pred_balanced)),
            'recall': float(recall_score(y_test, y_pred_balanced)),
            'f1': float(f1_score(y_test, y_pred_balanced)),
            'roc_auc': float(roc_auc_score(y_test, y_prob_balanced))
        }
    }
    
    # Get original feature names after encoding
    encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = list(numeric_features) + list(encoded_cat_names)
    
    # Feature Importances for Balanced Model
    importances = balanced_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Save Feature Importance plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Random Forest Feature Importance (Balanced Model)')
    plt.tight_layout()
    plt.savefig('docs/images/feature_importance.png')
    plt.close()
    
    # Save Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred_balanced)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Delinquent', 'Delinquent'],
                yticklabels=['Non-Delinquent', 'Delinquent'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - Balanced Model')
    plt.tight_layout()
    plt.savefig('docs/images/confusion_matrix.png')
    plt.close()
    
    # Compute fairness metrics on the original test set slice
    df_test = X_test.copy()
    df_test['delinquent'] = y_test
    fairness_results = compute_fairness_metrics(y_test, y_pred_balanced, df_test)
    
    # Save results dictionary
    pipeline_results = {
        'metrics': metrics,
        'fairness': fairness_results,
        'feature_importance': feat_imp_df.to_dict(orient='records')
    }
    
    with open('models/pipeline_results.json', 'w') as f:
        json.dump(pipeline_results, f, indent=4)
        
    # Serialize Preprocessor & Model
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    with open('models/balanced_model.pkl', 'wb') as f:
        pickle.dump(balanced_model, f)
        
    print("Model Training & Evaluation complete!")
    print(f"Baseline F1-Score: {metrics['baseline']['f1']:.4f} | Recall: {metrics['baseline']['recall']:.4f}")
    print(f"Balanced F1-Score: {metrics['balanced']['f1']:.4f} | Recall: {metrics['balanced']['recall']:.4f}")
    print(f"Disparate Impact Ratio: {fairness_results['Disparate_Impact_Ratio']:.4f}")
    print(f"Demographic Parity Difference: {fairness_results['Demographic_Parity_Difference']:.4f}")

if __name__ == '__main__':
    train_and_evaluate()
