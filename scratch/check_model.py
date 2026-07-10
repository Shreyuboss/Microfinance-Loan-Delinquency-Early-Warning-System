import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from src.modeling import load_data, preprocess_and_split

df = load_data()
X_train, X_test, y_train, y_test, preprocessor, numeric_features, categorical_features = preprocess_and_split(df)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

with open('models/balanced_model.pkl', 'rb') as f:
    model = pickle.load(f)

y_pred = model.predict(X_test_processed)
y_prob = model.predict_proba(X_test_processed)[:, 1]

print("Actual delinquency rate in test set:", y_test.mean())
print("Predicted delinquency rate in test set:", y_pred.mean())
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Max predicted probability:", y_prob.max())
print("Min predicted probability:", y_prob.min())
print("Mean predicted probability:", y_prob.mean())

# Print sample probabilities for positive cases
print("\nSample probabilities for actual delinquents:")
print(pd.DataFrame({'Actual': y_test, 'Prob': y_prob})[y_test == 1].head(10))

print("\nSample probabilities for actual non-delinquents:")
print(pd.DataFrame({'Actual': y_test, 'Prob': y_prob})[y_test == 0].head(10))
