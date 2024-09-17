import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import joblib
from scipy.stats import ks_2samp

def load_new_data():
    # In a real scenario, this would load new transaction data
    # For this example, we'll simulate new data
    return pd.read_csv('x_smoteenn.csv'), pd.read_csv('y_smoteenn.csv')['is_fraud']

def evaluate_model(model, X, y):
    y_pred = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_pred)

def detect_drift(X_ref, X_new, threshold=0.1):
    drift_scores = []
    for column in X_ref.columns:
        ks_statistic, p_value = ks_2samp(X_ref[column], X_new[column])
        drift_scores.append(ks_statistic)
    
    avg_drift = np.mean(drift_scores)
    return avg_drift > threshold, avg_drift

def monitor_model():
    # Load the model
    model = joblib.load('lightgbm_tuned_model.joblib')
    
    # Load reference data (data the model was trained on)
    X_ref = pd.read_csv('X_smoteenn.csv')
    
    # Load new data
    X_new, y_new = load_new_data()
    
    # Evaluate model on new data
    new_score = evaluate_model(model, X_new, y_new)
    print(f"Model performance on new data (ROC AUC): {new_score:.4f}")
    
    # Detect concept drift
    drift_detected, drift_score = detect_drift(X_ref, X_new)
    
    if drift_detected:
        print(f"Concept drift detected! Drift score: {drift_score:.4f}")
        print("Model retraining recommended.")
    else:
        print(f"No significant drift detected. Drift score: {drift_score:.4f}")

if __name__ == "__main__":
    monitor_model()