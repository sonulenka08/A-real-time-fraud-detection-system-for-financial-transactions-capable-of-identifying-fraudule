import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

def load_data():
    # Load a sample of data for explanation
    return pd.read_csv('X_smoteenn.csv').iloc[:1000]  # Use first 1000 rows for efficiency

def explain_predictions(model, X):
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)
   
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
   
    # Check if shap_values is a single array or a list of arrays
    if not isinstance(shap_values, list):
        shap_values = [shap_values]
   
    # Get feature importances
    feature_importance = np.abs(shap_values[0]).mean(0)
    feature_importance = pd.DataFrame(list(zip(X.columns, feature_importance)), columns=['feature', 'importance'])
    feature_importance = feature_importance.sort_values('importance', ascending=False)
   
    return shap_values, feature_importance

def get_explanation(model, X, index):
    shap_values, feature_importance = explain_predictions(model, X)
   
    # Get the SHAP values for the specific instance
    instance_shap = pd.DataFrame(list(zip(X.columns, shap_values[0][index])), columns=['feature', 'shap_value'])
    instance_shap = instance_shap.sort_values('shap_value', key=abs, ascending=False)
   
    return instance_shap, feature_importance

if __name__ == "__main__":
    # Load model and data
    model = joblib.load('lightgbm_tuned_model.joblib')
    X = load_data()
   
    # Get explanation for the first transaction
    instance_explanation, overall_importance = get_explanation(model, X, 0)
   
    print("Top 10 features influencing this prediction:")
    print(instance_explanation.head(10))
   
    print("\nOverall top 10 most important features:")
    print(overall_importance.head(10))
   
    # Create and save SHAP summary plot
    shap_values, _ = explain_predictions(model, X)
    shap.summary_plot(shap_values[0], X, plot_type="bar", show=False)
    plt.savefig('shap_summary_plot.png')
    print("\nSHAP summary plot saved as 'shap_summary_plot.png'")