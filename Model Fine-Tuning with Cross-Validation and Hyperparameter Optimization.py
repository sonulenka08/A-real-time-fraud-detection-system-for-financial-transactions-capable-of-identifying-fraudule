import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform

# Load the data
X = pd.read_csv('X_smoteenn.csv')
y = pd.read_csv('y_smoteenn.csv')['is_fraud']

# Define the parameter space
param_distributions = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'num_leaves': randint(20, 100),
    'min_child_samples': randint(10, 50),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}

# Create the model
model = LGBMClassifier(random_state=42)

# Set up RandomizedSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    model, param_distributions=param_distributions, n_iter=50, 
    cv=cv, scoring='roc_auc', n_jobs=-1, random_state=42, verbose=1
)

# Fit the model
random_search.fit(X, y)

# Print the best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best ROC AUC score:", random_search.best_score_)

# Save the best model
import joblib
joblib.dump(random_search.best_estimator_, 'lightgbm_tuned_model.joblib')

print("Tuned model has been saved.")