import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter

def load_and_preprocess_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df.drop(['is_fraud', 'timestamp'], axis=1)  # We drop 'timestamp' as it's not useful for modeling
    y = df['is_fraud']
    
    # Convert categorical variables to dummy variables
    X = pd.get_dummies(X, columns=['merchant_category'])
    
    return X, y

def check_missing_values(X):
    missing_values = X.isnull().sum()
    print("Missing values in each column:\n", missing_values[missing_values > 0])

def impute_missing_values(X):
    imputer = SimpleImputer(strategy='mean')  # You can also use 'median' or 'most_frequent'
    X_imputed = imputer.fit_transform(X)
    return pd.DataFrame(X_imputed, columns=X.columns)

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def apply_smote(X_train, y_train, sampling_strategy=0.1, random_state=42):
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def apply_random_undersampling(X_train, y_train, sampling_strategy=0.5, random_state=42):
    undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def apply_smoteenn(X_train, y_train, smote_ratio=0.1, under_ratio=0.5, random_state=42):
    over = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
    under = RandomUnderSampler(sampling_strategy=under_ratio, random_state=random_state)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

# Load and preprocess the data
X, y = load_and_preprocess_data('financial_transactions_engineered.csv')

# Check for missing values
check_missing_values(X)

# Impute missing values if any
X = impute_missing_values(X)

# Split the data
X_train, X_test, y_train, y_test = split_data(X, y)

print("Original dataset shape:", Counter(y))
print("Training set shape:", Counter(y_train))
print("Testing set shape:", Counter(y_test))

# Apply SMOTE
X_smote, y_smote = apply_smote(X_train, y_train)
print("\nSMOTE resampled dataset shape:", Counter(y_smote))

# Apply Random Undersampling
X_undersampled, y_undersampled = apply_random_undersampling(X_train, y_train)
print("Undersampled dataset shape:", Counter(y_undersampled))

# Apply combination of SMOTE and Random Undersampling
X_smoteenn, y_smoteenn = apply_smoteenn(X_train, y_train)
print("SMOTEENN resampled dataset shape:", Counter(y_smoteenn))

# Save the resampled datasets
pd.DataFrame(X_smote).to_csv('X_smote.csv', index=False)
pd.Series(y_smote).to_csv('y_smote.csv', index=False)
pd.DataFrame(X_undersampled).to_csv('X_undersampled.csv', index=False)
pd.Series(y_undersampled).to_csv('y_undersampled.csv', index=False)
pd.DataFrame(X_smoteenn).to_csv('X_smoteenn.csv', index=False)
pd.Series(y_smoteenn).to_csv('y_smoteenn.csv', index=False)

print("\nResampled datasets have been saved to CSV files.")