import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib

class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)  # Fit with training data
        return self
    
    def predict(self, X):
        return (self.model.predict(X) >= 0.5).astype(int)  # Convert probabilities to binary predictions
    
    def predict_proba(self, X):
        # Ensure that we return a 2D array with shape (n_samples, n_classes)
        return np.hstack((1 - self.model.predict(X), self.model.predict(X)))  # Returning probabilities for both classes

def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(X, y):
    """Preprocess the data by splitting and scaling."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_lightgbm(X_train, y_train):
    """Train a LightGBM classifier."""
    model = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def create_neural_network(input_dim):
    """Create a neural network model."""
    model = Sequential([
        Input(shape=(input_dim,)),  # Use Input layer as the first layer
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_neural_network(X_train, y_train):
    """Train the neural network model."""
    model = create_neural_network(X_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32)  # Train the neural network
    return model

def create_ensemble(lightgbm_model, nn_model):
    """Create an ensemble of LightGBM and neural network models."""
    return VotingClassifier(
        estimators=[
            ('lightgbm', lightgbm_model),
            ('neural_network', KerasClassifier(nn_model))
        ],
        voting='soft'
    )

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate the performance of the given model."""
    
    if hasattr(model, 'predict_proba'):
        # For LightGBM and VotingClassifier
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for class 1
        y_pred = (y_pred_proba >= 0.5).astype(int)  # Convert probabilities to binary predictions
    else:
        # For Keras Sequential model
        y_pred_proba = model.predict(X_test).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)  # Convert probabilities to binary predictions
    
    print(f"\n{model_name} Performance:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Load the data
X = load_data('X_smoteenn.csv')
y = load_data('y_smoteenn.csv')['is_fraud']

# Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(X,y)

# Train LightGBM model
lightgbm_model = train_lightgbm(X_train,y_train)
evaluate_model(lightgbm_model,X_test,y_test,"LightGBM")

# Train Neural Network model
nn_model = train_neural_network(X_train,y_train)
evaluate_model(nn_model,X_test,y_test,"Neural Network")

# Create and train ensemble model
ensemble_model = create_ensemble(lightgbm_model , nn_model)
ensemble_model.fit(X_train,y_train)
evaluate_model(ensemble_model,X_test,y_test,"Ensemble Model")

# Save models
joblib.dump(lightgbm_model,'lightgbm_model.joblib')
joblib.dump(ensemble_model,'ensemble_model.joblib')
nn_model.save('neural_network_model.h5')

print("\nModels have been saved.")