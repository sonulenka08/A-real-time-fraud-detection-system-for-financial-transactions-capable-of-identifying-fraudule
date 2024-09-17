import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify  # Import Flask and other required components

app = Flask(__name__)

# Load the model
model = joblib.load('lightgbm_tuned_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    
    # Convert data to DataFrame
    df = pd.DataFrame([data])
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Make prediction
    prediction = model.predict_proba(df)[0][1]  # Get probability of class 1 (fraud)
    
    # Return the prediction
    return jsonify({'fraud_probability': float(prediction)})

def preprocess_data(df):
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract hour and day of week from timestamp
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Calculate amount_log
    df['amount_log'] = np.log1p(df['amount'])
    
    # Calculate or provide default values for other features
    df['amount_vs_user_avg'] = 1.0  # Default value, replace with actual calculation if possible
    df['amount_vs_user_median'] = 1.0  # Default value, replace with actual calculation if possible
    df['user_transaction_count'] = 1  # Default value, replace with actual calculation if possible
    df['hours_since_last_transaction'] = 24  # Default value, replace with actual calculation if possible
    df['user_avg_lat'] = df['latitude']
    df['user_avg_lon'] = df['longitude']
    df['distance_from_user_avg'] = 0  # Default value, replace with actual calculation if possible
    df['merchant_category_fraud_rate'] = 0.01  # Default value, replace with actual data if available
    
    # 7-day and 30-day statistics (replace with actual calculations if possible)
    for period in [7, 30]:
        df[f'user_amount_mean_{period}D'] = df['amount']
        df[f'user_amount_std_{period}D'] = 0
        df[f'user_transaction_count_{period}D'] = 1
        df[f'amount_vs_mean_{period}D'] = 1
   
    # Convert categorical features to category dtype
    categorical_features = ['merchant_category', 'transaction_type', 'currency', 'device_type']
    for feature in categorical_features:
        df[feature] = df[feature].astype('category')
    
    # One-hot encode merchant categories (adjust based on your actual categories)
    merchant_categories = ['entertainment', 'grocery', 'online_shopping', 'restaurant', 'travel']
    for category in merchant_categories:
        df[f'merchant_category_{category}'] = (df['merchant_category'] == category).astype(int)
    
    # Ensure all required features are present and in the correct order
    required_features = [
        'amount', 'latitude', 'longitude', 'user_id', 'hour', 'day_of_week', 'is_weekend',
        'amount_log', 'amount_vs_user_avg', 'amount_vs_user_median', 'user_transaction_count',
        'hours_since_last_transaction', 'user_avg_lat', 'user_avg_lon', 'distance_from_user_avg',
        'merchant_category_fraud_rate', 'user_amount_mean_7D', 'user_amount_std_7D',
        'user_transaction_count_7D', 'amount_vs_mean_7D', 'user_amount_mean_30D',
        'user_amount_std_30D', 'user_transaction_count_30D', 'amount_vs_mean_30D',
        'merchant_category_entertainment', 'merchant_category_grocery',
        'merchant_category_online_shopping', 'merchant_category_restaurant',
        'merchant_category_travel'
    ]
    
    return df[required_features]

if __name__ == '__main__':
    app.run(port=5000, debug=True)