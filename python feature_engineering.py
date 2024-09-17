import pandas as pd
import numpy as np
from geopy.distance import great_circle

def engineer_features(df):
    # Convert timestamp to datetime if it's not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Amount-based features
    df['amount_log'] = np.log1p(df['amount'])
    
    # User-based features
    user_avg_amount = df.groupby('user_id')['amount'].transform('mean')
    user_median_amount = df.groupby('user_id')['amount'].transform('median')
    df['amount_vs_user_avg'] = df['amount'] / user_avg_amount
    df['amount_vs_user_median'] = df['amount'] / user_median_amount
    
    # Frequency-based features
    df['user_transaction_count'] = df.groupby('user_id')['timestamp'].transform('count')
    
    time_diff = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600
    df['hours_since_last_transaction'] = time_diff.fillna(0)
    
    # Location-based features
    user_locations = df.groupby('user_id')[['latitude', 'longitude']].mean()
    user_locations.columns = ['user_avg_lat', 'user_avg_lon']
    df = df.merge(user_locations, on='user_id', how='left')
    
    def calculate_distance(row):
        return great_circle((row['latitude'], row['longitude']), 
                            (row['user_avg_lat'], row['user_avg_lon'])).kilometers
    
    df['distance_from_user_avg'] = df.apply(calculate_distance, axis=1)
    
    # Merchant category features
    category_fraud_rate = df.groupby('merchant_category')['is_fraud'].mean()
    df['merchant_category_fraud_rate'] = df['merchant_category'].map(category_fraud_rate)
    
    # Time window features
    def calculate_window_features(window):
        df_sorted = df.sort_values(['user_id', 'timestamp'])
        
        def rolling_window(group, window):
            return group.rolling(window=window, on='timestamp')
        
        user_amount_mean = df_sorted.groupby('user_id').apply(lambda x: rolling_window(x, window)['amount'].mean()).reset_index(level=0, drop=True)
        user_amount_std = df_sorted.groupby('user_id').apply(lambda x: rolling_window(x, window)['amount'].std()).reset_index(level=0, drop=True)
        user_transaction_count = df_sorted.groupby('user_id').apply(lambda x: rolling_window(x, window)['amount'].count()).reset_index(level=0, drop=True)
        
        df[f'user_amount_mean_{window}'] = user_amount_mean
        df[f'user_amount_std_{window}'] = user_amount_std
        df[f'user_transaction_count_{window}'] = user_transaction_count
        df[f'amount_vs_mean_{window}'] = df['amount'] / df[f'user_amount_mean_{window}']
    
    calculate_window_features('7D')
    calculate_window_features('30D')
    
    return df

# Load the data
df = pd.read_csv('financial_transactions.csv')

# Apply feature engineering
df_engineered = engineer_features(df)

# Save the engineered dataset
df_engineered.to_csv('financial_transactions_engineered.csv', index=False)

print(df_engineered.head())
print(f"\nShape of the engineered dataset: {df_engineered.shape}")
print("\nNew features added:")
new_features = set(df_engineered.columns) - set(df.columns)
for feature in new_features:
    print(f"- {feature}")