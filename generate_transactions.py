import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_transactions(num_transactions, fraud_ratio=0.01):
    np.random.seed(42)
    
    # Generate transaction timestamps
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    timestamps = [start_date + timedelta(seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))) for _ in range(num_transactions)]
    
    # Generate transaction amounts
    amounts = np.exp(np.random.normal(4, 1, num_transactions))
    
    # Generate merchant categories
    categories = np.random.choice(['grocery', 'restaurant', 'entertainment', 'travel', 'online_shopping'], num_transactions)
    
    # Generate locations
    latitudes = np.random.uniform(25, 50, num_transactions)
    longitudes = np.random.uniform(-125, -65, num_transactions)
    
    # Generate user IDs
    user_ids = np.random.randint(1000, 10000, num_transactions)
    
    # Generate fraudulent transactions
    is_fraud = np.random.choice([0, 1], num_transactions, p=[1-fraud_ratio, fraud_ratio])
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'amount': amounts,
        'merchant_category': categories,
        'latitude': latitudes,
        'longitude': longitudes,
        'user_id': user_ids,
        'is_fraud': is_fraud
    })
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

# Generate 100,000 transactions with 1% fraud ratio
transactions_df = generate_transactions(100000, 0.01)

# Save to CSV
transactions_df.to_csv('financial_transactions.csv', index=False)

print(transactions_df.head())
print(f"\nShape of the dataset: {transactions_df.shape}")
print(f"\nFraud ratio: {transactions_df['is_fraud'].mean():.2%}")