import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the data
df = pd.read_csv('financial_transactions.csv')

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Basic statistics
print(df.describe())

# Correlation matrix
plt.figure(figsize=(10, 8))
# Select numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Calculate correlation matrix
corr_matrix = df[numeric_cols].corr()

# Create heatmap
plt.title('Correlation Matrix for Numeric Columns')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix of Transaction Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Distribution of transaction amounts
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='amount', hue='is_fraud', kde=True, element='step')
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount')
plt.ylabel('Count')
plt.legend(['Normal', 'Fraud'])
plt.savefig('amount_distribution.png')
plt.close()

# Fraudulent transactions by merchant category
fraud_by_category = df[df['is_fraud'] == 1]['merchant_category'].value_counts()
plt.figure(figsize=(10, 6))
fraud_by_category.plot(kind='bar')
plt.title('Fraudulent Transactions by Merchant Category')
plt.xlabel('Merchant Category')
plt.ylabel('Number of Fraudulent Transactions')
plt.tight_layout()
plt.savefig('fraud_by_category.png')
plt.close()

# Time series of transactions
df['date'] = df['timestamp'].dt.date
daily_transactions = df.groupby('date').size()
daily_fraud = df[df['is_fraud'] == 1].groupby('date').size()

plt.figure(figsize=(12, 6))
plt.plot(daily_transactions.index, daily_transactions.values, label='All Transactions')
plt.plot(daily_fraud.index, daily_fraud.values, label='Fraudulent Transactions')
plt.title('Daily Transaction Volume')
plt.xlabel('Date')
plt.ylabel('Number of Transactions')
plt.legend()
plt.tight_layout()
plt.savefig('daily_transactions.png')
plt.close()

# Geospatial distribution of fraudulent transactions
plt.figure(figsize=(12, 8))
plt.scatter(df[df['is_fraud'] == 0]['longitude'], df[df['is_fraud'] == 0]['latitude'], 
            alpha=0.5, s=1, c='blue', label='Normal')
plt.scatter(df[df['is_fraud'] == 1]['longitude'], df[df['is_fraud'] == 1]['latitude'], 
            alpha=0.5, s=5, c='red', label='Fraud')
plt.title('Geospatial Distribution of Transactions')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.tight_layout()
plt.savefig('geospatial_distribution.png')
plt.close()

# Statistical tests
normal_amounts = df[df['is_fraud'] == 0]['amount']
fraud_amounts = df[df['is_fraud'] == 1]['amount']
t_statistic, p_value = stats.ttest_ind(normal_amounts, fraud_amounts)
print(f"T-test for difference in transaction amounts:")
print(f"T-statistic: {t_statistic}, P-value: {p_value}")

# Output key findings
fraud_ratio = df['is_fraud'].mean()
avg_fraud_amount = df[df['is_fraud'] == 1]['amount'].mean()
avg_normal_amount = df[df['is_fraud'] == 0]['amount'].mean()

print(f"\nKey Findings:")
print(f"Fraud ratio: {fraud_ratio:.2%}")
print(f"Average fraudulent transaction amount: ${avg_fraud_amount:.2f}")
print(f"Average normal transaction amount: ${avg_normal_amount:.2f}")
print(f"Most common merchant category for fraud: {fraud_by_category.index[0]}")