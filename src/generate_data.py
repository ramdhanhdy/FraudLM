import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_synthetic_data(n_users=1000, n_transactions=50000, fraud_rate=0.02):
    """
    Generates a synthetic fraud dataset with patterns that ML models can pick up.
    """
    np.random.seed(42)
    random.seed(42)
    
    print(f"Generating {n_transactions} transactions for {n_users} users...")
    
    # User profiles
    users = [f"user_{i}" for i in range(n_users)]
    countries = ['US', 'UK', 'CA', 'FR', 'DE', 'JP', 'AU']
    user_countries = {u: np.random.choice(countries, p=[0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05]) for u in users}
    user_avg_spend = {u: np.random.lognormal(3, 0.5) for u in users}  # roughly $20 avg
    
    data = []
    
    start_time = datetime.now() - timedelta(days=30)
    
    for _ in range(n_transactions):
        user = np.random.choice(users)
        is_fraud = 0
        
        # Base transaction details
        amount = np.random.lognormal(np.log(user_avg_spend[user]), 0.5)
        timestamp = start_time + timedelta(seconds=np.random.randint(0, 30*24*3600))
        country = user_countries[user]
        device_id = f"dev_{user}_{np.random.randint(0, 3)}" # Mostly stable devices
        merchant_cat = np.random.choice(['groceries', 'entertainment', 'travel', 'utilities', 'luxury'])
        
        # Inject Fraud Patterns
        # 1. High Amount Fraud
        if np.random.random() < fraud_rate:
            is_fraud = 1
            fraud_type = np.random.choice(['high_amount', 'new_country', 'rapid_fire'])
            
            if fraud_type == 'high_amount':
                amount = amount * np.random.uniform(5, 20) # Much higher than avg
            
            elif fraud_type == 'new_country':
                country = np.random.choice([c for c in countries if c != user_countries[user]])
                
            elif fraud_type == 'rapid_fire':
                # We'll handle velocity features in FE, but here we just mark it
                # In a real simulator, we'd generate multiple txns close together.
                # For simplicity in this row-based gen, we just flag it.
                pass
                
            # New device often accompanies fraud
            if np.random.random() < 0.7:
                device_id = f"dev_new_{np.random.randint(1000, 9999)}"
                
        data.append({
            'transaction_id': f"tx_{np.random.randint(1000000, 9999999)}",
            'timestamp': timestamp,
            'user_id': user,
            'amount': round(amount, 2),
            'currency': 'USD',
            'merchant_category': merchant_cat,
            'country': country,
            'device_id': device_id,
            'is_fraud': is_fraud
        })
        
    df = pd.DataFrame(data)
    
    # Sort by time for realistic feature engineering later
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Ensure output directory exists
    os.makedirs('data', exist_ok=True)
    output_path = 'data/transactions.csv'
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(f"Fraud Rate: {df['is_fraud'].mean():.4f}")
    return df

if __name__ == "__main__":
    generate_synthetic_data()
