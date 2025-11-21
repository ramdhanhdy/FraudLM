import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, roc_curve, confusion_matrix

# Ensure directories exist
os.makedirs('src/model/artifacts', exist_ok=True)

def load_and_engineer_features(filepath='data/transactions.csv'):
    """
    Loads raw data and creates features for fraud detection.
    """
    print("Loading data...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    print("Engineering features...")
    
    # 1. Velocity Features (Number of transactions in last X time)
    # We use rolling windows grouped by user
    df_grouped = df.groupby('user_id').rolling(window='24h', on='timestamp')
    
    # Count tx in last 24h
    # Note: rolling produces a multi-index, we need to reset or map back
    # A simpler way for pandas < 2 or general stability:
    # Self-join or iterate is slow. Let's use a simple approximation or efficient rolling.
    
    # Efficient rolling count per user
    # We sort by user and time first
    df = df.sort_values(['user_id', 'timestamp'])
    
    # Feature: Hours since last transaction
    df['time_diff'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600
    df['time_diff'] = df['time_diff'].fillna(24) # Fill first with 24h
    
    # Feature: Amount ratio vs User Average (Expanding mean to prevent leakage from future)
    df['user_avg_amt_hist'] = df.groupby('user_id')['amount'].expanding().mean().reset_index(level=0, drop=True)
    df['amount_vs_avg'] = df['amount'] / df['user_avg_amt_hist']
    
    # Feature: Country Change (1 if different from last, 0 otherwise)
    df['prev_country'] = df.groupby('user_id')['country'].shift(1)
    df['country_change'] = (df['country'] != df['prev_country']).astype(int)
    # Handle first txn
    df.loc[df['prev_country'].isna(), 'country_change'] = 0
    
    # One-Hot Encoding for categorical variables
    # For production, we'd save the encoder. Here we use get_dummies for simplicity 
    # but we must align columns in inference. We'll stick to numericals + simple encodings for MVP.
    
    # Simple frequency encoding for Merchant Category
    cat_freq = df['merchant_category'].value_counts(normalize=True)
    df['merchant_cat_freq'] = df['merchant_category'].map(cat_freq)
    
    # Drop non-feature columns
    features = ['amount', 'time_diff', 'amount_vs_avg', 'country_change', 'merchant_cat_freq']
    target = 'is_fraud'
    
    # Clean up NAs created by shifts/rolling
    df = df.fillna(0)
    
    return df[features], df[target]

def train_model():
    X, y = load_and_engineer_features()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training XGBoost on {X_train.shape[0]} records...")
    
    # Train
    clf = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=sum(y==0)/sum(y==1), # Handle imbalance
        random_state=42,
        eval_metric='auc'
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cls_report = classification_report(y_test, y_pred)

    print("ROCAUC:", roc_auc)
    print("PR AUC:", pr_auc)
    print(cls_report)

    # Compute additional metrics for dashboard
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
        "class_report": cls_report,
    }

    with open("src/model/artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Train SHAP Explainer
    # We use TreeExplainer which is fast for trees
    print("Generating SHAP explainer...")
    explainer = shap.TreeExplainer(clf)
    
    # Save Artifacts
    print("Saving artifacts...")
    joblib.dump(clf, 'src/model/artifacts/model.joblib')
    joblib.dump(explainer, 'src/model/artifacts/shap_explainer.joblib')
    
    # Save column names to ensure inference matches training
    joblib.dump(X.columns.tolist(), 'src/model/artifacts/features.joblib')
    
    print("Done!")

if __name__ == "__main__":
    train_model()
