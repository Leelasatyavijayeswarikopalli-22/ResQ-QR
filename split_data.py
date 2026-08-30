import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load Dataset
df = pd.read_csv("telemetry_data.csv")

# 2. Feature Columns
feature_cols = [
    "gateway_latency_ms",
    "bank_latency_ms",
    "total_latency_ms",
    "packet_loss_pct",
    "payment_stage",
    "error_code_category",
    "is_timeout_flag",
    "order_amount_inr",
    "retry_count",
]

X = df[feature_cols]
y = df["action_label"]

# 3. Stratified Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Save CSVs
X_train.assign(action_label=y_train).to_csv("train_data.csv", index=False)
X_test.assign(action_label=y_test).to_csv("test_data.csv", index=False)

print("==================================================")
print("✅ DATA SPLIT SUCCESSFUL")
print("==================================================")
print(f"Total Dataset Size : {len(df)} samples")
print(f"Training Set Size  : {len(X_train)} samples (80%)")
print(f"Testing Set Size   : {len(X_test)} samples (20%)")
print("\nFiles saved: 'train_data.csv' & 'test_data.csv'")