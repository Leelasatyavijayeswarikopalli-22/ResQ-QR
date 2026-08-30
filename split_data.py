import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the generated dataset
df = pd.read_csv("telemetry_data.csv")

# 2. Separate Features (X) and Target Label (y)
feature_cols = [
    "latency_ms",
    "packet_loss_pct",
    "error_code_category",
    "is_timeout_flag",
    "order_amount_inr",
    "retry_count",
]
X = df[feature_cols]
y = df["action_label"]

# 3. Perform Stratified Train-Test Split (80% Train, 20% Test)
# Stratify ensures target class distribution is balanced across both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Save Train and Test Sets to CSV Files
X_train.assign(action_label=y_train).to_csv("train_data.csv", index=False)
X_test.assign(action_label=y_test).to_csv("test_data.csv", index=False)

print("==================================================")
print("✅ DATA SPLIT SUCCESSFUL")
print("==================================================")
print(f"Total Dataset Size : {len(df)} samples")
print(f"Training Set Size  : {len(X_train)} samples (80%)")
print(f"Testing Set Size   : {len(X_test)} samples (20%)")
print("\nFiles saved: 'train_data.csv' & 'test_data.csv'")