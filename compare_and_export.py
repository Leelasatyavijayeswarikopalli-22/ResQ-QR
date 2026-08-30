import time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

FEATURE_COLS = [
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

X_train, y_train = train_df[FEATURE_COLS], train_df["action_label"]
X_test, y_test = test_df[FEATURE_COLS], test_df["action_label"]

# 1. Train Random Forest
print("🌲 Training Random Forest Classifier...")
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

start_rf = time.perf_counter()
rf_clf.fit(X_train, y_train)
rf_preds = rf_clf.predict(X_test)
rf_time = (time.perf_counter() - start_rf) * 1000

rf_acc = accuracy_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds, average="weighted")

# 2. Train XGBoost
print("⚡ Training XGBoost Classifier...")
xgb_clf = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.08,
    random_state=42,
    eval_metric="mlogloss",
)

start_xgb = time.perf_counter()
xgb_clf.fit(X_train, y_train)
xgb_preds = xgb_clf.predict(X_test)
xgb_time = (time.perf_counter() - start_xgb) * 1000

xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_f1 = f1_score(y_test, xgb_preds, average="weighted")

# 3. Benchmarking
print("\n" + "=" * 58)
print("📊 MODEL PERFORMANCE BENCHMARK MATRIX")
print("=" * 58)
results_df = pd.DataFrame(
    {
        "Metric": [
            "Test Accuracy",
            "Weighted F1-Score",
            "Inference Overhead (ms)",
        ],
        "Random Forest": [
            f"{rf_acc * 100:.2f}%",
            f"{rf_f1:.4f}",
            f"{rf_time:.2f} ms",
        ],
        "XGBoost": [
            f"{xgb_acc * 100:.2f}%",
            f"{xgb_f1:.4f}",
            f"{xgb_time:.2f} ms",
        ],
    }
)
print(results_df.to_string(index=False))
print("=" * 58)

# 4. Save Best Model
target_names = ["NO_ACTION", "CONTEXTUAL_NUDGE", "GENERATE_DYNAMIC_QR"]

if xgb_f1 > rf_f1:
    winning_model = xgb_clf
    winner_name = "XGBoost"
    winning_preds = xgb_preds
else:
    winning_model = rf_clf
    winner_name = "Random Forest"
    winning_preds = rf_preds

print(f"\n🏆 WINNING MODEL: {winner_name}")
print("\n--- WINNING MODEL CLASSIFICATION REPORT ---")
print(classification_report(y_test, winning_preds, target_names=target_names))

joblib.dump(winning_model, "model.pkl")

with open("model_meta.txt", "w") as f:
    max_acc = max(rf_acc, xgb_acc) * 100
    f.write(f"Winner: {winner_name}\nAccuracy: {max_acc:.2f}%")

print(f"✅ Saved optimal model ({winner_name}) to 'model.pkl'.")
