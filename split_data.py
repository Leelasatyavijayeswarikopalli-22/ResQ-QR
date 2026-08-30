import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# RESQ-QR DATA SPLITTING
# ============================================================


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(
    "telemetry_data.csv"
)


# ------------------------------------------------------------
# 2. CLASSIFICATION FEATURES
# ------------------------------------------------------------

FEATURE_COLS = [

    # Network
    "network_latency_ms",
    "packet_loss_pct",
    "network_jitter_ms",

    # Gateway
    "gateway_latency_ms",
    "gateway_failure_rate_pct",
    "gateway_timeout_rate_pct",

    # Bank
    "bank_latency_ms",
    "bank_failure_rate_pct",
    "bank_timeout_rate_pct",

    # Transaction
    "total_latency_ms",
    "payment_stage",
    "transaction_age_ms",
    "retry_count",
    "order_amount_inr",

    # Error
    "error_code_category",
    "is_timeout_flag",
]


# ------------------------------------------------------------
# 3. REGRESSION TARGETS
# ------------------------------------------------------------

REGRESSION_TARGETS = [

    "true_network_degradation",
    "true_gateway_degradation",
    "true_bank_degradation",
]


# ------------------------------------------------------------
# 4. CLASSIFICATION TARGET
# ------------------------------------------------------------

CLASSIFICATION_TARGET = "action_label"


X = df[FEATURE_COLS]

y_class = df[CLASSIFICATION_TARGET]

y_reg = df[REGRESSION_TARGETS]


# ------------------------------------------------------------
# 5. STRATIFIED SPLIT
#
# We split everything using the same indices so that
# classification and regression use identical train/test rows.
# ------------------------------------------------------------

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = (
    train_test_split(
        X,
        y_class,
        y_reg,
        test_size=0.20,
        random_state=42,
        stratify=y_class,
    )
)


# ------------------------------------------------------------
# 6. SAVE CLASSIFICATION DATA
# ------------------------------------------------------------

classification_train = X_train.copy()

classification_train[
    "action_label"
] = y_class_train.values

classification_test = X_test.copy()

classification_test[
    "action_label"
] = y_class_test.values


classification_train.to_csv(
    "train_data.csv",
    index=False,
)

classification_test.to_csv(
    "test_data.csv",
    index=False,
)


# ------------------------------------------------------------
# 7. SAVE REGRESSION DATA
# ------------------------------------------------------------

regression_train = X_train.copy()

for target in REGRESSION_TARGETS:

    regression_train[target] = (
        y_reg_train[target].values
    )


regression_test = X_test.copy()

for target in REGRESSION_TARGETS:

    regression_test[target] = (
        y_reg_test[target].values
    )


regression_train.to_csv(
    "regression_train_data.csv",
    index=False,
)

regression_test.to_csv(
    "regression_test_data.csv",
    index=False,
)


# ------------------------------------------------------------
# 8. PRINT INFORMATION
# ------------------------------------------------------------

print("=" * 70)
print("RESQ-QR DATA SPLIT SUCCESSFUL")
print("=" * 70)

print(
    f"Total Dataset Size : {len(df)} samples"
)

print(
    f"Training Set Size  : {len(X_train)} samples (80%)"
)

print(
    f"Testing Set Size   : {len(X_test)} samples (20%)"
)


print("\nClassification files:")
print("  train_data.csv")
print("  test_data.csv")


print("\nRegression files:")
print("  regression_train_data.csv")
print("  regression_test_data.csv")


print("\nFeature count:")
print(len(FEATURE_COLS))


print("\nRegression targets:")

for target in REGRESSION_TARGETS:
    print(f"  - {target}")


print("\nClassification distribution:")

print(
    y_class.value_counts(
        normalize=True
    ).sort_index() * 100
)