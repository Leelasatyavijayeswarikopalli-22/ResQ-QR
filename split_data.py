import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# RESQ-QR DATA SPLITTING
# ============================================================
#
# DATASET LABELS
#
# 1 = GATEWAY DEGRADATION
#     -> CONTEXTUAL_NUDGE
#
# 2 = BANK DEGRADATION
#     -> NO_ACTION
#
# 3 = NETWORK DEGRADATION
#     -> GENERATE_DYNAMIC_QR
#
# ============================================================


RANDOM_STATE = 42


# ============================================================
# REGRESSION FEATURES
# ============================================================
#
# ONLY OBSERVABLE TELEMETRY
#
# IMPORTANT:
#
# The following are deliberately NOT used:
#
# total_latency_ms
# transaction_age_ms
# retry_count
# order_amount_inr
# payment_stage
#
# ============================================================


FEATURE_COLS = [

    # -------------------------
    # NETWORK TELEMETRY
    # -------------------------

    "network_latency_ms",
    "packet_loss_pct",
    "network_jitter_ms",

    # -------------------------
    # GATEWAY TELEMETRY
    # -------------------------

    "gateway_latency_ms",
    "gateway_failure_rate_pct",
    "gateway_timeout_rate_pct",

    # -------------------------
    # BANK TELEMETRY
    # -------------------------

    "bank_latency_ms",
    "bank_failure_rate_pct",
    "bank_timeout_rate_pct",

    # -------------------------
    # ERROR TELEMETRY
    # -------------------------

    "error_code_category",
    "is_timeout_flag",
]


# ============================================================
# REGRESSION TARGETS
# ============================================================

REGRESSION_TARGETS = [

    "true_network_degradation",
    "true_gateway_degradation",
    "true_bank_degradation",
]


# ============================================================
# CLASSIFICATION TARGET
# ============================================================

CLASSIFICATION_TARGET = "action_label"


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING RESQ-QR DATASET")
print("=" * 70)

df = pd.read_csv("telemetry_data.csv")


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\nDataset shape:", df.shape)

if len(df) != 10000:

    raise ValueError(
        "Expected exactly 10,000 samples."
        f" Found {len(df)} samples."
    )


# ============================================================
# VALIDATE CLASS LABELS
# ============================================================

valid_labels = {1, 2, 3}

actual_labels = set(
    df[CLASSIFICATION_TARGET]
    .astype(int)
    .unique()
)

if actual_labels != valid_labels:

    raise ValueError(
        "\nInvalid action labels found.\n"
        f"Expected: {sorted(valid_labels)}\n"
        f"Found: {sorted(actual_labels)}"
    )


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = (

    FEATURE_COLS

    + REGRESSION_TARGETS

    + [CLASSIFICATION_TARGET]
)


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns
]


if missing_columns:

    raise ValueError(

        "\nMissing required columns:\n"

        + "\n".join(missing_columns)
    )


# ============================================================
# CHECK CLASS DISTRIBUTION
# ============================================================

print("\nOriginal classification distribution:")

print(
    df[CLASSIFICATION_TARGET]
    .value_counts()
    .sort_index()
)


# ============================================================
# INPUT / OUTPUT
# ============================================================

X = df[FEATURE_COLS]

y_class = df[CLASSIFICATION_TARGET]

y_reg = df[REGRESSION_TARGETS]


# ============================================================
# 70 / 30 STRATIFIED SPLIT
# ============================================================

(
    X_train,
    X_test,
    y_class_train,
    y_class_test,
    y_reg_train,
    y_reg_test,
) = train_test_split(

    X,

    y_class,

    y_reg,

    test_size=0.30,

    random_state=RANDOM_STATE,

    stratify=y_class,
)


# ============================================================
# CLASSIFICATION TRAIN DATA
# ============================================================

classification_train = X_train.copy()

classification_train[
    "action_label"
] = y_class_train.values


# ============================================================
# CLASSIFICATION TEST DATA
# ============================================================

classification_test = X_test.copy()

classification_test[
    "action_label"
] = y_class_test.values


# ============================================================
# REGRESSION TRAIN DATA
# ============================================================

regression_train = X_train.copy()


for target in REGRESSION_TARGETS:

    regression_train[target] = (
        y_reg_train[target].values
    )


# ============================================================
# REGRESSION TEST DATA
# ============================================================

regression_test = X_test.copy()


for target in REGRESSION_TARGETS:

    regression_test[target] = (
        y_reg_test[target].values
    )


# ============================================================
# SAVE FILES
# ============================================================

classification_train.to_csv(
    "train_data.csv",
    index=False,
)

classification_test.to_csv(
    "test_data.csv",
    index=False,
)

regression_train.to_csv(
    "regression_train_data.csv",
    index=False,
)

regression_test.to_csv(
    "regression_test_data.csv",
    index=False,
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("RESQ-QR DATA SPLIT SUCCESSFUL")
print("=" * 70)

print(
    "\nTotal samples:",
    len(df),
)

print(
    "Training samples:",
    len(X_train),
)

print(
    "Testing samples:",
    len(X_test),
)

print(
    "\nTraining percentage:",
    len(X_train) / len(df) * 100,
)

print(
    "Testing percentage:",
    len(X_test) / len(df) * 100,
)


print("\nRegression features:")

for feature in FEATURE_COLS:

    print("  -", feature)


print("\nRegression feature count:")

print(len(FEATURE_COLS))


print("\nRegression targets:")

for target in REGRESSION_TARGETS:

    print("  -", target)


print("\nClassification labels:")

print("  1 = CONTEXTUAL_NUDGE")
print("  2 = NO_ACTION")
print("  3 = GENERATE_DYNAMIC_QR")


print("\nTraining classification distribution:")

print(
    y_class_train
    .value_counts()
    .sort_index()
)


print("\nTesting classification distribution:")

print(
    y_class_test
    .value_counts()
    .sort_index()
)


print("\nSaved files:")

print("  train_data.csv")
print("  test_data.csv")
print("  regression_train_data.csv")
print("  regression_test_data.csv")

print("\n" + "=" * 70)