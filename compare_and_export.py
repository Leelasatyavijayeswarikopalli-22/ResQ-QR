import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from xgboost import (
    XGBClassifier,
    XGBRegressor,
)


# ============================================================
# RESQ-QR MODEL TRAINING + BENCHMARKING
# ============================================================

RANDOM_STATE = 42


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLS = [

    # -------------------------
    # Network
    # -------------------------
    "network_latency_ms",
    "packet_loss_pct",
    "network_jitter_ms",

    # -------------------------
    # Gateway
    # -------------------------
    "gateway_latency_ms",
    "gateway_failure_rate_pct",
    "gateway_timeout_rate_pct",

    # -------------------------
    # Bank
    # -------------------------
    "bank_latency_ms",
    "bank_failure_rate_pct",
    "bank_timeout_rate_pct",

    # -------------------------
    # Transaction
    # -------------------------
    "total_latency_ms",
    "payment_stage",
    "transaction_age_ms",
    "retry_count",
    "order_amount_inr",

    # -------------------------
    # Error
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
# ACTION NAMES
# ============================================================

ACTION_NAMES = [
    "NO_ACTION",
    "CONTEXTUAL_NUDGE",
    "RETRY_PAYMENT",
    "GENERATE_DYNAMIC_QR",
]


# ============================================================
# HELPER FUNCTION
# BENCHMARK INFERENCE LATENCY
# ============================================================

def benchmark_inference(
    model,
    X_test,
    n_runs=500,
):
    """
    Measures model inference latency.

    We use a batch of up to 100 samples and calculate
    latency per individual sample.

    Returns:
        average
        p50
        p95
        p99
    """

    latencies = []

    sample_count = min(
        100,
        len(X_test),
    )

    sample = X_test.iloc[
        :sample_count
    ]

    # -------------------------
    # Warm-up
    # -------------------------

    model.predict(
        sample.iloc[:1]
    )

    # -------------------------
    # Benchmark
    # -------------------------

    for _ in range(n_runs):

        start = time.perf_counter()

        model.predict(sample)

        elapsed = (
            time.perf_counter()
            - start
        )

        per_sample_ms = (
            elapsed
            / sample_count
            * 1000
        )

        latencies.append(
            per_sample_ms
        )

    latencies = np.array(
        latencies
    )

    return {
        "average_ms": float(
            np.mean(latencies)
        ),

        "p50_ms": float(
            np.percentile(
                latencies,
                50,
            )
        ),

        "p95_ms": float(
            np.percentile(
                latencies,
                95,
            )
        ),

        "p99_ms": float(
            np.percentile(
                latencies,
                99,
            )
        ),
    }


# ============================================================
# HELPER FUNCTION
# MODEL SIZE
# ============================================================

def get_model_size_mb(
    model,
    filename,
):

    joblib.dump(
        model,
        filename,
    )

    size_bytes = os.path.getsize(
        filename
    )

    return (
        size_bytes
        / (1024 * 1024)
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)


train_df = pd.read_csv(
    "train_data.csv"
)

test_df = pd.read_csv(
    "test_data.csv"
)

reg_train_df = pd.read_csv(
    "regression_train_data.csv"
)

reg_test_df = pd.read_csv(
    "regression_test_data.csv"
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_classification_columns = (
    FEATURE_COLS
    + ["action_label"]
)

required_regression_columns = (
    FEATURE_COLS
    + REGRESSION_TARGETS
)


missing_classification = [
    col
    for col in required_classification_columns
    if col not in train_df.columns
    or col not in test_df.columns
]


missing_regression = [
    col
    for col in required_regression_columns
    if col not in reg_train_df.columns
    or col not in reg_test_df.columns
]


if missing_classification:

    raise ValueError(
        "\nMissing classification columns:\n"
        + "\n".join(
            missing_classification
        )
    )


if missing_regression:

    raise ValueError(
        "\nMissing regression columns:\n"
        + "\n".join(
            missing_regression
        )
    )


# ============================================================
# CLASSIFICATION DATA
# ============================================================

X_train = train_df[
    FEATURE_COLS
]

X_test = test_df[
    FEATURE_COLS
]

y_train = train_df[
    "action_label"
]

y_test = test_df[
    "action_label"
]


# ============================================================
# REGRESSION DATA
# ============================================================

X_reg_train = reg_train_df[
    FEATURE_COLS
]

X_reg_test = reg_test_df[
    FEATURE_COLS
]

y_reg_train = reg_train_df[
    REGRESSION_TARGETS
]

y_reg_test = reg_test_df[
    REGRESSION_TARGETS
]


# ============================================================
# CLASSIFICATION MODELS
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION MODEL TRAINING")
print("=" * 70)


# ============================================================
# RANDOM FOREST CLASSIFIER
# ============================================================

rf_classifier = RandomForestClassifier(

    n_estimators=250,

    max_depth=10,

    min_samples_leaf=2,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    class_weight="balanced",
)


# ============================================================
# XGBOOST CLASSIFIER
# ============================================================

xgb_classifier = XGBClassifier(

    n_estimators=250,

    max_depth=6,

    learning_rate=0.06,

    subsample=0.9,

    colsample_bytree=0.9,

    random_state=RANDOM_STATE,

    eval_metric="mlogloss",

    n_jobs=-1,
)


classification_models = {

    "Random Forest":
        rf_classifier,

    "XGBoost":
        xgb_classifier,
}


classification_results = {}


# ============================================================
# TRAIN BOTH CLASSIFIERS
# ============================================================

for name, model in (
    classification_models.items()
):

    print(
        f"\nTraining {name}..."
    )

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    training_time = (
        time.perf_counter()
        - start
    ) * 1000

    predictions = (
        model.predict(
            X_test
        )
    )

    accuracy = (
        accuracy_score(
            y_test,
            predictions,
        )
    )

    precision = (
        precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    recall = (
        recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    weighted_f1 = (
        f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    macro_f1 = (
        f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    micro_f1 = (
        f1_score(
            y_test,
            predictions,
            average="micro",
            zero_division=0,
        )
    )

    confusion = (
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    latency = (
        benchmark_inference(
            model,
            X_test,
        )
    )

    filename = (
        "classification_"
        + name.lower().replace(
            " ",
            "_",
        )
        + ".pkl"
    )

    model_size = (
        get_model_size_mb(
            model,
            filename,
        )
    )

    classification_results[
        name
    ] = {

        "accuracy":
            float(accuracy),

        "precision_weighted":
            float(precision),

        "recall_weighted":
            float(recall),

        "weighted_f1":
            float(weighted_f1),

        "macro_f1":
            float(macro_f1),

        "micro_f1":
            float(micro_f1),

        "training_time_ms":
            float(training_time),

        "inference_average_ms":
            latency[
                "average_ms"
            ],

        "inference_p50_ms":
            latency[
                "p50_ms"
            ],

        "inference_p95_ms":
            latency[
                "p95_ms"
            ],

        "inference_p99_ms":
            latency[
                "p99_ms"
            ],

        "model_size_mb":
            float(model_size),

        "confusion_matrix":
            confusion.tolist(),

        "feature_importances":
            model.feature_importances_.tolist(),
    }

    print(
        f"{name} Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"{name} Weighted F1: "
        f"{weighted_f1:.4f}"
    )

    print(
        f"{name} Macro F1: "
        f"{macro_f1:.4f}"
    )

    print(
        f"{name} P95 Inference: "
        f"{latency['p95_ms']:.4f} ms"
    )


# ============================================================
# SELECT BEST CLASSIFICATION MODEL
#
# PRIMARY:
# Macro F1
#
# SECONDARY:
# Weighted F1
#
# Why?
# Every action is important in a payment fallback system.
# Macro F1 gives equal importance to every class.
# ============================================================

rf_result = (
    classification_results[
        "Random Forest"
    ]
)

xgb_result = (
    classification_results[
        "XGBoost"
    ]
)


if (
    xgb_result["macro_f1"],
    xgb_result["weighted_f1"],
) > (
    rf_result["macro_f1"],
    rf_result["weighted_f1"],
):

    best_classifier_name = (
        "XGBoost"
    )

    best_classifier = (
        xgb_classifier
    )

else:

    best_classifier_name = (
        "Random Forest"
    )

    best_classifier = (
        rf_classifier
    )


# ============================================================
# SAVE BEST CLASSIFIER
# ============================================================

joblib.dump(
    best_classifier,
    "classification_model.pkl",
)

joblib.dump(
    best_classifier,
    "model.pkl",
)


# ============================================================
# BEST CLASSIFIER REPORT
# ============================================================

best_predictions = (
    best_classifier.predict(
        X_test
    )
)


print("\n" + "=" * 70)

print(
    "BEST CLASSIFICATION MODEL: "
    + best_classifier_name
)

print("=" * 70)


print(
    classification_report(
        y_test,
        best_predictions,
        target_names=ACTION_NAMES,
        zero_division=0,
    )
)


# ============================================================
# REGRESSION MODELS
# ============================================================

print("\n" + "=" * 70)
print("REGRESSION MODEL TRAINING")
print("=" * 70)


# ============================================================
# RANDOM FOREST REGRESSOR
#
# Random Forest naturally supports
# multiple regression targets.
# ============================================================

rf_regressor = RandomForestRegressor(

    n_estimators=250,

    max_depth=10,

    min_samples_leaf=2,

    random_state=RANDOM_STATE,

    n_jobs=-1,
)


# ============================================================
# XGBOOST REGRESSORS
#
# XGBoost does not directly behave like
# sklearn's multi-output RandomForest here,
# so we train one model for each target.
# ============================================================

xgb_regressors = {}


for target in REGRESSION_TARGETS:

    xgb_regressors[target] = (
        XGBRegressor(

            n_estimators=250,

            max_depth=6,

            learning_rate=0.06,

            subsample=0.9,

            colsample_bytree=0.9,

            random_state=RANDOM_STATE,

            objective="reg:squarederror",

            n_jobs=-1,
        )
    )


# ============================================================
# RANDOM FOREST REGRESSION
# ============================================================

print(
    "\nTraining Random Forest "
    "Regressor..."
)


start = time.perf_counter()


rf_regressor.fit(
    X_reg_train,
    y_reg_train,
)


rf_reg_training_time = (
    time.perf_counter()
    - start
) * 1000


rf_reg_predictions = (
    rf_regressor.predict(
        X_reg_test
    )
)


# ============================================================
# RANDOM FOREST REGRESSION METRICS
# ============================================================

rf_mae = (
    mean_absolute_error(
        y_reg_test,
        rf_reg_predictions,
    )
)

rf_mse = (
    mean_squared_error(
        y_reg_test,
        rf_reg_predictions,
    )
)

rf_rmse = np.sqrt(
    rf_mse
)

rf_r2 = (
    r2_score(
        y_reg_test,
        rf_reg_predictions,
        multioutput="uniform_average",
    )
)


rf_reg_latency = (
    benchmark_inference(
        rf_regressor,
        X_reg_test,
    )
)


rf_reg_size = (
    get_model_size_mb(
        rf_regressor,
        "regression_random_forest.pkl",
    )
)


# ============================================================
# XGBOOST REGRESSION
# ============================================================

print(
    "\nTraining XGBoost "
    "Regressors..."
)


xgb_reg_training_start = (
    time.perf_counter()
)


for target in REGRESSION_TARGETS:

    print(
        f"  Training target: {target}"
    )

    xgb_regressors[
        target
    ].fit(

        X_reg_train,

        y_reg_train[
            target
        ],
    )


xgb_reg_training_time = (
    time.perf_counter()
    - xgb_reg_training_start
) * 1000


# ============================================================
# XGBOOST PREDICTIONS
# ============================================================

xgb_reg_predictions = []


for target in REGRESSION_TARGETS:

    prediction = (
        xgb_regressors[
            target
        ].predict(
            X_reg_test
        )
    )

    xgb_reg_predictions.append(
        prediction
    )


xgb_reg_predictions = (
    np.column_stack(
        xgb_reg_predictions
    )
)


# ============================================================
# XGBOOST REGRESSION METRICS
# ============================================================

xgb_mae = (
    mean_absolute_error(
        y_reg_test,
        xgb_reg_predictions,
    )
)

xgb_mse = (
    mean_squared_error(
        y_reg_test,
        xgb_reg_predictions,
    )
)

xgb_rmse = np.sqrt(
    xgb_mse
)

xgb_r2 = (
    r2_score(
        y_reg_test,
        xgb_reg_predictions,
        multioutput="uniform_average",
    )
)


# ============================================================
# XGBOOST REGRESSION LATENCY
# ============================================================

xgb_latency_values = []


sample_count = min(
    100,
    len(X_reg_test),
)

sample = X_reg_test.iloc[
    :sample_count
]


# Warm-up

for target in REGRESSION_TARGETS:

    xgb_regressors[
        target
    ].predict(
        sample.iloc[:1]
    )


# Benchmark

for _ in range(500):

    start = time.perf_counter()

    for target in REGRESSION_TARGETS:

        xgb_regressors[
            target
        ].predict(
            sample
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    per_sample_ms = (
        elapsed
        / sample_count
        * 1000
    )

    xgb_latency_values.append(
        per_sample_ms
    )


xgb_latency_values = np.array(
    xgb_latency_values
)


xgb_reg_latency = {

    "average_ms":
        float(
            np.mean(
                xgb_latency_values
            )
        ),

    "p50_ms":
        float(
            np.percentile(
                xgb_latency_values,
                50,
            )
        ),

    "p95_ms":
        float(
            np.percentile(
                xgb_latency_values,
                95,
            )
        ),

    "p99_ms":
        float(
            np.percentile(
                xgb_latency_values,
                99,
            )
        ),
}


# ============================================================
# XGBOOST REGRESSION BUNDLE
# ============================================================

xgb_reg_bundle = {

    "models":
        xgb_regressors,

    "targets":
        REGRESSION_TARGETS,

    "features":
        FEATURE_COLS,
}


xgb_reg_size = (
    get_model_size_mb(
        xgb_reg_bundle,
        "regression_xgboost.pkl",
    )
)


# ============================================================
# XGBOOST FEATURE IMPORTANCES
#
# We have three XGBoost models.
# Therefore we calculate the average importance
# of each feature across the three models.
# ============================================================

xgb_importance_matrix = []


for target in REGRESSION_TARGETS:

    importance = (
        xgb_regressors[
            target
        ].feature_importances_
    )

    xgb_importance_matrix.append(
        importance
    )


xgb_importance_matrix = np.array(
    xgb_importance_matrix
)


xgb_average_importance = (
    np.mean(
        xgb_importance_matrix,
        axis=0,
    )
)


# ============================================================
# REGRESSION RESULTS
# ============================================================

regression_results = {

    "Random Forest": {

        "mae":
            float(rf_mae),

        "mse":
            float(rf_mse),

        "rmse":
            float(rf_rmse),

        "r2":
            float(rf_r2),

        "training_time_ms":
            float(
                rf_reg_training_time
            ),

        "inference_average_ms":
            rf_reg_latency[
                "average_ms"
            ],

        "inference_p50_ms":
            rf_reg_latency[
                "p50_ms"
            ],

        "inference_p95_ms":
            rf_reg_latency[
                "p95_ms"
            ],

        "inference_p99_ms":
            rf_reg_latency[
                "p99_ms"
            ],

        "model_size_mb":
            float(rf_reg_size),

        "feature_importances":
            rf_regressor
            .feature_importances_
            .tolist(),
    },


    "XGBoost": {

        "mae":
            float(xgb_mae),

        "mse":
            float(xgb_mse),

        "rmse":
            float(xgb_rmse),

        "r2":
            float(xgb_r2),

        "training_time_ms":
            float(
                xgb_reg_training_time
            ),

        "inference_average_ms":
            xgb_reg_latency[
                "average_ms"
            ],

        "inference_p50_ms":
            xgb_reg_latency[
                "p50_ms"
            ],

        "inference_p95_ms":
            xgb_reg_latency[
                "p95_ms"
            ],

        "inference_p99_ms":
            xgb_reg_latency[
                "p99_ms"
            ],

        "model_size_mb":
            float(xgb_reg_size),

        "feature_importances":
            xgb_average_importance
            .tolist(),
    },
}


# ============================================================
# SELECT BEST REGRESSION MODEL
#
# PRIMARY:
# Higher R2
#
# SECONDARY:
# Lower RMSE
# ============================================================

rf_reg_result = (
    regression_results[
        "Random Forest"
    ]
)

xgb_reg_result = (
    regression_results[
        "XGBoost"
    ]
)


if (
    xgb_reg_result["r2"],
    -xgb_reg_result["rmse"],
) > (
    rf_reg_result["r2"],
    -rf_reg_result["rmse"],
):

    best_regressor_name = (
        "XGBoost"
    )

    best_regressor = (
        xgb_reg_bundle
    )

else:

    best_regressor_name = (
        "Random Forest"
    )

    best_regressor = (
        rf_regressor
    )


# ============================================================
# SAVE BEST REGRESSION MODEL
# ============================================================

joblib.dump(
    best_regressor,
    "regression_model.pkl",
)


# ============================================================
# SAVE ALL BENCHMARK RESULTS
# ============================================================

all_results = {

    "classification":
        classification_results,

    "classification_winner":
        best_classifier_name,

    "regression":
        regression_results,

    "regression_winner":
        best_regressor_name,

    "feature_columns":
        FEATURE_COLS,

    "regression_targets":
        REGRESSION_TARGETS,

    "action_names":
        ACTION_NAMES,

    "selection_logic": {

        "classification_primary":
            "Macro F1",

        "classification_secondary":
            "Weighted F1",

        "regression_primary":
            "R2",

        "regression_secondary":
            "RMSE",
    },
}


with open(
    "benchmark_results.json",
    "w",
) as f:

    json.dump(
        all_results,
        f,
        indent=4,
    )


# ============================================================
# SAVE METADATA
# ============================================================

with open(
    "model_meta.txt",
    "w",
) as f:

    f.write(
        "Classification Winner: "
        + best_classifier_name
        + "\n"
    )

    f.write(
        "Regression Winner: "
        + best_regressor_name
        + "\n"
    )

    f.write(
        "Classification selection: "
        "Macro F1 followed by Weighted F1\n"
    )

    f.write(
        "Regression selection: "
        "R2 followed by RMSE\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print(
    "RESQ-QR FINAL MODEL BENCHMARK"
)

print("=" * 70)


# ============================================================
# CLASSIFICATION TABLE
# ============================================================

classification_table = (
    pd.DataFrame(
        classification_results
    ).T
)


print(
    "\nCLASSIFICATION RESULTS:"
)


print(
    classification_table[
        [
            "accuracy",
            "precision_weighted",
            "recall_weighted",
            "weighted_f1",
            "macro_f1",
            "training_time_ms",
            "inference_average_ms",
            "inference_p95_ms",
            "model_size_mb",
        ]
    ].round(5)
)


# ============================================================
# REGRESSION TABLE
# ============================================================

regression_table = (
    pd.DataFrame(
        regression_results
    ).T
)


print(
    "\nREGRESSION RESULTS:"
)


print(
    regression_table[
        [
            "mae",
            "mse",
            "rmse",
            "r2",
            "training_time_ms",
            "inference_average_ms",
            "inference_p95_ms",
            "model_size_mb",
        ]
    ].round(5)
)


# ============================================================
# WINNERS
# ============================================================

print(
    "\n🏆 Classification Winner:",
    best_classifier_name,
)

print(
    "🏆 Regression Winner:",
    best_regressor_name,
)


# ============================================================
# SAVED FILES
# ============================================================

print(
    "\nSaved files:"
)

print(
    "  classification_random_forest.pkl"
)

print(
    "  classification_xgboost.pkl"
)

print(
    "  classification_model.pkl"
)

print(
    "  regression_random_forest.pkl"
)

print(
    "  regression_xgboost.pkl"
)

print(
    "  regression_model.pkl"
)

print(
    "  model.pkl"
)

print(
    "  benchmark_results.json"
)

print(
    "  model_meta.txt"
)

print("=" * 70)
