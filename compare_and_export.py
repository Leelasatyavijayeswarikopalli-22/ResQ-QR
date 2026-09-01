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

from sklearn.model_selection import StratifiedKFold

from xgboost import (
    XGBClassifier,
    XGBRegressor,
)


# ============================================================
# RESQ-QR MODEL TRAINING + BENCHMARKING
# ============================================================

RANDOM_STATE = 42

OOF_FOLDS = 5


# ============================================================
# REGRESSION FEATURES
# ============================================================
#
# ONLY RAW OBSERVABLE TELEMETRY
#
# NOT INCLUDED:
#
# total_latency_ms
# transaction_age_ms
# retry_count
# order_amount_inr
# payment_stage
#
# ============================================================

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

    # Error

    "error_code_category",
    "is_timeout_flag",
]


# ============================================================
# CLASSIFICATION FEATURES
# ============================================================
#
# THE CLASSIFIER RECEIVES ONLY THESE THREE FEATURES.
#
# ============================================================

CLASSIFICATION_FEATURES = [

    "predicted_network_degradation",

    "predicted_gateway_degradation",

    "predicted_bank_degradation",
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
# ACTION LABELS
# ============================================================
#
# 1 = GATEWAY
# 2 = BANK
# 3 = NETWORK
#
# ============================================================

ACTION_NAMES = [

    "CONTEXTUAL_NUDGE",

    "NO_ACTION",

    "GENERATE_DYNAMIC_QR",
]


ACTION_LABEL_MAPPING = {

    1: "CONTEXTUAL_NUDGE",

    2: "NO_ACTION",

    3: "GENERATE_DYNAMIC_QR",
}


# ============================================================
# HELPER
# BENCHMARK INFERENCE
# ============================================================

def benchmark_inference(
    model,
    X_test,
    n_runs=500,
):

    latencies = []

    sample_count = min(
        100,
        len(X_test),
    )

    sample = X_test.iloc[
        :sample_count
    ]

    model.predict(
        sample.iloc[:1]
    )

    for _ in range(n_runs):

        start = time.perf_counter()

        model.predict(
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

        latencies.append(
            per_sample_ms
        )

    latencies = np.array(
        latencies
    )

    return {

        "average_ms":
            float(
                np.mean(latencies)
            ),

        "p50_ms":
            float(
                np.percentile(
                    latencies,
                    50,
                )
            ),

        "p95_ms":
            float(
                np.percentile(
                    latencies,
                    95,
                )
            ),

        "p99_ms":
            float(
                np.percentile(
                    latencies,
                    99,
                )
            ),
    }


# ============================================================
# HELPER
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
# CREATE REGRESSION MODEL
# ============================================================

def create_regression_model(
    model_type,
):

    if model_type == "Random Forest":

        return RandomForestRegressor(

            n_estimators=250,

            max_depth=10,

            min_samples_leaf=2,

            random_state=RANDOM_STATE,

            n_jobs=-1,
        )

    return XGBRegressor(

        n_estimators=250,

        max_depth=6,

        learning_rate=0.06,

        subsample=0.9,

        colsample_bytree=0.9,

        random_state=RANDOM_STATE,

        objective="reg:squarederror",

        n_jobs=-1,
    )


# ============================================================
# CREATE XGBOOST REGRESSORS
# ============================================================

def create_xgb_regressors():

    regressors = {}

    for target in REGRESSION_TARGETS:

        regressors[target] = XGBRegressor(

            n_estimators=250,

            max_depth=6,

            learning_rate=0.06,

            subsample=0.9,

            colsample_bytree=0.9,

            random_state=RANDOM_STATE,

            objective="reg:squarederror",

            n_jobs=-1,
        )

    return regressors


# ============================================================
# OOF REGRESSION PREDICTIONS
# ============================================================

def generate_oof_regression_predictions(

    model_type,

    X_train,

    y_train,

    classification_labels,
):

    print("\n" + "=" * 70)

    print(
        "GENERATING "
        + str(OOF_FOLDS)
        + "-FOLD OOF REGRESSION PREDICTIONS"
    )

    print("=" * 70)


    oof_predictions = np.zeros(

        (
            len(X_train),
            len(REGRESSION_TARGETS),
        )
    )


    stratified_kfold = StratifiedKFold(

        n_splits=OOF_FOLDS,

        shuffle=True,

        random_state=RANDOM_STATE,
    )


    for fold_number, (

        fold_train_indices,

        fold_validation_indices,

    ) in enumerate(

        stratified_kfold.split(

            X_train,

            classification_labels,
        ),

        start=1,
    ):

        print(
            f"\nOOF Fold "
            f"{fold_number}/{OOF_FOLDS}"
        )


        X_fold_train = X_train.iloc[
            fold_train_indices
        ]


        X_fold_validation = X_train.iloc[
            fold_validation_indices
        ]


        y_fold_train = y_train.iloc[
            fold_train_indices
        ]


        # ====================================================
        # RANDOM FOREST
        # ====================================================

        if model_type == "Random Forest":

            fold_model = (
                create_regression_model(
                    "Random Forest"
                )
            )

            fold_model.fit(

                X_fold_train,

                y_fold_train,
            )

            fold_predictions = (
                fold_model.predict(
                    X_fold_validation
                )
            )


        # ====================================================
        # XGBOOST
        # ====================================================

        else:

            fold_predictions = []


            for target in REGRESSION_TARGETS:

                fold_model = (
                    create_regression_model(
                        "XGBoost"
                    )
                )


                fold_model.fit(

                    X_fold_train,

                    y_fold_train[target],
                )


                prediction = (

                    fold_model.predict(

                        X_fold_validation
                    )
                )


                fold_predictions.append(
                    prediction
                )


            fold_predictions = (
                np.column_stack(
                    fold_predictions
                )
            )


        oof_predictions[
            fold_validation_indices
        ] = fold_predictions


        print(
            "  Validation samples:",
            len(
                fold_validation_indices
            )
        )


    oof_dataframe = pd.DataFrame(

        oof_predictions,

        columns=CLASSIFICATION_FEATURES,

        index=X_train.index,
    )


    print(
        "\nOOF predictions generated."
    )


    print(
        "OOF shape:",
        oof_dataframe.shape,
    )


    return oof_dataframe


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)

print(
    "LOADING DATA"
)

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
# VALIDATE LABELS
# ============================================================

for name, dataframe in [

    ("train_data.csv", train_df),

    ("test_data.csv", test_df),

]:

    labels = set(

        dataframe["action_label"]
        .astype(int)
        .unique()
    )


    if labels != {1, 2, 3}:

        raise ValueError(

            f"{name} contains invalid labels: "
            f"{sorted(labels)}. "
            "Expected exactly [1, 2, 3]."
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

    if (

        col not in train_df.columns

        or col not in test_df.columns
    )
]


missing_regression = [

    col

    for col in required_regression_columns

    if (

        col not in reg_train_df.columns

        or col not in reg_test_df.columns
    )
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
# DATA ALIGNMENT
# ============================================================

if len(train_df) != len(reg_train_df):

    raise ValueError(
        "Training datasets are not aligned."
    )


if len(test_df) != len(reg_test_df):

    raise ValueError(
        "Testing datasets are not aligned."
    )


print(
    "\nData alignment verified."
)

print(
    "Training rows:",
    len(train_df)
)

print(
    "Testing rows:",
    len(test_df)
)


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
# REGRESSION MODEL TRAINING
# ============================================================

print("\n" + "=" * 70)

print(
    "REGRESSION MODEL TRAINING"
)

print("=" * 70)


# ============================================================
# RANDOM FOREST REGRESSOR
# ============================================================

rf_regressor = RandomForestRegressor(

    n_estimators=250,

    max_depth=10,

    min_samples_leaf=2,

    random_state=RANDOM_STATE,

    n_jobs=-1,
)


print(
    "\nTraining Random Forest Regressor..."
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
# RANDOM FOREST METRICS
# ============================================================

rf_mae = mean_absolute_error(

    y_reg_test,

    rf_reg_predictions,
)


rf_mse = mean_squared_error(

    y_reg_test,

    rf_reg_predictions,
)


rf_rmse = np.sqrt(
    rf_mse
)


rf_r2 = r2_score(

    y_reg_test,

    rf_reg_predictions,

    multioutput="uniform_average",
)


rf_reg_latency = benchmark_inference(

    rf_regressor,

    X_reg_test,
)


rf_reg_size = get_model_size_mb(

    rf_regressor,

    "regression_random_forest.pkl",
)


# ============================================================
# XGBOOST REGRESSION
# ============================================================

print(
    "\nTraining XGBoost Regressors..."
)


xgb_regressors = (
    create_xgb_regressors()
)


xgb_reg_training_start = (
    time.perf_counter()
)


for target in REGRESSION_TARGETS:

    print(
        "  Training:",
        target
    )


    xgb_regressors[target].fit(

        X_reg_train,

        y_reg_train[target],
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

        xgb_regressors[target].predict(

            X_reg_test
        )
    )


    xgb_reg_predictions.append(
        prediction
    )


xgb_reg_predictions = np.column_stack(

    xgb_reg_predictions
)


# ============================================================
# XGBOOST METRICS
# ============================================================

xgb_mae = mean_absolute_error(

    y_reg_test,

    xgb_reg_predictions,
)


xgb_mse = mean_squared_error(

    y_reg_test,

    xgb_reg_predictions,
)


xgb_rmse = np.sqrt(
    xgb_mse
)


xgb_r2 = r2_score(

    y_reg_test,

    xgb_reg_predictions,

    multioutput="uniform_average",
)


# ============================================================
# XGBOOST LATENCY
# ============================================================

xgb_latency_values = []


sample_count = min(

    100,

    len(X_reg_test),
)


sample = X_reg_test.iloc[
    :sample_count
]


for target in REGRESSION_TARGETS:

    xgb_regressors[target].predict(

        sample.iloc[:1]
    )


for _ in range(500):

    start = time.perf_counter()


    for target in REGRESSION_TARGETS:

        xgb_regressors[target].predict(
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


xgb_reg_size = get_model_size_mb(

    xgb_reg_bundle,

    "regression_xgboost.pkl",
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

xgb_importance_matrix = []


for target in REGRESSION_TARGETS:

    xgb_importance_matrix.append(

        xgb_regressors[target]
        .feature_importances_
    )


xgb_importance_matrix = np.array(
    xgb_importance_matrix
)


xgb_average_importance = np.mean(

    xgb_importance_matrix,

    axis=0,
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
# SELECT BEST REGRESSION
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

    best_regressor_name = "XGBoost"

    best_regressor = xgb_reg_bundle

else:

    best_regressor_name = "Random Forest"

    best_regressor = rf_regressor


print(
    "\n🏆 Best Regression Model:",
    best_regressor_name,
)


# ============================================================
# OOF CLASSIFICATION FEATURES
# ============================================================

y_train = train_df[
    "action_label"
]


X_train_classification = (

    generate_oof_regression_predictions(

        best_regressor_name,

        X_reg_train,

        y_reg_train,

        y_train,
    )
)


# ============================================================
# TEST CLASSIFICATION FEATURES
# ============================================================

if best_regressor_name == "Random Forest":

    predicted_test = (

        rf_regressor.predict(

            X_reg_test
        )
    )

else:

    predicted_test_parts = []


    for target in REGRESSION_TARGETS:

        predicted_test_parts.append(

            xgb_regressors[target].predict(

                X_reg_test
            )
        )


    predicted_test = np.column_stack(

        predicted_test_parts
    )


X_test_classification = pd.DataFrame(

    predicted_test,

    columns=CLASSIFICATION_FEATURES,

    index=test_df.index,
)


y_test = test_df[
    "action_label"
]


# ============================================================
# CLASSIFICATION INPUT VERIFICATION
# ============================================================

print("\n" + "=" * 70)

print(
    "CLASSIFICATION INPUT VERIFICATION"
)

print("=" * 70)


print(
    "\nClassifier receives ONLY:"
)


for feature in CLASSIFICATION_FEATURES:

    print(
        "  -",
        feature
    )


print(
    "\nTraining classification shape:",
    X_train_classification.shape,
)


print(
    "Testing classification shape:",
    X_test_classification.shape,
)


print(
    "\nClassification columns:"
)


print(
    list(
        X_train_classification.columns
    )
)


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
#
# XGBoost requires classes 0,1,2.
#
# Our dataset MUST remain 1,2,3.
#
# Therefore:
#
# 1 -> 0
# 2 -> 1
# 3 -> 2
#
# only during XGBoost training.
#
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


# ============================================================
# CLASSIFICATION RESULTS
# ============================================================

classification_results = {}


# ============================================================
# RANDOM FOREST CLASSIFIER
# ============================================================

print(
    "\nTraining Random Forest Classifier..."
)


start = time.perf_counter()


rf_classifier.fit(

    X_train_classification,

    y_train,
)


training_time = (

    time.perf_counter()
    - start

) * 1000


rf_predictions = (

    rf_classifier.predict(

        X_test_classification
    )
)


accuracy = accuracy_score(

    y_test,

    rf_predictions,
)


precision = precision_score(

    y_test,

    rf_predictions,

    average="weighted",

    zero_division=0,
)


recall = recall_score(

    y_test,

    rf_predictions,

    average="weighted",

    zero_division=0,
)


weighted_f1 = f1_score(

    y_test,

    rf_predictions,

    average="weighted",

    zero_division=0,
)


macro_f1 = f1_score(

    y_test,

    rf_predictions,

    average="macro",

    zero_division=0,
)


micro_f1 = f1_score(

    y_test,

    rf_predictions,

    average="micro",

    zero_division=0,
)


confusion = confusion_matrix(

    y_test,

    rf_predictions,

    labels=[1, 2, 3],
)


latency = benchmark_inference(

    rf_classifier,

    X_test_classification,
)


model_size = get_model_size_mb(

    rf_classifier,

    "classification_random_forest.pkl",
)


classification_results[
    "Random Forest"
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
        rf_classifier
        .feature_importances_
        .tolist(),
}


print(
    f"Random Forest Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# XGBOOST CLASSIFIER
# ============================================================

print(
    "\nTraining XGBoost Classifier..."
)


# ------------------------------------------------------------
# CONVERT 1,2,3 -> 0,1,2
# ONLY FOR XGBOOST
# ------------------------------------------------------------

y_train_xgb = (

    y_train.astype(int) - 1
)


y_test_xgb = (

    y_test.astype(int) - 1
)


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

start = time.perf_counter()


xgb_classifier.fit(

    X_train_classification,

    y_train_xgb,
)


training_time = (

    time.perf_counter()
    - start

) * 1000


# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------

xgb_predictions_encoded = (

    xgb_classifier.predict(

        X_test_classification
    )
)


# ------------------------------------------------------------
# CONVERT BACK
#
# 0 -> 1
# 1 -> 2
# 2 -> 3
# ------------------------------------------------------------

xgb_predictions = (

    xgb_predictions_encoded
    .astype(int)
    + 1
)


# ============================================================
# XGBOOST METRICS
# ============================================================

accuracy = accuracy_score(

    y_test,

    xgb_predictions,
)


precision = precision_score(

    y_test,

    xgb_predictions,

    average="weighted",

    zero_division=0,
)


recall = recall_score(

    y_test,

    xgb_predictions,

    average="weighted",

    zero_division=0,
)


weighted_f1 = f1_score(

    y_test,

    xgb_predictions,

    average="weighted",

    zero_division=0,
)


macro_f1 = f1_score(

    y_test,

    xgb_predictions,

    average="macro",

    zero_division=0,
)


micro_f1 = f1_score(

    y_test,

    xgb_predictions,

    average="micro",

    zero_division=0,
)


confusion = confusion_matrix(

    y_test,

    xgb_predictions,

    labels=[1, 2, 3],
)


latency = benchmark_inference(

    xgb_classifier,

    X_test_classification,
)


model_size = get_model_size_mb(

    xgb_classifier,

    "classification_xgboost.pkl",
)


classification_results[
    "XGBoost"
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
        xgb_classifier
        .feature_importances_
        .tolist(),
}


print(
    f"XGBoost Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"XGBoost Weighted F1: "
    f"{weighted_f1:.4f}"
)

print(
    f"XGBoost Macro F1: "
    f"{macro_f1:.4f}"
)


# ============================================================
# SELECT BEST CLASSIFIER
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

    best_classifier_name = "XGBoost"

    best_classifier = xgb_classifier

else:

    best_classifier_name = "Random Forest"

    best_classifier = rf_classifier


print(
    "\n🏆 Best Classification Model:",
    best_classifier_name,
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
# SAVE PIPELINE METADATA
# ============================================================

classification_pipeline_meta = {

    "regression_features":
        FEATURE_COLS,

    "classification_features":
        CLASSIFICATION_FEATURES,

    "regression_targets":
        REGRESSION_TARGETS,

    "best_regression_model":
        best_regressor_name,

    "best_classification_model":
        best_classifier_name,

    "oof_folds":
        OOF_FOLDS,

    "classification_training_method":
        "5-fold out-of-fold regression predictions",

    "oof_used_for_classification_training":
        True,

    "action_label_mapping": {

        "1":
            "CONTEXTUAL_NUDGE",

        "2":
            "NO_ACTION",

        "3":
            "GENERATE_DYNAMIC_QR",
    },

    "xgboost_label_encoding": {

        "dataset_1":
            "internal_0",

        "dataset_2":
            "internal_1",

        "dataset_3":
            "internal_2",
    },

    "architecture":
        "Raw telemetry -> regression -> "
        "3 degradation scores -> classification -> "
        "action label 1/2/3",
}


with open(

    "classification_pipeline_meta.json",

    "w",

) as f:

    json.dump(

        classification_pipeline_meta,

        f,

        indent=4,
    )


# ============================================================
# BEST CLASSIFIER REPORT
# ============================================================

best_predictions = (

    best_classifier.predict(

        X_test_classification
    )
)


# XGBoost returns 0,1,2 internally.
# Convert to 1,2,3 for final evaluation.

if best_classifier_name == "XGBoost":

    best_predictions = (

        best_predictions
        .astype(int)
        + 1
    )


print("\n" + "=" * 70)

print(
    "BEST CLASSIFICATION MODEL"
)

print("=" * 70)

print(
    best_classifier_name
)


print(
    "\nClassification Report:\n"
)


print(

    classification_report(

        y_test,

        best_predictions,

        labels=[1, 2, 3],

        target_names=[

            "CONTEXTUAL_NUDGE",

            "NO_ACTION",

            "GENERATE_DYNAMIC_QR",
        ],

        zero_division=0,
    )
)


# ============================================================
# SAVE BEST REGRESSION MODEL
# ============================================================

joblib.dump(

    best_regressor,

    "regression_model.pkl",
)


# ============================================================
# SAVE BENCHMARK RESULTS
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

    "regression_feature_columns":
        FEATURE_COLS,

    "classification_feature_columns":
        CLASSIFICATION_FEATURES,

    "regression_targets":
        REGRESSION_TARGETS,

    "action_names":
        ACTION_NAMES,

    "action_label_mapping":
        ACTION_LABEL_MAPPING,

    "oof_folds":
        OOF_FOLDS,

    "oof_used_for_classification_training":
        True,

    "architecture":
        "Raw telemetry -> regression -> "
        "3 degradation scores -> classification",

    "excluded_from_regression": [

        "total_latency_ms",

        "transaction_age_ms",

        "retry_count",

        "order_amount_inr",

        "payment_stage",
    ],

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
# SAVE MODEL METADATA
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

        "Dataset action labels: "
        "1=CONTEXTUAL_NUDGE, "
        "2=NO_ACTION, "
        "3=GENERATE_DYNAMIC_QR\n"
    )


    f.write(

        "Classification input features: "
        "predicted_network_degradation, "
        "predicted_gateway_degradation, "
        "predicted_bank_degradation\n"
    )


    f.write(

        "Regression input: raw observable telemetry only\n"
    )


    f.write(

        "Regression excludes: "
        "total_latency_ms, "
        "transaction_age_ms, "
        "retry_count, "
        "order_amount_inr, "
        "payment_stage\n"
    )


    f.write(

        "Classification training: "
        "5-fold out-of-fold regression predictions\n"
    )


    f.write(

        "OOF folds: "
        + str(OOF_FOLDS)
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
# FINAL RESULTS
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
# ARCHITECTURE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "RESQ-QR ARCHITECTURE"
)

print(
    "=" * 70
)


print(
    "\nRAW NETWORK / GATEWAY / BANK TELEMETRY"
)


print(
    "        ↓"
)


print(
    "REGRESSION MODEL"
)


print(
    "        ↓"
)


print(
    "3 DEGRADATION SCORES"
)


print(
    "  • predicted_network_degradation"
)


print(
    "  • predicted_gateway_degradation"
)


print(
    "  • predicted_bank_degradation"
)


print(
    "        ↓"
)


print(
    "CLASSIFICATION MODEL"
)


print(
    "        ↓"
)


print(
    "ACTION LABEL"
)


print(
    "  • 1 = CONTEXTUAL_NUDGE"
)


print(
    "  • 2 = NO_ACTION"
)


print(
    "  • 3 = GENERATE_DYNAMIC_QR"
)


print(
    "\nClassification receives ONLY the three "
    "predicted degradation scores."
)


print(
    "\nRegression does NOT use:"
)


print(
    "  • total_latency_ms"
)


print(
    "  • transaction_age_ms"
)


print(
    "  • retry_count"
)


print(
    "  • order_amount_inr"
)


print(
    "  • payment_stage"
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
    "  classification_pipeline_meta.json"
)


print(
    "  model_meta.txt"
)


print(
    "\n" + "=" * 70
)

print(
    "RESQ-QR TRAINING COMPLETE"
)

print(
    "=" * 70
)