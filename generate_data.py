import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = 5000


# ============================================================
# 1. BASE PAYMENT TELEMETRY
# ============================================================

gateway_latency = np.random.gamma(
    shape=2.0,
    scale=450,
    size=N_SAMPLES
) + 30

bank_latency = np.random.gamma(
    shape=2.0,
    scale=500,
    size=N_SAMPLES
) + 40

total_latency = gateway_latency + bank_latency

packet_loss = np.random.gamma(
    shape=1.8,
    scale=2.0,
    size=N_SAMPLES
)

packet_loss = np.clip(packet_loss, 0, 30)

order_amount = np.random.lognormal(
    mean=np.log(1800),
    sigma=1.0,
    size=N_SAMPLES
)

order_amount = np.clip(order_amount, 100, 50000)

retry_count = np.random.choice(
    [0, 1, 2, 3],
    size=N_SAMPLES,
    p=[0.68, 0.17, 0.10, 0.05]
)

payment_stage = np.random.choice(
    [0, 1, 2, 3],
    size=N_SAMPLES,
    p=[0.10, 0.40, 0.40, 0.10]
)

error_code = np.random.choice(
    [0, 1, 2, 3],
    size=N_SAMPLES,
    p=[0.50, 0.23, 0.17, 0.10]
)


# ============================================================
# 2. HISTORICAL SYSTEM HEALTH
# ============================================================

gateway_failure_rate = np.random.beta(
    2, 15, N_SAMPLES
)

bank_failure_rate = np.random.beta(
    2, 18, N_SAMPLES
)

gateway_timeout_rate = np.random.beta(
    2, 20, N_SAMPLES
)

bank_timeout_rate = np.random.beta(
    2, 22, N_SAMPLES
)


# ============================================================
# 3. NETWORK / SYSTEM HEALTH SCORES
# ============================================================

network_quality = (
    100
    - packet_loss * 3.0
    - total_latency / 80
    + np.random.normal(0, 5, N_SAMPLES)
)

network_quality = np.clip(network_quality, 0, 100)


gateway_health = (
    100
    - gateway_failure_rate * 120
    - gateway_timeout_rate * 100
    - gateway_latency / 80
    + np.random.normal(0, 4, N_SAMPLES)
)

gateway_health = np.clip(gateway_health, 0, 100)


bank_health = (
    100
    - bank_failure_rate * 120
    - bank_timeout_rate * 100
    - bank_latency / 90
    + np.random.normal(0, 4, N_SAMPLES)
)

bank_health = np.clip(bank_health, 0, 100)


# ============================================================
# 4. TRANSACTION AGE
# ============================================================

transaction_age = (
    total_latency
    + retry_count * np.random.uniform(
        700,
        1800,
        N_SAMPLES
    )
    + np.random.uniform(
        100,
        1000,
        N_SAMPLES
    )
)


# ============================================================
# 5. DERIVED FEATURES
# ============================================================

latency_ratio = (
    gateway_latency /
    np.maximum(bank_latency, 1)
)

retry_pressure = (
    retry_count * 25
    + packet_loss * 2
)

network_degradation = (
    100 - network_quality
)

system_degradation = (
    100
    - (
        gateway_health * 0.45
        + bank_health * 0.35
        + network_quality * 0.20
    )
)


# ============================================================
# 6. TIMEOUT SIGNAL
# ============================================================

is_timeout = (
    (
        (total_latency > 2500)
        | (gateway_latency > 1800)
        | (error_code == 2)
        | (packet_loss > 10)
    )
    &
    (error_code != 1)
    &
    (error_code != 3)
).astype(int)


# ============================================================
# 7. REALISTIC GROUND-TRUTH DECISION LOGIC
#
# 0 = NO_ACTION
# 1 = CONTEXTUAL_NUDGE
# 2 = GENERATE_DYNAMIC_QR
# ============================================================

actions = np.zeros(N_SAMPLES, dtype=int)

for i in range(N_SAMPLES):

    # --------------------------------------------------------
    # USER-SIDE PROBLEMS
    # --------------------------------------------------------

    if error_code[i] == 1:

        actions[i] = 1

    # --------------------------------------------------------
    # ISSUER BANK IS UNAVAILABLE
    # --------------------------------------------------------

    elif (
        error_code[i] == 3
        and bank_health[i] < 45
    ):

        actions[i] = 0

    # --------------------------------------------------------
    # EXTREME RETRY / DUPLICATE RISK
    # --------------------------------------------------------

    elif (
        retry_count[i] >= 3
        and transaction_age[i] > 5000
    ):

        actions[i] = 0

    # --------------------------------------------------------
    # HIGH BANK FAILURE
    # --------------------------------------------------------

    elif (
        bank_failure_rate[i] > 0.35
        and bank_health[i] < 40
    ):

        actions[i] = 0

    # --------------------------------------------------------
    # DYNAMIC QR CONDITIONS
    # --------------------------------------------------------

    elif (
        (
            network_quality[i] < 45
            and bank_health[i] > 55
        )
        or
        (
            gateway_health[i] < 45
            and bank_health[i] > 60
        )
        or
        (
            is_timeout[i] == 1
            and gateway_health[i] < 55
            and bank_health[i] > 50
            and retry_count[i] <= 1
        )
        or
        (
            total_latency[i] > 2800
            and bank_health[i] > 55
            and retry_count[i] <= 1
        )
    ):

        actions[i] = 2

    # --------------------------------------------------------
    # MINOR DEGRADATION → USER NUDGE
    # --------------------------------------------------------

    elif (
        packet_loss[i] > 5
        or retry_count[i] >= 2
        or network_quality[i] < 65
    ):

        actions[i] = 1

    # --------------------------------------------------------
    # CLEAN TRANSACTION
    # --------------------------------------------------------

    else:

        actions[i] = 0


# ============================================================
# 8. BUILD DATAFRAME
# ============================================================

df = pd.DataFrame(
    {
        "gateway_latency_ms": np.round(
            gateway_latency, 2
        ),

        "bank_latency_ms": np.round(
            bank_latency, 2
        ),

        "total_latency_ms": np.round(
            total_latency, 2
        ),

        "packet_loss_pct": np.round(
            packet_loss, 2
        ),

        "payment_stage": payment_stage,

        "error_code_category": error_code,

        "is_timeout_flag": is_timeout,

        "order_amount_inr": np.round(
            order_amount, 2
        ),

        "retry_count": retry_count,

        "gateway_recent_failure_rate": np.round(
            gateway_failure_rate, 4
        ),

        "bank_recent_failure_rate": np.round(
            bank_failure_rate, 4
        ),

        "gateway_timeout_rate": np.round(
            gateway_timeout_rate, 4
        ),

        "bank_timeout_rate": np.round(
            bank_timeout_rate, 4
        ),

        "network_quality_score": np.round(
            network_quality, 2
        ),

        "gateway_health_score": np.round(
            gateway_health, 2
        ),

        "bank_health_score": np.round(
            bank_health, 2
        ),

        "transaction_age_ms": np.round(
            transaction_age, 2
        ),

        "latency_ratio": np.round(
            latency_ratio, 3
        ),

        "retry_pressure": np.round(
            retry_pressure, 2
        ),

        "network_degradation_score": np.round(
            network_degradation, 2
        ),

        "system_degradation_score": np.round(
            system_degradation, 2
        ),

        "action_label": actions,
    }
)


# ============================================================
# 9. SAVE DATASET
# ============================================================

df.to_csv(
    "telemetry_data.csv",
    index=False
)


# ============================================================
# 10. SUMMARY
# ============================================================

print("=" * 70)
print("RESQ-QR TELEMETRY DATASET GENERATED")
print("=" * 70)

print(f"Total samples : {len(df)}")
print(f"Total features: {len(df.columns) - 1}")

print("\nClass Distribution:")
print(
    df["action_label"]
    .value_counts()
    .sort_index()
)

print("\nClass Percentages:")
print(
    (
        df["action_label"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)
)

print("\nDataset saved as:")
print("telemetry_data.csv")