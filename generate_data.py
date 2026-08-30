import numpy as np
import pandas as pd


# ============================================================
# RESQ-QR SYNTHETIC PAYMENT TELEMETRY GENERATOR
# ============================================================

np.random.seed(42)

N_SAMPLES = 6000


# ------------------------------------------------------------
# 1. COMPONENT DEGRADATION
#
# These are hidden "true" system conditions.
# The ML model will NOT receive these directly.
#
# 0   = healthy
# 100 = severely degraded
# ------------------------------------------------------------

network_degradation = np.clip(
    np.random.beta(2.0, 4.0, N_SAMPLES) * 100,
    0,
    100,
)

gateway_degradation = np.clip(
    np.random.beta(2.0, 4.0, N_SAMPLES) * 100,
    0,
    100,
)

bank_degradation = np.clip(
    np.random.beta(2.0, 4.0, N_SAMPLES) * 100,
    0,
    100,
)


# ------------------------------------------------------------
# 2. OBSERVABLE TELEMETRY
# ------------------------------------------------------------

# Network latency increases with network degradation.
network_latency = (
    80
    + network_degradation * 18
    + np.random.normal(0, 80, N_SAMPLES)
)

network_latency = np.clip(network_latency, 30, 5000)


# Packet loss increases with network degradation.
packet_loss = (
    network_degradation * 0.16
    + np.random.normal(0, 0.8, N_SAMPLES)
)

packet_loss = np.clip(packet_loss, 0, 30)


# Network jitter also increases when the network is unhealthy.
network_jitter = (
    5
    + network_degradation * 0.35
    + np.random.normal(0, 3, N_SAMPLES)
)

network_jitter = np.clip(network_jitter, 1, 150)


# Gateway latency depends mainly on gateway degradation.
gateway_latency = (
    70
    + gateway_degradation * 20
    + network_degradation * 3
    + np.random.normal(0, 100, N_SAMPLES)
)

gateway_latency = np.clip(gateway_latency, 30, 5000)


# Bank latency depends mainly on bank degradation.
bank_latency = (
    90
    + bank_degradation * 22
    + np.random.normal(0, 100, N_SAMPLES)
)

bank_latency = np.clip(bank_latency, 30, 5000)


total_latency = (
    gateway_latency
    + bank_latency
    + network_latency
)


# ------------------------------------------------------------
# 3. FAILURE / HEALTH SIGNALS
# ------------------------------------------------------------

gateway_failure_rate = np.clip(
    gateway_degradation * 0.7
    + np.random.normal(0, 4, N_SAMPLES),
    0,
    100,
)

gateway_timeout_rate = np.clip(
    gateway_degradation * 0.6
    + network_degradation * 0.2
    + np.random.normal(0, 4, N_SAMPLES),
    0,
    100,
)

bank_failure_rate = np.clip(
    bank_degradation * 0.7
    + np.random.normal(0, 4, N_SAMPLES),
    0,
    100,
)

bank_timeout_rate = np.clip(
    bank_degradation * 0.6
    + np.random.normal(0, 4, N_SAMPLES),
    0,
    100,
)


# ------------------------------------------------------------
# 4. RETRIES
# ------------------------------------------------------------

retry_probability = (
    0.05
    + network_degradation / 250
    + gateway_degradation / 250
    + bank_degradation / 300
)

retry_probability = np.clip(retry_probability, 0.02, 0.9)

retry_count = np.random.binomial(
    3,
    retry_probability,
    N_SAMPLES,
)

retry_count = np.clip(retry_count, 0, 3)


# ------------------------------------------------------------
# 5. PAYMENT STAGE
#
# 0 = INITIATED
# 1 = AUTHENTICATING
# 2 = AUTHORIZING
# 3 = SETTLING
# ------------------------------------------------------------

payment_stage = np.random.choice(
    [0, 1, 2, 3],
    size=N_SAMPLES,
    p=[0.15, 0.30, 0.40, 0.15],
)


# ------------------------------------------------------------
# 6. TRANSACTION AGE
# ------------------------------------------------------------

transaction_age_ms = (
    500
    + total_latency * 0.8
    + retry_count * 1200
    + np.random.normal(0, 500, N_SAMPLES)
)

transaction_age_ms = np.clip(
    transaction_age_ms,
    100,
    20000,
)


# ------------------------------------------------------------
# 7. ORDER AMOUNT
# ------------------------------------------------------------

order_amount = np.random.uniform(
    100,
    15000,
    N_SAMPLES,
)


# ------------------------------------------------------------
# 8. ERROR CATEGORIES
#
# 0 = NONE
# 1 = INSUFFICIENT_FUNDS
# 2 = TIMEOUT
# 3 = BANK_OFFLINE
# ------------------------------------------------------------

error_code = np.zeros(N_SAMPLES, dtype=int)

for i in range(N_SAMPLES):

    # Bank degradation can result in bank-offline errors.
    if bank_degradation[i] > 82:
        error_code[i] = 3

    # User balance error is independent of infrastructure.
    elif np.random.random() < 0.12:
        error_code[i] = 1

    # High network/gateway degradation can cause timeouts.
    elif (
        network_degradation[i] > 70
        or gateway_degradation[i] > 75
        or total_latency[i] > 5000
    ):
        error_code[i] = 2

    else:
        error_code[i] = 0


# ------------------------------------------------------------
# 9. TIMEOUT FLAG
# ------------------------------------------------------------

is_timeout = (
    (
        (total_latency > 5000)
        | (gateway_timeout_rate > 45)
        | (network_degradation > 75)
        | (error_code == 2)
    )
).astype(int)


# ------------------------------------------------------------
# 10. HEALTH SCORES
# ------------------------------------------------------------

network_health = 100 - network_degradation
gateway_health = 100 - gateway_degradation
bank_health = 100 - bank_degradation


# ------------------------------------------------------------
# 11. DERIVED SIGNALS
# ------------------------------------------------------------

network_stress = np.clip(
    (
        network_latency / 50
        + packet_loss * 3
        + network_jitter / 10
    ) / 3,
    0,
    100,
)

gateway_stress = np.clip(
    (
        gateway_latency / 50
        + gateway_failure_rate
        + gateway_timeout_rate
    ) / 3,
    0,
    100,
)

bank_stress = np.clip(
    (
        bank_latency / 50
        + bank_failure_rate
        + bank_timeout_rate
    ) / 3,
    0,
    100,
)


# ------------------------------------------------------------
# 12. CLASSIFICATION TARGET
#
# 0 = NO_ACTION
# 1 = CONTEXTUAL_NUDGE
# 2 = RETRY_PAYMENT
# 3 = GENERATE_DYNAMIC_QR
#
# The target is based on the simulated payment environment.
# There are NO manually assigned component percentages.
# ------------------------------------------------------------

actions = np.zeros(N_SAMPLES, dtype=int)


for i in range(N_SAMPLES):

    # --------------------------------------------------------
    # User-side problem
    # --------------------------------------------------------
    if error_code[i] == 1:
        actions[i] = 1

    # --------------------------------------------------------
    # Settlement stage is highly sensitive.
    # We don't aggressively retry/redirect it.
    # --------------------------------------------------------
    elif payment_stage[i] == 3:

        if bank_degradation[i] > 80:
            actions[i] = 0

        elif transaction_age_ms[i] > 9000:
            actions[i] = 0

        elif retry_count[i] >= 2:
            actions[i] = 0

        else:
            actions[i] = 0

    # --------------------------------------------------------
    # Bank is severely unhealthy.
    # Avoid retrying or QR when issuer itself is unavailable.
    # --------------------------------------------------------
    elif error_code[i] == 3 or bank_degradation[i] > 82:

        actions[i] = 0

    # --------------------------------------------------------
    # Severe network/gateway degradation.
    # QR becomes the preferred fallback.
    # --------------------------------------------------------
    elif (
        network_degradation[i] > 72
        and gateway_degradation[i] > 60
    ):

        actions[i] = 3

    elif (
        network_degradation[i] > 78
        or gateway_degradation[i] > 82
        or packet_loss[i] > 12
        or gateway_timeout_rate[i] > 50
    ):

        actions[i] = 3

    # --------------------------------------------------------
    # Moderate degradation.
    # Retry can be useful if retry budget remains.
    # --------------------------------------------------------
    elif (
        (
            network_degradation[i] > 45
            or gateway_degradation[i] > 45
        )
        and retry_count[i] < 2
        and transaction_age_ms[i] < 7000
    ):

        actions[i] = 2

    # --------------------------------------------------------
    # Clean transaction.
    # --------------------------------------------------------
    else:
        actions[i] = 0


# ------------------------------------------------------------
# 13. CREATE DATAFRAME
# ------------------------------------------------------------

df = pd.DataFrame(
    {
        # Network
        "network_latency_ms": np.round(network_latency, 2),
        "packet_loss_pct": np.round(packet_loss, 2),
        "network_jitter_ms": np.round(network_jitter, 2),

        # Gateway
        "gateway_latency_ms": np.round(gateway_latency, 2),
        "gateway_failure_rate_pct": np.round(
            gateway_failure_rate,
            2,
        ),
        "gateway_timeout_rate_pct": np.round(
            gateway_timeout_rate,
            2,
        ),

        # Bank
        "bank_latency_ms": np.round(bank_latency, 2),
        "bank_failure_rate_pct": np.round(
            bank_failure_rate,
            2,
        ),
        "bank_timeout_rate_pct": np.round(
            bank_timeout_rate,
            2,
        ),

        # Transaction
        "total_latency_ms": np.round(total_latency, 2),
        "payment_stage": payment_stage,
        "transaction_age_ms": np.round(
            transaction_age_ms,
            2,
        ),
        "retry_count": retry_count,
        "order_amount_inr": np.round(
            order_amount,
            2,
        ),

        # Errors
        "error_code_category": error_code,
        "is_timeout_flag": is_timeout,

        # Hidden regression targets
        #
        # These are not used as input features.
        # The regression model learns to estimate them.
        "true_network_degradation": np.round(
            network_degradation,
            2,
        ),
        "true_gateway_degradation": np.round(
            gateway_degradation,
            2,
        ),
        "true_bank_degradation": np.round(
            bank_degradation,
            2,
        ),

        # Classification target
        "action_label": actions,
    }
)


# ------------------------------------------------------------
# 14. SAVE DATA
# ------------------------------------------------------------

df.to_csv(
    "telemetry_data.csv",
    index=False,
)


# ------------------------------------------------------------
# 15. PRINT DATASET INFORMATION
# ------------------------------------------------------------

print("=" * 70)
print("RESQ-QR TELEMETRY DATASET GENERATED")
print("=" * 70)

print(f"Total samples: {len(df)}")

print("\nAction distribution:")

action_names = {
    0: "NO_ACTION",
    1: "CONTEXTUAL_NUDGE",
    2: "RETRY_PAYMENT",
    3: "GENERATE_DYNAMIC_QR",
}

for label, count in df["action_label"].value_counts().sort_index().items():

    percentage = count / len(df) * 100

    print(
        f"{label} - "
        f"{action_names[label]:22s}: "
        f"{count:5d} "
        f"({percentage:.2f}%)"
    )


print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nSaved as:")
print("telemetry_data.csv")
