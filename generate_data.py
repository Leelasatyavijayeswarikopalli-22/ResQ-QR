import numpy as np
import pandas as pd


# ============================================================
# RESQ-QR SYNTHETIC PAYMENT TELEMETRY GENERATOR
# ============================================================
#
# TOTAL DATASET
# ----------------
# 10,000 samples
#
# CLASS 1:
# NO_ACTION
# Bank degradation
#
# CLASS 2:
# CONTEXTUAL_NUDGE
# Gateway degradation
#
# CLASS 3:
# GENERATE_DYNAMIC_QR
# Network degradation
#
# ============================================================
#
# IMPORTANT:
#
# payment_stage is completely removed.
#
# The following fields are NOT regression features:
#
# total_latency_ms
# transaction_age_ms
# retry_count
# order_amount_inr
#
# They are included only as transaction metadata.
#
# ============================================================


np.random.seed(42)


# ============================================================
# CONFIGURATION
# ============================================================

TOTAL_SAMPLES = 10000

CLASS_1_COUNT = 3333
CLASS_2_COUNT = 3333
CLASS_3_COUNT = 3334


# ============================================================
# NETWORK GENERATION PROFILES
# ============================================================

NETWORK_PROFILES = {

    "2G/EDGE": {

        "latency": (300, 900),

        "packet_loss": (1.0, 4.0),

        "jitter": (15, 40),

    },

    "3G": {

        "latency": (80, 300),

        "packet_loss": (0.3, 2.0),

        "jitter": (8, 25),

    },

    "4G/LTE": {

        "latency": (20, 120),

        "packet_loss": (0.05, 1.0),

        "jitter": (3, 15),

    },

    "5G": {

        "latency": (5, 60),

        "packet_loss": (0.01, 0.5),

        "jitter": (1, 8),

    },

    "Wi-Fi 6/7": {

        "latency": (5, 50),

        "packet_loss": (0.01, 0.4),

        "jitter": (1, 6),

    },

}


# ============================================================
# NETWORK GENERATION SAMPLER
# ============================================================

def choose_network_generation():

    generations = [

        "2G/EDGE",
        "3G",
        "4G/LTE",
        "5G",
        "Wi-Fi 6/7",

    ]

    probabilities = [

        0.08,
        0.12,
        0.35,
        0.30,
        0.15,

    ]

    return np.random.choice(

        generations,

        p=probabilities,

    )


# ============================================================
# CLAMP FUNCTION
# ============================================================

def clamp(value, low, high):

    return max(

        low,

        min(value, high),

    )


# ============================================================
# NETWORK TELEMETRY
# ============================================================

def generate_network_telemetry(

    generation,

    network_degraded,

):

    profile = NETWORK_PROFILES[generation]


    base_latency = np.random.uniform(

        profile["latency"][0],

        profile["latency"][1],

    )


    base_packet_loss = np.random.uniform(

        profile["packet_loss"][0],

        profile["packet_loss"][1],

    )


    base_jitter = np.random.uniform(

        profile["jitter"][0],

        profile["jitter"][1],

    )


    # --------------------------------------------------------
    # NETWORK DEGRADATION SCENARIO
    # --------------------------------------------------------

    if network_degraded:

        severity = np.random.uniform(

            0.65,

            1.00,

        )


        latency = (

            base_latency

            + severity * np.random.uniform(

                150,

                850,

            )

        )


        packet_loss = (

            base_packet_loss

            + severity * np.random.uniform(

                1.0,

                5.0,

            )

        )


        jitter = (

            base_jitter

            + severity * np.random.uniform(

                20,

                70,

            )

        )


    else:

        latency = base_latency

        packet_loss = base_packet_loss

        jitter = base_jitter


    return (

        round(latency, 2),

        round(

            clamp(

                packet_loss,

                0.001,

                10.0,

            ),

            4,

        ),

        round(

            clamp(

                jitter,

                0.1,

                100.0,

            ),

            2,

        ),

    )


# ============================================================
# GATEWAY TELEMETRY
# ============================================================

def generate_gateway_telemetry(

    gateway_degraded,

):

    # Healthy gateway baseline

    latency = np.random.uniform(

        40,

        140,

    )


    failure_rate = np.random.uniform(

        0.05,

        0.20,

    )


    timeout_rate = np.random.uniform(

        0.10,

        0.30,

    )


    # --------------------------------------------------------
    # Gateway degradation
    # --------------------------------------------------------

    if gateway_degraded:

        severity = np.random.uniform(

            0.70,

            1.00,

        )


        latency = np.random.uniform(

            150,

            1200,

        ) * severity + 100


        failure_rate = np.random.uniform(

            0.20,

            0.50,

        )


        timeout_rate = np.random.uniform(

            0.30,

            0.50,

        )


    return (

        round(latency, 2),

        round(

            clamp(

                failure_rate,

                0.05,

                0.50,

            ),

            4,

        ),

        round(

            clamp(

                timeout_rate,

                0.10,

                0.50,

            ),

            4,

        ),

    )


# ============================================================
# BANK TELEMETRY
# ============================================================
#
# Normal bank latency:
# 200–400 ms
#
# Failure:
# 0.1–1.5%
#
# Timeout:
# 1–3%
#
# ============================================================

def generate_bank_telemetry(

    bank_degraded,

):

    if bank_degraded:

        severity = np.random.uniform(

            0.65,

            1.00,

        )


        latency = np.random.uniform(

            200,

            400,

        ) + severity * np.random.uniform(

            250,

            900,

        )


        failure_rate = np.random.uniform(

            0.70,

            1.50,

        )


        timeout_rate = np.random.uniform(

            1.80,

            3.00,

        )


    else:

        latency = np.random.uniform(

            200,

            400,

        )


        failure_rate = np.random.uniform(

            0.10,

            0.30,

        )


        timeout_rate = np.random.uniform(

            1.00,

            1.80,

        )


    return (

        round(latency, 2),

        round(

            clamp(

                failure_rate,

                0.10,

                1.50,

            ),

            4,

        ),

        round(

            clamp(

                timeout_rate,

                1.00,

                3.00,

            ),

            4,

        ),

    )


# ============================================================
# NETWORK DEGRADATION SCORE
# ============================================================

def calculate_network_degradation(

    latency,

    packet_loss,

    jitter,

):

    latency_score = np.clip(

        (

            latency - 20

        ) / 850 * 100,

        0,

        100,

    )


    loss_score = np.clip(

        packet_loss / 5.0 * 100,

        0,

        100,

    )


    jitter_score = np.clip(

        (

            jitter - 2

        ) / 70 * 100,

        0,

        100,

    )


    score = (

        0.50 * latency_score

        + 0.30 * loss_score

        + 0.20 * jitter_score

    )


    return round(

        np.clip(score, 0, 100),

        2,

    )


# ============================================================
# GATEWAY DEGRADATION SCORE
# ============================================================

def calculate_gateway_degradation(

    latency,

    failure_rate,

    timeout_rate,

):

    latency_score = np.clip(

        (

            latency - 40

        ) / 1200 * 100,

        0,

        100,

    )


    failure_score = np.clip(

        (

            failure_rate - 0.05

        ) / 0.45 * 100,

        0,

        100,

    )


    timeout_score = np.clip(

        (

            timeout_rate - 0.10

        ) / 0.40 * 100,

        0,

        100,

    )


    score = (

        0.45 * latency_score

        + 0.30 * failure_score

        + 0.25 * timeout_score

    )


    return round(

        np.clip(score, 0, 100),

        2,

    )


# ============================================================
# BANK DEGRADATION SCORE
# ============================================================

def calculate_bank_degradation(

    latency,

    failure_rate,

    timeout_rate,

):

    latency_score = np.clip(

        (

            latency - 200

        ) / 900 * 100,

        0,

        100,

    )


    failure_score = np.clip(

        (

            failure_rate - 0.10

        ) / 1.40 * 100,

        0,

        100,

    )


    timeout_score = np.clip(

        (

            timeout_rate - 1.00

        ) / 2.00 * 100,

        0,

        100,

    )


    score = (

        0.45 * latency_score

        + 0.25 * failure_score

        + 0.30 * timeout_score

    )


    return round(

        np.clip(score, 0, 100),

        2,

    )


# ============================================================
# ERROR CATEGORY
# ============================================================

def generate_error_category(

    network_degraded,

    gateway_degraded,

    bank_degraded,

):

    if network_degraded:

        return np.random.choice(

            [0, 1],

            p=[0.30, 0.70],

        )


    if gateway_degraded:

        return np.random.choice(

            [0, 2],

            p=[0.25, 0.75],

        )


    if bank_degraded:

        return np.random.choice(

            [0, 3],

            p=[0.20, 0.80],

        )


    return 0


# ============================================================
# GENERATE DATA
# ============================================================

rows = []


# ============================================================
# CLASS 1
# BANK DEGRADATION
# ============================================================

for _ in range(CLASS_1_COUNT):

    generation = choose_network_generation()


    network_latency, packet_loss, network_jitter = (

        generate_network_telemetry(

            generation,

            network_degraded=False,

        )

    )


    gateway_latency, gateway_failure, gateway_timeout = (

        generate_gateway_telemetry(

            gateway_degraded=False,

        )

    )


    bank_latency, bank_failure, bank_timeout = (

        generate_bank_telemetry(

            bank_degraded=True,

        )

    )


    network_deg = calculate_network_degradation(

        network_latency,

        packet_loss,

        network_jitter,

    )


    gateway_deg = calculate_gateway_degradation(

        gateway_latency,

        gateway_failure,

        gateway_timeout,

    )


    bank_deg = calculate_bank_degradation(

        bank_latency,

        bank_failure,

        bank_timeout,

    )


    error_category = generate_error_category(

        False,

        False,

        True,

    )


    is_timeout = int(

        bank_timeout >= 2.0

    )


    total_latency = (

        network_latency

        + gateway_latency

        + bank_latency

    )


    transaction_age = np.random.uniform(

        300,

        8000,

    )


    retry_count = np.random.choice(

        [0, 1, 2, 3],

        p=[0.35, 0.35, 0.20, 0.10],

    )


    order_amount = np.random.uniform(

        50,

        50000,

    )


    rows.append({

        "action_label": 1,

        "network_generation": generation,

        "network_latency_ms": network_latency,

        "packet_loss_pct": packet_loss,

        "network_jitter_ms": network_jitter,

        "gateway_latency_ms": gateway_latency,

        "gateway_failure_rate_pct": gateway_failure,

        "gateway_timeout_rate_pct": gateway_timeout,

        "bank_latency_ms": bank_latency,

        "bank_failure_rate_pct": bank_failure,

        "bank_timeout_rate_pct": bank_timeout,

        "total_latency_ms": round(

            total_latency,

            2,

        ),

        "transaction_age_ms": round(

            transaction_age,

            2,

        ),

        "retry_count": retry_count,

        "order_amount_inr": round(

            order_amount,

            2,

        ),

        "error_code_category": error_category,

        "is_timeout_flag": is_timeout,

        "true_network_degradation": network_deg,

        "true_gateway_degradation": gateway_deg,

        "true_bank_degradation": bank_deg,

    })


# ============================================================
# CLASS 2
# GATEWAY DEGRADATION
# ============================================================

for _ in range(CLASS_2_COUNT):

    generation = choose_network_generation()


    network_latency, packet_loss, network_jitter = (

        generate_network_telemetry(

            generation,

            network_degraded=False,

        )

    )


    gateway_latency, gateway_failure, gateway_timeout = (

        generate_gateway_telemetry(

            gateway_degraded=True,

        )

    )


    bank_latency, bank_failure, bank_timeout = (

        generate_bank_telemetry(

            bank_degraded=False,

        )

    )


    network_deg = calculate_network_degradation(

        network_latency,

        packet_loss,

        network_jitter,

    )


    gateway_deg = calculate_gateway_degradation(

        gateway_latency,

        gateway_failure,

        gateway_timeout,

    )


    bank_deg = calculate_bank_degradation(

        bank_latency,

        bank_failure,

        bank_timeout,

    )


    error_category = generate_error_category(

        False,

        True,

        False,

    )


    is_timeout = int(

        gateway_timeout >= 0.35

    )


    total_latency = (

        network_latency

        + gateway_latency

        + bank_latency

    )


    transaction_age = np.random.uniform(

        300,

        8000,

    )


    retry_count = np.random.choice(

        [0, 1, 2, 3],

        p=[0.35, 0.35, 0.20, 0.10],

    )


    order_amount = np.random.uniform(

        50,

        50000,

    )


    rows.append({

        "action_label": 2,

        "network_generation": generation,

        "network_latency_ms": network_latency,

        "packet_loss_pct": packet_loss,

        "network_jitter_ms": network_jitter,

        "gateway_latency_ms": gateway_latency,

        "gateway_failure_rate_pct": gateway_failure,

        "gateway_timeout_rate_pct": gateway_timeout,

        "bank_latency_ms": bank_latency,

        "bank_failure_rate_pct": bank_failure,

        "bank_timeout_rate_pct": bank_timeout,

        "total_latency_ms": round(

            total_latency,

            2,

        ),

        "transaction_age_ms": round(

            transaction_age,

            2,

        ),

        "retry_count": retry_count,

        "order_amount_inr": round(

            order_amount,

            2,

        ),

        "error_code_category": error_category,

        "is_timeout_flag": is_timeout,

        "true_network_degradation": network_deg,

        "true_gateway_degradation": gateway_deg,

        "true_bank_degradation": bank_deg,

    })


# ============================================================
# CLASS 3
# NETWORK DEGRADATION
# ============================================================

for _ in range(CLASS_3_COUNT):

    generation = choose_network_generation()


    network_latency, packet_loss, network_jitter = (

        generate_network_telemetry(

            generation,

            network_degraded=True,

        )

    )


    gateway_latency, gateway_failure, gateway_timeout = (

        generate_gateway_telemetry(

            gateway_degraded=False,

        )

    )


    bank_latency, bank_failure, bank_timeout = (

        generate_bank_telemetry(

            bank_degraded=False,

        )

    )


    network_deg = calculate_network_degradation(

        network_latency,

        packet_loss,

        network_jitter,

    )


    gateway_deg = calculate_gateway_degradation(

        gateway_latency,

        gateway_failure,

        gateway_timeout,

    )


    bank_deg = calculate_bank_degradation(

        bank_latency,

        bank_failure,

        bank_timeout,

    )


    error_category = generate_error_category(

        True,

        False,

        False,

    )


    is_timeout = int(

        network_latency >= 500

    )


    total_latency = (

        network_latency

        + gateway_latency

        + bank_latency

    )


    transaction_age = np.random.uniform(

        300,

        8000,

    )


    retry_count = np.random.choice(

        [0, 1, 2, 3],

        p=[0.35, 0.35, 0.20, 0.10],

    )


    order_amount = np.random.uniform(

        50,

        50000,

    )


    rows.append({

        "action_label": 3,

        "network_generation": generation,

        "network_latency_ms": network_latency,

        "packet_loss_pct": packet_loss,

        "network_jitter_ms": network_jitter,

        "gateway_latency_ms": gateway_latency,

        "gateway_failure_rate_pct": gateway_failure,

        "gateway_timeout_rate_pct": gateway_timeout,

        "bank_latency_ms": bank_latency,

        "bank_failure_rate_pct": bank_failure,

        "bank_timeout_rate_pct": bank_timeout,

        "total_latency_ms": round(

            total_latency,

            2,

        ),

        "transaction_age_ms": round(

            transaction_age,

            2,

        ),

        "retry_count": retry_count,

        "order_amount_inr": round(

            order_amount,

            2,

        ),

        "error_code_category": error_category,

        "is_timeout_flag": is_timeout,

        "true_network_degradation": network_deg,

        "true_gateway_degradation": gateway_deg,

        "true_bank_degradation": bank_deg,

    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(rows)


# ============================================================
# SHUFFLE
# ============================================================

df = df.sample(

    frac=1,

    random_state=42,

).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

df.to_csv(

    "telemetry_data.csv",

    index=False,

)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 75)

print("RESQ-QR TELEMETRY DATASET GENERATED")

print("=" * 75)


print(

    "\nTotal samples:",

    len(df),

)


print("\nAction distribution:")

print(

    df["action_label"]

    .value_counts()

    .sort_index()

)


print("\nExpected:")

print("1 → NO_ACTION              = 3333")

print("2 → CONTEXTUAL_NUDGE       = 3333")

print("3 → GENERATE_DYNAMIC_QR    = 3334")


print("\nAction percentages:")

print(

    (

        df["action_label"]

        .value_counts(normalize=True)

        .sort_index()

        * 100

    ).round(2)

)


print("\nDegradation averages by action:")

print(

    df.groupby("action_label")[

        [

            "true_network_degradation",

            "true_gateway_degradation",

            "true_bank_degradation",

        ]

    ]

    .mean()

    .round(2)

)


print("\nMinimum degradation by action:")

print(

    df.groupby("action_label")[

        [

            "true_network_degradation",

            "true_gateway_degradation",

            "true_bank_degradation",

        ]

    ]

    .min()

    .round(2)

)


print("\nMaximum degradation by action:")

print(

    df.groupby("action_label")[

        [

            "true_network_degradation",

            "true_gateway_degradation",

            "true_bank_degradation",

        ]

    ]

    .max()

    .round(2)

)


print("\n✓ payment_stage is NOT present.")

print("\n✓ Dataset saved as telemetry_data.csv")

print("=" * 75)