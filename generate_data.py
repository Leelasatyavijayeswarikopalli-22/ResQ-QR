import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 2000

# 1. Feature Generation: Granular Payment Telemetry
gateway_latencies = np.random.exponential(scale=800, size=n_samples) + 20
bank_latencies = np.random.exponential(scale=1000, size=n_samples) + 30
total_latencies = gateway_latencies + bank_latencies

packet_losses = np.random.exponential(scale=3.5, size=n_samples)
order_amounts = np.random.uniform(100, 15000, n_samples)
retry_counts = np.random.choice(
    [0, 1, 2, 3], size=n_samples, p=[0.70, 0.15, 0.10, 0.05]
)

# Payment Stages: 0=INITIATED, 1=AUTH (OTP/PIN), 2=AUTHORIZING, 3=SETTLING
payment_stages = np.random.choice(
    [0, 1, 2, 3], size=n_samples, p=[0.1, 0.4, 0.4, 0.1]
)

# Error Categories: 0=NONE, 1=BAD_FUNDS, 2=TIMEOUT, 3=BANK_OFFLINE
error_codes = np.random.choice(
    [0, 1, 2, 3], size=n_samples, p=[0.50, 0.25, 0.15, 0.10]
)

# Timeout Flag: Triggered if latency exceeds threshold or timeout error
is_timeouts = np.where(
    (
        (total_latencies > 2500)
        | (error_codes == 2)
        | (packet_losses > 10.0)
    )
    & (~np.isin(error_codes, [1, 3])),
    1,
    0,
)

# 2. Target Variable Generation (Ground Truth Rules Hierarchy)
actions = np.zeros(n_samples, dtype=int)

for i in range(n_samples):
    # PRIORITY 1: User balance error -> ALWAYS CONTEXTUAL_NUDGE
    if error_codes[i] == 1:
        actions[i] = 1
    # PRIORITY 2: Issuer Bank Offline -> ALWAYS NO_ACTION
    elif error_codes[i] == 3:
        actions[i] = 0
    # PRIORITY 3: Gateway Timeout or Network Degradation
    # -> GENERATE_DYNAMIC_QR
    elif (
        error_codes[i] == 2
        or total_latencies[i] >= 2500
        or is_timeouts[i] == 1
        or packet_losses[i] >= 10.0
    ):
        actions[i] = 2
    # PRIORITY 4: Minor packet loss / retries -> CONTEXTUAL_NUDGE
    elif packet_losses[i] >= 5.0 or retry_counts[i] > 1:
        actions[i] = 1
    # PRIORITY 5: Clean path -> NO_ACTION
    else:
        actions[i] = 0

# 3. Combine into Pandas DataFrame
df = pd.DataFrame({
    "gateway_latency_ms": np.round(gateway_latencies, 2),
    "bank_latency_ms": np.round(bank_latencies, 2),
    "total_latency_ms": np.round(total_latencies, 2),
    "packet_loss_pct": np.round(packet_losses, 2),
    "payment_stage": payment_stages,
    "error_code_category": error_codes,
    "is_timeout_flag": is_timeouts,
    "order_amount_inr": np.round(order_amounts, 2),
    "retry_count": retry_counts,
    "action_label": actions
})

df.to_csv("telemetry_data.csv", index=False)
msg = (
    "✅ Real-time telemetry dataset generated successfully "
    "and saved to 'telemetry_data.csv'."
)
print(msg)
print("\n--- First 5 Rows of Dataset ---")
print(df.head())
