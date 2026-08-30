import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 1000

# 1. Feature Generation
latencies = np.random.exponential(scale=1200, size=n_samples) + 50  
packet_losses = np.random.exponential(scale=4, size=n_samples)      
order_amounts = np.random.uniform(100, 15000, n_samples)
retry_counts = np.random.choice(
    [0, 1, 2, 3], size=n_samples, p=[0.7, 0.15, 0.1, 0.05]
)

# Error Codes: 0=NONE, 1=BAD_REQUEST, 2=TIMEOUT, 3=BANK_OFFLINE
error_codes = np.random.choice(
    [0, 1, 2, 3], size=n_samples, p=[0.6, 0.2, 0.15, 0.05]
)
is_timeouts = np.where(
    (latencies > 2500) | (error_codes == 2) | (packet_losses > 10), 1, 0
)

# 2. Target Variable Generation (Ground Truth Rules)
# Target Classes: 0 = NO_ACTION, 1 = CONTEXTUAL_NUDGE, 2 = GENERATE_DYNAMIC_QR
actions = np.zeros(n_samples, dtype=int)

for i in range(n_samples):
    if latencies[i] >= 2500 or is_timeouts[i] == 1 or error_codes[i] == 2:
        actions[i] = 2  # GENERATE_DYNAMIC_QR (Severe network issues)
    elif error_codes[i] == 1 or packet_losses[i] >= 5.0 or retry_counts[i] > 1:
        actions[i] = 1  # CONTEXTUAL_NUDGE (User issue or minor lag)
    else:
        actions[i] = 0  # NO_ACTION (Normal flow / Host offline)

# 3. Combine into Pandas DataFrame
df = pd.DataFrame({
    "latency_ms": np.round(latencies, 2),
    "packet_loss_pct": np.round(packet_losses, 2),
    "error_code_category": error_codes,
    "is_timeout_flag": is_timeouts,
    "order_amount_inr": np.round(order_amounts, 2),
    "retry_count": retry_counts,
    "action_label": actions
})

# Save to CSV
df.to_csv("telemetry_data.csv", index=False)
msg = (
    "✅ Real Telemetry Dataset successfully generated and saved to "
    "'telemetry_data.csv'."
)
print(msg)
print("\n--- First 5 Rows of Dataset ---")
print(df.head())