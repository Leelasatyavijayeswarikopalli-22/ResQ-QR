# ⚡ ResQ-QR  
### Intelligent Network Degradation Detection & QR-Based Payment Recovery
**Deployed Link:** https://resq-qr.streamlit.app/

> **Detect network degradation before repeated payment failures — then switch to a lightweight QR fallback.**

---

## 🚨 Problem Statement

Digital payments depend on multiple infrastructure layers: **network → payment gateway → bank**.

When the **network path degrades**, high latency, packet loss and jitter can cause payment failures or timeouts. Users often retry repeatedly, increasing frustration and unnecessary load.

India processes **700M+ UPI transactions per day** at current scale. NPCI reported **22.7 billion UPI transactions in June 2026**, averaging roughly **757 million transactions/day**.

There is no official public dataset containing the detailed network/gateway/bank telemetry required for this problem. Therefore, ResQ-QR uses a **synthetically generated telemetry dataset based on publicly available reference ranges and realistic relationships between payment infrastructure variables**.

> **Illustrative impact estimate:** If only 1% of daily transactions are affected by network-related degradation, that represents ~**7.5M transactions/day**. If a fallback successfully recovers even 20% of those cases, ResQ-QR could potentially save ~**1.5M payment attempts/day**.
> *These are scenario estimates, not official NPCI failure statistics.*

---

## 💡 Solution

ResQ-QR uses a **two-stage ML pipeline**:

```text
Raw Payment Telemetry
        ↓
   Regression
        ↓
Network / Gateway / Bank
Degradation Scores
        ↓
  Classification
        ↓
Recovery Decision
        ↓
Network Degraded?
        ↓
Dynamic Lightweight QR
```

The key idea is:

**Predict the infrastructure condition first → then decide the appropriate recovery action.**

---

## 🧠 Why Regression + Classification?

Real payment failures can involve **multiple degrading components at the same time**.

- **Example:** Network degradation = High, Bank degradation = High, Gateway degradation = Low.
  A simple threshold might trigger QR because the network is degraded. However, the bank is also degraded, so QR may still fail and cause unnecessary retries.
  
  **Regression** estimates the severity of all three components, while **Classification** considers their combined condition and selects the appropriate recovery action.

This avoids relying on a single fixed threshold such as *“if network degradation > X, generate QR”* and helps prevent unnecessary retries which is against to our goal.

**Regression measures severity; Classification makes the final recovery decision.**

### Regression

Predicts continuous degradation scores:

* Network degradation
* Gateway degradation
* Bank degradation

Example:

```text
Network  → 82.4%
Gateway  → 18.7%
Bank     → 11.2%
```

This preserves the **severity information**.

### 🧠 Classification

The three predicted degradation scores are passed to the classifier to select the safest recovery action.

| Class | Recovery Action         | What it means                                                                                                                  |
| ----- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 1     | **Contextual Nudge**    | Bank-side degradation is detected. Inform the user and avoid unnecessary repeated retries.                                     |
| 2     | **No Action**           | Gateway-side degradation is detected. Do not trigger an automatic fallback; allow the gateway path to recover.                 |
| 3     | **Generate Dynamic QR** | Network-side degradation is detected. Generate a lightweight dynamic QR so the payment can continue through the fallback path. |

This creates a clear **Diagnose → Decide → Recover** pipeline.

---
## Architecture
```text
                         RAW TELEMETRY
                              │
          ┌───────────────────┼───────────────────┐
          ↓                   ↓                   ↓
       NETWORK             GATEWAY               BANK
       Latency              Latency              Latency
       Packet Loss          Failure Rate         Failure Rate
       Jitter               Timeout Rate         Timeout Rate
          └───────────────────┬───────────────────┘
                              │
                    PAYMENT ERROR TELEMETRY
                              │
                    Error Code Category
                              │
                    Timeout Flag
                 (derived from error code)
                              │
                              ↓
                    REGRESSION MODELS
                              │
          ┌───────────────────┼───────────────────┐
          ↓                   ↓                   ↓
       Network             Gateway               Bank
      Degradation         Degradation          Degradation
          └───────────────────┼───────────────────┘
                              ↓
                         CLASSIFIER
                              ↓
                    RECOVERY DECISION
                              │
             ┌────────────────┼────────────────┐
             ↓                ↓                ↓
       Contextual          No Action       Generate
          Nudge                              Dynamic QR
                                               │
                                               ↓
                                  Lightweight Resolver
                                               │
                                               ↓
                                          UPI Payment
```

---

## 📊 Synthetic Data Generation

Since detailed real-world telemetry is not publicly available, the dataset was generated synthetically.

### Process

1. Collected realistic parameter ranges from publicly available reference material.
2. Generated **10,000 payment telemetry records**.
3. Created realistic relationships between:

   * latency
   * packet loss
   * jitter
   * gateway failures
   * gateway timeouts
   * bank failures
   * bank timeouts
   * payment errors
4. Generated three continuous degradation targets:

   * `true_network_degradation`
   * `true_gateway_degradation`
   * `true_bank_degradation`
5. Created recovery classes from the resulting degradation conditions.
6. Used a **70% training / 30% testing split**.
7. Compared **Random Forest and XGBoost**.

The dataset is therefore designed to reproduce realistic infrastructure behaviour rather than being random numerical data.

---

## 🤖 Model Selection

Two model families are benchmarked:

* **Random Forest**
* **XGBoost**

The final deployment uses the **better-performing model based on the benchmark results**, rather than assuming one algorithm is always superior.

### Regression

Evaluated using:

* MAE
* MSE
* RMSE
* R²

### Classification

Evaluated using:

* Accuracy
* Precision
* Recall
* Weighted F1
* Macro F1

Production-related metrics are also measured:

* P95 inference latency
* Training time
* Model size

---

## 📈 What Do the Metrics Tell Us?

ResQ-QR benchmarks **Random Forest and XGBoost** and selects the best model for each ML stage based on accuracy, prediction quality and deployment performance.

### Regression — Infrastructure Degradation Prediction

| Model | MAE | RMSE | R² | P95 Inference |
|---|---:|---:|---:|---:|
| Random Forest | 2.24319 | 2.87269 | 0.98769 | 1.0659 ms |
| **XGBoost** | **0.27108** | **0.43873** | **0.99972** | **0.1319 ms** |

**XGBoost is selected for regression.**

- **MAE = 0.27108** → very small average degradation prediction error.
- **RMSE = 0.43873** → low error even when larger deviations are considered.
- **R² = 0.99972** → the model explains almost all variation in the generated degradation data.
- **P95 = 0.1319 ms** → fast inference suitable for real-time decision making.

### Classification — Recovery Action

| Model | Accuracy | Macro F1 | Precision | Recall | P95 Inference |
|---|---:|---:|---:|---:|---:|
| **Random Forest** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | 0.7914 ms |
| XGBoost | 99.93% | 0.9993 | 0.9993 | 0.9993 | **0.0491 ms** |

**Random Forest is selected for classification** because it achieved the highest overall predictive performance on the generated test data.

- **100% Accuracy** → all test cases were classified correctly.
- **Macro F1 = 1.0000** → balanced performance across all recovery classes.
- **Precision = 1.0000** → predicted recovery actions were correct.
- **Recall = 1.0000** → relevant recovery cases were successfully detected.

### Why These Results Matter

The regression model provides highly accurate continuous degradation estimates, while the classifier converts those estimates into an appropriate recovery action.

This two-stage approach allows ResQ-QR to distinguish between **network, gateway and bank degradation patterns** instead of relying on a single fixed threshold.

### Production Metrics

- **Regression P95: 0.1319 ms** — fast degradation prediction.
- **Classification P95: 0.7914 ms** — fast recovery-action selection.
- **Regression model size: 3.127 MB**
- **Classification model size: 0.754 MB**

> **Note:** These results are measured on the generated synthetic test dataset. They demonstrate the model's performance on the designed telemetry patterns and should not be interpreted as real-world accuracy on production UPI traffic.

---

## 📱 Lightweight Dynamic QR

When **network degradation is detected**, ResQ-QR generates a dynamic QR fallback.

Instead of placing a large payment payload directly inside the QR, ResQ-QR stores the payment session and puts only a **short resolver URL** into the QR.

```text
Full Payment Session
        ↓
Stored on Resolver
        ↓
Short Payment Token
        ↓
Lightweight Resolver URL
        ↓
QR Code
        ↓
Resolve Token
        ↓
UPI Payment Link
```

### Why?

A smaller QR payload means:

* less QR data density
* easier scanning
* faster generation
* cleaner QR representation
* better suitability for degraded connectivity scenarios

The QR is generated **only when the ML decision identifies network degradation**.

---

## 🧪 Test the System

The Streamlit application allows live testing using telemetry values.

### Network-degradation test

Try high:

```text
Network Latency
Packet Loss
Network Jitter
```

Expected behaviour:

```text
High Network Degradation
        ↓
Class 3
        ↓
GENERATE DYNAMIC QR
```

### Gateway-degradation test

Increase:

```text
Gateway Latency
Gateway Failure Rate
Gateway Timeout Rate
```

Expected recovery:

```text
Gateway Degradation
        ↓
NO ACTION
```

### Bank-degradation test

Increase:

```text
Bank Latency
Bank Failure Rate
Bank Timeout Rate
```

Expected recovery:

```text
Bank Degradation
        ↓
CONTEXTUAL NUDGE
```

### Insufficient-Funds Test

The error code has four values: `None`, `BANK_OFFLINE`, `TIMEOUT`, and `BAD_FUNDS`.

- `None`, `BANK_OFFLINE`, and `TIMEOUT` → handled by the **classification model** along with degradation scores to select the recovery action.
- `BAD_FUNDS` → unrelated to infrastructure degradation, so it **bypasses the recovery decision and never generates a QR**.

---

## 🖥️ Application

Built with:

* Python
* Streamlit
* Scikit-learn
* XGBoost
* Pandas
* NumPy
* Plotly
* Joblib
* QRCode

### Application Modules

**🚀 Live Payment Engine**
Enter telemetry and receive the ML recovery decision.

**📊 Model Performance**
Compare Random Forest vs XGBoost using model and production metrics.

**🧠 Feature Intelligence**
Explore features, degradation predictions and classifier feature importance.

---

## ⭐ Key Innovation

ResQ-QR does not simply predict **"payment failed."**

It identifies:

> **Which infrastructure layer is degrading → how severely → what action should be taken → whether a QR fallback can recover the payment.**

This makes the system a **data-driven payment recovery engine**, rather than a basic payment-failure detector.

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Required model files:

```text
regression_model.pkl
classification_model.pkl
benchmark_results.json
```

---

## 🎯 One-Line Summary

**ResQ-QR uses regression to quantify payment infrastructure degradation, classification to choose the safest recovery action, and a lightweight dynamic QR to recover payments specifically when the network path is degraded.**
