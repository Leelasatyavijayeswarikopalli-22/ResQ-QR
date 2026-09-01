# ⚡ ResQ-QR

### Intelligent Network Degradation Detection & QR-Based Payment Recovery

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

Using only classification would directly predict an action from raw telemetry.

ResQ-QR separates the problem because the real-world system first needs to understand **what is degrading and by how much**.

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

### Classification

The three predicted degradation scores are then used as classifier inputs to select the recovery action:

| Class | Action              |
| ----- | ------------------- |
| 1     | Contextual Nudge    |
| 2     | No Action           |
| 3     | Generate Dynamic QR |

This mirrors a real decision system:

**Measure → Diagnose → Decide → Recover**

---

## 🏗️ Architecture

```text
                 RAW TELEMETRY
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
     Network         Gateway          Bank
     Latency         Latency         Latency
     Packet Loss     Failures        Failures
     Jitter          Timeouts        Timeouts
        └──────────────┼──────────────┘
                       ↓
                REGRESSION MODELS
                       ↓
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
     Network        Gateway          Bank
    Degradation    Degradation    Degradation
        └──────────────┼──────────────┘
                       ↓
                 CLASSIFIER
                       ↓
             Recovery Decision
                       ↓
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       Nudge        No Action    Network QR
                                    ↓
                           Lightweight Resolver
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

### Regression

**MAE / RMSE → prediction error**

Lower values mean the predicted degradation is closer to the actual degradation.

**R² → explained variation**

Closer to `1` means the model explains the degradation patterns better.

### Classification

**Accuracy → overall correct decisions**

**Precision → how reliable predicted actions are**

**Recall → how many relevant cases are detected**

**Macro F1 → balanced performance across all recovery classes**

This is important because incorrectly missing a network-degradation case can prevent the QR fallback from being triggered.

### Production Metrics

**P95 inference latency** shows how quickly the model responds in near-worst-case normal conditions.

**Model size** indicates deployment memory/storage requirements.

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

### Insufficient-funds test

Select:

```text
1 — BAD_FUNDS
```

The system handles insufficient funds separately because a QR fallback cannot solve a genuine funds-related rejection.

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
