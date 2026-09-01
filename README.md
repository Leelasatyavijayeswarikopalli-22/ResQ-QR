# ⚡ ResQ-QR

### Intelligent Network Degradation Detection & QR-Based Payment Recovery

ResQ-QR is an ML-based payment recovery system that **detects network, gateway, and bank-side degradation from payment telemetry and selects an appropriate recovery action**.

## 🎯 Problem

Digital payments can fail because of:

* 📡 Network degradation
* 🔌 Payment gateway degradation
* 🏦 Bank-side degradation

A generic retry cannot identify **where the failure is occurring**, and repeated retries can worsen the user experience.

## 💡 Solution

ResQ-QR uses a **two-stage ML pipeline**:

```text
Raw Payment Telemetry
        ↓
   Regression
        ↓
Network | Gateway | Bank
Degradation Scores
        ↓
  Classification
        ↓
 Recovery Action
        ↓
Dynamic QR if Network is Degraded
```

## 🧠 Why Regression + Classification?

**Regression answers:**

> *How much is each infrastructure component degraded?*

It predicts continuous degradation scores for:

* Network
* Gateway
* Bank

**Classification answers:**

> *What should the system do now?*

It converts the three degradation scores into a recovery decision.

This separation makes the system **data-driven, interpretable, and closer to real-world payment decision-making**.

## 📡 ML Inputs

The regression stage uses **11 telemetry features**:

* Network latency
* Packet loss
* Network jitter
* Gateway latency
* Gateway failure rate
* Gateway timeout rate
* Bank latency
* Bank failure rate
* Bank timeout rate
* Error code category
* Timeout flag

## 🔄 Recovery Actions

| Class | Action              |
| ----- | ------------------- |
| **1** | Contextual Nudge    |
| **2** | No Action           |
| **3** | Generate Dynamic QR |

### 📱 Key Innovation

**Dynamic QR is generated only when network degradation is detected.**

The QR contains a **lightweight resolver URL**, which retrieves the payment session and constructs the UPI payment link.

This avoids placing the complete payment payload inside the QR.

## 📊 Model Evaluation

The system benchmarks **Random Forest and XGBoost** using:

* Classification: Accuracy, Precision, Recall, F1
* Regression: MAE, MSE, RMSE, R²
* Production: P95 inference latency
* Deployment: Model size

## 🏗️ Tech Stack

* **Python**
* **Scikit-learn**
* **XGBoost**
* **Pandas / NumPy**
* **Streamlit**
* **Plotly**
* **QR Code / UPI Deep Link**

## 🚀 Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Required model files:

```text
classification_model.pkl
regression_model.pkl
benchmark_results.json
```

## 🏆 Hackathon Value

**ResQ-QR moves payment recovery from blind retries to telemetry-driven decisions.**

It identifies **what is degrading → quantifies the degradation → chooses the recovery action → provides QR fallback when the network path is the problem.**
