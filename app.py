import io
import json
import os
import joblib
import numpy as np
import pandas as pd
import qrcode
import streamlit as st
import plotly.express as px
st.set_page_config(
    page_title="ResQ-QR | Intelligent Payment Recovery",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
FEATURE_COLS = [
    "network_latency_ms",
    "packet_loss_pct",
    "network_jitter_ms",
    "gateway_latency_ms",
    "gateway_failure_rate_pct",
    "gateway_timeout_rate_pct",
    "bank_latency_ms",
    "bank_failure_rate_pct",
    "bank_timeout_rate_pct",
    "total_latency_ms",
    "payment_stage",
    "transaction_age_ms",
    "retry_count",
    "order_amount_inr",
    "error_code_category",
    "is_timeout_flag",
]
REGRESSION_TARGETS = [
    "true_network_degradation",
    "true_gateway_degradation",
    "true_bank_degradation",
]
ACTION_NAMES = [
    "NO_ACTION",
    "CONTEXTUAL_NUDGE",
    "RETRY_PAYMENT",
    "GENERATE_DYNAMIC_QR",
]
ACTION_DESCRIPTIONS = {
    0: {
        "name": "NO ACTION",
        "emoji": "🛑",
        "color": "#ef4444",
        "meaning": (
            "The payment should not be retried automatically because"
            " another attempt may increase the risk of duplicate"
            " debits or cannot recover the transaction."
        ),
        "error": (
            "This usually indicates an unrecoverable banking-side"
            " problem such as an unavailable issuer bank."
        ),
        "solution": (
            "Wait for the bank or payment issuer to recover instead"
            " of repeatedly retrying."
        ),
    },
    1: {
        "name": "CONTEXTUAL NUDGE",
        "emoji": "🟠",
        "color": "#f59e0b",
        "meaning": (
            "The payment problem is likely recoverable by asking the"
            " customer to correct something on their side."
        ),
        "error": (
            "This can indicate insufficient funds, invalid payment"
            " information, or another customer-side payment problem."
        ),
        "solution": (
            "Notify the customer and guide them to correct the"
            " payment issue."
        ),
    },
    2: {
        "name": "RETRY PAYMENT",
        "emoji": "🔄",
        "color": "#3b82f6",
        "meaning": (
            "The payment appears temporarily recoverable, so another"
            " controlled payment attempt is appropriate."
        ),
        "error": (
            "This commonly represents a temporary gateway failure,"
            " transient latency spike, timeout, or communication"
            " problem."
        ),
        "solution": (
            "Retry the payment using controlled retry limits to avoid"
            " duplicate transactions."
        ),
    },
    3: {
        "name": "GENERATE DYNAMIC QR",
        "emoji": "🟢",
        "color": "#10b981",
        "meaning": (
            "The normal payment path is degraded, so a lightweight QR"
            " fallback can allow the customer to continue the payment."
        ),
        "error": (
            "This can indicate high network latency, packet loss,"
            " jitter, or gateway communication degradation."
        ),
        "solution": (
            "Move the customer to the QR fallback instead of"
            " repeatedly waiting for the degraded payment connection."
        ),
    },
}


def load_file(path, default=None):
    if os.path.exists(path):
        return joblib.load(path)
    return default


@st.cache_resource
def load_models():
    classifier = load_file("classification_model.pkl")
    regression = load_file("regression_model.pkl")
    return classifier, regression


@st.cache_data
def load_benchmark():
    if os.path.exists("benchmark_results.json"):
        with open("benchmark_results.json", "r") as f:
            return json.load(f)
    return {}


classifier, regression_model = load_models()
benchmark = load_benchmark()
if classifier is None:
    st.error(
        "classification_model.pkl was not found. "
        "Run your model training script first."
    )
    st.stop()
if regression_model is None:
    st.error(
        "regression_model.pkl was not found. "
        "Run your model training script first."
    )
    st.stop()
st.markdown("""
<style>
.stApp{
    background:linear-gradient(135deg,#070b1f 0%,#111a3a 45%,#062d3d 100%);
    color:#f8fafc;
}
.block-container{
    padding-top:2rem;
    padding-bottom:3rem;
    max-width:1450px;
}
.hero{
    background:linear-gradient(135deg,#7c3aed,#2563eb,#06b6d4);
    padding:38px;
    border-radius:28px;
    margin-bottom:25px;
    box-shadow:0 18px 50px rgba(0,0,0,.35);
    border:1px solid rgba(255,255,255,.2);
}
.hero h1{
    font-size:3.2rem;
    margin:0;
    color:white;
    font-weight:900;
}
.hero p{
    font-size:1.2rem;
    color:#e0f2fe;
    margin-top:8px;
}
.online{
    display:inline-block;
    background:#10b981;
    color:white;
    padding:8px 16px;
    border-radius:30px;
    font-weight:800;
    margin-top:12px;
}
.card{
    background:rgba(255,255,255,.09);
    border:1px solid rgba(255,255,255,.15);
    border-radius:22px;
    padding:24px;
    margin-bottom:18px;
    box-shadow:0 12px 35px rgba(0,0,0,.22);
}
.card h3{
    color:white;
    margin-top:0;
}
.section-title{
    font-size:1.8rem;
    font-weight:900;
    color:white;
    margin:25px 0 15px;
}
.action-card{
    border-radius:24px;
    padding:28px;
    color:white;
    box-shadow:0 18px 40px rgba(0,0,0,.35);
    margin-top:15px;
}
.reason-box{
    background:linear-gradient(135deg,rgba(30,41,59,.95),rgba(15,23,42,.9));
    border-left:6px solid #38bdf8;
    padding:22px;
    border-radius:16px;
    margin-top:22px;
}
.error-box{
    background:linear-gradient(135deg,rgba(239,68,68,.14),rgba(127,29,29,.18));
    border:1px solid rgba(248,113,113,.45);
    padding:20px;
    border-radius:16px;
    margin-top:15px;
}
.solution-box{
    background:linear-gradient(135deg,rgba(16,185,129,.14),rgba(6,78,59,.2));
    border:1px solid rgba(52,211,153,.45);
    padding:20px;
    border-radius:16px;
    margin-top:15px;
}
.success-box{
    background:linear-gradient(135deg,rgba(16,185,129,.14),rgba(6,78,59,.2));
    border:1px solid rgba(52,211,153,.45);
    padding:18px;
    border-radius:16px;
}
.info-box{
    background:linear-gradient(135deg,rgba(59,130,246,.14),rgba(30,58,138,.2));
    border:1px solid rgba(96,165,250,.45);
    padding:18px;
    border-radius:16px;
}
.metric-card{
    background:rgba(255,255,255,.08);
    border:1px solid rgba(255,255,255,.12);
    border-radius:18px;
    padding:18px;
    text-align:center;
}
.metric-title{
    color:#cbd5e1;
    font-size:.9rem;
}
.metric-value{
    color:white;
    font-size:1.6rem;
    font-weight:900;
}
div[data-testid="stMetric"]{
    background:rgba(255,255,255,.08);
    padding:15px;
    border-radius:15px;
    border:1px solid rgba(255,255,255,.12);
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="hero">
    <h1>⚡ ResQ-QR</h1>
    <p>Intelligent Payment Recovery & Dynamic Fallback Engine</p>
    <span class="online">● ML ENGINE ONLINE</span>
</div>
""", unsafe_allow_html=True)
winner_class = benchmark.get("classification_winner", "Unknown")
winner_reg = benchmark.get("regression_winner", "Unknown")
st.markdown(f"""
<div class="success-box">
    <b>🏆 Active Classification Engine:</b> {winner_class}
    &nbsp;&nbsp; | &nbsp;&nbsp;
    <b>📈 Active Regression Engine:</b> {winner_reg}
</div>
""", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(
    [
        "🚀 LIVE PAYMENT ENGINE",
        "📊 MODEL PERFORMANCE",
        "🧠 FEATURE INTELLIGENCE",
    ]
)
with tab1:
    st.markdown(
        '<div class="section-title">🚀 Live Payment '
        'Decision Simulator</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown(
            '<div class="card"><h3>💳 1. Transaction Details</h3>',
            unsafe_allow_html=True,
        )
        order_amount = st.number_input(
            "Transaction Value (₹)",
            min_value=1.0,
            value=499.0,
            step=50.0,
        )
        payment_stage_str = st.selectbox(
            "Payment Lifecycle Stage",
            [
                "0: INITIATED — Checkout launched",
                "1: AUTHENTICATING — PIN / OTP",
                "2: AUTHORIZING — PSP ↔ Bank handshake",
                "3: SETTLING — Final transaction confirmation",
            ],
        )
        payment_stage = int(payment_stage_str.split(":")[0])
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="card"><h3>🌐 2. Network Telemetry</h3>',
            unsafe_allow_html=True,
        )
        network_latency = st.slider(
            "Network Latency (ms)", 0, 3000, 500, 25
        )
        packet_loss = st.slider("Packet Loss (%)", 0.0, 30.0, 3.0, 0.5)
        network_jitter = st.slider(
            "Network Jitter (ms)", 0.0, 500.0, 30.0, 5.0
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="card"><h3>🏦 3. Gateway & Bank Telemetry</h3>',
            unsafe_allow_html=True,
        )
        gateway_latency = st.slider("Gateway Latency (ms)", 10, 3000, 600, 25)
        gateway_failure = st.slider(
            "Gateway Failure Rate (%)", 0.0, 30.0, 2.0, 0.5
        )
        gateway_timeout = st.slider(
            "Gateway Timeout Rate (%)", 0.0, 30.0, 2.0, 0.5
        )
        bank_latency = st.slider("Bank Latency (ms)", 10, 3000, 700, 25)
        bank_failure = st.slider("Bank Failure Rate (%)", 0.0, 30.0, 2.0, 0.5)
        bank_timeout = st.slider("Bank Timeout Rate (%)", 0.0, 30.0, 2.0, 0.5)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="card"><h3>🔁 4. Transaction Behaviour</h3>',
            unsafe_allow_html=True,
        )
        transaction_age = st.slider("Transaction Age (ms)", 0, 10000, 500, 100)
        retry_count = st.select_slider("Retry Attempts", [0, 1, 2, 3], value=0)
        error_type = st.selectbox(
            "Simulated Payment Error",
            [
                "NONE — No explicit error",
                "TIMEOUT — Gateway request timed out",
                "BAD_FUNDS — Insufficient funds / user-side issue",
                "BANK_OFFLINE — Issuer bank unavailable",
            ],
        )
        if "TIMEOUT" in error_type:
            error_category = 2
            timeout_flag = 1
        elif "BAD_FUNDS" in error_type:
            error_category = 1
            timeout_flag = 0
        elif "BANK_OFFLINE" in error_type:
            error_category = 3
            timeout_flag = 0
        else:
            error_category = 0
            timeout_flag = 0
        total_latency = network_latency + gateway_latency + bank_latency
        st.info(
            f"⏱️ Combined payment-path latency: **{total_latency:,} ms**"
        )
        run_sim = st.button(
            "⚡ ANALYZE PAYMENT", type="primary", use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown(
            '<div class="section-title">🧠 AI Decision Engine</div>',
            unsafe_allow_html=True,
        )
        if run_sim:
            input_values = [
                [
                    network_latency,
                    packet_loss,
                    network_jitter,
                    gateway_latency,
                    gateway_failure,
                    gateway_timeout,
                    bank_latency,
                    bank_failure,
                    bank_timeout,
                    total_latency,
                    payment_stage,
                    transaction_age,
                    retry_count,
                    order_amount,
                    error_category,
                    timeout_flag,
                ]
            ]
            input_df = pd.DataFrame(input_values, columns=FEATURE_COLS)
            pred_class = int(classifier.predict(input_df)[0])
            pred_probs = classifier.predict_proba(input_df)[0]
            confidence = float(np.max(pred_probs) * 100)
            action = ACTION_DESCRIPTIONS.get(
                pred_class, ACTION_DESCRIPTIONS[0]
            )
            st.markdown(
                f"""
            <div class="action-card" 
            style="background:linear-gradient(135deg,{action["color"]},
            #111827);">
                <div style="font-size:1rem;opacity:.85;font-weight:700;">
                AI RECOMMENDED SYSTEM ACTION</div>
                <div style="font-size:2.3rem;font-weight:900;margin:8px 0;">
                {action["emoji"]} {action["name"]}</div>
                <div style="font-size:1.05rem;">Model confidence: 
                <b>{confidence:.1f}%</b></div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-title">🎯 What the AI Thinks</div>',
                unsafe_allow_html=True,
            )
            probability_df = pd.DataFrame(
                {"Action": ACTION_NAMES, "Probability": pred_probs * 100}
            )
            fig = px.pie(
                probability_df,
                names="Action",
                values="Probability",
                hole=0.48,
                title="Recovery Action Probability",
            )
            fig.update_traces(textinfo="label+percent", textposition="outside")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                legend_font_color="white",
                margin=dict(t=70, b=30, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                (
                  '<div class="section-title">'
                  '📈 Infrastructure Degradation Probability'
                  '</div>'
                ),
                unsafe_allow_html=True,
            )
            regression_input = input_df
            try:
                if (
                    isinstance(regression_model, dict)
                    and "models" in regression_model
                ):
                    degradation_values = []
                    for target in REGRESSION_TARGETS:
                        value = float(
                            regression_model["models"][target].predict(
                                regression_input
                            )[0]
                        )
                        degradation_values.append(value)
                    degradation_values = np.clip(
                        np.array(degradation_values), 0, 1
                    )
                    degradation_df = pd.DataFrame(
                        {
                            "Component": [
                                "🌐 Network",
                                "🏦 Gateway",
                                "🏛️ Bank",
                            ],
                            "Degradation Probability": 
                                degradation_values * 100,
                        }
                    )
                    fig2 = px.bar(
                        degradation_df,
                        x="Component",
                        y="Degradation Probability",
                        text="Degradation Probability",
                        range_y=[0, 100],
                        title="Predicted Infrastructure Degradation",
                    )
                    fig2.update_traces(
                        texttemplate="%{text:.1f}%", textposition="outside"
                    )
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        yaxis_title="Probability (%)",
                        xaxis_title="",
                        margin=dict(t=70, b=30, l=20, r=20),
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.warning(
                    f"Regression prediction could not be displayed: {e}"
                )
            st.markdown(
                '<div class="section-title">🔎 AI Explanation</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
            <div class="reason-box">
                <h3>🧠 Why did the AI choose this?</h3>
                <p><b>Decision:</b> {action["name"]}</p>
                <p><b>What it means:</b> {action["meaning"]}</p>
            </div>
            <div class="error-box">
                <h3>🚨 What does this error mean?</h3>
                <p>{action["error"]}</p>
            </div>
            <div class="solution-box">
                <h3>✅ What should the system do?</h3>
                <p>{action["solution"]}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
            if pred_class == 3:
                deeplink = (
                    f"upi://pay?pa=resqstore@upi&pn=ResQStore&am="
                    f"{order_amount}&tr=TXN98765&cu=INR"
                )
                qr = qrcode.QRCode(version=1, box_size=6, border=2)
                qr.add_data(deeplink)
                qr.make(fit=True)
                img = qr.make_image(
                    fill_color="black", back_color="white"
                )
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                qr_bytes = buf.getvalue()
                st.markdown(
                    """
                <div class="success-box">
                    <h3>📱 Dynamic QR Fallback Generated</h3>
                    <p>The normal payment path is degraded. ResQ-QR has
                    generated a lightweight payment fallback.</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                c1, c2, c3 = st.columns([1, 1, 1])
                with c2:
                    st.image(qr_bytes, width=240)
                    st.caption(
                        f"Payload size: {len(qr_bytes) / 1024:.2f} KB"
                    )
        else:
            st.markdown("""
            <div class="card">
                <h2>👈 Configure Payment Telemetry</h2>
                <p style="color:#cbd5e1;font-size:1.1rem;">
                    Adjust the network, gateway, bank and transaction
                    conditions on the left.
                </p>
                <p style="color:#94a3b8;">
                    Then click <b>⚡ ANALYZE PAYMENT</b> to let the ML engine
                    determine the safest recovery action.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
with tab2:
    st.markdown(
        '<div class="section-title">📊 Model Performance Laboratory</div>',
        unsafe_allow_html=True,
    )
    if benchmark:
        classification = benchmark.get("classification", {})
        regression = benchmark.get("regression", {})
        st.markdown(
            """
        <div class="card">
            <h3>🤖 Classification: Random Forest vs XGBoost</h3>
            <p style="color:#cbd5e1;">
                The classification engine selects the safest payment recovery
                action.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if classification:
            rows = []
            for name, data in classification.items():
                rows.append(
                    {
                        "Model": name,
                        "Accuracy (%)": data.get("accuracy", 0) * 100,
                        "Weighted F1": data.get("weighted_f1", 0),
                        "Macro F1": data.get("macro_f1", 0),
                        "Precision": data.get("precision_weighted", 0),
                        "Recall": data.get("recall_weighted", 0),
                        "Train Time (ms)": data.get("training_time_ms", 0),
                        "P95 Inference (ms)": data.get("inference_p95_ms", 0),
                        "Model Size (MB)": data.get("model_size_mb", 0),
                    }
                )
            class_df = pd.DataFrame(rows)
            st.dataframe(
                class_df.style.format(
                    {
                        "Accuracy (%)": "{:.2f}",
                        "Weighted F1": "{:.4f}",
                        "Macro F1": "{:.4f}",
                        "Precision": "{:.4f}",
                        "Recall": "{:.4f}",
                        "Train Time (ms)": "{:.2f}",
                        "P95 Inference (ms)": "{:.4f}",
                        "Model Size (MB)": "{:.3f}",
                    }
                ),
                use_container_width=True,
            )
            metric_choice = st.selectbox(
                "Classification metric",
                [
                    "Accuracy (%)",
                    "Weighted F1",
                    "Macro F1",
                    "Precision",
                    "Recall",
                    "P95 Inference (ms)",
                ],
                key="classification_metric",
            )
            fig = px.bar(
                class_df,
                x="Model",
                y=metric_choice,
                text=metric_choice,
                color="Model",
                title=f"Classification — {metric_choice}",
            )
            fig.update_traces(
                texttemplate="%{text:.3f}", textposition="outside"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"""
        <div class="success-box">
            <b>🏆 Classification Winner: {winner_class}</b>
            <br>
            Macro F1 is prioritized because every payment action should be
            represented fairly rather than allowing a majority class to
            dominate the evaluation.
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
        <div class="card">
            <h3>📈 Regression: Random Forest vs XGBoost</h3>
            <p style="color:#cbd5e1;">
                The regression engine estimates continuous infrastructure
                degradation probability.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if regression:
            rows = []
            for name, data in regression.items():
                rows.append({"Model": name, "MAE": data.get("mae", 0),"MSE":data.get("mse",0),"RMSE":data.get("rmse",0),"R²":data.get("r2",0),"Train Time (ms)":data.get("training_time_ms",0),"P95 Inference (ms)":data.get("inference_p95_ms",0),"Model Size (MB)":data.get("model_size_mb",0)})
            reg_df=pd.DataFrame(rows)
            st.dataframe(reg_df.style.format({"MAE":"{:.5f}","MSE":"{:.5f}","RMSE":"{:.5f}","R²":"{:.5f}","Train Time (ms)":"{:.2f}","P95 Inference (ms)":"{:.4f}","Model Size (MB)":"{:.3f}"}),use_container_width=True)
            reg_metric=st.selectbox("Regression metric",["MAE","MSE","RMSE","R²","P95 Inference (ms)"],key="regression_metric")
            fig=px.bar(reg_df,x="Model",y=reg_metric,text=reg_metric,color="Model",title=f"Regression — {reg_metric}")
            fig.update_traces(texttemplate="%{text:.4f}",textposition="outside")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="white")
            st.plotly_chart(fig,use_container_width=True)
        st.markdown(f"""
        <div class="info-box">
            <b>📐 Regression Winner: {winner_reg}</b>
            <br>
            Regression performance is evaluated using R², MAE, MSE and RMSE. Higher R² is better, while lower MAE/MSE/RMSE indicates smaller prediction errors.
        </div>
        """,unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚡ Why These Metrics Matter</div>',unsafe_allow_html=True)
        m1,m2,m3,m4=st.columns(4)
        with m1:
            st.metric("Classification","Macro F1","Balanced class performance")
        with m2:
            st.metric("Regression","R²","Explained variation")
        with m3:
            st.metric("Production","P95","Inference latency")
        with m4:
            st.metric("Deployment","Model Size","Memory footprint")
    else:
        st.warning("benchmark_results.json was not found. Run the training script first.")
with tab3:
    st.markdown('<div class="section-title">🧠 Feature Intelligence</div>',unsafe_allow_html=True)
    if hasattr(classifier,"feature_importances_"):
        importance_df=pd.DataFrame({"Feature":FEATURE_COLS,"Importance":classifier.feature_importances_}).sort_values("Importance",ascending=False)
        fig=px.bar(importance_df.head(12).sort_values("Importance"),x="Importance",y="Feature",orientation="h",text="Importance",color="Importance",title="Top Classification Features")
        fig.update_traces(texttemplate="%{text:.3f}",textposition="outside")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="white",margin=dict(l=20,r=80,t=70,b=20))
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(importance_df.style.format({"Importance":"{:.5f}"}),use_container_width=True)
    st.markdown("""
    <div class="card">
        <h3>🔬 How ResQ-QR Thinks</h3>
        <p style="color:#cbd5e1;font-size:1.05rem;">
            The system does not rely on one hard-coded rule such as "latency above X means QR".
            The classification model learns relationships between network, gateway, bank and transaction telemetry from labelled examples.
        </p>
        <p style="color:#cbd5e1;font-size:1.05rem;">
            The regression model provides continuous degradation probabilities for the network, gateway and bank.
            These signals can support the final recovery decision.
        </p>
    </div>
    """,unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <h3>⚡ ResQ-QR Decision Pipeline</h3>
        <p style="font-size:1.1rem;color:#cbd5e1;">
            📡 Telemetry
            →
            🧠 Classification
            →
            📈 Degradation Probability
            →
            🎯 Recovery Decision
            →
            📱 QR Fallback
        </p>
    </div>
    """,unsafe_allow_html=True)