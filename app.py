import io
import os
import joblib
import numpy as np
import pandas as pd
import qrcode
import streamlit as st

st.set_page_config(
    page_title="ResQ-QR ML Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
    .stCard {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .metric-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .badge-green { background-color: #059669; color: white; }
    .badge-orange { background-color: #d97706; color: white; }
    .badge-red { background-color: #dc2626; color: white; }
    </style>
""",
    unsafe_allow_html=True,
)

FEATURE_COLS = [
    "gateway_latency_ms",
    "bank_latency_ms",
    "total_latency_ms",
    "packet_loss_pct",
    "payment_stage",
    "error_code_category",
    "is_timeout_flag",
    "order_amount_inr",
    "retry_count",
]


@st.cache_resource
def load_best_model():
    model = joblib.load("model.pkl")
    meta_info = "Optimal ML Engine"
    if os.path.exists("model_meta.txt"):
        with open("model_meta.txt", "r") as f:
            meta_info = f.read().replace("\n", " | ")
    return model, meta_info


model, meta_info = load_best_model()

st.title("⚡ ResQ-QR: Payments Fallback Engine")
st.caption(f"🧠 Active Engine: `{meta_info}`")

tab1, tab2 = st.tabs(
    [
        "🚀 Interactive Sandbox Demo",
        "📊 Model Telemetry & Weights",
    ]
)

with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("1. Payment Initialization & Telemetry")

        order_amount = st.number_input(
            "Transaction Value (INR)", value=499.00, step=50.0
        )

        payment_stage_str = st.selectbox(
            "Payment Lifecycle Stage",
            [
                "0: INITIATED (Checkout Launched)",
                "1: AUTHENTICATING (User Entering PIN/OTP)",
                "2: AUTHORIZING (PSP <-> Bank Handshake)",
                "3: SETTLING (Finalizing Transaction)",
            ],
        )
        payment_stage = int(payment_stage_str.split(":")[0])

        error_type = st.selectbox(
            "Simulated Gateway Exception",
            [
                "GATEWAY_TIMEOUT (504 Socket Timeout)",
                "BAD_INSUFFICIENT_FUNDS (User Account Balance Low)",
                "ISSUER_BANK_OFFLINE (503 Host Unreachable)",
            ],
        )

        st.markdown("---")
        st.markdown("**Real-Time Latency Breakdown (ms)**")
        col_lat1, col_lat2 = st.columns(2)
        with col_lat1:
            gateway_latency = st.slider("Gateway Latency", 10, 3000, 1800, 50)
        with col_lat2:
            bank_latency = st.slider("Bank Host Latency", 10, 3000, 1400, 50)

        total_latency = gateway_latency + bank_latency
        st.caption(f"⏱ Total Round-Trip Latency: **{total_latency} ms**")

        packet_loss_pct = st.slider(
            "Packet Loss Rate (%)", 0.0, 30.0, 12.0, 0.5
        )
        retry_count = st.select_slider(
            "Automated Retry Attempts",
            options=[0, 1, 2, 3],
            value=0,
        )

        # Map Error Categories and Timeout Flags accurately
        if "FUNDS" in error_type:
            error_category = 1
            is_timeout = 0
        elif "TIMEOUT" in error_type:
            error_category = 2
            is_timeout = 1
        else:
            error_category = 3
            is_timeout = 0

        run_sim = st.button(
            "⚡ Process Telemetry & Predict Action",
            type="primary",
            use_container_width=True,
        )

    with col_right:
        st.subheader("2. AI Decision Engine Output")

        if run_sim:
            input_df = pd.DataFrame(
                [
                    [
                        gateway_latency,
                        bank_latency,
                        total_latency,
                        packet_loss_pct,
                        payment_stage,
                        error_category,
                        is_timeout,
                        order_amount,
                        retry_count,
                    ]
                ],
                columns=FEATURE_COLS,
            )

            pred_class = model.predict(input_df)[0]
            pred_probs = model.predict_proba(input_df)[0]
            confidence = np.max(pred_probs) * 100

            actions = {
                0: (
                    "NO_ACTION",
                    "badge-red",
                    "Bank host offline or unrecoverable error. "
                    "Halting retries to prevent duplicate debits.",
                ),
                1: (
                    "CONTEXTUAL_NUDGE",
                    "badge-orange",
                    "User-side error detected (Insufficient Funds / "
                    "VPA). Nudge user via SMS/Push notification.",
                ),
                2: (
                    "GENERATE_DYNAMIC_QR",
                    "badge-green",
                    "Degraded connectivity detected. Offloading "
                    "payment flow to client-rendered micro-QR.",
                ),
            }

            label, badge_class, explanation = actions[pred_class]

            st.markdown(
                f"""
                <div class="stCard">
                    <h4>Recommended System Action</h4>
                    <span class="metric-badge {badge_class}">{label}
                    </span>
                    <p style="margin-top: 10px; color: #cbd5e1;">
                    {explanation}</p>
                    <small>Model Confidence: <strong>{confidence:.1f}%
                    </strong></small>
                </div>
            """,
                unsafe_allow_html=True,
            )

            st.write("**Model Probability Distribution:**")
            probs_df = pd.DataFrame(
                {
                    "Action": [
                        "NO_ACTION",
                        "CONTEXTUAL_NUDGE",
                        "GENERATE_DYNAMIC_QR",
                    ],
                    "Probability": pred_probs,
                }
            )
            st.bar_chart(probs_df.set_index("Action"))

            if pred_class == 2:
                deeplink = (
                    f"upi://pay?pa=resqstore@upi&pn=ResQStore"
                    f"&am={order_amount}&tr=TXN98765&cu=INR"
                )
                qr = qrcode.QRCode(version=1, box_size=4, border=1)
                qr.add_data(deeplink)
                qr.make(fit=True)
                img = qr.make_image(fill_color="black", back_color="white")

                buf = io.BytesIO()
                img.save(buf, format="PNG")
                qr_bytes = buf.getvalue()
                size_kb = round(len(qr_bytes) / 1024, 2)

                st.markdown("---")
                st.markdown("### 📱 Client-Side Micro-QR Asset")
                st.image(
                    qr_bytes,
                    width=200,
                    caption=f"Rendered in memory | Payload size: {size_kb} KB",
                )
        else:
            st.info(
                "👈 Adjust telemetry parameters and click "
                "**Process Telemetry** to trigger decision engine."
            )

with tab2:
    st.subheader("Predictive Feature Weights")
    st.caption(
        "Shows feature importance ranking computed by the "
        "trained model."
    )

    if hasattr(model, "feature_importances_"):
        importances = pd.DataFrame(
            {
                "Telemetry Feature": FEATURE_COLS,
                "Relative Weight": model.feature_importances_,
            }
        ).sort_values(by="Relative Weight", ascending=False)

        st.bar_chart(importances.set_index("Telemetry Feature"))
        st.dataframe(importances, use_container_width=True)
