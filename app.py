import io
import json
import os
import uuid
import secrets
from urllib.parse import quote
from html import escape

import joblib
import numpy as np
import pandas as pd
import qrcode
import streamlit as st
import plotly.express as px


# ===================================================================
# PAGE CONFIGURATION
# ===================================================================

st.set_page_config(
    page_title="ResQ-QR | Intelligent Payment Recovery",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===================================================================
# RESQ-QR PAYMENT CONFIGURATION
# ===================================================================

UPI_ID = "resqstore@upi"
MERCHANT_NAME = "ResQStore"

RESQ_PUBLIC_URL = os.getenv(
    "RESQ_PUBLIC_URL",
    "http://localhost:8501",
).rstrip("/")

TOKEN_STORE_FILE = "dynamic_payment_tokens.json"


# ===================================================================
# MODEL FEATURES
# ===================================================================

REGRESSION_FEATURES = [
    "network_latency_ms",
    "packet_loss_pct",
    "network_jitter_ms",
    "gateway_latency_ms",
    "gateway_failure_rate_pct",
    "gateway_timeout_rate_pct",
    "bank_latency_ms",
    "bank_failure_rate_pct",
    "bank_timeout_rate_pct",
    "error_code_category",
    "is_timeout_flag",
]

CLASSIFICATION_FEATURES = [
    "predicted_network_degradation",
    "predicted_gateway_degradation",
    "predicted_bank_degradation",
]

REGRESSION_TARGETS = [
    "true_network_degradation",
    "true_gateway_degradation",
    "true_bank_degradation",
]


# ===================================================================
# ACTION DEFINITIONS
# ===================================================================

ACTION_DESCRIPTIONS = {
    1: {
        "name": "CONTEXTUAL NUDGE",
        "emoji": "🟠",
        "color": "#f59e0b",
        "meaning": (
            "The payment issue is associated with bank-side "
            "degradation. The customer should be informed and "
            "guided rather than repeatedly retrying the payment."
        ),
        "error": (
            "The bank-side payment infrastructure is showing "
            "elevated degradation according to the regression stage."
        ),
        "solution": (
            "Provide a contextual message to the customer and "
            "avoid unnecessary repeated payment attempts."
        ),
    },
    2: {
        "name": "NO ACTION",
        "emoji": "🛑",
        "color": "#ef4444",
        "meaning": (
            "The payment infrastructure indicates gateway-side "
            "degradation where an automatic recovery action "
            "is not recommended."
        ),
        "error": (
            "The payment gateway is showing elevated degradation "
            "according to the regression stage."
        ),
        "solution": (
            "Do not automatically generate a fallback or repeatedly "
            "retry the transaction. Allow the gateway path to recover."
        ),
    },
    3: {
        "name": "GENERATE DYNAMIC QR",
        "emoji": "🟢",
        "color": "#10b981",
        "meaning": (
            "The network path is degraded. A lightweight QR payment "
            "fallback can allow the customer to continue the payment."
        ),
        "error": (
            "Network telemetry such as latency, packet loss and "
            "jitter indicates network-side degradation."
        ),
        "solution": (
            "Move the customer to the QR fallback instead of "
            "continuously waiting on the degraded network path."
        ),
    },
}

ACTION_NAMES = [
    "CONTEXTUAL NUDGE",
    "NO ACTION",
    "GENERATE DYNAMIC QR",
]


# ===================================================================
# FILE LOADING
# ===================================================================

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
        with open(
            "benchmark_results.json",
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    return {}


classifier, regression_model = load_models()
benchmark = load_benchmark()


# ===================================================================
# SESSION STATE
# ===================================================================

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "pred_class" not in st.session_state:
    st.session_state.pred_class = None

if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0

if "probability_map" not in st.session_state:
    st.session_state.probability_map = {}

if "regression_predictions" not in st.session_state:
    st.session_state.regression_predictions = None

if "action" not in st.session_state:
    st.session_state.action = None

if "payment_data" not in st.session_state:
    st.session_state.payment_data = None


# ===================================================================
# DYNAMIC PAYMENT TOKEN STORAGE
# ===================================================================

def load_token_store():
    if not os.path.exists(TOKEN_STORE_FILE):
        return {}

    try:
        with open(
            TOKEN_STORE_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

            if isinstance(data, dict):
                return data

    except Exception:
        pass

    return {}


def save_token_store(store):
    temp_file = TOKEN_STORE_FILE + ".tmp"

    with open(
        temp_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            store,
            file,
            indent=2,
        )

    os.replace(
        temp_file,
        TOKEN_STORE_FILE,
    )


# ===================================================================
# DYNAMIC PAYMENT GENERATION
# ===================================================================

def generate_dynamic_payment(amount=499.00):

    transaction_id = (
        "RSQ-"
        + uuid.uuid4().hex[:12].upper()
    )

    token = (
        "RSQ-"
        + secrets.token_urlsafe(9)
    )

    payment_data = {
        "token": token,
        "transaction_id": transaction_id,
        "amount": float(amount),
        "upi_id": UPI_ID,
        "merchant_name": MERCHANT_NAME,
        "currency": "INR",
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }

    token_store = load_token_store()

    token_store[token] = payment_data

    save_token_store(token_store)

    return payment_data


# ===================================================================
# UPI DEEP LINK
# ===================================================================

def build_upi_payment_url(payment):

    return (
        "upi://pay"
        f"?pa={quote(str(payment['upi_id']))}"
        f"&pn={quote(str(payment['merchant_name']))}"
        f"&am={float(payment['amount']):.2f}"
        f"&tr={quote(str(payment['transaction_id']))}"
        f"&cu={quote(str(payment['currency']))}"
    )


# ===================================================================
# RESOLVER URL
# ===================================================================

def build_resolver_url(token):

    return (
        f"{RESQ_PUBLIC_URL}/"
        f"?payment_token={quote(token)}"
    )


# ===================================================================
# RESOLVE PAYMENT TOKEN
# ===================================================================

def resolve_payment_token(token):

    if not token:
        return None

    token_store = load_token_store()

    return token_store.get(token)


# ===================================================================
# PAYMENT RESOLVER
# ===================================================================

payment_token = st.query_params.get(
    "payment_token"
)

if isinstance(payment_token, list):
    payment_token = payment_token[0]


if payment_token:

    resolved_payment = resolve_payment_token(
        payment_token
    )

    if resolved_payment is None:

        st.error(
            "❌ Payment session not found."
        )

        st.stop()

    resolved_upi_url = build_upi_payment_url(
        resolved_payment
    )

    safe_transaction = escape(
        resolved_payment["transaction_id"]
    )

    safe_merchant = escape(
        resolved_payment["merchant_name"]
    )

    safe_upi = escape(
        resolved_payment["upi_id"]
    )

    safe_token = escape(
        resolved_payment["token"]
    )

    amount_display = (
        f"₹{resolved_payment['amount']:.2f}"
    )

    st.markdown(
        f"""
        <div style="
            max-width:650px;
            margin:60px auto;
            padding:35px;
            border-radius:24px;
            background:
                linear-gradient(
                    145deg,
                    rgba(25,55,95,0.96),
                    rgba(18,45,82,0.94)
                );
            border:1px solid rgba(96,165,250,0.40);
            box-shadow:0 20px 60px rgba(0,0,0,0.35);
            text-align:center;
        ">

            <div style="
                font-size:3rem;
                margin-bottom:10px;
            ">
                ⚡
            </div>

            <h1 style="
                color:#f8fafc;
                margin-bottom:8px;
            ">
                ResQ-QR Payment
            </h1>

            <p style="
                color:#bfdbfe;
                font-size:1.05rem;
            ">
                Dynamic payment session resolved successfully.
            </p>

            <div style="
                margin-top:25px;
                padding:20px;
                border-radius:16px;
                background:rgba(15,23,42,0.65);
                text-align:left;
            ">

                <p style="color:#f8fafc;">
                    <b>Merchant:</b> {safe_merchant}
                </p>

                <p style="color:#f8fafc;">
                    <b>Amount:</b> {amount_display}
                </p>

                <p style="color:#f8fafc;">
                    <b>UPI ID:</b> {safe_upi}
                </p>

                <p style="color:#f8fafc;">
                    <b>Transaction:</b> {safe_transaction}
                </p>

                <p style="color:#f8fafc;">
                    <b>Token:</b> {safe_token}
                </p>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="
            max-width:650px;
            margin:20px auto;
            text-align:center;
        ">

            <a
                href="{escape(resolved_upi_url)}"
                style="
                    display:inline-block;
                    padding:15px 30px;
                    border-radius:14px;
                    background:#10b981;
                    color:white;
                    text-decoration:none;
                    font-weight:800;
                    font-size:1.05rem;
                "
            >
                💳 OPEN UPI PAYMENT
            </a>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "If your device supports UPI deep links, "
        "the button above can open an installed UPI application."
    )

    st.stop()


# ===================================================================
# STYLING
# ===================================================================

st.markdown(
    """
    <style>

    /* ============================================================
       GLOBAL
       ============================================================ */

    html,
    body,
    .stApp {
        color:#f8fafc !important;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(59,130,246,0.18),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 20%,
                rgba(16,185,129,0.12),
                transparent 25%
            ),
            #08111f;
    }

    .block-container {
        max-width:1400px;
        padding-top:2rem;
        padding-bottom:3rem;
    }

    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        color:#f8fafc !important;
        letter-spacing:-0.02em;
    }

    p,
    label,
    span,
    div {
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }


    /* ============================================================
       HERO
       ============================================================ */

    .hero {
    position: relative;
    width: 100%;
    box-sizing: border-box;

    padding: 30px 34px;
    margin-top: 30px;
    margin-bottom: 24px;

    border-radius: 24px;

    background:
        linear-gradient(
            135deg,
            #172554 0%,
            #1d4ed8 52%,
            #0e7490 100%
        );

    border:
        1px solid rgba(147, 197, 253, 0.25);

    box-shadow:
        0 20px 55px rgba(0, 0, 0, 0.32);

    overflow: hidden;
}
    .hero-title {
        font-size:2.9rem;
        font-weight:900;
        color:white !important;
        margin:0;
    }

    .hero-subtitle {
        color:#dbeafe !important;
        font-size:1.08rem;
        margin:7px 0 15px;
    }

    .status-pill {
        display:inline-block;
        padding:7px 14px;
        border-radius:999px;

        background:
            rgba(16,185,129,0.18);

        border:
            1px solid rgba(110,231,183,0.4);

        color:#a7f3d0 !important;
        font-weight:800;
        font-size:0.88rem;
    }


    /* ============================================================
       PANELS
       ============================================================ */

    .panel {
        background:
            rgba(18,38,68,0.88);

        border:
            1px solid rgba(96,165,250,0.25);

        border-radius:20px;
        padding:22px;
        margin-bottom:16px;

        box-shadow:
            0 12px 35px rgba(0,0,0,0.22);
    }

    .panel-title {
        font-size:1.12rem;
        font-weight:800;
        color:#f8fafc !important;
        margin-bottom:14px;
    }

    .panel-subtitle {
        color:#bfdbfe !important;
        font-size:0.92rem;
        line-height:1.55;
    }

    .muted {
        color:#cbd5e1 !important;
        line-height:1.6;
    }


    /* ============================================================
       STATUS BAR
       ============================================================ */

    .status-bar {
        padding:14px 18px;
        border-radius:15px;

        background:
            rgba(16,185,129,0.10);

        border:
            1px solid rgba(52,211,153,0.28);

        color:#d1fae5 !important;
        margin-bottom:18px;
    }


    /* ============================================================
       DECISION CARD
       ============================================================ */

    .decision {
        padding:24px;
        border-radius:20px;
        color:white !important;
        margin-bottom:18px;

        box-shadow:
            0 18px 45px rgba(0,0,0,0.3);
    }

    .decision-label {
        font-size:0.78rem;
        font-weight:800;
        opacity:0.82;
        letter-spacing:0.08em;
    }

    .decision-name {
        font-size:2rem;
        font-weight:900;
        margin:7px 0;
    }

    .decision-confidence {
        font-size:1rem;
        opacity:0.92;
    }


    /* ============================================================
       EXPLANATION
       ============================================================ */

    .explain {
        padding:20px;
        border-radius:17px;

        background:
            rgba(30,41,59,0.72);

        border:
            1px solid rgba(96,165,250,0.24);

        margin-top:14px;
    }

    .explain h4 {
        margin:0 0 8px;
    }

    .error-panel {
        padding:20px;
        border-radius:17px;

        background:
            rgba(127,29,29,0.18);

        border:
            1px solid rgba(248,113,113,0.28);

        margin-top:14px;
    }

    .solution-panel {
        padding:20px;
        border-radius:17px;

        background:
            rgba(6,78,59,0.20);

        border:
            1px solid rgba(52,211,153,0.28);

        margin-top:14px;
    }


    /* ============================================================
       STREAMLIT WIDGET LABELS
       ============================================================ */

    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] span {
        color:#f8fafc !important;
        font-weight:700 !important;
    }

    div[data-testid="stSlider"] label,
    div[data-testid="stSlider"] label *,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stNumberInput"] label *,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] label *,
    div[data-testid="stSelectSlider"] label,
    div[data-testid="stSelectSlider"] label * {
        color:#f8fafc !important;
        font-weight:700 !important;
    }


    /* ============================================================
       INPUTS
       ============================================================ */

    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        background-color:#0f172a !important;
        color:#f8fafc !important;
        caret-color:#f8fafc !important;

        border:
            1px solid rgba(148,163,184,0.35) !important;
    }

    div[data-testid="stNumberInput"] input::placeholder,
    div[data-testid="stTextInput"] input::placeholder {
        color:#94a3b8 !important;
        opacity:1 !important;
    }


    /* ============================================================
       SELECTBOX / DROPDOWN
       ============================================================ */

    div[data-testid="stSelectbox"] {
        width:100%;
    }

    div[data-testid="stSelectbox"]
    div[data-baseweb="select"] {
        background-color:#0f172a !important;

        border:
            1px solid rgba(148,163,184,0.40) !important;

        border-radius:10px !important;
    }

    div[data-testid="stSelectbox"]
    div[data-baseweb="select"] > div {
        background-color:#0f172a !important;
    }

    div[data-testid="stSelectbox"]
    div[data-baseweb="select"]
    span {
        color:#f8fafc !important;
    }

    /* Dropdown popup */

    div[data-baseweb="popover"] {
        background-color:#0f172a !important;
    }

    div[data-baseweb="menu"] {
        background-color:#0f172a !important;

        border:
            1px solid rgba(96,165,250,0.35) !important;
    }

    div[data-baseweb="menu"] li {
        background-color:#0f172a !important;
        color:#f8fafc !important;
    }

    div[data-baseweb="menu"] li span {
        color:#f8fafc !important;
    }

    div[data-baseweb="menu"] li:hover {
        background-color:#1e3a5f !important;
        color:#ffffff !important;
    }

    div[data-baseweb="menu"]
    li[aria-selected="true"] {
        background-color:#1d4ed8 !important;
        color:#ffffff !important;
    }

    div[data-baseweb="menu"]
    li[aria-selected="true"] span {
        color:#ffffff !important;
    }


    /* ============================================================
       SLIDERS
       ============================================================ */

    div[data-testid="stSlider"] p,
    div[data-testid="stSelectSlider"] p {
        color:#cbd5e1 !important;
    }


    /* ============================================================
       BUTTONS
       ============================================================ */

    .stButton > button {
        min-height:48px;
        border-radius:13px;
        font-weight:800;

        border:
            1px solid rgba(96,165,250,0.35);

        color:#f8fafc !important;
    }

    .stButton > button * {
        color:#f8fafc !important;
    }


    /* ============================================================
       METRICS
       ============================================================ */

    div[data-testid="stMetric"] {
        background:
            rgba(18,38,68,0.90);

        border:
            1px solid rgba(96,165,250,0.25);

        border-radius:16px;
        padding:14px;
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {
        color:#bfdbfe !important;
    }

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color:#f8fafc !important;
    }

    /* ==========================================================
   RESQ-QR DECISION PIPELINE
   ========================================================== */

.pipeline-wrapper {
    width: 100%;
    display: flex;
    align-items: stretch;
    justify-content: center;
    gap: 8px;
    margin-top: 18px;
    margin-bottom: 14px;
    overflow-x: auto;
    padding: 8px 2px 18px;
}

.pipeline-step {
    min-width: 150px;
    flex: 1;

    padding: 18px 14px;

    border-radius: 18px;

    background:
        linear-gradient(
            145deg,
            rgba(25, 55, 95, 0.96),
            rgba(18, 45, 82, 0.92)
        );

    border:
        1px solid rgba(96, 165, 250, 0.28);

    box-shadow:
        0 10px 28px rgba(0, 0, 0, 0.22);

    text-align: center;

    transition:
        transform 0.2s ease,
        border-color 0.2s ease,
        box-shadow 0.2s ease;
}

.pipeline-step:hover {
    transform: translateY(-4px);

    border-color:
        rgba(147, 197, 253, 0.65);

    box-shadow:
        0 15px 35px rgba(0, 0, 0, 0.30);
}

.pipeline-icon {
    width: 54px;
    height: 54px;

    margin:
        0 auto 10px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 15px;

    background:
        rgba(59, 130, 246, 0.14);

    border:
        1px solid rgba(96, 165, 250, 0.30);

    font-size: 1.7rem;
}

.pipeline-number {
    color:
        #93c5fd;

    font-size:
        0.68rem;

    font-weight:
        900;

    letter-spacing:
        0.10em;

    margin-bottom:
        6px;
}

.pipeline-title {
    color:
        #f8fafc;

    font-size:
        1rem;

    font-weight:
        900;

    margin-bottom:
        7px;
}

.pipeline-description {
    color:
        #cbd5e1;

    font-size:
        0.76rem;

    line-height:
        1.45;
}

.pipeline-arrow {
    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    min-width:
        25px;

    color:
        #60a5fa;

    font-size:
        1.45rem;

    font-weight:
        900;
}


/* FINAL AI ACTION */

.pipeline-final {
    background:
        linear-gradient(
            145deg,
            rgba(6, 78, 59, 0.92),
            rgba(5, 46, 38, 0.96)
        );

    border:
        1px solid rgba(52, 211, 153, 0.45);

    box-shadow:
        0 12px 35px rgba(16, 185, 129, 0.14);
}

.pipeline-final .pipeline-icon {
    background:
        rgba(16, 185, 129, 0.15);

    border:
        1px solid rgba(52, 211, 153, 0.40);
}

.pipeline-final .pipeline-number {
    color:
        #6ee7b7;
}

.pipeline-final:hover {
    border-color:
        rgba(110, 231, 183, 0.70);

    box-shadow:
        0 18px 40px rgba(16, 185, 129, 0.20);
}


/* FLOW LABEL */

.pipeline-flow-label {
    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    gap:
        12px;

    margin:
        2px 0 28px;

    color:
        #94a3b8;

    font-size:
        0.68rem;

    font-weight:
        800;

    letter-spacing:
        0.08em;

    text-align:
        center;
}

.flow-line {
    width:
        45px;

    height:
        1px;

    background:
        rgba(96, 165, 250, 0.35);
}


/* RESPONSIVE */

@media (max-width: 1100px) {

    .pipeline-wrapper {
        justify-content:
            flex-start;
    }

    .pipeline-step {
        min-width:
            145px;
        flex:
            0 0 145px;
    }

    .pipeline-arrow {
        flex:
            0 0 22px;
    }

}
    /* ============================================================
       TABS
       ============================================================ */

    .stTabs [data-baseweb="tab-list"] {
        gap:8px;
        margin-bottom:20px;
    }

    .stTabs [data-baseweb="tab"] {
        padding:10px 18px;
        border-radius:11px;
        font-weight:800;
        color:#f8fafc !important;
    }

    .stTabs [data-baseweb="tab"] * {
        color:#f8fafc !important;
    }


    /* ============================================================
       DATAFRAMES
       ============================================================ */

    div[data-testid="stDataFrame"] {
        border-radius:14px;
        overflow:hidden;

        border:
            1px solid rgba(96,165,250,0.25);
    }


    /* ============================================================
       ALERTS
       ============================================================ */

    div[data-testid="stAlert"] {
        border-radius:14px;
    }

    div[data-testid="stAlert"] * {
        color:#f8fafc !important;
    }


    /* ============================================================
       CONTAINER CARDS
       ============================================================ */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background:
            linear-gradient(
                145deg,
                rgba(25,55,95,0.94),
                rgba(18,45,82,0.90)
            ) !important;

        border:
            1px solid rgba(96,165,250,0.38) !important;

        border-radius:20px !important;

        box-shadow:
            0 14px 35px rgba(0,0,0,0.24) !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background:transparent !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ===================================================================
# MODEL FILE CHECK
# ===================================================================

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


# ===================================================================
# HEADER
# ===================================================================

hero_html = """
<div class="hero">
    <div class="hero-title">⚡ ResQ-QR</div>
    <div class="hero-subtitle">
        Intelligent Payment Recovery & Dynamic Fallback Engine
    </div>
    <span class="status-pill">
        ● ML ENGINE ONLINE
    </span>
</div>
"""

st.markdown(
    hero_html,
    unsafe_allow_html=True,
)


winner_class = benchmark.get(
    "classification_winner",
    "Unknown",
)

winner_reg = benchmark.get(
    "regression_winner",
    "Unknown",
)


status_col1, status_col2 = st.columns(2)

with status_col1:
    st.success(
        f"🏆 Classification Winner: {winner_class}"
    )

with status_col2:
    st.info(
        f"📈 Regression Winner: {winner_reg}"
    )


# ===================================================================
# TABS
# ===================================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🚀 LIVE PAYMENT ENGINE",
        "📊 MODEL PERFORMANCE",
        "🧠 FEATURE INTELLIGENCE",
    ]
)


# ===================================================================
# TAB 1
# ===================================================================

with tab1:

    st.header(
        "🚀 Live Payment Decision Simulator"
    )

    st.caption(
        "Configure the raw payment telemetry and let the ML engine "
        "estimate infrastructure degradation before selecting "
        "the safest recovery action."
    )

    left, right = st.columns(
        [1, 1],
        gap="large",
    )


    # ===============================================================
    # LEFT SIDE
    # ===============================================================

    with left:

        # -----------------------------------------------------------
        # NETWORK
        # -----------------------------------------------------------

        st.markdown(
            '<div class="panel-title">'
            '🌐 Network Telemetry'
            '</div>',
            unsafe_allow_html=True,
        )

        network_latency = st.slider(
            "Network Latency (ms)",
            min_value=5,
            max_value=1750,
            value=500,
            step=1,
            help="Dataset range: 5–1750 ms",
        )

        packet_loss = st.slider(
            "Packet Loss (%)",
            min_value=0.01,
            max_value=9.00,
            value=3.00,
            step=0.01,
            help="Dataset range: 0.01–9.00%",
        )

        network_jitter = st.slider(
            "Network Jitter (ms)",
            min_value=1.0,
            max_value=100.0,
            value=30.0,
            step=0.1,
            help="Dataset range: 1–100 ms",
        )


        st.divider()


        # -----------------------------------------------------------
        # GATEWAY
        # -----------------------------------------------------------

        st.markdown(
            '<div class="panel-title">'
            '🔌 Gateway Telemetry'
            '</div>',
            unsafe_allow_html=True,
        )

        gateway_latency = st.slider(
            "Gateway Latency (ms)",
            min_value=40,
            max_value=1300,
            value=600,
            step=1,
            help="Dataset range: 40–1300 ms",
        )

        gateway_failure = st.slider(
            "Gateway Failure Rate (%)",
            min_value=0.05,
            max_value=0.50,
            value=0.20,
            step=0.01,
            help="Dataset range: 0.05–0.50%",
        )

        gateway_timeout = st.slider(
            "Gateway Timeout Rate (%)",
            min_value=0.10,
            max_value=0.50,
            value=0.20,
            step=0.01,
            help="Dataset range: 0.10–0.50%",
        )


        st.divider()


        # -----------------------------------------------------------
        # BANK
        # -----------------------------------------------------------

        st.markdown(
            '<div class="panel-title">'
            '🏦 Bank Telemetry'
            '</div>',
            unsafe_allow_html=True,
        )

        bank_latency = st.slider(
            "Bank Latency (ms)",
            min_value=200,
            max_value=1300,
            value=500,
            step=1,
            help="Dataset range: 200–1300 ms",
        )

        bank_failure = st.slider(
            "Bank Failure Rate (%)",
            min_value=0.10,
            max_value=1.50,
            value=0.50,
            step=0.01,
            help="Dataset range: 0.10–1.50%",
        )

        bank_timeout = st.slider(
            "Bank Timeout Rate (%)",
            min_value=1.00,
            max_value=3.00,
            value=2.00,
            step=0.01,
            help="Dataset range: 1.00–3.00%",
        )


        st.divider()


        # -----------------------------------------------------------
        # PAYMENT ERROR
        # -----------------------------------------------------------

        st.markdown(
            '<div class="panel-title">'
            '⚠️ Payment Error Telemetry'
            '</div>',
            unsafe_allow_html=True,
        )

        error_type = st.selectbox(
            "Error Code Category",
            [
                "0 — NONE",
                "1 — BAD_FUNDS",
                "2 — TIMEOUT",
                "3 — BANK_OFFLINE",
            ],
            key="payment_error_type",
        )


        if error_type.startswith("0"):

            error_category = 0
            timeout_flag = 0

        elif error_type.startswith("1"):

            error_category = 1
            timeout_flag = 0

        elif error_type.startswith("2"):

            error_category = 2
            timeout_flag = 1

        else:

            error_category = 3
            timeout_flag = 0


        timeout_display = (
            "1 — Timeout detected"
            if timeout_flag
            else "0 — No timeout"
        )


        st.metric(
            "Timeout Flag",
            timeout_display,
        )


        st.divider()


        # -----------------------------------------------------------
        # PAYMENT CONTEXT
        # -----------------------------------------------------------

        st.markdown(
            '<div class="panel-title">'
            'ℹ️ Payment Context'
            '</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Transaction amount, payment stage, transaction age "
            "and retry count are intentionally excluded from the "
            "ML pipeline because they were not training features."
        )


        analyze_button = st.button(
            "⚡ ANALYZE PAYMENT",
            type="primary",
            use_container_width=True,
        )


    # ===============================================================
    # RIGHT SIDE
    # ===============================================================

    with right:

        st.header(
            "🧠 AI Decision Engine"
        )


        # ===========================================================
        # BEFORE ANALYSIS
        # ===========================================================

        if not analyze_button:

            with st.container(border=True):

                st.subheader(
                    "👈 Configure Raw Telemetry"
                )

                st.write(
                    "Adjust the network, gateway, bank and error "
                    "telemetry on the left."
                )

                st.write(
                    "ResQ-QR first predicts infrastructure "
                    "degradation using regression models."
                )

                st.write(
                    "Those three degradation predictions are then "
                    "passed to the classification model to select "
                    "the recovery action."
                )


        # ===========================================================
        # ANALYSIS
        # ===========================================================

        if analyze_button:

            st.session_state.analysis_done = True


            # =======================================================
            # STEP 1 — RAW TELEMETRY
            # =======================================================

            input_values = [[

                network_latency,
                packet_loss,
                network_jitter,

                gateway_latency,
                gateway_failure,
                gateway_timeout,

                bank_latency,
                bank_failure,
                bank_timeout,

                error_category,
                timeout_flag,

            ]]


            input_df = pd.DataFrame(
                input_values,
                columns=REGRESSION_FEATURES,
            )


            # =======================================================
            # STEP 2 — REGRESSION
            # =======================================================

            st.subheader(
                "📈 Regression — Infrastructure Degradation"
            )


            try:

                if (
                    isinstance(
                        regression_model,
                        dict,
                    )
                    and "models" in regression_model
                ):

                    regression_predictions = []

                    for target in REGRESSION_TARGETS:

                        if target not in regression_model["models"]:

                            raise KeyError(
                                f"Regression model for '{target}' "
                                "was not found."
                            )


                        reg_model = (
                            regression_model["models"][target]
                        )


                        prediction = float(
                            reg_model.predict(
                                input_df
                            )[0]
                        )


                        regression_predictions.append(
                            prediction
                        )

                else:

                    regression_predictions = np.asarray(
                        regression_model.predict(
                            input_df
                        )[0],
                        dtype=float,
                    ).reshape(-1)


                regression_predictions = np.clip(
                    regression_predictions,
                    0,
                    100,
                )


            except Exception as error:

                st.error(
                    "Regression prediction failed. "
                    "Please ensure regression_model.pkl was trained "
                    f"with these 11 features: "
                    f"{REGRESSION_FEATURES}. "
                    f"Error: {error}"
                )

                st.stop()


            # =======================================================
            # REGRESSION DISPLAY
            # =======================================================

            degradation_df = pd.DataFrame(
                {
                    "Component": [
                        "🌐 Network",
                        "🔌 Gateway",
                        "🏦 Bank",
                    ],
                    "Degradation Score":
                        regression_predictions,
                }
            )


            fig2 = px.bar(
                degradation_df,
                x="Component",
                y="Degradation Score",
                text="Degradation Score",
                title="Predicted Infrastructure Degradation",
            )


            fig2.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
            )


            fig2.update_layout(
                height=380,
                yaxis_title="Degradation Score (%)",
                xaxis_title="",
                yaxis_range=[0, 100],
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                margin=dict(
                    t=60,
                    b=30,
                    l=20,
                    r=20,
                ),
            )


            st.plotly_chart(
                fig2,
                use_container_width=True,
            )


            # =======================================================
            # STEP 3 — REGRESSION → CLASSIFIER
            # =======================================================

            classification_input = pd.DataFrame(
                [regression_predictions],
                columns=CLASSIFICATION_FEATURES,
            )


            # =======================================================
            # STEP 4 — CLASSIFICATION
            # =======================================================

            try:

                pred_class = int(
                    classifier.predict(
                        classification_input
                    )[0]
                )


                if hasattr(
                    classifier,
                    "predict_proba",
                ):

                    pred_probs_raw = (
                        classifier.predict_proba(
                            classification_input
                        )[0]
                    )


                    model_classes = getattr(
                        classifier,
                        "classes_",
                        np.array([1, 2, 3]),
                    )


                    probability_map = {

                        int(cls): float(prob)

                        for cls, prob in zip(
                            model_classes,
                            pred_probs_raw,
                        )

                    }


                else:

                    probability_map = {
                        1: 0.0,
                        2: 0.0,
                        3: 0.0,
                    }


                    probability_map[
                        pred_class
                    ] = 1.0


                confidence = (
                    probability_map.get(
                        pred_class,
                        0.0,
                    )
                    * 100
                )


            except Exception as error:

                st.error(
                    "Classification prediction failed. "
                    "The classifier must accept these three features: "
                    f"{CLASSIFICATION_FEATURES}. "
                    f"Error: {error}"
                )

                st.stop()


            # =======================================================
            # SAVE MODEL OUTPUT TO SESSION STATE
            # =======================================================

            st.session_state.pred_class = pred_class

            st.session_state.confidence = confidence

            st.session_state.probability_map = (
                probability_map
            )

            st.session_state.regression_predictions = (
                regression_predictions
            )


            # =======================================================
            # STEP 5 — ACTION
            # =======================================================

            if pred_class not in ACTION_DESCRIPTIONS:

                st.error(
                    f"Unexpected classifier output: {pred_class}. "
                    "Expected class 1, 2 or 3."
                )

                st.stop()


            action = ACTION_DESCRIPTIONS[
                pred_class
            ]


            # =======================================================
            # HARD PAYMENT ERROR OVERRIDE
            # =======================================================

            funds_error = (
                error_category == 1
            )

            network_degraded = (
                regression_predictions[0] >= 50.0
            )


            if funds_error:

                if network_degraded:

                    action = {

                        "name":
                            "INSUFFICIENT FUNDS + NETWORK DEGRADED",

                        "emoji": "⚠️",

                        "color": "#f59e0b",

                        "meaning": (
                            "The payment cannot currently be completed "
                            "because the available funds are insufficient. "
                            "At the same time, the network path is showing "
                            "elevated degradation."
                        ),

                        "error": (
                            "Insufficient funds were reported for this "
                            "payment. Network degradation is also present, "
                            "so repeated payment attempts should be avoided."
                        ),

                        "solution": (
                            "Please ensure sufficient funds are available "
                            "first. Because the network is also degraded, "
                            "try the payment again later rather than "
                            "repeatedly retrying now."
                        ),
                    }


                else:

                    action = {

                        "name":
                            "INSUFFICIENT FUNDS",

                        "emoji": "💳",

                        "color": "#f59e0b",

                        "meaning": (
                            "The payment was rejected because sufficient "
                            "funds are not currently available."
                        ),

                        "error": (
                            "The payment system reported an "
                            "insufficient-funds condition."
                        ),

                        "solution": (
                            "Please ensure sufficient funds are available "
                            "and then retry the payment."
                        ),
                    }


            # =======================================================
            # SAVE FINAL ACTION
            # =======================================================

            st.session_state.action = action


            # =======================================================
            # AI RECOMMENDED ACTION
            # =======================================================

            with st.container(border=True):

                st.subheader(
                    f"{action['emoji']} "
                    "AI Recommended System Action"
                )

                st.markdown(
                    f"## {action['name']}"
                )

                st.write(
                    f"**Model confidence:** "
                    f"{confidence:.1f}%"
                )

                st.write(
                    f"**Classification output:** "
                    f"Class {pred_class}"
                )


            # =======================================================
            # REGRESSION → CLASSIFICATION TABLE
            # =======================================================

            st.subheader(
                "🔗 Regression → Classification Input"
            )


            classifier_display_df = pd.DataFrame(
                {
                    "Classifier Feature":
                        CLASSIFICATION_FEATURES,

                    "Predicted Value (%)":
                        regression_predictions,
                }
            )


            st.dataframe(
                classifier_display_df.style.format(
                    {
                        "Predicted Value (%)":
                            "{:.2f}"
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )


            # =======================================================
            # CLASS PROBABILITIES
            # =======================================================

            st.subheader(
                "🎯 Recovery Action Probabilities"
            )


            probability_df = pd.DataFrame(
                {
                    "Class": [
                        1,
                        2,
                        3,
                    ],

                    "Action": [
                        "CONTEXTUAL NUDGE",
                        "NO ACTION",
                        "GENERATE DYNAMIC QR",
                    ],

                    "Probability": [

                        probability_map.get(
                            1,
                            0.0,
                        ) * 100,

                        probability_map.get(
                            2,
                            0.0,
                        ) * 100,

                        probability_map.get(
                            3,
                            0.0,
                        ) * 100,
                    ],
                }
            )


            fig = px.bar(
                probability_df,
                x="Action",
                y="Probability",
                text="Probability",
                title="ML Confidence by Recovery Action",
            )


            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
            )


            fig.update_layout(
                height=400,
                yaxis_title="Probability (%)",
                xaxis_title="",
                yaxis_range=[0, 100],
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                margin=dict(
                    t=60,
                    b=30,
                    l=20,
                    r=20,
                ),
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
            )


            # =======================================================
            # AI EXPLANATION
            # =======================================================

            st.subheader(
                "🔎 AI Explanation"
            )


            with st.container(border=True):

                st.subheader(
                    "🧠 Why did the AI choose this?"
                )

                st.write(
                    f"**Classification:** "
                    f"Class {pred_class}"
                )

                st.write(
                    f"**Decision:** "
                    f"{action['name']}"
                )

                st.write(
                    f"**What it means:** "
                    f"{action['meaning']}"
                )


            with st.container(border=True):

                st.subheader(
                    "🚨 What does this error mean?"
                )

                st.write(
                    action["error"]
                )


            with st.container(border=True):

                st.subheader(
                    "✅ What should the system do?"
                )

                st.write(
                    action["solution"]
                )


            # =======================================================
            # DYNAMIC QR FALLBACK
            # =======================================================

            if pred_class == 3:

                st.divider()

                with st.container(border=True):

                    st.subheader(
                        "📱 Dynamic QR Fallback"
                    )

                    st.caption(
                        "Network degradation detected. "
                        "ResQ-QR has generated a lightweight "
                        "dynamic payment QR for fallback recovery."
                    )


                    # ------------------------------------------------
                    # PAYMENT AMOUNT
                    # ------------------------------------------------

                    qr_amount = 499.00


                    # ------------------------------------------------
                    # CREATE PAYMENT SESSION ONLY ONCE
                    # ------------------------------------------------

                    if (
                        st.session_state.payment_data
                        is None
                    ):

                        st.session_state.payment_data = (
                            generate_dynamic_payment(
                                amount=qr_amount
                            )
                        )


                    payment_data = (
                        st.session_state.payment_data
                    )


                    payment_token = (
                        payment_data["token"]
                    )


                    # ------------------------------------------------
                    # BUILD RESOLVER URL
                    # ------------------------------------------------

                    resolver_url = (
                        build_resolver_url(
                            payment_token
                        )
                    )


                    # ------------------------------------------------
                    # CREATE QR
                    # ------------------------------------------------

                    qr = qrcode.QRCode(
                        version=1,
                        error_correction=(
                            qrcode.constants.ERROR_CORRECT_L
                        ),
                        box_size=7,
                        border=2,
                    )


                    qr.add_data(
                        resolver_url
                    )

                    qr.make(
                        fit=True
                    )


                    img = qr.make_image(
                        fill_color="black",
                        back_color="white",
                    )


                    buf = io.BytesIO()

                    img.save(
                        buf,
                        format="PNG",
                    )

                    qr_bytes = (
                        buf.getvalue()
                    )


                    # ------------------------------------------------
                    # QR DISPLAY
                    # ------------------------------------------------

                    qr_left, qr_center, qr_right = (
                        st.columns(
                            [1, 1.2, 1]
                        )
                    )


                    with qr_center:

                        st.image(
                            qr_bytes,
                            width=260,
                        )

                        st.caption(
                            "Lightweight resolver QR"
                        )


                    # ------------------------------------------------
                    # PAYMENT DETAILS
                    # ------------------------------------------------

                    info_left, info_right = (
                        st.columns(2)
                    )


                    with info_left:

                        st.metric(
                            "Payment Amount",
                            f"₹{payment_data['amount']:.2f}",
                        )


                    with info_right:

                        st.metric(
                            "Transaction",
                            payment_data[
                                "transaction_id"
                            ],
                        )


                    st.success(
                        "⚡ Lightweight Dynamic QR Generated"
                    )


                    st.info(
                        "The QR contains only a short resolver URL. "
                        "The resolver retrieves the payment session "
                        "and constructs the UPI payment link."
                    )


                    # ------------------------------------------------
                    # QR PAYLOAD
                    # ------------------------------------------------

                    with st.expander(
                        "🔍 View QR Payload"
                    ):

                        st.code(
                            resolver_url,
                            language="text",
                        )


                    # ------------------------------------------------
                    # PAYMENT SESSION DETAILS
                    # ------------------------------------------------

                    with st.expander(
                        "💳 Payment Session Details"
                    ):

                        st.write(
                            f"**Merchant:** "
                            f"{payment_data['merchant_name']}"
                        )

                        st.write(
                            f"**UPI ID:** "
                            f"{payment_data['upi_id']}"
                        )

                        st.write(
                            f"**Amount:** "
                            f"₹{payment_data['amount']:.2f}"
                        )

                        st.write(
                            f"**Transaction ID:** "
                            f"{payment_data['transaction_id']}"
                        )

                        st.write(
                            f"**Payment Token:** "
                            f"{payment_data['token']}"
                        )


# ===================================================================
# TAB 2 — MODEL PERFORMANCE
# ===================================================================

with tab2:

    st.header(
        "📊 Model Performance Laboratory"
    )

    st.caption(
        "Compare classification and regression models using "
        "accuracy, F1, error and latency metrics."
    )


    if not benchmark:

        st.warning(
            "benchmark_results.json was not found. "
            "Run the training script first."
        )


    else:

        classification = benchmark.get(
            "classification",
            {},
        )

        regression = benchmark.get(
            "regression",
            {},
        )


        # ===========================================================
        # CLASSIFICATION
        # ===========================================================

        st.subheader(
            "🤖 Classification — Random Forest vs XGBoost"
        )

        st.caption(
            "Classification receives ONLY the three degradation "
            "predictions produced by regression."
        )


        if classification:

            rows = []


            for name, data in classification.items():

                rows.append(
                    {
                        "Model": name,

                        "Accuracy (%)":
                            data.get(
                                "accuracy",
                                0,
                            ) * 100,

                        "Weighted F1":
                            data.get(
                                "weighted_f1",
                                0,
                            ),

                        "Macro F1":
                            data.get(
                                "macro_f1",
                                0,
                            ),

                        "Precision":
                            data.get(
                                "precision_weighted",
                                0,
                            ),

                        "Recall":
                            data.get(
                                "recall_weighted",
                                0,
                            ),

                        "Train Time (ms)":
                            data.get(
                                "training_time_ms",
                                0,
                            ),

                        "P95 Inference (ms)":
                            data.get(
                                "inference_p95_ms",
                                0,
                            ),

                        "Model Size (MB)":
                            data.get(
                                "model_size_mb",
                                0,
                            ),
                    }
                )


            class_df = pd.DataFrame(
                rows
            )


            st.dataframe(
                class_df.style.format(
                    {
                        "Accuracy (%)":
                            "{:.2f}",

                        "Weighted F1":
                            "{:.4f}",

                        "Macro F1":
                            "{:.4f}",

                        "Precision":
                            "{:.4f}",

                        "Recall":
                            "{:.4f}",

                        "Train Time (ms)":
                            "{:.2f}",

                        "P95 Inference (ms)":
                            "{:.4f}",

                        "Model Size (MB)":
                            "{:.3f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
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
                title=(
                    f"Classification — "
                    f"{metric_choice}"
                ),
            )


            fig.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside",
            )


            fig.update_layout(
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                margin=dict(
                    t=60,
                    b=30,
                    l=20,
                    r=20,
                ),
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
            )


            st.success(
                f"🏆 Active Classification Winner: "
                f"{winner_class}"
            )


        else:

            st.info(
                "No classification benchmark results "
                "are available."
            )


        st.divider()


        # ===========================================================
        # REGRESSION
        # ===========================================================

        st.subheader(
            "📈 Regression — Random Forest vs XGBoost"
        )

        st.caption(
            "Regression estimates continuous degradation scores "
            "for network, gateway and bank infrastructure."
        )


        if regression:

            rows = []


            for name, data in regression.items():

                rows.append(
                    {
                        "Model": name,

                        "MAE":
                            data.get(
                                "mae",
                                0,
                            ),

                        "MSE":
                            data.get(
                                "mse",
                                0,
                            ),

                        "RMSE":
                            data.get(
                                "rmse",
                                0,
                            ),

                        "R²":
                            data.get(
                                "r2",
                                0,
                            ),

                        "Train Time (ms)":
                            data.get(
                                "training_time_ms",
                                0,
                            ),

                        "P95 Inference (ms)":
                            data.get(
                                "inference_p95_ms",
                                0,
                            ),

                        "Model Size (MB)":
                            data.get(
                                "model_size_mb",
                                0,
                            ),
                    }
                )


            reg_df = pd.DataFrame(
                rows
            )


            st.dataframe(
                reg_df.style.format(
                    {
                        "MAE":
                            "{:.5f}",

                        "MSE":
                            "{:.5f}",

                        "RMSE":
                            "{:.5f}",

                        "R²":
                            "{:.5f}",

                        "Train Time (ms)":
                            "{:.2f}",

                        "P95 Inference (ms)":
                            "{:.4f}",

                        "Model Size (MB)":
                            "{:.3f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )


            reg_metric = st.selectbox(
                "Regression metric",
                [
                    "MAE",
                    "MSE",
                    "RMSE",
                    "R²",
                    "P95 Inference (ms)",
                ],
                key="regression_metric",
            )


            fig = px.bar(
                reg_df,
                x="Model",
                y=reg_metric,
                text=reg_metric,
                title=(
                    f"Regression — "
                    f"{reg_metric}"
                ),
            )


            fig.update_traces(
                texttemplate="%{text:.4f}",
                textposition="outside",
            )


            fig.update_layout(
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                margin=dict(
                    t=60,
                    b=30,
                    l=20,
                    r=20,
                ),
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
            )


            st.success(
                f"🏆 Active Regression Winner: "
                f"{winner_reg}"
            )


        else:

            st.info(
                "No regression benchmark results "
                "are available."
            )


        st.divider()


        # ===========================================================
        # WHY METRICS MATTER
        # ===========================================================

        st.subheader(
            "⚡ Why These Metrics Matter"
        )


        metric_cols = st.columns(4)


        with metric_cols[0]:

            st.metric(
                "Classification",
                "Macro F1",
                "Balanced class performance",
            )


        with metric_cols[1]:

            st.metric(
                "Regression",
                "R²",
                "Explained variation",
            )


        with metric_cols[2]:

            st.metric(
                "Production",
                "P95",
                "Inference latency",
            )


        with metric_cols[3]:

            st.metric(
                "Deployment",
                "Model Size",
                "Memory footprint",
            )


        st.info(
            "Macro F1 is useful when every recovery action matters. "
            "For regression, lower MAE/MSE/RMSE means smaller errors, "
            "while higher R² indicates better explained variation."
        )


# ===================================================================
# TAB 3 — FEATURE INTELLIGENCE
# ===================================================================

with tab3:

    st.header(
        "🧠 Feature Intelligence"
    )

    st.caption(
        "Understand the two-stage ML architecture used by ResQ-QR."
    )


    # ===============================================================
    # REGRESSION FEATURES
    # ===============================================================

    st.subheader(
        "📡 Regression Input Features"
    )

    st.caption(
        "These 11 raw telemetry features are passed to the "
        "regression models."
    )


    regression_feature_df = pd.DataFrame(
        {
            "Feature":
                REGRESSION_FEATURES,

            "Role": [
                "Network telemetry",
                "Network telemetry",
                "Network telemetry",

                "Gateway telemetry",
                "Gateway telemetry",
                "Gateway telemetry",

                "Bank telemetry",
                "Bank telemetry",
                "Bank telemetry",

                "Payment error telemetry",
                "Payment error telemetry",
            ],
        }
    )


    st.dataframe(
        regression_feature_df,
        use_container_width=True,
        hide_index=True,
    )


    st.divider()


    # ===============================================================
    # CLASSIFIER FEATURES
    # ===============================================================

    st.subheader(
        "🧠 Classification Input Features"
    )

    st.caption(
        "The classifier does NOT receive raw telemetry. "
        "It receives only these three regression predictions."
    )


    classification_feature_df = pd.DataFrame(
        {
            "Feature":
                CLASSIFICATION_FEATURES,

            "Source": [
                "Regression",
                "Regression",
                "Regression",
            ],
        }
    )


    st.dataframe(
        classification_feature_df,
        use_container_width=True,
        hide_index=True,
    )


    st.divider()


    # ===============================================================
    # FEATURE IMPORTANCE
    # ===============================================================

    st.subheader(
        "🔬 Classification Feature Importance"
    )


    if hasattr(
        classifier,
        "feature_importances_",
    ):

        classifier_features = list(
            getattr(
                classifier,
                "feature_names_in_",
                CLASSIFICATION_FEATURES,
            )
        )


        importance_df = pd.DataFrame(
            {
                "Feature":
                    classifier_features,

                "Importance":
                    classifier.feature_importances_,
            }
        ).sort_values(
            "Importance",
            ascending=False,
        )


        top_features = (
            importance_df
            .head(12)
            .sort_values("Importance")
        )


        fig = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            text="Importance",
            title="Classification Feature Importance",
        )


        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
        )


        fig.update_layout(
            height=520,
            xaxis_title="Feature Importance",
            yaxis_title="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            margin=dict(
                l=20,
                r=70,
                t=70,
                b=30,
            ),
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
        )


        st.subheader(
            "📋 Complete Classification Feature Ranking"
        )


        st.dataframe(
            importance_df.style.format(
                {
                    "Importance":
                        "{:.5f}"
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


    else:

        st.info(
            "The loaded classifier does not expose "
            "feature_importances_."
        )


    st.divider()


    # ===============================================================
    # HOW RESQ-QR THINKS
    # ===============================================================

    st.subheader(
        "🔬 How ResQ-QR Thinks"
    )


    info1, info2 = st.columns(
        2,
        gap="large",
    )


    with info1:

        with st.container(border=True):

            st.subheader(
                "📈 Regression"
            )

            st.write(
                "The regression stage receives **11 raw telemetry "
                "features** and predicts **three continuous "
                "infrastructure degradation scores**."
            )

            st.markdown(
                "**Flow**"
            )

            st.info(
                "Raw telemetry → Network degradation\n\n"
                "Raw telemetry → Gateway degradation\n\n"
                "Raw telemetry → Bank degradation"
            )


    with info2:

        with st.container(border=True):

            st.subheader(
                "🧠 Classification"
            )

            st.write(
                "The classification stage receives **only the three "
                "predicted degradation values** and maps them to "
                "**Class 1, Class 2 or Class 3**."
            )

            st.markdown(
                "**Flow**"
            )

            st.info(
                "3 degradation predictions\n\n"
                "↓\n\n"
                "Class 1 / Class 2 / Class 3\n\n"
                "↓\n\n"
                "Recovery action"
            )


    # ===============================================================
    # DECISION PIPELINE
    # ===============================================================

    st.subheader(
        "⚡ ResQ-QR Decision Pipeline"
    )

    st.caption(
        "End-to-end flow from raw payment telemetry to intelligent "
        "fallback recovery."
    )


    # ---------------------------------------------------------------
    # PIPELINE STEPS
    # ---------------------------------------------------------------

    pipeline_steps = [
        (
            "1",
            "📡",
            "Raw Telemetry",
            "Collect network, gateway, bank and payment error telemetry."
        ),
        (
            "2",
            "📈",
            "Regression",
            "Predict infrastructure degradation scores."
        ),
        (
            "3",
            "🌐",
            "Network Degradation",
            "Estimate degradation of the network path."
        ),
        (
            "4",
            "🔌",
            "Gateway Degradation",
            "Estimate degradation of the payment gateway."
        ),
        (
            "5",
            "🏦",
            "Bank Degradation",
            "Estimate degradation of the banking infrastructure."
        ),
        (
            "6",
            "🧠",
            "Classification → Action",
            "Map degradation predictions to the safest recovery action."
        ),
        (
            "7",
            "🔄",
            "Convert to Lite String",
            "If network degradation is detected, create a lightweight "
            "payment reference instead of placing the complete payment "
            "payload inside the QR."
        ),
        (
            "8",
            "📱",
            "Generate Dynamic QR",
            "Generate a QR containing the lightweight resolver URL "
            "for payment recovery."
        ),
    ]


    # ---------------------------------------------------------------
    # FIRST ROW — STEPS 1 TO 4
    # ---------------------------------------------------------------

    row1 = st.columns(4, gap="medium")

    for col, (number, icon, title, description) in zip(
        row1,
        pipeline_steps[:4],
    ):

        with col:

            with st.container(border=True):

                st.markdown(
                    f"### {icon}"
                )

                st.markdown(
                    f"**STEP {number} — {title}**"
                )

                st.caption(
                    description
                )


    # ---------------------------------------------------------------
    # FLOW INDICATOR
    # ---------------------------------------------------------------

    st.markdown(
        "⬇️  **Infrastructure degradation is passed forward**  ⬇️"
    )


    # ---------------------------------------------------------------
    # SECOND ROW — STEPS 5 TO 8
    # ---------------------------------------------------------------

    row2 = st.columns(4, gap="medium")

    for col, (number, icon, title, description) in zip(
        row2,
        pipeline_steps[4:],
    ):

        with col:

            with st.container(border=True):

                st.markdown(
                    f"### {icon}"
                )

                st.markdown(
                    f"**STEP {number} — {title}**"
                )

                st.caption(
                    description
                )
        