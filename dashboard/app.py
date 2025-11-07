#!/usr/bin/env python3
# 🧩 DeepHawk – AI-Assisted Malware Classifier (SOC Themed Dashboard)

import os
import time
import datetime
import pandas as pd
import numpy as np
import joblib
import shap
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ================================================================
# 📂 PATHS
# ================================================================
PROJECT = os.path.expanduser("~/Projects/deephawk")
MODEL_PATH = os.path.join(PROJECT, "data", "processed", "ember_model.pkl")
DATA_DIR = os.path.join(PROJECT, "data", "processed")
LOG_PATH = os.path.join(DATA_DIR, "prediction_log.csv")
LOGO_PATH = os.path.join(DATA_DIR, "deep_hawk_logo.png")  # optional

# ================================================================
# ⚙️ STREAMLIT CONFIG
# ================================================================
st.set_page_config(page_title="🧩 DeepHawk Malware Classifier", layout="wide", page_icon="🦅")

# ================================================================
# 🎨 THEME + HUD
# ================================================================
def inject_css():
    css = """
    <style>
    .reportview-container, .main {
        background: radial-gradient(ellipse at top left, #07111a 0%, #02040a 70%);
        color: #e6f2ff;
    }
    .block-container { padding: 1rem 1.4rem 2rem 1.4rem; }
    /* === SIDEBAR (Dark Matte SOC Style) === */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0b0e 0%, #121417 100%) !important;
    color: #e5e7eb;
    border-right: 1px solid rgba(255,255,255,0.05);
}

.stSidebar > div:first-child {
    padding: 1.5rem 1rem;
}

/* Sidebar title */
.stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4 {
    color: #f3f4f6 !important;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* Sidebar radio & labels */
.stSidebar label, .stRadio > label {
    color: #d1d5db !important;
    font-weight: 400;
}

.stSidebar .stRadio > div > label {
    color: #d1d5db !important;
}

.stSidebar .stRadio div[role="radiogroup"] label {
    color: #d1d5db !important;
    transition: all 0.2s ease;
}

.stSidebar .stRadio div[role="radiogroup"] label:hover {
    color: #f9fafb !important;
}

/* Sidebar buttons */
.stSidebar button {
    background: #1f2937 !important;
    border: 1px solid rgba(255,255,255,0.08);
    color: #e5e7eb !important;
    border-radius: 8px;
    transition: all 0.2s ease;
}
.stSidebar button:hover {
    background: #374151 !important;
    border-color: rgba(255,255,255,0.15);
}

/* Sidebar footer text */
.stSidebar .stMarkdown {
    color: #9ca3af !important;
}

/* Highlight active radio selection */
.stSidebar div[role="radio"][aria-checked="true"] label {
    color: #ffffff !important;
    background: #1c1f25 !important;
    border-radius: 6px;
    padding: 4px 8px;
}

/* Sidebar model info badge */
.stSidebar code {
    background: #1f2937 !important;
    color: #e5e7eb !important;
    border-radius: 4px;
    padding: 2px 5px;
    font-size: 12px;
}


    /* Metrics */
    @keyframes pulseGreen {
        0% { box-shadow: 0 0 5px #00f1a1aa; }
        50% { box-shadow: 0 0 25px #00f1a1; }
        100% { box-shadow: 0 0 5px #00f1a1aa; }
    }
    @keyframes pulseRed {
        0% { box-shadow: 0 0 5px #ff004466; }
        50% { box-shadow: 0 0 25px #ff0044; }
        100% { box-shadow: 0 0 5px #ff004466; }
    }
    .stMetric {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 10px;
        padding: 8px 12px;
        border: 1px solid rgba(255,255,255,0.03);
    }
    .stMetric .stMetricValue {
        color: #e6f2ff;
        font-weight: 700;
        text-shadow: 0 0 8px #00f1a1;
    }
    .stMetric:has(div[data-testid="stMetricValue"]:contains("Malicious")) { animation: pulseRed 3s infinite; }
    .stMetric:has(div[data-testid="stMetricValue"]:contains("Benign")),
    .stMetric:has(div[data-testid="stMetricValue"]:contains("%")) { animation: pulseGreen 3s infinite; }

    //* === HUD TOP BAR (Dark Matte Version) === */
.hud-bar {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    background: linear-gradient(90deg, #0b0c10, #1a1f25);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    z-index: 999;
    padding: 8px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'Consolas', monospace;
    color: #d1d5db; /* soft gray text */
    font-size: 14px;
    letter-spacing: 0.3px;
}

/* Title */
.hud-title {
    color: #f0f0f0;
    font-weight: 700;
}

/* Status */
.hud-status.ok {
    color: #9ae6b4; /* soft green */
}

/* Clock & Metrics */
.hud-clock {
    color: #cbd5e1; /* light gray-blue */
}
.hud-preds {
    color: #a3bffa; /* soft blue */
    font-weight: 500;
}

/* Divider line under HUD */
.hud-bar::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    height: 1px;
    width: 100%;
    background: rgba(255,255,255,0.08);
}

.main > div:first-child {
    margin-top: 65px !important;
}

	
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_hud():
    """Live top HUD bar."""
    count = 0
    if os.path.exists(LOG_PATH):
        try:
            df = pd.read_csv(LOG_PATH)
            count = len(df)
        except Exception:
            pass
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    hud_html = f"""
    <div class="hud-bar">
        <div class="hud-section">
            <span class="hud-title">🦅 DeepHawk SOC Console</span>
        </div>
        <div class="hud-section hud-center">
            <span class="hud-status ok">● System Status: <b>Operational</b></span>
            <span class="hud-clock">🕒 {current_time}</span>
        </div>
        <div class="hud-section hud-right">
            <span class="hud-preds">📈 Predictions Logged: <b>{count}</b></span>
        </div>
    </div>
    """
    st.markdown(hud_html, unsafe_allow_html=True)


inject_css()
with st.container():
    render_hud()

# ================================================================
# 🧠 MODEL LOADING
# ================================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ================================================================
# 🧾 LOGGING
# ================================================================
def log_prediction(pred, prob):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = pd.DataFrame([{
        "timestamp": ts,
        "prediction": int(pred),
        "confidence": float(prob)
    }])
    row.to_csv(LOG_PATH, mode="a", index=False, header=not os.path.exists(LOG_PATH))

# ================================================================
# 🎛️ SIDEBAR CONTROLS
# ================================================================
with st.sidebar:
    if os.path.isfile(LOGO_PATH):
        st.image(LOGO_PATH, width=140, caption="DeepHawk", output_format="PNG")
    st.title("🧭 DeepHawk Controls")
    mode = st.radio(
        "Mode",
        ["Single Sample", "Batch Evaluation", "Explainability", "Metrics & History"]
    )
    st.markdown("---")
    st.markdown("**Model:** `LightGBM`  •  **Dataset:** EMBER subset")
    st.caption("Built by Slim Ben Tanfous — Virtual SOC demo")

# ================================================================
# 🧠 1️⃣ SINGLE SAMPLE
# ================================================================
if mode == "Single Sample":
    st.header("🧠 Malware Prediction (Single Sample)")
    uploaded_file = st.file_uploader("Upload a CSV sample (1 row)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        for col in ["label", "mal_prob"]:
            if col in df.columns:
                df = df.drop(columns=[col])
        st.subheader("Uploaded Sample Preview")
        st.dataframe(df.head())

        model_features = model.booster_.feature_name()
        if len(df.columns) != len(model_features):
            st.warning(f"⚠️ Model expects {len(model_features)} features, but CSV has {len(df.columns)}")

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][pred]

        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown(f"### Prediction: {'🦠 Malicious' if pred else '✅ Benign'}")
            st.write("Confidence")
            st.progress(min(max(prob, 0.0), 1.0))
        with col2:
            label = "Malicious" if pred else "Benign"
            st.metric("Result", label, delta=f"{prob*100:.2f} %")

        log_prediction(pred, prob)
        st.success("Prediction logged ✅")

# ================================================================
# 📊 2️⃣ BATCH EVALUATION
# ================================================================
elif mode == "Batch Evaluation":
    st.header("📊 Batch Evaluation")
    selected = st.selectbox("Choose test dataset", [
        "malicious_samples.csv",
        "top_malicious_by_confidence.csv",
        "synth_malicious.csv",
        "ember_expanded.csv"
    ])

    path = os.path.join(DATA_DIR, selected)
    if os.path.isfile(path):
        df = pd.read_csv(path)
        X = df.drop(columns=["label", "mal_prob"], errors="ignore")
        y = df["label"] if "label" in df.columns else None

        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        st.write(f"Samples: {len(preds)}")
        st.write(f"Mean malicious probability: {probs.mean():.4f}")

        if y is not None:
            report = classification_report(y, preds, output_dict=True, zero_division=0)
            st.subheader("Classification Report")
            st.dataframe(pd.DataFrame(report).transpose())
    else:
        st.error("Dataset not found.")

# ================================================================
# 🔍 3️⃣ EXPLAINABILITY
# ================================================================
elif mode == "Explainability":
    st.header("🔍 Model Explainability (SHAP)")
    df = pd.read_csv(os.path.join(DATA_DIR, "ember_expanded.csv"))
    X = df.drop(columns=["label"], errors="ignore").sample(200, random_state=42)

    with st.spinner("Computing SHAP values — this may take a moment..."):
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

    st.subheader("📈 Global Feature Importance")
    fig, ax = plt.subplots(figsize=(10,6))
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig)

    st.subheader("💡 Single Sample Explanation")
    i = st.slider("Select sample index", 0, len(X)-1, 0)
    fig2, ax2 = plt.subplots(figsize=(8,4))
    shap.waterfall_plot(shap_values[i], show=False)
    st.pyplot(fig2)

# ================================================================
# 📈 4️⃣ METRICS & HISTORY
# ================================================================
elif mode == "Metrics & History":
    st.header("📈 DeepHawk Metrics & Prediction History")

    if not os.path.isfile(LOG_PATH):
        st.warning("No prediction log found yet. Run a few predictions first.")
    else:
        log_df = pd.read_csv(LOG_PATH)
        st.subheader("Recent Predictions")
        st.dataframe(log_df.tail(10))

        total = len(log_df)
        mal = (log_df["prediction"] == 1).sum()
        ben = total - mal
        avg_conf = log_df["confidence"].mean() * 100 if total > 0 else 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Predictions", total)
        k2.metric("Malicious Count", mal, delta_color="inverse")
        k3.metric("Benign Count", ben)
        k4.metric("Avg Confidence", f"{avg_conf:.2f} %")

        log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
        log_df = log_df.sort_values("timestamp")

        left, right = st.columns(2)
        with left:
            st.subheader("Confidence Over Time")
            st.line_chart(log_df.set_index("timestamp")["confidence"])
        with right:
            st.subheader("Class Distribution")
            st.bar_chart(log_df["prediction"].map({0:"Benign",1:"Malicious"}).value_counts())

        st.download_button(
            label="📥 Download Full Log CSV",
            data=log_df.to_csv(index=False),
            file_name="deephawk_prediction_log.csv",
            mime="text/csv",
        )

        if st.button("🗑️ Reset Logs"):
            os.remove(LOG_PATH)
            st.success("Prediction log reset.")

