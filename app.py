#!/usr/bin/env python3
"""
app.py — Network Intrusion Detection System Dashboard
Streamlit-based cybersecurity monitoring interface.

Run with:
    streamlit run app.py
"""

import os
import sys
import io
import time
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (
    load_preprocessing_artefacts,
    preprocess_for_prediction,
    generate_demo_dataset,
    preprocess_dataset,
    clean_data,
    get_dataset_stats,
)
from src.training import (
    load_models,
    models_exist,
    evaluate_model,
    get_feature_importance,
    train_random_forest,
    train_isolation_forest,
    save_models,
)
from src.prediction import predict_single, batch_predict
from src.hybrid_detector import (
    hybrid_decision_batch,
    hybrid_decision_single,
    build_results_dataframe,
    get_summary_counts,
    RESULT_LABELS,
    RESULT_EMOJI,
    RESULT_COLORS,
    NORMAL,
    KNOWN_ATTACK,
    ANOMALOUS,
)
from src.preprocessing import save_preprocessing_artefacts

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ PAGE CONFIG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="Network IDS — Hybrid ML Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ CUSTOM CSS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def inject_css():
    st.markdown("""
    <style>
    /* ── Global dark theme overrides ── */
    .stApp {
        background: linear-gradient(135deg, #0a0e17 0%, #0d1321 50%, #0f1729 100%);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #111927 100%);
        border-right: 1px solid rgba(0, 230, 118, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #00e676 !important;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(13, 25, 39, 0.9), rgba(17, 34, 51, 0.9));
        border: 1px solid rgba(0, 230, 118, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetric"] label {
        color: #8892a4 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e0e6ed !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* ── Headers ── */
    .main-header {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00e676, #00bcd4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #8892a4;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* ── Cards ── */
    .card {
        background: linear-gradient(135deg, rgba(13, 25, 39, 0.95), rgba(17, 34, 51, 0.85));
        border: 1px solid rgba(0, 230, 118, 0.15);
        border-radius: 14px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }

    /* ── Result badges ── */
    .result-normal {
        background: rgba(0, 230, 118, 0.15);
        color: #00e676;
        padding: 8px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid rgba(0, 230, 118, 0.3);
    }
    .result-attack {
        background: rgba(255, 23, 68, 0.15);
        color: #ff1744;
        padding: 8px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid rgba(255, 23, 68, 0.3);
    }
    .result-anomalous {
        background: rgba(255, 145, 0, 0.15);
        color: #ff9100;
        padding: 8px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid rgba(255, 145, 0, 0.3);
    }

    /* ── Demo warning banner ── */
    .demo-banner {
        background: rgba(255, 145, 0, 0.12);
        border: 1px solid rgba(255, 145, 0, 0.35);
        border-radius: 10px;
        padding: 14px 20px;
        color: #ff9100;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(13, 25, 39, 0.8);
        border: 1px solid rgba(0, 230, 118, 0.15);
        border-radius: 8px 8px 0 0;
        color: #8892a4;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0, 230, 118, 0.1) !important;
        border-color: #00e676 !important;
        color: #00e676 !important;
    }

    /* ── Divider glow ── */
    hr {
        border-color: rgba(0, 230, 118, 0.2) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e17; }
    ::-webkit-scrollbar-thumb { background: #1a3a4a; border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ HELPERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource
def cached_load_models():
    """Cache-load models so they persist across reruns."""
    if not models_exist(MODELS_DIR):
        return None
    try:
        m = load_models(MODELS_DIR)
        p = load_preprocessing_artefacts(MODELS_DIR)
        return {**m, **p}
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


def is_demo_mode():
    """Check if the current models were trained on demo data."""
    meta_path = os.path.join(MODELS_DIR, "selected_features.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        # Demo data uses our synthetic feature names
        if "Flow Duration" in meta.get("selected_features", []):
            return True
    return False


def plotly_dark_layout(fig, title=""):
    """Apply consistent dark theme to plotly figures."""
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e0e6ed", size=16)),
        paper_bgcolor="rgba(13, 25, 39, 0.0)",
        plot_bgcolor="rgba(13, 25, 39, 0.6)",
        font=dict(color="#8892a4"),
        xaxis=dict(gridcolor="rgba(0, 230, 118, 0.08)", zerolinecolor="rgba(0, 230, 118, 0.1)"),
        yaxis=dict(gridcolor="rgba(0, 230, 118, 0.08)", zerolinecolor="rgba(0, 230, 118, 0.1)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ SIDEBAR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_sidebar():
    with st.sidebar:
        st.markdown("# 🛡️ Network IDS")
        st.markdown("**Hybrid ML Detection System**")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "📂 Analyze CSV", "🔍 Single Prediction", "📈 Model Performance", "⚙️ Train Models"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Model status indicator
        if models_exist(MODELS_DIR):
            st.success("Models loaded ✓")
            if is_demo_mode():
                st.warning("Demo models active")
        else:
            st.error("Models not trained")
            st.caption("Run `python train.py` first")

        st.markdown("---")
        st.caption("B.Tech Semester Project")
        st.caption("Hybrid ML Approach for")
        st.caption("Network Intrusion Detection")

        return page


# ━━━━━━━━━━━━━━━━━━━━━━━━ PAGE 1: DASHBOARD ━━━━━━━━━━━━━━━━━━━━━━━━━━

def page_dashboard(ctx):
    st.markdown('<div class="main-header">📊 Security Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time hybrid intrusion detection overview</div>', unsafe_allow_html=True)

    if ctx is None:
        st.error("⚠️ **Models not trained yet.** Run `python train.py` first.")
        st.info("Or use `python train.py --demo` to train with synthetic data for quick testing.")
        _render_architecture()
        return

    if is_demo_mode():
        st.markdown(
            '<div class="demo-banner">⚠ DEMO MODE — Predictions are generated using '
            'demonstration data and must not be used as research results.</div>',
            unsafe_allow_html=True,
        )

    metrics = ctx.get("metrics", {})

    # ── Row 1: Key metrics ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    with c2:
        st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
    with c3:
        st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
    with c4:
        st.metric("F1 Score", f"{metrics.get('f1_score', 0):.2%}")

    st.markdown("---")

    # ── Row 2: Model status cards ──
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🌲 Random Forest")
        st.markdown("**Status:** Trained ✓")
        st.markdown(f"**Type:** Supervised Classifier")
        st.markdown(f"**Role:** Detect known attack patterns")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🌀 Isolation Forest")
        st.markdown("**Status:** Trained ✓")
        st.markdown(f"**Type:** Anomaly Detector")
        st.markdown(f"**Role:** Detect unseen / zero-day anomalies")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts ──
    c1, c2 = st.columns(2)

    with c1:
        # Classification report chart
        report = metrics.get("classification_report", {})
        chart_data = {}
        for key, val in report.items():
            if isinstance(val, dict) and "f1-score" in val:
                chart_data[key] = {
                    "Precision": val.get("precision", 0),
                    "Recall": val.get("recall", 0),
                    "F1-Score": val.get("f1-score", 0),
                }
        if chart_data:
            chart_df = pd.DataFrame(chart_data).T
            chart_df = chart_df.reset_index().rename(columns={"index": "Class"})
            chart_df = chart_df[~chart_df["Class"].isin(["accuracy", "macro avg", "weighted avg"])]
            if not chart_df.empty:
                fig = px.bar(
                    chart_df.melt(id_vars="Class", var_name="Metric", value_name="Score"),
                    x="Class", y="Score", color="Metric", barmode="group",
                    color_discrete_sequence=["#00e676", "#00bcd4", "#7c4dff"],
                )
                fig = plotly_dark_layout(fig, "Per-Class Performance")
                st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Feature importance chart
        feat_imp = metrics.get("feature_importance", [])
        if feat_imp:
            fi_df = pd.DataFrame(feat_imp[:10])
            fig = px.bar(
                fi_df, x="importance", y="feature", orientation="h",
                color="importance",
                color_continuous_scale=["#0d1321", "#00e676"],
            )
            fig = plotly_dark_layout(fig, "Top 10 Important Features")
            fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # Architecture diagram
    _render_architecture()


def _render_architecture():
    """Display the hybrid detection architecture."""
    st.markdown("### 🏗️ Hybrid Detection Architecture")
    st.markdown("""
    ```
    Incoming Network Traffic
              │
              ▼
      ┌───────────────┐
      │ Preprocessing │  ← Feature scaling + selection
      └───────┬───────┘
              │
              ▼
      ┌───────────────┐
      │ Random Forest │  ← Supervised classifier
      └───────┬───────┘
              │
        Known Attack?
         /         \\
       YES          NO
        │            │
        ▼            ▼
    🔴 ATTACK   ┌──────────────────┐
                │ Isolation Forest │  ← Anomaly detector
                └────────┬─────────┘
                         │
                   Is Anomaly?
                    /        \\
                  YES         NO
                   │           │
                   ▼           ▼
            🟠 ANOMALOUS   🟢 NORMAL
    ```
    """)


# ━━━━━━━━━━━━━━━━━━━━━━━ PAGE 2: ANALYZE CSV ━━━━━━━━━━━━━━━━━━━━━━━━━

def page_analyze_csv(ctx):
    st.markdown('<div class="main-header">📂 Analyze CSV</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload network traffic CSV for hybrid analysis</div>', unsafe_allow_html=True)

    if ctx is None:
        st.error("⚠️ **Models not trained yet.** Run `python train.py` first.")
        return

    if is_demo_mode():
        st.markdown(
            '<div class="demo-banner">⚠ DEMO MODE — Predictions are generated using '
            'demonstration data and must not be used as research results.</div>',
            unsafe_allow_html=True,
        )

    uploaded = st.file_uploader("Upload a CSV file with network traffic data", type=["csv"])

    if uploaded is not None:
        with st.spinner("Reading CSV ..."):
            df = pd.read_csv(uploaded, low_memory=False)
            df.columns = df.columns.str.strip()

        st.success(f"Loaded **{len(df):,}** records with **{len(df.columns)}** columns.")

        with st.expander("Preview uploaded data", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)

        if st.button("🚀 Run Hybrid Analysis", type="primary", use_container_width=True):
            _run_csv_analysis(df, ctx)


def _run_csv_analysis(df: pd.DataFrame, ctx: dict):
    """Execute the full hybrid analysis pipeline on uploaded CSV."""
    progress = st.progress(0, text="Starting analysis ...")

    # Step 1: Preprocess
    progress.progress(20, text="Preprocessing data ...")
    try:
        X = preprocess_for_prediction(
            df,
            scaler=ctx["scaler"],
            selector=ctx["selector"],
            selected_features=ctx["selected_features"],
            all_feature_names=ctx.get("all_feature_names"),
        )
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return

    # Step 2: Predict
    progress.progress(50, text="Running Random Forest predictions ...")
    results = batch_predict(ctx["random_forest"], ctx["isolation_forest"], X)

    # Step 3: Hybrid decision
    progress.progress(75, text="Computing hybrid decisions ...")
    hybrid_preds = hybrid_decision_batch(
        results["supervised_predictions"],
        results["anomaly_predictions"],
    )

    # Step 4: Build results
    progress.progress(90, text="Building results ...")
    results_df = build_results_dataframe(
        results["supervised_predictions"],
        results["anomaly_predictions"],
        hybrid_preds,
        label_names=ctx.get("label_names"),
    )
    summary = get_summary_counts(hybrid_preds)

    progress.progress(100, text="Done!")
    time.sleep(0.3)
    progress.empty()

    # ── Summary metrics ──
    st.markdown("### 📊 Analysis Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Records", f"{results['total_samples']:,}")
    with c2:
        st.metric("🟢 Normal", f"{summary.get('NORMAL', 0):,}")
    with c3:
        st.metric("🔴 Known Attacks", f"{summary.get('KNOWN ATTACK', 0):,}")
    with c4:
        st.metric("🟠 Anomalous", f"{summary.get('ANOMALOUS', 0):,}")

    st.caption(f"Processing time: {results['processing_time_ms']:.1f} ms")

    # ── Distribution chart ──
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(data=[go.Pie(
            labels=list(summary.keys()),
            values=list(summary.values()),
            marker=dict(colors=[RESULT_COLORS[NORMAL], RESULT_COLORS[KNOWN_ATTACK], RESULT_COLORS[ANOMALOUS]]),
            hole=0.4,
        )])
        fig = plotly_dark_layout(fig, "Detection Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        bar_df = pd.DataFrame(list(summary.items()), columns=["Category", "Count"])
        fig = px.bar(
            bar_df, x="Category", y="Count",
            color="Category",
            color_discrete_map={
                "NORMAL": RESULT_COLORS[NORMAL],
                "KNOWN ATTACK": RESULT_COLORS[KNOWN_ATTACK],
                "ANOMALOUS": RESULT_COLORS[ANOMALOUS],
            },
        )
        fig = plotly_dark_layout(fig, "Category Counts")
        st.plotly_chart(fig, use_container_width=True)

    # ── Results table ──
    st.markdown("### 📋 Detailed Results")
    st.dataframe(
        results_df,
        use_container_width=True,
        height=400,
        column_config={
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Record": st.column_config.NumberColumn("Record", width="small"),
        },
    )

    # ── Download ──
    csv_data = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name="hybrid_detection_results.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━ PAGE 3: SINGLE PREDICTION ━━━━━━━━━━━━━━━━━━━━━

def page_single_prediction(ctx):
    st.markdown('<div class="main-header">🔍 Single Traffic Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyze individual network traffic samples</div>', unsafe_allow_html=True)

    if ctx is None:
        st.error("⚠️ **Models not trained yet.** Run `python train.py` first.")
        return

    if is_demo_mode():
        st.markdown(
            '<div class="demo-banner">⚠ DEMO MODE — Predictions are generated using '
            'demonstration data and must not be used as research results.</div>',
            unsafe_allow_html=True,
        )

    selected_features = ctx.get("selected_features", [])
    all_features = ctx.get("all_feature_names", [])

    if not all_features:
        st.warning("Feature metadata not available.")
        return

    st.markdown("### Enter Network Traffic Features")
    st.caption(f"Using **{len(selected_features)}** selected features (out of {len(all_features)} total).")

    # Build input form
    feature_values = {}
    cols = st.columns(3)
    for i, feat in enumerate(all_features):
        with cols[i % 3]:
            highlight = "⭐ " if feat in selected_features else ""
            feature_values[feat] = st.number_input(
                f"{highlight}{feat}",
                value=0.0,
                format="%.4f",
                key=f"feat_{i}",
            )

    st.markdown("---")

    if st.button("⚡ Analyze Traffic", type="primary", use_container_width=True):
        _run_single_prediction(feature_values, ctx)


def _run_single_prediction(feature_values: dict, ctx: dict):
    """Run prediction on a single sample."""
    # Build DataFrame from input
    df = pd.DataFrame([feature_values])

    # Preprocess
    try:
        X = preprocess_for_prediction(
            df,
            scaler=ctx["scaler"],
            selector=ctx["selector"],
            selected_features=ctx["selected_features"],
            all_feature_names=ctx.get("all_feature_names"),
        )
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return

    # Predict
    result = predict_single(ctx["random_forest"], ctx["isolation_forest"], X)

    # Hybrid decision
    hybrid = hybrid_decision_single(result["supervised_prediction"], result["anomaly_prediction"])

    # Display result
    st.markdown("---")
    st.markdown("### 🎯 Analysis Result")

    # Large result badge
    result_label = RESULT_LABELS[hybrid]
    result_emoji = RESULT_EMOJI[hybrid]
    css_class = {NORMAL: "result-normal", KNOWN_ATTACK: "result-attack", ANOMALOUS: "result-anomalous"}[hybrid]

    st.markdown(f'<div style="text-align:center; margin: 20px 0;">'
                f'<span class="{css_class}" style="font-size:1.4rem;">'
                f'{result_emoji} {result_label}</span></div>',
                unsafe_allow_html=True)

    # Detail cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🌲 Supervised Prediction**")
        label_names = ctx.get("label_names", {})
        sup_label = label_names.get(str(result["supervised_prediction"]), "Unknown")
        st.markdown(f"### {sup_label}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🌀 Anomaly Detection**")
        ano_label = "Normal" if result["anomaly_prediction"] == 1 else "Anomalous"
        st.markdown(f"### {ano_label}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**⏱️ Processing Time**")
        st.markdown(f"### {result['processing_time_ms']:.2f} ms")
        st.markdown('</div>', unsafe_allow_html=True)

    # Explanation
    st.markdown("---")
    st.markdown("#### 🔎 How this result was determined")
    if hybrid == KNOWN_ATTACK:
        st.info("The **Random Forest** classifier identified this traffic as a **known attack pattern** "
                "matching patterns learned during training.")
    elif hybrid == ANOMALOUS:
        st.warning("The **Random Forest** classified this as benign, but the **Isolation Forest** "
                   "detected anomalous behaviour. This could indicate a **previously unseen / zero-day attack**.")
    else:
        st.success("Both the **Random Forest** (supervised) and **Isolation Forest** (anomaly detection) "
                   "agree this traffic is **normal**.")


# ━━━━━━━━━━━━━━━━━━━━ PAGE 4: MODEL PERFORMANCE ━━━━━━━━━━━━━━━━━━━━━━━

def page_model_performance(ctx):
    st.markdown('<div class="main-header">📈 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detailed evaluation metrics and analysis</div>', unsafe_allow_html=True)

    if ctx is None:
        st.error("⚠️ **Models not trained yet.** Run `python train.py` first.")
        return

    if is_demo_mode():
        st.markdown(
            '<div class="demo-banner">⚠ DEMO MODE — Predictions are generated using '
            'demonstration data and must not be used as research results.</div>',
            unsafe_allow_html=True,
        )

    metrics = ctx.get("metrics", {})

    # ── Overall metrics ──
    st.markdown("### 📊 Overall Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    with c2:
        st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    with c3:
        st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
    with c4:
        st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")

    st.markdown("---")

    # ── Confusion Matrix ──
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔢 Confusion Matrix")
        cm = metrics.get("confusion_matrix", [])
        if cm:
            label_names = ctx.get("label_names", {})
            labels = [label_names.get(str(i), str(i)) for i in range(len(cm))]

            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale=[[0, "#0d1321"], [0.5, "#00695c"], [1, "#00e676"]],
                text=[[str(val) for val in row] for row in cm],
                texttemplate="%{text}",
                textfont=dict(size=16, color="#e0e6ed"),
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            ))
            fig = plotly_dark_layout(fig, "Confusion Matrix")
            fig.update_layout(
                xaxis_title="Predicted",
                yaxis_title="Actual",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### 📋 Classification Report")
        report = metrics.get("classification_report", {})
        if report:
            report_rows = []
            for key, val in report.items():
                if isinstance(val, dict) and "precision" in val:
                    report_rows.append({
                        "Class": key,
                        "Precision": f"{val['precision']:.4f}",
                        "Recall": f"{val['recall']:.4f}",
                        "F1-Score": f"{val['f1-score']:.4f}",
                        "Support": int(val.get("support", 0)),
                    })
            if report_rows:
                st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Feature Importance ──
    st.markdown("### 🏆 Top Important Features")
    feat_imp = metrics.get("feature_importance", [])
    if feat_imp:
        fi_df = pd.DataFrame(feat_imp[:10])

        fig = px.bar(
            fi_df, x="importance", y="feature", orientation="h",
            color="importance",
            color_continuous_scale=["#1a237e", "#00e676"],
        )
        fig = plotly_dark_layout(fig, "Random Forest Feature Importance (Top 10)")
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        # Table
        for i, fi in enumerate(feat_imp[:10], 1):
            pct = fi["importance"] * 100
            st.markdown(f"**{i}.** `{fi['feature']}` — {pct:.2f}%")

    st.markdown("---")

    # ── Selected Features ──
    st.markdown("### 🎯 Selected Features (SelectKBest)")
    selected = ctx.get("selected_features", [])
    if selected:
        for i, f in enumerate(selected, 1):
            st.markdown(f"`{i}.` {f}")
    else:
        st.info("Feature list not available.")


# ━━━━━━━━━━━━━━━━━━━━━ PAGE 5: TRAIN MODELS ━━━━━━━━━━━━━━━━━━━━━━━━━━

def page_train_models():
    st.markdown('<div class="main-header">⚙️ Train Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Train or retrain the hybrid detection models</div>', unsafe_allow_html=True)

    st.markdown("### 📌 Quick Training")
    st.markdown("Train models directly from the dashboard using demo data or uploaded CSV files.")

    method = st.radio(
        "Select training data source:",
        ["🧪 Demo Data (Synthetic)", "📂 Upload CSV"],
        horizontal=True,
    )

    n_features = st.slider("Number of features to select (SelectKBest)", 5, 30, 20)

    train_df = None

    if method == "🧪 Demo Data (Synthetic)":
        n_samples = st.slider("Number of demo samples", 500, 5000, 2000, step=500)
        if st.button("🚀 Generate & Train", type="primary", use_container_width=True):
            train_df = generate_demo_dataset(n_samples=n_samples)
            st.info("⚠️ Using synthetic demo data. Results must NOT be used as research results.")
    else:
        uploaded = st.file_uploader("Upload CIC-IDS2017 CSV", type=["csv"], key="train_upload")
        if uploaded and st.button("🚀 Train on Uploaded Data", type="primary", use_container_width=True):
            train_df = pd.read_csv(uploaded, low_memory=False)
            train_df.columns = train_df.columns.str.strip()

    if train_df is not None:
        _run_training(train_df, n_features)


def _run_training(df: pd.DataFrame, n_features: int):
    """Run the training pipeline from the UI."""
    progress = st.progress(0, text="Starting training ...")

    try:
        # Step 1: Stats
        progress.progress(10, text="Computing dataset statistics ...")
        stats = get_dataset_stats(df)
        st.write(f"📊 **{stats['total_records']:,}** records, **{stats['total_features']}** features")

        # Step 2: Preprocess
        progress.progress(25, text="Preprocessing ...")
        result = preprocess_dataset(df, n_features=n_features)
        st.write(f"✓ Preprocessed — {result['X_train'].shape[0]:,} train / {result['X_test'].shape[0]:,} test")

        # Step 3: Train RF
        progress.progress(45, text="Training Random Forest ...")
        rf = train_random_forest(result["X_train"], result["y_train"])

        # Step 4: Train IF
        progress.progress(60, text="Training Isolation Forest ...")
        normal_mask = result["y_train"] == 0
        X_normal = result["X_train"][normal_mask]
        iso = train_isolation_forest(X_normal)

        # Step 5: Evaluate
        progress.progress(75, text="Evaluating ...")
        metrics = evaluate_model(rf, result["X_test"], result["y_test"], result["label_names"])
        feat_imp = get_feature_importance(rf, result["selected_features"])
        metrics["feature_importance"] = feat_imp

        # Step 6: Save
        progress.progress(90, text="Saving models ...")
        os.makedirs(MODELS_DIR, exist_ok=True)
        save_models(rf, iso, metrics, MODELS_DIR)
        save_preprocessing_artefacts(result, MODELS_DIR)

        progress.progress(100, text="Done!")
        time.sleep(0.3)
        progress.empty()

        st.success("✓ Models trained and saved successfully!")
        st.markdown(f"- **Accuracy:** {metrics['accuracy']:.4f}")
        st.markdown(f"- **F1 Score:** {metrics['f1_score']:.4f}")
        st.info("Refresh the page or switch tabs to use the new models.")

        # Clear the cached models so they reload
        cached_load_models.clear()

    except Exception as e:
        st.error(f"Training failed: {e}")
        import traceback
        st.code(traceback.format_exc())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ MAIN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    inject_css()
    page = render_sidebar()
    ctx = cached_load_models()

    if page == "📊 Dashboard":
        page_dashboard(ctx)
    elif page == "📂 Analyze CSV":
        page_analyze_csv(ctx)
    elif page == "🔍 Single Prediction":
        page_single_prediction(ctx)
    elif page == "📈 Model Performance":
        page_model_performance(ctx)
    elif page == "⚙️ Train Models":
        page_train_models()


if __name__ == "__main__":
    main()
