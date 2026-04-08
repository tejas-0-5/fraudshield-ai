"""
Streamlit Dashboard — Fraud Detection Platform
================================================
Interactive, production-grade fraud monitoring dashboard.

Sections:
    1. 📊 Overview & Dataset Statistics
    2. 🔬 EDA & Visualizations
    3. 🤖 Model Performance & Comparison
    4. 🔍 Transaction Explainability
    5. ⚡ Real-Time Simulation
    6. 🚨 Alert Dashboard

Run with:
    streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import random
from datetime import datetime
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI | Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main theme */
    .main { background-color: #f8fafc; }
    .stApp { font-family: 'Inter', sans-serif; }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
        border-left: 4px solid #3b82f6;
        margin-bottom: 12px;
    }
    .metric-card.fraud { border-left-color: #ef4444; }
    .metric-card.success { border-left-color: #22c55e; }
    .metric-card.warning { border-left-color: #f59e0b; }
    
    /* Alert badges */
    .alert-safe { 
        background: #dcfce7; color: #166534; 
        padding: 4px 12px; border-radius: 20px; font-weight: 600;
        display: inline-block;
    }
    .alert-suspicious { 
        background: #fef3c7; color: #92400e; 
        padding: 4px 12px; border-radius: 20px; font-weight: 600;
        display: inline-block;
    }
    .alert-fraud { 
        background: #fee2e2; color: #991b1b; 
        padding: 4px 12px; border-radius: 20px; font-weight: 600;
        display: inline-block;
    }
    
    /* Section headers */
    .section-header {
        font-size: 22px; font-weight: 700; color: #1e293b;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px; margin-bottom: 20px;
    }
    
    /* Score gauge */
    .score-container {
        text-align: center; padding: 20px;
        background: white; border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Sidebar */
    .css-1d391kg { background-color: #1e293b; }
    
    /* Hide default streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Helper functions ───────────────────────────────────────────────────────────

@st.cache_data
def load_dataset_stats():
    path = os.path.join(os.path.dirname(__file__), 'dataset_stats.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_model_metrics():
    path = os.path.join(os.path.dirname(__file__), 'model_metrics.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_feature_importance():
    path = os.path.join(os.path.dirname(__file__), 'feature_importance.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def get_figure_path(name: str) -> str:
    base = os.path.join(os.path.dirname(__file__), 'static', 'figures')
    return os.path.join(base, f"{name}.png")

def show_figure(name: str, caption: str = ""):
    path = get_figure_path(name)
    if os.path.exists(path):
        st.image(path, caption=caption, use_column_width=True)
    else:
        st.info(f"📊 Run `train_pipeline.py` to generate: {name}")

def risk_badge(risk: str) -> str:
    mapping = {
        'SAFE': '<span class="alert-safe">✅ SAFE</span>',
        'SUSPICIOUS': '<span class="alert-suspicious">⚠️ SUSPICIOUS</span>',
        'FRAUD': '<span class="alert-fraud">🚨 FRAUD</span>'
    }
    return mapping.get(risk, risk)

def score_color(score: float) -> str:
    if score < 30:
        return "#22c55e"
    elif score < 70:
        return "#f59e0b"
    return "#ef4444"

def make_gauge_html(score: float, risk: str) -> str:
    color = score_color(score)
    return f"""
    <div class="score-container">
        <div style="font-size: 48px; font-weight: 900; color: {color};">{score:.1f}</div>
        <div style="font-size: 14px; color: #64748b; margin-top: 4px;">Fraud Score (0–100)</div>
        <div style="margin-top: 12px; font-size: 20px; font-weight: 700; color: {color};">
            {'🚨 ' if risk == 'FRAUD' else '⚠️ ' if risk == 'SUSPICIOUS' else '✅ '}{risk}
        </div>
        <div style="margin-top: 8px;">
            <div style="background: #e2e8f0; border-radius: 8px; height: 12px; width: 100%;">
                <div style="background: {color}; border-radius: 8px; height: 12px; 
                     width: {score}%; transition: width 0.5s;"></div>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 4px; 
                    font-size: 11px; color: #94a3b8;">
            <span>SAFE</span><span>SUSPICIOUS</span><span>FRAUD</span>
        </div>
    </div>
    """


# ── Sidebar Navigation ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <div style="font-size: 36px;">🛡️</div>
        <div style="font-size: 20px; font-weight: 800; color: white; margin-top: 8px;">
            FraudShield AI
        </div>
        <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">
            Enterprise Fraud Intelligence
        </div>
    </div>
    <hr style="border-color: #334155; margin: 0 0 20px 0;">
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        options=[
            "📊 Overview",
            "🔬 Data Analysis",
            "🤖 Model Performance",
            "🔍 Explainability",
            "⚡ Real-Time Simulator",
            "🚨 Alert Dashboard"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Threshold slider (global)
    st.markdown("**Decision Threshold**")
    threshold = st.slider(
        "Fraud Threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Lower = more sensitive (higher recall, more false positives)",
        label_visibility="collapsed"
    )
    st.caption(f"Threshold: {threshold:.2f} | Score cutoff: {threshold*100:.0f}")
    
    if threshold < 0.3:
        st.warning("⚠️ Very sensitive — expect many false positives")
    elif threshold > 0.7:
        st.warning("⚠️ Conservative — may miss real fraud")
    
    st.markdown("---")
    st.markdown("""
    <div style="color: #64748b; font-size: 11px;">
        <b>Tech Stack</b><br>
        Python • scikit-learn<br>
        Pandas • NumPy<br>
        Matplotlib • Seaborn<br>
        Streamlit • FastAPI
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.title("🛡️ FraudShield AI — Fraud Detection Platform")
    st.markdown("**Enterprise-Grade AI-Powered Transaction Risk Intelligence**")
    st.markdown("---")
    
    stats = load_dataset_stats()
    metrics = load_model_metrics()
    
    if stats:
        # KPI Row 1
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📦 Total Transactions", f"{stats['total_transactions']:,}")
        with col2:
            st.metric("🚨 Fraud Cases", f"{stats['fraud_count']:,}",
                      delta=f"{stats['fraud_rate_pct']:.3f}% fraud rate",
                      delta_color="inverse")
        with col3:
            st.metric("✅ Legitimate", f"{stats['legitimate_count']:,}")
        with col4:
            st.metric("⚖️ Imbalance Ratio", f"{stats['imbalance_ratio']:.0f}:1",
                      help="Legitimate:Fraud ratio")
        
        st.markdown("---")
        
        # KPI Row 2
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Avg Transaction", f"${stats['amount_stats']['mean']:.2f}")
        with col2:
            st.metric("💸 Max Transaction", f"${stats['amount_stats']['max']:,.2f}")
        with col3:
            st.metric("⏱️ Time Span", f"{stats['time_range_hours']:.0f}h")
        with col4:
            st.metric("💾 Dataset Size", f"{stats['memory_mb']:.1f} MB")
    else:
        st.info("📋 Run `python train_pipeline.py` to populate statistics.")
    
    if metrics:
        st.markdown("---")
        st.markdown("### 🏆 Best Model Performance")
        
        # Find best by recall
        best = max(metrics, key=lambda m: m['recall'])
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🏅 Best Model", best['model_name'])
        with col2:
            st.metric("🎯 Precision", f"{best['precision']:.4f}")
        with col3:
            st.metric("🔍 Recall", f"{best['recall']:.4f}",
                      help="Highest priority metric — catching fraud")
        with col4:
            st.metric("⚡ F1 Score", f"{best['f1_score']:.4f}")
        with col5:
            st.metric("📈 ROC-AUC", f"{best['roc_auc']:.4f}")
        
        st.markdown("---")
        st.markdown("### 📊 Model Comparison")
        
        # Display metrics table
        metrics_df = pd.DataFrame(metrics)[['model_name', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']]
        metrics_df.columns = ['Model', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'PR-AUC']
        
        styled = metrics_df.style.highlight_max(
            subset=['Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'PR-AUC'],
            color='#dcfce7'
        ).format({c: '{:.4f}' for c in ['Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'PR-AUC']})
        
        st.dataframe(styled, use_container_width=True)
    
    # Architecture diagram description
    st.markdown("---")
    st.markdown("### 🏗️ System Architecture")
    
    arch_col1, arch_col2 = st.columns([2, 1])
    with arch_col1:
        st.markdown("""
        ```
        ┌─────────────────────────────────────────────────────────┐
        │                  FRAUDSHIELD AI PLATFORM                 │
        ├──────────────┬──────────────┬──────────────┬────────────┤
        │  DATA LAYER  │  ML LAYER    │  API LAYER   │  UI LAYER  │
        │              │              │              │            │
        │ • CSV Ingest │ • Log. Reg.  │ • FastAPI    │ • Streamlit│
        │ • Chunked    │ • Random     │ • /predict   │ • Dashboard│
        │   Loading    │   Forest     │ • /explain   │ • Realtime │
        │ • Memory     │ • Gradient   │ • /metrics   │ • Alerts   │
        │   Optimize   │   Boosting   │ • /threshold │            │
        │ • Feature    │              │              │            │
        │   Engineering│ • SMOTE      │              │            │
        │              │ • Undersamp  │              │            │
        │              │ • Class Wt.  │              │            │
        ├──────────────┴──────────────┴──────────────┴────────────┤
        │  EXPLAINABILITY: Feature Importance + Local Perturbation │
        │  MONITORING: Prediction logging + Performance tracking   │
        └─────────────────────────────────────────────────────────┘
        ```
        """)
    with arch_col2:
        st.markdown("""
        **Pipeline Steps:**
        1. 📥 Ingest & validate CSV
        2. 🔧 Feature engineering  
        3. ⚖️ Handle class imbalance
        4. 🤖 Train 3 models
        5. 📊 Evaluate & compare
        6. 💾 Save with versioning
        7. 🔍 XAI explanations
        8. ⚡ Real-time simulation
        """)


# ══════════════════════════════════════════════════════════════
# PAGE 2: DATA ANALYSIS
# ══════════════════════════════════════════════════════════════

elif page == "🔬 Data Analysis":
    st.markdown('<div class="section-header">🔬 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Class Distribution", "💰 Amount Analysis", "⏱️ Time Analysis"])
    
    with tab1:
        st.markdown("#### Transaction Class Distribution")
        st.markdown("The dataset is **highly imbalanced** — a key challenge in fraud detection.")
        show_figure('class_distribution')
        
        stats = load_dataset_stats()
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"🏦 **{stats['legitimate_count']:,}** legitimate transactions ({100 - stats['fraud_rate_pct']:.3f}%)")
            with col2:
                st.error(f"🚨 **{stats['fraud_count']:,}** fraudulent transactions ({stats['fraud_rate_pct']:.3f}%)")
    
    with tab2:
        st.markdown("#### Transaction Amount Distribution by Class")
        show_figure('amount_distribution')
        st.markdown("""
        **Key Observations:**
        - Fraud transactions tend to have a different amount distribution than legitimate ones
        - Most transactions are small-value; fraud often targets specific amount ranges
        - The `Amount` feature is RobustScaler-normalized before modeling to handle outliers
        """)
    
    with tab3:
        st.markdown("#### Temporal Transaction Patterns")
        show_figure('time_trends')
        st.markdown("""
        **Key Observations:**
        - Transaction volume varies significantly by time of day
        - Some hours show higher fraud rates — captured via cyclical `hour_sin/cos` encoding
        - The dataset covers ~48 hours of transaction data
        """)


# ══════════════════════════════════════════════════════════════
# PAGE 3: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════

elif page == "🤖 Model Performance":
    st.markdown('<div class="section-header">🤖 Model Performance & Comparison</div>', unsafe_allow_html=True)
    
    metrics = load_model_metrics()
    
    if not metrics:
        st.warning("⚠️ No metrics found. Run `python train_pipeline.py` first.")
        st.stop()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Comparison", "📈 ROC Curves", "📉 PR Curves",
        "🎯 Confusion Matrices", "⚙️ Threshold Analysis"
    ])
    
    with tab1:
        st.markdown("#### Model Performance Comparison")
        show_figure('model_comparison', "Higher is better for all metrics")
        
        st.markdown("#### 📋 Detailed Metrics Table")
        metrics_df = pd.DataFrame(metrics)
        display_cols = ['model_name', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc', 'tp', 'fp', 'fn', 'tn']
        metrics_df = metrics_df[[c for c in display_cols if c in metrics_df.columns]]
        metrics_df.columns = [c.replace('_', ' ').title() for c in metrics_df.columns]
        st.dataframe(metrics_df.set_index('Model Name'), use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 💡 Model Strategy")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **Logistic Regression (Baseline)**
            - Fast training
            - Interpretable coefficients
            - Good calibration
            - Struggles with non-linear patterns
            """)
        with col2:
            st.markdown("""
            **Random Forest**
            - Robust to outliers
            - Natural feature importance
            - Good with imbalanced data
            - Ensemble reduces overfitting
            """)
        with col3:
            st.markdown("""
            **Gradient Boosting ⭐**
            - Sequential error correction
            - Best F1 / Recall typically
            - Handles class imbalance well
            - Production recommended model
            """)
    
    with tab2:
        show_figure('roc_curves', "Area Under Curve (AUC) closer to 1.0 is better")
        st.markdown("The ROC curve shows the tradeoff between **True Positive Rate** (fraud caught) "
                    "and **False Positive Rate** (false alarms) across thresholds.")
    
    with tab3:
        show_figure('pr_curves', "PR-AUC is more informative for imbalanced datasets")
        st.markdown("For highly imbalanced datasets, the **Precision-Recall curve** is more "
                    "informative than ROC. High PR-AUC means the model is good at identifying "
                    "fraud without too many false alarms.")
    
    with tab4:
        model_names = [m['model_name'] for m in metrics]
        selected = st.selectbox("Select Model", model_names)
        safe_name = selected.lower().replace(' ', '_')
        show_figure(f'confusion_matrix_{safe_name}')
        
        selected_metrics = next(m for m in metrics if m['model_name'] == selected)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("✅ True Positives (Fraud Caught)", selected_metrics['tp'])
        with col2:
            st.metric("❌ False Negatives (Fraud Missed)", selected_metrics['fn'],
                      delta_color="inverse")
        with col3:
            st.metric("⚠️ False Positives (False Alarms)", selected_metrics['fp'],
                      delta_color="inverse")
        with col4:
            st.metric("✅ True Negatives (Legit Correct)", selected_metrics['tn'])
    
    with tab5:
        model_names = [m['model_name'] for m in metrics]
        selected = st.selectbox("Select Model for Threshold Analysis", model_names, key='thresh_sel')
        safe_name = selected.lower().replace(' ', '_')
        show_figure(f'threshold_analysis_{safe_name}')
        st.markdown("""
        **How to choose threshold:**
        - **Maximize Recall** → Use lower threshold (catch all fraud, accept more false positives)
        - **Maximize Precision** → Use higher threshold (fewer false alarms, may miss some fraud)
        - **Maximize F1** → Use the optimal point shown on the chart
        """)


# ══════════════════════════════════════════════════════════════
# PAGE 4: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════

elif page == "🔍 Explainability":
    st.markdown('<div class="section-header">🔍 Explainable AI — Why is this Fraud?</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🌐 Global Feature Importance", "🔎 Transaction Explanation"])
    
    with tab1:
        st.markdown("#### Top Features Driving Fraud Predictions (Global)")
        show_figure('feature_importance_gradient_boosting',
                    "Feature importance from Gradient Boosting model")
        
        imp_df = load_feature_importance()
        if imp_df is not None:
            st.markdown("#### 📋 Feature Importance Table (Top 20)")
            st.dataframe(
                imp_df.head(20)[['rank', 'feature', 'importance_pct']].set_index('rank'),
                use_container_width=True
            )
        
        st.markdown("""
        #### 📖 Feature Descriptions
        | Feature | Description |
        |---------|-------------|
        | V1–V28 | PCA-transformed behavioral features (anonymized for privacy) |
        | Amount_scaled | Normalized transaction amount (RobustScaler) |
        | V14 | Strong negative fraud signal (high importance) |
        | V17 | Strong negative fraud signal |
        | V14_V17_interaction | Interaction term: amplifies combined signal |
        | tx_velocity | Rolling transaction count (velocity fraud detection) |
        | amount_deviation | Deviation from rolling average (anomaly indicator) |
        | is_high_amount | Flag for top 5% high-value transactions |
        """)
    
    with tab2:
        st.markdown("#### Explain a Single Transaction")
        st.markdown("Enter transaction values to see why it would be classified as fraud or legitimate.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            amount = st.number_input("💰 Amount ($)", min_value=0.0, value=100.0, step=10.0)
            time_val = st.number_input("⏱️ Time (seconds)", min_value=0, value=3600, step=100)
            
            st.markdown("**Key V-Features** (others default to 0)")
            v14 = st.slider("V14 (strong fraud signal)", -20.0, 10.0, 0.0, 0.1)
            v17 = st.slider("V17 (fraud signal)", -20.0, 10.0, 0.0, 0.1)
            v4 = st.slider("V4", -10.0, 10.0, 0.0, 0.1)
            v11 = st.slider("V11", -10.0, 10.0, 0.0, 0.1)
            
            preset = st.selectbox("Or load preset scenario", [
                "Custom", "Normal Purchase", "High-Value Night", "Card Testing", "Suspicious Pattern"
            ])
            
            presets = {
                "Normal Purchase": {'amount': 45.0, 'v14': 0.5, 'v17': 0.2, 'v4': 0.1, 'v11': -0.3},
                "High-Value Night": {'amount': 5000.0, 'v14': -12.0, 'v17': -8.0, 'v4': -3.0, 'v11': 2.0},
                "Card Testing": {'amount': 0.99, 'v14': -8.0, 'v17': -5.0, 'v4': 3.0, 'v11': 1.0},
                "Suspicious Pattern": {'amount': 299.0, 'v14': -4.0, 'v17': -3.0, 'v4': 0.5, 'v11': -1.0}
            }
            
            if preset != "Custom" and preset in presets:
                p = presets[preset]
                amount = p['amount']
                v14 = p['v14']
                v17 = p['v17']
                v4 = p['v4']
                v11 = p['v11']
        
        with col2:
            if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
                try:
                    # Build feature vector
                    import pandas as pd
                    from src.feature_engineering import engineer_features, get_feature_columns
                    from src.explainability import explain_transaction_local, prob_to_score, score_to_risk
                    from src.modeling import load_model
                    
                    raw_df = pd.DataFrame([{
                        'Time': float(time_val), 'Amount': float(amount),
                        'V1': 0, 'V2': 0, 'V3': 0, 'V4': float(v4), 'V5': 0,
                        'V6': 0, 'V7': 0, 'V8': 0, 'V9': 0, 'V10': 0,
                        'V11': float(v11), 'V12': 0, 'V13': 0, 'V14': float(v14), 'V15': 0,
                        'V16': 0, 'V17': float(v17), 'V18': 0, 'V19': 0, 'V20': 0,
                        'V21': 0, 'V22': 0, 'V23': 0, 'V24': 0, 'V25': 0,
                        'V26': 0, 'V27': 0, 'V28': 0, 'Class': 0
                    }])
                    
                    df_eng = engineer_features(raw_df)
                    feat_cols = get_feature_columns()
                    for col in feat_cols:
                        if col not in df_eng.columns:
                            df_eng[col] = 0.0
                    features = df_eng[feat_cols].values[0].astype(np.float32)
                    
                    model = load_model('gradient_boosting')
                    explanation = explain_transaction_local(model, features, feat_cols, n_perturbations=300)
                    
                    st.markdown(make_gauge_html(explanation['fraud_score'], explanation['risk_level']),
                                unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("**Top Contributing Features:**")
                    
                    for feat in explanation['top_features'][:7]:
                        direction = "↑ Increases Risk" if feat['contribution'] > 0 else "↓ Decreases Risk"
                        color = "#ef4444" if feat['contribution'] > 0 else "#22c55e"
                        bar_width = min(abs(feat['contribution']) * 200, 100)
                        
                        st.markdown(f"""
                        <div style="margin: 6px 0; padding: 8px 12px; background: white; 
                                    border-radius: 8px; border-left: 3px solid {color};">
                            <b>{feat['feature']}</b> 
                            <span style="color: {color}; float: right;">{direction}</span>
                            <div style="color: #64748b; font-size: 12px;">
                                value = {feat['value']:.4f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.info(explanation['explanation_text'])
                    
                except FileNotFoundError:
                    st.error("⚠️ Model not found. Run `python train_pipeline.py` first.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.markdown("""
                <div style="text-align: center; padding: 40px; color: #94a3b8;">
                    <div style="font-size: 48px;">🔍</div>
                    <p>Configure transaction values on the left,<br>then click Analyze Transaction</p>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 5: REAL-TIME SIMULATOR
# ══════════════════════════════════════════════════════════════

elif page == "⚡ Real-Time Simulator":
    st.markdown('<div class="section-header">⚡ Real-Time Transaction Simulation</div>', unsafe_allow_html=True)
    
    st.markdown("Simulate a live fraud detection pipeline processing transactions in real-time.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        n_tx = st.slider("Transactions to simulate", 10, 500, 100)
    with col2:
        delay_ms = st.slider("Processing delay (ms)", 0, 200, 50)
    with col3:
        use_fraud_heavy = st.toggle("Include synthetic fraud scenarios", value=True)
    
    if st.button("▶️ Start Simulation", type="primary", use_container_width=True):
        try:
            from src.modeling import load_model
            from src.feature_engineering import engineer_features, get_feature_columns
            from src.realtime_engine import RealTimeSimulator, generate_synthetic_fraud_scenarios
            from src.explainability import score_to_risk, prob_to_score
            
            model = load_model('gradient_boosting')
            feat_cols = get_feature_columns()
            sim = RealTimeSimulator(model, feat_cols, threshold=threshold)
            
            # Generate test transactions
            if use_fraud_heavy:
                synth_df = generate_synthetic_fraud_scenarios(
                    n_legitimate=int(n_tx * 0.7),
                    n_fraud=int(n_tx * 0.3)
                )
                # Add Time column if missing
                if 'Time' not in synth_df.columns:
                    synth_df['Time'] = np.linspace(0, 3600*48, len(synth_df))
                if 'Class' not in synth_df.columns:
                    synth_df['Class'] = (synth_df['type'] == 'FRAUD').astype(int)
                    
                df_eng = engineer_features(synth_df[['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)] + ['Class']])
            else:
                # Load a sample from the real dataset
                import pandas as pd
                df_sample = pd.read_csv('/mnt/user-data/uploads/creditcard.csv', nrows=n_tx*2)
                df_eng = engineer_features(df_sample)
            
            for col in feat_cols:
                if col not in df_eng.columns:
                    df_eng[col] = 0.0
            
            # Setup display
            progress_bar = st.progress(0)
            stats_placeholder = st.empty()
            alert_feed = st.empty()
            
            recent_alerts = []
            stats_tracker = {'SAFE': 0, 'SUSPICIOUS': 0, 'FRAUD': 0, 'total': 0}
            
            for i, result in enumerate(sim.stream_transactions(df_eng, delay_ms=delay_ms, max_transactions=n_tx)):
                # Update progress
                progress_bar.progress((i + 1) / n_tx)
                
                risk = result['risk_level']
                stats_tracker[risk] = stats_tracker.get(risk, 0) + 1
                stats_tracker['total'] += 1
                
                # Add to recent feed
                color = {'SAFE': '🟢', 'SUSPICIOUS': '🟡', 'FRAUD': '🔴'}[risk]
                recent_alerts.insert(0, {
                    'icon': color,
                    'tx_id': result['tx_id'],
                    'score': result['fraud_score'],
                    'risk': risk,
                    'prob': result['fraud_probability']
                })
                recent_alerts = recent_alerts[:15]
                
                # Update stats display
                with stats_placeholder.container():
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("🔄 Processed", stats_tracker['total'])
                    c2.metric("✅ Safe", stats_tracker.get('SAFE', 0))
                    c3.metric("⚠️ Suspicious", stats_tracker.get('SUSPICIOUS', 0))
                    c4.metric("🚨 Fraud", stats_tracker.get('FRAUD', 0))
                
                # Update alert feed
                with alert_feed.container():
                    st.markdown("**📡 Live Alert Feed:**")
                    for alert in recent_alerts:
                        st.markdown(
                            f"{alert['icon']} `{alert['tx_id']}` — Score: **{alert['score']:.1f}** | "
                            f"Risk: **{alert['risk']}** | Prob: {alert['prob']:.3f}"
                        )
            
            progress_bar.progress(1.0)
            
            # Final summary
            st.success(f"✅ Simulation complete! Processed {n_tx} transactions.")
            
            fraud_rate = stats_tracker['FRAUD'] / max(stats_tracker['total'], 1) * 100
            col1, col2, col3 = st.columns(3)
            col1.metric("Detection Rate", f"{fraud_rate:.1f}%")
            col2.metric("Threshold Used", f"{threshold:.2f}")
            col3.metric("Avg Latency", f"{delay_ms}ms")
            
        except FileNotFoundError:
            st.error("⚠️ Model not found. Run `python train_pipeline.py` first.")
        except Exception as e:
            st.error(f"Simulation error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════
# PAGE 6: ALERT DASHBOARD
# ══════════════════════════════════════════════════════════════

elif page == "🚨 Alert Dashboard":
    st.markdown('<div class="section-header">🚨 Alert Dashboard — Live Fraud Monitor</div>', unsafe_allow_html=True)
    
    # Check for prediction logs
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    log_files = sorted([f for f in os.listdir(logs_dir) if f.endswith('.jsonl')]) if os.path.exists(logs_dir) else []
    
    if log_files:
        latest_log = os.path.join(logs_dir, log_files[-1])
        
        records = []
        with open(latest_log) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except:
                    pass
        
        if records:
            log_df = pd.DataFrame(records)
            
            st.markdown(f"**Showing {len(log_df)} predictions from today's session**")
            
            col1, col2, col3, col4 = st.columns(4)
            counts = log_df['risk_level'].value_counts() if 'risk_level' in log_df.columns else {}
            col1.metric("Total Alerts", len(log_df))
            col2.metric("✅ Safe", counts.get('SAFE', 0))
            col3.metric("⚠️ Suspicious", counts.get('SUSPICIOUS', 0))
            col4.metric("🚨 Fraud", counts.get('FRAUD', 0))
            
            st.markdown("---")
            
            # Filter
            filter_risk = st.multiselect("Filter by Risk Level", ['SAFE', 'SUSPICIOUS', 'FRAUD'],
                                         default=['SUSPICIOUS', 'FRAUD'])
            
            if 'risk_level' in log_df.columns and filter_risk:
                filtered = log_df[log_df['risk_level'].isin(filter_risk)]
            else:
                filtered = log_df
            
            if len(filtered) > 0:
                display_cols = ['tx_id', 'timestamp', 'fraud_score', 'fraud_probability', 'risk_level', 'decision']
                available_cols = [c for c in display_cols if c in filtered.columns]
                
                st.dataframe(
                    filtered[available_cols].sort_values('fraud_score', ascending=False).head(100),
                    use_container_width=True
                )
        else:
            st.info("No prediction logs found. Run the Real-Time Simulator first.")
    else:
        st.info("🔄 No logs yet. Run the **Real-Time Simulator** to generate predictions.")
        st.markdown("""
        The alert dashboard shows:
        - All predictions from the current session
        - Risk levels (SAFE / SUSPICIOUS / FRAUD)
        - Fraud scores (0–100)
        - Recommended actions
        - Timestamps for audit trail
        """)
    
    st.markdown("---")
    st.markdown("### 📊 Score Distribution (Static Demo)")
    
    # Show score distribution from model evaluation
    show_figure('score_distribution_gradient_boosting')
    
    st.markdown("""
    **Risk Zones:**
    - 🟢 **0–30 (SAFE)**: Normal transactions, approve automatically
    - 🟡 **30–70 (SUSPICIOUS)**: Unusual patterns, flag for manual review
    - 🔴 **70–100 (FRAUD)**: High confidence fraud, block immediately
    """)
