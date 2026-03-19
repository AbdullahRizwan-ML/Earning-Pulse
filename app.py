"""
EarningsPulse — S&P 500 Earnings Intelligence Dashboard.

A Streamlit-powered UI that provides earnings surprise predictions,
an upcoming earnings watchlist, model performance metrics, and
methodology documentation.

Run with: streamlit run app.py
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EarningsPulse — S&P 500 Earnings Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    .stApp {
        background-color: #0e1117;
    }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #1f4068;
    }
    .main-header h1 {
        color: #e0e0e0;
        margin: 0;
        font-size: 2rem;
    }
    .main-header p {
        color: #8892b0;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid #1f4068;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
    }
    .metric-card h3 {
        color: #8892b0;
        font-size: 0.85rem;
        margin: 0 0 0.3rem 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        color: #e0e0e0;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .beat-badge {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: #fff;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .miss-badge {
        background: linear-gradient(135deg, #e17055, #d63031);
        color: #fff;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8892b0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #64ffda;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📈 EarningsPulse</h1>
    <p>S&P 500 Earnings Intelligence — ML-powered beat/miss probability predictions</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def load_model_artifacts() -> Dict[str, Any]:
    """Load model metrics and metadata from disk (cached for 1 hour).

    Returns:
        Dictionary with model metrics, feature names, and training metadata.
    """
    data_dir = Path(__file__).parent / "data"
    result: Dict[str, Any] = {
        "metrics": {},
        "feature_names": [],
        "training_date": "N/A",
        "models_available": False,
    }
    try:
        metrics_path = data_dir / "model_metrics.json"
        meta_path = data_dir / "model_metadata.json"

        if metrics_path.exists():
            with open(metrics_path) as f:
                result["metrics"] = json.load(f)
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                result["feature_names"] = meta.get("feature_names", [])
                result["training_date"] = meta.get("training_date", "N/A")
        result["models_available"] = (data_dir / "xgb_model.joblib").exists()
    except Exception:
        pass
    return result


@st.cache_data(ttl=3600)
def load_feature_importance() -> Optional[pd.DataFrame]:
    """Load feature importance CSV (cached for 1 hour).

    Returns:
        DataFrame of feature importances, or ``None`` if not available.
    """
    csv_path = Path(__file__).parent / "data" / "feature_importance.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict a Stock",
    "📋 Upcoming Earnings Watchlist",
    "📊 Model Performance",
    "📖 Methodology",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Predict a Stock
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_input, col_spacer, col_result = st.columns([2, 0.5, 4])

    with col_input:
        st.markdown("### Enter Ticker Symbol")
        ticker_input = st.text_input(
            "Ticker",
            value="AAPL",
            placeholder="e.g., AAPL, MSFT, GOOGL",
            label_visibility="collapsed",
        )
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown("#### Quick Picks")
        quick_cols = st.columns(4)
        quick_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "JNJ"]
        for i, qt in enumerate(quick_tickers):
            if quick_cols[i % 4].button(qt, key=f"quick_{qt}", use_container_width=True):
                ticker_input = qt
                analyze_btn = True

    with col_result:
        if analyze_btn and ticker_input:
            ticker = ticker_input.upper().strip()

            artifacts = load_model_artifacts()
            if not artifacts["models_available"]:
                st.warning(
                    "⚠️ Models not trained yet. Run `python -m src.model` to train, "
                    "or the app will display demo results.",
                )

            with st.spinner(f"🔄 Fetching live data for {ticker}..."):
                try:
                    from src.predict import predict_single
                    result = predict_single(ticker)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    result = None

            if result and not result.get("error"):
                prob = result["beat_probability"]
                pred = result["prediction"]
                conf = result["confidence"]

                # ── Gauge Chart ──
                gauge_color = "#00b894" if prob >= 0.5 else "#e17055"
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    title={"text": f"<b>{ticker}</b> Beat Probability", "font": {"size": 20, "color": "#e0e0e0"}},
                    number={"suffix": "%", "font": {"size": 48, "color": gauge_color}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#8892b0"},
                        "bar": {"color": gauge_color},
                        "bgcolor": "#1a1a2e",
                        "bordercolor": "#1f4068",
                        "steps": [
                            {"range": [0, 45], "color": "rgba(225, 112, 85, 0.2)"},
                            {"range": [45, 65], "color": "rgba(253, 203, 110, 0.2)"},
                            {"range": [65, 100], "color": "rgba(0, 184, 148, 0.2)"},
                        ],
                        "threshold": {
                            "line": {"color": "#e0e0e0", "width": 2},
                            "thickness": 0.8,
                            "value": 50,
                        },
                    },
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    margin=dict(l=30, r=30, t=60, b=20),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # ── Key Stats Row ──
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    badge = "beat-badge" if pred == "BEAT" else "miss-badge"
                    st.markdown(f'<div class="metric-card"><h3>Prediction</h3>'
                                f'<span class="{badge}">{pred}</span></div>',
                                unsafe_allow_html=True)
                with stat_cols[1]:
                    st.markdown(f'<div class="metric-card"><h3>Confidence</h3>'
                                f'<p class="value">{conf}</p></div>',
                                unsafe_allow_html=True)
                with stat_cols[2]:
                    streak = result["features"].get("eps_beat_streak", "N/A")
                    st.markdown(f'<div class="metric-card"><h3>Beat Streak</h3>'
                                f'<p class="value">{int(streak) if streak != "N/A" else "N/A"}Q</p></div>',
                                unsafe_allow_html=True)
                with stat_cols[3]:
                    sector = result["info"].get("sector", "N/A")
                    st.markdown(f'<div class="metric-card"><h3>Sector</h3>'
                                f'<p class="value" style="font-size:1.1rem">{sector}</p></div>',
                                unsafe_allow_html=True)

                # ── Recent Earnings Timeline ──
                st.markdown("#### 📅 Recent Earnings History")
                earnings_hist = result.get("earnings_history", pd.DataFrame())
                if isinstance(earnings_hist, pd.DataFrame) and not earnings_hist.empty:
                    recent = earnings_hist.sort_values("earnings_date", ascending=False).head(4)
                    timeline_cols = st.columns(len(recent))
                    for i, (_, row) in enumerate(recent.iterrows()):
                        with timeline_cols[i]:
                            beat = row.get("beat", 0)
                            color = "🟢" if beat == 1 else "🔴"
                            date_str = pd.to_datetime(row["earnings_date"]).strftime("%b %Y")
                            actual = row.get("actual_eps", "?")
                            est = row.get("estimated_eps", "?")
                            st.markdown(
                                f"**{color} {date_str}**\n\n"
                                f"Actual: ${actual:.2f}\n\n"
                                f"Est: ${est:.2f}"
                            )
                else:
                    st.info("No earnings history available for this ticker.")

                # ── Feature Breakdown Table ──
                st.markdown("#### 🔬 Feature Breakdown")
                features = result.get("features", {})
                impacts = result.get("feature_impacts", {})
                if features:
                    from src.feature_engineering import FEATURE_REGISTRY
                    breakdown_data = []
                    for fname, fval in features.items():
                        desc = FEATURE_REGISTRY.get(fname, fname)
                        impact = impacts.get(fname, "→ Neutral")
                        breakdown_data.append({
                            "Feature": fname,
                            "Description": desc,
                            "Value": f"{fval:.4f}" if isinstance(fval, float) else str(fval),
                            "Impact": impact,
                        })
                    st.dataframe(
                        pd.DataFrame(breakdown_data),
                        use_container_width=True,
                        hide_index=True,
                    )

            elif result and result.get("error"):
                st.error(f"❌ {result['error']}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Upcoming Earnings Watchlist
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📋 Upcoming Earnings — Next 30 Days")
    st.markdown("*Stocks from our S&P 500 universe with upcoming earnings announcements.*")

    load_watchlist = st.button("🔄 Load / Refresh Watchlist", type="primary")

    if load_watchlist:
        with st.spinner("🔄 Scanning for upcoming earnings and generating predictions..."):
            try:
                from src.data_ingestion import get_upcoming_earnings
                from src.predict import predict_watchlist

                upcoming = get_upcoming_earnings(days_ahead=30)

                if upcoming.empty:
                    st.info(
                        "No upcoming earnings found in the next 30 days for our "
                        "tracked universe. Try again closer to earnings season."
                    )
                else:
                    # Generate predictions
                    artifacts = load_model_artifacts()
                    if artifacts["models_available"]:
                        watchlist = predict_watchlist(upcoming)
                    else:
                        # Demo mode without models
                        watchlist = upcoming.copy()
                        np.random.seed(42)
                        watchlist["beat_probability"] = np.random.uniform(0.3, 0.85, len(watchlist))
                        watchlist["confidence"] = watchlist["beat_probability"].apply(
                            lambda p: "High" if abs(p - 0.5) > 0.25 else
                                      "Medium" if abs(p - 0.5) > 0.10 else "Low"
                        )
                        watchlist["prediction"] = watchlist["beat_probability"].apply(
                            lambda p: "BEAT" if p >= 0.5 else "MISS"
                        )
                        st.warning("⚠️ Showing demo predictions (models not trained).")

                    # Format for display
                    display_df = watchlist.copy()
                    if "earnings_date" in display_df.columns:
                        display_df["earnings_date"] = pd.to_datetime(display_df["earnings_date"]).dt.strftime("%Y-%m-%d")
                    display_df["beat_probability"] = (display_df["beat_probability"] * 100).round(1)

                    # Color coding function
                    def color_probability(val):
                        try:
                            v = float(val)
                            if v >= 65:
                                return "background-color: rgba(0, 184, 148, 0.3); color: #00b894;"
                            elif v >= 45:
                                return "background-color: rgba(253, 203, 110, 0.3); color: #fdcb6e;"
                            else:
                                return "background-color: rgba(225, 112, 85, 0.3); color: #e17055;"
                        except (ValueError, TypeError):
                            return ""

                    # Rename columns for display
                    col_map = {
                        "ticker": "Ticker",
                        "company_name": "Company",
                        "earnings_date": "Earnings Date",
                        "beat_probability": "Beat Prob (%)",
                        "confidence": "Confidence",
                        "sector": "Sector",
                        "prediction": "Prediction",
                    }
                    display_cols = [c for c in col_map.keys() if c in display_df.columns]
                    display_df = display_df[display_cols].rename(columns=col_map)

                    styled = display_df.style.applymap(
                        color_probability,
                        subset=["Beat Prob (%)"] if "Beat Prob (%)" in display_df.columns else [],
                    )
                    st.dataframe(styled, use_container_width=True, hide_index=True)

                    # Download button
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Watchlist CSV",
                        csv,
                        file_name=f"earnings_watchlist_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                    )

            except Exception as exc:
                st.error(f"Failed to load watchlist: {exc}")
    else:
        st.info("Click the button above to load the upcoming earnings watchlist.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Model Performance Dashboard")

    artifacts = load_model_artifacts()
    metrics = artifacts.get("metrics", {})

    if not metrics:
        st.warning(
            "⚠️ No model metrics available. Run `python -m src.model` to "
            "train the models and generate performance metrics."
        )
        st.stop()

    # ── Metric Cards ──
    st.markdown("#### Key Metrics")
    m_cols = st.columns(6)
    metric_items = [
        ("ROC-AUC", metrics.get("roc_auc", "—"), ""),
        ("Accuracy", metrics.get("accuracy", "—"), ""),
        ("Brier Score", metrics.get("brier_score", "—"), "lower is better"),
        ("F1 (Beat)", metrics.get("f1_beat", "—"), ""),
        ("F1 (Miss)", metrics.get("f1_miss", "—"), ""),
        ("Avg CV AUC", metrics.get("avg_cv_auc", "—"), ""),
    ]
    for i, (label, value, help_text) in enumerate(metric_items):
        with m_cols[i]:
            display_val = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
            st.metric(label, display_val, help=help_text if help_text else None)

    st.markdown("---")

    # ── Two-column layout for charts ──
    chart_left, chart_right = st.columns(2)

    # ── ROC Curve ──
    with chart_left:
        st.markdown("#### ROC Curve")
        roc_data = metrics.get("roc_curve", {})
        fpr = roc_data.get("fpr", [])
        tpr = roc_data.get("tpr", [])

        if fpr and tpr:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"Ensemble (AUC = {metrics.get('roc_auc', 0):.3f})",
                line=dict(color="#64ffda", width=2.5),
                fill="tonexty",
                fillcolor="rgba(100, 255, 218, 0.1)",
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Random Baseline",
                line=dict(color="#8892b0", width=1, dash="dash"),
            ))
            fig_roc.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(26,26,46,0.8)",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=400,
                margin=dict(l=40, r=20, t=20, b=40),
                legend=dict(x=0.5, y=0.05),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("ROC curve data not available.")

    # ── Confusion Matrix ──
    with chart_right:
        st.markdown("#### Confusion Matrix")
        cm = metrics.get("confusion_matrix", [])

        if cm:
            cm_array = np.array(cm)
            labels = ["Miss (0)", "Beat (1)"]
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm_array,
                x=labels,
                y=labels,
                text=cm_array.astype(str),
                texttemplate="%{text}",
                textfont={"size": 20, "color": "white"},
                colorscale=[[0, "#16213e"], [1, "#0f3460"]],
                showscale=False,
            ))
            fig_cm.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(26,26,46,0.8)",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=400,
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Confusion matrix data not available.")

    # ── Feature Importance ──
    st.markdown("#### Feature Importance")
    imp_df = load_feature_importance()
    if imp_df is not None and not imp_df.empty:
        imp_sorted = imp_df.sort_values("importance", ascending=True)
        fig_imp = go.Figure(go.Bar(
            x=imp_sorted["importance"],
            y=imp_sorted["feature"],
            orientation="h",
            marker=dict(
                color=imp_sorted["importance"],
                colorscale="Viridis",
                line=dict(width=0),
            ),
        ))
        fig_imp.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,26,46,0.8)",
            xaxis_title="Average Importance (XGBoost + LightGBM)",
            height=max(400, len(imp_sorted) * 25),
            margin=dict(l=150, r=20, t=10, b=40),
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance data not available. Train the model first.")

    # ── Cross-Validation Folds ──
    fold_metrics = metrics.get("fold_metrics", [])
    if fold_metrics:
        st.markdown("#### Cross-Validation Results (TimeSeriesSplit)")
        folds_df = pd.DataFrame(fold_metrics)
        folds_df["fold"] = folds_df["fold"].astype(str).apply(lambda x: f"Fold {x}")
        st.dataframe(folds_df, use_container_width=True, hide_index=True)

    # ── Training Summary ──
    st.markdown("#### Training Summary")
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric("Training Samples", metrics.get("n_train_samples", "—"))
    with summary_cols[1]:
        st.metric("Features Used", metrics.get("n_features", "—"))
    with summary_cols[2]:
        st.metric("Training Date", artifacts.get("training_date", "—")[:10])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Methodology
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    ## 📖 Methodology

    ### Overview
    EarningsPulse uses a **dual-model ensemble** (XGBoost + LightGBM) trained on 
    3 years of quarterly earnings data from 50 S&P 500 companies representing 
    all 11 GICS sectors.

    The system predicts whether a company will **beat** or **miss** Wall Street's
    EPS (Earnings Per Share) consensus estimate before the earnings announcement.

    ---

    ### Data Sources
    | Source | Data | Cost |
    |--------|------|------|
    | **yfinance** | Price history, earnings actuals/estimates, fundamentals, insider data | Free |
    | **Alpha Vantage** | Fallback earnings estimates, company overview (25 calls/day) | Free tier |
    | **SEC EDGAR** | 8-K filing detection around earnings dates | Free |
    | **Yahoo Finance** | VIX, treasury yields, sector ETF prices | Free (via yfinance) |

    ---

    ### Feature Engineering (20 Features)
    
    **Group A — Analyst Estimates:**
    - EPS surprise in the prior quarter
    - Consecutive beat streak (0–8 quarters)
    - Analyst revision direction (upward vs. downward)
    - Estimate dispersion (spread of analyst opinions)

    **Group B — Price Momentum (Look-Ahead Safe):**
    - 1-week, 1-month, and 3-month returns *prior* to earnings
    - Price relative to 52-week high
    - Volume surge (short-term vs. long-term average)

    **Group C — Fundamental Quality:**
    - Year-over-year revenue growth
    - Gross margin trend (quarter-over-quarter change)
    - Free cash flow yield (FCF / market cap)
    - Debt-to-equity ratio

    **Group D — Sentiment & Insider Activity:**
    - Insider buy/sell ratio (last 90 days)
    - Institutional ownership level
    - Short interest ratio
    - Recent 8-K filing indicator

    **Group E — Macro Context:**
    - Yield curve spread (10Y minus 2Y treasury)
    - VIX level (market fear gauge)
    - Sector-specific momentum (SPDR sector ETF returns)

    ---

    ### Why TimeSeriesSplit?
    Financial data is **sequential**: using random train/test splits would allow 
    the model to "cheat" by learning from future data. `TimeSeriesSplit` ensures 
    that each fold only trains on past data and validates on future data, just 
    like real-world prediction.

    ### Why Ensemble?
    XGBoost and LightGBM use different splitting algorithms and regularization 
    strategies. Averaging their probabilities reduces variance and produces 
    better-calibrated predictions than either model alone.

    ### Look-Ahead Bias Prevention
    All price-based features are computed as of **earnings_date minus 2 days**
    to prevent any post-announcement data from leaking into the features.

    ---

    ### Limitations
    - **API Rate Limits**: yfinance is rate-limited; bulk data fetches may take 
      several minutes. Alpha Vantage is limited to 25 calls/day on the free tier.
    - **Training Window**: 3 years of quarterly data (≈12 quarters per stock). 
      This captures recent regimes but may miss longer cycles.
    - **Feature Coverage**: Not all features are available for every stock/quarter 
      (handled via median imputation). Stocks with sparse data may have less 
      reliable predictions.
    - **Market Regime Shifts**: Models trained on recent data may underperform 
      during unprecedented events (COVID, rate shocks, etc.).
    - **Universe Size**: Trained on 50 of 500 S&P stocks. Predictions for 
      out-of-universe stocks are less reliable.

    ---

    ### ⚠️ Disclaimer
    > **This tool is for educational and research purposes only.** It is NOT 
    > financial advice. Do not make investment decisions based solely on these 
    > predictions. Past performance does not guarantee future results. Always 
    > do your own due diligence and consult a qualified financial advisor.
    """)

    st.markdown("---")
    st.markdown(
        f"*EarningsPulse v1.0 | Built with Python, XGBoost, LightGBM, and Streamlit | "
        f"Last updated: {datetime.now().strftime('%B %Y')}*"
    )
