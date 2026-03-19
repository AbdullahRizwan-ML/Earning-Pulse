<p align="center">
  <h1 align="center">📈 EarningsPulse</h1>
  <p align="center">S&P 500 Earnings Surprise Predictor — ML-powered beat/miss probability intelligence</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Built%20with-XGBoost-orange?logo=xgboost" alt="XGBoost">
  <img src="https://img.shields.io/badge/Built%20with-LightGBM-brightgreen" alt="LightGBM">
  <img src="https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Data-yfinance-yellow" alt="yfinance">
</p>

---

## Overview

**EarningsPulse** is a machine learning system that predicts whether a company will **beat** or **miss** Wall Street's EPS consensus estimates before the earnings announcement. It uses 20 engineered features spanning analyst estimates, price momentum, fundamental quality, insider sentiment, and macroeconomic context.

The ensemble model (XGBoost + LightGBM) is trained on 3 years of quarterly earnings from 50 S&P 500 companies across all 11 GICS sectors, with **TimeSeriesSplit** cross-validation to prevent data leakage.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                  │
│  yfinance │ Alpha Vantage (fallback) │ SEC EDGAR │ Yahoo Finance    │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION (src/data_ingestion.py)           │
│  Earnings History │ Price Data │ Fundamentals │ Sentiment │ Macro   │
│  • Retry logic (3x with backoff)                                    │
│  • Parquet caching with TTL                                         │
│  • Rate limiting (0.5s between requests)                            │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING (src/feature_engineering.py)      │
│  20 Features across 5 groups:                                        │
│  A: Analyst Estimates │ B: Price Momentum │ C: Fundamentals          │
│  D: Sentiment/Insider │ E: Macro Context                             │
│  • Median imputation for missing data                                │
│  • Lag features (prior quarter surprise)                             │
│  • FEATURE_REGISTRY for UI descriptions                              │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     MODEL TRAINING (src/model.py)                    │
│  XGBoost   ──┐                                                       │
│              ├── Ensemble (average probabilities)                    │
│  LightGBM  ──┘                                                       │
│  • TimeSeriesSplit(5) cross-validation                               │
│  • Early stopping │ Feature importance │ Model persistence           │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   STREAMLIT DASHBOARD (app.py)                       │
│  Tab 1: Predict a Stock │ Tab 2: Earnings Watchlist                  │
│  Tab 3: Model Performance │ Tab 4: Methodology                      │
│  • Plotly charts │ Dark theme │ CSV export │ Live data               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/yourusername/earnings-predictor.git
cd earnings-predictor
pip install -r requirements.txt
```

### 2. Configure API keys (optional)

```bash
cp .env.example .env
# Edit .env with your Alpha Vantage and SEC API keys
# All keys are optional — the system works with yfinance alone
```

### 3. Train the model

```bash
python -m src.model
```

This fetches 3 years of earnings data for 50 S&P 500 tickers, engineers all 20 features, trains the XGBoost + LightGBM ensemble, and saves the models to `data/`.

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` to access the EarningsPulse dashboard.

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Features (20 total)

| # | Feature | Group | Description | Source |
|---|---------|-------|-------------|--------|
| 1 | `eps_surprise_last_q` | Analyst | EPS surprise magnitude in prior quarter | yfinance |
| 2 | `eps_beat_streak` | Analyst | Consecutive quarters of beating estimates (0–8) | yfinance |
| 3 | `eps_revision_direction` | Analyst | Analyst estimate revision trend | yfinance |
| 4 | `estimate_dispersion` | Analyst | Spread of analyst price targets | yfinance |
| 5 | `ret_1w` | Momentum | 1-week return prior to earnings | yfinance |
| 6 | `ret_1m` | Momentum | 1-month return prior to earnings | yfinance |
| 7 | `ret_3m` | Momentum | 3-month return prior to earnings | yfinance |
| 8 | `price_vs_52w_high` | Momentum | Price as ratio of 52-week high | yfinance |
| 9 | `volume_surge` | Momentum | 5-day / 30-day avg volume ratio | yfinance |
| 10 | `revenue_growth_yoy` | Fundamental | Year-over-year revenue growth | yfinance |
| 11 | `gross_margin_trend` | Fundamental | QoQ gross margin change | yfinance |
| 12 | `fcf_yield` | Fundamental | Free cash flow / market cap | yfinance |
| 13 | `debt_to_equity` | Fundamental | Balance sheet leverage ratio | yfinance |
| 14 | `insider_buy_sell_ratio` | Sentiment | Net insider purchase ratio (90 days) | yfinance |
| 15 | `institutional_ownership_change` | Sentiment | Institutional ownership level | yfinance |
| 16 | `short_interest_ratio` | Sentiment | Days to cover short interest | yfinance |
| 17 | `recent_8k_filing` | Sentiment | 8-K filed in last 7 days | SEC EDGAR |
| 18 | `yield_curve_spread` | Macro | 10Y–2Y treasury spread | yfinance |
| 19 | `vix_level` | Macro | VIX volatility index | yfinance |
| 20 | `sector_momentum` | Macro | SPDR sector ETF 1-month return | yfinance |

---

## Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | *train to see* |
| Accuracy | *train to see* |
| Brier Score | *train to see* |
| F1 (Beat) | *train to see* |
| F1 (Miss) | *train to see* |
| Avg CV AUC | *train to see* |

> Run `python -m src.model` to populate these metrics. View them in the Model Performance tab of the dashboard.

---

## Screenshots

*Launch the dashboard with `streamlit run app.py` to see the full UI.*

---

## Methodology

### Why TimeSeriesSplit?
Financial data is inherently sequential. Using random train/test splits would allow the model to learn from future earnings results, creating an unrealistically optimistic evaluation. `TimeSeriesSplit` ensures each fold only trains on historical data and validates on subsequent periods, mimicking real-world prediction conditions.

### Why Ensemble?
XGBoost and LightGBM employ different tree-building algorithms (histogram vs. exact greedy) and regularization strategies. Averaging their predicted probabilities reduces model variance and produces better-calibrated predictions than either model alone.

### Look-Ahead Bias Prevention
All price-based features (returns, volume, 52-week high ratio) are computed as of **earnings_date − 2 business days** to ensure no post-announcement price moves contaminate the features. This is validated in the test suite.

---

## Limitations

- **API Rate Limits**: yfinance has implicit rate limits; bulk fetches are throttled with 0.5s delays. Alpha Vantage free tier is limited to 25 calls/day.
- **Training Window**: 3 years of quarterly data (~12 quarters per stock). Captures recent market regimes but may miss longer-term cycles.
- **Feature Coverage**: Not all features are available for every stock/quarter. Handled via median imputation — stocks with sparse data yield less reliable predictions.
- **Universe Size**: Trained on 50 of ~500 S&P constituents. Out-of-universe predictions should be treated with lower confidence.
- **Market Regime Shifts**: The model may underperform during unprecedented events (pandemics, rate shocks, geopolitical crises).

---

## Project Structure

```
earnings_predictor/
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Pinned dependencies
├── .env.example                    # API key template
├── README.md                       # This file
├── data/
│   └── .gitkeep
├── src/
│   ├── __init__.py                 # Package init
│   ├── __main__.py                 # python -m src.model entry point
│   ├── utils.py                    # Logging, caching, constants
│   ├── data_ingestion.py           # All data fetching logic
│   ├── feature_engineering.py      # Feature construction & registry
│   ├── model.py                    # Training, evaluation, persistence
│   └── predict.py                  # Inference pipeline
└── tests/
    ├── __init__.py
    ├── test_features.py            # Feature engineering tests
    └── test_model.py               # Model pipeline tests
```

---

## ⚠️ Disclaimer

**This tool is for educational and portfolio demonstration purposes only.** It is NOT financial advice. Do not make investment decisions based solely on these predictions. Past model performance does not guarantee future results. Always do your own due diligence and consult a qualified financial advisor before making any investment decisions.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
