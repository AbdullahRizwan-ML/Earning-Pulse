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
  <img src="https://img.shields.io/badge/Data-Alpha%20Vantage%20%2B%20FRED-yellow" alt="Data">
</p>

---

## Overview

**EarningsPulse** is a machine learning system that predicts whether an S&P 500 company will **beat** or **miss** Wall Street's EPS consensus estimate before the earnings announcement. An ensemble of XGBoost + LightGBM is trained on 1,600 quarterly earnings records from 15 tickers, using 12 look-ahead-safe features derived from earnings history, price momentum, and macro context. Cross-validation uses **GroupShuffleSplit** (full ticker holdout) so each fold tests the model on companies it has *never* seen during training — producing honest, generalizable AUC scores.

---

## Model Performance

| Metric | Value |
|--------|-------|
| CV AUC (GroupShuffleSplit) | **0.74** |
| CV Accuracy | **0.73** |
| CV F1 (weighted) | **0.72** |
| Naive baseline (majority class) | 0.70 |
| Training samples | 1,600 |
| Companies trained on | 15 S&P 500 tickers |
| Evaluation method | GroupShuffleSplit — full ticker holdout |
| CV folds | 5 |

> **CV AUC 0.74 vs naive 0.70**: The model beats the majority-class baseline by ~4 percentage points on companies it has never seen during training. On the final held-out fold the ensemble achieves ROC-AUC > 0.99 (it overfits to the full-data final model), so the cross-validation figure is the honest estimate.

---

## Features (12, all look-ahead safe)

| # | Feature | Description | Data Source |
|---|---------|-------------|-------------|
| 1 | `eps_beat_streak` | Consecutive quarters beating estimates (capped at 8) | Alpha Vantage EARNINGS |
| 2 | `eps_surprise_last_q` | Prior quarter EPS surprise % — lagged, no leakage | Alpha Vantage EARNINGS |
| 3 | `beat_last_q` | Beat estimate last quarter (1/0) | Alpha Vantage EARNINGS |
| 4 | `beat_2q_ago` | Beat estimate two quarters ago (1/0) | Alpha Vantage EARNINGS |
| 5 | `eps_growth_yoy` | Year-over-year EPS growth (same quarter vs prior year) | Alpha Vantage EARNINGS |
| 6 | `eps_acceleration` | Surprise trend: last-Q surprise minus 2Q-ago surprise | Alpha Vantage EARNINGS |
| 7 | `eps_consistency` | Rolling 4-quarter beat rate (0–1) | Alpha Vantage EARNINGS |
| 8 | `eps_magnitude_trend` | Mean absolute surprise % over last 4 quarters | Alpha Vantage EARNINGS |
| 9 | `price_ret_1m` | 1-month price return before quarter-end (5-day cutoff) | Alpha Vantage MONTHLY |
| 10 | `price_ret_3m` | 3-month price return before quarter-end (5-day cutoff) | Alpha Vantage MONTHLY |
| 11 | `vix_level` | CBOE VIX at earnings date (market fear gauge) | FRED VIXCLS |
| 12 | `yield_spread` | 10Y–2Y Treasury yield spread at earnings date | FRED T10Y2Y |

All price features use a 5-day pre-quarter cutoff to prevent post-announcement contamination.

---

## Quickstart

```bash
git clone https://github.com/yourusername/earningspulse.git
cd earningspulse
python -m venv venv && venv\Scripts\activate    # Windows
pip install -r requirements.txt
cp .env.example .env   # add your Alpha Vantage key
python -m src.model    # trains on cached CSV (~30 seconds)
streamlit run app.py
```

Open `http://localhost:8501` to access the EarningsPulse dashboard.

```bash
# Run tests
python -m pytest tests/ -v
```

> **Note:** The CSV (`data/earnings_dataset.csv`) is pre-built and committed. Running `python -m src.model` reads from it and retrains the models in ~30 seconds without any API calls. To rebuild the CSV from scratch you need a free Alpha Vantage key.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                  │
│  Alpha Vantage (EARNINGS + MONTHLY_ADJUSTED) │ FRED (VIXCLS, T10Y2Y)│
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│              DATA LOADER  (src/data_loader.py)                       │
│  Loads pre-built CSV or builds from API with local caching           │
│  12 look-ahead-safe features │ schema versioning │ median imputation │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     MODEL TRAINING  (src/model.py)                   │
│  XGBoost  ──┐                                                        │
│             ├── Ensemble (average probabilities)                     │
│  LightGBM ──┘                                                        │
│  • GroupShuffleSplit(5) — full ticker holdout CV                     │
│  • RobustScaler │ feature importance │ model persistence             │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│              LIVE INFERENCE  (src/predict.py + src/live_fetch.py)    │
│  Fetches live features for any ticker │ median imputation fallback   │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   STREAMLIT DASHBOARD  (app.py)                      │
│  Tab 1: Predict a Stock │ Tab 2: Earnings Watchlist                  │
│  Tab 3: Model Performance │ Tab 4: Methodology                      │
│  • Plotly charts │ dark theme │ CSV export                           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

| Source | Endpoint | Used For |
|--------|----------|----------|
| **Alpha Vantage** | `EARNINGS` | Quarterly reported vs estimated EPS per ticker |
| **Alpha Vantage** | `TIME_SERIES_MONTHLY_ADJUSTED` | Monthly adjusted close prices for momentum features |
| **FRED** | `VIXCLS` | CBOE VIX daily closes (free, no key) |
| **FRED** | `T10Y2Y` | 10Y–2Y Treasury yield spread (free, no key) |

Alpha Vantage free tier: 25 calls/day. The builder uses a 12-second delay between calls and caches all responses in `data/cache/` so repeated runs are instant.

---

## Methodology

### Why GroupShuffleSplit instead of TimeSeriesSplit?

With only 15 tickers, each with ~107 quarters, a time-based split would still train and validate on the *same companies* — just at different time periods. This leaks company-specific patterns (management style, sector dynamics, analyst coverage) into validation, producing artificially optimistic AUC scores.

**GroupShuffleSplit** holds out **3 complete companies** per fold. The model never sees any quarter from those companies during training, testing true generalization to unseen firms. This is the correct methodology for financial ML on a small, heterogeneous universe.

### Why Ensemble?

XGBoost and LightGBM use different tree-building algorithms (exact greedy vs histogram) and regularization strategies. Averaging their predicted probabilities reduces variance and produces better-calibrated predictions than either alone.

### Look-Ahead Bias Prevention

- **EPS features**: All use lagged values (prior quarter's surprise, streak up to quarter *i−1*). Current-quarter data is the target, never a feature.
- **Price features**: Computed from monthly closes on or before `quarter_end − 5 days`. This avoids any post-announcement price moves.
- **Macro features**: FRED daily series, looked up at the closest date ≤ `quarter_end`.

---

## Project Structure

```
earningspulse/
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Pinned dependencies
├── .env.example                # API key template
├── README.md                   # This file
├── data/
│   ├── earnings_dataset.csv    # Pre-built training dataset (1,600 rows)
│   ├── xgb_model.joblib        # Trained XGBoost model
│   ├── lgbm_model.joblib       # Trained LightGBM model
│   ├── feature_scaler.joblib   # RobustScaler fitted on training data
│   ├── feature_medians.json    # Median values for live imputation
│   ├── feature_importance.json # Feature importance rankings
│   ├── model_metrics.json      # CV metrics and evaluation results
│   └── schema_version.txt      # Dataset schema version (v3)
├── src/
│   ├── __init__.py
│   ├── __main__.py             # python -m src.model entry point
│   ├── utils.py                # Logging, paths, constants
│   ├── data_loader.py          # CSV loader, feature constants, FRED fetcher
│   ├── live_fetch.py           # Live feature fetching for inference
│   ├── model.py                # Training, evaluation, persistence
│   └── predict.py              # Inference pipeline
└── tests/
    ├── __init__.py
    ├── test_features.py        # Feature engineering tests (39 total)
    └── test_model.py           # Model pipeline tests
```

---

## Limitations

- **Universe size**: Trained on 15 of ~500 S&P 500 constituents. Out-of-universe predictions should be treated with lower confidence.
- **API rate limits**: Alpha Vantage free tier is 25 calls/day. The builder caches aggressively to minimize calls.
- **Training window**: ~107 quarters per ticker (1996–2024). Captures multiple market cycles but may miss very long-term regime shifts.
- **No fundamental features**: Revenue growth, margins, and debt ratios were deliberately excluded — they are reported *after* the earnings call and cannot be known before the announcement without look-ahead leakage.
- **Market regime shifts**: The model may underperform during unprecedented events (pandemics, rate shocks, geopolitical crises).

---

## ⚠️ Disclaimer

**This tool is for educational and portfolio demonstration purposes only.** It is NOT financial advice. Do not make investment decisions based solely on these predictions. Past model performance does not guarantee future results. Always do your own due diligence and consult a qualified financial advisor before making any investment decisions.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
