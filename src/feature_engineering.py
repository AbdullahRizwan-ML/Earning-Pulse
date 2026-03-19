"""
Feature engineering module for EarningsPulse.

Orchestrates data ingestion functions to build a clean feature matrix for
model training and inference. Implements the complete feature registry,
missing data imputation, and lag feature creation.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data_ingestion import (
    fetch_all_training_data,
    get_analyst_features,
    get_earnings_history,
    get_fundamental_features,
    get_macro_features,
    get_price_features,
    get_sentiment_features,
)
from src.utils import (
    DATA_DIR,
    SECTOR_MAP,
    SP500_TICKERS,
    is_demo_mode,
    load_from_cache,
    save_to_cache,
    setup_logger,
)

log = setup_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE REGISTRY — powers the UI breakdown table
# ═══════════════════════════════════════════════════════════════════════════════
FEATURE_REGISTRY: Dict[str, str] = {
    # Group A — Analyst Estimate Features
    "eps_surprise_last_q": "EPS surprise magnitude in prior quarter",
    "eps_beat_streak": "Consecutive quarters of beating estimates (0–8)",
    "eps_revision_direction": "Analyst estimate revision trend (up vs. down)",
    "estimate_dispersion": "Dispersion / spread of analyst estimates",
    # Group B — Price Momentum Features
    "ret_1w": "1-week stock return prior to earnings",
    "ret_1m": "1-month stock return prior to earnings",
    "ret_3m": "3-month stock return prior to earnings",
    "price_vs_52w_high": "Current price as ratio of 52-week high",
    "volume_surge": "5-day avg volume / 30-day avg volume ratio",
    # Group C — Fundamental Quality Features
    "revenue_growth_yoy": "Year-over-year revenue growth rate",
    "gross_margin_trend": "Gross margin change vs. prior quarter",
    "fcf_yield": "Free cash flow yield (FCF / market cap)",
    "debt_to_equity": "Debt-to-equity ratio from balance sheet",
    # Group D — Sentiment / Insider Features
    "insider_buy_sell_ratio": "Net insider purchase ratio (last 90 days)",
    "institutional_ownership_change": "Institutional ownership level",
    "short_interest_ratio": "Short interest ratio (days to cover)",
    "recent_8k_filing": "Whether an 8-K was filed in the last 7 days",
    # Group E — Macro Context Features
    "yield_curve_spread": "10Y minus 2Y treasury yield spread",
    "vix_level": "VIX volatility index level",
    "sector_momentum": "Sector ETF 1-month return",
}

FEATURE_COLUMNS: List[str] = list(FEATURE_REGISTRY.keys())
TARGET_COLUMN: str = "beat"


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD FEATURE MATRIX (for training)
# ═══════════════════════════════════════════════════════════════════════════════
def build_feature_matrix(
    tickers: Optional[List[str]] = None,
    years: int = 3,
) -> pd.DataFrame:
    """Build the complete feature matrix for model training.

    Iterates through all tickers, fetches their earnings history, and for
    each earnings quarter computes all 20 features plus the beat/miss label.

    Args:
        tickers: List of ticker symbols. Defaults to ``SP500_TICKERS``.
        years: Years of earnings history to pull per ticker.

    Returns:
        DataFrame with all feature columns, the target ``beat`` column,
        and metadata columns ``[ticker, earnings_date]``.
    """
    if tickers is None:
        tickers = SP500_TICKERS

    # Check for cached full matrix
    cached = load_from_cache("feature_matrix", "full", str(years), ttl_hours=24)
    if cached is not None and not cached.empty:
        log.info("Loaded cached feature matrix: %d rows x %d cols", *cached.shape)
        return cached

    # Demo mode fallback
    if is_demo_mode():
        return _generate_demo_data(tickers)

    all_rows: List[Dict[str, Any]] = []
    total = len(tickers)
    batch_size = 5

    for idx, ticker in enumerate(tickers, 1):
        log.info("Building features for %s (%d/%d)", ticker, idx, total)
        try:
            # Get earnings history
            earnings_df = get_earnings_history(ticker, years=years)
            if earnings_df is None or earnings_df.empty:
                log.warning("No earnings data for %s — skipping", ticker)
                continue

            # Get analyst features (computed from earnings history)
            analyst_feats = get_analyst_features(ticker, earnings_df)

            # Get fundamental features (current snapshot applied to all quarters)
            fund_feats = get_fundamental_features(ticker)

            # Get sentiment features
            sent_feats = get_sentiment_features(ticker)

            # For each earnings quarter, compute per-quarter features
            for _, row in earnings_df.iterrows():
                earnings_date = row["earnings_date"]
                record: Dict[str, Any] = {
                    "ticker": ticker,
                    "sector": SECTOR_MAP.get(ticker, "Unknown"),
                    "earnings_date": earnings_date,
                    "actual_eps": row.get("actual_eps"),
                    "estimated_eps": row.get("estimated_eps"),
                    "beat": row.get("beat", 0),
                }

                # Group A — Analyst features
                record.update(analyst_feats)

                # Group B — Price features (per-quarter, look-ahead safe)
                price_feats = get_price_features(ticker, earnings_date)
                record.update(price_feats)

                # Group C — Fundamentals (apply current to all quarters)
                record.update(fund_feats)

                # Group D — Sentiment
                record.update(sent_feats)

                # Group E — Macro (per-date)
                macro_feats = get_macro_features(date=earnings_date, ticker=ticker)
                record.update(macro_feats)

                all_rows.append(record)

        except Exception as exc:
            log.warning("Failed to build features for %s: %s", ticker, exc)
            continue

        # Pause between batches of 5 tickers to avoid 429s
        if idx % batch_size == 0 and idx < total:
            log.info(
                "Batch %d/%d complete — pausing 10s to respect rate limits",
                idx // batch_size,
                (total + batch_size - 1) // batch_size,
            )
            time.sleep(10)

    if not all_rows:
        log.error("Feature matrix is empty — falling back to demo data")
        return _generate_demo_data(tickers)

    df = pd.DataFrame(all_rows)

    # Ensure all feature columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Handle missing data
    df = handle_missing_data(df)

    # Create lag features
    df = create_lag_features(df)

    # Cache the result
    save_to_cache(df, "feature_matrix", "full", str(years))
    log.info("Feature matrix built: %d rows x %d cols", *df.shape)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD FEATURES FOR SINGLE TICKER (inference)
# ═══════════════════════════════════════════════════════════════════════════════
def build_single_ticker_features(ticker: str) -> Dict[str, Any]:
    """Build the feature vector for a single ticker (live inference).

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dictionary containing all feature values for the ticker, ready
        for model prediction.
    """
    log.info("Building live features for %s", ticker)
    features: Dict[str, Any] = {"ticker": ticker}

    try:
        # Earnings history and analyst features
        earnings_df = get_earnings_history(ticker, years=3)
        analyst_feats = get_analyst_features(ticker, earnings_df)
        features.update(analyst_feats)

        # Price features (use today as reference)
        price_feats = get_price_features(ticker, datetime.now())
        features.update(price_feats)

        # Fundamentals
        fund_feats = get_fundamental_features(ticker)
        features.update(fund_feats)

        # Sentiment
        sent_feats = get_sentiment_features(ticker)
        features.update(sent_feats)

        # Macro
        macro_feats = get_macro_features(date=datetime.now(), ticker=ticker)
        features.update(macro_feats)

        # Add lag from earnings history
        if earnings_df is not None and not earnings_df.empty:
            features["prev_surprise"] = float(
                earnings_df.sort_values("earnings_date", ascending=False)
                .iloc[0]
                .get("surprise_pct", 0)
            )
            features["earnings_history"] = earnings_df
        else:
            features["prev_surprise"] = 0.0
            features["earnings_history"] = pd.DataFrame()

    except Exception as exc:
        log.error("Feature build failed for %s: %s", ticker, exc)

    return features


# ═══════════════════════════════════════════════════════════════════════════════
# MISSING DATA HANDLING
# ═══════════════════════════════════════════════════════════════════════════════
def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in the feature matrix.

    Uses median imputation for numeric features and zero-fill for ratio
    features. Logs columns with >30% missing data as a quality warning.

    Args:
        df: Raw feature matrix with potential NaN values.

    Returns:
        DataFrame with all NaN values imputed.
    """
    log.info("Handling missing data in %d rows x %d cols", *df.shape)

    # Report columns with high missingness
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            pct_missing = df[col].isna().sum() / len(df) * 100
            if pct_missing > 30:
                log.warning(
                    "Column '%s' has %.1f%% missing values", col, pct_missing
                )

    # Separate numeric features for imputation
    numeric_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    # Ratio features get zero-fill (makes semantic sense)
    ratio_features = [
        "insider_buy_sell_ratio",
        "institutional_ownership_change",
        "recent_8k_filing",
        "eps_beat_streak",
    ]
    for col in ratio_features:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Everything else gets median imputation
    remaining_cols = [c for c in numeric_cols if c not in ratio_features]
    for col in remaining_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

    log.info("Missing data imputation complete")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# LAG FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged features from the previous quarter's earnings surprise.

    The prior quarter's EPS surprise is one of the strongest predictors
    of future beats, capturing the momentum in earnings quality.

    Args:
        df: Feature matrix sorted by ticker and date.

    Returns:
        DataFrame with an additional ``prev_surprise`` column.
    """
    if df.empty:
        df["prev_surprise"] = np.nan
        return df

    df = df.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)

    # Create lagged surprise within each ticker group
    df["prev_surprise"] = df.groupby("ticker")["eps_surprise_last_q"].shift(1)

    # Fill NaN lag values with 0 (first quarter for each ticker)
    df["prev_surprise"] = df["prev_surprise"].fillna(0.0)

    log.info("Created lag features (prev_surprise)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
def _generate_demo_data(
    tickers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate synthetic demo data for offline/demo mode.

    Creates realistic-looking feature distributions so the app can run
    without internet connectivity.

    Args:
        tickers: List of ticker symbols to generate data for.

    Returns:
        Synthetic feature matrix with proper column structure.
    """
    if tickers is None:
        tickers = SP500_TICKERS

    log.info("Generating demo data for %d tickers", len(tickers))
    np.random.seed(42)

    records: List[Dict[str, Any]] = []
    quarters = pd.date_range(
        end=datetime.now(),
        periods=12,
        freq="QE",  # Quarterly end
    )

    for ticker in tickers:
        sector = SECTOR_MAP.get(ticker, "Unknown")
        for q_date in quarters:
            actual_eps = np.random.normal(1.5, 0.8)
            estimate_eps = actual_eps + np.random.normal(-0.05, 0.15)
            beat = 1 if actual_eps > estimate_eps else 0

            record = {
                "ticker": ticker,
                "sector": sector,
                "earnings_date": q_date,
                "actual_eps": round(actual_eps, 2),
                "estimated_eps": round(estimate_eps, 2),
                "beat": beat,
                # Group A
                "eps_surprise_last_q": np.random.normal(0.02, 0.05),
                "eps_beat_streak": np.random.randint(0, 6),
                "eps_revision_direction": np.random.normal(0.01, 0.03),
                "estimate_dispersion": abs(np.random.normal(0.1, 0.05)),
                # Group B
                "ret_1w": np.random.normal(0.005, 0.03),
                "ret_1m": np.random.normal(0.01, 0.06),
                "ret_3m": np.random.normal(0.03, 0.10),
                "price_vs_52w_high": np.random.uniform(0.7, 1.0),
                "volume_surge": np.random.lognormal(0, 0.2),
                # Group C
                "revenue_growth_yoy": np.random.normal(0.08, 0.15),
                "gross_margin_trend": np.random.normal(0.005, 0.02),
                "fcf_yield": np.random.uniform(0.01, 0.08),
                "debt_to_equity": abs(np.random.normal(0.8, 0.5)),
                # Group D
                "insider_buy_sell_ratio": np.random.uniform(0, 1),
                "institutional_ownership_change": np.random.uniform(0.5, 0.95),
                "short_interest_ratio": abs(np.random.normal(3, 2)),
                "recent_8k_filing": float(np.random.choice([0, 1], p=[0.7, 0.3])),
                # Group E
                "yield_curve_spread": np.random.normal(0.01, 0.005),
                "vix_level": np.random.uniform(12, 35),
                "sector_momentum": np.random.normal(0.02, 0.05),
            }
            records.append(record)

    df = pd.DataFrame(records)
    df = create_lag_features(df)
    df = handle_missing_data(df)

    # Save as demo file
    demo_path = DATA_DIR / "demo_feature_matrix.parquet"
    df.to_parquet(demo_path, index=False)
    log.info("Demo data generated: %d rows x %d cols", *df.shape)
    return df


def get_all_features_with_metadata() -> List[Dict[str, str]]:
    """Return the feature registry as a list of dicts for the UI.

    Returns:
        List of dictionaries, each with ``name`` and ``description`` keys.
    """
    return [
        {"name": name, "description": desc}
        for name, desc in FEATURE_REGISTRY.items()
    ]
