"""
Live feature fetcher for EarningsPulse.

Fetches features for a SINGLE ticker at prediction time using at most
3 Alpha Vantage API calls + 2 free FRED calls.  Results are cached for
24 hours to minimize API usage.

Computes the same 12 features as the training pipeline (data_loader.py),
all derived from earnings history + monthly prices + FRED macro data.
"""

import json
import os
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from src.data_loader import FEATURE_COLUMNS, FEATURE_REGISTRY, _safe_float
from src.utils import DATA_DIR, SECTOR_MAP, setup_logger

log = setup_logger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────────────
LIVE_CACHE_DIR = DATA_DIR / "cache"
LIVE_CACHE_TTL_HOURS = 24
MEDIANS_PATH = DATA_DIR / "feature_medians.json"


def _load_medians() -> Dict[str, float]:
    """Load training-set feature medians for imputation of missing values.

    Returns:
        Dictionary of feature_name → median value.
        Returns zeros for all features if the medians file is not found.
    """
    if MEDIANS_PATH.exists():
        with open(MEDIANS_PATH) as f:
            return json.load(f)
    log.warning("feature_medians.json not found — using 0 for missing features")
    return {col: 0.0 for col in FEATURE_COLUMNS}


def fetch_live_features(ticker: str) -> Dict[str, Any]:
    """Fetch features for a single ticker for live Streamlit prediction.

    Uses at most 3 Alpha Vantage calls (EARNINGS, TIME_SERIES_MONTHLY_ADJUSTED)
    plus 2 free FRED calls.   Results are cached for 24 hours.

    All 12 features are derived from:
      - Earnings history (beat streak, lag surprises, growth, consistency)
      - Monthly price data (1m / 3m returns)
      - FRED macro data (VIX, yield spread)

    Args:
        ticker: Stock ticker symbol (e.g. ``"AAPL"``).

    Returns:
        Dictionary with keys ``features`` (dict of 12 feature values),
        ``earnings_history`` (list of recent quarters), ``ticker``, ``sector``.

    Raises:
        ValueError: If the ticker is not found by Alpha Vantage.
    """
    ticker = ticker.upper().strip()
    LIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Check cache ──
    cache_file = LIVE_CACHE_DIR / f"live_{ticker}.json"
    if cache_file.exists():
        try:
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_hours < LIVE_CACHE_TTL_HOURS:
                with open(cache_file) as f:
                    cached = json.load(f)
                log.info("Live cache hit for %s (%.1fh old)", ticker, age_hours)
                return cached
        except Exception:
            pass

    av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not av_key or av_key == "your_key_here":
        raise ValueError(
            "ALPHA_VANTAGE_API_KEY not set. "
            "Get a free key at https://www.alphavantage.co/support/#api-key"
        )

    medians = _load_medians()

    # Start with median defaults for every feature
    features: Dict[str, Any] = {col: medians.get(col, 0.0) for col in FEATURE_COLUMNS}

    # ── 1. Alpha Vantage EARNINGS ──
    earnings = _av_request("EARNINGS", ticker, av_key)
    earnings_history_rows: List[Dict[str, Any]] = []

    if earnings and "quarterlyEarnings" in earnings:
        quarters = earnings["quarterlyEarnings"]

        for q in quarters[:12]:  # last 12 quarters
            try:
                reported = q.get("reportedEPS", "None")
                estimated = q.get("estimatedEPS", "None")
                if reported == "None" or estimated == "None":
                    continue
                actual = float(reported)
                est = float(estimated)
                beat = 1 if actual > est else 0
                surprise = (
                    (actual - est) / abs(est) * 100 if est != 0 else 0.0
                )
                earnings_history_rows.append({
                    "earnings_date": q.get("fiscalDateEnding", ""),
                    "actual_eps": actual,
                    "estimated_eps": est,
                    "beat": beat,
                    "surprise_pct": surprise,
                })
            except (TypeError, ValueError):
                continue

        if not earnings_history_rows:
            raise ValueError(f"No valid earnings data found for ticker '{ticker}'")

        # earnings_history_rows[0] = most recent (already announced)
        # For live prediction these are all PAST quarters — no leakage.

        # eps_beat_streak: count consecutive beats from most recent backwards
        beat_streak = 0
        for q in earnings_history_rows:
            if q["beat"] == 1:
                beat_streak += 1
            else:
                break
        features["eps_beat_streak"] = beat_streak

        # eps_surprise_last_q
        features["eps_surprise_last_q"] = earnings_history_rows[0]["surprise_pct"]

        # beat_last_q
        features["beat_last_q"] = float(earnings_history_rows[0]["beat"])

        # beat_2q_ago
        if len(earnings_history_rows) >= 2:
            features["beat_2q_ago"] = float(earnings_history_rows[1]["beat"])
        else:
            features["beat_2q_ago"] = 0.0

        # eps_growth_yoy: compare most recent quarter to same quarter last year (4 back)
        if len(earnings_history_rows) >= 5:
            curr_eps = earnings_history_rows[0]["actual_eps"]
            prev_year_eps = earnings_history_rows[4]["actual_eps"]
            if prev_year_eps != 0:
                features["eps_growth_yoy"] = (curr_eps - prev_year_eps) / abs(prev_year_eps)

        # eps_acceleration: surprise_last_q - surprise_2q_ago
        if len(earnings_history_rows) >= 2:
            s1 = earnings_history_rows[0]["surprise_pct"]
            s2 = earnings_history_rows[1]["surprise_pct"]
            features["eps_acceleration"] = s1 - s2

        # eps_consistency: beat rate in last 4 quarters
        lookback = min(len(earnings_history_rows), 4)
        beats_in_window = sum(q["beat"] for q in earnings_history_rows[:lookback])
        features["eps_consistency"] = beats_in_window / lookback

        # eps_magnitude_trend: mean(|surprise|) last 4 quarters
        lookback_mag = min(len(earnings_history_rows), 4)
        mag_values = [abs(q["surprise_pct"]) for q in earnings_history_rows[:lookback_mag]]
        features["eps_magnitude_trend"] = sum(mag_values) / len(mag_values)

    else:
        raise ValueError(f"No earnings data found for ticker '{ticker}'")

    # ── 2. Alpha Vantage TIME_SERIES_MONTHLY_ADJUSTED (price features) ──
    monthly = _av_request("TIME_SERIES_MONTHLY_ADJUSTED", ticker, av_key)
    if monthly and "Monthly Adjusted Time Series" in monthly:
        ts = monthly["Monthly Adjusted Time Series"]

        # Parse and sort monthly dates (most recent first)
        dated_prices = []
        for date_str, values in ts.items():
            try:
                dt = pd.to_datetime(date_str)
                close = float(values.get("5. adjusted close", values.get("4. close", 0)))
                if close > 0:
                    dated_prices.append((dt, close))
            except (ValueError, TypeError):
                continue

        dated_prices.sort(key=lambda x: x[0], reverse=True)

        if len(dated_prices) >= 2:
            curr_price = dated_prices[0][1]
            prev_1m = dated_prices[1][1]
            if prev_1m > 0:
                features["price_ret_1m"] = (curr_price - prev_1m) / prev_1m

        if len(dated_prices) >= 4:
            curr_price = dated_prices[0][1]
            prev_3m = dated_prices[3][1]
            if prev_3m > 0:
                features["price_ret_3m"] = (curr_price - prev_3m) / prev_3m

    # ── 3. FRED VIX (free, no key) ──
    try:
        vix_resp = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS",
            timeout=10,
        )
        if vix_resp.status_code == 200:
            vix_df = pd.read_csv(StringIO(vix_resp.text))
            vix_df.columns = vix_df.columns.str.strip()
            vix_df = vix_df[vix_df.iloc[:, 0].astype(str) != "."]
            vix_df["VALUE"] = pd.to_numeric(vix_df.iloc[:, 1], errors="coerce")
            vix_df = vix_df.dropna(subset=["VALUE"])
            if not vix_df.empty:
                features["vix_level"] = float(vix_df["VALUE"].iloc[-1])
    except Exception as exc:
        log.debug("FRED VIX fetch failed: %s", exc)

    # ── 4. FRED yield spread (free, no key) ──
    try:
        spread_resp = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y",
            timeout=10,
        )
        if spread_resp.status_code == 200:
            sp_df = pd.read_csv(StringIO(spread_resp.text))
            sp_df.columns = sp_df.columns.str.strip()
            sp_df = sp_df[sp_df.iloc[:, 0].astype(str) != "."]
            sp_df["VALUE"] = pd.to_numeric(sp_df.iloc[:, 1], errors="coerce")
            sp_df = sp_df.dropna(subset=["VALUE"])
            if not sp_df.empty:
                features["yield_spread"] = float(sp_df["VALUE"].iloc[-1])
    except Exception as exc:
        log.debug("FRED T10Y2Y fetch failed: %s", exc)

    # ── Persist cache ──
    result = {
        "features": {k: features[k] for k in FEATURE_COLUMNS},
        "ticker": ticker,
        "sector": SECTOR_MAP.get(ticker, "Unknown"),
        "earnings_history": earnings_history_rows[:8],
        "fetched_at": datetime.now().isoformat(),
    }
    try:
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception:
        pass

    log.info("Fetched live features for %s", ticker)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ALPHA VANTAGE REQUEST HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def _av_request(
    function: str,
    ticker: str,
    api_key: str,
) -> Optional[Dict[str, Any]]:
    """Single Alpha Vantage API call with 12s inter-call delay.

    Args:
        function: AV function name.
        ticker: Stock ticker symbol.
        api_key: Alpha Vantage API key.

    Returns:
        Parsed JSON dict or ``None`` on failure.
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": ticker,
        "apikey": api_key,
    }

    log.info("Alpha Vantage %s → %s", function, ticker)
    time.sleep(12)  # respect rate limit

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "Note" in data or "Information" in data:
            log.warning("AV rate-limit: %s", data.get("Note", data.get("Information")))
            return None
        return data
    except Exception as exc:
        log.warning("AV %s failed for %s: %s", function, ticker, exc)
        return None
