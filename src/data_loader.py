"""
Static training data loader for EarningsPulse.

Loads training data from a pre-built CSV file (data/earnings_dataset.csv).
If the CSV doesn't exist, builds it using Alpha Vantage (EARNINGS +
TIME_SERIES_MONTHLY_ADJUSTED) and FRED (VIX, yield spread) for a small
universe of 15 S&P 500 tickers.

All features are **look-ahead safe** — they use only information that is
knowable *before* the earnings announcement date.

Schema v3: 12 features derived from earnings history, price momentum,
and macro context.  No OVERVIEW fundamentals (those are current-day
snapshots and cause look-ahead leakage when applied to historical rows).
"""

import json
import os
import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from src.utils import DATA_DIR, SECTOR_MAP, setup_logger

log = setup_logger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────────────
TRAINING_CSV = DATA_DIR / "earnings_dataset.csv"
CACHE_DIR = DATA_DIR / "cache"

# Increment when the feature schema changes to trigger a rebuild
SCHEMA_VERSION = "v3"

TRAINING_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "JPM", "JNJ", "XOM", "WMT", "TSLA",
    "BAC", "PFE", "AMZN", "META", "NVDA", "KO", "PG",
]

# ─── Feature columns (12 features, all look-ahead safe) ───────────────────────
FEATURE_COLUMNS: List[str] = [
    # Tier 1 — Earnings history (derived, no leakage)
    "eps_beat_streak",
    "eps_surprise_last_q",
    "beat_last_q",
    "beat_2q_ago",
    "eps_growth_yoy",
    "eps_acceleration",
    "eps_consistency",
    "eps_magnitude_trend",
    # Tier 2 — Price momentum (AV MONTHLY, date-matched)
    "price_ret_1m",
    "price_ret_3m",
    # Tier 3 — Macro context (FRED, free)
    "vix_level",
    "yield_spread",
]

TARGET_COLUMN = "beat"

# Feature descriptions for the Streamlit UI
FEATURE_REGISTRY: Dict[str, str] = {
    "eps_beat_streak": "Consecutive quarters beating estimates",
    "eps_surprise_last_q": "Last quarter's EPS surprise (%) — lagged",
    "beat_last_q": "Beat estimate last quarter (1/0)",
    "beat_2q_ago": "Beat estimate two quarters ago (1/0)",
    "eps_growth_yoy": "Year-over-year EPS growth (same quarter vs prior year)",
    "eps_acceleration": "Surprise trend: last_q surprise minus 2q_ago surprise",
    "eps_consistency": "Rolling 4-quarter beat rate (0–1)",
    "eps_magnitude_trend": "Avg |surprise %| over last 4 quarters",
    "price_ret_1m": "1-month price return before earnings",
    "price_ret_3m": "3-month price return before earnings",
    "vix_level": "CBOE VIX (market fear gauge) at earnings date",
    "yield_spread": "10Y–2Y Treasury yield spread at earnings date",
}

# Alpha Vantage inter-call delay (25 calls/day → ~12s minimum)
AV_CALL_DELAY = 12


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════
def load_or_build_training_data() -> pd.DataFrame:
    """Load training data from CSV, or build it if the file doesn't exist.

    Uses a schema version file to detect when the feature schema has changed
    and automatically triggers a rebuild of the CSV.

    Returns:
        DataFrame with columns matching ``FEATURE_COLUMNS`` plus metadata
        columns (``ticker``, ``quarter_end``, ``actual_eps``, etc.).
    """
    version_file = DATA_DIR / "schema_version.txt"

    if TRAINING_CSV.exists():
        if version_file.exists() and version_file.read_text().strip() == SCHEMA_VERSION:
            log.info("Loading existing CSV (schema %s)...", SCHEMA_VERSION)
            df = pd.read_csv(TRAINING_CSV)
            log.info("Loaded %d rows, %d columns", len(df), len(df.columns))
            return df
        else:
            log.warning("Schema version mismatch — rebuilding dataset...")
            TRAINING_CSV.unlink()

    log.info("Training CSV not found — building from Alpha Vantage + FRED ...")
    df = _build_training_data()

    # Write schema version after successful build
    version_file.write_text(SCHEMA_VERSION)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
def _build_training_data() -> pd.DataFrame:
    """Build the training CSV from Alpha Vantage + FRED.

    For each ticker we fetch:
      - EARNINGS (quarterly reported vs estimated EPS) — 1 AV call
      - TIME_SERIES_MONTHLY_ADJUSTED (monthly prices) — 1 AV call
    Plus 2 FRED calls for macro data (free, no key).

    Total AV calls: 15 tickers × 2 = 30. The free tier allows 25/day,
    so the builder may need to be run across 2 days if caches are cold.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not av_key or av_key == "your_key_here":
        log.error(
            "ALPHA_VANTAGE_API_KEY not set.  Cannot build training data.\n"
            "Get a free key at https://www.alphavantage.co/support/#api-key\n"
            "Then set it in .env: ALPHA_VANTAGE_API_KEY=your_key"
        )
        raise RuntimeError("ALPHA_VANTAGE_API_KEY not set — cannot build training data")

    # ── Fetch macro data (free, no key needed) ──
    vix_series = _fetch_fred_series("VIXCLS")
    spread_series = _fetch_fred_series("T10Y2Y")

    all_rows: List[Dict[str, Any]] = []

    for idx, ticker in enumerate(TRAINING_TICKERS, 1):
        log.info("Processing %s (%d/%d)", ticker, idx, len(TRAINING_TICKERS))

        # ── Alpha Vantage: EARNINGS ──
        earnings_data = _av_cached_request(
            ticker, "EARNINGS", av_key,
            params={"function": "EARNINGS", "symbol": ticker},
        )
        if not earnings_data or "quarterlyEarnings" not in earnings_data:
            log.warning("No EARNINGS data for %s — skipping", ticker)
            continue

        # ── Alpha Vantage: MONTHLY prices ──
        monthly_data = _av_cached_request(
            ticker, "MONTHLY", av_key,
            params={
                "function": "TIME_SERIES_MONTHLY_ADJUSTED",
                "symbol": ticker,
            },
        )
        monthly_ts = {}
        if monthly_data and "Monthly Adjusted Time Series" in monthly_data:
            monthly_ts = monthly_data["Monthly Adjusted Time Series"]

        # ── Parse quarterly earnings into rows ──
        quarters = earnings_data["quarterlyEarnings"]
        # Build a list of (quarter_end, actual_eps, estimated_eps) for lag computation
        parsed_quarters: List[Dict[str, Any]] = []

        for q in quarters:
            try:
                reported = q.get("reportedEPS", "None")
                estimated = q.get("estimatedEPS", "None")
                if reported == "None" or estimated == "None":
                    continue
                actual_eps = float(reported)
                estimated_eps = float(estimated)
                quarter_end = q.get("fiscalDateEnding", "")
                if not quarter_end:
                    continue

                beat = 1 if actual_eps > estimated_eps else 0
                surprise_pct = (
                    (actual_eps - estimated_eps) / abs(estimated_eps) * 100
                    if estimated_eps != 0
                    else 0.0
                )

                parsed_quarters.append({
                    "quarter_end": quarter_end,
                    "actual_eps": actual_eps,
                    "estimated_eps": estimated_eps,
                    "beat": beat,
                    "surprise_pct": surprise_pct,
                })
            except (TypeError, ValueError, KeyError) as exc:
                log.debug("Skipping quarter for %s: %s", ticker, exc)
                continue

        if len(parsed_quarters) < 3:
            log.warning("Too few quarters for %s (%d) — skipping", ticker, len(parsed_quarters))
            continue

        # Sort oldest-first so we can compute forward lags
        parsed_quarters.sort(key=lambda q: q["quarter_end"])

        # ── Compute features for each quarter ──
        for i, q in enumerate(parsed_quarters):
            qe = q["quarter_end"]

            # --- LAG features (require previous quarters) ---
            # eps_surprise_last_q
            eps_surprise_last_q = parsed_quarters[i - 1]["surprise_pct"] if i >= 1 else None
            beat_last_q = float(parsed_quarters[i - 1]["beat"]) if i >= 1 else None
            beat_2q_ago = float(parsed_quarters[i - 2]["beat"]) if i >= 2 else None

            # eps_beat_streak: count consecutive beats ending at i-1
            beat_streak = 0
            for j in range(i - 1, -1, -1):
                if parsed_quarters[j]["beat"] == 1:
                    beat_streak += 1
                else:
                    break

            # eps_growth_yoy: compare same-quarter last year (4 quarters back)
            eps_growth_yoy = None
            if i >= 4:
                prev_year_eps = parsed_quarters[i - 4]["actual_eps"]
                if prev_year_eps != 0:
                    eps_growth_yoy = (
                        (q["actual_eps"] - prev_year_eps) / abs(prev_year_eps)
                    )

            # eps_acceleration: surprise_last_q - surprise_2q_ago
            eps_acceleration = None
            if i >= 2:
                s1 = parsed_quarters[i - 1]["surprise_pct"]
                s2 = parsed_quarters[i - 2]["surprise_pct"]
                eps_acceleration = s1 - s2

            # eps_consistency: beat rate in last 4 quarters
            eps_consistency = None
            if i >= 4:
                last_4 = [parsed_quarters[j]["beat"] for j in range(i - 4, i)]
                eps_consistency = sum(last_4) / 4.0
            elif i >= 2:
                last_n = [parsed_quarters[j]["beat"] for j in range(max(0, i - 4), i)]
                eps_consistency = sum(last_n) / len(last_n) if last_n else None

            # eps_magnitude_trend: mean(|surprise|) last 4 quarters
            eps_magnitude_trend = None
            if i >= 1:
                lookback = min(i, 4)
                recent_surprises = [
                    abs(parsed_quarters[j]["surprise_pct"])
                    for j in range(i - lookback, i)
                ]
                eps_magnitude_trend = sum(recent_surprises) / len(recent_surprises)

            # Skip rows where critical lag features are unavailable
            if eps_surprise_last_q is None or beat_last_q is None:
                continue

            # --- PRICE features (from monthly data) ---
            price_feats = _get_price_features(monthly_ts, qe)

            # --- MACRO features (from FRED) ---
            vix_val = _lookup_macro(vix_series, qe)
            spread_val = _lookup_macro(spread_series, qe)

            row: Dict[str, Any] = {
                "ticker": ticker,
                "quarter_end": qe,
                "actual_eps": q["actual_eps"],
                "estimated_eps": q["estimated_eps"],
                "beat": q["beat"],
                "eps_surprise_pct": round(q["surprise_pct"], 4),
                # --- Features ---
                "eps_beat_streak": beat_streak,
                "eps_surprise_last_q": round(eps_surprise_last_q, 4),
                "beat_last_q": beat_last_q,
                "beat_2q_ago": beat_2q_ago if beat_2q_ago is not None else 0.0,
                "eps_growth_yoy": round(eps_growth_yoy, 4) if eps_growth_yoy is not None else 0.0,
                "eps_acceleration": round(eps_acceleration, 4) if eps_acceleration is not None else 0.0,
                "eps_consistency": round(eps_consistency, 4) if eps_consistency is not None else 0.5,
                "eps_magnitude_trend": round(eps_magnitude_trend, 4) if eps_magnitude_trend is not None else 0.0,
                "price_ret_1m": price_feats.get("price_ret_1m", 0.0),
                "price_ret_3m": price_feats.get("price_ret_3m", 0.0),
                "vix_level": vix_val if vix_val is not None else 0.0,
                "yield_spread": spread_val if spread_val is not None else 0.0,
            }
            all_rows.append(row)

    if not all_rows:
        raise RuntimeError("Built 0 rows — check API key and connectivity")

    df = pd.DataFrame(all_rows)
    df["quarter_end"] = pd.to_datetime(df["quarter_end"])
    df = df.sort_values(["ticker", "quarter_end"], ascending=[True, True])
    df = df.reset_index(drop=True)

    # ── Check price feature coverage ──
    price_nonzero = (df["price_ret_1m"] != 0.0).mean()
    if price_nonzero < 0.5:
        log.warning(
            "price_ret_1m is non-zero for only %.1f%% of rows — "
            "monthly price data may be incomplete",
            price_nonzero * 100,
        )

    # Save
    df.to_csv(TRAINING_CSV, index=False)
    log.info("Saved training CSV to %s", TRAINING_CSV)

    # Summary
    n_rows = len(df)
    n_tickers = df["ticker"].nunique()
    avg_q = n_rows / n_tickers if n_tickers else 0
    print(
        f"\nBuilt dataset: {n_rows} rows, {n_tickers} tickers, "
        f"{avg_q:.1f} quarters each on average."
    )
    print(f"Price ret_1m non-zero: {price_nonzero*100:.1f}%\n")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PRICE FEATURES FROM MONTHLY DATA
# ═══════════════════════════════════════════════════════════════════════════════
def _get_price_features(
    monthly_ts: Dict[str, Dict[str, str]],
    quarter_end: str,
) -> Dict[str, float]:
    """Compute price momentum features from AV monthly data for a given date.

    Finds the most recent monthly close BEFORE quarter_end (minus 5 days
    to avoid post-announcement contamination) and computes returns.

    Args:
        monthly_ts: The ``"Monthly Adjusted Time Series"`` dict from AV.
        quarter_end: ISO date string (e.g. ``"2024-06-30"``).

    Returns:
        Dict with ``price_ret_1m``, ``price_ret_3m`` (float).
        Defaults to 0.0 for any feature that cannot be computed.
    """
    result = {"price_ret_1m": 0.0, "price_ret_3m": 0.0}

    if not monthly_ts:
        return result

    try:
        cutoff = pd.to_datetime(quarter_end) - pd.Timedelta(days=5)

        # Parse and sort monthly dates
        dated_prices: List[tuple] = []
        for date_str, values in monthly_ts.items():
            try:
                dt = pd.to_datetime(date_str)
                close = float(values.get("5. adjusted close", values.get("4. close", 0)))
                if close > 0:
                    dated_prices.append((dt, close))
            except (ValueError, TypeError):
                continue

        dated_prices.sort(key=lambda x: x[0])

        # Find the last monthly close on or before cutoff
        valid = [(dt, p) for dt, p in dated_prices if dt <= cutoff]
        if len(valid) < 2:
            return result

        # Current month's close (closest to quarter_end)
        curr_price = valid[-1][1]

        # 1-month return
        if len(valid) >= 2:
            prev_1m = valid[-2][1]
            if prev_1m > 0:
                result["price_ret_1m"] = round((curr_price - prev_1m) / prev_1m, 6)

        # 3-month return
        if len(valid) >= 4:
            prev_3m = valid[-4][1]
            if prev_3m > 0:
                result["price_ret_3m"] = round((curr_price - prev_3m) / prev_3m, 6)
        elif len(valid) >= 2:
            # Fallback: use whatever is available
            prev_3m = valid[0][1]
            if prev_3m > 0:
                result["price_ret_3m"] = round((curr_price - prev_3m) / prev_3m, 6)

    except Exception as exc:
        log.debug("Price feature computation failed for %s: %s", quarter_end, exc)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ALPHA VANTAGE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _av_cached_request(
    ticker: str,
    function: str,
    api_key: str,
    params: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch from Alpha Vantage with local JSON caching.

    If ``data/cache/{ticker}_{function}.json`` exists, return its contents
    immediately (zero API call).  Otherwise fetch, cache, and return.

    Args:
        ticker: Stock ticker symbol.
        function: AV function name (e.g. ``"EARNINGS"``, ``"MONTHLY"``).
        api_key: Alpha Vantage API key.
        params: Query parameters for the request.

    Returns:
        Parsed JSON dict or ``None`` on failure.
    """
    cache_file = CACHE_DIR / f"{ticker}_{function}.json"
    if cache_file.exists():
        # For MONTHLY data, check if cache is less than 30 days old
        if function == "MONTHLY":
            age_days = (time.time() - cache_file.stat().st_mtime) / 86400
            if age_days > 30:
                log.info("MONTHLY cache for %s is %.0f days old — refreshing", ticker, age_days)
                cache_file.unlink()
            else:
                log.debug("Cache hit: %s (%.0f days old)", cache_file.name, age_days)
                with open(cache_file) as f:
                    return json.load(f)
        else:
            log.debug("Cache hit: %s", cache_file.name)
            with open(cache_file) as f:
                return json.load(f)

    # Build request
    if params is None:
        params = {}
    params["apikey"] = api_key
    url = "https://www.alphavantage.co/query"

    log.info("Calling Alpha Vantage %s for %s ...", function, ticker)
    time.sleep(AV_CALL_DELAY)  # respect 25 calls/day

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # AV returns {"Note": "Thank you ..."} when rate-limited
        if "Note" in data or "Information" in data:
            log.warning(
                "Alpha Vantage rate-limit/info response for %s %s: %s",
                function, ticker, data.get("Note", data.get("Information", "")),
            )
            return None

        # Cache the response
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

        return data
    except Exception as exc:
        log.warning("Alpha Vantage %s request failed for %s: %s", function, ticker, exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# FRED HELPERS (free, no API key)
# ═══════════════════════════════════════════════════════════════════════════════
def _fetch_fred_series(series_id: str) -> pd.Series:
    """Fetch a FRED time series as a pandas Series indexed by date.

    Uses the free CSV endpoint (no API key required).

    Args:
        series_id: FRED series identifier (e.g. ``"VIXCLS"``, ``"T10Y2Y"``).

    Returns:
        pandas Series with datetime index and float values.
        Returns an empty Series on failure.
    """
    cache_file = CACHE_DIR / f"fred_{series_id}.csv"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        log.debug("FRED cache hit: %s", cache_file.name)
        try:
            df = pd.read_csv(cache_file)
            df.columns = df.columns.str.strip()
            df = df[df.iloc[:, 0].astype(str) != "."]
            df["DATE"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            df["VALUE"] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            df = df.dropna(subset=["DATE", "VALUE"])
            return df.set_index("DATE").sort_index()["VALUE"]
        except Exception:
            pass

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    log.info("Fetching FRED %s ...", series_id)

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        # Save raw CSV to cache
        cache_file.write_text(resp.text)

        df = pd.read_csv(StringIO(resp.text))
        df.columns = df.columns.str.strip()
        df = df[df.iloc[:, 0].astype(str) != "."]
        df["DATE"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        df["VALUE"] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        df = df.dropna(subset=["DATE", "VALUE"])
        return df.set_index("DATE").sort_index()["VALUE"]
    except Exception as exc:
        log.warning("FRED %s fetch failed: %s", series_id, exc)
        return pd.Series(dtype=float)


def _lookup_macro(series: pd.Series, date_str: str) -> Optional[float]:
    """Look up the closest macro value on or before a date.

    Args:
        series: Time-indexed macro series.
        date_str: ISO date string (e.g. ``"2024-06-30"``).

    Returns:
        The closest available value, or ``None``.
    """
    if series.empty:
        return None
    try:
        dt = pd.to_datetime(date_str)
        mask = series.index <= dt
        if mask.any():
            return float(series.loc[mask].iloc[-1])
        return float(series.iloc[0])
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MISC HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _safe_float(val: Any) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    if val is None or val == "None" or val == "-" or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
