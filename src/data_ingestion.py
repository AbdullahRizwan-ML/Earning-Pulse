"""
Data ingestion module for EarningsPulse.

Handles all external data fetching from yfinance, SEC EDGAR, Alpha Vantage,
and FRED. Every function includes retry logic, caching, rate limiting, and
graceful error handling so the application never crashes on bad data.
"""

import os
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests as req_lib
import yfinance as yf

from src.utils import (
    DATA_DIR,
    SECTOR_ETFS,
    SECTOR_MAP,
    SP500_TICKERS,
    get_alpha_vantage_key,
    get_sec_api_key,
    load_from_cache,
    logger,
    retry,
    safe_divide,
    save_to_cache,
    setup_logger,
)

log = setup_logger(__name__)

# ─── Module-level yfinance session ─────────────────────────────────────────────
# Shared across all functions to maintain connection pooling and set headers
# that reduce 429 rate-limit responses from Yahoo Finance.
_yf_session = req_lib.Session()
_yf_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
})


# ─── Per-source rate limiter ───────────────────────────────────────────────────
# Tracks timestamps of recent calls to each data source.  If >20 calls have
# been made to a source within the last 60 seconds the helper sleeps until
# the oldest call in the window expires.  This replaces the simple
# _last_request_time counter with source-aware throttling.
RATE_LIMITER: Dict[str, List[float]] = {
    "yfinance": [],
    "alpha_vantage": [],
    "sec_edgar": [],
}
_RATE_LIMIT_MAX_CALLS = 20
_RATE_LIMIT_WINDOW = 60  # seconds

_last_request_time: float = 0.0


def _source_rate_limit(source: str = "yfinance") -> None:
    """Enforce per-source rate limiting.

    If more than ``_RATE_LIMIT_MAX_CALLS`` have been made to *source* within
    the last ``_RATE_LIMIT_WINDOW`` seconds, sleep until the oldest call
    falls outside the window.

    Args:
        source: One of ``'yfinance'``, ``'alpha_vantage'``, ``'sec_edgar'``.
    """
    now = time.time()
    timestamps = RATE_LIMITER.get(source, [])

    # Prune calls outside the window
    timestamps = [t for t in timestamps if now - t < _RATE_LIMIT_WINDOW]
    RATE_LIMITER[source] = timestamps

    if len(timestamps) >= _RATE_LIMIT_MAX_CALLS:
        oldest = timestamps[0]
        wait = _RATE_LIMIT_WINDOW - (now - oldest) + 1.0
        if wait > 0:
            log.info(
                "Source '%s' hit %d calls in %ds — sleeping %.1fs",
                source, _RATE_LIMIT_MAX_CALLS, _RATE_LIMIT_WINDOW, wait,
            )
            time.sleep(wait)

    RATE_LIMITER[source].append(time.time())


def _rate_limit(min_interval: float = 3.0, source: str = "yfinance") -> None:
    """Enforce minimum interval between API requests with random jitter.

    Also delegates to :func:`_source_rate_limit` for per-source windowed
    throttling.

    Args:
        min_interval: Minimum seconds between calls (default 3.0s).
        source: Data source name for per-source tracking.
    """
    global _last_request_time
    # Per-source windowed throttle
    _source_rate_limit(source)
    # Per-call jittered delay
    jittered = random.uniform(max(min_interval - 0.5, 0.5), min_interval + 1.0)
    elapsed = time.time() - _last_request_time
    if elapsed < jittered:
        time.sleep(jittered - elapsed)
    _last_request_time = time.time()


# ─── 429-aware retry helper ─────────────────────────────────────────────────────
def _retry_with_429_backoff(
    func: Any,
    max_retries: int = 3,
    backoff_schedule: tuple = (15, 30, 60),
) -> Any:
    """Call *func* with aggressive backoff on HTTP 429 errors.

    Args:
        func: Zero-argument callable to execute.
        max_retries: Maximum retry attempts.
        backoff_schedule: Seconds to wait on each successive 429.

    Returns:
        The return value of *func*, or ``None`` on exhausted retries.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as exc:
            exc_str = str(exc).lower()
            is_429 = "429" in exc_str or "too many requests" in exc_str
            wait = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]
            if is_429:
                log.warning(
                    "429 Too Many Requests (attempt %d/%d) — waiting %ds",
                    attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
            elif attempt < max_retries - 1:
                log.warning(
                    "Request failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, max_retries, exc, wait,
                )
                time.sleep(wait)
            else:
                log.error("Request failed after %d attempts: %s", max_retries, exc)
                return None
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# EARNINGS HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
def get_earnings_history(
    ticker: str,
    years: int = 3,
) -> Optional[pd.DataFrame]:
    """Fetch historical earnings data: actual EPS vs. consensus estimates.

    **Primary**: Alpha Vantage ``EARNINGS`` endpoint (``quarterlyEarnings``)
    provides clean ``reportedEPS`` / ``estimatedEPS`` pairs.  **Secondary**:
    ``yf.Ticker.get_earnings_dates()`` as fallback when no AV key is set or
    the AV call fails.

    Args:
        ticker: Stock ticker symbol (e.g., ``"AAPL"``).
        years: Number of years of history to retrieve (default 3).

    Returns:
        DataFrame with columns ``[ticker, quarter, earnings_date, actual_eps,
        estimated_eps, surprise_pct, beat]`` sorted by date descending, or
        ``None`` on failure.
    """
    # Check cache first
    cached = load_from_cache("earnings", ticker, str(years), ttl_hours=12)
    if cached is not None:
        return cached

    log.info("Fetching earnings history for %s (%d years)", ticker, years)

    records: List[Dict[str, Any]] = []
    cutoff = datetime.now() - timedelta(days=years * 365)

    # ── PRIMARY: Alpha Vantage EARNINGS endpoint ──
    av_key = get_alpha_vantage_key()
    if av_key:
        try:
            _rate_limit(min_interval=3.0, source="alpha_vantage")
            url = (
                f"https://www.alphavantage.co/query"
                f"?function=EARNINGS&symbol={ticker}&apikey={av_key}"
            )
            resp = req_lib.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("quarterlyEarnings", []):
                    try:
                        date_val = pd.to_datetime(item["fiscalDateEnding"])
                        if date_val < cutoff:
                            continue
                        reported = item.get("reportedEPS", "None")
                        estimated = item.get("estimatedEPS", "None")
                        if reported == "None" or estimated == "None":
                            continue
                        actual = float(reported)
                        estimate = float(estimated)
                        surprise_raw = item.get("surprisePercentage", "None")
                        if surprise_raw != "None":
                            surprise = float(surprise_raw) / 100
                        else:
                            surprise = safe_divide(
                                actual - estimate,
                                abs(estimate) if estimate != 0 else 1,
                            )
                        records.append({
                            "ticker": ticker,
                            "quarter": date_val.strftime("%Y-%m"),
                            "earnings_date": date_val,
                            "actual_eps": actual,
                            "estimated_eps": estimate,
                            "surprise_pct": surprise,
                            "beat": 1 if actual > estimate else 0,
                        })
                    except (TypeError, ValueError, KeyError):
                        continue
                log.info(
                    "Alpha Vantage returned %d earnings records for %s",
                    len(records), ticker,
                )
        except Exception as exc:
            log.warning("Alpha Vantage EARNINGS failed for %s: %s", ticker, exc)

    # ── SECONDARY: yfinance get_earnings_dates() ──
    if len(records) < 4:
        try:
            _rate_limit(source="yfinance")

            def _fetch_earnings_dates() -> pd.DataFrame:
                tk = yf.Ticker(ticker, session=_yf_session)
                return tk.get_earnings_dates(limit=years * 4 + 4)

            ed_df = _retry_with_429_backoff(_fetch_earnings_dates)

            if ed_df is not None and isinstance(ed_df, pd.DataFrame) and not ed_df.empty:
                existing_dates = {r["earnings_date"] for r in records}
                for idx, row in ed_df.iterrows():
                    try:
                        date_val = (
                            pd.to_datetime(idx)
                            if not isinstance(idx, (datetime, pd.Timestamp))
                            else idx
                        )
                        # get_earnings_dates returns tz-aware timestamps
                        if hasattr(date_val, "tz") and date_val.tz is not None:
                            date_val = date_val.tz_localize(None)
                        if date_val < cutoff:
                            continue
                        actual = row.get("Reported EPS", row.get("epsActual", None))
                        estimate = row.get("EPS Estimate", row.get("epsEstimate", None))
                        if actual is None or estimate is None:
                            continue
                        if pd.isna(actual) or pd.isna(estimate):
                            continue
                        actual = float(actual)
                        estimate = float(estimate)
                        if date_val in existing_dates:
                            continue
                        surprise = safe_divide(
                            actual - estimate,
                            abs(estimate) if estimate != 0 else 1,
                        )
                        records.append({
                            "ticker": ticker,
                            "quarter": date_val.strftime("%Y-%m"),
                            "earnings_date": date_val,
                            "actual_eps": actual,
                            "estimated_eps": estimate,
                            "surprise_pct": surprise,
                            "beat": 1 if actual > estimate else 0,
                        })
                    except (TypeError, ValueError) as exc:
                        log.debug("Skipping yfinance row for %s: %s", ticker, exc)
                        continue
                log.info(
                    "yfinance added records — total now %d for %s",
                    len(records), ticker,
                )
        except Exception as exc:
            log.warning("yfinance get_earnings_dates failed for %s: %s", ticker, exc)

    if not records:
        log.warning("No earnings data found for %s", ticker)
        return None

    df = pd.DataFrame(records)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df = df.sort_values("earnings_date", ascending=False).reset_index(drop=True)

    # Deduplicate by quarter
    df = df.drop_duplicates(subset=["ticker", "earnings_date"], keep="first")

    save_to_cache(df, "earnings", ticker, str(years))
    log.info("Retrieved %d earnings records for %s", len(df), ticker)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PRICE MOMENTUM FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
@retry(max_retries=3, backoff_seconds=15.0)
def get_price_features(
    ticker: str,
    earnings_date: datetime,
) -> Dict[str, Optional[float]]:
    """Compute price momentum features relative to an earnings date.

    All price data is taken as of ``earnings_date - 2 days`` to prevent
    look-ahead bias.

    Args:
        ticker: Stock ticker symbol.
        earnings_date: The earnings announcement date.

    Returns:
        Dictionary with keys: ``ret_1w``, ``ret_1m``, ``ret_3m``,
        ``price_vs_52w_high``, ``volume_surge``.
    """
    features: Dict[str, Optional[float]] = {
        "ret_1w": None,
        "ret_1m": None,
        "ret_3m": None,
        "price_vs_52w_high": None,
        "volume_surge": None,
    }

    # Avoid look-ahead bias: use data up to 2 days before earnings
    end_date = pd.to_datetime(earnings_date) - timedelta(days=2)
    start_date = end_date - timedelta(days=400)  # ~13 months for 52w high

    try:
        _rate_limit()
        hist = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
            session=_yf_session,
        )

        if hist.empty or len(hist) < 10:
            log.warning("Insufficient price data for %s around %s", ticker, earnings_date)
            return features

        # Flatten multi-level columns if present
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        close = hist["Close"]
        volume = hist["Volume"]

        # Current price (last available)
        current_price = float(close.iloc[-1])

        # Returns
        if len(close) >= 5:
            features["ret_1w"] = float((current_price / close.iloc[-5]) - 1)
        if len(close) >= 21:
            features["ret_1m"] = float((current_price / close.iloc[-21]) - 1)
        if len(close) >= 63:
            features["ret_3m"] = float((current_price / close.iloc[-63]) - 1)

        # Price vs 52-week high
        high_252 = close.tail(252).max()
        features["price_vs_52w_high"] = safe_divide(current_price, float(high_252))

        # Volume surge: avg volume last 5 days / avg volume last 30 days
        if len(volume) >= 30:
            vol_5d = float(volume.tail(5).mean())
            vol_30d = float(volume.tail(30).mean())
            features["volume_surge"] = safe_divide(vol_5d, vol_30d)

    except Exception as exc:
        log.warning("Price features failed for %s: %s", ticker, exc)

    return features


# ═══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL QUALITY FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
@retry(max_retries=3, backoff_seconds=15.0)
def get_fundamental_features(ticker: str) -> Dict[str, Optional[float]]:
    """Extract fundamental quality features from quarterly financials.

    Primary source is ``yf.Ticker`` (quarterly financials, balance sheet,
    cashflow).  If yfinance returns empty data, falls back to Alpha Vantage
    ``OVERVIEW`` endpoint.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dictionary with keys: ``revenue_growth_yoy``, ``gross_margin_trend``,
        ``fcf_yield``, ``debt_to_equity``.
    """
    features: Dict[str, Optional[float]] = {
        "revenue_growth_yoy": None,
        "gross_margin_trend": None,
        "fcf_yield": None,
        "debt_to_equity": None,
    }

    yf_success = False  # Track whether yfinance produced any data

    try:
        _rate_limit(source="yfinance")
        tk = yf.Ticker(ticker, session=_yf_session)
        info = tk.info or {}

        # ── Revenue growth YoY ──
        try:
            fin = tk.quarterly_financials
            if fin is not None and not fin.empty:
                revenue_row = None
                for label in ["Total Revenue", "Revenue", "Operating Revenue"]:
                    if label in fin.index:
                        revenue_row = fin.loc[label]
                        break
                if revenue_row is not None and len(revenue_row) >= 5:
                    latest = float(revenue_row.iloc[0])
                    year_ago = float(revenue_row.iloc[4])
                    features["revenue_growth_yoy"] = safe_divide(latest - year_ago, abs(year_ago))
                    yf_success = True
                elif revenue_row is not None and len(revenue_row) >= 2:
                    latest = float(revenue_row.iloc[0])
                    prev = float(revenue_row.iloc[1])
                    features["revenue_growth_yoy"] = safe_divide(latest - prev, abs(prev))
                    yf_success = True
        except Exception as exc:
            log.debug("Revenue growth failed for %s: %s", ticker, exc)

        # ── Gross margin trend ──
        try:
            fin = tk.quarterly_financials
            if fin is not None and not fin.empty:
                gp_row = None
                rev_row = None
                for label in ["Gross Profit"]:
                    if label in fin.index:
                        gp_row = fin.loc[label]
                for label in ["Total Revenue", "Revenue"]:
                    if label in fin.index:
                        rev_row = fin.loc[label]
                        break
                if gp_row is not None and rev_row is not None and len(gp_row) >= 2:
                    margin_now = safe_divide(float(gp_row.iloc[0]), float(rev_row.iloc[0]))
                    margin_prev = safe_divide(float(gp_row.iloc[1]), float(rev_row.iloc[1]))
                    features["gross_margin_trend"] = margin_now - margin_prev
                    yf_success = True
        except Exception as exc:
            log.debug("Gross margin trend failed for %s: %s", ticker, exc)

        # ── FCF yield ──
        try:
            cf = tk.quarterly_cashflow
            if cf is not None and not cf.empty:
                fcf_row = None
                for label in ["Free Cash Flow", "Operating Cash Flow"]:
                    if label in cf.index:
                        fcf_row = cf.loc[label]
                        break
                if fcf_row is not None:
                    fcf_annual = float(fcf_row.iloc[:4].sum())
                    market_cap = info.get("marketCap", None)
                    if market_cap and market_cap > 0:
                        features["fcf_yield"] = fcf_annual / market_cap
                        yf_success = True
        except Exception as exc:
            log.debug("FCF yield failed for %s: %s", ticker, exc)

        # ── Debt to equity ──
        try:
            bs = tk.quarterly_balance_sheet
            if bs is not None and not bs.empty:
                total_debt = None
                total_equity = None
                for label in ["Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation"]:
                    if label in bs.index:
                        total_debt = float(bs.loc[label].iloc[0])
                        break
                for label in ["Total Stockholders Equity", "Stockholders Equity", "Common Stock Equity"]:
                    if label in bs.index:
                        total_equity = float(bs.loc[label].iloc[0])
                        break
                features["debt_to_equity"] = safe_divide(total_debt, total_equity)
                if features["debt_to_equity"] is not None:
                    yf_success = True
            # Fallback to info dict
            if features["debt_to_equity"] is None:
                features["debt_to_equity"] = info.get("debtToEquity", None)
                if features["debt_to_equity"] is not None:
                    features["debt_to_equity"] = float(features["debt_to_equity"]) / 100
                    yf_success = True
        except Exception as exc:
            log.debug("D/E failed for %s: %s", ticker, exc)

    except Exception as exc:
        log.warning("yfinance fundamentals failed for %s: %s", ticker, exc)

    # ── Alpha Vantage OVERVIEW fallback ──
    if not yf_success:
        av_key = get_alpha_vantage_key()
        if av_key:
            try:
                _rate_limit(min_interval=3.0, source="alpha_vantage")
                url = (
                    f"https://www.alphavantage.co/query"
                    f"?function=OVERVIEW&symbol={ticker}&apikey={av_key}"
                )
                resp = req_lib.get(url, timeout=15)
                if resp.status_code == 200:
                    ov = resp.json()
                    # Map Alpha Vantage fields to our features
                    rev_growth = ov.get("QuarterlyRevenueGrowthYOY", ov.get("RevenueGrowthYOY", "None"))
                    if rev_growth != "None" and rev_growth is not None:
                        try:
                            features["revenue_growth_yoy"] = float(rev_growth)
                        except (TypeError, ValueError):
                            pass

                    gross_profit = ov.get("GrossProfitTTM", "None")
                    revenue_ttm = ov.get("RevenueTTM", "None")
                    if gross_profit != "None" and revenue_ttm != "None":
                        try:
                            gp = float(gross_profit)
                            rev = float(revenue_ttm)
                            features["gross_margin_trend"] = safe_divide(gp, rev)
                        except (TypeError, ValueError):
                            pass

                    de_ratio = ov.get("DebtToEquity", "None")
                    if de_ratio != "None" and de_ratio is not None:
                        try:
                            features["debt_to_equity"] = float(de_ratio) / 100
                        except (TypeError, ValueError):
                            pass

                    pe_ratio = ov.get("PERatio", "None")
                    eps = ov.get("EPS", "None")
                    market_cap = ov.get("MarketCapitalization", "None")
                    if eps != "None" and market_cap != "None":
                        try:
                            eps_val = float(eps)
                            mc = float(market_cap)
                            if mc > 0:
                                features["fcf_yield"] = safe_divide(eps_val, mc / 1e6)
                        except (TypeError, ValueError):
                            pass

                    log.info("Alpha Vantage OVERVIEW filled fundamentals for %s", ticker)
            except Exception as exc:
                log.warning("Alpha Vantage OVERVIEW failed for %s: %s", ticker, exc)

    return features


# ═══════════════════════════════════════════════════════════════════════════════
# SENTIMENT / INSIDER FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
@retry(max_retries=3, backoff_seconds=1.0)
def get_sentiment_features(ticker: str) -> Dict[str, Optional[float]]:
    """Compute sentiment and insider activity features.

    Uses yfinance for institutional holdings, insider transactions, and
    short interest. Optionally checks SEC EDGAR for recent 8-K filings.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dictionary with keys: ``insider_buy_sell_ratio``,
        ``institutional_ownership_change``, ``short_interest_ratio``,
        ``recent_8k_filing``.
    """
    features: Dict[str, Optional[float]] = {
        "insider_buy_sell_ratio": None,
        "institutional_ownership_change": None,
        "short_interest_ratio": None,
        "recent_8k_filing": 0.0,
    }

    try:
        _rate_limit(source="yfinance")
        tk = yf.Ticker(ticker, session=_yf_session)
        info = tk.info or {}

        # ── Short interest ratio ──
        features["short_interest_ratio"] = info.get("shortRatio", None)
        if features["short_interest_ratio"] is not None:
            features["short_interest_ratio"] = float(features["short_interest_ratio"])

        # ── Insider buy/sell ratio ──
        try:
            insiders = tk.insider_transactions
            if insiders is not None and isinstance(insiders, pd.DataFrame) and not insiders.empty:
                cutoff = datetime.now() - timedelta(days=90)
                # Filter to recent transactions
                if "Start Date" in insiders.columns:
                    insiders["Start Date"] = pd.to_datetime(insiders["Start Date"], errors="coerce")
                    recent = insiders[insiders["Start Date"] >= cutoff]
                else:
                    recent = insiders.head(10)  # Fallback: most recent 10

                if not recent.empty:
                    buys = 0
                    sells = 0
                    for _, row in recent.iterrows():
                        text = str(row.get("Transaction", row.get("Text", ""))).lower()
                        shares = abs(float(row.get("Shares", row.get("Value", 1))))
                        if "purchase" in text or "buy" in text or "acquisition" in text:
                            buys += shares
                        elif "sale" in text or "sell" in text or "disposition" in text:
                            sells += shares
                    features["insider_buy_sell_ratio"] = safe_divide(buys, buys + sells)
        except Exception as exc:
            log.debug("Insider transactions failed for %s: %s", ticker, exc)

        # ── Institutional ownership change ──
        try:
            inst = tk.institutional_holders
            if inst is not None and isinstance(inst, pd.DataFrame) and not inst.empty:
                # Use pctHeld or calculate from shares
                holders_pct = info.get("heldPercentInstitutions", None)
                if holders_pct is not None:
                    features["institutional_ownership_change"] = float(holders_pct)
                else:
                    features["institutional_ownership_change"] = 0.0
        except Exception as exc:
            log.debug("Institutional ownership failed for %s: %s", ticker, exc)

        # ── 8-K filing check (SEC EDGAR) ──
        sec_key = get_sec_api_key()
        if sec_key:
            try:
                _rate_limit(min_interval=1.0, source="sec_edgar")
                end_dt = datetime.now().strftime("%Y-%m-%d")
                start_dt = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                url = (
                    f"https://efts.sec.gov/LATEST/search-index"
                    f"?q={ticker}&dateRange=custom"
                    f"&startdt={start_dt}&enddt={end_dt}&forms=8-K"
                )
                headers = {"User-Agent": "EarningsPulse/1.0 earningspulse@example.com"}
                resp = req_lib.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    hits = data.get("hits", {}).get("total", {}).get("value", 0)
                    features["recent_8k_filing"] = 1.0 if hits > 0 else 0.0
            except Exception as exc:
                log.debug("SEC EDGAR check failed for %s: %s", ticker, exc)

    except Exception as exc:
        log.warning("Sentiment features failed for %s: %s", ticker, exc)

    return features


# ═══════════════════════════════════════════════════════════════════════════════
# MACRO CONTEXT FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
@retry(max_retries=3, backoff_seconds=1.0)
def get_macro_features(
    date: Optional[datetime] = None,
    ticker: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """Fetch macroeconomic context features for a given date.

    Uses yfinance for VIX and treasury yields when FRED is unavailable.
    Computes sector momentum from SPDR sector ETFs.

    Args:
        date: Reference date for macro features (default: today).
        ticker: Optional ticker to determine sector for sector momentum.

    Returns:
        Dictionary with keys: ``yield_curve_spread``, ``vix_level``,
        ``sector_momentum``.
    """
    if date is None:
        date = datetime.now()

    features: Dict[str, Optional[float]] = {
        "yield_curve_spread": None,
        "vix_level": None,
        "sector_momentum": None,
    }

    end_date = pd.to_datetime(date)
    start_date = end_date - timedelta(days=60)

    # ── VIX level ──
    try:
        _rate_limit()
        vix = yf.download(
            "^VIX",
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
            session=_yf_session,
        )
        if not vix.empty:
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            features["vix_level"] = float(vix["Close"].iloc[-1])
    except Exception as exc:
        log.debug("VIX fetch failed: %s", exc)

    # ── Yield curve spread (10Y - 2Y) ──
    try:
        _rate_limit()
        tnx = yf.download(
            "^TNX",
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
            session=_yf_session,
        )
        _rate_limit()
        irx = yf.download(
            "^IRX",
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
            session=_yf_session,
        )
        if not tnx.empty and not irx.empty:
            if isinstance(tnx.columns, pd.MultiIndex):
                tnx.columns = tnx.columns.get_level_values(0)
            if isinstance(irx.columns, pd.MultiIndex):
                irx.columns = irx.columns.get_level_values(0)
            # ^TNX is 10-year yield, ^IRX is 13-week T-bill
            # Approximate 2Y as average of 10Y and 13-week
            ten_yr = float(tnx["Close"].iloc[-1])
            short_rate = float(irx["Close"].iloc[-1])
            features["yield_curve_spread"] = (ten_yr - short_rate) / 100  # Convert from % to decimal spread
    except Exception as exc:
        log.debug("Yield curve fetch failed: %s", exc)

    # ── Sector momentum ──
    if ticker:
        sector = SECTOR_MAP.get(ticker, None)
        if sector:
            etf = SECTOR_ETFS.get(sector, None)
            if etf:
                try:
                    _rate_limit()
                    etf_hist = yf.download(
                        etf,
                        start=(end_date - timedelta(days=35)).strftime("%Y-%m-%d"),
                        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                        progress=False,
                        auto_adjust=True,
                        session=_yf_session,
                    )
                    if not etf_hist.empty and len(etf_hist) >= 5:
                        if isinstance(etf_hist.columns, pd.MultiIndex):
                            etf_hist.columns = etf_hist.columns.get_level_values(0)
                        close = etf_hist["Close"]
                        features["sector_momentum"] = float(
                            (close.iloc[-1] / close.iloc[0]) - 1
                        )
                except Exception as exc:
                    log.debug("Sector momentum failed for %s (%s): %s", etf, sector, exc)

    return features


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYST ESTIMATE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
def get_analyst_features(
    ticker: str,
    earnings_history: Optional[pd.DataFrame] = None,
) -> Dict[str, Optional[float]]:
    """Compute analyst estimate features from earnings history.

    Derives beat streak, surprise magnitude, and dispersion metrics.

    Args:
        ticker: Stock ticker symbol.
        earnings_history: Pre-fetched earnings history DataFrame.
            If ``None``, fetches it automatically.

    Returns:
        Dictionary with keys: ``eps_surprise_last_q``, ``eps_beat_streak``,
        ``eps_revision_direction``, ``estimate_dispersion``.
    """
    features: Dict[str, Optional[float]] = {
        "eps_surprise_last_q": None,
        "eps_beat_streak": 0,
        "eps_revision_direction": None,
        "estimate_dispersion": None,
    }

    if earnings_history is None:
        earnings_history = get_earnings_history(ticker)

    if earnings_history is None or earnings_history.empty:
        return features

    # Sort chronologically (oldest first) for streak counting
    eh = earnings_history.sort_values("earnings_date", ascending=True).reset_index(drop=True)

    # ── EPS surprise last quarter ──
    if len(eh) >= 1:
        last = eh.iloc[-1]
        features["eps_surprise_last_q"] = float(last.get("surprise_pct", 0))

    # ── Beat streak ──
    streak = 0
    for i in range(len(eh) - 1, -1, -1):
        if eh.iloc[i].get("beat", 0) == 1:
            streak += 1
        else:
            break
    features["eps_beat_streak"] = min(streak, 8)

    # ── Revision direction (estimate trend) ──
    if len(eh) >= 2:
        estimates = eh["estimated_eps"].dropna().values
        if len(estimates) >= 2:
            # Positive = estimates trending up, negative = trending down
            recent_avg = float(np.mean(estimates[-2:]))
            older_avg = float(np.mean(estimates[:-2])) if len(estimates) > 2 else float(estimates[0])
            features["eps_revision_direction"] = safe_divide(
                recent_avg - older_avg, abs(older_avg) if older_avg != 0 else 1
            )

    # ── Estimate dispersion ──
    try:
        _rate_limit()
        tk = yf.Ticker(ticker, session=_yf_session)
        info = tk.info or {}
        n_analysts = info.get("numberOfAnalystOpinions", None)
        if n_analysts is not None and n_analysts > 0:
            # Use number of analysts as dispersion proxy (more analysts = less
            # dispersion typically). Normalize to 0-1 range.
            features["estimate_dispersion"] = safe_divide(1.0, float(n_analysts))
        # Also try to get mean/std from analyst estimates
        target_mean = info.get("targetMeanPrice", None)
        target_low = info.get("targetLowPrice", None)
        target_high = info.get("targetHighPrice", None)
        if target_mean and target_low and target_high:
            spread = (float(target_high) - float(target_low)) / float(target_mean)
            features["estimate_dispersion"] = spread
    except Exception as exc:
        log.debug("Analyst dispersion failed for %s: %s", ticker, exc)

    return features


# ═══════════════════════════════════════════════════════════════════════════════
# UPCOMING EARNINGS CALENDAR
# ═══════════════════════════════════════════════════════════════════════════════
def get_upcoming_earnings(
    days_ahead: int = 30,
    tickers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fetch the earnings calendar for the next N days.

    Scans across the S&P 500 universe (or a custom list) to find companies
    with upcoming earnings announcements.

    Args:
        days_ahead: How many days ahead to search (default 30).
        tickers: List of tickers to check. Defaults to ``SP500_TICKERS``.

    Returns:
        DataFrame with columns: ``[ticker, company_name, earnings_date,
        sector]`` sorted by earnings date.
    """
    if tickers is None:
        tickers = SP500_TICKERS

    # Check cache (short TTL since calendar changes)
    cached = load_from_cache("upcoming_earnings", str(days_ahead), ttl_hours=6)
    if cached is not None:
        return cached

    records: List[Dict[str, Any]] = []
    today = datetime.now()
    cutoff = today + timedelta(days=days_ahead)

    log.info("Scanning %d tickers for upcoming earnings (next %d days)", len(tickers), days_ahead)

    for ticker in tickers:
        try:
            _rate_limit()
            tk = yf.Ticker(ticker, session=_yf_session)

            # Try to get next earnings date from calendar
            cal = getattr(tk, "calendar", None)
            earnings_date = None

            if cal is not None:
                if isinstance(cal, pd.DataFrame) and not cal.empty:
                    if "Earnings Date" in cal.columns:
                        earnings_date = pd.to_datetime(cal["Earnings Date"].iloc[0])
                    elif "Earnings Date" in cal.index:
                        earnings_date = pd.to_datetime(cal.loc["Earnings Date"].iloc[0])
                elif isinstance(cal, dict):
                    ed = cal.get("Earnings Date", [])
                    if ed:
                        earnings_date = pd.to_datetime(ed[0]) if isinstance(ed, list) else pd.to_datetime(ed)

            if earnings_date is None:
                info = tk.info or {}
                # Some yfinance versions expose earningsTimestamp
                ts = info.get("mostRecentQuarter") or info.get("earningsTimestamp")
                if ts:
                    # earningsTimestamp is unix timestamp for next earnings
                    if isinstance(ts, (int, float)) and ts > time.time():
                        earnings_date = datetime.fromtimestamp(ts)

            if earnings_date and today <= earnings_date <= cutoff:
                info = tk.info or {}
                records.append({
                    "ticker": ticker,
                    "company_name": info.get("shortName", info.get("longName", ticker)),
                    "earnings_date": earnings_date,
                    "sector": SECTOR_MAP.get(ticker, "Unknown"),
                })
        except Exception as exc:
            log.debug("Calendar check failed for %s: %s", ticker, exc)
            continue

    df = pd.DataFrame(records)
    if not df.empty:
        df["earnings_date"] = pd.to_datetime(df["earnings_date"])
        df = df.sort_values("earnings_date").reset_index(drop=True)
        save_to_cache(df, "upcoming_earnings", str(days_ahead))
        log.info("Found %d upcoming earnings in the next %d days", len(df), days_ahead)
    else:
        log.info("No upcoming earnings found in the next %d days", days_ahead)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# BULK DATA FETCH (for training)
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_all_training_data(
    tickers: Optional[List[str]] = None,
    years: int = 3,
) -> pd.DataFrame:
    """Fetch all earnings history for a list of tickers (for model training).

    Iterates through each ticker, fetches earnings history, and concatenates
    into a single DataFrame.

    Args:
        tickers: List of ticker symbols. Defaults to ``SP500_TICKERS``.
        years: Years of history per ticker.

    Returns:
        Concatenated DataFrame of all earnings records across all tickers.
    """
    if tickers is None:
        tickers = SP500_TICKERS

    all_records: List[pd.DataFrame] = []
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        log.info("Fetching training data: %s (%d/%d)", ticker, i, total)
        try:
            df = get_earnings_history(ticker, years=years)
            if df is not None and not df.empty:
                all_records.append(df)
        except Exception as exc:
            log.warning("Skipping %s: %s", ticker, exc)
        time.sleep(random.uniform(2.5, 4.0))  # Rate limiting between tickers

    if not all_records:
        log.error("No training data fetched — check API connectivity")
        return pd.DataFrame()

    combined = pd.concat(all_records, ignore_index=True)
    log.info("Total training records: %d across %d tickers", len(combined), len(all_records))
    return combined
