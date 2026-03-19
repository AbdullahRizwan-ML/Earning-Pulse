"""
Utility module for EarningsPulse.

Provides centralized logging configuration, caching helpers, retry decorators,
and shared constants used across the entire application.
"""

import functools
import hashlib
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

# ─── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# ─── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─── S&P 500 Ticker Universe ───────────────────────────────────────────────────
SP500_TICKERS: List[str] = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    # Financials
    "JPM", "BAC", "GS", "MS", "WFC",
    # Healthcare
    "JNJ", "PFE", "UNH", "ABBV", "MRK",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Consumer Discretionary
    "TSLA", "GM", "F", "TM", "HMC",
    # Consumer Staples
    "WMT", "COST", "PG", "KO", "PEP",
    # Industrials
    "CAT", "GE", "HON", "UPS", "BA",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP",
    # Real Estate
    "AMT", "PLD", "CCI", "EQIX", "SPG",
    # Materials
    "LIN", "APD", "SHW", "ECL", "NEM",
]

SECTOR_MAP: Dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Technology", "META": "Technology",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "MS": "Financials", "WFC": "Financials",
    "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy",
    "TSLA": "Consumer Discretionary", "GM": "Consumer Discretionary",
    "F": "Consumer Discretionary", "TM": "Consumer Discretionary",
    "HMC": "Consumer Discretionary",
    "WMT": "Consumer Staples", "COST": "Consumer Staples",
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "CAT": "Industrials", "GE": "Industrials", "HON": "Industrials",
    "UPS": "Industrials", "BA": "Industrials",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D": "Utilities", "AEP": "Utilities",
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    "EQIX": "Real Estate", "SPG": "Real Estate",
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    "ECL": "Materials", "NEM": "Materials",
}

# Sector ETF tickers (SPDR) for sector momentum calculation
SECTOR_ETFS: Dict[str, str] = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}


# ─── Logging Setup ─────────────────────────────────────────────────────────────
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger with consistent formatting.

    Args:
        name: The logger name, typically ``__name__`` of the calling module.
        level: Logging level (default ``logging.INFO``).

    Returns:
        Configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = setup_logger("earningspulse")


# ─── Retry Decorator ───────────────────────────────────────────────────────────
def retry(
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """Decorator that retries a function on failure with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        backoff_seconds: Initial wait time between retries (doubles each attempt).
        exceptions: Tuple of exception types to catch and retry on.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt < max_retries:
                        wait = backoff_seconds * (2 ** (attempt - 1))
                        logger.warning(
                            "%s attempt %d/%d failed: %s — retrying in %.1fs",
                            func.__name__,
                            attempt,
                            max_retries,
                            exc,
                            wait,
                        )
                        time.sleep(wait)
                    else:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__,
                            max_retries,
                            exc,
                        )
            return None  # Graceful fallback

        return wrapper

    return decorator


# ─── Parquet Cache Helpers ──────────────────────────────────────────────────────
def _cache_key(prefix: str, *identifiers: str) -> str:
    """Generate a deterministic cache filename from identifiers.

    Args:
        prefix: Cache category (e.g. ``"earnings"``, ``"price"``).
        *identifiers: Variable number of string identifiers to hash.

    Returns:
        A filename-safe cache key string.
    """
    raw = "_".join(str(i) for i in identifiers)
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
    safe_name = raw.replace("/", "_").replace("\\", "_").replace(":", "_")[:60]
    return f"{prefix}_{safe_name}_{short_hash}"


def save_to_cache(
    df: pd.DataFrame,
    prefix: str,
    *identifiers: str,
    ttl_hours: int = 24,
) -> Path:
    """Save a DataFrame to the data/ cache directory as a parquet file.

    Args:
        df: The DataFrame to cache.
        prefix: Cache category prefix for the filename.
        *identifiers: Variable identifiers used to generate the cache key.
        ttl_hours: Time-to-live in hours (metadata only, checked on load).

    Returns:
        The ``Path`` to the saved parquet file.
    """
    key = _cache_key(prefix, *identifiers)
    filepath = DATA_DIR / f"{key}.parquet"
    df.to_parquet(filepath, index=False)
    # Save metadata for TTL tracking
    meta_path = DATA_DIR / f"{key}.meta"
    meta_path.write_text(
        f"created={datetime.utcnow().isoformat()}\nttl_hours={ttl_hours}\n"
    )
    logger.debug("Cached %d rows to %s", len(df), filepath.name)
    return filepath


def load_from_cache(
    prefix: str,
    *identifiers: str,
    ttl_hours: int = 24,
) -> Optional[pd.DataFrame]:
    """Load a cached DataFrame if it exists and is within TTL.

    Args:
        prefix: Cache category prefix.
        *identifiers: Variable identifiers matching the save call.
        ttl_hours: Maximum age in hours before the cache is considered stale.

    Returns:
        The cached ``pd.DataFrame`` or ``None`` if missing/stale.
    """
    key = _cache_key(prefix, *identifiers)
    filepath = DATA_DIR / f"{key}.parquet"
    meta_path = DATA_DIR / f"{key}.meta"

    if not filepath.exists():
        return None

    # Check TTL
    if meta_path.exists():
        try:
            meta_text = meta_path.read_text()
            for line in meta_text.strip().split("\n"):
                if line.startswith("created="):
                    created = datetime.fromisoformat(line.split("=", 1)[1])
                    age_hours = (datetime.utcnow() - created).total_seconds() / 3600
                    if age_hours > ttl_hours:
                        logger.debug("Cache expired for %s (%.1fh old)", key, age_hours)
                        return None
        except (ValueError, IndexError):
            pass  # If metadata is corrupt, use the cache anyway

    try:
        df = pd.read_parquet(filepath)
        logger.debug("Cache hit for %s (%d rows)", key, len(df))
        return df
    except Exception as exc:
        logger.warning("Failed to read cache %s: %s", filepath.name, exc)
        return None


def clear_cache(prefix: Optional[str] = None) -> int:
    """Remove cached parquet files from the data directory.

    Args:
        prefix: If provided, only clear files matching this prefix.
            If ``None``, clear all cached parquet files.

    Returns:
        Number of files removed.
    """
    removed = 0
    for f in DATA_DIR.glob("*.parquet"):
        if prefix is None or f.stem.startswith(prefix):
            f.unlink(missing_ok=True)
            meta = f.with_suffix(".meta")
            meta.unlink(missing_ok=True)
            removed += 1
    logger.info("Cleared %d cached files (prefix=%s)", removed, prefix)
    return removed


# ─── API Key Helpers ────────────────────────────────────────────────────────────
def get_alpha_vantage_key() -> Optional[str]:
    """Retrieve the Alpha Vantage API key from environment variables.

    Returns:
        The API key string, or ``None`` if not configured.
    """
    key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if key and key != "your_key_here":
        return key
    return None


def get_sec_api_key() -> Optional[str]:
    """Retrieve the SEC API key from environment variables.

    Returns:
        The API key string, or ``None`` if not configured.
    """
    key = os.getenv("SEC_API_KEY")
    if key and key != "your_key_here":
        return key
    return None


def get_fred_api_key() -> Optional[str]:
    """Retrieve the FRED API key from environment variables.

    Returns:
        The API key string, or ``None`` if not configured.
    """
    key = os.getenv("FRED_API_KEY")
    if key and key != "your_key_here":
        return key
    return None


# ─── Demo Mode ──────────────────────────────────────────────────────────────────
def is_demo_mode() -> bool:
    """Check whether the application should run in demo mode.

    Demo mode is activated when the environment variable ``DEMO_MODE=1``
    is set, or when pre-saved sample data exists in the data directory
    and no internet connection is detected.

    Returns:
        ``True`` if the app should use sample data, ``False`` otherwise.
    """
    if os.getenv("DEMO_MODE", "0") == "1":
        return True
    sample_file = DATA_DIR / "demo_feature_matrix.parquet"
    if sample_file.exists():
        # Quick connectivity check
        try:
            import socket
            socket.create_connection(("finance.yahoo.com", 443), timeout=3)
            return False
        except OSError:
            logger.info("No internet connection detected — entering DEMO MODE")
            return True
    return False


# ─── Formatting Helpers ─────────────────────────────────────────────────────────
def format_pct(value: Optional[float], decimals: int = 1) -> str:
    """Format a decimal ratio as a percentage string.

    Args:
        value: The raw decimal value (e.g., ``0.053`` → ``"5.3%"``).
        decimals: Number of decimal places in the output.

    Returns:
        Human-readable percentage string, or ``"N/A"`` if value is ``None``.
    """
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: Optional[float], decimals: int = 2) -> str:
    """Format a number as a currency string.

    Args:
        value: The numeric value to format.
        decimals: Number of decimal places.

    Returns:
        Formatted string like ``"$1,234.56"``, or ``"N/A"`` if ``None``.
    """
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.{decimals}f}"


def safe_divide(
    numerator: Optional[float],
    denominator: Optional[float],
    default: float = 0.0,
) -> float:
    """Safely divide two numbers, returning a default on failure.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value to return if division is impossible.

    Returns:
        Result of division, or ``default`` if denominator is zero/None.
    """
    try:
        if numerator is None or denominator is None:
            return default
        if pd.isna(numerator) or pd.isna(denominator):
            return default
        if denominator == 0:
            return default
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return default
