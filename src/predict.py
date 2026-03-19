"""
Inference pipeline for EarningsPulse.

Provides functions to generate beat/miss probability predictions for
individual stocks and batch predictions for the earnings watchlist.
Uses the trained ensemble models for scoring.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.feature_engineering import (
    FEATURE_COLUMNS,
    build_single_ticker_features,
    handle_missing_data,
)
from src.model import load_models
from src.utils import DATA_DIR, SECTOR_MAP, setup_logger

log = setup_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE TICKER PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
def predict_single(
    ticker: str,
    xgb_model: Optional[Any] = None,
    lgbm_model: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate a beat/miss probability prediction for a single stock.

    Fetches live features, runs them through both models, and returns
    the ensemble probability along with a feature breakdown.

    Args:
        ticker: Stock ticker symbol (e.g., ``"AAPL"``).
        xgb_model: Pre-loaded XGBoost model. If ``None``, loads from disk.
        lgbm_model: Pre-loaded LightGBM model. If ``None``, loads from disk.
        feature_names: Feature column names. If ``None``, loads from metadata.

    Returns:
        Dictionary with keys:
        - ``ticker``: The ticker symbol.
        - ``beat_probability``: Ensemble probability (0–1).
        - ``prediction``: ``"BEAT"`` or ``"MISS"``.
        - ``confidence``: Confidence level (``"High"``, ``"Medium"``, ``"Low"``).
        - ``features``: Dict of feature name → value.
        - ``feature_impacts``: Dict of feature name → impact direction.
        - ``earnings_history``: Recent earnings history DataFrame.
        - ``info``: Additional info (P/E, sector, etc.).
    """
    result: Dict[str, Any] = {
        "ticker": ticker.upper(),
        "beat_probability": 0.5,
        "prediction": "UNCERTAIN",
        "confidence": "Low",
        "features": {},
        "feature_impacts": {},
        "earnings_history": pd.DataFrame(),
        "info": {},
        "error": None,
    }

    try:
        # Load models if not provided
        if xgb_model is None or lgbm_model is None:
            xgb_model, lgbm_model, _, feature_names = load_models()

        if not feature_names:
            feature_names = FEATURE_COLUMNS + ["prev_surprise"]

        # Build features
        raw_features = build_single_ticker_features(ticker)

        # Extract earnings history before converting to model input
        earnings_hist = raw_features.pop("earnings_history", pd.DataFrame())
        result["earnings_history"] = earnings_hist

        # Build feature vector in correct order
        feature_vector = {}
        for fname in feature_names:
            val = raw_features.get(fname, None)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                feature_vector[fname] = 0.0
            else:
                feature_vector[fname] = float(val)

        result["features"] = feature_vector

        # Create DataFrame for prediction
        X = pd.DataFrame([feature_vector])
        X = X[feature_names]  # Ensure correct column order
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Predict with both models
        xgb_proba = float(xgb_model.predict_proba(X)[:, 1][0])
        lgbm_proba = float(lgbm_model.predict_proba(X)[:, 1][0])
        ensemble_proba = (xgb_proba + lgbm_proba) / 2

        result["beat_probability"] = ensemble_proba
        result["xgb_probability"] = xgb_proba
        result["lgbm_probability"] = lgbm_proba
        result["prediction"] = "BEAT" if ensemble_proba >= 0.5 else "MISS"

        # Confidence level
        distance_from_05 = abs(ensemble_proba - 0.5)
        if distance_from_05 > 0.25:
            result["confidence"] = "High"
        elif distance_from_05 > 0.10:
            result["confidence"] = "Medium"
        else:
            result["confidence"] = "Low"

        # Feature impacts (positive = pushes toward beat)
        result["feature_impacts"] = _compute_feature_impacts(
            xgb_model, feature_names, feature_vector
        )

        # Additional info
        result["info"] = {
            "sector": SECTOR_MAP.get(ticker.upper(), "Unknown"),
            "model_agreement": "Yes" if (xgb_proba >= 0.5) == (lgbm_proba >= 0.5) else "No",
        }

        log.info(
            "Prediction for %s: %.1f%% beat probability (%s confidence)",
            ticker,
            ensemble_proba * 100,
            result["confidence"],
        )

    except FileNotFoundError:
        result["error"] = "Models not trained yet. Run 'python -m src.model' first."
        log.error(result["error"])
    except Exception as exc:
        result["error"] = f"Prediction failed: {str(exc)}"
        log.error("Prediction failed for %s: %s", ticker, exc)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH PREDICTIONS (for watchlist)
# ═══════════════════════════════════════════════════════════════════════════════
def predict_watchlist(
    upcoming_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate beat probabilities for all stocks in the upcoming earnings list.

    Args:
        upcoming_df: DataFrame with at least a ``ticker`` column from
            ``get_upcoming_earnings()``.

    Returns:
        The input DataFrame augmented with ``beat_probability``,
        ``confidence``, and ``prediction`` columns.
    """
    if upcoming_df.empty:
        log.info("No upcoming earnings to predict")
        return upcoming_df

    try:
        xgb_model, lgbm_model, _, feature_names = load_models()
    except FileNotFoundError:
        log.error("Models not found — cannot generate watchlist predictions")
        upcoming_df["beat_probability"] = 0.5
        upcoming_df["confidence"] = "N/A"
        upcoming_df["prediction"] = "UNKNOWN"
        return upcoming_df

    predictions: List[Dict[str, Any]] = []

    for _, row in upcoming_df.iterrows():
        ticker = row["ticker"]
        try:
            pred = predict_single(
                ticker,
                xgb_model=xgb_model,
                lgbm_model=lgbm_model,
                feature_names=feature_names,
            )
            predictions.append({
                "ticker": ticker,
                "beat_probability": pred["beat_probability"],
                "confidence": pred["confidence"],
                "prediction": pred["prediction"],
            })
        except Exception as exc:
            log.warning("Watchlist prediction failed for %s: %s", ticker, exc)
            predictions.append({
                "ticker": ticker,
                "beat_probability": 0.5,
                "confidence": "N/A",
                "prediction": "UNKNOWN",
            })

    pred_df = pd.DataFrame(predictions)
    result = upcoming_df.merge(pred_df, on="ticker", how="left")

    # Fill any missing predictions
    result["beat_probability"] = result["beat_probability"].fillna(0.5)
    result["confidence"] = result["confidence"].fillna("N/A")
    result["prediction"] = result["prediction"].fillna("UNKNOWN")

    # Sort by probability descending
    result = result.sort_values("beat_probability", ascending=False).reset_index(drop=True)

    log.info("Generated predictions for %d watchlist stocks", len(result))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPACT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def _compute_feature_impacts(
    model: Any,
    feature_names: List[str],
    feature_values: Dict[str, float],
) -> Dict[str, str]:
    """Determine directional impact of each feature on the prediction.

    Uses the model's feature importances combined with the sign of each
    feature value relative to the training distribution to estimate
    whether each feature pushes toward BEAT or MISS.

    Args:
        model: Trained model with ``feature_importances_`` attribute.
        feature_names: List of feature names.
        feature_values: Current feature values for the prediction.

    Returns:
        Dictionary mapping feature names to impact strings
        (``"↑ Bullish"``, ``"↓ Bearish"``, or ``"→ Neutral"``).
    """
    impacts: Dict[str, str] = {}

    try:
        importances = model.feature_importances_
        imp_dict = dict(zip(feature_names, importances))
    except Exception:
        imp_dict = {f: 1.0 / len(feature_names) for f in feature_names}

    # Features where higher = more likely to beat
    bullish_when_high = {
        "eps_surprise_last_q", "eps_beat_streak", "eps_revision_direction",
        "ret_1m", "ret_3m", "ret_1w", "price_vs_52w_high",
        "revenue_growth_yoy", "gross_margin_trend", "fcf_yield",
        "insider_buy_sell_ratio", "sector_momentum", "prev_surprise",
    }

    # Features where higher = more likely to miss
    bearish_when_high = {
        "estimate_dispersion", "volume_surge", "debt_to_equity",
        "short_interest_ratio", "vix_level",
    }

    for fname in feature_names:
        val = feature_values.get(fname, 0)
        imp = imp_dict.get(fname, 0)

        if imp < 0.01:
            impacts[fname] = "→ Neutral"
        elif fname in bullish_when_high:
            impacts[fname] = "↑ Bullish" if val > 0 else "↓ Bearish"
        elif fname in bearish_when_high:
            impacts[fname] = "↓ Bearish" if val > 0 else "↑ Bullish"
        else:
            impacts[fname] = "→ Neutral"

    return impacts


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: Quick prediction (used by dashboard)
# ═══════════════════════════════════════════════════════════════════════════════
def quick_predict(ticker: str) -> Tuple[float, str, str]:
    """Fast prediction returning just the essentials.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Tuple of ``(probability, prediction_label, confidence_level)``.
    """
    result = predict_single(ticker)
    return (
        result["beat_probability"],
        result["prediction"],
        result["confidence"],
    )
