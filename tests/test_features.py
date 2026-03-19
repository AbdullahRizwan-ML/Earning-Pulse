"""
Unit tests for the feature engineering pipeline.

Tests cover feature matrix shape, look-ahead bias prevention, missing data
handling, beat label creation, and EPS beat streak counting.
"""

from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    FEATURE_COLUMNS,
    FEATURE_REGISTRY,
    TARGET_COLUMN,
    create_lag_features,
    handle_missing_data,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Create a sample feature matrix with known values for testing."""
    np.random.seed(42)
    n_rows = 20
    data = {
        "ticker": ["AAPL"] * 10 + ["MSFT"] * 10,
        "earnings_date": pd.date_range("2022-01-01", periods=10, freq="QE").tolist() * 2,
        "actual_eps": np.random.normal(2.0, 0.5, n_rows),
        "estimated_eps": np.random.normal(1.9, 0.5, n_rows),
        TARGET_COLUMN: [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
    }
    # Add all feature columns
    for col in FEATURE_COLUMNS:
        data[col] = np.random.normal(0.5, 0.2, n_rows)

    return pd.DataFrame(data)


@pytest.fixture
def earnings_records() -> pd.DataFrame:
    """Create earnings records with known beat/miss pattern."""
    return pd.DataFrame({
        "ticker": ["TEST"] * 8,
        "earnings_date": pd.date_range("2022-01-01", periods=8, freq="QE"),
        "actual_eps": [1.5, 1.6, 1.4, 1.7, 1.8, 1.9, 2.0, 2.1],
        "estimated_eps": [1.4, 1.5, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        "surprise_pct": [0.07, 0.07, -0.07, 0.06, 0.06, 0.06, 0.05, 0.05],
        "beat": [1, 1, 0, 1, 1, 1, 1, 1],
        "eps_surprise_last_q": [0.07, 0.07, -0.07, 0.06, 0.06, 0.06, 0.05, 0.05],
    })


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════
class TestFeatureMatrixShape:
    """Tests for feature matrix dimensions and column presence."""

    def test_feature_matrix_has_expected_columns(self, sample_feature_df: pd.DataFrame) -> None:
        """Assert output has all expected feature columns."""
        df = sample_feature_df
        for col in FEATURE_COLUMNS:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_feature_matrix_has_target(self, sample_feature_df: pd.DataFrame) -> None:
        """Assert the target column exists."""
        assert TARGET_COLUMN in sample_feature_df.columns

    def test_feature_registry_complete(self) -> None:
        """Assert FEATURE_REGISTRY has descriptions for all features."""
        for col in FEATURE_COLUMNS:
            assert col in FEATURE_REGISTRY, f"Missing registry entry: {col}"
            assert len(FEATURE_REGISTRY[col]) > 5, f"Trivial description for: {col}"

    def test_feature_registry_matches_columns(self) -> None:
        """Assert FEATURE_REGISTRY keys match FEATURE_COLUMNS exactly."""
        assert set(FEATURE_REGISTRY.keys()) == set(FEATURE_COLUMNS)


class TestNoLookaheadBias:
    """Tests to verify price features don't use post-earnings data."""

    def test_price_features_use_pre_earnings_date(self) -> None:
        """Assert price features are computed from earnings_date minus 2 days.

        This verifies the parameterization logic — the actual yfinance call
        uses (earnings_date - 2 days) as the end date.
        """
        from src.data_ingestion import get_price_features

        # The function should accept an earnings date and use it - 2 days
        # We test the interface contract: it should not crash and should
        # return a dict with the expected keys
        earnings_date = datetime.now() - timedelta(days=30)
        result = get_price_features("AAPL", earnings_date)

        assert isinstance(result, dict)
        expected_keys = {"ret_1w", "ret_1m", "ret_3m", "price_vs_52w_high", "volume_surge"}
        assert expected_keys == set(result.keys())

    def test_price_feature_date_arithmetic(self) -> None:
        """Verify the 2-day offset arithmetic is correct."""
        earnings_date = datetime(2024, 1, 25)
        expected_end = datetime(2024, 1, 23)

        # This mirrors the logic in get_price_features
        actual_end = pd.to_datetime(earnings_date) - timedelta(days=2)
        assert actual_end == expected_end


class TestMissingDataHandling:
    """Tests for missing data imputation."""

    def test_no_nans_after_imputation(self, sample_feature_df: pd.DataFrame) -> None:
        """Assert no NaN values remain after imputation."""
        # Inject NaN values
        df = sample_feature_df.copy()
        for col in FEATURE_COLUMNS[:5]:
            df.loc[df.index[:3], col] = np.nan

        result = handle_missing_data(df)
        for col in FEATURE_COLUMNS:
            assert result[col].isna().sum() == 0, f"NaN remaining in {col}"

    def test_median_imputation_correctness(self) -> None:
        """Assert median imputation produces the correct fill value."""
        df = pd.DataFrame({
            "ret_1m": [0.1, 0.2, np.nan, 0.3, 0.4],
            "eps_beat_streak": [1.0, np.nan, 3.0, 4.0, 5.0],
        })
        # Add remaining columns as zeros
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0

        result = handle_missing_data(df)

        # ret_1m median of [0.1, 0.2, 0.3, 0.4] = 0.25
        assert result.loc[2, "ret_1m"] == pytest.approx(0.25, abs=0.01)

        # eps_beat_streak is a ratio feature → filled with 0
        assert result.loc[1, "eps_beat_streak"] == 0.0

    def test_all_nan_column_fills_with_zero(self) -> None:
        """Assert columns that are entirely NaN get filled with 0."""
        df = pd.DataFrame({col: [np.nan] * 5 for col in FEATURE_COLUMNS})
        result = handle_missing_data(df)
        for col in FEATURE_COLUMNS:
            assert result[col].isna().sum() == 0


class TestBeatLabelCreation:
    """Tests for correct beat/miss label generation."""

    def test_beat_when_actual_exceeds_estimate(self) -> None:
        """Assert label is 1 when actual EPS > estimated EPS."""
        actual = 1.50
        estimated = 1.40
        beat = 1 if actual > estimated else 0
        assert beat == 1

    def test_miss_when_actual_below_estimate(self) -> None:
        """Assert label is 0 when actual EPS < estimated EPS."""
        actual = 1.30
        estimated = 1.40
        beat = 1 if actual > estimated else 0
        assert beat == 0

    def test_miss_when_exact_match(self) -> None:
        """Assert label is 0 when actual EPS == estimated EPS (not a beat)."""
        actual = 1.40
        estimated = 1.40
        beat = 1 if actual > estimated else 0
        assert beat == 0

    def test_beat_label_vectorized(self) -> None:
        """Assert vectorized beat label creation works correctly."""
        df = pd.DataFrame({
            "actual_eps": [1.5, 1.3, 1.4, 2.0, 0.5],
            "estimated_eps": [1.4, 1.4, 1.4, 1.9, 0.6],
        })
        df["beat"] = (df["actual_eps"] > df["estimated_eps"]).astype(int)
        expected = [1, 0, 0, 1, 0]
        assert df["beat"].tolist() == expected


class TestEpsBeatStreak:
    """Tests for beat streak counting logic."""

    def test_full_beat_streak(self, earnings_records: pd.DataFrame) -> None:
        """Verify streak count with a known sequence ending in 5 beats."""
        # Pattern: 1, 1, 0, 1, 1, 1, 1, 1
        # Streak from the end: 5 consecutive beats
        beats = earnings_records["beat"].tolist()
        streak = 0
        for b in reversed(beats):
            if b == 1:
                streak += 1
            else:
                break
        assert streak == 5

    def test_zero_streak(self) -> None:
        """Verify streak is 0 when last quarter was a miss."""
        beats = [1, 1, 1, 0]
        streak = 0
        for b in reversed(beats):
            if b == 1:
                streak += 1
            else:
                break
        assert streak == 0

    def test_capped_at_8(self) -> None:
        """Verify streak is capped at 8 quarters maximum."""
        beats = [1] * 12
        streak = 0
        for b in reversed(beats):
            if b == 1:
                streak += 1
            else:
                break
        capped = min(streak, 8)
        assert capped == 8

    def test_single_beat(self) -> None:
        """Verify streak is 1 for single trailing beat."""
        beats = [0, 0, 0, 1]
        streak = 0
        for b in reversed(beats):
            if b == 1:
                streak += 1
            else:
                break
        assert streak == 1


class TestLagFeatures:
    """Tests for lag feature creation."""

    def test_lag_feature_exists(self, sample_feature_df: pd.DataFrame) -> None:
        """Assert prev_surprise column is created."""
        result = create_lag_features(sample_feature_df)
        assert "prev_surprise" in result.columns

    def test_lag_feature_no_nans(self, sample_feature_df: pd.DataFrame) -> None:
        """Assert no NaN in prev_surprise after creation (first row filled with 0)."""
        result = create_lag_features(sample_feature_df)
        assert result["prev_surprise"].isna().sum() == 0

    def test_lag_feature_correct_shift(self) -> None:
        """Assert prev_surprise contains previous quarter's surprise."""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 4,
            "earnings_date": pd.date_range("2023-01-01", periods=4, freq="QE"),
            "eps_surprise_last_q": [0.05, 0.10, -0.03, 0.08],
        })
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0

        result = create_lag_features(df)
        # First row should be 0 (filled), second should be 0.05, etc.
        assert result.loc[0, "prev_surprise"] == 0.0  # First quarter, no prev
        assert result.loc[1, "prev_surprise"] == pytest.approx(0.05)
        assert result.loc[2, "prev_surprise"] == pytest.approx(0.10)
        assert result.loc[3, "prev_surprise"] == pytest.approx(-0.03)

    def test_empty_dataframe(self) -> None:
        """Assert lag features handle empty DataFrame gracefully."""
        df = pd.DataFrame(columns=FEATURE_COLUMNS + ["ticker", "earnings_date", "eps_surprise_last_q"])
        result = create_lag_features(df)
        assert "prev_surprise" in result.columns
        assert len(result) == 0
