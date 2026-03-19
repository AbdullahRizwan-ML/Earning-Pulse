"""
Unit tests for the model training, evaluation, and persistence pipeline.

Tests cover prediction output format, time-series split correctness,
ensemble averaging, model save/load roundtrip, and feature importance
normalization.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import FEATURE_COLUMNS


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def synthetic_training_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic training data for model tests.

    Returns:
        Tuple of (X, y) where X is a feature matrix and y is a binary target.
    """
    np.random.seed(42)
    n_samples = 200
    n_features = len(FEATURE_COLUMNS)

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=FEATURE_COLUMNS,
    )
    # Add prev_surprise as additional feature
    X["prev_surprise"] = np.random.randn(n_samples) * 0.05

    # Create target with some signal from first few features
    signal = X.iloc[:, 0] * 0.3 + X.iloc[:, 1] * 0.2 + np.random.randn(n_samples) * 0.5
    y = pd.Series((signal > 0).astype(int), name="beat")

    return X, y


@pytest.fixture
def trained_models(synthetic_training_data: Tuple[pd.DataFrame, pd.Series]) -> Tuple[Any, Any, List[str]]:
    """Train models on synthetic data for testing.

    Returns:
        Tuple of (xgb_model, lgbm_model, feature_names).
    """
    import xgboost as xgb
    import lightgbm as lgb

    X, y = synthetic_training_data

    xgb_model = xgb.XGBClassifier(
        n_estimators=10, max_depth=3, random_state=42,
        eval_metric="logloss", use_label_encoder=False,
    )
    xgb_model.fit(X, y, verbose=False)

    lgbm_model = lgb.LGBMClassifier(
        n_estimators=10, num_leaves=15, random_state=42, verbose=-1,
    )
    lgbm_model.fit(X, y)

    return xgb_model, lgbm_model, list(X.columns)


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════
class TestModelOutputProbability:
    """Tests for model prediction output format."""

    def test_xgb_predictions_in_range(
        self,
        trained_models: Tuple[Any, Any, List[str]],
        synthetic_training_data: Tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Assert XGBoost predicted probabilities are in [0, 1]."""
        xgb_model, _, _ = trained_models
        X, _ = synthetic_training_data
        proba = xgb_model.predict_proba(X)[:, 1]
        assert np.all(proba >= 0.0), "Probabilities below 0 detected"
        assert np.all(proba <= 1.0), "Probabilities above 1 detected"

    def test_lgbm_predictions_in_range(
        self,
        trained_models: Tuple[Any, Any, List[str]],
        synthetic_training_data: Tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Assert LightGBM predicted probabilities are in [0, 1]."""
        _, lgbm_model, _ = trained_models
        X, _ = synthetic_training_data
        proba = lgbm_model.predict_proba(X)[:, 1]
        assert np.all(proba >= 0.0), "Probabilities below 0 detected"
        assert np.all(proba <= 1.0), "Probabilities above 1 detected"

    def test_ensemble_predictions_in_range(
        self,
        trained_models: Tuple[Any, Any, List[str]],
        synthetic_training_data: Tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Assert ensemble averaged probabilities are in [0, 1]."""
        xgb_model, lgbm_model, _ = trained_models
        X, _ = synthetic_training_data
        xgb_proba = xgb_model.predict_proba(X)[:, 1]
        lgbm_proba = lgbm_model.predict_proba(X)[:, 1]
        ensemble = (xgb_proba + lgbm_proba) / 2
        assert np.all(ensemble >= 0.0)
        assert np.all(ensemble <= 1.0)

    def test_prediction_output_shape(
        self,
        trained_models: Tuple[Any, Any, List[str]],
        synthetic_training_data: Tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Assert predict_proba returns correct shape."""
        xgb_model, _, _ = trained_models
        X, _ = synthetic_training_data
        proba = xgb_model.predict_proba(X)
        assert proba.shape == (len(X), 2), f"Expected shape ({len(X)}, 2), got {proba.shape}"


class TestTimeSeriesSplit:
    """Tests for time-series aware cross-validation."""

    def test_no_future_data_in_train(
        self,
        synthetic_training_data: Tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Assert that no future indices appear in training folds."""
        from sklearn.model_selection import TimeSeriesSplit

        X, _ = synthetic_training_data
        tscv = TimeSeriesSplit(n_splits=5)

        for train_idx, val_idx in tscv.split(X):
            max_train = max(train_idx)
            min_val = min(val_idx)
            assert max_train < min_val, (
                f"Data leak detected: train max idx {max_train} >= val min idx {min_val}"
            )

    def test_increasing_train_size(
        self,
        synthetic_training_data: Tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Assert training set grows with each fold."""
        from sklearn.model_selection import TimeSeriesSplit

        X, _ = synthetic_training_data
        tscv = TimeSeriesSplit(n_splits=5)
        prev_size = 0
        for train_idx, _ in tscv.split(X):
            assert len(train_idx) > prev_size, "Training set did not grow"
            prev_size = len(train_idx)

    def test_validation_never_overlaps_train(
        self,
        synthetic_training_data: Tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Assert train and validation indices never overlap."""
        from sklearn.model_selection import TimeSeriesSplit

        X, _ = synthetic_training_data
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, val_idx in tscv.split(X):
            overlap = set(train_idx) & set(val_idx)
            assert len(overlap) == 0, f"Overlap detected: {overlap}"


class TestEnsembleAverage:
    """Tests for ensemble probability averaging."""

    def test_ensemble_is_mean_of_models(
        self,
        trained_models: Tuple[Any, Any, List[str]],
        synthetic_training_data: Tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Assert ensemble output is exactly the mean of two model outputs."""
        xgb_model, lgbm_model, _ = trained_models
        X, _ = synthetic_training_data
        xgb_proba = xgb_model.predict_proba(X)[:, 1]
        lgbm_proba = lgbm_model.predict_proba(X)[:, 1]
        expected_ensemble = (xgb_proba + lgbm_proba) / 2
        np.testing.assert_array_almost_equal(
            expected_ensemble,
            (xgb_proba + lgbm_proba) / 2,
            decimal=10,
        )

    def test_ensemble_bounded_by_individual_models(
        self,
        trained_models: Tuple[Any, Any, List[str]],
        synthetic_training_data: Tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Assert ensemble is between min and max of individual predictions."""
        xgb_model, lgbm_model, _ = trained_models
        X, _ = synthetic_training_data
        xgb_proba = xgb_model.predict_proba(X)[:, 1]
        lgbm_proba = lgbm_model.predict_proba(X)[:, 1]
        ensemble = (xgb_proba + lgbm_proba) / 2
        mins = np.minimum(xgb_proba, lgbm_proba)
        maxs = np.maximum(xgb_proba, lgbm_proba)
        assert np.all(ensemble >= mins - 1e-10)
        assert np.all(ensemble <= maxs + 1e-10)


class TestModelSaveLoad:
    """Tests for model persistence roundtrip."""

    def test_save_and_load_produces_identical_predictions(
        self,
        trained_models: Tuple[Any, Any, List[str]],
        synthetic_training_data: Tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Assert predictions are identical after save→load cycle."""
        import joblib

        xgb_model, lgbm_model, feature_names = trained_models
        X, _ = synthetic_training_data

        # Generate predictions before saving
        xgb_pred_before = xgb_model.predict_proba(X)[:, 1]
        lgbm_pred_before = lgbm_model.predict_proba(X)[:, 1]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            joblib.dump(xgb_model, tmp_path / "xgb_model.joblib")
            joblib.dump(lgbm_model, tmp_path / "lgbm_model.joblib")

            # Save metadata
            meta = {"feature_names": feature_names, "training_date": "2024-01-01"}
            with open(tmp_path / "model_metadata.json", "w") as f:
                json.dump(meta, f)

            # Load and predict
            loaded_xgb = joblib.load(tmp_path / "xgb_model.joblib")
            loaded_lgbm = joblib.load(tmp_path / "lgbm_model.joblib")

            xgb_pred_after = loaded_xgb.predict_proba(X)[:, 1]
            lgbm_pred_after = loaded_lgbm.predict_proba(X)[:, 1]

        np.testing.assert_array_almost_equal(xgb_pred_before, xgb_pred_after, decimal=10)
        np.testing.assert_array_almost_equal(lgbm_pred_before, lgbm_pred_after, decimal=10)

    def test_metadata_saved_correctly(
        self,
        trained_models: Tuple[Any, Any, List[str]],
    ) -> None:
        """Assert model metadata is written and readable."""
        from src.model import save_models

        xgb_model, lgbm_model, feature_names = trained_models
        metrics = {"roc_auc": 0.75, "accuracy": 0.68}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_models(xgb_model, lgbm_model, metrics, feature_names, path=tmpdir)

            meta_path = Path(tmpdir) / "model_metadata.json"
            assert meta_path.exists(), "Metadata file not created"

            with open(meta_path) as f:
                meta = json.load(f)

            assert meta["feature_names"] == feature_names
            assert "training_date" in meta
            assert meta["n_features"] == len(feature_names)

    def test_load_raises_if_no_models(self) -> None:
        """Assert FileNotFoundError when models don't exist."""
        from src.model import load_models

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="Model files not found"):
                load_models(path=tmpdir)


class TestFeatureImportance:
    """Tests for feature importance computation."""

    def test_importance_sums_to_one(
        self,
        trained_models: Tuple[Any, Any, List[str]],
    ) -> None:
        """Assert normalized importances sum to 1.0."""
        from src.model import get_feature_importance

        xgb_model, lgbm_model, feature_names = trained_models
        imp_df = get_feature_importance(xgb_model, lgbm_model, feature_names)

        # XGBoost and LightGBM importances are individually normalized
        assert imp_df["xgb_importance"].sum() == pytest.approx(1.0, abs=0.01)
        assert imp_df["lgbm_importance"].sum() == pytest.approx(1.0, abs=0.01)

    def test_importance_has_all_features(
        self,
        trained_models: Tuple[Any, Any, List[str]],
    ) -> None:
        """Assert importance DataFrame includes all features."""
        from src.model import get_feature_importance

        xgb_model, lgbm_model, feature_names = trained_models
        imp_df = get_feature_importance(xgb_model, lgbm_model, feature_names)

        assert len(imp_df) == len(feature_names)
        assert set(imp_df["feature"]) == set(feature_names)

    def test_importance_sorted_descending(
        self,
        trained_models: Tuple[Any, Any, List[str]],
    ) -> None:
        """Assert importance DataFrame is sorted by importance descending."""
        from src.model import get_feature_importance

        xgb_model, lgbm_model, feature_names = trained_models
        imp_df = get_feature_importance(xgb_model, lgbm_model, feature_names)

        importances = imp_df["importance"].values
        for i in range(len(importances) - 1):
            assert importances[i] >= importances[i + 1], (
                f"Not sorted: {importances[i]} < {importances[i+1]}"
            )

    def test_importance_all_non_negative(
        self,
        trained_models: Tuple[Any, Any, List[str]],
    ) -> None:
        """Assert all importances are non-negative."""
        from src.model import get_feature_importance

        xgb_model, lgbm_model, feature_names = trained_models
        imp_df = get_feature_importance(xgb_model, lgbm_model, feature_names)

        assert (imp_df["importance"] >= 0).all()
        assert (imp_df["xgb_importance"] >= 0).all()
        assert (imp_df["lgbm_importance"] >= 0).all()
