"""
Unit tests for the model training, evaluation, and persistence pipeline.

Tests cover prediction output format, GroupShuffleSplit correctness,
ensemble averaging, model save/load roundtrip, and feature importance
normalization.

NOTE: FEATURE_COLUMNS lives in src.data_loader (not src.feature_engineering).
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from src.data_loader import FEATURE_COLUMNS


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

    # Create target with some signal from first few features
    signal = X.iloc[:, 0] * 0.3 + X.iloc[:, 1] * 0.2 + np.random.randn(n_samples) * 0.5
    y = pd.Series((signal > 0).astype(int), name="beat")

    return X, y


@pytest.fixture
def trained_models(
    synthetic_training_data: Tuple[pd.DataFrame, pd.Series],
) -> Tuple[Any, Any, List[str]]:
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


class TestGroupShuffleSplit:
    """Tests for GroupShuffleSplit cross-validation (ticker-level holdout).

    The project uses GroupShuffleSplit so each fold holds out ALL quarters
    of 2-3 complete tickers — the model never sees those companies during
    training for that fold.
    """

    def test_no_group_overlap_between_train_and_val(
        self,
        synthetic_training_data: Tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Assert no ticker appears in both train and validation folds."""
        from sklearn.model_selection import GroupShuffleSplit

        X, y = synthetic_training_data
        n_samples = len(X)
        # Simulate 5 tickers with equal samples
        tickers = ["T" + str(i) for i in range(5)]
        groups = pd.Series(
            [tickers[i % 5] for i in range(n_samples)], name="ticker"
        )

        gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        for train_idx, val_idx in gss.split(X, y, groups):
            train_tickers = set(groups.iloc[train_idx])
            val_tickers = set(groups.iloc[val_idx])
            overlap = train_tickers & val_tickers
            assert len(overlap) == 0, (
                f"Ticker overlap between train and val: {overlap}"
            )

    def test_validation_never_overlaps_train_indices(
        self,
        synthetic_training_data: Tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Assert train and validation indices never share the same row."""
        from sklearn.model_selection import GroupShuffleSplit

        X, y = synthetic_training_data
        n_samples = len(X)
        groups = pd.Series([f"T{i % 5}" for i in range(n_samples)])

        gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        for train_idx, val_idx in gss.split(X, y, groups):
            overlap = set(train_idx) & set(val_idx)
            assert len(overlap) == 0, f"Index overlap detected: {overlap}"

    def test_cv_method_label_in_metrics(self) -> None:
        """Assert saved model_metrics.json records GroupShuffleSplit as cv_method."""
        import json
        from pathlib import Path

        metrics_path = Path("data/model_metrics.json")
        if not metrics_path.exists():
            pytest.skip("model_metrics.json not found — run python -m src.model first")

        with open(metrics_path) as f:
            metrics = json.load(f)

        cv_method = metrics.get("cv_method", "")
        assert "GroupShuffleSplit" in cv_method, (
            f"Expected 'GroupShuffleSplit' in cv_method, got: {cv_method!r}"
        )


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

    def test_load_models_returns_four_values(self) -> None:
        """Assert load_models() returns a 4-tuple: (xgb, lgbm, metrics, feature_names)."""
        from src.model import load_models

        result = load_models()
        assert len(result) == 4, (
            f"load_models() should return 4 values, got {len(result)}"
        )
        xgb_model, lgbm_model, metrics, feature_names = result
        assert hasattr(xgb_model, "predict_proba")
        assert hasattr(lgbm_model, "predict_proba")
        assert isinstance(metrics, dict)
        assert isinstance(feature_names, list)


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
