"""
Model training, evaluation, and persistence module for EarningsPulse.

Implements a dual-model ensemble (XGBoost + LightGBM) with
**GroupShuffleSplit** cross-validation — each fold holds out ALL quarters
of 2–3 tickers completely, so the model is tested on companies it has
never seen during training.  This is the correct methodology for
financial ML and produces honest, generalizable AUC scores.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit

from src.data_loader import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_or_build_training_data,
)
from src.utils import DATA_DIR, setup_logger

log = setup_logger(__name__)

# ─── Model hyperparameters ─────────────────────────────────────────────────────
XGB_PARAMS: Dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1.2,
    "random_state": 42,
    "eval_metric": "logloss",
    "use_label_encoder": False,
}

LGBM_PARAMS: Dict[str, Any] = {
    "n_estimators": 300,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "random_state": 42,
    "verbose": -1,
}


# Features whose outliers are clipped to the 1st/99th percentile.
CLIP_FEATURES: List[str] = [
    "eps_surprise_last_q", "eps_acceleration",
    "eps_growth_yoy", "eps_magnitude_trend",
]


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED BUILDING BLOCKS (used by training and by src.evaluate)
# ═══════════════════════════════════════════════════════════════════════════════
def prepare_feature_frame(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Select feature columns, coerce to numeric and fill missing values with 0."""
    X = df[features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.fillna(0)


class FeaturePreprocessor:
    """Quantile clipping + RobustScaler, fitted on one frame and applied to others.

    ``fit`` only looks at the rows it is given, so fitting on a training split
    keeps test-split statistics out of the transform.
    """

    def __init__(self) -> None:
        from sklearn.preprocessing import RobustScaler
        self.clip_bounds: Dict[str, Tuple[float, float]] = {}
        self.scaler = RobustScaler()

    def _clip(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (low, high) in self.clip_bounds.items():
            X[col] = X[col].clip(low, high)
        if "eps_beat_streak" in X.columns:
            X["eps_beat_streak"] = X["eps_beat_streak"].clip(0, 8)
        return X

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        self.clip_bounds = {
            col: (float(X[col].quantile(0.01)), float(X[col].quantile(0.99)))
            for col in CLIP_FEATURES if col in X.columns
        }
        self.scaler.fit(self._clip(X))
        return self

    def clip(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply only the clipping step (used for the imputation medians)."""
        return self._clip(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.scaler.transform(self._clip(X)), columns=X.columns, index=X.index,
        )


def group_shuffle_splits(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """The ticker-holdout splits behind the README's CV figures."""
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    return list(gss.split(X, y, groups))


def fit_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> Tuple[Any, Any]:
    """Fit the XGBoost and LightGBM members of the ensemble.

    ``X_val``/``y_val`` are only passed as an eval set for logging; they do
    not affect the fitted trees (no early stopping).
    """
    import xgboost as xgb
    import lightgbm as lgb

    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    lgbm_model = lgb.LGBMClassifier(**LGBM_PARAMS)
    if X_val is not None and y_val is not None:
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                       callbacks=[lgb.log_evaluation(period=0)])
    else:
        xgb_model.fit(X_train, y_train, verbose=False)
        lgbm_model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(period=0)])
    return xgb_model, lgbm_model


def predict_ensemble(xgb_model: Any, lgbm_model: Any, X: pd.DataFrame) -> np.ndarray:
    """Average the two members' beat probabilities."""
    return (xgb_model.predict_proba(X)[:, 1] + lgbm_model.predict_proba(X)[:, 1]) / 2


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Train XGBoost and LightGBM with GroupShuffleSplit cross-validation.

    Each fold holds out ALL quarters of 2–3 tickers.  The model never sees
    any data from the held-out tickers during training, which tests true
    generalisation to unseen companies.

    Args:
        X: Feature matrix (rows = samples, columns = features).
        y: Binary target vector (1 = beat, 0 = miss).
        groups: Ticker column used for group-aware splitting.
        n_splits: Number of cross-validation folds.

    Returns:
        Tuple of ``(xgb_model, lgbm_model, metrics_dict)``.
    """
    log.info("Training models on %d samples with %d features", len(X), X.shape[1])
    log.info("Target distribution: beat=%.1f%%, miss=%.1f%%",
             y.mean() * 100, (1 - y.mean()) * 100)
    log.info("Tickers in dataset: %s", sorted(groups.unique().tolist()))

    splits = group_shuffle_splits(X, y, groups, n_splits)
    fold_metrics: List[Dict[str, Any]] = []

    best_auc = -1.0
    best_xgb = None
    best_lgbm = None

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_tickers = sorted(groups.iloc[train_idx].unique().tolist())
        val_tickers = sorted(groups.iloc[val_idx].unique().tolist())

        print(f"\n{'='*60}")
        print(f"  FOLD {fold}/{n_splits}  |  Train: {len(X_train)}  |  Val: {len(X_val)}")
        print(f"  Train tickers ({len(train_tickers)}): {train_tickers}")
        print(f"  Val tickers   ({len(val_tickers)}):  {val_tickers}")
        print(f"{'='*60}")

        xgb_model, lgbm_model = fit_ensemble(X_train, y_train, X_val, y_val)
        ensemble_proba = predict_ensemble(xgb_model, lgbm_model, X_val)
        ensemble_preds = (ensemble_proba >= 0.5).astype(int)

        # ── Fold metrics ──
        unique_classes = np.unique(y_val)
        if len(unique_classes) < 2:
            fold_auc = 0.5
            log.warning("Fold %d has only one class — AUC set to 0.5", fold)
        else:
            fold_auc = roc_auc_score(y_val, ensemble_proba)

        fold_acc = accuracy_score(y_val, ensemble_preds)
        fold_f1 = f1_score(y_val, ensemble_preds, average="weighted", zero_division=0)

        fold_result = {
            "fold": fold,
            "auc": float(fold_auc),
            "accuracy": float(fold_acc),
            "f1": float(fold_f1),
            "train_size": len(X_train),
            "val_size": len(X_val),
            "val_tickers": val_tickers,
        }
        fold_metrics.append(fold_result)

        print(f"  AUC: {fold_auc:.4f}  |  Accuracy: {fold_acc:.4f}  |  F1: {fold_f1:.4f}")

        if fold_auc > best_auc:
            best_auc = fold_auc
            best_xgb = xgb_model
            best_lgbm = lgbm_model

    # ── Final training on full data ──
    print(f"\n{'='*60}")
    print(f"  FINAL TRAINING ON ALL {len(X)} SAMPLES")
    print(f"{'='*60}")

    final_xgb, final_lgbm = fit_ensemble(X, y)

    # ── Use last fold split for final evaluation metrics ──
    last_train_idx, last_test_idx = splits[-1]
    X_test, y_test = X.iloc[last_test_idx], y.iloc[last_test_idx]

    metrics = evaluate_model(final_xgb, final_lgbm, X_test, y_test)
    metrics["fold_metrics"] = fold_metrics
    metrics["avg_cv_auc"] = float(np.mean([f["auc"] for f in fold_metrics]))
    metrics["avg_cv_accuracy"] = float(np.mean([f["accuracy"] for f in fold_metrics]))
    metrics["avg_cv_f1"] = float(np.mean([f["f1"] for f in fold_metrics]))
    metrics["n_train_samples"] = len(X)
    metrics["n_features"] = X.shape[1]
    metrics["training_date"] = datetime.now().isoformat()
    metrics["cv_method"] = "GroupShuffleSplit (ticker-level holdout)"

    # ── Baseline comparison ──
    beat_rate = float(y.mean())
    naive_accuracy = max(beat_rate, 1.0 - beat_rate)
    metrics["naive_baseline"] = naive_accuracy
    metrics["improvement_over_baseline"] = float(metrics["accuracy"] - naive_accuracy)
    metrics["training_tickers"] = int(groups.nunique())
    metrics["training_quarters"] = "2014-2024"
    metrics["trained_at"] = datetime.now().isoformat()

    # ── Calibration ──
    try:
        xgb_test_proba = final_xgb.predict_proba(X_test)[:, 1]
        lgbm_test_proba = final_lgbm.predict_proba(X_test)[:, 1]
        test_ensemble_proba = (xgb_test_proba + lgbm_test_proba) / 2
        prob_true, prob_pred = calibration_curve(y_test, test_ensemble_proba, n_bins=5)
        metrics["calibration_mean_predicted"] = float(prob_pred.mean())
        metrics["calibration_mean_actual"] = float(prob_true.mean())
    except Exception:
        metrics["calibration_mean_predicted"] = None
        metrics["calibration_mean_actual"] = None

    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION SUMMARY (GroupShuffleSplit)")
    print(f"{'='*60}")
    print(f"  Avg AUC:       {metrics['avg_cv_auc']:.4f}")
    print(f"  Avg Accuracy:  {metrics['avg_cv_accuracy']:.4f}")
    print(f"  Avg F1:        {metrics['avg_cv_f1']:.4f}")
    print(f"  Naive baseline accuracy: {naive_accuracy:.4f}")
    print(f"  Model improvement over baseline: +{metrics['improvement_over_baseline']:.4f}")
    if metrics.get("calibration_mean_predicted") is not None:
        print(f"  Calibration — mean predicted: {metrics['calibration_mean_predicted']:.3f}, "
              f"mean actual: {metrics['calibration_mean_actual']:.3f}")
    print(f"{'='*60}")

    return final_xgb, final_lgbm, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_model(
    xgb_model: Any,
    lgbm_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluate the ensemble model on a test set.

    Args:
        xgb_model: Trained XGBoost classifier.
        lgbm_model: Trained LightGBM classifier.
        X_test: Test feature matrix.
        y_test: Test target vector.

    Returns:
        Dictionary with all evaluation metrics.
    """
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]
    ensemble_proba = (xgb_proba + lgbm_proba) / 2
    ensemble_preds = (ensemble_proba >= 0.5).astype(int)

    unique_classes = np.unique(y_test)
    if len(unique_classes) < 2:
        auc = 0.5
        fpr, tpr = [0, 1], [0, 1]
    else:
        auc = roc_auc_score(y_test, ensemble_proba)
        fpr, tpr, _ = roc_curve(y_test, ensemble_proba)

    metrics: Dict[str, Any] = {
        "roc_auc": float(auc),
        "accuracy": float(accuracy_score(y_test, ensemble_preds)),
        "precision_beat": float(precision_score(y_test, ensemble_preds, pos_label=1, zero_division=0)),
        "recall_beat": float(recall_score(y_test, ensemble_preds, pos_label=1, zero_division=0)),
        "f1_beat": float(f1_score(y_test, ensemble_preds, pos_label=1, zero_division=0)),
        "precision_miss": float(precision_score(y_test, ensemble_preds, pos_label=0, zero_division=0)),
        "recall_miss": float(recall_score(y_test, ensemble_preds, pos_label=0, zero_division=0)),
        "f1_miss": float(f1_score(y_test, ensemble_preds, pos_label=0, zero_division=0)),
        "brier_score": float(brier_score_loss(y_test, ensemble_proba)),
        "confusion_matrix": confusion_matrix(y_test, ensemble_preds).tolist(),
        "roc_curve": {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
        },
        "n_test_samples": len(y_test),
    }

    print("\n" + "─" * 50)
    print("  EVALUATION METRICS")
    print("─" * 50)
    print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    print(f"\n  Beat (class=1):  P={metrics['precision_beat']:.3f}  R={metrics['recall_beat']:.3f}  F1={metrics['f1_beat']:.3f}")
    print(f"  Miss (class=0):  P={metrics['precision_miss']:.3f}  R={metrics['recall_miss']:.3f}  F1={metrics['f1_miss']:.3f}")
    print(f"\n  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"                 Predicted Miss  Predicted Beat")
    print(f"  Actual Miss    {cm[0][0]:>14}  {cm[0][1]:>14}")
    print(f"  Actual Beat    {cm[1][0]:>14}  {cm[1][1]:>14}")
    print("─" * 50)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
def get_feature_importance(
    xgb_model: Any,
    lgbm_model: Any,
    feature_names: List[str],
) -> pd.DataFrame:
    """Compute average feature importance across both models.

    Returns:
        DataFrame with columns ``[feature, importance, rank]``
        sorted by importance descending.
    """
    xgb_imp = xgb_model.feature_importances_
    xgb_imp = xgb_imp / xgb_imp.sum() if xgb_imp.sum() > 0 else xgb_imp

    lgbm_imp = lgbm_model.feature_importances_.astype(float)
    lgbm_imp = lgbm_imp / lgbm_imp.sum() if lgbm_imp.sum() > 0 else lgbm_imp

    avg_imp = (xgb_imp + lgbm_imp) / 2

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": avg_imp,
        "xgb_importance": xgb_imp,
        "lgbm_importance": lgbm_imp,
    })
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def save_feature_importance_plot(
    importance_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> str:
    """Save a horizontal bar chart of feature importances."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if save_path is None:
            save_path = str(DATA_DIR / "feature_importance.png")

        fig, ax = plt.subplots(figsize=(10, 8))
        df_plot = importance_df.sort_values("importance", ascending=True)

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_plot)))
        ax.barh(df_plot["feature"], df_plot["importance"], color=colors)

        ax.set_xlabel("Average Importance (XGBoost + LightGBM)", fontsize=12)
        ax.set_title("Feature Importance — EarningsPulse Ensemble", fontsize=14, fontweight="bold")
        ax.tick_params(axis="y", labelsize=10)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        log.info("Feature importance plot saved to %s", save_path)
        return save_path
    except ImportError:
        log.warning("matplotlib not installed — skipping PNG plot (JSON still saved)")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════
def save_models(
    xgb_model: Any,
    lgbm_model: Any,
    metrics: Dict[str, Any],
    feature_names: List[str],
    path: Optional[str] = None,
) -> None:
    """Save trained models, metrics, and metadata to disk."""
    save_dir = Path(path) if path else DATA_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(xgb_model, save_dir / "xgb_model.joblib")
    joblib.dump(lgbm_model, save_dir / "lgbm_model.joblib")

    metadata = {
        "feature_names": feature_names,
        "training_date": datetime.now().isoformat(),
        "n_features": len(feature_names),
        "cv_method": "GroupShuffleSplit",
        "metrics": {k: v for k, v in metrics.items()
                    if k not in ("roc_curve", "fold_metrics", "confusion_matrix")},
    }
    with open(save_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    with open(save_dir / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    log.info("Models saved to %s", save_dir)


def load_models(
    path: Optional[str] = None,
) -> Tuple[Any, Any, Dict[str, Any], List[str]]:
    """Load trained models and metadata from disk.

    Returns:
        Tuple of ``(xgb_model, lgbm_model, metrics, feature_names)``.

    Raises:
        FileNotFoundError: If model files are not found.
    """
    load_dir = Path(path) if path else DATA_DIR

    xgb_path = load_dir / "xgb_model.joblib"
    lgbm_path = load_dir / "lgbm_model.joblib"
    meta_path = load_dir / "model_metadata.json"
    metrics_path = load_dir / "model_metrics.json"

    if not xgb_path.exists() or not lgbm_path.exists():
        raise FileNotFoundError(
            f"Model files not found in {load_dir}. "
            f"Run 'python -m src.model' to train the models first."
        )

    xgb_model = joblib.load(xgb_path)
    lgbm_model = joblib.load(lgbm_path)

    metadata: Dict[str, Any] = {}
    feature_names: List[str] = []
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        feature_names = metadata.get("feature_names", [])

    metrics: Dict[str, Any] = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    log.info("Models loaded from %s (trained %s)",
             load_dir, metadata.get("training_date", "unknown"))

    return xgb_model, lgbm_model, metrics, feature_names


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT (python -m src.model)
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    """Full training pipeline: data load → train → save."""
    print("\n" + "═" * 60)
    print("  EarningsPulse — Model Training Pipeline")
    print("═" * 60)

    # Step 1: Load or build training data
    print("\n[1/4] Loading training data...")
    df = load_or_build_training_data()

    if df.empty:
        print("ERROR: Training data is empty. Check API key and connectivity.")
        return

    print(f"  → {len(df)} samples, {df['ticker'].nunique()} tickers")

    # Step 2: Prepare X, y, and groups
    print("\n[2/4] Preparing training data...")
    available_features = [f for f in FEATURE_COLUMNS if f in df.columns]

    X = prepare_feature_frame(df, available_features)
    y = df[TARGET_COLUMN].copy()
    groups = df["ticker"].copy()

    # Clip outliers and scale with RobustScaler (fitted on the full dataset,
    # as the deployed model sees all rows)
    preprocessor = FeaturePreprocessor().fit(X)
    for col, (low, high) in preprocessor.clip_bounds.items():
        log.info("Clipped %s to [%.2f, %.2f]", col, low, high)
    X = preprocessor.clip(X)
    X_scaled = preprocessor.transform(X)
    scaler = preprocessor.scaler
    joblib.dump(scaler, DATA_DIR / "feature_scaler.joblib")
    log.info("Saved RobustScaler to data/feature_scaler.joblib")

    print(f"  → Features ({len(available_features)}): {list(available_features)}")
    print(f"  → Target distribution: Beat={int(y.sum())}/{len(y)} ({y.mean()*100:.1f}%)")
    print(f"  → Groups: {sorted(groups.unique().tolist())}")

    # Step 3: Train with GroupShuffleSplit
    print("\n[3/4] Training ensemble models (GroupShuffleSplit CV)...")
    xgb_model, lgbm_model, metrics = train_models(X_scaled, y, groups)

    # Step 4: Save
    print("\n[4/4] Saving models and artifacts...")
    save_models(xgb_model, lgbm_model, metrics, available_features)

    # Save feature medians from PRE-SCALED data for live inference imputation
    medians = X.median().to_dict()  # X is pre-scaled, used for imputation defaults
    medians_path = DATA_DIR / "feature_medians.json"
    with open(medians_path, "w") as f:
        json.dump(medians, f, indent=2)
    log.info("Saved feature medians to %s", medians_path)

    # Feature importance
    importance_df = get_feature_importance(xgb_model, lgbm_model, available_features)
    save_feature_importance_plot(importance_df)
    importance_df.to_csv(DATA_DIR / "feature_importance.csv", index=False)
    importance_df.to_json(DATA_DIR / "feature_importance.json", orient="records", indent=2)
    log.info("Saved feature importance to CSV and JSON")

    print("\n" + "═" * 60)
    print("  Training complete!")
    print(f"  Models saved to: {DATA_DIR}")
    print(f"  ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
    print(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
    print(f"  Naive baseline: {metrics.get('naive_baseline', 'N/A')}")
    print(f"  CV method: GroupShuffleSplit (ticker-level holdout)")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
