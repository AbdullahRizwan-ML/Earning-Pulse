"""
Model training, evaluation, and persistence module for EarningsPulse.

Implements a dual-model ensemble (XGBoost + LightGBM) with time-series
aware cross-validation, comprehensive metric reporting, and feature
importance analysis.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
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
from sklearn.model_selection import TimeSeriesSplit

from src.feature_engineering import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_feature_matrix,
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


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Train XGBoost and LightGBM models with TimeSeriesSplit CV.

    Uses time-series aware splitting to prevent future data from leaking
    into the training set — critical for financial data.

    Args:
        X: Feature matrix (rows = samples, columns = features).
        y: Binary target vector (1 = beat, 0 = miss).
        n_splits: Number of cross-validation folds.

    Returns:
        Tuple of ``(xgb_model, lgbm_model, metrics_dict)`` where
        ``metrics_dict`` contains all evaluation metrics and fold results.
    """
    import xgboost as xgb
    import lightgbm as lgb

    log.info("Training models on %d samples with %d features", len(X), X.shape[1])
    log.info("Target distribution: beat=%.1f%%, miss=%.1f%%",
             y.mean() * 100, (1 - y.mean()) * 100)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: List[Dict[str, float]] = []

    # Track best models from the fold with the best AUC
    best_auc = -1.0
    best_xgb = None
    best_lgbm = None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"\n{'='*60}")
        print(f"  FOLD {fold}/{n_splits}  |  Train: {len(X_train)}  |  Val: {len(X_val)}")
        print(f"{'='*60}")

        # ── XGBoost ──
        xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        xgb_proba = xgb_model.predict_proba(X_val)[:, 1]

        # ── LightGBM ──
        lgbm_model = lgb.LGBMClassifier(**LGBM_PARAMS)
        lgbm_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(period=0)],
        )
        lgbm_proba = lgbm_model.predict_proba(X_val)[:, 1]

        # ── Ensemble ──
        ensemble_proba = (xgb_proba + lgbm_proba) / 2
        ensemble_preds = (ensemble_proba >= 0.5).astype(int)

        # ── Fold metrics ──
        fold_auc = roc_auc_score(y_val, ensemble_proba) if len(np.unique(y_val)) > 1 else 0.5
        fold_acc = accuracy_score(y_val, ensemble_preds)
        fold_f1 = f1_score(y_val, ensemble_preds, average="weighted")

        fold_result = {
            "fold": fold,
            "auc": fold_auc,
            "accuracy": fold_acc,
            "f1": fold_f1,
            "train_size": len(X_train),
            "val_size": len(X_val),
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

    final_xgb = xgb.XGBClassifier(**XGB_PARAMS)
    final_xgb.fit(X, y, verbose=False)

    final_lgbm = lgb.LGBMClassifier(**LGBM_PARAMS)
    final_lgbm.fit(X, y, callbacks=[lgb.log_evaluation(period=0)])

    # ── Aggregate metrics (using last fold as test metrics) ──
    last_fold_idx = list(tscv.split(X))[-1]
    _, test_idx = last_fold_idx
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    metrics = evaluate_model(final_xgb, final_lgbm, X_test, y_test)
    metrics["fold_metrics"] = fold_metrics
    metrics["avg_cv_auc"] = float(np.mean([f["auc"] for f in fold_metrics]))
    metrics["avg_cv_accuracy"] = float(np.mean([f["accuracy"] for f in fold_metrics]))
    metrics["avg_cv_f1"] = float(np.mean([f["f1"] for f in fold_metrics]))
    metrics["n_train_samples"] = len(X)
    metrics["n_features"] = X.shape[1]
    metrics["training_date"] = datetime.now().isoformat()

    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Avg AUC:      {metrics['avg_cv_auc']:.4f}")
    print(f"  Avg Accuracy:  {metrics['avg_cv_accuracy']:.4f}")
    print(f"  Avg F1:        {metrics['avg_cv_f1']:.4f}")
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

    Computes ROC-AUC, accuracy, precision, recall, F1, Brier score,
    and confusion matrix for the ensemble predictions.

    Args:
        xgb_model: Trained XGBoost classifier.
        lgbm_model: Trained LightGBM classifier.
        X_test: Test feature matrix.
        y_test: Test target vector.

    Returns:
        Dictionary with all evaluation metrics, including per-class
        precision/recall and the confusion matrix.
    """
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]
    ensemble_proba = (xgb_proba + lgbm_proba) / 2
    ensemble_preds = (ensemble_proba >= 0.5).astype(int)

    # Handle edge case where only one class is present
    unique_classes = np.unique(y_test)
    if len(unique_classes) < 2:
        auc = 0.5
        fpr, tpr, thresholds = [0, 1], [0, 1], [1, 0]
    else:
        auc = roc_auc_score(y_test, ensemble_proba)
        fpr, tpr, thresholds = roc_curve(y_test, ensemble_proba)

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

    # Print detailed report
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

    Normalizes importances from each model to sum to 1.0, then averages
    them for the ensemble ranking.

    Args:
        xgb_model: Trained XGBoost classifier.
        lgbm_model: Trained LightGBM classifier.
        feature_names: List of feature column names.

    Returns:
        DataFrame with columns ``[feature, importance, rank]``
        sorted by importance descending.
    """
    # XGBoost importances
    xgb_imp = xgb_model.feature_importances_
    xgb_imp = xgb_imp / xgb_imp.sum() if xgb_imp.sum() > 0 else xgb_imp

    # LightGBM importances
    lgbm_imp = lgbm_model.feature_importances_.astype(float)
    lgbm_imp = lgbm_imp / lgbm_imp.sum() if lgbm_imp.sum() > 0 else lgbm_imp

    # Average
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
    """Save a horizontal bar chart of feature importances.

    Args:
        importance_df: Output of ``get_feature_importance()``.
        save_path: File path for the output image. Defaults to
            ``data/feature_importance.png``.

    Returns:
        Path to the saved image file.
    """
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
    """Save trained models, metrics, and metadata to disk.

    Saves as joblib files with an accompanying metadata JSON for
    reproducibility tracking.

    Args:
        xgb_model: Trained XGBoost classifier.
        lgbm_model: Trained LightGBM classifier.
        metrics: Evaluation metrics dictionary.
        feature_names: List of feature names used during training.
        path: Directory to save models in. Defaults to ``data/``.
    """
    save_dir = Path(path) if path else DATA_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    joblib.dump(xgb_model, save_dir / "xgb_model.joblib")
    joblib.dump(lgbm_model, save_dir / "lgbm_model.joblib")

    # Save metadata
    metadata = {
        "feature_names": feature_names,
        "training_date": datetime.now().isoformat(),
        "n_features": len(feature_names),
        "metrics": {k: v for k, v in metrics.items()
                    if k not in ("roc_curve", "fold_metrics", "confusion_matrix")},
    }
    with open(save_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Save full metrics for the dashboard
    with open(save_dir / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    log.info("Models saved to %s", save_dir)


def load_models(
    path: Optional[str] = None,
) -> Tuple[Any, Any, Dict[str, Any], List[str]]:
    """Load trained models and metadata from disk.

    Args:
        path: Directory containing saved model files.
            Defaults to ``data/``.

    Returns:
        Tuple of ``(xgb_model, lgbm_model, metrics, feature_names)``.

    Raises:
        FileNotFoundError: If model files are not found at the specified path.
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

    # Load metadata
    metadata: Dict[str, Any] = {}
    feature_names: List[str] = []
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        feature_names = metadata.get("feature_names", [])

    # Load full metrics
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
    """Full training pipeline: data fetch → features → train → save.

    Run via ``python -m src.model`` from the project root.
    """
    print("\n" + "═" * 60)
    print("  EarningsPulse — Model Training Pipeline")
    print("═" * 60)

    # Step 1: Build feature matrix
    print("\n[1/4] Building feature matrix...")
    df = build_feature_matrix()

    if df.empty:
        print("ERROR: Feature matrix is empty. Check data connectivity.")
        return

    print(f"  → {len(df)} samples, {len(FEATURE_COLUMNS)} features")

    # Step 2: Prepare X and y
    print("\n[2/4] Preparing training data...")
    # Include prev_surprise as additional feature
    all_features = FEATURE_COLUMNS + ["prev_surprise"]
    available_features = [f for f in all_features if f in df.columns]

    X = df[available_features].copy()
    y = df[TARGET_COLUMN].copy()

    # Convert any remaining object columns
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)

    print(f"  → Features: {list(available_features)}")
    print(f"  → Target distribution: Beat={y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")

    # Step 3: Train
    print("\n[3/4] Training ensemble models...")
    xgb_model, lgbm_model, metrics = train_models(X, y)

    # Step 4: Save
    print("\n[4/4] Saving models and artifacts...")
    save_models(xgb_model, lgbm_model, metrics, available_features)

    # Feature importance
    importance_df = get_feature_importance(xgb_model, lgbm_model, available_features)
    save_feature_importance_plot(importance_df)
    importance_df.to_csv(DATA_DIR / "feature_importance.csv", index=False)

    print("\n" + "═" * 60)
    print("  Training complete!")
    print(f"  Models saved to: {DATA_DIR}")
    print(f"  ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
    print(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
