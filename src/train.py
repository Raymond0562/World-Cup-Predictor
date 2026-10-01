"""
Train, evaluate, and persist match-outcome prediction models.

Usage:
    python -m src.train [--no-shap]

Time-based splits (no shuffling):
    train  : date <= TRAIN_END_DATE
    val    : TRAIN_END_DATE < date <= VAL_END_DATE
    test   : date > VAL_END_DATE

Three models are compared: Logistic Regression, Random Forest, XGBoost.
The best by validation log-loss is saved with isotonic probability calibration.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from config import (
    HOME_WIN, DRAW, AWAY_WIN, OUTCOME_LABELS,
    TRAIN_END_DATE, VAL_END_DATE, FINAL_TRAIN_END_DATE,
    LR_PARAMS, RF_PARAMS, XGB_PARAMS, LGBM_PARAMS,
    RANDOM_STATE, WEIGHT_DECAY_RATE, WC_2026_WEIGHT_MULTIPLIER,
)
from src.data_loader import load_all
from src.feature_engineering import build_features, FEATURE_COLS

# silence XGBoost/LightGBM version warnings
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample weighting
# ---------------------------------------------------------------------------

def compute_sample_weights(
    slice_df: pd.DataFrame,
    decay_rate: float = WEIGHT_DECAY_RATE,
    wc_2026_multiplier: float = WC_2026_WEIGHT_MULTIPLIER,
) -> np.ndarray:
    """
    Combine exponential recency decay with a WC-2026 tournament boost.

    - Recency: exp(decay_rate * t) where t ∈ [0,1], oldest→0, newest→1.
      With decay_rate=3 the newest match gets ~20× the weight of the oldest.
    - WC 2026 boost: multiply those matches by wc_2026_multiplier on top.
    """
    dates = pd.to_datetime(slice_df["date"])
    span = max((dates.max() - dates.min()).days, 1)
    t = (dates - dates.min()).dt.days / span
    weights = np.exp(decay_rate * t.values)

    if "tournament" in slice_df.columns:
        is_wc_2026 = (
            (slice_df["tournament"].values == "FIFA World Cup") &
            (dates.dt.year.values == 2026)
        )
        weights[is_wc_2026] *= wc_2026_multiplier

    return weights


# ---------------------------------------------------------------------------
# Model registry  (name → (class, params))
# ---------------------------------------------------------------------------

def _build_model_registry() -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(**LR_PARAMS),
        "random_forest": RandomForestClassifier(**RF_PARAMS),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(**XGB_PARAMS)
        logger.info("XGBoost available — added to comparison.")
    except ImportError:
        logger.warning("xgboost not installed — skipping.")

    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(**LGBM_PARAMS)
        logger.info("LightGBM available — added to comparison.")
    except ImportError:
        pass

    return models


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _brier_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Average one-vs-rest Brier score across all three outcomes."""
    n_classes = y_prob.shape[1]
    scores = []
    for c in range(n_classes):
        binary = (y_true == c).astype(float)
        scores.append(brier_score_loss(binary, y_prob[:, c]))
    return float(np.mean(scores))


def _evaluate(name: str, model: Any, X: np.ndarray, y: np.ndarray) -> dict:
    y_prob = model.predict_proba(X)
    y_pred = model.predict(X)
    return {
        "model": name,
        "log_loss": round(log_loss(y, y_prob), 4),
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "brier": round(_brier_multiclass(y, y_prob), 4),
    }


# ---------------------------------------------------------------------------
# Calibration check
# ---------------------------------------------------------------------------

def _calibration_stats(model: Any, X: np.ndarray, y: np.ndarray) -> list[dict]:
    rows = []
    y_prob = model.predict_proba(X)
    for c, label in OUTCOME_LABELS.items():
        binary = (y == c).astype(float)
        frac_pos, mean_pred = calibration_curve(binary, y_prob[:, c], n_bins=10)
        rows.append({"outcome": label, "mean_pred": list(mean_pred.round(3)),
                     "frac_pos": list(frac_pos.round(3))})
    return rows


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def _feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """Return a DataFrame with feature names and importance scores."""
    base = model.estimator if hasattr(model, "estimator") else model

    if hasattr(base, "feature_importances_"):
        importance = base.feature_importances_
    elif hasattr(base, "coef_"):
        # Logistic regression: mean absolute coefficient across classes
        importance = np.mean(np.abs(base.coef_), axis=0)
    else:
        return pd.DataFrame()

    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _shap_importance(model: Any, X: np.ndarray, feature_names: list[str]) -> pd.DataFrame | None:
    try:
        import shap
        base = model.estimator if hasattr(model, "estimator") else model
        explainer = shap.TreeExplainer(base)
        shap_values = explainer.shap_values(X[:200])  # sample for speed
        # For multiclass shap_values is a list; take mean absolute across classes
        if isinstance(shap_values, list):
            mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)
        df = pd.DataFrame({"feature": feature_names, "shap_importance": mean_shap})
        return df.sort_values("shap_importance", ascending=False).reset_index(drop=True)
    except Exception as exc:
        logger.warning("SHAP failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(use_shap: bool = False) -> dict:
    # ── Load data ──────────────────────────────────────────────────────────
    logger.info("Loading data …")
    matches, teams = load_all()

    logger.info("Engineering features …")
    features, engineer = build_features(matches, teams)

    features = features.dropna(subset=FEATURE_COLS + ["outcome"])
    logger.info("Feature matrix: %d rows × %d features", len(features), len(FEATURE_COLS))

    # ── Phase 1: historical splits for model selection ────────────────────
    train_mask = features["date"] <= TRAIN_END_DATE
    val_mask   = (features["date"] > TRAIN_END_DATE) & (features["date"] <= VAL_END_DATE)
    test_mask  = features["date"] > VAL_END_DATE

    X_train = features.loc[train_mask, FEATURE_COLS].values.astype(float)
    y_train = features.loc[train_mask, "outcome"].values.astype(int)
    X_val   = features.loc[val_mask,   FEATURE_COLS].values.astype(float)
    y_val   = features.loc[val_mask,   "outcome"].values.astype(int)
    X_test  = features.loc[test_mask,  FEATURE_COLS].values.astype(float)
    y_test  = features.loc[test_mask,  "outcome"].values.astype(int)

    logger.info(
        "Historical split — train: %d  val: %d  test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    if len(X_train) < 50:
        raise RuntimeError("Too few training samples.  Run generate_mock_data.py to create data.")

    scaler_sel = StandardScaler()
    X_train_s = scaler_sel.fit_transform(X_train)
    X_val_s   = scaler_sel.transform(X_val)
    X_test_s  = scaler_sel.transform(X_test)

    registry = _build_model_registry()
    calibrated_models: dict[str, Any] = {}
    val_metrics: list[dict] = []

    sw_train = compute_sample_weights(features.loc[train_mask])

    for name, base_model in registry.items():
        logger.info("Training %s ...", name)
        cal_model = CalibratedClassifierCV(base_model, cv=5, method="isotonic")
        cal_model.fit(X_train_s, y_train, sample_weight=sw_train)
        calibrated_models[name] = cal_model

        vm = _evaluate(name, cal_model, X_val_s, y_val)
        val_metrics.append(vm)
        logger.info("  val log-loss=%.4f  accuracy=%.4f  brier=%.4f",
                    vm["log_loss"], vm["accuracy"], vm["brier"])

    best_name = min(val_metrics, key=lambda r: r["log_loss"])["model"]
    logger.info("Best model (by val log-loss): %s", best_name)

    test_metrics_all = [_evaluate(n, m, X_test_s, y_test)
                        for n, m in calibrated_models.items()]
    logger.info("=== Test-set results ===")
    for tm in test_metrics_all:
        logger.info("  %-20s  log-loss=%.4f  accuracy=%.4f  brier=%.4f",
                    tm["model"], tm["log_loss"], tm["accuracy"], tm["brier"])

    # ── Phase 2: retrain winner on ALL completed data ─────────────────────
    # This incorporates the 2026 WC matches that have already been played.
    final_mask = features["date"] <= FINAL_TRAIN_END_DATE
    X_final = features.loc[final_mask, FEATURE_COLS].values.astype(float)
    y_final = features.loc[final_mask, "outcome"].values.astype(int)

    logger.info(
        "Retraining %s on all completed data (%d matches, up to %s) ...",
        best_name, len(X_final), FINAL_TRAIN_END_DATE,
    )

    # Fresh scaler and model fitted on full dataset
    scaler = StandardScaler()
    X_final_s = scaler.fit_transform(X_final)
    sw_final = compute_sample_weights(features.loc[final_mask])

    # Instantiate a fresh copy of the best base model
    final_base = _build_model_registry()[best_name]
    best_model = CalibratedClassifierCV(final_base, cv=5, method="isotonic")
    best_model.fit(X_final_s, y_final, sample_weight=sw_final)

    # ── Feature importance ────────────────────────────────────────────────
    fi_df = _feature_importance(best_model, FEATURE_COLS)
    if not fi_df.empty:
        logger.info("Top features:\n%s", fi_df.head(10).to_string(index=False))

    shap_df = None
    if use_shap:
        shap_df = _shap_importance(best_model, X_final_s, FEATURE_COLS)
        if shap_df is not None:
            logger.info("Top features (SHAP):\n%s", shap_df.head(10).to_string(index=False))

    # Calibration check on historical test set (best available held-out data)
    cal_stats = _calibration_stats(calibrated_models[best_name], X_test_s, y_test)

    # ── Persist artifacts ─────────────────────────────────────────────────
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)

    with open(config.MODEL_FILE, "wb") as f:
        pickle.dump(best_model, f)
    with open(config.FEATURE_NAMES_FILE, "wb") as f:
        pickle.dump(FEATURE_COLS, f)
    with open(config.SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    with open(config.MODEL_DIR / "engineer.pkl", "wb") as f:
        pickle.dump(engineer, f)

    metrics_out = {
        "best_model": best_name,
        "final_train_samples": int(len(X_final)),
        "final_train_cutoff": FINAL_TRAIN_END_DATE,
        "validation": val_metrics,
        "test": test_metrics_all,
        "calibration": cal_stats,
        "feature_importance": fi_df.to_dict("records") if not fi_df.empty else [],
    }
    with open(config.METRICS_FILE, "w") as f:
        json.dump(metrics_out, f, indent=2)

    logger.info("Artifacts saved to %s", config.MODEL_DIR)
    return metrics_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Train the World Cup outcome model.")
    p.add_argument("--no-shap", action="store_true", help="Skip SHAP computation.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = train(use_shap=not args.no_shap)
    print(f"\nBest model: {result['best_model']}")
    best_test = next(r for r in result["test"] if r["model"] == result["best_model"])
    print(f"  Test log-loss : {best_test['log_loss']:.4f}")
    print(f"  Test accuracy : {best_test['accuracy']:.4f}")
    print(f"  Test brier    : {best_test['brier']:.4f}")
