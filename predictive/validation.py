from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit


def metric_bundle(problem_type: str, y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> Dict[str, float | None]:
    if problem_type in {"regression", "forecasting"}:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else None
        metrics = {
            "mae": float(mean_absolute_error(y_true, y_pred)) if len(y_true) else None,
            "rmse": rmse,
            "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
        }
        if problem_type == "forecasting":
            safe_true = np.where(np.asarray(y_true) == 0, np.nan, np.asarray(y_true))
            metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred)) if len(y_true) else None
            metrics["bias"] = float(np.nanmean(np.asarray(y_pred) - np.asarray(y_true))) if len(y_true) else None
            if np.isnan(metrics["mape"]):
                metrics["mape"] = None
        return metrics

    metrics = {
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": None,
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            metrics["roc_auc"] = None
        try:
            metrics["brier_score"] = float(brier_score_loss(y_true, y_score))
        except Exception:
            metrics["brier_score"] = None
        try:
            metrics["calibration_gap"] = float(abs(float(np.mean(y_score)) - float(np.mean(y_true))))
        except Exception:
            metrics["calibration_gap"] = None
    return metrics


def selection_score(problem_type: str, metrics: Dict[str, float | None]) -> float:
    if problem_type in {"regression", "forecasting"}:
        rmse = metrics.get("rmse")
        mae = metrics.get("mae")
        r2 = metrics.get("r2") or 0.0
        if rmse is None or mae is None:
            return float("-inf")
        return float(-rmse - (0.3 * mae) + (10 * r2))
    return float(metrics.get("f1") or 0.0) + float(metrics.get("roc_auc") or 0.0)


def build_baseline_predictions(problem_type: str, y_train: pd.Series, y_test: pd.Series) -> tuple[np.ndarray, np.ndarray | None, str]:
    if problem_type in {"regression", "forecasting"}:
        if problem_type == "forecasting" and len(y_train) > 0:
            baseline_value = float(y_train.iloc[-1])
            label = "naive_last_value"
        else:
            baseline_value = float(pd.to_numeric(y_train, errors="coerce").mean())
            label = "naive_mean"
        return np.full(len(y_test), baseline_value), None, label

    majority_class = pd.Series(y_train).mode(dropna=True)
    baseline_value = majority_class.iloc[0] if not majority_class.empty else 0
    baseline_pred = np.full(len(y_test), baseline_value)
    baseline_score = np.full(len(y_test), float(pd.Series(y_train).eq(baseline_value).mean()))
    return baseline_pred, baseline_score, "majority_class"


def cross_validate_model(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
) -> Dict[str, Any]:
    if len(X_train) < 25:
        return {"fold_metrics": [], "mean": {}, "std": {}, "folds": 0}

    if problem_type == "classification":
        splitter = StratifiedKFold(n_splits=min(5, max(3, pd.Series(y_train).value_counts().min())), shuffle=True, random_state=42)
    elif problem_type == "forecasting":
        splitter = TimeSeriesSplit(n_splits=min(5, max(3, len(X_train) // 20)))
    else:
        splitter = KFold(n_splits=min(5, max(3, len(X_train) // 20)), shuffle=True, random_state=42)

    fold_metrics: List[Dict[str, float | None]] = []
    for train_idx, valid_idx in splitter.split(X_train, y_train if problem_type == "classification" else None):
        X_fit = X_train.iloc[train_idx]
        X_valid = X_train.iloc[valid_idx]
        y_fit = y_train.iloc[train_idx]
        y_valid = y_train.iloc[valid_idx]

        estimator = clone(model)
        estimator.fit(X_fit, y_fit)
        preds = estimator.predict(X_valid)
        score_vector = None
        if problem_type == "classification" and hasattr(estimator, "predict_proba"):
            try:
                score_vector = estimator.predict_proba(X_valid)[:, 1]
            except Exception:
                score_vector = None
        fold_metrics.append(metric_bundle(problem_type, y_valid, preds, score_vector))

    if not fold_metrics:
        return {"fold_metrics": [], "mean": {}, "std": {}, "folds": 0}

    keys = {key for fold in fold_metrics for key in fold.keys()}
    mean_values = {}
    std_values = {}
    for key in keys:
        numeric_values = [float(fold[key]) for fold in fold_metrics if fold.get(key) is not None]
        if not numeric_values:
            mean_values[key] = None
            std_values[key] = None
            continue
        mean_values[key] = float(np.mean(numeric_values))
        std_values[key] = float(np.std(numeric_values))
    return {
        "fold_metrics": fold_metrics,
        "mean": mean_values,
        "std": std_values,
        "folds": len(fold_metrics),
    }


def residual_diagnostics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float | None]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    residuals = actual - predicted
    if len(residuals) == 0:
        return {}
    denom = float(np.std(actual)) or 1.0
    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_skew": float(pd.Series(residuals).skew()) if len(residuals) > 2 else None,
        "outlier_residual_ratio": float(np.mean(np.abs(residuals) > (2 * np.std(residuals)))) if len(residuals) > 1 else 0.0,
        "prediction_interval_half_width": float(1.96 * np.std(residuals)) if len(residuals) > 1 else None,
        "relative_interval_width": float((2 * 1.96 * np.std(residuals)) / denom) if len(residuals) > 1 else None,
    }


def confusion_details(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    labels = sorted(pd.Series(y_true).dropna().unique().tolist())
    matrix = pd.crosstab(pd.Series(y_true, name="actual"), pd.Series(y_pred, name="predicted"), dropna=False)
    return {
        "labels": labels,
        "matrix": matrix.to_dict(),
    }


def overfit_gap(problem_type: str, train_metrics: Dict[str, float | None], test_metrics: Dict[str, float | None]) -> float:
    if problem_type in {"regression", "forecasting"}:
        return float((train_metrics.get("r2") or 0.0) - (test_metrics.get("r2") or 0.0))
    return float((train_metrics.get("f1") or 0.0) - (test_metrics.get("f1") or 0.0))
