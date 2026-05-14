from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _numeric_drift(train: pd.Series, score: pd.Series) -> Dict[str, float | None]:
    train_num = pd.to_numeric(train, errors="coerce").dropna()
    score_num = pd.to_numeric(score, errors="coerce").dropna()
    if train_num.empty or score_num.empty:
        return {"mean_shift": None, "std_ratio": None, "drift_score": None}
    train_mean = float(train_num.mean())
    score_mean = float(score_num.mean())
    denom = float(train_num.std(ddof=0)) or 1.0
    mean_shift = abs(score_mean - train_mean) / denom
    std_ratio = float((score_num.std(ddof=0) or 1.0) / (train_num.std(ddof=0) or 1.0))
    drift_score = min(mean_shift + abs(std_ratio - 1.0), 5.0) / 5.0
    return {
        "mean_shift": float(mean_shift),
        "std_ratio": float(std_ratio),
        "drift_score": float(drift_score),
    }


def _categorical_drift(train: pd.Series, score: pd.Series) -> Dict[str, float | None]:
    train_dist = train.astype(str).fillna("__missing__").value_counts(normalize=True)
    score_dist = score.astype(str).fillna("__missing__").value_counts(normalize=True)
    categories = sorted(set(train_dist.index).union(score_dist.index))
    drift = 0.0
    for category in categories:
        drift += abs(float(train_dist.get(category, 0.0)) - float(score_dist.get(category, 0.0)))
    return {
        "population_shift": float(drift / 2.0),
        "drift_score": float(min(drift / 2.0, 1.0)),
    }


def detect_data_drift(
    training_frame: pd.DataFrame,
    scoring_frame: pd.DataFrame,
    feature_columns: List[str],
) -> Dict[str, Any]:
    drift_by_feature: Dict[str, Any] = {}
    warnings: List[str] = []
    scores: List[float] = []

    for column in feature_columns:
        if column not in training_frame.columns or column not in scoring_frame.columns:
            continue
        if pd.api.types.is_numeric_dtype(training_frame[column]) and pd.api.types.is_numeric_dtype(scoring_frame[column]):
            details = _numeric_drift(training_frame[column], scoring_frame[column])
        else:
            details = _categorical_drift(training_frame[column], scoring_frame[column])
        drift_by_feature[column] = details
        score = details.get("drift_score")
        if score is not None:
            scores.append(float(score))
            if float(score) >= 0.4:
                warnings.append(f"{column} shows material drift.")

    average_drift = float(np.mean(scores)) if scores else 0.0
    health_score = max(0, min(100, int(round(100 - (average_drift * 100)))))
    retrain_triggered = average_drift >= 0.35 or len(warnings) >= 3

    return {
        "feature_drift": drift_by_feature,
        "warnings": warnings,
        "average_drift_score": average_drift,
        "health_score": health_score,
        "retrain_triggered": retrain_triggered,
        "monitoring_plan": [
            "Track feature drift on each new scoring batch.",
            "Recompute holdout-like performance when actuals arrive.",
            "Retrain when drift or realized performance exceeds thresholds.",
        ],
    }


def performance_decay_monitor(
    metrics: Dict[str, float | None],
    baseline_metrics: Dict[str, float | None],
    problem_type: str,
) -> Dict[str, Any]:
    warnings: List[str] = []
    decay_score = 0.0
    if problem_type in {"regression", "forecasting"}:
        current_r2 = float(metrics.get("r2") or 0.0)
        baseline_r2 = float(baseline_metrics.get("r2") or 0.0)
        decay_score = max(0.0, baseline_r2 - current_r2)
        if decay_score > 0.1:
            warnings.append("Explained variance has decayed materially from baseline.")
    else:
        current_f1 = float(metrics.get("f1") or 0.0)
        baseline_f1 = float(baseline_metrics.get("f1") or 0.0)
        decay_score = max(0.0, baseline_f1 - current_f1)
        if decay_score > 0.08:
            warnings.append("Classification performance has decayed materially from baseline.")
    return {
        "decay_score": float(decay_score),
        "warnings": warnings,
        "healthy": decay_score <= (0.1 if problem_type in {"regression", "forecasting"} else 0.08),
    }
