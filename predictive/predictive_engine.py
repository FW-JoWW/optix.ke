from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone

from core.predictive_contracts import ConfidenceAssessment, ModelResult, PredictiveMetrics, PredictiveResult, ReadinessWarning
from predictive.confidence import calibrate_confidence
from predictive.feature_engineering import build_feature_frame, detect_date_column, infer_problem_type, split_features
from predictive.monitoring import detect_data_drift, performance_decay_monitor
from predictive.model_registry import get_candidate_models
from predictive.readiness import assess_readiness
from predictive.runtime import apply_runtime_optimizations, determine_runtime_mode
from predictive.validation import (
    build_baseline_predictions,
    confusion_details,
    cross_validate_model,
    metric_bundle,
    overfit_gap,
    residual_diagnostics,
    selection_score,
)


def _tokenize(text: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", str(text).lower()) if token]


NON_TARGET_HINTS = {
    "predict",
    "prediction",
    "forecast",
    "forecasting",
    "future",
    "estimate",
    "estimated",
    "project",
    "projection",
    "recommend",
    "recommended",
    "recommendation",
    "optimize",
    "optimization",
    "actions",
    "action",
    "plan",
    "planning",
    "and",
    "the",
    "a",
    "an",
    "for",
    "of",
    "to",
}


def _target_match_score(question_tokens: List[str], column: str) -> float:
    column_tokens = _tokenize(column)
    if not column_tokens:
        return 0.0
    exact_overlap = len(set(question_tokens) & set(column_tokens))
    similarity = max((SequenceMatcher(None, token, part).ratio() for token in question_tokens for part in column_tokens), default=0.0)
    return float(exact_overlap) + similarity


def _infer_target_column(question: str, selected_columns: List[str], df: pd.DataFrame) -> str | None:
    question_tokens = _tokenize(question)
    candidate_columns = selected_columns or list(df.columns)
    if not candidate_columns:
        return None

    scored = []
    for column in candidate_columns:
        score = _target_match_score(question_tokens, column)
        if score > 0:
            scored.append((column, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    if scored and scored[0][1] >= 1.6:
        return scored[0][0]

    candidate_token_set = {token for column in candidate_columns for token in _tokenize(column)}
    unresolved_target_tokens = [
        token for token in question_tokens
        if token not in NON_TARGET_HINTS and token not in candidate_token_set
    ]
    if unresolved_target_tokens:
        return None

    numeric_selected = [col for col in candidate_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_selected) == 1:
        return numeric_selected[0]
    if len(candidate_columns) == 1:
        return candidate_columns[0]
    return None


def _extract_feature_importance(model: object, feature_names: List[str]) -> List[Dict[str, float | str]]:
    values = None
    if hasattr(model, "feature_importances_"):
        values = getattr(model, "feature_importances_", None)
    elif hasattr(model, "coef_"):
        coef = getattr(model, "coef_", None)
        if coef is not None:
            values = np.ravel(np.abs(coef))
    if values is None:
        return []
    ranked = sorted(zip(feature_names, values), key=lambda item: abs(float(item[1])), reverse=True)[:10]
    return [{"feature": feature, "importance": round(float(importance), 4)} for feature, importance in ranked]


def run_predictive_analysis(df: pd.DataFrame, task: Dict[str, Any], state_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    state_context = state_context or {}
    question = state_context.get("business_question", "")
    selected_columns = state_context.get("selected_columns", list(df.columns))
    parameters = task.get("parameters", {}) or {}
    target_column = parameters.get("target_column") or _infer_target_column(question, selected_columns, df)
    date_column = parameters.get("date_column") or detect_date_column(df)
    runtime_mode = determine_runtime_mode(state_context)
    df, runtime_report = apply_runtime_optimizations(df, target_column or "", date_column, runtime_mode)
    if date_column and date_column not in df.columns:
        date_column = detect_date_column(df)

    if not target_column:
        return {"tool": "predictive_analysis", "error": "Could not resolve a prediction target column."}

    problem_type = parameters.get("problem_type") or infer_problem_type(question, target_column, df, date_column)
    readiness_payload, leakage_columns = assess_readiness(df, target_column, problem_type, date_column=date_column)
    if any(item["severity"] == "high" and item["code"] in {"missing_target", "missing_date"} for item in readiness_payload):
        return {
            "tool": "predictive_analysis",
            "error": "Dataset is not ready for the requested predictive workflow.",
            "readiness_warnings": readiness_payload,
            "leakage_columns": leakage_columns,
        }

    X, y, metadata = build_feature_frame(df, target_column, problem_type, date_column, leakage_columns)
    if len(X) < 20:
        return {
            "tool": "predictive_analysis",
            "error": "Not enough usable rows remain after feature preparation.",
            "readiness_warnings": readiness_payload,
            "leakage_columns": leakage_columns,
        }
    if X.shape[1] == 0:
        return {
            "tool": "predictive_analysis",
            "error": "No predictor columns remain after feature preparation.",
            "readiness_warnings": readiness_payload,
            "leakage_columns": leakage_columns,
        }

    X_train, X_test, y_train, y_test = split_features(X, y, problem_type, date_column=date_column)
    if len(X_train) < 10 or len(X_test) < 3:
        return {
            "tool": "predictive_analysis",
            "error": "Not enough train/test coverage remains after deterministic splitting.",
            "readiness_warnings": readiness_payload,
            "leakage_columns": leakage_columns,
        }

    if problem_type == "classification" and pd.Series(y_train).nunique(dropna=True) < 2:
        return {
            "tool": "predictive_analysis",
            "error": "Classification target does not contain enough class variation after preprocessing.",
            "readiness_warnings": readiness_payload,
            "leakage_columns": leakage_columns,
        }

    model_results: List[ModelResult] = []
    best_score = float("-inf")
    best_metrics: Dict[str, float | None] = {}
    best_train_metrics: Dict[str, float | None] = {}
    best_cv_summary: Dict[str, Any] = {}
    best_baseline_metrics: Dict[str, float | None] = {}
    best_diagnostics: Dict[str, Any] = {}
    best_predictions: np.ndarray | None = None
    best_model_name: str | None = None
    best_feature_importance: List[Dict[str, float | str]] = []
    best_truthfulness_flags: List[str] = []
    best_no_reliable_recommendation = False

    for model_name, candidate in get_candidate_models(problem_type).items():
        model = clone(candidate)
        validation_notes: List[str] = []
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
        except Exception as exc:
            validation_notes.append(f"Model failed during fit/predict: {exc}")
            model_results.append(
                ModelResult(
                    model_name=model_name,
                    problem_type=problem_type,
                    metrics=PredictiveMetrics(values={}),
                    feature_importance=[],
                    validation_notes=validation_notes,
                )
            )
            continue
        score_vector = None
        train_score_vector = None
        if problem_type == "classification" and hasattr(model, "predict_proba"):
            try:
                score_vector = model.predict_proba(X_test)[:, 1]
            except Exception:
                score_vector = None
            try:
                train_score_vector = model.predict_proba(X_train)[:, 1]
            except Exception:
                train_score_vector = None
        train_predictions = model.predict(X_train)
        metrics = metric_bundle(problem_type, y_test, predictions, score_vector)
        train_metrics = metric_bundle(problem_type, y_train, train_predictions, train_score_vector)
        cv_summary = cross_validate_model(model, X_train, y_train, problem_type)
        baseline_pred, baseline_score, baseline_label = build_baseline_predictions(problem_type, y_train, y_test)
        baseline_metrics = metric_bundle(problem_type, y_test, baseline_pred, baseline_score)
        diagnostics = {
            "overfit_gap": overfit_gap(problem_type, train_metrics, metrics),
        }
        if problem_type in {"regression", "forecasting"}:
            diagnostics["residuals"] = residual_diagnostics(y_test, predictions)
        else:
            diagnostics["confusion_matrix"] = confusion_details(y_test, predictions)
        selection_metric = selection_score(problem_type, metrics)
        cv_primary = (cv_summary.get("mean", {}) or {}).get("r2" if problem_type in {"regression", "forecasting"} else "f1")
        stability_penalty = float((cv_summary.get("std", {}) or {}).get("r2" if problem_type in {"regression", "forecasting"} else "f1") or 0.0)
        combined_score = selection_metric + (float(cv_primary or 0.0) * 5.0) - (stability_penalty * 10.0)
        feature_importance = _extract_feature_importance(model, list(X.columns))
        weak_model = False
        truthfulness_flags: List[str] = []
        if problem_type in {"regression", "forecasting"}:
            if (metrics.get("r2") or 0.0) <= 0.1:
                weak_model = True
                truthfulness_flags.append("Explained variance is too weak for a dependable predictive recommendation.")
            if ((cv_summary.get("mean", {}) or {}).get("r2") or 0.0) <= 0.05:
                weak_model = True
                truthfulness_flags.append("Cross-validation shows unstable or weak generalization.")
            if (metrics.get("r2") or 0.0) <= (baseline_metrics.get("r2") or 0.0) + 0.02:
                weak_model = True
                truthfulness_flags.append("Model does not materially outperform the naive baseline.")
            if diagnostics["overfit_gap"] > 0.2:
                weak_model = True
                truthfulness_flags.append("Model appears overfit relative to holdout performance.")
        else:
            if (metrics.get("f1") or 0.0) <= (baseline_metrics.get("f1") or 0.0) + 0.03:
                weak_model = True
                truthfulness_flags.append("Classifier does not materially outperform the majority-class baseline.")
            if diagnostics["overfit_gap"] > 0.15:
                weak_model = True
                truthfulness_flags.append("Classifier appears overfit relative to holdout performance.")

        validation_summary = {
            "train_metrics": train_metrics,
            "test_metrics": metrics,
            "cross_validation": cv_summary,
            "baseline": {"model": baseline_label, "metrics": baseline_metrics},
            "diagnostics": diagnostics,
            "weak_model": weak_model,
        }
        model_results.append(
            ModelResult(
                model_name=model_name,
                problem_type=problem_type,
                metrics=PredictiveMetrics(values=metrics),
                feature_importance=feature_importance,
                validation_notes=validation_notes,
                validation_summary=validation_summary,
            )
        )
        if combined_score > best_score:
            best_score = combined_score
            best_metrics = metrics
            best_train_metrics = train_metrics
            best_cv_summary = cv_summary
            best_baseline_metrics = baseline_metrics
            best_diagnostics = diagnostics
            best_predictions = predictions
            best_model_name = model_name
            best_feature_importance = feature_importance
            best_truthfulness_flags = truthfulness_flags
            best_no_reliable_recommendation = weak_model

    if best_model_name is None or best_predictions is None:
        return {
            "tool": "predictive_analysis",
            "error": "No candidate model completed successfully on the prepared dataset.",
            "model_comparison": [item.model_dump() for item in model_results],
            "readiness_warnings": readiness_payload,
            "leakage_columns": leakage_columns,
        }

    preview = pd.DataFrame({"actual": y_test.reset_index(drop=True), "predicted": pd.Series(best_predictions).reset_index(drop=True)}).head(10)
    limitations = []
    if leakage_columns:
        limitations.append("Some candidate leakage columns were removed before model training.")
    if readiness_payload:
        limitations.extend(item["message"] for item in readiness_payload if item["severity"] in {"medium", "high"})
    limitations.extend(best_truthfulness_flags)

    confidence_payload = calibrate_confidence(
        problem_type=problem_type,
        sample_size=len(X),
        test_metrics=best_metrics,
        baseline_metrics=best_baseline_metrics,
        cv_summary=best_cv_summary,
        diagnostics={
            "overfit_gap": best_diagnostics.get("overfit_gap"),
            **(best_diagnostics.get("residuals", {}) or {}),
        },
        readiness_warnings=readiness_payload,
        weak_model=best_no_reliable_recommendation,
    )
    if best_no_reliable_recommendation:
        limitations.append("No reliable operational recommendation should be made until model quality improves.")

    monitoring = detect_data_drift(
        training_frame=X_train,
        scoring_frame=X_test,
        feature_columns=list(X_train.columns),
    )
    decay = performance_decay_monitor(best_metrics, best_baseline_metrics, problem_type)
    if decay["warnings"]:
        limitations.extend(decay["warnings"])

    result = PredictiveResult(
        problem_type=problem_type,
        target_column=target_column,
        chosen_model=best_model_name,
        model_comparison=model_results,
        readiness_warnings=[ReadinessWarning(**item) for item in readiness_payload],
        leakage_columns=leakage_columns,
        feature_columns=list(X.columns),
        top_drivers=best_feature_importance[:5],
        predictions_preview=preview.to_dict(orient="records"),
        metrics=PredictiveMetrics(values=best_metrics),
        confidence_level=confidence_payload["label"],
        confidence=ConfidenceAssessment(**confidence_payload),
        validation_summary={
            "train_metrics": best_train_metrics,
            "test_metrics": best_metrics,
            "cross_validation": best_cv_summary,
            "baseline_metrics": best_baseline_metrics,
            "diagnostics": best_diagnostics,
            "runtime_report": runtime_report,
            "monitoring": monitoring,
            "performance_decay": decay,
        },
        truthfulness_flags=best_truthfulness_flags,
        no_reliable_recommendation=best_no_reliable_recommendation,
        limitations=limitations,
    )
    return result.model_dump()
