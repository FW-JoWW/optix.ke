from __future__ import annotations

from typing import Any, Dict, List


def _label(score: int) -> str:
    if score >= 75:
        return "high"
    if score >= 45:
        return "moderate"
    return "low"


def calibrate_confidence(
    problem_type: str,
    sample_size: int,
    test_metrics: Dict[str, float | None],
    baseline_metrics: Dict[str, float | None],
    cv_summary: Dict[str, Any],
    diagnostics: Dict[str, Any],
    readiness_warnings: List[Dict[str, str]],
    weak_model: bool,
    drift_summary: Dict[str, Any] | None = None,
    feature_diagnostics: Dict[str, Any] | None = None,
    scenario_uncertainty: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    score = 55
    reasons: List[str] = []
    drift_summary = drift_summary or {}
    feature_diagnostics = feature_diagnostics or {}
    scenario_uncertainty = scenario_uncertainty or {}

    if sample_size >= 500:
        score += 12
        reasons.append("Large sample size supports more stable estimates.")
    elif sample_size >= 120:
        score += 6
        reasons.append("Sample size is adequate for a holdout and cross-validation read.")
    else:
        score -= 10
        reasons.append("Limited sample size reduces trust in model stability.")

    if problem_type in {"regression", "forecasting"}:
        test_r2 = float(test_metrics.get("r2") or 0.0)
        base_r2 = float(baseline_metrics.get("r2") or 0.0)
        improvement = test_r2 - base_r2
        if improvement >= 0.15:
            score += 10
            reasons.append("Model materially outperforms the naive baseline.")
        elif improvement <= 0.02:
            score -= 12
            reasons.append("Model does not materially outperform the naive baseline.")
        interval_width = diagnostics.get("relative_interval_width")
        if interval_width is not None:
            if interval_width <= 0.75:
                score += 8
                reasons.append("Prediction intervals are reasonably tight relative to target variation.")
            elif interval_width > 1.5:
                score -= 10
                reasons.append("Prediction intervals are wide, indicating substantial uncertainty.")
    else:
        test_f1 = float(test_metrics.get("f1") or 0.0)
        base_f1 = float(baseline_metrics.get("f1") or 0.0)
        if test_f1 - base_f1 >= 0.1:
            score += 10
            reasons.append("Classifier meaningfully improves on the majority-class baseline.")
        elif test_f1 - base_f1 <= 0.03:
            score -= 12
            reasons.append("Classifier is only marginally better than the baseline.")
        calibration_gap = test_metrics.get("calibration_gap")
        if calibration_gap is not None:
            if float(calibration_gap) <= 0.05:
                score += 6
                reasons.append("Predicted probabilities are reasonably calibrated.")
            elif float(calibration_gap) > 0.12:
                score -= 8
                reasons.append("Probability calibration is weak, so action confidence should be reduced.")

    cv_mean = cv_summary.get("mean", {}) or {}
    cv_std = cv_summary.get("std", {}) or {}
    primary_metric = "r2" if problem_type in {"regression", "forecasting"} else "f1"
    primary_cv_std = cv_std.get(primary_metric)
    if primary_cv_std is not None:
        if float(primary_cv_std) <= 0.05:
            score += 8
            reasons.append("Cross-validation performance is stable across folds.")
        elif float(primary_cv_std) > 0.15:
            score -= 12
            reasons.append("Cross-validation varies materially across folds.")

    overfit_gap = diagnostics.get("overfit_gap")
    if overfit_gap is not None:
        if float(overfit_gap) > 0.22:
            score -= 18
            reasons.append("Train-vs-test gap is large enough to make the model operationally fragile.")
        elif float(overfit_gap) > 0.15:
            score -= 12
            reasons.append("Train-vs-test gap suggests meaningful overfitting.")
        elif float(overfit_gap) <= 0.05:
            score += 4
            reasons.append("Train-vs-test performance gap is controlled.")

    residual_skew = diagnostics.get("residual_skew")
    if residual_skew is not None:
        skew = abs(float(residual_skew))
        if skew > 1.0:
            score -= 10
            reasons.append("Residual skew suggests the model error profile is asymmetric and less stable in deployment.")
        elif skew > 0.6:
            score -= 5
            reasons.append("Residual skew is noticeable, so confidence is reduced modestly.")

    top_driver_share = float(feature_diagnostics.get("top_driver_share", 0.0) or 0.0)
    concentration_score = float(feature_diagnostics.get("driver_concentration_score", 0.0) or 0.0)
    if top_driver_share >= 0.6:
        score -= 12
        reasons.append("The model depends heavily on a single dominant driver, which makes operational recommendations less diversified and less stable.")
    elif concentration_score >= 0.45:
        score -= 6
        reasons.append("Driver influence is concentrated in a narrow set of features.")

    average_drift = float(drift_summary.get("average_drift_score", 0.0) or 0.0)
    if average_drift >= 0.2:
        score -= 8
        reasons.append("Feature drift between train and score windows reduces reliability.")
    elif average_drift >= 0.1:
        score -= 4
        reasons.append("Mild drift is present, so monitoring should stay active.")

    scenario_spread = float(scenario_uncertainty.get("spread_ratio", 0.0) or 0.0)
    if scenario_spread >= 1.2:
        score -= 10
        reasons.append("Scenario uncertainty is wide, so operational confidence should be lower than predictive fit alone suggests.")
    elif scenario_spread >= 0.7:
        score -= 5
        reasons.append("Scenario outcomes vary materially, so action confidence should remain moderated.")

    penalty = sum(8 if item.get("severity") == "high" else 4 for item in readiness_warnings if item.get("severity") in {"medium", "high"})
    score -= penalty
    if penalty:
        reasons.append("Data-readiness warnings lowered confidence.")

    if weak_model:
        score = min(score, 40)
        reasons.append("Weak or unstable model behavior triggered a confidence cap.")

    score = max(0, min(100, int(round(score))))
    return {
        "score": score,
        "label": _label(score),
        "explanation": " ".join(reasons) if reasons else "Confidence is based on validation stability, benchmark performance, and data quality.",
        "factors": {
            "sample_size": sample_size,
            "baseline_comparison": {
                "test_metrics": test_metrics,
                "baseline_metrics": baseline_metrics,
            },
            "cross_validation": cv_summary,
            "diagnostics": diagnostics,
            "feature_diagnostics": feature_diagnostics,
            "drift_summary": drift_summary,
            "scenario_uncertainty": scenario_uncertainty,
            "weak_model": weak_model,
        },
    }
