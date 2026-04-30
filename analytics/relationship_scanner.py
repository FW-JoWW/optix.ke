from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer

from analytics.correlation_engine import run_smart_correlation
from analytics.temporal_analysis import analyze_temporal_precedence
from causal.bias_checks import run_bias_checks
from causal.causal_score import grade_causal_evidence
from causal.confounder_detector import detect_confounders
from causal.experiment_recommender import recommend_next_step
from core.output_contracts import (
    BiasCheckResult,
    CausalEvidenceResult,
    ConfounderCandidate,
    NonlinearSignal,
    PartialCorrelationResult,
    RelationshipEvidenceReport,
    RelationshipStats,
    SegmentRelationship,
    TemporalSignal,
)


def _kind(column: str, df: pd.DataFrame, state_context: Dict[str, Any]) -> str:
    dataset_profile = state_context.get("dataset_profile", {}) or {}
    column_registry = state_context.get("column_registry", {}) or {}
    series = df[column]
    if column in dataset_profile.get("numeric_columns", []):
        unique_count = int(series.dropna().nunique())
        if unique_count == 2:
            return "binary"
        return "numeric"
    if column in dataset_profile.get("categorical_columns", []):
        unique_count = int(series.dropna().nunique())
        if unique_count == 2:
            return "binary"
        return "categorical"
    role = (column_registry.get(column, {}) or {}).get("semantic_role")
    if role in {"numeric_measure", "derived_metric"}:
        unique_count = int(series.dropna().nunique())
        if unique_count == 2:
            return "binary"
        return "numeric"
    unique_count = int(series.dropna().nunique())
    if pd.api.types.is_numeric_dtype(series):
        return "binary" if unique_count == 2 else "numeric"
    return "binary" if unique_count == 2 else "categorical"


def _partial_corr(frame: pd.DataFrame, x_col: str, y_col: str, controls: List[str]) -> PartialCorrelationResult:
    if not controls:
        return PartialCorrelationResult(computed=False, note="No confounders were selected for control.")
    usable = [col for col in controls if col in frame.columns]
    if not usable:
        return PartialCorrelationResult(computed=False, controls=controls, note="Requested controls were not present in the analysis frame.")

    work = frame[[x_col, y_col, *usable]].copy().dropna()
    if len(work) < 10:
        return PartialCorrelationResult(computed=False, controls=usable, sample_size=len(work), note="Too few complete rows for partial correlation.")

    control_matrix = pd.get_dummies(work[usable], drop_first=True)
    if control_matrix.empty:
        return PartialCorrelationResult(computed=False, controls=usable, sample_size=len(work), note="Control variables did not produce a usable design matrix.")

    design = np.column_stack([np.ones(len(control_matrix)), control_matrix.to_numpy(dtype=float)])
    x_values = pd.to_numeric(work[x_col], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(work[y_col], errors="coerce").to_numpy(dtype=float)
    beta_x, *_ = np.linalg.lstsq(design, x_values, rcond=None)
    beta_y, *_ = np.linalg.lstsq(design, y_values, rcond=None)
    resid_x = x_values - design @ beta_x
    resid_y = y_values - design @ beta_y
    if len(resid_x) < 3:
        return PartialCorrelationResult(computed=False, controls=usable, sample_size=len(work), note="Too few residual observations after control adjustment.")
    coeff, p_value = stats.pearsonr(resid_x, resid_y)
    return PartialCorrelationResult(
        computed=True,
        controls=usable,
        coefficient=float(coeff),
        p_value=float(p_value),
        sample_size=len(work),
        note="Computed by correlating residualized variables after linear control adjustment.",
    )


def _segment_correlations(df: pd.DataFrame, x_col: str, y_col: str, state_context: Dict[str, Any]) -> List[SegmentRelationship]:
    dataset_profile = state_context.get("dataset_profile", {}) or {}
    candidates = [
        col for col in dataset_profile.get("categorical_columns", []) or []
        if col not in {x_col, y_col} and col in df.columns and df[col].dropna().nunique() <= 8
    ][:2]
    results: List[SegmentRelationship] = []
    for column in candidates:
        for group_value, group in df[[x_col, y_col, column]].dropna().groupby(column):
            if len(group) < 12:
                continue
            corr = pd.to_numeric(group[x_col], errors="coerce").corr(pd.to_numeric(group[y_col], errors="coerce"), method="spearman")
            if pd.isna(corr):
                continue
            results.append(
                SegmentRelationship(
                    segment_column=column,
                    segment_value=str(group_value),
                    method_used="spearman",
                    coefficient=float(corr),
                    p_value=None,
                    sample_size=int(len(group)),
                )
            )
    return results[:12]


def _nonlinear_signal(df: pd.DataFrame, x_col: str, y_col: str) -> NonlinearSignal:
    frame = pd.DataFrame({"x": pd.to_numeric(df[x_col], errors="coerce"), "y": pd.to_numeric(df[y_col], errors="coerce")}).dropna()
    if len(frame) < 20:
        return NonlinearSignal(detected=False, warning="Too few observations for nonlinear diagnostics.")
    if frame["x"].nunique() < 5 or frame["y"].nunique() < 5:
        return NonlinearSignal(detected=False, warning="Too little numeric variation for nonlinear diagnostics.")

    x_values = frame["x"].to_numpy(dtype=float).reshape(-1, 1)
    y_values = frame["y"].to_numpy(dtype=float)
    base_corr = abs(float(frame["x"].corr(frame["y"], method="pearson") or 0.0))
    try:
        mi = float(mutual_info_regression(x_values, y_values, random_state=42)[0])
        forest = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=4)
        forest.fit(x_values, y_values)
        tree_importance = float(forest.feature_importances_[0])

        linear_fit = np.polyfit(frame["x"], frame["y"], 1)
        linear_pred = np.polyval(linear_fit, frame["x"])
        ss_res_linear = float(((frame["y"] - linear_pred) ** 2).sum())
        ss_tot = float(((frame["y"] - frame["y"].mean()) ** 2).sum()) or 1.0
        linear_r2 = 1.0 - ss_res_linear / ss_tot

        poly_fit = np.polyfit(frame["x"], frame["y"], 2)
        poly_pred = np.polyval(poly_fit, frame["x"])
        ss_res_poly = float(((frame["y"] - poly_pred) ** 2).sum())
        poly_r2 = 1.0 - ss_res_poly / ss_tot
        gain = float(poly_r2 - linear_r2)
    except Exception:
        return NonlinearSignal(detected=False, warning="Nonlinear diagnostics could not be estimated reliably.")

    detected = base_corr < 0.2 and (mi >= 0.08 or gain >= 0.08)
    warning = None
    if detected:
        warning = "Linear correlation is weak, but nonlinear diagnostics suggest a relationship may still exist."

    return NonlinearSignal(
        detected=detected,
        mutual_information=round(mi, 4),
        tree_importance=round(tree_importance, 4),
        polynomial_gain_r2=round(gain, 4),
        warning=warning,
    )


def _confidence_score(
    stats_result: Dict[str, Any],
    bias_result: Dict[str, Any],
    confounders: List[Dict[str, Any]],
    temporal_signal: Dict[str, Any],
    segment_results: List[SegmentRelationship],
) -> int:
    score = 55
    sample_size = int(stats_result.get("sample_size") or 0)
    coefficient = abs(float(stats_result.get("coefficient") or 0.0))
    p_value = stats_result.get("p_value")

    if sample_size >= 500:
        score += 15
    elif sample_size >= 100:
        score += 8
    elif sample_size < 30:
        score -= 12

    if p_value is not None and p_value < 0.01:
        score += 10
    elif p_value is not None and p_value >= 0.1:
        score -= 10

    if coefficient >= 0.6:
        score += 8
    elif coefficient < 0.15:
        score -= 8

    score -= 6 * len(bias_result.get("bias_risks", []))
    score -= 3 * min(len(confounders), 3)

    if temporal_signal.get("lag_direction") == "x_precedes_y":
        score += 6
    elif temporal_signal.get("lag_direction") == "y_precedes_x":
        score -= 8

    if segment_results:
        signs = {1 if float(item.coefficient or 0) > 0 else -1 for item in segment_results if item.coefficient is not None and abs(float(item.coefficient)) >= 0.1}
        if len(signs) == 1:
            score += 6
        elif len(signs) > 1:
            score -= 6

    return max(0, min(100, int(round(score))))


def _relationship_label(method_used: str, coefficient: Optional[float]) -> str:
    if coefficient is None:
        return "No reliable relationship could be estimated."
    strength = abs(float(coefficient))
    if method_used in {"cramers_v", "phi"}:
        label = "categorical association"
    else:
        label = "relationship"
    if strength >= 0.6:
        return f"Strong {label} detected."
    if strength >= 0.3:
        return f"Moderate {label} detected."
    return f"Weak {label} detected."


def analyze_relationship_evidence(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    question: str,
    state_context: Dict[str, Any],
) -> Dict[str, Any]:
    x_kind = _kind(x_col, df, state_context)
    y_kind = _kind(y_col, df, state_context)
    stats_result = run_smart_correlation(df[x_col], df[y_col], x_kind=x_kind, y_kind=y_kind)

    confounders = []
    nonlinear = None
    partial = None
    temporal = {"applicable": False}
    segments: List[SegmentRelationship] = []
    bias_payload = {"bias_risks": [], "warnings": []}

    if x_kind == "numeric" and y_kind == "numeric":
        work = df[[x_col, y_col, *[c for c in df.columns if c not in {x_col, y_col}]]].copy()
        confounders = detect_confounders(df, x_col, y_col, state_context)
        controls = [item["column"] for item in confounders[:3]]
        partial = _partial_corr(work, x_col, y_col, controls)
        segments = _segment_correlations(df, x_col, y_col, state_context)
        temporal = analyze_temporal_precedence(df, x_col, y_col, state_context)
        nonlinear = _nonlinear_signal(df, x_col, y_col)
        bias_payload = run_bias_checks(df, x_col, y_col, state_context, temporal, confounders)

    causal = grade_causal_evidence(
        stats_result=stats_result,
        temporal_signal=temporal,
        confounders=confounders,
        segment_correlations=[item.model_dump() for item in segments],
        bias_result=bias_payload,
        nonlinear_signal=(nonlinear.model_dump() if nonlinear else {}),
        question=question,
    )
    recommended_next_step = recommend_next_step(
        causal_grade=causal["grade"],
        temporal_signal=temporal,
        confounders=confounders,
        bias_risks=bias_payload.get("bias_risks", []),
        question=question,
    )
    confidence = _confidence_score(stats_result, bias_payload, confounders, temporal, segments)

    assumptions = list(stats_result.get("warnings", []))
    if partial and partial.note:
        assumptions.append(partial.note)
    if nonlinear and nonlinear.warning:
        assumptions.append(nonlinear.warning)

    summary_parts = [_relationship_label(str(stats_result.get("method_used")), stats_result.get("coefficient"))]
    if confounders:
        summary_parts.append(f"Likely confounders include {', '.join(item['column'] for item in confounders[:3])}.")
    if temporal.get("reverse_causality_warning"):
        summary_parts.append(temporal["reverse_causality_warning"])
    summary_parts.append(f"Causal evidence is {causal['grade']}.")

    report = RelationshipEvidenceReport(
        question=question,
        relationship_found=_relationship_label(str(stats_result.get("method_used")), stats_result.get("coefficient")),
        method_used=str(stats_result.get("method_used")),
        stats=RelationshipStats(
            coefficient=stats_result.get("coefficient"),
            p_value=stats_result.get("p_value"),
            sample_size=int(stats_result.get("sample_size") or 0),
            test_statistic=stats_result.get("test_statistic"),
            confidence_interval_95=stats_result.get("confidence_interval_95"),
            additional_metrics={
                "x_kind": x_kind,
                "y_kind": y_kind,
                **({"contingency_table": stats_result.get("contingency_table")} if stats_result.get("contingency_table") else {}),
            },
        ),
        assumptions=assumptions,
        assumption_checks_passed=bool(stats_result.get("assumptions_met")),
        warnings=bias_payload.get("warnings", []),
        bias_risks=bias_payload.get("bias_risks", []),
        confounders=[ConfounderCandidate(**item) for item in confounders],
        nonlinear_signal=nonlinear,
        partial_correlation=partial,
        segment_correlations=segments,
        temporal_signal=TemporalSignal(**temporal),
        causal_evidence=CausalEvidenceResult(**causal),
        recommended_next_step=recommended_next_step,
        confidence=confidence,
        human_summary=" ".join(summary_parts),
        structured_summary={
            "x_column": x_col,
            "y_column": y_col,
            "x_kind": x_kind,
            "y_kind": y_kind,
        },
    )
    return report.model_dump()
