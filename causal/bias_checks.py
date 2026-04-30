from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def _outlier_dominance(df: pd.DataFrame, x_col: str, y_col: str) -> bool:
    frame = pd.DataFrame({"x": pd.to_numeric(df[x_col], errors="coerce"), "y": pd.to_numeric(df[y_col], errors="coerce")}).dropna()
    if len(frame) < 12:
        return False
    base = frame["x"].corr(frame["y"], method="spearman")
    if pd.isna(base):
        return False
    trimmed = frame[
        frame["x"].between(frame["x"].quantile(0.05), frame["x"].quantile(0.95))
        & frame["y"].between(frame["y"].quantile(0.05), frame["y"].quantile(0.95))
    ]
    if len(trimmed) < 8:
        return False
    trimmed_corr = trimmed["x"].corr(trimmed["y"], method="spearman")
    if pd.isna(trimmed_corr):
        return False
    return abs(float(base) - float(trimmed_corr)) >= 0.2


def detect_simpsons_paradox(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    segment_columns: List[str],
) -> Dict[str, Any]:
    frame = pd.DataFrame({"x": pd.to_numeric(df[x_col], errors="coerce"), "y": pd.to_numeric(df[y_col], errors="coerce")}).dropna()
    if len(frame) < 20:
        return {"detected": False}
    overall = frame["x"].corr(frame["y"], method="spearman")
    if pd.isna(overall) or abs(float(overall)) < 0.1:
        return {"detected": False}
    overall_sign = 1 if overall > 0 else -1

    for column in segment_columns:
        if column not in df.columns:
            continue
        signs = []
        for _, group in df[[x_col, y_col, column]].dropna().groupby(column):
            if len(group) < 10:
                continue
            corr = pd.to_numeric(group[x_col], errors="coerce").corr(pd.to_numeric(group[y_col], errors="coerce"), method="spearman")
            if pd.isna(corr) or abs(float(corr)) < 0.1:
                continue
            signs.append(1 if corr > 0 else -1)
        if signs and all(sign != overall_sign for sign in signs):
            return {
                "detected": True,
                "segment_column": column,
                "overall_sign": overall_sign,
                "segment_signs": signs,
            }
    return {"detected": False}


def run_bias_checks(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    state_context: Dict[str, Any],
    temporal_signal: Dict[str, Any],
    confounders: List[Dict[str, Any]],
) -> Dict[str, Any]:
    warnings: List[str] = []
    bias_risks: List[str] = []
    dataset_profile = state_context.get("dataset_profile", {}) or {}
    selected_columns = state_context.get("analysis_evidence", {}).get("selected_columns") or state_context.get("selected_columns") or []

    subset = df[[x_col, y_col]].copy()
    missing_ratio = float(subset.isna().any(axis=1).mean()) if len(subset) else 0.0
    if missing_ratio > 0.2:
        bias_risks.append("missingness_bias")
        warnings.append("More than 20% of rows are incomplete for the relationship subset.")

    if len(subset.dropna()) < 30:
        bias_risks.append("small_sample_risk")
        warnings.append("Small sample size limits the reliability of relationship estimates.")

    full_row_count = int(dataset_profile.get("row_count") or len(df) or 0)
    subset_row_count = int(len(df))
    if full_row_count and subset_row_count / full_row_count < 0.5:
        bias_risks.append("selection_bias")
        warnings.append("The analysis subset is much smaller than the full dataset, so selection bias is plausible.")

    if _outlier_dominance(df, x_col, y_col):
        bias_risks.append("outlier_domination")
        warnings.append("The measured relationship changes materially when extreme values are trimmed.")

    segment_candidates = [
        column for column in dataset_profile.get("categorical_columns", []) or []
        if column not in {x_col, y_col} and column in df.columns
    ][:5]
    simpson = detect_simpsons_paradox(df, x_col, y_col, segment_candidates)
    if simpson.get("detected"):
        bias_risks.append("simpsons_paradox")
        warnings.append(f"Overall and within-segment relationships disagree when segmented by {simpson['segment_column']}.")

    derived_columns = (state_context.get("relationship_signals", {}) or {}).get("derived_columns", []) or []
    if x_col in derived_columns or y_col in derived_columns:
        bias_risks.append("leakage")
        warnings.append("One side of the relationship appears derived from other variables, so leakage is possible.")

    if temporal_signal.get("reverse_causality_warning"):
        bias_risks.append("reverse_causality")
        warnings.append(temporal_signal["reverse_causality_warning"])

    if confounders:
        warnings.append("Candidate confounders were detected and should be considered before causal claims.")

    return {
        "bias_risks": bias_risks,
        "simpsons_paradox_detected": bool(simpson.get("detected")),
        "outlier_domination_detected": "outlier_domination" in bias_risks,
        "small_sample_risk": "small_sample_risk" in bias_risks,
        "missingness_bias_risk": "missingness_bias" in bias_risks,
        "selection_bias_risk": "selection_bias" in bias_risks,
        "leakage_risk": "leakage" in bias_risks,
        "warnings": warnings,
    }
