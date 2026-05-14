from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd


def assess_readiness(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    date_column: str | None = None,
) -> Tuple[List[Dict[str, str]], List[str]]:
    warnings: List[Dict[str, str]] = []
    leakage_columns: List[str] = []

    if target_column not in df.columns:
        warnings.append({"code": "missing_target", "message": f"Target column '{target_column}' is not present.", "severity": "high"})
        return warnings, leakage_columns

    if len(df) < 40:
        warnings.append({"code": "too_few_rows", "message": "Too few rows for a reliable predictive workflow.", "severity": "high"})
    elif len(df) < 120:
        warnings.append({"code": "limited_rows", "message": "Row count is limited; model estimates may be unstable.", "severity": "medium"})

    if df.duplicated().mean() > 0.1:
        warnings.append({"code": "duplicate_records", "message": "Dataset contains a high duplicate-record ratio.", "severity": "medium"})

    target_missing = float(df[target_column].isna().mean())
    if target_missing > 0.05:
        warnings.append({"code": "target_missingness", "message": "Target column has material missingness.", "severity": "high"})

    non_null_target = df[target_column].dropna()
    if non_null_target.nunique() <= 1:
        warnings.append({"code": "zero_variance_target", "message": "Target column has no usable variation for modeling.", "severity": "high"})

    feature_missing = float(df.drop(columns=[target_column]).isna().mean().mean()) if len(df.columns) > 1 else 0.0
    if feature_missing > 0.35:
        warnings.append({"code": "severe_feature_missingness", "message": "Feature matrix has severe missingness.", "severity": "high"})
    elif feature_missing > 0.15:
        warnings.append({"code": "moderate_feature_missingness", "message": "Feature matrix has moderate missingness.", "severity": "medium"})

    if problem_type == "classification":
        target_distribution = df[target_column].dropna().value_counts(normalize=True)
        if not target_distribution.empty and float(target_distribution.max()) > 0.9:
            warnings.append({"code": "class_imbalance", "message": "Classification target is extremely imbalanced.", "severity": "high"})
        if non_null_target.nunique() > max(25, int(len(non_null_target) * 0.2)):
            warnings.append({"code": "too_many_classes", "message": "Classification target has too many distinct classes for a stable generic classifier.", "severity": "high"})

    if problem_type == "forecasting" and not date_column:
        warnings.append({"code": "missing_date", "message": "Forecasting requires a usable date or time column.", "severity": "high"})

    target_base = str(target_column).lower()
    for column in df.columns:
        if column == target_column:
            continue
        column_lower = str(column).lower()
        if target_base and target_base in column_lower:
            leakage_columns.append(column)

    numeric_target = pd.to_numeric(df[target_column], errors="coerce")
    if numeric_target.notna().mean() > 0.8:
        correlations = df.select_dtypes(include="number").corr(numeric_only=True).get(target_column)
        if correlations is not None:
            for column, corr_value in correlations.drop(labels=[target_column], errors="ignore").items():
                if pd.notna(corr_value) and abs(float(corr_value)) > 0.995:
                    leakage_columns.append(column)

    return warnings, sorted(set(leakage_columns))
