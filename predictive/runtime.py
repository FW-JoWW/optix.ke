from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd


def determine_runtime_mode(state_context: Dict[str, Any] | None = None) -> str:
    state_context = state_context or {}
    question = str(state_context.get("business_question", "")).lower()
    explicit = str(state_context.get("runtime_mode", "")).lower()
    if explicit in {"exploratory", "full", "fast_enterprise"}:
        return explicit
    if any(token in question for token in ["fast enterprise", "fast-enterprise", "accelerated final"]):
        return "fast_enterprise"
    if any(token in question for token in ["quick", "fast", "explor", "sample"]):
        return "exploratory"
    return "full"


def apply_runtime_optimizations(
    df: pd.DataFrame,
    target_column: str,
    date_column: str | None,
    runtime_mode: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    optimized = df.copy()
    report: Dict[str, Any] = {
        "runtime_mode": runtime_mode,
        "row_count_before": int(len(df)),
        "column_count_before": int(len(df.columns)),
        "sampled": False,
        "dropped_columns": [],
    }

    row_limit = 20000 if runtime_mode == "exploratory" else 40000 if runtime_mode == "fast_enterprise" else None
    if row_limit is not None and len(optimized) > row_limit:
        if date_column and date_column in optimized.columns:
            optimized = optimized.sort_values(date_column).tail(row_limit).copy()
        else:
            optimized = optimized.sample(row_limit, random_state=42).copy()
        report["sampled"] = True
        report["row_count_after_sampling"] = int(len(optimized))

    drop_columns = []
    for column in optimized.columns:
        if column == target_column:
            continue
        if date_column and column == date_column:
            continue
        non_null = optimized[column].dropna()
        if non_null.empty:
            drop_columns.append(column)
            continue
        if pd.api.types.is_numeric_dtype(optimized[column]):
            continue
        if pd.api.types.is_datetime64_any_dtype(optimized[column]):
            continue
        unique_count = int(non_null.nunique())
        unique_ratio = float(unique_count / max(len(non_null), 1))
        is_identifier_like = (
            pd.api.types.is_object_dtype(optimized[column])
            or pd.api.types.is_string_dtype(optimized[column])
        )
        if is_identifier_like and unique_count > 5000 and unique_ratio > 0.9:
            drop_columns.append(column)

    if drop_columns:
        optimized = optimized.drop(columns=drop_columns, errors="ignore")
        report["dropped_columns"] = drop_columns

    report["row_count_after"] = int(len(optimized))
    report["column_count_after"] = int(len(optimized.columns))
    return optimized, report
