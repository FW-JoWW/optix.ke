from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd


def determine_runtime_mode(state_context: Dict[str, Any] | None = None) -> str:
    state_context = state_context or {}
    question = str(state_context.get("business_question", "")).lower()
    explicit = str(state_context.get("runtime_mode", "")).lower()
    if explicit in {"exploratory", "full"}:
        return explicit
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

    if runtime_mode == "exploratory" and len(optimized) > 20000:
        if date_column and date_column in optimized.columns:
            optimized = optimized.sort_values(date_column).tail(20000).copy()
        else:
            optimized = optimized.sample(20000, random_state=42).copy()
        report["sampled"] = True
        report["row_count_after_sampling"] = int(len(optimized))

    drop_columns = []
    for column in optimized.columns:
        if column == target_column:
            continue
        non_null = optimized[column].dropna()
        if non_null.empty:
            drop_columns.append(column)
            continue
        unique_count = int(non_null.nunique())
        unique_ratio = float(unique_count / max(len(non_null), 1))
        if unique_count > 5000 and unique_ratio > 0.9:
            drop_columns.append(column)

    if drop_columns:
        optimized = optimized.drop(columns=drop_columns, errors="ignore")
        report["dropped_columns"] = drop_columns

    report["row_count_after"] = int(len(optimized))
    report["column_count_after"] = int(len(optimized.columns))
    return optimized, report
