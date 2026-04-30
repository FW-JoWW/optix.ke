from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def _strength_numeric(series_a: pd.Series, series_b: pd.Series) -> float:
    frame = pd.DataFrame({"a": pd.to_numeric(series_a, errors="coerce"), "b": pd.to_numeric(series_b, errors="coerce")}).dropna()
    if len(frame) < 5:
        return 0.0
    corr = frame["a"].corr(frame["b"], method="spearman")
    return abs(float(corr)) if pd.notna(corr) else 0.0


def _strength_categorical(cat_series: pd.Series, num_series: pd.Series) -> float:
    frame = pd.DataFrame({"cat": cat_series.astype(str), "num": pd.to_numeric(num_series, errors="coerce")}).dropna()
    if len(frame) < 10 or frame["cat"].nunique() < 2:
        return 0.0
    grouped = frame.groupby("cat")["num"].mean()
    if len(grouped) < 2:
        return 0.0
    return min(abs(float(grouped.max() - grouped.min())) / max(float(frame["num"].std(ddof=0) or 1.0), 1.0), 1.0)


def detect_confounders(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    state_context: Dict[str, Any],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    dataset_profile = state_context.get("dataset_profile", {}) or {}
    column_registry = state_context.get("column_registry", {}) or {}
    candidates: List[Dict[str, Any]] = []
    numeric_columns = set(dataset_profile.get("numeric_columns", []) or [])
    categorical_columns = set(dataset_profile.get("categorical_columns", []) or [])

    for column in df.columns:
        if column in {x_col, y_col}:
            continue
        role = (column_registry.get(column, {}) or {}).get("semantic_role")
        if role == "identifier":
            continue
        if column in numeric_columns:
            x_strength = _strength_numeric(df[column], df[x_col])
            y_strength = _strength_numeric(df[column], df[y_col])
        elif column in categorical_columns:
            x_strength = _strength_categorical(df[column], df[x_col])
            y_strength = _strength_categorical(df[column], df[y_col])
        else:
            continue
        combined = (x_strength + y_strength) / 2.0
        if min(x_strength, y_strength) < 0.1 or combined < 0.15:
            continue
        candidates.append(
            {
                "column": column,
                "score": round(min(combined, 1.0), 4),
                "relates_to_x": round(x_strength, 4),
                "relates_to_y": round(y_strength, 4),
                "reason": "Variable shows material association with both sides of the relationship.",
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:limit]
