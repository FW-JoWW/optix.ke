from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from utils.numeric_parsing import normalize_numeric_token


def _normalize_categories(series: pd.Series) -> pd.Series:
    cleaned = series.astype("object").where(series.notna(), None)
    stripped = cleaned.map(lambda value: str(value).strip() if value is not None else value)
    canonical: Dict[str, str] = {}

    def _canon(value: Any) -> Any:
        if value is None:
            return None
        key = str(value).lower()
        canonical.setdefault(key, str(value).strip())
        return canonical[key]

    return stripped.map(_canon)


def _convert_type(series: pd.Series, inferred_type: str) -> pd.Series:
    if inferred_type == "numeric":
        return pd.to_numeric(series.map(normalize_numeric_token), errors="coerce")
    if inferred_type == "datetime":
        return pd.to_datetime(series, errors="coerce")
    return series


def _impute_numeric(series: pd.Series, strategy: str) -> pd.Series:
    numeric = pd.to_numeric(series.map(normalize_numeric_token), errors="coerce")
    if strategy == "mean":
        fill_value = numeric.mean()
    elif strategy == "median":
        fill_value = numeric.median()
    else:
        fill_value = None

    if pd.isna(fill_value):
        return numeric
    return numeric.fillna(fill_value)


def _impute_categorical(series: pd.Series, strategy: str) -> pd.Series:
    cleaned = series.astype("object")
    if strategy == "mode":
        mode = cleaned.dropna().mode()
        if mode.empty:
            return cleaned
        return cleaned.fillna(mode.iloc[0])
    if strategy == "forward_fill":
        return cleaned.ffill()
    if strategy == "backward_fill":
        return cleaned.bfill()
    return cleaned


def _recompute_if_possible(
    df: pd.DataFrame,
    column: str,
    relationships: List[Dict[str, Any]],
) -> pd.Series:
    for rel in relationships:
        if rel.get("target_column") != column:
            continue
        operator = rel.get("operator")
        left = rel.get("source_columns", [None, None])[0]
        right = rel.get("source_columns", [None, None])[-1]
        if left not in df.columns or right not in df.columns:
            continue
        if operator == "multiply":
            return df[left] * df[right]
        if operator == "add":
            return df[left] + df[right]
    return df[column]


def execute_cleaning_actions(
    df: pd.DataFrame,
    actions: List[Dict[str, Any]],
    dataset_profile: Dict[str, Any],
    relationships: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    cleaned = df.copy()
    execution_log: List[Dict[str, Any]] = []
    relationships = relationships or []
    column_info = dataset_profile.get("columns", {})

    for item in actions:
        column = item.get("column")
        action = item.get("action")

        if action == "leave_unchanged":
            execution_log.append({"column": column, "action": action, "status": "skipped"})
            continue

        if column not in cleaned.columns:
            execution_log.append({"column": column, "action": action, "status": "missing_column"})
            continue

        if action == "forward_fill":
            cleaned[column] = cleaned[column].ffill()
        elif action == "backward_fill":
            cleaned[column] = cleaned[column].bfill()
        elif action == "impute_mean":
            cleaned[column] = _impute_numeric(cleaned[column], "mean")
        elif action == "impute_median":
            cleaned[column] = _impute_numeric(cleaned[column], "median")
        elif action == "impute_mode":
            cleaned[column] = _impute_categorical(cleaned[column], "mode")
        elif action == "remove_duplicates":
            cleaned = cleaned.drop_duplicates().copy()
        elif action == "investigate_or_cap" and pd.api.types.is_numeric_dtype(cleaned[column]):
            q1 = cleaned[column].quantile(0.25)
            q3 = cleaned[column].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            cleaned[column] = cleaned[column].clip(lower=lower, upper=upper)
        elif action in {"review_only", "leave_unchanged"}:
            pass
        elif action == "drop_rows":
            cleaned = cleaned[cleaned[column].notna()].copy()
        elif action == "convert_type":
            inferred_type = column_info.get(column, {}).get("inferred_type", "unknown")
            cleaned[column] = _convert_type(cleaned[column], inferred_type)
        elif action == "standardize_categories":
            cleaned[column] = _normalize_categories(cleaned[column])
        elif action == "recompute_if_possible":
            cleaned[column] = _recompute_if_possible(cleaned, column, relationships)

        execution_log.append({"column": column, "action": action, "status": "applied"})

    return {
        "cleaned_df": cleaned,
        "execution_log": execution_log,
    }
