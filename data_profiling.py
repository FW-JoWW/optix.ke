from __future__ import annotations

import re
import warnings
from difflib import SequenceMatcher
from typing import Any, Dict, List

import pandas as pd

from utils.numeric_parsing import normalize_numeric_token


def _non_null_ratio(mask: pd.Series) -> float:
    total = len(mask)
    if total == 0:
        return 0.0
    return float(mask.sum() / total)


def _numeric_ratio(series: pd.Series) -> float:
    parsed = pd.to_numeric(series.map(normalize_numeric_token), errors="coerce")
    return _non_null_ratio(parsed.notna())


def _datetime_ratio(series: pd.Series) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, errors="coerce")
    return _non_null_ratio(parsed.notna())


def _base_name(column: str) -> str:
    lowered = str(column).strip().lower()
    lowered = re.sub(r"[\s\-]+", "_", lowered)
    lowered = re.sub(r"_+\d+$", "", lowered)
    return lowered


def _infer_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    numeric_ratio = _numeric_ratio(series)
    datetime_ratio = _datetime_ratio(series)
    unique_ratio = float(series.nunique(dropna=True) / max(len(series), 1))

    if numeric_ratio >= 0.8:
        return "numeric"
    if datetime_ratio >= 0.8:
        return "datetime"
    if unique_ratio >= 0.95:
        return "identifier_like"
    return "categorical"


def _numeric_summary(series: pd.Series) -> Dict[str, Any]:
    parsed = pd.to_numeric(series.map(normalize_numeric_token), errors="coerce").dropna()
    if parsed.empty:
        return {}
    return {
        "min": float(parsed.min()),
        "max": float(parsed.max()),
        "mean": float(parsed.mean()),
        "median": float(parsed.median()),
        "std": float(parsed.std(ddof=0)) if len(parsed) > 1 else 0.0,
    }


def _top_value_patterns(series: pd.Series, limit: int = 5) -> List[Dict[str, Any]]:
    values = series.dropna().astype(str)
    if values.empty:
        return []
    counts = values.value_counts().head(limit)
    return [
        {"value": idx, "count": int(count), "ratio": round(float(count / max(len(series), 1)), 4)}
        for idx, count in counts.items()
    ]


def _detect_similar_columns(df: pd.DataFrame, threshold: float = 0.82) -> List[Dict[str, Any]]:
    columns = list(df.columns)
    groups: List[Dict[str, Any]] = []
    seen_pairs = set()

    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            key = tuple(sorted((left, right)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            score = SequenceMatcher(None, _base_name(left), _base_name(right)).ratio()
            if _base_name(left) == _base_name(right):
                score = 1.0

            if score >= threshold:
                groups.append(
                    {
                        "columns": [left, right],
                        "similarity": round(float(score), 4),
                        "reason": "name_similarity",
                    }
                )

    return groups


def _detect_sparsity_patterns(df: pd.DataFrame, limit: int = 10) -> Dict[str, Any]:
    row_patterns = df.isna().astype(int).astype(str).agg("".join, axis=1)
    top_patterns = row_patterns.value_counts().head(limit)
    return {
        "column_missing_ratios": {
            col: round(float(df[col].isna().mean()), 4) for col in df.columns
        },
        "row_pattern_examples": [
            {"pattern": pattern, "count": int(count)}
            for pattern, count in top_patterns.items()
        ],
    }


def _detect_repeated_row_blocks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    patterns = df.isna().astype(int).astype(str).agg("".join, axis=1)
    blocks: List[Dict[str, Any]] = []

    start = 0
    while start < len(patterns):
        end = start + 1
        while end < len(patterns) and patterns.iloc[end] == patterns.iloc[start]:
            end += 1
        if end - start >= 3:
            blocks.append(
                {
                    "start_row": int(start),
                    "end_row": int(end - 1),
                    "length": int(end - start),
                    "pattern": patterns.iloc[start],
                }
            )
        start = end

    return blocks


def profile_dataset(df: pd.DataFrame, sample_rows: int = 5) -> Dict[str, Any]:
    columns: Dict[str, Any] = {}

    for col in df.columns:
        series = df[col]
        inferred_type = _infer_type(series)
        columns[col] = {
            "inferred_type": inferred_type,
            "dtype": str(series.dtype),
            "missing_ratio": round(float(series.isna().mean()), 4),
            "missing_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
            "unique_ratio": round(float(series.nunique(dropna=True) / max(len(df), 1)), 4),
            "numeric_like_ratio": round(_numeric_ratio(series), 4),
            "datetime_like_ratio": round(_datetime_ratio(series), 4),
            "distribution_summary": _numeric_summary(series) if inferred_type == "numeric" else {},
            "value_patterns": _top_value_patterns(series),
        }

    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "column_names": list(df.columns),
        "columns": columns,
        "pattern_detection": {
            "repeated_row_blocks": _detect_repeated_row_blocks(df),
            "sparsity_patterns": _detect_sparsity_patterns(df),
            "column_similarity": _detect_similar_columns(df),
        },
        "sample_rows": df.head(sample_rows).replace({pd.NA: None}).to_dict(orient="records"),
    }
