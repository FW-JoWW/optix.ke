from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def _timestamp_candidate_score(series: pd.Series) -> float:
    parsed = pd.to_datetime(series, errors="coerce")
    valid = parsed.dropna()
    if valid.empty:
        return -1.0

    valid_ratio = float(parsed.notna().mean())
    unique_count = int(valid.nunique())
    unique_ratio = float(unique_count / max(len(valid), 1))
    monotonic = bool(valid.is_monotonic_increasing or valid.is_monotonic_decreasing)

    minimum_unique = 3 if len(valid) <= 20 else 10
    if valid_ratio < 0.8 or unique_count < minimum_unique:
        return -1.0
    if unique_ratio < 0.01 and unique_count < 20:
        return -1.0

    return valid_ratio + min(unique_ratio, 1.0) + (0.25 if monotonic else 0.0)


def _best_timestamp_column(df: pd.DataFrame, state_context: Dict[str, Any]) -> Optional[str]:
    dataset_profile = state_context.get("dataset_profile", {}) or {}
    column_registry = state_context.get("column_registry", {}) or {}

    candidates = set(dataset_profile.get("datetime_columns", []) or [])
    for column, meta in column_registry.items():
        if (meta or {}).get("semantic_role") == "timestamp":
            candidates.add(column)

    best_column: Optional[str] = None
    best_score = -1.0
    for column in candidates:
        if column not in df.columns:
            continue
        score = _timestamp_candidate_score(df[column])
        if score > best_score:
            best_score = score
            best_column = column

    return best_column


def analyze_temporal_precedence(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    state_context: Dict[str, Any],
    max_lag: int = 6,
) -> Dict[str, Any]:
    timestamp_col = _best_timestamp_column(df, state_context)
    if not timestamp_col or any(col not in df.columns for col in [x_col, y_col]):
        return {"applicable": False}

    frame = df[[timestamp_col, x_col, y_col]].copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="coerce")
    frame[x_col] = pd.to_numeric(frame[x_col], errors="coerce")
    frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
    frame = frame.dropna().sort_values(timestamp_col)
    if len(frame) < max(12, max_lag * 3):
        return {
            "applicable": True,
            "timestamp_column": timestamp_col,
            "best_lag": None,
            "best_lag_correlation": None,
            "lag_direction": "unclear",
            "reverse_causality_warning": "Too few ordered observations for a reliable lag analysis.",
        }

    best_lag = 0
    best_corr = None
    for lag in range(-max_lag, max_lag + 1):
        shifted = frame[y_col].shift(-lag)
        corr = frame[x_col].corr(shifted)
        if pd.isna(corr):
            continue
        if best_corr is None or abs(float(corr)) > abs(float(best_corr)):
            best_corr = float(corr)
            best_lag = lag

    if best_corr is None:
        return {"applicable": True, "timestamp_column": timestamp_col, "best_lag": None, "best_lag_correlation": None, "lag_direction": "unclear"}

    if best_lag > 0:
        lag_direction = "x_precedes_y"
        warning = None
    elif best_lag < 0:
        lag_direction = "y_precedes_x"
        warning = "The strongest lag suggests the claimed driver may follow the outcome, so reverse causality is plausible."
    else:
        lag_direction = "same_time"
        warning = None

    return {
        "applicable": True,
        "timestamp_column": timestamp_col,
        "best_lag": int(best_lag),
        "best_lag_correlation": float(best_corr),
        "lag_direction": lag_direction,
        "reverse_causality_warning": warning,
    }
