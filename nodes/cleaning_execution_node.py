# nodes/cleaning_execution_node.py
import re

import numpy as np
import pandas as pd

from state.state import AnalystState

NUMERIC_LIKE_THRESHOLD = 0.8


def _normalize_numeric_token(value) -> float | None:
    if pd.isna(value):
        return None

    s = str(value).strip()
    if not s:
        return None

    s = re.sub(r"[^\d,.\-]", "", s)
    if not s:
        return None

    if re.match(r"^\d{1,3}(\.\d{3})+,\d+$", s):
        s = s.replace(".", "").replace(",", ".")
    elif re.match(r"^\d{1,3}(,\d{3})+$", s):
        s = s.replace(",", "")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")

    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _detect_numeric_like_columns(df: pd.DataFrame, state: AnalystState) -> list[str]:
    column_registry = state.get("column_registry", {})
    detected: list[str] = []

    for col in df.columns:
        if column_registry.get(col, {}).get("semantic_role") == "identifier":
            continue

        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            detected.append(col)
            continue

        normalized = series.map(_normalize_numeric_token)
        if normalized.notna().mean() >= NUMERIC_LIKE_THRESHOLD:
            detected.append(col)

    return detected


def apply_numeric_cleaning(
    df: pd.DataFrame,
    state: AnalystState,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    numeric_cols = _detect_numeric_like_columns(df, state)
    cleaned_columns: list[str] = []
    warnings: list[str] = []

    for col in numeric_cols:
        if col not in df.columns:
            continue

        before_na = df[col].isna().sum()
        if pd.api.types.is_numeric_dtype(df[col]):
            cleaned_columns.append(col)
            continue

        df[col] = df[col].map(_normalize_numeric_token)
        after_na = df[col].isna().sum()
        cleaned_columns.append(col)

        if after_na > before_na:
            warnings.append(f"{col}: NaN increased ({before_na} -> {after_na})")

    return df, cleaned_columns, warnings


def _cap_outliers(series: pd.Series) -> pd.Series:
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return pd.Series(np.clip(series, lower, upper), index=series.index)


def cleaning_execution_node(state: AnalystState) -> AnalystState:
    df = state.get("dataframe")
    if df is None:
        raise ValueError("No dataframe found in state.")

    if "cleaning_plan" not in state:
        raise ValueError("No cleaning plan found in state.")

    df = df.copy()
    plan = state["cleaning_plan"]
    evidence = state.setdefault("analysis_evidence", {})

    for step in plan:
        col = step.get("column")
        action = step.get("action")

        if action == "numeric_cleaning":
            df, cleaned_cols, warnings = apply_numeric_cleaning(df, state)
            evidence["numeric_cleaning"] = {
                "cleaned_columns": cleaned_cols,
                "warnings": warnings,
            }

        elif action == "remove_duplicates":
            df = df.drop_duplicates()

        elif action == "impute" and col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val.iloc[0])

        elif action == "drop_column" and col in df.columns:
            df = df.drop(columns=[col])

        elif action == "convert_to_numeric" and col in df.columns:
            normalized = df[col].map(_normalize_numeric_token)
            if normalized.notna().mean() >= NUMERIC_LIKE_THRESHOLD:
                df[col] = normalized
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        elif action == "investigate_or_cap" and col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                normalized = df[col].map(_normalize_numeric_token)
                if normalized.notna().mean() >= NUMERIC_LIKE_THRESHOLD:
                    df[col] = normalized

            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = _cap_outliers(df[col])

    for col in df.select_dtypes(include=[np.number]).columns:
        col_mean = df[col].mean()
        col_max = df[col].max()

        if pd.notna(col_mean) and pd.notna(col_max) and col_mean != 0:
            ratio = col_max / col_mean

            if ratio > 1000:
                evidence.setdefault("scaling_warnings", []).append(
                    f"{col}: highly skewed distribution"
                )

            if col_mean < 1:
                evidence.setdefault("scaling_warnings", []).append(
                    f"{col}: mean very small -> possible scaling issue"
                )

    state["cleaned_data"] = df

    print("\n=== CLEANING EXECUTION COMPLETE ===")
    print(f"Cleaned dataset shape: {df.shape}")

    return state
