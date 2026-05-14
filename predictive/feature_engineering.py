from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return pd.to_datetime(series, errors="coerce")


def infer_problem_type(question: str, target_column: str, df: pd.DataFrame, date_column: str | None = None) -> str:
    query = (question or "").lower()
    if any(word in query for word in ["forecast", "next", "future", "trend over time", "time series"]) and date_column:
        return "forecasting"
    series = df[target_column]
    if pd.api.types.is_numeric_dtype(series):
        unique_count = int(series.dropna().nunique())
        if unique_count <= 10 and any(word in query for word in ["class", "risk", "likely", "default", "churn", "fraud", "segment", "attrition"]):
            return "classification"
        return "regression"
    return "classification"


def detect_date_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            return column
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        sample = _safe_to_datetime(df[column])
        if sample.notna().mean() >= 0.8 and sample.nunique(dropna=True) >= max(10, int(len(sample) * 0.05)):
            return column
    return None


def _add_datetime_parts(frame: pd.DataFrame, column: str, metadata: Dict[str, List[str]]) -> pd.DataFrame:
    parsed = _safe_to_datetime(frame[column])
    frame[column] = parsed
    generated = [
        f"{column}_year",
        f"{column}_month",
        f"{column}_day",
        f"{column}_dayofweek",
        f"{column}_quarter",
    ]
    frame[generated[0]] = parsed.dt.year
    frame[generated[1]] = parsed.dt.month
    frame[generated[2]] = parsed.dt.day
    frame[generated[3]] = parsed.dt.dayofweek
    frame[generated[4]] = parsed.dt.quarter
    metadata["date_features"].extend(generated)
    return frame.drop(columns=[column])


def build_feature_frame(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    date_column: str | None,
    leakage_columns: List[str],
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
    frame = df.copy()
    feature_columns = [col for col in frame.columns if col not in {target_column, *leakage_columns}]
    X = frame[feature_columns].copy()
    y = frame[target_column].copy()

    metadata: Dict[str, List[str]] = {"date_features": [], "lag_features": [], "categorical_features": [], "numeric_features": []}

    # Treat datetime-like columns structurally instead of one-hot encoding every timestamp.
    datetime_candidates = [col for col in X.columns if pd.api.types.is_datetime64_any_dtype(X[col])]
    if date_column and date_column in X.columns and date_column not in datetime_candidates:
        datetime_candidates.append(date_column)
    for column in list(X.columns):
        if column in datetime_candidates:
            continue
        if X[column].dtype == "object" or pd.api.types.is_string_dtype(X[column]):
            parsed = _safe_to_datetime(X[column])
            if parsed.notna().mean() >= 0.8 and parsed.nunique(dropna=True) >= max(10, int(len(parsed) * 0.05)):
                datetime_candidates.append(column)

    for column in datetime_candidates:
        if column in X.columns:
            X = _add_datetime_parts(X, column, metadata)

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]

    for column in numeric_columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")
        X[column] = X[column].fillna(X[column].median())
    metadata["numeric_features"] = numeric_columns[:]

    if problem_type == "forecasting" and date_column and date_column in frame.columns:
        ordering_series = _safe_to_datetime(frame[date_column])
        ordered = pd.concat(
            [ordering_series.rename("__ordering_date__"), X, y.rename(target_column)],
            axis=1,
        ).dropna(subset=["__ordering_date__", target_column]).sort_values("__ordering_date__").reset_index(drop=True)
        for lag in [1, 2, 3, 7]:
            ordered[f"{target_column}_lag_{lag}"] = ordered[target_column].shift(lag)
            metadata["lag_features"].append(f"{target_column}_lag_{lag}")
        ordered[f"{target_column}_rolling_mean_3"] = ordered[target_column].shift(1).rolling(3).mean()
        ordered[f"{target_column}_rolling_std_3"] = ordered[target_column].shift(1).rolling(3).std()
        metadata["lag_features"].extend([f"{target_column}_rolling_mean_3", f"{target_column}_rolling_std_3"])
        ordered = ordered.dropna().reset_index(drop=True)
        y = ordered[target_column]
        X = ordered.drop(columns=[target_column, "__ordering_date__"])
        numeric_columns = X.select_dtypes(include="number").columns.tolist()
        categorical_columns = [col for col in X.columns if col not in numeric_columns]

    retained_categorical: List[str] = []
    for column in categorical_columns:
        non_null = X[column].dropna().astype(str)
        unique_count = int(non_null.nunique())
        unique_ratio = float(unique_count / len(non_null)) if len(non_null) else 0.0
        if unique_count > 100 and unique_ratio > 0.5:
            X = X.drop(columns=[column])
            continue
        if unique_count > 30:
            frequencies = X[column].astype(str).fillna("__missing__").value_counts(normalize=True)
            encoded_name = f"{column}__freq"
            X[encoded_name] = X[column].astype(str).fillna("__missing__").map(frequencies).astype(float)
            metadata["numeric_features"].append(encoded_name)
            X = X.drop(columns=[column])
            continue
        value_counts = X[column].astype(str).fillna("__missing__").value_counts(normalize=True)
        rare_values = value_counts[value_counts < 0.01].index
        X[column] = X[column].astype(str).fillna("__missing__").replace({value: "__rare__" for value in rare_values})
        retained_categorical.append(column)
    categorical_columns = retained_categorical
    metadata["categorical_features"] = categorical_columns[:]

    valid_target_mask = y.notna()
    if X.shape[1] == 0:
        return X.loc[valid_target_mask].copy(), y.loc[valid_target_mask].copy(), metadata

    X = pd.get_dummies(X, columns=categorical_columns, drop_first=False)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[valid_target_mask].copy()
    y = y.loc[valid_target_mask].copy()
    X = X.fillna(0)
    zero_variance = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
    if zero_variance:
        X = X.drop(columns=zero_variance)

    return X, y, metadata


def split_features(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    date_column: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if problem_type == "forecasting":
        split_idx = max(int(len(X) * 0.8), len(X) - min(12, max(4, len(X) // 5)))
        return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

    stratify = y if problem_type == "classification" and y.nunique(dropna=True) > 1 else None
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
