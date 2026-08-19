# utils/issue_detector.py
import warnings

import numpy as np
import pandas as pd

from backend.utils.numeric_parsing import normalize_numeric_token
from backend.utils.dataset_artifact_cache import get_cached_artifact, set_cached_artifact


PROFILE_SAMPLE_SIZE = 1000
LARGE_DATASET_ROW_THRESHOLD = 25000

_DATETIME_NAME_HINTS = ("date", "time", "timestamp", "month", "year", "day", "dob")
_DATETIME_TOKEN_PATTERN = r"(\d{4}[-/]\d{1,2}([-/]\d{1,2})?|(?:\d{1,2}[-/]){2}\d{2,4}|\d{1,2}:\d{2}(:\d{2})?|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})"


def _sample_non_null(series: pd.Series, max_non_null: int = PROFILE_SAMPLE_SIZE) -> pd.Series:
    non_null = series.dropna()
    if len(non_null) <= max_non_null:
        return non_null
    return non_null.sample(max_non_null, random_state=42)


def _parse_ratio_on_non_null(parsed: pd.Series, original: pd.Series) -> float:
    non_null = int(original.notna().sum())
    if non_null == 0:
        return 0.0
    return float(parsed.notna().sum() / non_null)


def _datetime_signal_ratio(series: pd.Series) -> float:
    sample = _sample_non_null(series).astype(str)
    if sample.empty:
        return 0.0
    token_signal = sample.str.contains(_DATETIME_TOKEN_PATTERN, case=False, regex=True, na=False)
    return float(token_signal.mean())


def detect_issues(df: pd.DataFrame) -> dict:
    """
    Rule-based deterministic detection of data quality issues.
    Returns a list of detected issues (without LLM interpretation yet).
    """
    cached = get_cached_artifact("detect_issues", df)
    if cached is not None:
        return cached

    issues = []
    total_rows = len(df)
    large_dataset = total_rows > LARGE_DATASET_ROW_THRESHOLD or df.shape[1] > 25
    evidence_scope = "sampled" if large_dataset else "exact"

    numeric_cache = {}
    for col in df.columns:
        sample = _sample_non_null(df[col])
        if sample.empty:
            continue
        coerced_sample = pd.to_numeric(sample.map(normalize_numeric_token), errors="coerce")
        if float(coerced_sample.notna().mean()) >= 0.8:
            if large_dataset:
                numeric_cache[col] = coerced_sample.dropna()
            else:
                numeric_cache[col] = pd.to_numeric(df[col].map(normalize_numeric_token), errors="coerce")

    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_percent = (missing_count / total_rows) * 100 if total_rows else 0.0
            severity = "low"
            if missing_percent > 30:
                severity = "high"
            elif missing_percent > 10:
                severity = "medium"
            issues.append(
                {
                    "column": col,
                    "issue_type": "missing_values",
                    "severity": severity,
                    "missing_count": missing_count,
                    "evidence_scope": "exact",
                }
            )

    dup_count = df.head(5000).duplicated().sum() if large_dataset else df.duplicated().sum()
    if dup_count > 0:
        issues.append(
            {
                "column": None,
                "issue_type": "duplicate_rows",
                "severity": "medium",
                "duplicate_count": dup_count,
                "evidence_scope": "sampled" if large_dataset else "exact",
            }
        )

    for col, series in numeric_cache.items():
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_count = series[(series < lower) | (series > upper)].count()
        if outlier_count > 0:
            issues.append(
                    {
                        "column": col,
                        "issue_type": "outliers",
                        "severity": "high" if total_rows and outlier_count / total_rows > 0.05 else "medium",
                        "outlier_count": outlier_count,
                        "evidence_scope": "exact",
                    }
                )

    for col in df.columns:
        if large_dataset:
            sample = _sample_non_null(df[col])
            if sample.empty:
                continue
            if sample.nunique(dropna=False) == 1 and df[col].isna().sum() < total_rows:
                issues.append(
                    {
                        "column": col,
                        "issue_type": "constant_column",
                        "severity": "low",
                        "evidence_scope": "sampled" if large_dataset else "exact",
                    }
                )
            continue

        if df[col].nunique(dropna=False) == 1:
            issues.append(
                {
                    "column": col,
                    "issue_type": "constant_column",
                    "severity": "low",
                    "evidence_scope": "exact",
                }
            )

    categorical_cols = df.select_dtypes(include=["str", "category"]).columns.tolist()
    for col in categorical_cols:
        if total_rows == 0:
            continue
        sample = _sample_non_null(df[col])
        if sample.empty:
            continue
        unique_ratio = sample.nunique() / max(len(sample), 1) if large_dataset else df[col].nunique() / total_rows
        if unique_ratio > 0.8:
            issues.append(
                {
                    "column": col,
                    "issue_type": "high_cardinality",
                    "severity": "medium",
                    "evidence_scope": "sampled" if large_dataset else "exact",
                }
            )

    for col in df.columns:
        if df[col].dtype == "object" and col in numeric_cache:
            issues.append(
                {
                    "column": col,
                    "issue_type": "numeric_as_object",
                    "severity": "medium",
                    "evidence_scope": "sampled" if large_dataset else "exact",
                }
            )

    for col in df.columns:
        if df[col].dtype != "object":
            continue
        sample = _sample_non_null(df[col])
        if sample.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce")
        parse_ratio = float(parsed.notna().mean())
        token_ratio = _datetime_signal_ratio(df[col])
        name_hint = any(hint in col.lower() for hint in _DATETIME_NAME_HINTS)
        if parse_ratio >= 0.9 and (token_ratio >= 0.3 or name_hint):
            issues.append(
                {
                    "column": col,
                    "issue_type": "datetime_as_object",
                    "severity": "medium",
                    "evidence_scope": "sampled" if large_dataset else "exact",
                }
            )

    for col in categorical_cols:
        non_null = _sample_non_null(df[col]).astype(str) if large_dataset else df[col].dropna().astype(str)
        if non_null.empty:
            continue
        stripped = non_null.str.strip()
        normalized = stripped.str.lower()
        if bool((non_null != stripped).any()) or normalized.nunique() < stripped.nunique():
            issues.append(
                {
                    "column": col,
                    "issue_type": "inconsistent_labels",
                    "severity": "low",
                    "evidence_scope": "sampled" if large_dataset else "exact",
                }
            )

    result = {
        "detected_issues": issues,
        "evidence_scope": evidence_scope,
        "provenance": {
            "source": "issue_detector",
            "scope": evidence_scope,
            "verified": not large_dataset,
            "method": "heuristic reconnaissance" if large_dataset else "exact detection",
        },
    }
    return set_cached_artifact("detect_issues", df, result)


