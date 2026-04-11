from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

try:
    from scipy.stats import chi2_contingency
except Exception:  # pragma: no cover - graceful fallback when scipy is unavailable
    chi2_contingency = None


@dataclass(frozen=True)
class CategoricalAnalysisConfig:
    low_unique_numeric_threshold: int = 20
    low_unique_numeric_ratio: float = 0.05
    high_cardinality_threshold: int = 50
    rare_category_threshold: float = 0.01
    max_categories_for_frequency: int = 500
    max_categories_for_cross_analysis: int = 50
    max_contingency_cells: int = 2_500
    max_numeric_interaction_groups: int = 100


def detect_categorical_columns(
    df: pd.DataFrame,
    config: Optional[CategoricalAnalysisConfig] = None,
) -> List[str]:
    config = config or CategoricalAnalysisConfig()
    categorical_columns: List[str] = []
    total_rows = max(len(df), 1)

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        unique_count = int(non_null.nunique())
        unique_ratio = unique_count / total_rows

        if (
            is_object_dtype(series)
            or is_string_dtype(series)
            or is_categorical_dtype(series)
            or is_bool_dtype(series)
        ):
            categorical_columns.append(col)
            continue

        if is_numeric_dtype(series):
            if (
                unique_count <= config.low_unique_numeric_threshold
                and unique_ratio <= config.low_unique_numeric_ratio
            ):
                categorical_columns.append(col)

    return categorical_columns


def _safe_mode(series: pd.Series) -> Any:
    mode = series.dropna().mode()
    if mode.empty:
        return None
    return _to_python_value(mode.iloc[0])


def _to_python_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _label_key(value: Any) -> str:
    python_value = _to_python_value(value)
    if python_value is None:
        return "__MISSING__"
    return str(python_value)


def _series_as_clean_strings(series: pd.Series) -> pd.Series:
    return series.dropna().astype(str)


def _frequency_distribution(
    series: pd.Series,
    config: CategoricalAnalysisConfig,
) -> Tuple[Dict[str, int], Dict[str, float], bool]:
    counts = series.value_counts(dropna=True)
    truncated = len(counts) > config.max_categories_for_frequency
    if truncated:
        counts = counts.head(config.max_categories_for_frequency)

    total = max(int(series.dropna().shape[0]), 1)
    percentages = {
        _label_key(idx): round((int(val) / total) * 100, 4)
        for idx, val in counts.items()
    }
    frequency = {_label_key(idx): int(val) for idx, val in counts.items()}
    return frequency, percentages, truncated


def _quality_issues(series: pd.Series) -> Dict[str, Any]:
    issues: List[str] = []
    suggestions: Dict[str, str] = {}

    string_values = _series_as_clean_strings(series)
    if string_values.empty:
        return {"issues": issues, "suggestions": suggestions}

    trimmed = string_values.str.strip()
    if not string_values.equals(trimmed):
        issues.append("whitespace_inconsistency")

    normalized_map: Dict[str, List[str]] = {}
    for raw_value in string_values.unique().tolist():
        normalized = raw_value.strip().casefold()
        normalized_map.setdefault(normalized, []).append(raw_value)

    inconsistent_groups = {
        key: sorted(set(values))
        for key, values in normalized_map.items()
        if len(set(values)) > 1
    }

    if inconsistent_groups:
        issues.append("case_or_label_inconsistency")
        for values in inconsistent_groups.values():
            canonical = min(values, key=lambda v: (len(v.strip()), v.strip().casefold()))
            cleaned = canonical.strip()
            for value in values:
                if value != cleaned:
                    suggestions[value] = cleaned

    return {
        "issues": issues,
        "suggestions": suggestions,
    }


def _rare_categories(
    series: pd.Series,
    config: CategoricalAnalysisConfig,
) -> Dict[str, Dict[str, float]]:
    counts = series.value_counts(dropna=True)
    total = max(int(series.dropna().shape[0]), 1)
    rare: Dict[str, Dict[str, float]] = {}

    for label, count in counts.items():
        proportion = count / total
        if proportion < config.rare_category_threshold:
            rare[_label_key(label)] = {
                "count": int(count),
                "percentage": round(proportion * 100, 4),
            }

    return rare


def _cardinality(unique_count: int, config: CategoricalAnalysisConfig) -> Dict[str, Any]:
    return {
        "unique_values": unique_count,
        "is_high_cardinality": unique_count > config.high_cardinality_threshold,
        "threshold": config.high_cardinality_threshold,
    }


def _contingency_analysis(
    df: pd.DataFrame,
    source_col: str,
    other_col: str,
    config: CategoricalAnalysisConfig,
) -> Dict[str, Any]:
    source_unique = int(df[source_col].dropna().nunique())
    other_unique = int(df[other_col].dropna().nunique())

    if source_unique == 0 or other_unique == 0:
        return {"status": "skipped", "reason": "empty_categories"}

    if (
        source_unique > config.max_categories_for_cross_analysis
        or other_unique > config.max_categories_for_cross_analysis
    ):
        return {"status": "skipped", "reason": "high_cardinality"}

    if source_unique * other_unique > config.max_contingency_cells:
        return {"status": "skipped", "reason": "contingency_too_large"}

    contingency = pd.crosstab(df[source_col], df[other_col], dropna=False)
    formatted_table = {
        _label_key(col_key): {
            _label_key(row_key): int(cell_value)
            for row_key, cell_value in column_values.items()
        }
        for col_key, column_values in contingency.to_dict().items()
    }
    result: Dict[str, Any] = {
        "status": "ok",
        "contingency_table": formatted_table,
        "shape": list(contingency.shape),
    }

    if chi2_contingency is None:
        result["chi_square"] = {
            "status": "unavailable",
            "reason": "scipy_not_installed",
        }
        return result

    try:
        chi2, p_value, dof, _ = chi2_contingency(contingency)
        result["chi_square"] = {
            "status": "ok",
            "chi2": float(chi2),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
        }
    except Exception as exc:
        result["chi_square"] = {
            "status": "failed",
            "reason": str(exc),
        }

    return result


def _numeric_interactions(
    df: pd.DataFrame,
    categorical_col: str,
    numeric_columns: List[str],
    config: CategoricalAnalysisConfig,
) -> Dict[str, Any]:
    group_count = int(df[categorical_col].dropna().nunique())
    if group_count == 0:
        return {}

    if group_count > config.max_numeric_interaction_groups:
        return {
            "_status": "skipped",
            "_reason": "too_many_groups",
        }

    interactions: Dict[str, Any] = {}

    for num_col in numeric_columns:
        safe_df = df[[categorical_col, num_col]].copy()
        safe_df[num_col] = pd.to_numeric(
            safe_df[num_col].astype(str).str.replace(r"[^\d\.\-]", "", regex=True),
            errors="coerce",
        )
        safe_df = safe_df.dropna(subset=[num_col])
        if safe_df.empty:
            continue

        summary = safe_df.groupby(categorical_col, dropna=False)[num_col].agg(["mean", "median", "count"])
        interactions[num_col] = {
            _label_key(idx): {
                "mean": None if pd.isna(row["mean"]) else float(row["mean"]),
                "median": None if pd.isna(row["median"]) else float(row["median"]),
                "count": int(row["count"]),
            }
            for idx, row in summary.iterrows()
        }

    return interactions


def analyze_categorical_columns(
    df: pd.DataFrame,
    *,
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    config: Optional[CategoricalAnalysisConfig] = None,
) -> Dict[str, Dict[str, Any]]:
    config = config or CategoricalAnalysisConfig()
    categorical_columns = categorical_columns or detect_categorical_columns(df, config)
    numeric_columns = numeric_columns or [
        col for col in df.select_dtypes(include="number").columns.tolist()
        if col not in categorical_columns
    ]

    results: Dict[str, Dict[str, Any]] = {}

    for col in categorical_columns:
        series = df[col]
        non_null = series.dropna()
        unique_count = int(non_null.nunique())
        frequency, percentages, truncated = _frequency_distribution(series, config)
        quality = _quality_issues(series)

        cross_analysis = {}
        for other_col in categorical_columns:
            if other_col == col:
                continue
            cross_analysis[other_col] = _contingency_analysis(df, col, other_col, config)

        results[col] = {
            "type": "categorical",
            "frequency": frequency,
            "percentages": percentages,
            "frequency_truncated": truncated,
            "cardinality": _cardinality(unique_count, config),
            "mode": _to_python_value(_safe_mode(series)),
            "missing": {
                "count": int(series.isna().sum()),
                "percentage": round((float(series.isna().mean()) * 100), 4),
            },
            "rare_categories": _rare_categories(series, config),
            "quality_issues": quality,
            "cross_analysis": cross_analysis,
            "numeric_interactions": _numeric_interactions(
                df,
                col,
                numeric_columns,
                config,
            ),
        }

    return results


def categorical_analysis_config_to_dict(
    config: Optional[CategoricalAnalysisConfig] = None,
) -> Dict[str, Any]:
    return asdict(config or CategoricalAnalysisConfig())
