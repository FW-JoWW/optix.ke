from __future__ import annotations

from typing import Any, Dict, List

import math
import pandas as pd

from inferential_engine import run_inferential_analysis
from tools.anova_tool import anova_tool
from tools.categorical_analysis_tool import categorical_analysis_tool
from tools.correlation_tool import correlation_tool
from tools.outlier_tool import outlier_tool
from tools.regression_tool import regression_tool
from tools.summary_statistics_tool import summary_statistics_tool
from tools.ttest_tool import ttest_tool
from utils.numeric_parsing import normalize_numeric_token

TOOL_MAPPING = {
    "direct_computation": None,
    "correlation": correlation_tool,
    "chi_square": None,
    "ttest": ttest_tool,
    "detect_outliers": outlier_tool,
    "summary_statistics": summary_statistics_tool,
    "regression": regression_tool,
    "anova": anova_tool,
    "categorical_analysis": categorical_analysis_tool,
}


def _coerce_numeric_like(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series

    coerced = pd.to_numeric(series.map(normalize_numeric_token), errors="coerce")
    non_null = int(series.notna().sum())
    ratio = float(coerced.notna().sum() / non_null) if non_null else 0.0
    return coerced if ratio >= 0.8 else series


def _coerce_datetime_like(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    parsed = pd.to_datetime(series, errors="coerce")
    non_null = int(series.notna().sum())
    ratio = float(parsed.notna().sum() / non_null) if non_null else 0.0
    return parsed if ratio >= 0.5 else series


def _prepare_frame(df: pd.DataFrame, columns: List[str], tool_name: str) -> pd.DataFrame:
    prepared = df.copy()
    numeric_tools = {"correlation", "summary_statistics", "detect_outliers", "regression", "ttest", "anova", "direct_computation"}
    if tool_name in numeric_tools:
        for col in columns:
            if col in prepared.columns:
                prepared[col] = _coerce_numeric_like(prepared[col])
    return prepared


def _contains_invalid_number(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_invalid_number(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_invalid_number(v) for v in value)
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    return False


def _result_is_invalid(result: Any) -> bool:
    if result is None:
        return True
    if isinstance(result, dict) and result.get("error"):
        return True
    return _contains_invalid_number(result)


def _fallback_result(
    df: pd.DataFrame,
    original_tool: str,
    columns: List[str],
) -> Dict[str, Any] | None:
    numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(_coerce_numeric_like(df[col]))]
    if not numeric_columns:
        return None

    fallback = summary_statistics_tool(df.assign(**{col: _coerce_numeric_like(df[col]) for col in numeric_columns}), numeric_columns)
    if isinstance(fallback, dict):
        fallback["tool"] = "summary_statistics"
        fallback["fallback_used"] = True
        fallback["original_tool"] = original_tool
    return fallback


def _run_direct_computation(df: pd.DataFrame, task: Dict[str, Any]) -> Dict[str, Any]:
    parameters = task.get("parameters", {}) or {}
    steps = parameters.get("computation_plan", [])
    current = df.copy()
    strategy = parameters.get("strategy", "aggregation")
    result_payload: Dict[str, Any] = {"strategy": strategy, "steps": steps}
    current_group = None
    current_series = None

    for step in steps:
        operation = step.get("operation")
        column = step.get("column")
        params = step.get("parameters", {}) or {}

        if operation == "group_by" and column in current.columns:
            current[column] = _coerce_datetime_like(current[column])
            current_group = column
        elif operation == "aggregate" and column in current.columns:
            method = params.get("method", "mean")
            if params.get("scope") == "group_results" and current_series is not None:
                reducer_source = pd.to_numeric(current_series, errors="coerce")
                if method == "mean":
                    result_payload["value"] = float(reducer_source.mean())
                elif method == "median":
                    result_payload["value"] = float(reducer_source.median())
                elif method == "min":
                    result_payload["value"] = float(reducer_source.min())
                elif method == "max":
                    result_payload["value"] = float(reducer_source.max())
                elif method == "sum":
                    result_payload["value"] = float(reducer_source.sum())
                result_payload["rows"] = current_series.reset_index().to_dict(orient="records")
                continue

            if current_group:
                grouped = current.groupby(current_group)[column]
                if method == "sum":
                    current_series = grouped.sum()
                elif method == "mean":
                    current_series = grouped.mean()
                elif method == "median":
                    current_series = grouped.median()
                elif method == "min":
                    current_series = grouped.min()
                elif method == "max":
                    current_series = grouped.max()
                else:
                    current_series = grouped.mean()
            else:
                series = pd.to_numeric(current[column], errors="coerce")
                if method == "sum":
                    result_payload["value"] = float(series.sum())
                elif method == "mean":
                    result_payload["value"] = float(series.mean())
                elif method == "median":
                    result_payload["value"] = float(series.median())
                elif method == "min":
                    result_payload["value"] = float(series.min())
                elif method == "max":
                    result_payload["value"] = float(series.max())
        elif operation == "group_compare":
            group_by = params.get("group_by")
            aggregate = params.get("aggregate", "mean")
            compare_col = column or (step.get("columns") or [None])[0]
            if group_by in current.columns and compare_col in current.columns:
                grouped = current.groupby(group_by)[compare_col]
                if aggregate == "sum":
                    current_series = grouped.sum()
                elif aggregate == "median":
                    current_series = grouped.median()
                else:
                    current_series = grouped.mean()
        elif operation == "frequency_distribution" and column in current.columns:
            current_series = current[column].value_counts(dropna=False)
        elif operation == "numeric_distribution" and column in current.columns:
            series = pd.to_numeric(current[column], errors="coerce")
            result_payload["summary"] = {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "min": float(series.min()),
                "max": float(series.max()),
            }

    if current_series is not None:
        if isinstance(current_series, pd.Series):
            result_payload["rows"] = current_series.reset_index().to_dict(orient="records")
        else:
            result_payload["value"] = current_series

    return {"tool": "direct_computation", "results": result_payload}


def execute_analysis_plan(
    df: pd.DataFrame,
    plan: List[Dict[str, Any]],
    config: Dict[str, Any] | None = None,
    state_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    config = config or {}
    state_context = state_context or {}

    for task in plan:
        tool_name = task["tool"]
        columns = task.get("columns", [])
        tool_func = TOOL_MAPPING.get(tool_name)
        if tool_name not in {"direct_computation", "chi_square"} and tool_func is None:
            continue
        if any(col not in df.columns for col in columns):
            continue
        prepared_df = _prepare_frame(df, columns, tool_name)

        try:
            if tool_name == "direct_computation":
                result = _run_direct_computation(prepared_df, task)
            elif tool_name in {"ttest", "anova", "correlation", "chi_square"}:
                result = run_inferential_analysis(prepared_df, task, state_context=state_context)
            elif tool_name in {"detect_outliers", "summary_statistics"}:
                result = tool_func(prepared_df, columns)
            elif tool_name == "regression":
                result = tool_func(prepared_df, x_col=columns[0], y_col=columns[1])
            elif tool_name == "categorical_analysis":
                result = tool_func(prepared_df, columns=columns, config=config)
            else:
                continue
        except Exception as exc:
            result = {"tool": tool_name, "error": str(exc)}

        if _result_is_invalid(result):
            fallback = _fallback_result(prepared_df, tool_name, columns)
            if fallback is not None and not _result_is_invalid(fallback):
                result = fallback

        if result is not None and isinstance(result, dict) and "tool" not in result:
            result["tool"] = tool_name
        key = f"{tool_name}_{'_'.join(columns)}" if columns else tool_name
        results[key] = result

    return results
