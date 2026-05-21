from __future__ import annotations

from typing import Any, Dict, List

import math
import pandas as pd

from inferential_engine import run_inferential_analysis
from predictive.predictive_engine import run_predictive_analysis
from prescriptive.prescriptive_engine import run_prescriptive_analysis
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
    "predictive_analysis": None,
    "prescriptive_analysis": None,
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


def _sanitize_numbers(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_numbers(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _inferential_result_is_invalid(result: Dict[str, Any]) -> bool:
    if result.get("error"):
        return True
    method = result.get("method_selected")
    if not method:
        return True
    hypothesis = result.get("hypothesis_test", {}) or {}
    effect_size = result.get("effect_size", {}) or {}
    if hypothesis.get("p_value") is None and hypothesis.get("test_statistic") is None:
        return True
    if effect_size and effect_size.get("metric") and effect_size.get("value") is None and hypothesis.get("p_value") is None:
        return True
    return False


def _result_is_invalid(result: Any) -> bool:
    if result is None:
        return True
    if isinstance(result, dict) and result.get("error"):
        return True
    if isinstance(result, dict) and result.get("tool") == "inferential_analysis":
        return _inferential_result_is_invalid(result)
    return _contains_invalid_number(result)


def _fallback_result(
    df: pd.DataFrame,
    original_tool: str,
    columns: List[str],
) -> Dict[str, Any] | None:
    if original_tool in {"predictive_analysis", "prescriptive_analysis"}:
        return None
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
            bucket = params.get("bucket")
            if bucket and pd.api.types.is_datetime64_any_dtype(current[column]):
                if bucket == "month":
                    bucket_col = f"{column}__month"
                    current[bucket_col] = current[column].dt.to_period("M").astype(str)
                elif bucket == "quarter":
                    bucket_col = f"{column}__quarter"
                    current[bucket_col] = current[column].dt.to_period("Q").astype(str)
                elif bucket == "year":
                    bucket_col = f"{column}__year"
                    current[bucket_col] = current[column].dt.to_period("Y").astype(str)
                else:
                    bucket_col = column
                current_group = bucket_col
            else:
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
        elif operation == "distinct_count" and column in current.columns:
            if current_group:
                current_series = current.groupby(current_group)[column].nunique(dropna=True)
            else:
                result_payload["value"] = int(current[column].nunique(dropna=True))
        elif operation == "row_expression" and column in current.columns:
            subtract_column = params.get("subtract_column")
            expression_name = params.get("expression_name") or column
            if subtract_column in current.columns:
                left = pd.to_numeric(current[column], errors="coerce")
                right = pd.to_numeric(current[subtract_column], errors="coerce")
                derived = left - right
                current[expression_name] = derived
                current_series = derived
                result_payload["derived_column"] = expression_name
        elif operation == "share_of_total" and column in current.columns:
            entity_column = params.get("entity_column")
            top_n = int(params.get("top_n", 10) or 10)
            if entity_column in current.columns:
                grouped = current.groupby(entity_column)[column].sum().sort_values(ascending=False)
                top_share = float(grouped.head(top_n).sum() / grouped.sum()) if float(grouped.sum()) else 0.0
                result_payload["value"] = top_share
                result_payload["rows"] = grouped.head(top_n).reset_index().to_dict(orient="records")
                result_payload["top_n"] = top_n
        elif operation == "repeat_rate" and column in current.columns:
            entity_column = params.get("entity_column")
            if entity_column in current.columns:
                grouped = current.groupby(entity_column)[column].nunique(dropna=True)
                repeat_share = float((grouped > 1).mean()) if len(grouped) else 0.0
                result_payload["value"] = repeat_share
                result_payload["summary"] = {
                    "repeat_entities": int((grouped > 1).sum()),
                    "total_entities": int(len(grouped)),
                }
        elif operation == "growth_rate" and column in current.columns:
            method = params.get("method", "sum")
            if current_group:
                grouped = current.groupby(current_group)[column]
                if method == "distinct_count":
                    entity_column = params.get("entity_column") or column
                    if entity_column in current.columns:
                        series = current.groupby(current_group)[entity_column].nunique(dropna=True)
                    else:
                        series = grouped.nunique(dropna=True)
                elif method == "sum":
                    series = grouped.sum()
                elif method == "mean":
                    series = grouped.mean()
                else:
                    series = grouped.sum()
                growth = series.pct_change().replace([float("inf"), float("-inf")], pd.NA)
                growth_rows = pd.DataFrame(
                    {
                        "period": series.index.astype(str),
                        "value": series.values,
                        "growth_rate": growth.values,
                    }
                )
                result_payload["rows"] = _sanitize_numbers(growth_rows.to_dict(orient="records"))
                valid_growth = growth.dropna()
                result_payload["value"] = float(valid_growth.iloc[-1]) if not valid_growth.empty else None
        elif operation == "rank_entities" and column in current.columns:
            entity_column = params.get("entity_column")
            method = params.get("method", "sum")
            top_n = int(params.get("top_n", 10) or 10)
            ascending = str(params.get("sort", "desc")).lower() == "asc"
            if entity_column in current.columns:
                if method == "distinct_count":
                    grouped = current.groupby(entity_column)[column].nunique(dropna=True)
                elif method == "mean":
                    grouped = current.groupby(entity_column)[column].mean()
                elif method == "count":
                    grouped = current.groupby(entity_column)[column].count()
                else:
                    grouped = current.groupby(entity_column)[column].sum()
                ranked = grouped.sort_values(ascending=ascending).head(top_n)
                result_payload["rows"] = ranked.reset_index().to_dict(orient="records")
                result_payload["value"] = float(ranked.iloc[0]) if len(ranked) else None
                result_payload["ranking_sort"] = "asc" if ascending else "desc"
        elif operation == "segment_contrast":
            entity_column = params.get("entity_column")
            primary_metric = params.get("primary_metric")
            primary_method = params.get("primary_method", "distinct_count")
            secondary_metric = params.get("secondary_metric")
            secondary_method = params.get("secondary_method", "sum")
            pattern = params.get("pattern", "high_low")
            top_n = int(params.get("top_n", 10) or 10)
            if entity_column in current.columns and primary_metric in current.columns and secondary_metric in current.columns:
                if primary_method == "distinct_count":
                    primary_series = current.groupby(entity_column)[primary_metric].nunique(dropna=True)
                elif primary_method == "mean":
                    primary_series = current.groupby(entity_column)[primary_metric].mean()
                else:
                    primary_series = current.groupby(entity_column)[primary_metric].sum()

                if secondary_method == "mean":
                    secondary_series = current.groupby(entity_column)[secondary_metric].mean()
                elif secondary_method == "distinct_count":
                    secondary_series = current.groupby(entity_column)[secondary_metric].nunique(dropna=True)
                else:
                    secondary_series = current.groupby(entity_column)[secondary_metric].sum()

                contrast_df = pd.DataFrame({
                    entity_column: primary_series.index,
                    "primary_value": primary_series.values,
                    "secondary_value": secondary_series.reindex(primary_series.index).values,
                }).dropna()
                if not contrast_df.empty:
                    if pattern == "high_low":
                        p_cut = contrast_df["primary_value"].quantile(0.75)
                        s_cut = contrast_df["secondary_value"].quantile(0.25)
                        subset = contrast_df[(contrast_df["primary_value"] >= p_cut) & (contrast_df["secondary_value"] <= s_cut)].copy()
                        subset["contrast_score"] = subset["primary_value"].rank(pct=True) - subset["secondary_value"].rank(pct=True)
                        subset = subset.sort_values("contrast_score", ascending=False)
                    else:
                        p_cut = contrast_df["primary_value"].quantile(0.25)
                        s_cut = contrast_df["secondary_value"].quantile(0.75)
                        subset = contrast_df[(contrast_df["primary_value"] <= p_cut) & (contrast_df["secondary_value"] >= s_cut)].copy()
                        subset["contrast_score"] = subset["secondary_value"].rank(pct=True) - subset["primary_value"].rank(pct=True)
                        subset = subset.sort_values("contrast_score", ascending=False)
                    if subset.empty:
                        subset = contrast_df.copy()
                        subset["contrast_score"] = (
                            subset["primary_value"].rank(pct=True, ascending=(pattern != "high_low"))
                            + subset["secondary_value"].rank(pct=True, ascending=(pattern == "high_low"))
                        )
                        subset = subset.sort_values("contrast_score", ascending=False)
                    subset = subset.head(top_n)
                    result_payload["rows"] = subset.to_dict(orient="records")
                    result_payload["value"] = float(subset.iloc[0]["contrast_score"]) if len(subset) else None
                    result_payload["contrast_pattern"] = pattern
        elif operation == "segment_growth_rank":
            entity_column = params.get("entity_column")
            time_column = params.get("time_column")
            bucket = params.get("bucket", "month")
            method = params.get("method", "sum")
            top_n = int(params.get("top_n", 10) or 10)
            ascending = str(params.get("sort", "desc")).lower() == "asc"
            if entity_column in current.columns and time_column in current.columns and column in current.columns:
                time_series = _coerce_datetime_like(current[time_column])
                if pd.api.types.is_datetime64_any_dtype(time_series):
                    temp = current.copy()
                    temp[time_column] = time_series
                    if bucket == "quarter":
                        temp["_period_bucket"] = temp[time_column].dt.to_period("Q").astype(str)
                    elif bucket == "year":
                        temp["_period_bucket"] = temp[time_column].dt.to_period("Y").astype(str)
                    else:
                        temp["_period_bucket"] = temp[time_column].dt.to_period("M").astype(str)
                    if method == "distinct_count":
                        grouped = temp.groupby([entity_column, "_period_bucket"])[column].nunique(dropna=True)
                    elif method == "mean":
                        grouped = temp.groupby([entity_column, "_period_bucket"])[column].mean()
                    else:
                        grouped = temp.groupby([entity_column, "_period_bucket"])[column].sum()
                    growth_rows = []
                    for entity, series in grouped.groupby(level=0):
                        entity_series = series.droplevel(0).sort_index()
                        growth = entity_series.pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna()
                        if growth.empty:
                            continue
                        growth_rows.append({
                            entity_column: entity,
                            "latest_growth_rate": growth.iloc[-1],
                            "average_growth_rate": growth.mean(),
                            "periods": int(len(entity_series)),
                        })
                    growth_df = pd.DataFrame(growth_rows)
                    if not growth_df.empty:
                        sort_col = "average_growth_rate"
                        growth_df = growth_df.sort_values(sort_col, ascending=ascending).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(growth_df.to_dict(orient="records"))
                        result_payload["value"] = float(growth_df.iloc[0][sort_col]) if len(growth_df) else None
                        result_payload["growth_sort"] = "asc" if ascending else "desc"
        elif operation == "segment_seasonality":
            entity_column = params.get("entity_column")
            time_column = params.get("time_column")
            method = params.get("method", "sum")
            top_n = int(params.get("top_n", 10) or 10)
            if entity_column in current.columns and time_column in current.columns and column in current.columns:
                time_series = _coerce_datetime_like(current[time_column])
                if pd.api.types.is_datetime64_any_dtype(time_series):
                    temp = current.copy()
                    temp[time_column] = time_series
                    temp["_month_bucket"] = temp[time_column].dt.month
                    if method == "distinct_count":
                        grouped = temp.groupby([entity_column, "_month_bucket"])[column].nunique(dropna=True)
                    elif method == "mean":
                        grouped = temp.groupby([entity_column, "_month_bucket"])[column].mean()
                    else:
                        grouped = temp.groupby([entity_column, "_month_bucket"])[column].sum()
                    rows = []
                    for entity, series in grouped.groupby(level=0):
                        entity_series = series.droplevel(0)
                        mean_value = float(entity_series.mean()) if len(entity_series) else 0.0
                        std_value = float(entity_series.std(ddof=0)) if len(entity_series) > 1 else 0.0
                        seasonality_score = float(std_value / mean_value) if mean_value else 0.0
                        rows.append({
                            entity_column: entity,
                            "seasonality_score": seasonality_score,
                            "peak_month": int(entity_series.idxmax()) if len(entity_series) else None,
                        })
                    seasonality_df = pd.DataFrame(rows)
                    if not seasonality_df.empty:
                        seasonality_df = seasonality_df.sort_values("seasonality_score", ascending=False).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(seasonality_df.to_dict(orient="records"))
                        result_payload["value"] = float(seasonality_df.iloc[0]["seasonality_score"]) if len(seasonality_df) else None
        elif operation == "concentration_score":
            parent_column = params.get("parent_column")
            child_column = params.get("child_column")
            method = params.get("method", "sum")
            top_n = int(params.get("top_n", 10) or 10)
            if parent_column in current.columns and child_column in current.columns and column in current.columns:
                if method == "distinct_count":
                    grouped = current.groupby([parent_column, child_column])[column].nunique(dropna=True)
                elif method == "mean":
                    grouped = current.groupby([parent_column, child_column])[column].mean()
                else:
                    grouped = current.groupby([parent_column, child_column])[column].sum()
                rows = []
                for parent, series in grouped.groupby(level=0):
                    child_series = series.droplevel(0).sort_values(ascending=False)
                    total = float(child_series.sum())
                    if not total:
                        continue
                    top_share = float(child_series.iloc[0] / total)
                    hhi = float(((child_series / total) ** 2).sum())
                    rows.append({
                        parent_column: parent,
                        "top_child_share": top_share,
                        "concentration_score": hhi,
                        "top_child": str(child_series.index[0]),
                    })
                concentration_df = pd.DataFrame(rows)
                if not concentration_df.empty:
                    concentration_df = concentration_df.sort_values("concentration_score", ascending=False).head(top_n)
                    result_payload["rows"] = _sanitize_numbers(concentration_df.to_dict(orient="records"))
                    result_payload["value"] = float(concentration_df.iloc[0]["concentration_score"]) if len(concentration_df) else None
        elif operation == "trend_classification" and column in current.columns:
            source = params.get("source")
            if source == "growth_rows" and result_payload.get("rows"):
                growth_values = [row.get("growth_rate") for row in result_payload["rows"] if row.get("growth_rate") is not None]
                if growth_values:
                    avg_growth = sum(growth_values) / len(growth_values)
                    if avg_growth > 0.03:
                        trend = "growing"
                    elif avg_growth < -0.03:
                        trend = "declining"
                    else:
                        trend = "flat"
                    result_payload["trend"] = trend
                    result_payload["value"] = avg_growth

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
        if tool_name not in {"direct_computation", "chi_square", "predictive_analysis", "prescriptive_analysis"} and tool_func is None:
            continue
        if any(col not in df.columns for col in columns):
            continue
        prepared_df = _prepare_frame(df, columns, tool_name)

        try:
            if tool_name == "direct_computation":
                result = _run_direct_computation(prepared_df, task)
            elif tool_name == "predictive_analysis":
                result = run_predictive_analysis(prepared_df, task, state_context=state_context)
            elif tool_name == "prescriptive_analysis":
                predictive_context = state_context.get("analysis_evidence", {}).get("predictive_result")
                if predictive_context is None:
                    predictive_context = next(
                        (
                            value for value in results.values()
                            if isinstance(value, dict) and value.get("tool") == "predictive_analysis"
                        ),
                        None,
                    )
                result = run_prescriptive_analysis(
                    predictive_result=predictive_context or {"error": "No predictive result found."},
                    question=state_context.get("business_question", ""),
                )
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

        result = _sanitize_numbers(result)

        if _result_is_invalid(result):
            fallback = _fallback_result(prepared_df, tool_name, columns)
            if fallback is not None and not _result_is_invalid(fallback):
                result = fallback

        if result is not None and isinstance(result, dict) and "tool" not in result:
            result["tool"] = tool_name
        if isinstance(result, dict) and result.get("tool") == "predictive_analysis":
            state_context.setdefault("analysis_evidence", {})["predictive_result"] = result
        key = f"{tool_name}_{'_'.join(columns)}" if columns else tool_name
        results[key] = result

    return results
