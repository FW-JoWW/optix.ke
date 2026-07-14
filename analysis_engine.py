from __future__ import annotations

from typing import Any, Dict, List

import math
import warnings
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, errors="coerce")
    non_null = int(series.notna().sum())
    ratio = float(parsed.notna().sum() / non_null) if non_null else 0.0
    return parsed if ratio >= 0.5 else series


def _prepare_frame(df: pd.DataFrame, columns: List[str], tool_name: str) -> pd.DataFrame:
    prepared = df.copy()
    numeric_tools = {"correlation", "summary_statistics", "detect_outliers", "regression", "ttest", "anova"}
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


def _adaptive_support_threshold(counts: pd.Series, floor: int = 30, fraction: float = 0.25) -> int:
    valid = pd.to_numeric(counts, errors="coerce").dropna()
    if valid.empty:
        return floor
    median_count = float(valid.median())
    return max(floor, int(round(median_count * fraction)))


def _unique_columns(columns: List[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for column in columns:
        if column and column not in seen:
            unique.append(column)
            seen.add(column)
    return unique


def _assign_period_bucket(frame: pd.DataFrame, time_column: str, bucket: str) -> pd.DataFrame:
    result = frame.copy()
    result[time_column] = _coerce_datetime_like(result[time_column])
    bucket = str(bucket or "month").lower()
    if bucket == "day":
        result["_period_bucket"] = result[time_column].dt.date.astype(str)
    elif bucket == "week":
        result["_period_bucket"] = result[time_column].dt.to_period("W").astype(str)
    elif bucket == "quarter":
        result["_period_bucket"] = result[time_column].dt.to_period("Q").astype(str)
    elif bucket == "year":
        result["_period_bucket"] = result[time_column].dt.to_period("Y").astype(str)
    else:
        result["_period_bucket"] = result[time_column].dt.to_period("M").astype(str)
    return result


def _event_dates(event_name: str, years: List[int]) -> List[pd.Timestamp]:
    event = str(event_name or "").lower()
    dates: List[pd.Timestamp] = []
    for year in sorted(set(int(y) for y in years if pd.notna(y))):
        if "black friday" in event:
            november = pd.date_range(f"{year}-11-01", f"{year}-11-30", freq="D")
            fridays = [day for day in november if day.weekday() == 4]
            if fridays:
                dates.append(fridays[-1])
    return dates


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
                if bucket == "day":
                    bucket_col = f"{column}__day"
                    current[bucket_col] = current[column].dt.date.astype(str)
                elif bucket == "week":
                    bucket_col = f"{column}__week"
                    current[bucket_col] = current[column].dt.to_period("W").astype(str)
                elif bucket == "month":
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
        elif operation == "period_extremes" and current_series is not None:
            series = pd.to_numeric(current_series, errors="coerce").dropna()
            if len(series):
                top_n = int(params.get("top_n", 1) or 1)
                bottom_n = int(params.get("bottom_n", 1) or 1)
                best = series.sort_values(ascending=False).head(top_n)
                worst = series.sort_values(ascending=True).head(bottom_n)
                result_payload["best_periods"] = _sanitize_numbers(best.reset_index().to_dict(orient="records"))
                result_payload["worst_periods"] = _sanitize_numbers(worst.reset_index().to_dict(orient="records"))
                result_payload["rows"] = _sanitize_numbers(current_series.reset_index().to_dict(orient="records"))
                result_payload["value"] = {
                    "best_period": best.index[0] if len(best) else None,
                    "best_value": float(best.iloc[0]) if len(best) else None,
                    "worst_period": worst.index[0] if len(worst) else None,
                    "worst_value": float(worst.iloc[0]) if len(worst) else None,
                }
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
        elif operation == "missingness_report":
            target_columns = [col for col in params.get("target_columns", []) if col in current.columns] or list(current.columns)
            row_count = int(len(current))
            rows = []
            for target in target_columns:
                missing_count = int(current[target].isna().sum())
                if missing_count:
                    rows.append({"column": target, "missing_count": missing_count, "missing_rate": float(missing_count / row_count) if row_count else 0.0, "dtype": str(current[target].dtype)})
            rows = sorted(rows, key=lambda item: (item["missing_rate"], item["missing_count"]), reverse=True)
            result_payload["rows"] = _sanitize_numbers(rows)
            result_payload["value"] = float(rows[0]["missing_rate"]) if rows else 0.0
            result_payload["summary"] = {"row_count": row_count, "columns_checked": int(len(target_columns)), "columns_with_missing": int(len(rows)), "total_missing_cells": int(sum(row["missing_count"] for row in rows))}
        elif operation == "duplicate_rows_report":
            target_columns = [col for col in params.get("target_columns", []) if col in current.columns] or list(current.columns)
            temp = current[target_columns].copy()
            duplicate_mask = temp.duplicated(keep=False)
            duplicate_rows = int(duplicate_mask.sum())
            rows = []
            duplicate_group_count = 0
            if duplicate_rows:
                duplicate_groups = temp.loc[duplicate_mask].groupby(target_columns, dropna=False).size().reset_index(name="duplicate_count")
                duplicate_groups = duplicate_groups[duplicate_groups["duplicate_count"] > 1].sort_values("duplicate_count", ascending=False)
                duplicate_group_count = int(len(duplicate_groups))
                preview_columns = target_columns[:8]
                rows = duplicate_groups[preview_columns + ["duplicate_count"]].head(int(params.get("top_n", 10) or 10)).to_dict(orient="records")
            result_payload["rows"] = _sanitize_numbers(rows)
            result_payload["value"] = float(duplicate_rows)
            result_payload["summary"] = {"row_count": int(len(current)), "columns_checked": int(len(target_columns)), "duplicate_rows": duplicate_rows, "duplicate_groups": duplicate_group_count}
        elif operation == "timestamp_consistency_report":
            timestamp_columns = [col for col in params.get("timestamp_columns", []) if col in current.columns]
            purchase_column = params.get("purchase_column")
            delivered_column = params.get("delivered_column")
            estimated_column = params.get("estimated_column")
            rows = []
            parsed = {}
            for target in timestamp_columns:
                series = _coerce_datetime_like(current[target])
                invalid_count = int(current[target].notna().sum() - series.notna().sum())
                parsed[target] = series
                if invalid_count:
                    rows.append({"check": "invalid_timestamp_parse", "column": target, "issue_count": invalid_count})
            def add_order_check(left_col: str | None, right_col: str | None, check_name: str) -> None:
                if left_col in parsed and right_col in parsed:
                    mask = parsed[left_col].notna() & parsed[right_col].notna() & (parsed[left_col] > parsed[right_col])
                    count = int(mask.sum())
                    if count:
                        rows.append({"check": check_name, "left_column": left_col, "right_column": right_col, "issue_count": count})
            add_order_check(purchase_column, delivered_column, "purchase_after_delivery")
            add_order_check(purchase_column, estimated_column, "purchase_after_estimated_delivery")
            issue_total = int(sum(row.get("issue_count", 0) for row in rows))
            result_payload["rows"] = _sanitize_numbers(sorted(rows, key=lambda item: item.get("issue_count", 0), reverse=True))
            result_payload["value"] = float(issue_total)
            result_payload["summary"] = {"timestamp_columns_checked": int(len(timestamp_columns)), "issue_count": issue_total}
        elif operation == "numeric_validity_check":
            value_column = params.get("value_column") or column
            min_allowed = float(params.get("min_allowed", 0) or 0)
            allow_zero = bool(params.get("allow_zero", True))
            if value_column in current.columns:
                numeric = pd.to_numeric(current[value_column], errors="coerce")
                non_numeric_count = int(current[value_column].notna().sum() - numeric.notna().sum())
                invalid_mask = numeric < min_allowed if allow_zero else numeric <= min_allowed
                invalid_count = int(invalid_mask.fillna(False).sum()) + non_numeric_count
                rows = []
                if invalid_count:
                    invalid_rows = current.loc[invalid_mask.fillna(False), [value_column]].copy()
                    invalid_rows["issue"] = "below_minimum" if allow_zero else "not_positive"
                    rows = invalid_rows.head(int(params.get("top_n", 10) or 10)).to_dict(orient="records")
                    if non_numeric_count:
                        rows.append({"column": value_column, "issue": "non_numeric_values", "issue_count": non_numeric_count})
                result_payload["rows"] = _sanitize_numbers(rows)
                result_payload["value"] = float(invalid_count)
                result_payload["summary"] = {"column": value_column, "min_allowed": min_allowed, "allow_zero": allow_zero, "invalid_count": invalid_count, "non_numeric_count": non_numeric_count, "min_observed": float(numeric.min()) if numeric.notna().any() else None}
        elif operation == "delivery_date_validity":
            purchase_column = params.get("purchase_column")
            delivered_column = params.get("delivered_column")
            estimated_column = params.get("estimated_column")
            order_column = params.get("order_column")
            needed = _unique_columns([order_column, purchase_column, delivered_column, estimated_column])
            if purchase_column in current.columns and delivered_column in current.columns:
                temp = current[needed].copy()
                for target in [purchase_column, delivered_column, estimated_column]:
                    if target in temp.columns:
                        temp[target] = _coerce_datetime_like(temp[target])
                rows = []
                def collect(mask: pd.Series, issue: str) -> None:
                    if int(mask.fillna(False).sum()):
                        sample = temp.loc[mask.fillna(False)].head(int(params.get("top_n", 10) or 10)).copy()
                        sample["issue"] = issue
                        rows.extend(sample.to_dict(orient="records"))
                collect(temp[delivered_column].notna() & temp[purchase_column].notna() & (temp[delivered_column] < temp[purchase_column]), "delivered_before_purchase")
                if estimated_column in temp.columns:
                    collect(temp[estimated_column].notna() & temp[purchase_column].notna() & (temp[estimated_column] < temp[purchase_column]), "estimated_before_purchase")
                issue_count = int(len(rows))
                result_payload["rows"] = _sanitize_numbers(rows[: int(params.get("top_n", 10) or 10)])
                result_payload["value"] = float(issue_count)
                result_payload["summary"] = {"issue_count": issue_count, "rows_checked": int(len(temp))}
        elif operation == "categorical_label_quality":
            category_column = params.get("category_column") or column
            top_n = int(params.get("top_n", 10) or 10)
            if category_column in current.columns:
                series = current[category_column]
                text = series.astype("string")
                blank_mask = series.isna() | text.str.strip().str.lower().isin(["", "nan", "none", "null", "unknown"])
                normalized = text.str.strip().str.lower().str.replace(r"[\s\-]+", "_", regex=True)
                variants = pd.DataFrame({"raw": text, "normalized": normalized}).dropna()
                variant_counts = variants.groupby("normalized")["raw"].nunique(dropna=True).reset_index(name="variant_count")
                broken = variant_counts[variant_counts["variant_count"] > 1].sort_values("variant_count", ascending=False)
                rows = []
                if int(blank_mask.sum()):
                    rows.append({"issue": "blank_or_missing_label", "issue_count": int(blank_mask.sum()), "column": category_column})
                rows.extend(broken.head(top_n).to_dict(orient="records"))
                result_payload["rows"] = _sanitize_numbers(rows)
                result_payload["value"] = float(sum(row.get("issue_count", row.get("variant_count", 0)) for row in rows))
                result_payload["summary"] = {"column": category_column, "unique_labels": int(series.nunique(dropna=True)), "blank_or_missing_count": int(blank_mask.sum()), "variant_groups": int(len(broken))}
        elif operation == "customer_order_frequency" and column in current.columns:
            entity_column = params.get("entity_column")
            if entity_column in current.columns:
                grouped = current.groupby(entity_column)[column].nunique(dropna=True)
                result_payload["value"] = float(grouped.mean()) if len(grouped) else 0.0
                result_payload["summary"] = {
                    "mean_orders_per_customer": float(grouped.mean()) if len(grouped) else 0.0,
                    "median_orders_per_customer": float(grouped.median()) if len(grouped) else 0.0,
                    "repeat_rate": float((grouped > 1).mean()) if len(grouped) else 0.0,
                    "total_customers": int(len(grouped)),
                }
                current_series = grouped.sort_values(ascending=False).head(10)
        elif operation == "purchase_gap" and column in current.columns:
            entity_column = params.get("entity_column")
            order_column = params.get("order_column")
            if entity_column in current.columns and order_column in current.columns:
                temp = current[[entity_column, order_column, column]].copy()
                temp[column] = _coerce_datetime_like(temp[column])
                temp = temp.dropna(subset=[entity_column, order_column, column])
                if not temp.empty and pd.api.types.is_datetime64_any_dtype(temp[column]):
                    unique_orders = temp.groupby([entity_column, order_column])[column].min().reset_index()
                    unique_orders = unique_orders.sort_values([entity_column, column])
                    gap_rows = []
                    for entity, group in unique_orders.groupby(entity_column):
                        if len(group) < 2:
                            continue
                        ordered = group[column].sort_values().tolist()
                        gap_days = (ordered[1] - ordered[0]).days
                        gap_rows.append({entity_column: entity, "days_between_first_second": float(gap_days)})
                    gap_df = pd.DataFrame(gap_rows)
                    if not gap_df.empty:
                        result_payload["rows"] = gap_df.sort_values("days_between_first_second").head(10).to_dict(orient="records")
                        result_payload["value"] = float(gap_df["days_between_first_second"].mean())
                        result_payload["summary"] = {
                            "mean_days_between_first_second": float(gap_df["days_between_first_second"].mean()),
                            "median_days_between_first_second": float(gap_df["days_between_first_second"].median()),
                            "customers_with_repeat_purchase": int(len(gap_df)),
                        }
        elif operation == "single_purchase_share" and column in current.columns:
            entity_column = params.get("entity_column")
            if entity_column in current.columns:
                grouped = current.groupby(entity_column)[column].nunique(dropna=True)
                single_share = float((grouped == 1).mean()) if len(grouped) else 0.0
                result_payload["value"] = single_share
                result_payload["summary"] = {
                    "single_purchase_entities": int((grouped == 1).sum()),
                    "total_entities": int(len(grouped)),
                }
        elif operation == "basket_value_pattern" and column in current.columns:
            entity_column = params.get("entity_column")
            if entity_column in current.columns:
                temp = current[[entity_column, column]].copy()
                temp[column] = pd.to_numeric(temp[column], errors="coerce")
                temp = temp.dropna(subset=[entity_column, column])
                if not temp.empty:
                    grouped = temp.groupby(entity_column)[column]
                    order_value = grouped.sum()
                    item_count = grouped.count()
                    avg_item_price = order_value / item_count.replace(0, pd.NA)
                    mean_items = float(item_count.mean()) if len(item_count) else 0.0
                    mean_item_price = float(avg_item_price.mean()) if len(avg_item_price.dropna()) else 0.0
                    item_threshold = float(item_count.median()) if len(item_count) else 0.0
                    price_threshold = float(avg_item_price.median()) if len(avg_item_price.dropna()) else 0.0
                    pattern = "many_cheap_items" if mean_items > item_threshold and mean_item_price <= price_threshold else "few_expensive_items"
                    result_payload["value"] = mean_item_price
                    result_payload["summary"] = {
                        "mean_items_per_order": mean_items,
                        "mean_item_price": mean_item_price,
                        "median_items_per_order": item_threshold,
                        "median_item_price": price_threshold,
                        "pattern": pattern,
                    }
        elif operation == "threshold_value_comparison":
            threshold_column = params.get("threshold_column")
            value_column = params.get("value_column") or column
            threshold = float(params.get("threshold", 0) or 0)
            high_label = params.get("higher_group_label", "high_group")
            low_label = params.get("lower_group_label", "low_group")
            if threshold_column in current.columns and value_column in current.columns:
                temp = current[[threshold_column, value_column]].copy()
                temp[threshold_column] = pd.to_numeric(temp[threshold_column], errors="coerce")
                temp[value_column] = pd.to_numeric(temp[value_column], errors="coerce")
                temp = temp.dropna(subset=[threshold_column, value_column])
                if not temp.empty:
                    high_mask = temp[threshold_column] > threshold
                    high_values = temp.loc[high_mask, value_column]
                    low_values = temp.loc[~high_mask, value_column]
                    rows = []
                    if len(high_values):
                        rows.append({"group": high_label, "mean_value": float(high_values.mean()), "count": int(len(high_values))})
                    if len(low_values):
                        rows.append({"group": low_label, "mean_value": float(low_values.mean()), "count": int(len(low_values))})
                    result_payload["rows"] = _sanitize_numbers(rows)
                    if len(high_values) and len(low_values):
                        result_payload["value"] = float(high_values.mean() - low_values.mean())
                    result_payload["summary"] = {
                        "threshold": threshold,
                        "higher_group_label": high_label,
                        "lower_group_label": low_label,
                        "higher_group_mean": float(high_values.mean()) if len(high_values) else None,
                        "lower_group_mean": float(low_values.mean()) if len(low_values) else None,
                    }
        elif operation == "loyalty_trend" and column in current.columns:
            entity_column = params.get("entity_column")
            order_column = params.get("order_column")
            bucket = params.get("bucket", "month")
            if entity_column in current.columns and order_column in current.columns:
                temp = current[[entity_column, order_column, column]].copy()
                temp[column] = _coerce_datetime_like(temp[column])
                temp = temp.dropna(subset=[entity_column, order_column, column])
                if not temp.empty and pd.api.types.is_datetime64_any_dtype(temp[column]):
                    unique_orders = temp.groupby([entity_column, order_column])[column].min().reset_index()
                    if bucket == "quarter":
                        unique_orders["_period_bucket"] = unique_orders[column].dt.to_period("Q").astype(str)
                    elif bucket == "year":
                        unique_orders["_period_bucket"] = unique_orders[column].dt.to_period("Y").astype(str)
                    else:
                        unique_orders["_period_bucket"] = unique_orders[column].dt.to_period("M").astype(str)
                    first_order = unique_orders.groupby(entity_column)[column].min()
                    rows = []
                    for period, group in unique_orders.groupby("_period_bucket"):
                        active_customers = group[entity_column].nunique()
                        repeat_customers = int((first_order.reindex(group[entity_column]).values < group[column].values).sum())
                        loyalty_rate = float(repeat_customers / active_customers) if active_customers else 0.0
                        rows.append({"period": period, "value": active_customers, "loyalty_rate": loyalty_rate})
                    rows = sorted(rows, key=lambda item: item["period"])
                    result_payload["rows"] = rows
                    loyalty_values = [row["loyalty_rate"] for row in rows]
                    if loyalty_values:
                        result_payload["value"] = float(loyalty_values[-1])
                        result_payload["summary"] = {
                            "average_loyalty_rate": float(sum(loyalty_values) / len(loyalty_values)),
                            "latest_loyalty_rate": float(loyalty_values[-1]),
                        }
        elif operation == "dormancy_count" and column in current.columns:
            entity_column = params.get("entity_column")
            if entity_column in current.columns:
                temp = current[[entity_column, column]].copy()
                temp[column] = _coerce_datetime_like(temp[column])
                temp = temp.dropna(subset=[entity_column, column])
                if not temp.empty and pd.api.types.is_datetime64_any_dtype(temp[column]):
                    last_seen = temp.groupby(entity_column)[column].max()
                    latest_date = last_seen.max()
                    earliest_date = temp[column].min()
                    span_days = max(1, int((latest_date - earliest_date).days))
                    dormancy_threshold = max(30, min(180, span_days // 4))
                    dormant = ((latest_date - last_seen).dt.days > dormancy_threshold)
                    result_payload["value"] = int(dormant.sum())
                    result_payload["summary"] = {
                        "dormant_customers": int(dormant.sum()),
                        "total_customers": int(len(last_seen)),
                        "dormancy_threshold_days": int(dormancy_threshold),
                        "dormancy_rate": float(dormant.mean()) if len(last_seen) else 0.0,
                    }
        elif operation == "segment_repeat_rate" and column in current.columns:
            entity_column = params.get("entity_column")
            group_column = params.get("group_column")
            if entity_column in current.columns and group_column in current.columns:
                temp = current[[entity_column, group_column, column]].copy().dropna(subset=[entity_column, group_column, column])
                if not temp.empty:
                    order_counts = temp.groupby([group_column, entity_column])[column].nunique(dropna=True)
                    rows = []
                    for group_name, series in order_counts.groupby(level=0):
                        customer_counts = series.droplevel(0)
                        repeat_rate = float((customer_counts > 1).mean()) if len(customer_counts) else 0.0
                        rows.append({
                            group_column: group_name,
                            "repeat_rate": repeat_rate,
                            "repeat_customers": int((customer_counts > 1).sum()),
                            "total_customers": int(len(customer_counts)),
                        })
                    repeat_df = pd.DataFrame(rows)
                    if not repeat_df.empty:
                        support_threshold = _adaptive_support_threshold(repeat_df["total_customers"], floor=20, fraction=0.25)
                        global_repeat_rate = float(((order_counts > 1).groupby(level=0).sum().sum()) / len(order_counts)) if len(order_counts) else 0.0
                        prior_strength = 20.0
                        repeat_df["smoothed_repeat_rate"] = (
                            repeat_df["repeat_customers"] + prior_strength * global_repeat_rate
                        ) / (repeat_df["total_customers"] + prior_strength)
                        filtered = repeat_df[repeat_df["total_customers"] >= support_threshold].copy()
                        if filtered.empty:
                            filtered = repeat_df[repeat_df["total_customers"] >= max(5, support_threshold // 2)].copy()
                        if filtered.empty:
                            filtered = repeat_df.copy()
                        filtered = filtered.sort_values(["smoothed_repeat_rate", "total_customers"], ascending=[False, False])
                        result_payload["rows"] = _sanitize_numbers(filtered.head(10).to_dict(orient="records"))
                        result_payload["value"] = float(filtered.iloc[0]["smoothed_repeat_rate"])
                        result_payload["summary"] = {
                            "support_threshold": int(support_threshold),
                            "prior_strength": prior_strength,
                        }
        elif operation == "review_repeat_comparison" and column in current.columns:
            entity_column = params.get("entity_column")
            order_column = params.get("order_column")
            if entity_column in current.columns and order_column in current.columns:
                temp = current[[entity_column, order_column, column]].copy()
                temp[column] = pd.to_numeric(temp[column], errors="coerce")
                temp = temp.dropna(subset=[entity_column, order_column, column])
                if not temp.empty:
                    customer_review = temp.groupby(entity_column)[column].mean()
                    order_counts = temp.groupby(entity_column)[order_column].nunique(dropna=True)
                    repeat_flag = order_counts > 1
                    review_threshold = 4.0 if customer_review.max() <= 5.5 else float(customer_review.quantile(0.75))
                    labels = pd.Series("lower_review", index=customer_review.index)
                    labels.loc[customer_review >= review_threshold] = "high_review"
                    rows = []
                    for label, customers in labels.groupby(labels):
                        idx = customers.index
                        rows.append({
                            "review_segment": label,
                            "repeat_rate": float(repeat_flag.reindex(idx).mean()) if len(idx) else 0.0,
                            "customer_count": int(len(idx)),
                            "average_review": float(customer_review.reindex(idx).mean()) if len(idx) else None,
                        })
                    comparison_df = pd.DataFrame(rows)
                    if not comparison_df.empty:
                        result_payload["rows"] = _sanitize_numbers(comparison_df.to_dict(orient="records"))
                        high = comparison_df.loc[comparison_df["review_segment"] == "high_review", "repeat_rate"]
                        low = comparison_df.loc[comparison_df["review_segment"] == "lower_review", "repeat_rate"]
                        if not high.empty and not low.empty:
                            result_payload["value"] = float(high.iloc[0] - low.iloc[0])
        elif operation == "cohort_repeat_rate":
            customer_column = params.get("customer_column")
            order_column = params.get("order_column")
            time_column = params.get("time_column") or column
            bucket = params.get("bucket", "month")
            cohort_filter = str(params.get("cohort_filter", "") or "").lower()
            top_n = int(params.get("top_n", 10) or 10)
            if customer_column in current.columns and order_column in current.columns and time_column in current.columns:
                temp = current[[customer_column, order_column, time_column]].copy()
                temp[time_column] = _coerce_datetime_like(temp[time_column])
                temp = temp.dropna(subset=[customer_column, order_column, time_column])
                if not temp.empty:
                    orders = temp.groupby([customer_column, order_column])[time_column].min().reset_index()
                    first_purchase = orders.groupby(customer_column)[time_column].min()
                    orders["_first_purchase"] = orders[customer_column].map(first_purchase)
                    orders = _assign_period_bucket(orders.rename(columns={"_first_purchase": "_cohort_time"}), "_cohort_time", bucket)
                    rows = []
                    for cohort, frame in orders.groupby("_period_bucket"):
                        if cohort_filter and cohort_filter not in str(cohort).lower():
                            month_name = pd.to_datetime(frame["_cohort_time"]).dt.month_name().str.lower().iloc[0]
                            month_abbrev = pd.to_datetime(frame["_cohort_time"]).dt.strftime("%b").str.lower().iloc[0]
                            if cohort_filter not in month_name and cohort_filter not in month_abbrev:
                                continue
                        customer_counts = frame.groupby(customer_column)[order_column].nunique(dropna=True)
                        repeat_customers = int((customer_counts > 1).sum())
                        total_customers = int(len(customer_counts))
                        rows.append({
                            "cohort": cohort,
                            "repeat_rate": float(repeat_customers / total_customers) if total_customers else 0.0,
                            "repeat_customers": repeat_customers,
                            "total_customers": total_customers,
                        })
                    cohort_df = pd.DataFrame(rows)
                    if not cohort_df.empty:
                        if not cohort_filter:
                            support_threshold = _adaptive_support_threshold(cohort_df["total_customers"], floor=30, fraction=0.25)
                            global_repeat_rate = (
                                float(cohort_df["repeat_customers"].sum() / cohort_df["total_customers"].sum())
                                if float(cohort_df["total_customers"].sum()) else 0.0
                            )
                            prior_strength = 20.0
                            cohort_df["smoothed_repeat_rate"] = (
                                cohort_df["repeat_customers"] + prior_strength * global_repeat_rate
                            ) / (cohort_df["total_customers"] + prior_strength)
                            supported = cohort_df[cohort_df["total_customers"] >= support_threshold].copy()
                            if supported.empty:
                                supported = cohort_df[cohort_df["total_customers"] >= max(5, support_threshold // 2)].copy()
                            if supported.empty:
                                supported = cohort_df.copy()
                            cohort_df = supported.sort_values(["smoothed_repeat_rate", "total_customers"], ascending=[False, False])
                            result_payload["summary"] = {
                                "bucket": bucket,
                                "cohort_filter": cohort_filter,
                                "support_threshold": int(support_threshold),
                                "prior_strength": prior_strength,
                            }
                        else:
                            cohort_df = cohort_df.sort_values(["repeat_rate", "total_customers"], ascending=[False, False])
                            result_payload["summary"] = {"bucket": bucket, "cohort_filter": cohort_filter}
                        cohort_df = cohort_df.head(top_n)
                        result_payload["rows"] = _sanitize_numbers(cohort_df.to_dict(orient="records"))
                        result_payload["value"] = float(cohort_df.iloc[0]["repeat_rate"])
        elif operation == "cohort_value_rank":
            customer_column = params.get("customer_column")
            order_column = params.get("order_column")
            time_column = params.get("time_column")
            value_column = params.get("value_column") or column
            bucket = params.get("bucket", "month")
            top_n = int(params.get("top_n", 10) or 10)
            if customer_column in current.columns and order_column in current.columns and time_column in current.columns and value_column in current.columns:
                temp = current[[customer_column, order_column, time_column, value_column]].copy()
                temp[time_column] = _coerce_datetime_like(temp[time_column])
                temp[value_column] = pd.to_numeric(temp[value_column], errors="coerce")
                temp = temp.dropna(subset=[customer_column, order_column, time_column, value_column])
                if not temp.empty:
                    order_level = temp.groupby([customer_column, order_column]).agg(first_order_time=(time_column, "min"), order_value=(value_column, "max" if "payment" in value_column else "sum")).reset_index()
                    first_purchase = order_level.groupby(customer_column)["first_order_time"].min()
                    order_level["_cohort_time"] = order_level[customer_column].map(first_purchase)
                    order_level = _assign_period_bucket(order_level, "_cohort_time", bucket)
                    cohort_df = order_level.groupby("_period_bucket").agg(
                        long_term_value=("order_value", "sum"),
                        avg_customer_value=("order_value", "mean"),
                        customers=(customer_column, "nunique"),
                        orders=(order_column, "nunique"),
                    ).reset_index().rename(columns={"_period_bucket": "cohort"})
                    if not cohort_df.empty:
                        cohort_df = cohort_df.sort_values(["long_term_value", "customers"], ascending=[False, False]).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(cohort_df.to_dict(orient="records"))
                        result_payload["value"] = float(cohort_df.iloc[0]["long_term_value"])
                        result_payload["summary"] = {"bucket": bucket, "value_column": value_column}
        elif operation == "churn_speed_proxy":
            customer_column = params.get("customer_column")
            order_column = params.get("order_column")
            time_column = params.get("time_column") or column
            if customer_column in current.columns and order_column in current.columns and time_column in current.columns:
                temp = current[[customer_column, order_column, time_column]].copy()
                temp[time_column] = _coerce_datetime_like(temp[time_column])
                temp = temp.dropna(subset=[customer_column, order_column, time_column])
                if not temp.empty:
                    orders = temp.groupby([customer_column, order_column])[time_column].min().reset_index().sort_values([customer_column, time_column])
                    gaps = []
                    for customer, frame in orders.groupby(customer_column):
                        times = frame[time_column].sort_values().tolist()
                        if len(times) > 1:
                            gaps.extend([(times[idx] - times[idx - 1]).days for idx in range(1, len(times))])
                    last_seen = orders.groupby(customer_column)[time_column].max()
                    latest = orders[time_column].max()
                    inactive_days = (latest - last_seen).dt.days
                    result_payload["value"] = float(inactive_days.median()) if len(inactive_days) else None
                    result_payload["summary"] = {
                        "median_days_since_last_purchase": float(inactive_days.median()) if len(inactive_days) else None,
                        "mean_days_since_last_purchase": float(inactive_days.mean()) if len(inactive_days) else None,
                        "median_repeat_gap_days": float(pd.Series(gaps).median()) if gaps else None,
                        "customers": int(len(last_seen)),
                    }
        elif operation == "segment_retention_rate":
            customer_column = params.get("customer_column")
            order_column = params.get("order_column")
            segment_column = params.get("segment_column")
            top_n = int(params.get("top_n", 10) or 10)
            if customer_column in current.columns and order_column in current.columns and segment_column in current.columns:
                temp = current[[customer_column, order_column, segment_column]].copy().dropna(subset=[customer_column, order_column, segment_column])
                if not temp.empty:
                    customer_orders = temp.groupby([segment_column, customer_column])[order_column].nunique(dropna=True)
                    rows = []
                    for segment, series in customer_orders.groupby(level=0):
                        counts = series.droplevel(0)
                        repeat_customers = int((counts > 1).sum())
                        total_customers = int(len(counts))
                        rows.append({
                            segment_column: segment,
                            "repeat_rate": float(repeat_customers / total_customers) if total_customers else 0.0,
                            "repeat_customers": repeat_customers,
                            "total_customers": total_customers,
                        })
                    retention_df = pd.DataFrame(rows)
                    if not retention_df.empty:
                        support_threshold = _adaptive_support_threshold(retention_df["total_customers"], floor=20, fraction=0.25)
                        filtered = retention_df[retention_df["total_customers"] >= support_threshold].copy()
                        if filtered.empty:
                            filtered = retention_df.copy()
                        filtered = filtered.sort_values(["repeat_rate", "total_customers"], ascending=[False, False]).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(filtered.to_dict(orient="records"))
                        result_payload["value"] = float(filtered.iloc[0]["repeat_rate"])
                        result_payload["summary"] = {"support_threshold": int(support_threshold)}
        elif operation in {"categorical_preference_by_entity", "payment_preference_by_entity"}:
            entity_column = params.get("entity_column")
            category_column = params.get("category_column") or params.get("payment_column") or column
            preferred_values = [str(item).lower() for item in (params.get("preferred_values") or [])]
            top_n = int(params.get("top_n", 10) or 10)
            if entity_column in current.columns and category_column in current.columns and preferred_values:
                temp = current[[entity_column, category_column]].copy()
                temp[category_column] = temp[category_column].astype(str).str.lower()
                temp = temp.dropna(subset=[entity_column, category_column])
                rows = []
                for entity, frame in temp.groupby(entity_column):
                    total = int(len(frame))
                    if total < 2:
                        continue
                    category_text = frame[category_column]
                    shares = {value: float(category_text.str.contains(value, na=False).mean()) for value in preferred_values}
                    dominant_value = max(shares, key=shares.get)
                    contrast = float(shares[preferred_values[0]] - shares[preferred_values[1]]) if len(preferred_values) >= 2 else float(shares[dominant_value])
                    row = {
                        entity_column: entity,
                        "dominant_category_value": dominant_value,
                        "dominant_payment_type": dominant_value,
                        "dominant_share": float(shares[dominant_value]),
                        "preference_strength": abs(contrast),
                        "total_orders": total,
                    }
                    for value in preferred_values:
                        row[f"{value}_share"] = float(shares[value])
                    rows.append(row)
                pref_df = pd.DataFrame(rows)
                if not pref_df.empty:
                    support_threshold = _adaptive_support_threshold(pref_df["total_orders"], floor=5, fraction=0.2)
                    filtered = pref_df[pref_df["total_orders"] >= support_threshold].copy()
                    if filtered.empty:
                        filtered = pref_df.copy()
                    pref_df = filtered.sort_values(["preference_strength", "total_orders"], ascending=[False, False]).head(top_n)
                    result_payload["rows"] = _sanitize_numbers(pref_df.to_dict(orient="records"))
                    result_payload["value"] = float(pref_df.iloc[0]["preference_strength"]) if len(pref_df) else None
                    result_payload["summary"] = {"support_threshold": int(support_threshold), "preferred_values": preferred_values, "category_column": category_column}
        elif operation == "segment_order_value" and column in current.columns:
            entity_column = params.get("entity_column")
            order_column = params.get("order_column")
            value_method = params.get("value_method", "sum")
            group_method = params.get("group_method", "mean")
            top_n = int(params.get("top_n", 10) or 10)
            sort_desc = str(params.get("sort", "desc")).lower() != "asc"
            if entity_column in current.columns and order_column in current.columns:
                selected_columns = list(dict.fromkeys([entity_column, order_column, column]))
                temp = current.loc[:, selected_columns].copy()
                value_series = temp[column]
                if isinstance(value_series, pd.DataFrame):
                    value_series = value_series.iloc[:, 0]
                temp[column] = pd.to_numeric(value_series, errors="coerce")
                temp = temp.dropna(subset=[entity_column, order_column, column])
                if not temp.empty:
                    if value_method == "mean":
                        order_values = temp.groupby([entity_column, order_column])[column].mean()
                    else:
                        order_values = temp.groupby([entity_column, order_column])[column].sum()
                    if group_method == "sum":
                        grouped = order_values.groupby(level=0).sum()
                    elif group_method == "median":
                        grouped = order_values.groupby(level=0).median()
                    else:
                        grouped = order_values.groupby(level=0).mean()
                    ranked = grouped.sort_values(ascending=not sort_desc).head(top_n)
                    ranked_df = ranked.rename("avg_order_value").reset_index()
                    result_payload["rows"] = _sanitize_numbers(ranked_df.to_dict(orient="records"))
                    result_payload["value"] = float(ranked.iloc[0]) if len(ranked) else None
                    result_payload["ranking_sort"] = "desc" if sort_desc else "asc"
        elif operation == "grouped_pairwise_relationship":
            entity_column = params.get("entity_column")
            left_column = params.get("left_column") or column
            right_column = params.get("right_column")
            left_method = params.get("left_method", "mean")
            right_method = params.get("right_method", "mean")
            method = str(params.get("method", "spearman")).lower()
            if entity_column in current.columns and left_column in current.columns and right_column in current.columns:
                temp = current[[entity_column, left_column, right_column]].copy()
                temp[left_column] = pd.to_numeric(temp[left_column], errors="coerce")
                temp[right_column] = pd.to_numeric(temp[right_column], errors="coerce")
                temp = temp.dropna(subset=[entity_column, left_column, right_column])
                if not temp.empty:
                    left_group = getattr(temp.groupby(entity_column)[left_column], left_method)()
                    right_group = getattr(temp.groupby(entity_column)[right_column], right_method)()
                    grouped = pd.DataFrame({
                        entity_column: left_group.index,
                        "left_value": left_group.values,
                        "right_value": right_group.reindex(left_group.index).values,
                    }).dropna()
                    if len(grouped) >= 3 and grouped["left_value"].nunique() >= 2 and grouped["right_value"].nunique() >= 2:
                        corr = grouped["left_value"].corr(grouped["right_value"], method=method)
                        result_payload["rows"] = _sanitize_numbers(grouped.to_dict(orient="records"))
                        result_payload["value"] = float(corr) if pd.notna(corr) else None
                        result_payload["summary"] = {
                            "method": method,
                            "group_column": entity_column,
                            "x_column": left_column,
                            "y_column": right_column,
                            "correlation": float(corr) if pd.notna(corr) else None,
                            "groups_used": int(len(grouped)),
                        }
        elif operation == "delivery_duration_rank" and column in current.columns:
            entity_column = params.get("entity_column")
            start_column = params.get("start_column")
            end_column = params.get("end_column") or column
            top_n = int(params.get("top_n", 10) or 10)
            sort_desc = str(params.get("sort", "desc")).lower() != "asc"
            if entity_column in current.columns and start_column in current.columns and end_column in current.columns:
                temp = current[[entity_column, start_column, end_column]].copy()
                temp[start_column] = _coerce_datetime_like(temp[start_column])
                temp[end_column] = _coerce_datetime_like(temp[end_column])
                temp = temp.dropna(subset=[entity_column, start_column, end_column])
                if not temp.empty and pd.api.types.is_datetime64_any_dtype(temp[start_column]) and pd.api.types.is_datetime64_any_dtype(temp[end_column]):
                    temp["_delivery_days"] = (temp[end_column] - temp[start_column]).dt.total_seconds() / 86400.0
                    temp = temp[temp["_delivery_days"] >= 0]
                    if not temp.empty:
                        grouped = temp.groupby(entity_column)["_delivery_days"].mean()
                        ranked = grouped.sort_values(ascending=not sort_desc).head(top_n)
                        result_payload["rows"] = ranked.reset_index().rename(columns={"_delivery_days": "delivery_days"}).to_dict(orient="records")
                        result_payload["value"] = float(ranked.iloc[0]) if len(ranked) else None
                        result_payload["ranking_sort"] = "desc" if sort_desc else "asc"
        elif operation == "delivery_duration_summary":
            start_column = params.get("start_column")
            end_column = params.get("end_column") or column
            if start_column in current.columns and end_column in current.columns:
                temp = current[_unique_columns([start_column, end_column])].copy()
                temp[start_column] = _coerce_datetime_like(temp[start_column])
                temp[end_column] = _coerce_datetime_like(temp[end_column])
                temp = temp.dropna()
                if not temp.empty and pd.api.types.is_datetime64_any_dtype(temp[start_column]) and pd.api.types.is_datetime64_any_dtype(temp[end_column]):
                    delivery_days = (temp[end_column] - temp[start_column]).dt.total_seconds() / 86400.0
                    delivery_days = delivery_days[delivery_days >= 0]
                    if not delivery_days.empty:
                        result_payload["value"] = float(delivery_days.mean())
                        result_payload["summary"] = {
                            "mean_delivery_days": float(delivery_days.mean()),
                            "median_delivery_days": float(delivery_days.median()),
                            "min_delivery_days": float(delivery_days.min()),
                            "max_delivery_days": float(delivery_days.max()),
                        }
        elif operation == "delivery_gap_summary":
            actual_column = params.get("actual_column") or column
            estimated_column = params.get("estimated_column")
            if actual_column in current.columns and estimated_column in current.columns:
                temp = current[_unique_columns([actual_column, estimated_column])].copy()
                temp[actual_column] = _coerce_datetime_like(temp[actual_column])
                temp[estimated_column] = _coerce_datetime_like(temp[estimated_column])
                temp = temp.dropna()
                if not temp.empty and pd.api.types.is_datetime64_any_dtype(temp[actual_column]) and pd.api.types.is_datetime64_any_dtype(temp[estimated_column]):
                    gap_days = (temp[actual_column] - temp[estimated_column]).dt.total_seconds() / 86400.0
                    if not gap_days.empty:
                        result_payload["value"] = float(gap_days.mean())
                        result_payload["summary"] = {
                            "mean_gap_days": float(gap_days.mean()),
                            "median_gap_days": float(gap_days.median()),
                            "late_share": float((gap_days > 0).mean()),
                            "early_share": float((gap_days < 0).mean()),
                        }
        elif operation == "delivery_timing_share":
            actual_column = params.get("actual_column")
            estimated_column = params.get("estimated_column")
            mode = params.get("mode", "late")
            order_column = params.get("order_column") or column
            if actual_column in current.columns and estimated_column in current.columns and order_column in current.columns:
                temp = current[_unique_columns([actual_column, estimated_column, order_column])].copy()
                temp[actual_column] = _coerce_datetime_like(temp[actual_column])
                temp[estimated_column] = _coerce_datetime_like(temp[estimated_column])
                temp = temp.dropna()
                if not temp.empty:
                    order_level = temp.groupby(order_column).agg(actual=(actual_column, "max"), estimated=(estimated_column, "max")).dropna()
                    gap_days = (order_level["actual"] - order_level["estimated"]).dt.total_seconds() / 86400.0
                    share = float((gap_days > 0).mean()) if mode == "late" else float((gap_days < 0).mean())
                    result_payload["value"] = share
                    result_payload["summary"] = {
                        "mode": mode,
                        "matching_orders": int(len(order_level)),
                        "share": share,
                    }
        elif operation == "delay_burden_rank":
            entity_column = params.get("entity_column")
            start_column = params.get("start_column")
            end_column = params.get("end_column") or column
            top_n = int(params.get("top_n", 10) or 10)
            ascending = str(params.get("sort", "desc")).lower() == "asc"
            min_orders = int(params.get("min_orders", 0) or 0)
            if entity_column in current.columns and start_column in current.columns and end_column in current.columns:
                temp = current[_unique_columns([entity_column, start_column, end_column])].copy()
                temp[start_column] = _coerce_datetime_like(temp[start_column])
                temp[end_column] = _coerce_datetime_like(temp[end_column])
                temp = temp.dropna()
                if not temp.empty:
                    temp["_delay_days"] = (temp[end_column] - temp[start_column]).dt.total_seconds() / 86400.0
                    temp = temp[temp["_delay_days"] >= 0]
                    if not temp.empty:
                        grouped = temp.groupby(entity_column).agg(delay_days=("_delay_days", "mean"), total_orders=("_delay_days", "count")).reset_index()
                        if not min_orders:
                            min_orders = _adaptive_support_threshold(grouped["total_orders"], floor=25, fraction=0.25)
                        filtered = grouped[grouped["total_orders"] >= min_orders].copy()
                        if filtered.empty:
                            filtered = grouped[grouped["total_orders"] >= max(10, min_orders // 2)].copy()
                        if filtered.empty:
                            filtered = grouped.copy()
                        ranked = filtered.sort_values(["delay_days", "total_orders"], ascending=[ascending, False]).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(ranked.to_dict(orient="records"))
                        result_payload["value"] = float(ranked.iloc[0]["delay_days"]) if len(ranked) else None
                        result_payload["ranking_sort"] = "asc" if ascending else "desc"
                        result_payload["summary"] = {"support_threshold": int(min_orders)}
        elif operation == "delay_trend":
            start_column = params.get("start_column")
            end_column = params.get("end_column") or column
            time_column = params.get("time_column")
            bucket = params.get("bucket", "month")
            min_orders = int(params.get("min_orders", 0) or 0)
            if start_column in current.columns and end_column in current.columns and time_column in current.columns:
                temp = current[_unique_columns([start_column, end_column, time_column])].copy()
                temp[start_column] = _coerce_datetime_like(temp[start_column])
                temp[end_column] = _coerce_datetime_like(temp[end_column])
                temp[time_column] = _coerce_datetime_like(temp[time_column])
                temp = temp.dropna()
                if not temp.empty:
                    temp["_delay_days"] = (temp[end_column] - temp[start_column]).dt.total_seconds() / 86400.0
                    temp = temp[temp["_delay_days"] >= 0]
                    if bucket == "quarter":
                        temp["_period_bucket"] = temp[time_column].dt.to_period("Q").astype(str)
                    elif bucket == "year":
                        temp["_period_bucket"] = temp[time_column].dt.to_period("Y").astype(str)
                    else:
                        temp["_period_bucket"] = temp[time_column].dt.to_period("M").astype(str)
                    trend_df = temp.groupby("_period_bucket").agg(delay_days=("_delay_days", "mean"), total_orders=("_delay_days", "count")).reset_index()
                    if not min_orders:
                        min_orders = _adaptive_support_threshold(trend_df["total_orders"], floor=25, fraction=0.25)
                    filtered = trend_df[trend_df["total_orders"] >= min_orders].copy()
                    if filtered.empty:
                        filtered = trend_df[trend_df["total_orders"] >= max(10, min_orders // 2)].copy()
                    if filtered.empty:
                        filtered = trend_df.copy()
                    result_payload["rows"] = _sanitize_numbers(filtered.rename(columns={"_period_bucket": "period"}).to_dict(orient="records"))
                    result_payload["value"] = float(filtered.iloc[-1]["delay_days"]) if len(filtered) else None
                    if len(filtered) >= 2:
                        delta = float(filtered.iloc[-1]["delay_days"] - filtered.iloc[0]["delay_days"])
                        tolerance = max(0.1, float(filtered["delay_days"].std(ddof=0) or 0.0) * 0.1)
                        if delta < -tolerance:
                            trend = "improving"
                        elif delta > tolerance:
                            trend = "worsening"
                        else:
                            trend = "flat"
                        result_payload["trend"] = trend
                    result_payload["summary"] = {"support_threshold": int(min_orders), "periods_used": int(len(filtered))}
        elif operation == "delay_quality_relationship":
            start_column = params.get("start_column")
            end_column = params.get("end_column")
            review_column = params.get("review_column") or column
            if start_column in current.columns and end_column in current.columns and review_column in current.columns:
                temp = current[[start_column, end_column, review_column]].copy()
                temp[start_column] = _coerce_datetime_like(temp[start_column])
                temp[end_column] = _coerce_datetime_like(temp[end_column])
                temp[review_column] = pd.to_numeric(temp[review_column], errors="coerce")
                temp = temp.dropna()
                if not temp.empty:
                    temp["_delay_days"] = (temp[end_column] - temp[start_column]).dt.total_seconds() / 86400.0
                    temp = temp[temp["_delay_days"] >= 0]
                    if len(temp) >= 3 and temp["_delay_days"].nunique() >= 2 and temp[review_column].nunique() >= 2:
                        corr = temp["_delay_days"].corr(temp[review_column], method="spearman")
                        result_payload["value"] = float(corr) if pd.notna(corr) else None
                        result_payload["summary"] = {
                            "method": "spearman",
                            "delay_review_correlation": float(corr) if pd.notna(corr) else None,
                            "mean_delay_days": float(temp["_delay_days"].mean()),
                            "mean_review_score": float(temp[review_column].mean()),
                        }
        elif operation == "distance_proxy_cancellation_relationship":
            seller_geo_column = params.get("seller_geo_column")
            customer_geo_column = params.get("customer_geo_column")
            status_column = params.get("status_column")
            order_column = params.get("order_column") or column
            if seller_geo_column in current.columns and customer_geo_column in current.columns and status_column in current.columns and order_column in current.columns:
                temp = current[[seller_geo_column, customer_geo_column, status_column, order_column]].copy()
                temp[seller_geo_column] = pd.to_numeric(temp[seller_geo_column], errors="coerce")
                temp[customer_geo_column] = pd.to_numeric(temp[customer_geo_column], errors="coerce")
                temp = temp.dropna()
                if not temp.empty:
                    order_level = temp.groupby(order_column).agg(
                        seller_geo=(seller_geo_column, "mean"),
                        customer_geo=(customer_geo_column, "mean"),
                        cancelled=(status_column, lambda s: float(s.astype(str).str.lower().str.contains("cancel", na=False).mean() > 0)),
                    ).reset_index()
                    order_level["distance_proxy"] = (order_level["seller_geo"] - order_level["customer_geo"]).abs()
                    if len(order_level) >= 3 and order_level["distance_proxy"].nunique() >= 2:
                        corr = order_level["distance_proxy"].corr(order_level["cancelled"], method="spearman")
                        result_payload["value"] = float(corr) if pd.notna(corr) else None
                        result_payload["summary"] = {
                            "method": "spearman_proxy",
                            "distance_proxy_cancellation_correlation": float(corr) if pd.notna(corr) else None,
                            "proxy_warning": "Uses geographic prefix separation as a distance proxy, not true shipping distance.",
                        }
        elif operation == "status_share":
            status_column = params.get("status_column")
            order_column = params.get("order_column") or column
            mode = params.get("mode", "delivered")
            if status_column in current.columns and order_column in current.columns:
                temp = current[[status_column, order_column]].copy().dropna()
                if not temp.empty:
                    order_level = temp.groupby(order_column)[status_column].agg(lambda s: str(s.iloc[0]).lower())
                    if mode == "invoiced_not_delivered":
                        match = order_level.str.contains("invoiced", na=False)
                    elif mode == "failure":
                        match = order_level.str.contains("cancel|unavailable", na=False)
                    else:
                        match = order_level.str.contains(mode, na=False)
                    share = float(match.mean()) if len(order_level) else 0.0
                    result_payload["value"] = share
                    result_payload["summary"] = {"mode": mode, "matching_orders": int(match.sum()), "total_orders": int(len(order_level))}
        elif operation == "status_rate_by_entity":
            entity_column = params.get("entity_column")
            status_column = params.get("status_column")
            order_column = params.get("order_column") or column
            mode = params.get("mode", "canceled")
            top_n = int(params.get("top_n", 10) or 10)
            prior_strength = float(params.get("prior_strength", 25.0) or 25.0)
            min_orders = int(params.get("min_orders", 0) or 0)
            if entity_column in current.columns and status_column in current.columns and order_column in current.columns:
                temp = current[[entity_column, status_column, order_column]].copy().dropna()
                if not temp.empty:
                    grouped = temp.groupby([entity_column, order_column])[status_column].agg(lambda s: str(s.iloc[0]).lower()).reset_index()
                    if mode == "failure":
                        grouped["_match"] = grouped[status_column].str.contains("cancel|unavailable", na=False).astype(float)
                    else:
                        grouped["_match"] = grouped[status_column].str.contains(mode, na=False).astype(float)
                    rate_df = grouped.groupby(entity_column).agg(failure_rate=("_match", "mean"), affected_orders=("_match", "sum"), total_orders=(order_column, "nunique")).reset_index()
                    if not min_orders:
                        min_orders = _adaptive_support_threshold(rate_df["total_orders"], floor=30, fraction=0.25)
                    global_rate = float(grouped["_match"].mean()) if len(grouped) else 0.0
                    rate_df["smoothed_failure_rate"] = (
                        rate_df["affected_orders"] + prior_strength * global_rate
                    ) / (rate_df["total_orders"] + prior_strength)
                    filtered = rate_df[rate_df["total_orders"] >= min_orders].copy()
                    if filtered.empty:
                        fallback_threshold = max(10, min_orders // 2)
                        filtered = rate_df[rate_df["total_orders"] >= fallback_threshold].copy()
                    if filtered.empty:
                        filtered = rate_df.copy()
                    rate_df = filtered.sort_values(["smoothed_failure_rate", "total_orders"], ascending=[False, False]).head(top_n)
                    result_payload["rows"] = _sanitize_numbers(rate_df.to_dict(orient="records"))
                    result_payload["value"] = float(rate_df.iloc[0]["smoothed_failure_rate"]) if len(rate_df) else None
                    result_payload["summary"] = {
                        "global_failure_rate": global_rate,
                        "support_threshold": int(min_orders),
                        "prior_strength": prior_strength,
                    }
        elif operation == "operational_issue_score":
            status_column = params.get("status_column")
            order_column = params.get("order_column") or column
            time_column = params.get("time_column")
            bucket = params.get("bucket", "month")
            prior_strength = float(params.get("prior_strength", 20.0) or 20.0)
            if status_column in current.columns and order_column in current.columns and time_column in current.columns:
                temp = current[[status_column, order_column, time_column]].copy()
                temp[time_column] = _coerce_datetime_like(temp[time_column])
                temp = temp.dropna()
                if not temp.empty:
                    if bucket == "quarter":
                        temp["_period_bucket"] = temp[time_column].dt.to_period("Q").astype(str)
                    elif bucket == "year":
                        temp["_period_bucket"] = temp[time_column].dt.to_period("Y").astype(str)
                    else:
                        temp["_period_bucket"] = temp[time_column].dt.to_period("M").astype(str)
                    order_level = temp.groupby([order_column, "_period_bucket"])[status_column].agg(lambda s: str(s.iloc[0]).lower()).reset_index()
                    order_level["_issue"] = order_level[status_column].str.contains("cancel|unavailable|invoiced", na=False).astype(float)
                    issue_df = order_level.groupby("_period_bucket").agg(issue_rate=("_issue", "mean"), issue_orders=("_issue", "sum"), total_orders=(order_column, "nunique")).reset_index()
                    min_orders = _adaptive_support_threshold(issue_df["total_orders"], floor=30, fraction=0.25)
                    global_issue_rate = float(order_level["_issue"].mean()) if len(order_level) else 0.0
                    issue_df["smoothed_issue_rate"] = (
                        issue_df["issue_orders"] + prior_strength * global_issue_rate
                    ) / (issue_df["total_orders"] + prior_strength)
                    issue_df = issue_df[issue_df["total_orders"] >= min_orders].copy()
                    if issue_df.empty:
                        issue_df = order_level.groupby("_period_bucket").agg(issue_rate=("_issue", "mean"), issue_orders=("_issue", "sum"), total_orders=(order_column, "nunique")).reset_index()
                        issue_df["smoothed_issue_rate"] = (
                            issue_df["issue_orders"] + prior_strength * global_issue_rate
                        ) / (issue_df["total_orders"] + prior_strength)
                    issue_df = issue_df.sort_values(["smoothed_issue_rate", "total_orders"], ascending=[False, False])
                    result_payload["rows"] = _sanitize_numbers(issue_df.to_dict(orient="records"))
                    result_payload["value"] = float(issue_df.iloc[0]["smoothed_issue_rate"]) if len(issue_df) else None
                    result_payload["summary"] = {
                        "global_issue_rate": global_issue_rate,
                        "support_threshold": int(min_orders),
                        "prior_strength": prior_strength,
                    }
        elif operation == "status_rate_trend":
            status_column = params.get("status_column")
            order_column = params.get("order_column") or column
            time_column = params.get("time_column")
            mode = params.get("mode", "canceled")
            bucket = params.get("bucket", "month")
            prior_strength = float(params.get("prior_strength", 20.0) or 20.0)
            if status_column in current.columns and order_column in current.columns and time_column in current.columns:
                temp = current[[status_column, order_column, time_column]].copy()
                temp[time_column] = _coerce_datetime_like(temp[time_column])
                temp = temp.dropna()
                if not temp.empty:
                    if bucket == "quarter":
                        temp["_period_bucket"] = temp[time_column].dt.to_period("Q").astype(str)
                    elif bucket == "year":
                        temp["_period_bucket"] = temp[time_column].dt.to_period("Y").astype(str)
                    else:
                        temp["_period_bucket"] = temp[time_column].dt.to_period("M").astype(str)
                    order_level = temp.groupby([order_column, "_period_bucket"])[status_column].agg(lambda s: str(s.iloc[0]).lower()).reset_index()
                    match = order_level[status_column].str.contains(mode, na=False).astype(float)
                    trend_df = order_level.assign(_match=match).groupby("_period_bucket").agg(rate=("_match", "mean"), total_orders=(order_column, "nunique")).reset_index()
                    min_orders = _adaptive_support_threshold(trend_df["total_orders"], floor=30, fraction=0.25)
                    global_rate = float(match.mean()) if len(order_level) else 0.0
                    trend_df["smoothed_rate"] = (
                        trend_df["rate"] * trend_df["total_orders"] + prior_strength * global_rate
                    ) / (trend_df["total_orders"] + prior_strength)
                    filtered = trend_df[trend_df["total_orders"] >= min_orders].copy()
                    if filtered.empty:
                        filtered = trend_df.copy()
                    result_payload["rows"] = _sanitize_numbers(filtered.to_dict(orient="records"))
                    result_payload["value"] = float(filtered.iloc[-1]["smoothed_rate"]) if len(filtered) else None
                    if len(filtered) >= 2:
                        delta = float(filtered.iloc[-1]["smoothed_rate"] - filtered.iloc[0]["smoothed_rate"])
                        tolerance = max(0.001, abs(global_rate) * 0.1)
                        if delta > tolerance:
                            trend = "increasing"
                        elif delta < -tolerance:
                            trend = "decreasing"
                        else:
                            trend = "flat"
                        result_payload["trend"] = trend
                    result_payload["summary"] = {
                        "global_rate": global_rate,
                        "support_threshold": int(min_orders),
                        "prior_strength": prior_strength,
                        "periods_used": int(len(filtered)),
                    }
        elif operation == "time_series_metric":
            time_column = params.get("time_column")
            metric_column = params.get("metric_column") or column
            entity_column = params.get("entity_column")
            method = params.get("method", "sum")
            bucket = params.get("bucket", "month")
            needed = _unique_columns([time_column, metric_column, entity_column])
            if time_column in current.columns and metric_column in current.columns:
                temp = current[needed].copy()
                temp[time_column] = _coerce_datetime_like(temp[time_column])
                temp = temp.dropna(subset=[time_column])
                if not temp.empty:
                    temp = _assign_period_bucket(temp, time_column, bucket)
                    if method == "distinct_count":
                        target = entity_column if entity_column in temp.columns else metric_column
                        series = temp.groupby("_period_bucket")[target].nunique(dropna=True)
                    elif method == "mean":
                        temp[metric_column] = pd.to_numeric(temp[metric_column], errors="coerce")
                        series = temp.dropna(subset=[metric_column]).groupby("_period_bucket")[metric_column].mean()
                    elif method == "count":
                        series = temp.groupby("_period_bucket")[metric_column].count()
                    else:
                        temp[metric_column] = pd.to_numeric(temp[metric_column], errors="coerce")
                        series = temp.dropna(subset=[metric_column]).groupby("_period_bucket")[metric_column].sum()
                    rows = pd.DataFrame({"period": series.index.astype(str), "value": series.values})
                    result_payload["rows"] = _sanitize_numbers(rows.to_dict(orient="records"))
                    result_payload["value"] = float(series.iloc[-1]) if len(series) else None
                    result_payload["summary"] = {"bucket": bucket, "method": method, "periods_used": int(len(series))}
        elif operation == "weekday_segment_compare":
            time_column = params.get("time_column")
            metric_column = params.get("metric_column") or column
            entity_column = params.get("entity_column")
            method = params.get("method", "sum")
            needed = _unique_columns([time_column, metric_column, entity_column])
            if time_column in current.columns and metric_column in current.columns:
                temp = current[needed].copy()
                temp[time_column] = _coerce_datetime_like(temp[time_column])
                temp = temp.dropna(subset=[time_column])
                if not temp.empty:
                    temp["_weekday_segment"] = temp[time_column].dt.weekday.apply(lambda day: "weekend" if day >= 5 else "weekday")
                    if method == "distinct_count":
                        target = entity_column if entity_column in temp.columns else metric_column
                        series = temp.groupby("_weekday_segment")[target].nunique(dropna=True)
                    elif method == "mean":
                        temp[metric_column] = pd.to_numeric(temp[metric_column], errors="coerce")
                        series = temp.dropna(subset=[metric_column]).groupby("_weekday_segment")[metric_column].mean()
                    else:
                        temp[metric_column] = pd.to_numeric(temp[metric_column], errors="coerce")
                        series = temp.dropna(subset=[metric_column]).groupby("_weekday_segment")[metric_column].sum()
                    rows = series.reset_index().rename(columns={"_weekday_segment": "segment", metric_column: "value", 0: "value"})
                    if "value" not in rows.columns:
                        rows = rows.rename(columns={rows.columns[-1]: "value"})
                    result_payload["rows"] = _sanitize_numbers(rows.to_dict(orient="records"))
                    result_payload["value"] = float(series.max()) if len(series) else None
                    result_payload["summary"] = {"method": method}
        elif operation == "temporal_spike_detection":
            time_column = params.get("time_column")
            metric_column = params.get("metric_column") or column
            entity_column = params.get("entity_column")
            method = params.get("method", "sum")
            bucket = params.get("bucket", "day")
            needed = _unique_columns([time_column, metric_column, entity_column])
            if time_column in current.columns and metric_column in current.columns:
                temp = current[needed].copy()
                temp[time_column] = _coerce_datetime_like(temp[time_column])
                temp = temp.dropna(subset=[time_column])
                if not temp.empty:
                    temp = _assign_period_bucket(temp, time_column, bucket)
                    if method == "distinct_count":
                        target = entity_column if entity_column in temp.columns else metric_column
                        series = temp.groupby("_period_bucket")[target].nunique(dropna=True)
                    else:
                        temp[metric_column] = pd.to_numeric(temp[metric_column], errors="coerce")
                        series = temp.dropna(subset=[metric_column]).groupby("_period_bucket")[metric_column].sum()
                    mean = float(series.mean()) if len(series) else 0.0
                    std = float(series.std(ddof=0)) if len(series) else 0.0
                    spike_df = series.reset_index()
                    spike_df.columns = ["period", "value"]
                    spike_df["spike_score"] = (spike_df["value"] - mean) / std if std else 0.0
                    spike_df = spike_df.sort_values(["spike_score", "value"], ascending=[False, False]).head(int(params.get("top_n", 10) or 10))
                    result_payload["rows"] = _sanitize_numbers(spike_df.to_dict(orient="records"))
                    result_payload["value"] = float(spike_df.iloc[0]["spike_score"]) if len(spike_df) else None
                    result_payload["summary"] = {"bucket": bucket, "baseline_mean": mean, "baseline_std": std}
        elif operation == "rapid_repeat_order_anomaly":
            customer_column = params.get("customer_column")
            order_column = params.get("order_column")
            time_column = params.get("time_column") or column
            threshold_hours = float(params.get("threshold_hours", 24) or 24)
            top_n = int(params.get("top_n", 10) or 10)
            if customer_column in current.columns and order_column in current.columns and time_column in current.columns:
                temp = current[_unique_columns([customer_column, order_column, time_column])].copy()
                temp[time_column] = _coerce_datetime_like(temp[time_column])
                temp = temp.dropna(subset=[customer_column, order_column, time_column])
                if not temp.empty:
                    orders = temp.groupby([customer_column, order_column])[time_column].min().reset_index()
                    orders = orders.sort_values([customer_column, time_column])
                    orders["_previous_order_time"] = orders.groupby(customer_column)[time_column].shift(1)
                    orders["gap_hours"] = (orders[time_column] - orders["_previous_order_time"]).dt.total_seconds() / 3600.0
                    rapid = orders[(orders["gap_hours"].notna()) & (orders["gap_hours"] >= 0) & (orders["gap_hours"] <= threshold_hours)].copy()
                    if not rapid.empty:
                        grouped = rapid.groupby(customer_column).agg(
                            rapid_order_pairs=("gap_hours", "count"),
                            min_gap_hours=("gap_hours", "min"),
                            median_gap_hours=("gap_hours", "median"),
                            latest_rapid_order=(time_column, "max"),
                        ).reset_index()
                        order_counts = orders.groupby(customer_column)[order_column].nunique(dropna=True)
                        grouped["total_orders"] = grouped[customer_column].map(order_counts).fillna(0).astype(int)
                        grouped["anomaly_score"] = grouped["rapid_order_pairs"].rank(pct=True) + (1.0 - grouped["min_gap_hours"].rank(pct=True))
                        grouped = grouped.sort_values(["anomaly_score", "rapid_order_pairs", "min_gap_hours"], ascending=[False, False, True]).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(grouped.to_dict(orient="records"))
                        result_payload["value"] = float(grouped.iloc[0]["anomaly_score"]) if len(grouped) else None
                    result_payload["summary"] = {"threshold_hours": threshold_hours, "rapid_customers": int(rapid[customer_column].nunique()) if not rapid.empty else 0}
        elif operation == "transaction_value_outlier_rank":
            order_column = params.get("order_column")
            value_column = params.get("value_column") or column
            top_n = int(params.get("top_n", 10) or 10)
            if order_column in current.columns and value_column in current.columns:
                temp = current[_unique_columns([order_column, value_column])].copy()
                temp[value_column] = pd.to_numeric(temp[value_column], errors="coerce")
                temp = temp.dropna(subset=[order_column, value_column])
                if not temp.empty:
                    order_values = temp.groupby(order_column)[value_column].max() if "payment" in value_column else temp.groupby(order_column)[value_column].sum()
                    q1 = float(order_values.quantile(0.25))
                    q3 = float(order_values.quantile(0.75))
                    iqr = q3 - q1
                    upper = q3 + 1.5 * iqr
                    mean = float(order_values.mean())
                    std = float(order_values.std(ddof=0) or 0.0)
                    outlier_df = order_values.reset_index().rename(columns={value_column: "transaction_value"})
                    outlier_df["z_score"] = (outlier_df["transaction_value"] - mean) / std if std else 0.0
                    outlier_df["iqr_excess"] = outlier_df["transaction_value"] - upper
                    outlier_df["anomaly_score"] = outlier_df["z_score"].clip(lower=0) + (outlier_df["iqr_excess"].clip(lower=0) / max(iqr, 1.0))
                    outlier_df = outlier_df[outlier_df["transaction_value"] >= upper].sort_values(["anomaly_score", "transaction_value"], ascending=[False, False]).head(top_n)
                    result_payload["rows"] = _sanitize_numbers(outlier_df.to_dict(orient="records"))
                    result_payload["value"] = float(outlier_df.iloc[0]["anomaly_score"]) if len(outlier_df) else None
                    result_payload["summary"] = {"upper_fence": upper, "mean": mean, "std": std, "outlier_orders": int(len(outlier_df))}
        elif operation == "contextual_metric_mismatch":
            entity_column = params.get("entity_column")
            low_column = params.get("low_column")
            high_column = params.get("high_column") or column
            order_column = params.get("order_column")
            top_n = int(params.get("top_n", 10) or 10)
            needed = _unique_columns([entity_column, order_column, low_column, high_column])
            if low_column in current.columns and high_column in current.columns:
                temp = current[needed].copy()
                temp[low_column] = pd.to_numeric(temp[low_column], errors="coerce")
                temp[high_column] = pd.to_numeric(temp[high_column], errors="coerce")
                subset = [low_column, high_column] + ([entity_column] if entity_column in temp.columns else [])
                temp = temp.dropna(subset=subset)
                if not temp.empty:
                    low_threshold = float(temp[low_column].quantile(0.25))
                    high_threshold = float(temp[high_column].quantile(0.75))
                    flagged = temp[(temp[low_column] <= low_threshold) & (temp[high_column] >= high_threshold)].copy()
                    if not flagged.empty:
                        flagged["mismatch_score"] = (
                            (high_threshold and flagged[high_column] / max(high_threshold, 1.0))
                            + (low_threshold / flagged[low_column].clip(lower=0.01))
                        )
                        sort_cols = ["mismatch_score", high_column]
                        flagged = flagged.sort_values(sort_cols, ascending=[False, False]).head(top_n)
                        rename = {low_column: "low_context_value", high_column: "high_signal_value"}
                        result_payload["rows"] = _sanitize_numbers(flagged.rename(columns=rename).to_dict(orient="records"))
                        result_payload["value"] = float(flagged.iloc[0]["mismatch_score"]) if len(flagged) else None
                    result_payload["summary"] = {"low_threshold": low_threshold, "high_threshold": high_threshold, "flagged_records": int(len(flagged))}
        elif operation == "entity_temporal_spike_detection":
            entity_column = params.get("entity_column")
            time_column = params.get("time_column")
            metric_column = params.get("metric_column") or column
            method = params.get("method", "distinct_count")
            bucket = params.get("bucket", "day")
            top_n = int(params.get("top_n", 10) or 10)
            if entity_column in current.columns and time_column in current.columns and metric_column in current.columns:
                temp = current[_unique_columns([entity_column, time_column, metric_column])].copy()
                temp[time_column] = _coerce_datetime_like(temp[time_column])
                temp = temp.dropna(subset=[entity_column, time_column])
                if not temp.empty:
                    temp = _assign_period_bucket(temp, time_column, bucket)
                    if method == "distinct_count":
                        grouped = temp.groupby([entity_column, "_period_bucket"])[metric_column].nunique(dropna=True)
                    else:
                        temp[metric_column] = pd.to_numeric(temp[metric_column], errors="coerce")
                        grouped = temp.dropna(subset=[metric_column]).groupby([entity_column, "_period_bucket"])[metric_column].sum()
                    rows = []
                    for entity, series in grouped.groupby(level=0):
                        values = series.droplevel(0)
                        if len(values) < 3:
                            continue
                        mean = float(values.mean())
                        std = float(values.std(ddof=0) or 0.0)
                        if not std:
                            continue
                        for period, value in values.items():
                            spike_score = float((value - mean) / std)
                            if spike_score > 0:
                                rows.append({entity_column: entity, "period": period, "value": float(value), "spike_score": spike_score, "baseline_mean": mean})
                    spike_df = pd.DataFrame(rows)
                    if not spike_df.empty:
                        spike_df = spike_df.sort_values(["spike_score", "value"], ascending=[False, False]).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(spike_df.to_dict(orient="records"))
                        result_payload["value"] = float(spike_df.iloc[0]["spike_score"])
                        result_payload["summary"] = {"bucket": bucket, "entity_column": entity_column, "method": method}
        elif operation == "duplicate_behavior_fingerprint":
            customer_column = params.get("customer_column")
            order_column = params.get("order_column")
            fingerprint_columns = [col for col in params.get("fingerprint_columns", []) if col in current.columns]
            value_column = params.get("value_column")
            top_n = int(params.get("top_n", 10) or 10)
            if customer_column in current.columns and order_column in current.columns and fingerprint_columns:
                needed = _unique_columns([customer_column, order_column, value_column] + fingerprint_columns)
                temp = current[needed].copy().dropna(subset=[customer_column])
                if value_column in temp.columns:
                    temp[value_column] = pd.to_numeric(temp[value_column], errors="coerce")
                customer_count = int(temp[customer_column].nunique(dropna=True))
                adaptive_columns = []
                for fp_col in fingerprint_columns:
                    unique_count = int(temp[fp_col].nunique(dropna=True))
                    lower = str(fp_col).lower()
                    if unique_count <= max(100, int(customer_count * 0.05)) or any(token in lower for token in ["state", "category", "payment", "type", "status"]):
                        adaptive_columns.append(fp_col)
                if adaptive_columns:
                    fingerprint_columns = adaptive_columns
                customer_rows = []
                for customer, frame in temp.groupby(customer_column):
                    row = {customer_column: customer, "order_count": int(frame[order_column].nunique(dropna=True))}
                    for fp_col in fingerprint_columns:
                        mode = frame[fp_col].dropna().mode()
                        row[fp_col] = str(mode.iloc[0]) if not mode.empty else ""
                    if value_column in temp.columns:
                        avg_value = frame[value_column].mean()
                        row["avg_value_band"] = int(round(float(avg_value if pd.notna(avg_value) else 0.0) / 50.0))
                    customer_rows.append(row)
                customer_df = pd.DataFrame(customer_rows)
                if not customer_df.empty:
                    fp_cols = [col for col in customer_df.columns if col != customer_column]
                    grouped = customer_df.groupby(fp_cols).agg(matching_customers=(customer_column, "nunique")).reset_index()
                    grouped = grouped[grouped["matching_customers"] > 1].copy()
                    if grouped.empty and len(fp_cols) > 2:
                        coarse_cols = [col for col in fp_cols if col in {"order_count", "avg_value_band"} or any(token in str(col).lower() for token in ["state", "category", "payment", "type", "status"])]
                        if coarse_cols:
                            grouped = customer_df.groupby(coarse_cols).agg(matching_customers=(customer_column, "nunique")).reset_index()
                            grouped = grouped[grouped["matching_customers"] > 1].copy()
                    if not grouped.empty:
                        grouped["duplicate_score"] = grouped["matching_customers"].rank(pct=True) + grouped.get("order_count", pd.Series(0, index=grouped.index)).rank(pct=True) * 0.25
                        grouped = grouped.sort_values(["duplicate_score", "matching_customers"], ascending=[False, False]).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(grouped.to_dict(orient="records"))
                        result_payload["value"] = float(grouped.iloc[0]["duplicate_score"])
                    result_payload["summary"] = {"fingerprint_columns": fingerprint_columns, "duplicate_patterns": int(len(grouped))}
        elif operation == "review_pattern_anomaly":
            entity_column = params.get("entity_column")
            review_column = params.get("review_column") or column
            order_column = params.get("order_column")
            top_n = int(params.get("top_n", 10) or 10)
            if entity_column in current.columns and review_column in current.columns:
                temp = current[_unique_columns([entity_column, review_column, order_column])].copy()
                temp[review_column] = pd.to_numeric(temp[review_column], errors="coerce")
                temp = temp.dropna(subset=[entity_column, review_column])
                if not temp.empty:
                    rows = []
                    max_score = float(temp[review_column].max())
                    min_score = float(temp[review_column].min())
                    for entity, frame in temp.groupby(entity_column):
                        support = int(frame[order_column].nunique(dropna=True)) if order_column in frame.columns else int(len(frame))
                        if support < 3:
                            continue
                        counts = frame[review_column].value_counts(normalize=True)
                        repeated_share = float(counts.max()) if not counts.empty else 0.0
                        perfect_rate = float((frame[review_column] >= max_score).mean())
                        low_rate = float((frame[review_column] <= min_score + 1).mean())
                        review_std = float(frame[review_column].std(ddof=0) or 0.0)
                        rows.append({entity_column: entity, "review_count": support, "repeated_score_share": repeated_share, "perfect_review_rate": perfect_rate, "low_review_rate": low_rate, "review_std": review_std})
                    review_df = pd.DataFrame(rows)
                    if not review_df.empty:
                        support_threshold = _adaptive_support_threshold(review_df["review_count"], floor=5, fraction=0.2)
                        filtered = review_df[review_df["review_count"] >= support_threshold].copy()
                        if filtered.empty:
                            filtered = review_df.copy()
                        filtered["review_anomaly_score"] = (
                            filtered["repeated_score_share"].rank(pct=True)
                            + filtered["perfect_review_rate"].rank(pct=True) * 0.5
                            + filtered["low_review_rate"].rank(pct=True) * 0.5
                            + filtered["review_count"].rank(pct=True) * 0.25
                        )
                        filtered = filtered.sort_values(["review_anomaly_score", "review_count"], ascending=[False, False]).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(filtered.to_dict(orient="records"))
                        result_payload["value"] = float(filtered.iloc[0]["review_anomaly_score"])
                        result_payload["summary"] = {"support_threshold": int(support_threshold), "entity_column": entity_column}
        elif operation == "geographic_anomaly_score":
            geography_column = params.get("geography_column") or params.get("entity_column")
            order_column = params.get("order_column")
            value_column = params.get("value_column")
            review_column = params.get("review_column")
            freight_column = params.get("freight_column")
            top_n = int(params.get("top_n", 10) or 10)
            needed = _unique_columns([geography_column, order_column, value_column, review_column, freight_column])
            if geography_column in current.columns and order_column in current.columns:
                temp = current[needed].copy().dropna(subset=[geography_column, order_column])
                for metric in [value_column, review_column, freight_column]:
                    if metric in temp.columns:
                        temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
                rows = []
                for geo, frame in temp.groupby(geography_column):
                    row = {geography_column: geo, "total_orders": int(frame[order_column].nunique(dropna=True))}
                    if value_column in frame.columns:
                        row["avg_value"] = float(frame[value_column].mean())
                    if review_column in frame.columns:
                        row["avg_review"] = float(frame[review_column].mean())
                    if freight_column in frame.columns:
                        row["avg_freight"] = float(frame[freight_column].mean())
                    rows.append(row)
                geo_df = pd.DataFrame(rows)
                if not geo_df.empty:
                    support_threshold = _adaptive_support_threshold(geo_df["total_orders"], floor=20, fraction=0.25)
                    filtered = geo_df[geo_df["total_orders"] >= support_threshold].copy()
                    if filtered.empty:
                        filtered = geo_df.copy()
                    score = pd.Series(0.0, index=filtered.index)
                    for metric in ["total_orders", "avg_value", "avg_freight"]:
                        if metric in filtered.columns and filtered[metric].nunique() > 1:
                            score += ((filtered[metric] - filtered[metric].mean()) / (filtered[metric].std(ddof=0) or 1.0)).abs()
                    if "avg_review" in filtered.columns and filtered["avg_review"].nunique() > 1:
                        score += ((filtered["avg_review"] - filtered["avg_review"].mean()) / (filtered["avg_review"].std(ddof=0) or 1.0)).abs() * 0.75
                    filtered["geo_anomaly_score"] = score
                    filtered = filtered.sort_values(["geo_anomaly_score", "total_orders"], ascending=[False, False]).head(top_n)
                    result_payload["rows"] = _sanitize_numbers(filtered.to_dict(orient="records"))
                    result_payload["value"] = float(filtered.iloc[0]["geo_anomaly_score"]) if len(filtered) else None
                    result_payload["summary"] = {"support_threshold": int(support_threshold), "geography_column": geography_column}
        elif operation == "event_window_impact":
            time_column = params.get("time_column")
            metric_column = params.get("metric_column") or column
            entity_column = params.get("entity_column")
            event_name = params.get("event_name", "event")
            window_days = int(params.get("window_days", 7) or 7)
            needed = _unique_columns([time_column, metric_column, entity_column])
            if time_column in current.columns and metric_column in current.columns:
                temp = current[needed].copy()
                temp[time_column] = _coerce_datetime_like(temp[time_column])
                temp = temp.dropna(subset=[time_column])
                if not temp.empty:
                    years = temp[time_column].dt.year.dropna().astype(int).tolist()
                    events = _event_dates(event_name, years)
                    if events:
                        if params.get("method") == "distinct_count":
                            target = entity_column if entity_column in temp.columns else metric_column
                            daily = temp.groupby(temp[time_column].dt.date)[target].nunique(dropna=True)
                        else:
                            temp[metric_column] = pd.to_numeric(temp[metric_column], errors="coerce")
                            daily = temp.dropna(subset=[metric_column]).groupby(temp[time_column].dt.date)[metric_column].sum()
                        daily.index = pd.to_datetime(daily.index)
                        rows = []
                        for event_date in events:
                            event_mask = (daily.index >= event_date - pd.Timedelta(days=window_days)) & (daily.index <= event_date + pd.Timedelta(days=window_days))
                            baseline_mask = (daily.index >= event_date - pd.Timedelta(days=window_days * 4)) & (daily.index < event_date - pd.Timedelta(days=window_days))
                            event_mean = float(daily.loc[event_mask].mean()) if event_mask.any() else 0.0
                            baseline_mean = float(daily.loc[baseline_mask].mean()) if baseline_mask.any() else 0.0
                            impact = (event_mean - baseline_mean) / baseline_mean if baseline_mean else None
                            rows.append({
                                "event": event_name,
                                "event_date": str(event_date.date()),
                                "event_window_mean": event_mean,
                                "baseline_mean": baseline_mean,
                                "impact_ratio": float(impact) if impact is not None else None,
                            })
                        result_payload["rows"] = _sanitize_numbers(rows)
                        valid = [row["impact_ratio"] for row in rows if row.get("impact_ratio") is not None]
                        result_payload["value"] = float(sum(valid) / len(valid)) if valid else None
                        result_payload["summary"] = {"event_name": event_name, "window_days": window_days}
        elif operation == "customer_acquisition_trend":
            customer_column = params.get("customer_column") or column
            time_column = params.get("time_column")
            bucket = params.get("bucket", "month")
            if customer_column in current.columns and time_column in current.columns:
                temp = current[[customer_column, time_column]].copy()
                temp[time_column] = _coerce_datetime_like(temp[time_column])
                temp = temp.dropna(subset=[customer_column, time_column])
                if not temp.empty:
                    first_seen = temp.groupby(customer_column)[time_column].min().reset_index()
                    first_seen = _assign_period_bucket(first_seen, time_column, bucket)
                    series = first_seen.groupby("_period_bucket")[customer_column].nunique(dropna=True)
                    growth = series.pct_change().replace([float("inf"), float("-inf")], pd.NA)
                    rows = pd.DataFrame({"period": series.index.astype(str), "new_customers": series.values, "growth_rate": growth.values})
                    result_payload["rows"] = _sanitize_numbers(rows.to_dict(orient="records"))
                    result_payload["value"] = float(series.iloc[-1]) if len(series) else None
                    result_payload["summary"] = {"bucket": bucket, "periods_used": int(len(series))}
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
                    ranked = grouped.sort_values(ascending=ascending).head(top_n)
                    result_payload["rows"] = ranked.reset_index().to_dict(orient="records")
                    result_payload["value"] = float(ranked.iloc[0]) if len(ranked) else None
                    result_payload["ranking_sort"] = "asc" if ascending else "desc"
                elif method == "mean":
                    temp = current[[entity_column, column]].copy()
                    temp[column] = pd.to_numeric(temp[column], errors="coerce")
                    temp = temp.dropna(subset=[entity_column, column])
                    if not temp.empty:
                        grouped = temp.groupby(entity_column).agg(
                            mean_value=(column, "mean"),
                            total_orders=(column, "count"),
                        ).reset_index()
                        support_threshold = _adaptive_support_threshold(grouped["total_orders"], floor=10, fraction=0.2)
                        global_mean = float(temp[column].mean()) if len(temp) else 0.0
                        prior_strength = 20.0
                        grouped["smoothed_mean"] = (
                            grouped["mean_value"] * grouped["total_orders"] + prior_strength * global_mean
                        ) / (grouped["total_orders"] + prior_strength)
                        filtered = grouped[grouped["total_orders"] >= support_threshold].copy()
                        if filtered.empty:
                            filtered = grouped[grouped["total_orders"] >= max(5, support_threshold // 2)].copy()
                        if filtered.empty:
                            filtered = grouped.copy()
                        ranked_df = filtered.sort_values(
                            ["smoothed_mean", "total_orders"],
                            ascending=[ascending, False],
                        ).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(ranked_df.rename(columns={"smoothed_mean": column}).to_dict(orient="records"))
                        result_payload["value"] = float(ranked_df.iloc[0]["smoothed_mean"]) if len(ranked_df) else None
                        result_payload["ranking_sort"] = "asc" if ascending else "desc"
                        result_payload["summary"] = {
                            "support_threshold": int(support_threshold),
                            "prior_strength": prior_strength,
                        }
                elif method == "count":
                    grouped = current.groupby(entity_column)[column].count()
                    ranked = grouped.sort_values(ascending=ascending).head(top_n)
                    result_payload["rows"] = ranked.reset_index().to_dict(orient="records")
                    result_payload["value"] = float(ranked.iloc[0]) if len(ranked) else None
                    result_payload["ranking_sort"] = "asc" if ascending else "desc"
                else:
                    grouped = current.groupby(entity_column)[column].sum()
                    ranked = grouped.sort_values(ascending=ascending).head(top_n)
                    result_payload["rows"] = ranked.reset_index().to_dict(orient="records")
                    result_payload["value"] = float(ranked.iloc[0]) if len(ranked) else None
                    result_payload["ranking_sort"] = "asc" if ascending else "desc"
        elif operation == "filtered_rank_entities" and column in current.columns:
            entity_column = params.get("entity_column")
            method = params.get("method", "sum")
            top_n = int(params.get("top_n", 10) or 10)
            ascending = str(params.get("sort", "desc")).lower() == "asc"
            filter_column = params.get("filter_column")
            filter_contains = str(params.get("filter_contains", "")).lower()
            if entity_column in current.columns and filter_column in current.columns:
                temp = current.copy()
                mask = temp[filter_column].astype(str).str.lower().str.contains(filter_contains, na=False)
                temp = temp[mask]
                if not temp.empty:
                    if method == "distinct_count":
                        affected = temp.groupby(entity_column)[column].nunique(dropna=True).reset_index(name="affected_orders")
                        baseline = current.groupby(entity_column)[column].nunique(dropna=True).reset_index(name="total_orders")
                        grouped = affected.merge(baseline, on=entity_column, how="left").fillna({"total_orders": 0})
                        support_threshold = _adaptive_support_threshold(grouped["total_orders"], floor=10, fraction=0.2)
                        total_affected_orders = float(temp[column].nunique(dropna=True))
                        total_orders_all = float(current[column].nunique(dropna=True))
                        global_rate = (total_affected_orders / total_orders_all) if total_orders_all else 0.0
                        prior_strength = 20.0
                        grouped["filtered_rate"] = grouped["affected_orders"] / grouped["total_orders"].replace(0, pd.NA)
                        grouped["smoothed_filtered_rate"] = (
                            grouped["affected_orders"] + prior_strength * global_rate
                        ) / (grouped["total_orders"] + prior_strength)
                        filtered = grouped[grouped["total_orders"] >= support_threshold].copy()
                        if filtered.empty:
                            filtered = grouped[grouped["total_orders"] >= max(5, support_threshold // 2)].copy()
                        if filtered.empty:
                            filtered = grouped.copy()
                        ranked_df = filtered.sort_values(
                            ["smoothed_filtered_rate", "affected_orders", "total_orders"],
                            ascending=[ascending, False, False],
                        ).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(ranked_df.to_dict(orient="records"))
                        result_payload["value"] = float(ranked_df.iloc[0]["smoothed_filtered_rate"]) if len(ranked_df) else None
                        result_payload["ranking_sort"] = "asc" if ascending else "desc"
                        result_payload["summary"] = {
                            "support_threshold": int(support_threshold),
                            "prior_strength": prior_strength,
                        }
                    elif method == "mean":
                        grouped = temp.groupby(entity_column)[column].mean()
                        ranked = grouped.sort_values(ascending=ascending).head(top_n)
                        result_payload["rows"] = ranked.reset_index().to_dict(orient="records")
                        result_payload["value"] = float(ranked.iloc[0]) if len(ranked) else None
                        result_payload["ranking_sort"] = "asc" if ascending else "desc"
                    elif method == "count":
                        grouped = temp.groupby(entity_column)[column].count()
                        ranked = grouped.sort_values(ascending=ascending).head(top_n)
                        result_payload["rows"] = ranked.reset_index().to_dict(orient="records")
                        result_payload["value"] = float(ranked.iloc[0]) if len(ranked) else None
                        result_payload["ranking_sort"] = "asc" if ascending else "desc"
                    else:
                        grouped = temp.groupby(entity_column)[column].sum()
                        ranked = grouped.sort_values(ascending=ascending).head(top_n)
                        result_payload["rows"] = ranked.reset_index().to_dict(orient="records")
                        result_payload["value"] = float(ranked.iloc[0]) if len(ranked) else None
                        result_payload["ranking_sort"] = "asc" if ascending else "desc"
                    result_payload["filter_applied"] = {filter_column: filter_contains}
        elif operation == "relative_burden_rank":
            entity_column = params.get("entity_column")
            numerator_column = params.get("numerator_column") or column
            denominator_column = params.get("denominator_column")
            top_n = int(params.get("top_n", 10) or 10)
            ascending = str(params.get("sort", "desc")).lower() == "asc"
            min_orders = int(params.get("min_orders", 0) or 0)
            if entity_column in current.columns and numerator_column in current.columns and denominator_column in current.columns:
                temp = current[[entity_column, numerator_column, denominator_column]].copy()
                temp[numerator_column] = pd.to_numeric(temp[numerator_column], errors="coerce")
                temp[denominator_column] = pd.to_numeric(temp[denominator_column], errors="coerce")
                temp = temp.dropna()
                temp = temp[temp[denominator_column] > 0]
                if not temp.empty:
                    temp["_burden_ratio"] = temp[numerator_column] / temp[denominator_column]
                    grouped = temp.groupby(entity_column).agg(
                        burden_ratio=("_burden_ratio", "mean"),
                        total_orders=("_burden_ratio", "count"),
                    ).reset_index()
                    if not min_orders:
                        min_orders = _adaptive_support_threshold(grouped["total_orders"], floor=25, fraction=0.25)
                    filtered = grouped[grouped["total_orders"] >= min_orders].copy()
                    if filtered.empty:
                        filtered = grouped[grouped["total_orders"] >= max(10, min_orders // 2)].copy()
                    if filtered.empty:
                        filtered = grouped.copy()
                    ranked_df = filtered.sort_values(["burden_ratio", "total_orders"], ascending=[ascending, False]).head(top_n)
                    ranked = ranked_df["burden_ratio"]
                    result_payload["rows"] = _sanitize_numbers(ranked_df.to_dict(orient="records"))
                    result_payload["value"] = float(ranked.iloc[0]) if len(ranked) else None
                    result_payload["ranking_sort"] = "asc" if ascending else "desc"
                    result_payload["summary"] = {"support_threshold": int(min_orders)}
        elif operation == "low_outcome_driver_analysis":
            outcome_column = params.get("outcome_column") or column
            candidate_columns = _unique_columns(list(params.get("candidate_columns") or []))
            start_column = params.get("start_column")
            end_column = params.get("end_column")
            top_n = int(params.get("top_n", 10) or 10)
            if outcome_column in current.columns:
                temp = current.copy()
                temp[outcome_column] = pd.to_numeric(temp[outcome_column], errors="coerce")
                temp = temp.dropna(subset=[outcome_column])
                if not temp.empty:
                    threshold = params.get("threshold")
                    threshold = float(threshold) if threshold is not None else float(temp[outcome_column].quantile(0.25))
                    low_mask = temp[outcome_column] <= threshold
                    low = temp.loc[low_mask].copy()
                    rows = []

                    if (
                        start_column in temp.columns
                        and end_column in temp.columns
                        and start_column not in candidate_columns
                        and end_column not in candidate_columns
                    ):
                        temp[start_column] = _coerce_datetime_like(temp[start_column])
                        temp[end_column] = _coerce_datetime_like(temp[end_column])
                        delay = (temp[end_column] - temp[start_column]).dt.total_seconds() / 86400.0
                        temp["_derived_delay_days"] = delay
                        low = temp.loc[low_mask].copy()
                        candidate_columns.append("_derived_delay_days")

                    for candidate in candidate_columns:
                        if candidate not in temp.columns or candidate == outcome_column:
                            continue
                        series_all = pd.to_numeric(temp[candidate], errors="coerce")
                        numeric_non_null = series_all.dropna()
                        if len(numeric_non_null) >= max(20, int(len(temp) * 0.01)):
                            series_low = pd.to_numeric(low[candidate], errors="coerce").dropna()
                            if len(series_low) < 5:
                                continue
                            baseline_mean = float(numeric_non_null.mean())
                            low_mean = float(series_low.mean())
                            std = float(numeric_non_null.std()) or 0.0
                            score = (low_mean - baseline_mean) / std if std else 0.0
                            rows.append({
                                "driver": candidate,
                                "driver_type": "numeric",
                                "column": candidate,
                                "low_outcome_mean": low_mean,
                                "baseline_mean": baseline_mean,
                                "score": float(score),
                                "low_count": int(len(series_low)),
                                "baseline_count": int(len(numeric_non_null)),
                            })
                            continue

                        categorical = temp[[candidate, outcome_column]].dropna(subset=[candidate]).copy()
                        if categorical.empty:
                            continue
                        categorical[candidate] = categorical[candidate].astype(str)
                        low_categorical = categorical.loc[categorical[outcome_column] <= threshold]
                        if low_categorical.empty:
                            continue
                        low_counts = low_categorical[candidate].value_counts()
                        all_counts = categorical[candidate].value_counts()
                        min_low_count = max(5, int(len(low_categorical) * 0.005))
                        for value, low_count in low_counts.head(50).items():
                            baseline_count = int(all_counts.get(value, 0))
                            if low_count < min_low_count or baseline_count <= 0:
                                continue
                            low_share = float(low_count / len(low_categorical))
                            baseline_share = float(baseline_count / len(categorical))
                            if baseline_share <= 0:
                                continue
                            lift = low_share / baseline_share
                            rows.append({
                                "driver": f"{candidate}={value}",
                                "driver_type": "categorical",
                                "column": candidate,
                                "category_value": value,
                                "low_outcome_share": low_share,
                                "baseline_share": baseline_share,
                                "score": float(lift - 1.0),
                                "lift": float(lift),
                                "low_count": int(low_count),
                                "baseline_count": int(baseline_count),
                            })

                    if rows:
                        driver_df = pd.DataFrame(rows)
                        driver_df["_abs_score"] = driver_df["score"].abs()
                        driver_df = driver_df.sort_values(["score", "low_count"], ascending=[False, False]).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(driver_df.drop(columns=["_abs_score"], errors="ignore").to_dict(orient="records"))
                        result_payload["value"] = float(driver_df.iloc[0]["score"])
                        result_payload["summary"] = {
                            "outcome_column": outcome_column,
                            "threshold": threshold,
                            "low_outcome_rows": int(low_mask.sum()),
                            "baseline_rows": int(len(temp)),
                        }
        elif operation == "pairwise_relationship":
            columns = [col for col in ([column] if column else []) + list(params.get("columns", []) or []) if col]
            comparison_column = params.get("comparison_column")
            if comparison_column and comparison_column not in columns:
                columns.append(comparison_column)
            if len(columns) < 2:
                fallback_columns = [col for col in step.get("columns", []) if col]
                for fallback in fallback_columns:
                    if fallback not in columns:
                        columns.append(fallback)
            if len(columns) >= 2 and all(col in current.columns for col in columns[:2]):
                left_col, right_col = columns[:2]
                method = str(params.get("method", "spearman")).lower()
                temp = current[[left_col, right_col]].copy()
                temp[left_col] = pd.to_numeric(temp[left_col], errors="coerce")
                temp[right_col] = pd.to_numeric(temp[right_col], errors="coerce")
                temp = temp.dropna(subset=[left_col, right_col])
                if not temp.empty and temp[left_col].nunique() >= 2 and temp[right_col].nunique() >= 2:
                    corr = temp[left_col].corr(temp[right_col], method=method)
                    result_payload["value"] = float(corr) if pd.notna(corr) else None
                    result_payload["summary"] = {
                        "method": method,
                        "x_column": left_col,
                        "y_column": right_col,
                        "correlation": float(corr) if pd.notna(corr) else None,
                        "sample_size": int(len(temp)),
                    }
        elif operation == "ratio_metric":
            numerator_column = params.get("numerator_column") or column
            denominator_column = params.get("denominator_column")
            numerator_method = params.get("numerator_method", "sum")
            denominator_method = params.get("denominator_method", "sum")
            as_percentage = bool(params.get("as_percentage"))
            if numerator_column in current.columns and denominator_column in current.columns:
                temp = current[[numerator_column, denominator_column]].copy()
                temp[numerator_column] = pd.to_numeric(temp[numerator_column], errors="coerce")
                temp[denominator_column] = pd.to_numeric(temp[denominator_column], errors="coerce")
                temp = temp.dropna()
                if not temp.empty:
                    numerator = float(temp[numerator_column].mean()) if numerator_method == "mean" else float(temp[numerator_column].sum())
                    denominator = float(temp[denominator_column].mean()) if denominator_method == "mean" else float(temp[denominator_column].sum())
                    ratio = (numerator / denominator) if denominator else None
                    result_payload["value"] = float(ratio * 100.0) if ratio is not None and as_percentage else ratio
                    result_payload["summary"] = {
                        "numerator_total": numerator,
                        "denominator_total": denominator,
                        "ratio": ratio,
                        "as_percentage": as_percentage,
                    }
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
        elif operation == "derived_delay_relationship":
            entity_column = params.get("entity_column")
            size_columns = [col for col in (params.get("size_columns") or []) if col in current.columns]
            start_column = params.get("start_column")
            end_column = params.get("end_column")
            if entity_column in current.columns and size_columns and start_column in current.columns and end_column in current.columns:
                temp = current[[entity_column, start_column, end_column] + size_columns].copy()
                temp[start_column] = _coerce_datetime_like(temp[start_column])
                temp[end_column] = _coerce_datetime_like(temp[end_column])
                temp = temp.dropna(subset=[entity_column, start_column, end_column])
                if not temp.empty:
                    temp["_delay_days"] = (temp[end_column] - temp[start_column]).dt.total_seconds() / 86400.0
                    temp = temp[temp["_delay_days"] >= 0]
                    for size_col in size_columns:
                        temp[size_col] = pd.to_numeric(temp[size_col], errors="coerce")
                    agg_map = {col: "mean" for col in size_columns}
                    agg_map["_delay_days"] = "mean"
                    entity_df = temp.groupby(entity_column).agg(agg_map).dropna()
                    rows = []
                    if not entity_df.empty:
                        for size_col in size_columns:
                            subset = entity_df[[size_col, "_delay_days"]].dropna()
                            if len(subset) < 3 or subset[size_col].nunique() < 2:
                                continue
                            corr = subset[size_col].corr(subset["_delay_days"], method="spearman")
                            if pd.notna(corr):
                                rows.append({"size_metric": size_col, "delay_correlation": float(corr)})
                        if rows:
                            rel_df = pd.DataFrame(rows)
                            rel_df["abs_correlation"] = rel_df["delay_correlation"].abs()
                            rel_df = rel_df.sort_values("abs_correlation", ascending=False)
                            result_payload["rows"] = _sanitize_numbers(rel_df.drop(columns=["abs_correlation"]).to_dict(orient="records"))
                            result_payload["value"] = float(rel_df.iloc[0]["delay_correlation"]) if len(rel_df) else None
                            result_payload["method"] = "spearman"
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
        elif operation == "top_dependency_share":
            entity_column = params.get("entity_column")
            value_column = params.get("value_column") or column
            method = params.get("method", "sum")
            top_n = int(params.get("top_n", 10) or 10)
            if entity_column in current.columns and value_column in current.columns:
                temp = current[[entity_column, value_column]].copy().dropna(subset=[entity_column, value_column])
                if not temp.empty:
                    if method == "distinct_count":
                        grouped = temp.groupby(entity_column)[value_column].nunique(dropna=True)
                    elif method == "mean":
                        temp[value_column] = pd.to_numeric(temp[value_column], errors="coerce")
                        grouped = temp.dropna(subset=[value_column]).groupby(entity_column)[value_column].mean()
                    else:
                        temp[value_column] = pd.to_numeric(temp[value_column], errors="coerce")
                        grouped = temp.dropna(subset=[value_column]).groupby(entity_column)[value_column].sum()
                    ranked = grouped.sort_values(ascending=False)
                    total = float(ranked.sum()) if len(ranked) else 0.0
                    top = ranked.head(top_n)
                    rows = [
                        {
                            entity_column: entity,
                            "contribution": float(value),
                            "share": float(value / total) if total else 0.0,
                            "rank": int(idx + 1),
                        }
                        for idx, (entity, value) in enumerate(top.items())
                    ]
                    top_share = float(top.sum() / total) if total else 0.0
                    hhi = float(((ranked / total) ** 2).sum()) if total else 0.0
                    result_payload["rows"] = _sanitize_numbers(rows)
                    result_payload["value"] = top_share
                    result_payload["summary"] = {"top_n": top_n, "top_share": top_share, "hhi": hhi, "entity_count": int(len(ranked)), "method": method}
        elif operation == "review_crisis_rank":
            entity_column = params.get("entity_column")
            review_column = params.get("review_column") or column
            order_column = params.get("order_column")
            top_n = int(params.get("top_n", 10) or 10)
            threshold = float(params.get("threshold", 2.0) or 2.0)
            if entity_column in current.columns and review_column in current.columns:
                needed = _unique_columns([entity_column, review_column, order_column])
                temp = current[needed].copy()
                temp[review_column] = pd.to_numeric(temp[review_column], errors="coerce")
                temp = temp.dropna(subset=[entity_column, review_column])
                if not temp.empty:
                    rows = []
                    for entity, frame in temp.groupby(entity_column):
                        support = int(frame[order_column].nunique(dropna=True)) if order_column in frame.columns else int(len(frame))
                        avg_review = float(frame[review_column].mean())
                        low_rate = float((frame[review_column] <= threshold).mean())
                        rows.append({entity_column: entity, "avg_review": avg_review, "low_review_rate": low_rate, "total_orders": support})
                    crisis_df = pd.DataFrame(rows)
                    if not crisis_df.empty:
                        min_orders = _adaptive_support_threshold(crisis_df["total_orders"], floor=25, fraction=0.25)
                        filtered = crisis_df[crisis_df["total_orders"] >= min_orders].copy()
                        if filtered.empty:
                            filtered = crisis_df[crisis_df["total_orders"] >= max(10, min_orders // 2)].copy()
                        if filtered.empty:
                            filtered = crisis_df.copy()
                        filtered["crisis_score"] = filtered["low_review_rate"].rank(pct=True) + (1.0 - filtered["avg_review"].rank(pct=True)) + filtered["total_orders"].rank(pct=True) * 0.25
                        filtered = filtered.sort_values(["crisis_score", "low_review_rate", "total_orders"], ascending=[False, False, False]).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(filtered.to_dict(orient="records"))
                        result_payload["value"] = float(filtered.iloc[0]["crisis_score"]) if len(filtered) else None
                        result_payload["summary"] = {"support_threshold": int(min_orders), "low_review_threshold": threshold}
        elif operation == "late_delivery_period_cluster":
            actual_column = params.get("actual_column")
            estimated_column = params.get("estimated_column")
            order_column = params.get("order_column") or column
            time_column = params.get("time_column")
            bucket = params.get("bucket", "month")
            if actual_column in current.columns and estimated_column in current.columns and order_column in current.columns and time_column in current.columns:
                temp = current[[actual_column, estimated_column, order_column, time_column]].copy()
                temp[actual_column] = _coerce_datetime_like(temp[actual_column])
                temp[estimated_column] = _coerce_datetime_like(temp[estimated_column])
                temp[time_column] = _coerce_datetime_like(temp[time_column])
                temp = temp.dropna()
                if not temp.empty:
                    temp = _assign_period_bucket(temp, time_column, bucket)
                    order_level = temp.groupby([order_column, "_period_bucket"]).agg(actual=(actual_column, "max"), estimated=(estimated_column, "max")).reset_index()
                    order_level["_late"] = ((order_level["actual"] - order_level["estimated"]).dt.total_seconds() > 0).astype(float)
                    period_df = order_level.groupby("_period_bucket").agg(late_rate=("_late", "mean"), late_orders=("_late", "sum"), total_orders=(order_column, "nunique")).reset_index()
                    min_orders = _adaptive_support_threshold(period_df["total_orders"], floor=25, fraction=0.25)
                    filtered = period_df[period_df["total_orders"] >= min_orders].copy()
                    if filtered.empty:
                        filtered = period_df.copy()
                    global_rate = float(order_level["_late"].mean()) if len(order_level) else 0.0
                    filtered["cluster_score"] = filtered["late_rate"] - global_rate
                    filtered = filtered.sort_values(["cluster_score", "late_orders"], ascending=[False, False])
                    result_payload["rows"] = _sanitize_numbers(filtered.rename(columns={"_period_bucket": "period"}).to_dict(orient="records"))
                    result_payload["value"] = float(filtered.iloc[0]["late_rate"]) if len(filtered) else None
                    result_payload["summary"] = {"global_late_rate": global_rate, "support_threshold": int(min_orders), "bucket": bucket}
        elif operation == "aggregate_forecast":
            time_column = params.get("time_column")
            metric_column = params.get("metric_column") or column
            segment_column = params.get("segment_column")
            method = params.get("method", "sum")
            bucket = params.get("bucket", "month")
            top_n = int(params.get("top_n", 10) or 10)
            if time_column in current.columns and metric_column in current.columns:
                needed = _unique_columns([time_column, metric_column, segment_column])
                temp = current[needed].copy()
                temp[time_column] = _coerce_datetime_like(temp[time_column])
                temp = temp.dropna(subset=[time_column])
                if not temp.empty:
                    temp = _assign_period_bucket(temp, time_column, bucket)
                    if method == "distinct_count":
                        grouped = temp.groupby(([segment_column] if segment_column in temp.columns else []) + ["_period_bucket"])[metric_column].nunique(dropna=True)
                    elif method == "mean":
                        temp[metric_column] = pd.to_numeric(temp[metric_column], errors="coerce")
                        grouped = temp.dropna(subset=[metric_column]).groupby(([segment_column] if segment_column in temp.columns else []) + ["_period_bucket"])[metric_column].mean()
                    else:
                        temp[metric_column] = pd.to_numeric(temp[metric_column], errors="coerce")
                        grouped = temp.dropna(subset=[metric_column]).groupby(([segment_column] if segment_column in temp.columns else []) + ["_period_bucket"])[metric_column].sum()
                    rows = []
                    groups = grouped.groupby(level=0) if segment_column in temp.columns else [(None, grouped)]
                    for segment, series in groups:
                        values = series.droplevel(0).sort_index() if segment_column in temp.columns else series.sort_index()
                        if values.empty:
                            continue
                        values = pd.to_numeric(values, errors="coerce").dropna()
                        if values.empty:
                            continue
                        recent = values.tail(min(3, len(values)))
                        baseline = float(recent.mean())
                        trend = float((recent.iloc[-1] - recent.iloc[0]) / max(1, len(recent) - 1)) if len(recent) > 1 else 0.0
                        trend_forecast = baseline + trend
                        seasonal_forecast = None
                        if len(values) >= 13:
                            seasonal_forecast = float(values.iloc[-12])
                        if seasonal_forecast is not None:
                            forecast_value = max(0.0, 0.55 * trend_forecast + 0.30 * baseline + 0.15 * seasonal_forecast)
                            model = "trend_recent_seasonal_ensemble"
                        else:
                            forecast_value = max(0.0, 0.70 * trend_forecast + 0.30 * baseline)
                            model = "trend_recent_ensemble"
                        backtest_errors = []
                        backtest_count = min(3, max(0, len(values) - 3))
                        for offset in range(backtest_count, 0, -1):
                            train = values.iloc[:-offset]
                            actual = float(values.iloc[-offset])
                            if len(train) < 2:
                                continue
                            train_recent = train.tail(min(3, len(train)))
                            train_baseline = float(train_recent.mean())
                            train_trend = float((train_recent.iloc[-1] - train_recent.iloc[0]) / max(1, len(train_recent) - 1)) if len(train_recent) > 1 else 0.0
                            predicted = max(0.0, train_baseline + train_trend)
                            backtest_errors.append(abs(actual - predicted))
                        mae = float(sum(backtest_errors) / len(backtest_errors)) if backtest_errors else None
                        mape = None
                        if mae is not None and float(values.tail(min(3, len(values))).mean()) != 0:
                            mape = float(mae / max(abs(float(values.tail(min(3, len(values))).mean())), 1e-9))
                        residual_buffer = mae if mae is not None else max(abs(forecast_value) * 0.20, 1.0)
                        reliability = "high" if len(values) >= 12 and (mape is None or mape <= 0.25) else "medium" if len(values) >= 6 else "low"
                        row = {
                            "forecast_period": "next_period",
                            "forecast_value": forecast_value,
                            "forecast_low": max(0.0, forecast_value - 1.28 * residual_buffer),
                            "forecast_high": forecast_value + 1.28 * residual_buffer,
                            "recent_average": baseline,
                            "trend_adjustment": trend,
                            "seasonal_reference": seasonal_forecast,
                            "backtest_mae": mae,
                            "backtest_mape": mape,
                            "forecast_model": model,
                            "forecast_reliability": reliability,
                            "periods_used": int(len(values)),
                        }
                        if segment_column in temp.columns:
                            row[segment_column] = segment
                        rows.append(row)
                    forecast_df = pd.DataFrame(rows)
                    if not forecast_df.empty:
                        forecast_df = forecast_df.sort_values("forecast_value", ascending=False).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(forecast_df.to_dict(orient="records"))
                        result_payload["value"] = float(forecast_df.iloc[0]["forecast_value"])
                        result_payload["summary"] = {
                            "bucket": bucket,
                            "method": method,
                            "segment_column": segment_column,
                            "forecast_method": "backtested_recent_trend_with_optional_seasonality",
                            "interval": "approximate_80_percent_prediction_interval",
                        }
        elif operation == "capacity_need_score":
            entity_column = params.get("entity_column")
            time_column = params.get("time_column")
            demand_column = params.get("demand_column") or column
            bucket = params.get("bucket", "week")
            top_n = int(params.get("top_n", 10) or 10)
            if entity_column in current.columns and time_column in current.columns and demand_column in current.columns:
                temp = current[_unique_columns([entity_column, time_column, demand_column])].copy()
                temp[time_column] = _coerce_datetime_like(temp[time_column])
                temp = temp.dropna(subset=[entity_column, time_column])
                if not temp.empty:
                    temp = _assign_period_bucket(temp, time_column, bucket)
                    grouped = temp.groupby([entity_column, "_period_bucket"])[demand_column].nunique(dropna=True)
                    rows = []
                    for entity, series in grouped.groupby(level=0):
                        values = series.droplevel(0).sort_index()
                        if values.empty:
                            continue
                        peak = float(values.max())
                        recent = float(values.tail(min(4, len(values))).mean())
                        capacity_need = max(peak, recent)
                        rows.append({entity_column: entity, "capacity_need": capacity_need, "peak_period_demand": peak, "recent_average_demand": recent, "periods": int(len(values))})
                    cap_df = pd.DataFrame(rows)
                    if not cap_df.empty:
                        cap_df = cap_df.sort_values(["capacity_need", "peak_period_demand"], ascending=[False, False]).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(cap_df.to_dict(orient="records"))
                        result_payload["value"] = float(cap_df.iloc[0]["capacity_need"])
                        result_payload["summary"] = {"bucket": bucket}
        elif operation == "strategic_opportunity_score":
            entity_column = params.get("entity_column")
            order_column = params.get("order_column")
            value_column = params.get("value_column")
            review_column = params.get("review_column")
            freight_column = params.get("freight_column")
            mode = params.get("mode", "growth")
            top_n = int(params.get("top_n", 10) or 10)
            if entity_column in current.columns and order_column in current.columns:
                needed = _unique_columns([entity_column, order_column, value_column, review_column, freight_column])
                temp = current[needed].copy().dropna(subset=[entity_column, order_column])
                for metric in [value_column, review_column, freight_column]:
                    if metric in temp.columns:
                        temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
                rows = []
                for entity, frame in temp.groupby(entity_column):
                    row = {entity_column: entity, "orders": float(frame[order_column].nunique(dropna=True))}
                    if value_column in frame.columns:
                        row["revenue"] = float(frame[value_column].sum())
                        row["avg_value"] = float(frame[value_column].mean())
                    if review_column in frame.columns:
                        row["avg_review"] = float(frame[review_column].mean())
                    if freight_column in frame.columns:
                        row["avg_freight"] = float(frame[freight_column].mean())
                    rows.append(row)
                score_df = pd.DataFrame(rows)
                if not score_df.empty:
                    support_threshold = _adaptive_support_threshold(score_df["orders"], floor=10, fraction=0.2)
                    filtered = score_df[score_df["orders"] >= support_threshold].copy()
                    if filtered.empty:
                        filtered = score_df.copy()
                    demand = filtered["orders"].rank(pct=True).fillna(0.0)
                    revenue = filtered.get("revenue", filtered["orders"]).rank(pct=True).fillna(0.0)
                    avg_value = filtered.get("avg_value", filtered["orders"]).rank(pct=True).fillna(0.0)
                    review = filtered.get("avg_review", pd.Series(0.5, index=filtered.index)).rank(pct=True).fillna(0.5)
                    freight_penalty = filtered.get("avg_freight", pd.Series(0.0, index=filtered.index)).rank(pct=True).fillna(0.0)
                    filtered["demand_score"] = demand
                    filtered["revenue_score"] = revenue
                    filtered["value_score"] = avg_value
                    filtered["review_score_component"] = review
                    filtered["freight_penalty"] = freight_penalty
                    if mode == "drop":
                        filtered["strategic_score"] = (1.0 - demand) * 0.35 + (1.0 - revenue) * 0.35 + (1.0 - review) * 0.2 + freight_penalty * 0.1
                        weights = {"weak_demand": 0.35, "weak_revenue": 0.35, "weak_review": 0.2, "freight_penalty": 0.1}
                    elif mode == "premium":
                        filtered["strategic_score"] = avg_value * 0.5 + review * 0.3 + demand * 0.2
                        weights = {"value": 0.5, "review": 0.3, "demand": 0.2}
                    elif mode == "recruit":
                        filtered["strategic_score"] = demand * 0.45 + revenue * 0.35 + (1.0 - freight_penalty) * 0.2
                        weights = {"demand": 0.45, "revenue": 0.35, "low_freight": 0.2}
                    else:
                        filtered["strategic_score"] = demand * 0.35 + revenue * 0.35 + review * 0.2 + (1.0 - freight_penalty) * 0.1
                        weights = {"demand": 0.35, "revenue": 0.35, "review": 0.2, "low_freight": 0.1}
                    filtered["evidence_grade"] = filtered["orders"].apply(
                        lambda count: "high" if count >= support_threshold * 3 else "medium" if count >= support_threshold else "low"
                    )
                    component_columns = ["demand_score", "revenue_score", "value_score", "review_score_component", "freight_penalty"]
                    filtered["dominant_signal"] = filtered[component_columns].idxmax(axis=1)
                    filtered = filtered.sort_values(["strategic_score", "orders"], ascending=[False, False]).head(top_n)
                    result_payload["rows"] = _sanitize_numbers(filtered.to_dict(orient="records"))
                    result_payload["value"] = float(filtered.iloc[0]["strategic_score"])
                    result_payload["summary"] = {
                        "mode": mode,
                        "support_threshold": int(support_threshold),
                        "weights": weights,
                        "scoring": "Rank-normalized composite score with explicit demand, value/revenue, review, and freight components.",
                    }
        elif operation == "predictive_target_profile":
            target_type = params.get("target_type", "generic")
            customer_column = params.get("customer_column")
            order_column = params.get("order_column")
            status_column = params.get("status_column")
            review_column = params.get("review_column")
            time_column = params.get("time_column")
            if order_column in current.columns:
                rows = []
                target = pd.Series(dtype=float)
                if target_type == "cancellation" and status_column in current.columns:
                    statuses = current[status_column].astype(str).str.lower()
                    target = statuses.str.contains("cancel", na=False).astype(float)
                    rows = [{"target": "cancellation", "positive_rate": float(target.mean()), "positive_count": int(target.sum()), "total_records": int(len(target))}]
                elif target_type == "bad_review" and review_column in current.columns:
                    review = pd.to_numeric(current[review_column], errors="coerce")
                    target = (review <= 2).astype(float)
                    rows = [{"target": "bad_review", "positive_rate": float(target.mean()), "positive_count": int(target.sum()), "total_records": int(review.notna().sum())}]
                elif target_type in {"repeat_customer", "churn"} and customer_column in current.columns:
                    counts = current.groupby(customer_column)[order_column].nunique(dropna=True)
                    if target_type == "repeat_customer":
                        target = (counts > 1).astype(float)
                    else:
                        target = (counts <= 1).astype(float)
                    rows = [{"target": target_type, "positive_rate": float(target.mean()), "positive_count": int(target.sum()), "total_records": int(len(target))}]
                if rows:
                    positive_count = int(rows[0]["positive_count"])
                    total_records = int(rows[0]["total_records"])
                    negative_count = max(0, total_records - positive_count)
                    minority_count = min(positive_count, negative_count)
                    positive_rate = float(rows[0]["positive_rate"])
                    if total_records >= 1000 and minority_count >= 100:
                        readiness = "model_ready"
                    elif total_records >= 200 and minority_count >= 20:
                        readiness = "model_possible_with_validation"
                    else:
                        readiness = "target_constructed_but_sparse"
                    baseline_accuracy = max(positive_rate, 1.0 - positive_rate) if total_records else None
                    rows[0].update({
                        "negative_count": negative_count,
                        "minority_class_count": minority_count,
                        "baseline_accuracy": baseline_accuracy,
                        "class_balance": float(min(positive_rate, 1.0 - positive_rate)) if total_records else None,
                        "recommended_model_family": "regularized_tree_or_logistic_classifier" if target_type in {"cancellation", "bad_review"} else "customer_level_classification",
                        "readiness": readiness,
                    })
                    result_payload["rows"] = _sanitize_numbers(rows)
                    result_payload["value"] = float(rows[0]["positive_rate"])
                    result_payload["summary"] = {
                        "target_type": target_type,
                        "predictive_readiness": readiness,
                        "time_column": time_column,
                        "minimum_recommended_positive_cases": 100,
                    }
        elif operation == "customer_ltv_estimate":
            customer_column = params.get("customer_column")
            order_column = params.get("order_column")
            value_column = params.get("value_column") or column
            top_n = int(params.get("top_n", 10) or 10)
            if customer_column in current.columns and value_column in current.columns:
                temp = current[_unique_columns([customer_column, order_column, value_column])].copy()
                temp[value_column] = pd.to_numeric(temp[value_column], errors="coerce")
                temp = temp.dropna(subset=[customer_column, value_column])
                if not temp.empty:
                    grouped = temp.groupby(customer_column).agg(lifetime_value=(value_column, "sum"), orders=(order_column, "nunique") if order_column in temp.columns else (value_column, "count")).reset_index()
                    grouped["value_per_order"] = grouped["lifetime_value"] / grouped["orders"].replace(0, pd.NA)
                    top = grouped.sort_values("lifetime_value", ascending=False).head(top_n)
                    result_payload["rows"] = _sanitize_numbers(top.to_dict(orient="records"))
                    result_payload["value"] = float(grouped["lifetime_value"].mean())
                    result_payload["summary"] = {
                        "average_lifetime_value": float(grouped["lifetime_value"].mean()),
                        "median_lifetime_value": float(grouped["lifetime_value"].median()),
                        "p75_lifetime_value": float(grouped["lifetime_value"].quantile(0.75)),
                        "p90_lifetime_value": float(grouped["lifetime_value"].quantile(0.90)),
                        "repeat_customer_rate": float((grouped["orders"] > 1).mean()),
                        "average_orders_per_customer": float(grouped["orders"].mean()),
                        "customers": int(len(grouped)),
                        "method": "observed_customer_value_rollup",
                    }
        elif operation == "customer_clustering_segments":
            customer_column = params.get("customer_column")
            order_column = params.get("order_column")
            value_column = params.get("value_column")
            time_column = params.get("time_column")
            if customer_column in current.columns and order_column in current.columns:
                needed = _unique_columns([customer_column, order_column, value_column, time_column])
                temp = current[needed].copy()
                if value_column in temp.columns:
                    temp[value_column] = pd.to_numeric(temp[value_column], errors="coerce")
                rows = []
                grouped = temp.groupby(customer_column)
                latest = _coerce_datetime_like(temp[time_column]).max() if time_column in temp.columns else None
                customer_rows = []
                for customer, frame in grouped:
                    row = {"orders": float(frame[order_column].nunique(dropna=True))}
                    row["value"] = float(frame[value_column].sum()) if value_column in frame.columns else row["orders"]
                    if time_column in frame.columns and pd.notna(latest):
                        dates = _coerce_datetime_like(frame[time_column])
                        row["recency_days"] = float((latest - dates.max()).days) if pd.notna(dates.max()) else None
                    customer_rows.append(row)
                cdf = pd.DataFrame(customer_rows).dropna()
                if not cdf.empty:
                    cdf["frequency_band"] = pd.qcut(cdf["orders"].rank(method="first"), q=min(3, len(cdf)), labels=False, duplicates="drop")
                    cdf["value_band"] = pd.qcut(cdf["value"].rank(method="first"), q=min(3, len(cdf)), labels=False, duplicates="drop")
                    if "recency_days" in cdf.columns:
                        cdf["recency_band"] = pd.qcut((-cdf["recency_days"]).rank(method="first"), q=min(3, len(cdf)), labels=False, duplicates="drop")
                    else:
                        cdf["recency_band"] = 1
                    cdf["segment_id"] = cdf[["frequency_band", "value_band", "recency_band"]].astype(str).agg("-".join, axis=1)
                    seg = cdf.groupby("segment_id").agg(
                        customers=("orders", "count"),
                        avg_orders=("orders", "mean"),
                        avg_value=("value", "mean"),
                        avg_recency_days=("recency_days", "mean") if "recency_days" in cdf.columns else ("orders", "mean"),
                    ).reset_index()
                    seg["segment_score"] = seg["avg_orders"].rank(pct=True) * 0.4 + seg["avg_value"].rank(pct=True) * 0.6
                    seg["segment_label"] = seg.apply(
                        lambda row: "high_value_loyal" if row["avg_value"] >= cdf["value"].quantile(0.75) and row["avg_orders"] >= cdf["orders"].quantile(0.75)
                        else "high_value_infrequent" if row["avg_value"] >= cdf["value"].quantile(0.75)
                        else "frequent_lower_value" if row["avg_orders"] >= cdf["orders"].quantile(0.75)
                        else "emerging_or_low_value",
                        axis=1,
                    )
                    seg = seg.sort_values(["segment_score", "customers"], ascending=[False, False])
                    result_payload["rows"] = _sanitize_numbers(seg.head(10).to_dict(orient="records"))
                    result_payload["value"] = float(len(seg))
                    result_payload["summary"] = {
                        "segments": int(len(seg)),
                        "customers": int(len(cdf)),
                        "method": "rfm_quantile_segments",
                        "features": ["recency_days", "orders", "lifetime_value"],
                        "note": "Deterministic quantile segmentation; production clustering can replace labels with k-means or hierarchical clusters.",
                    }
        elif operation == "delivery_promise_optimization":
            entity_column = params.get("entity_column")
            start_column = params.get("start_column")
            end_column = params.get("end_column")
            estimated_column = params.get("estimated_column")
            order_column = params.get("order_column")
            top_n = int(params.get("top_n", 10) or 10)
            if entity_column in current.columns and start_column in current.columns and end_column in current.columns:
                temp = current[_unique_columns([entity_column, order_column, start_column, end_column, estimated_column])].copy()
                temp[start_column] = _coerce_datetime_like(temp[start_column])
                temp[end_column] = _coerce_datetime_like(temp[end_column])
                temp = temp.dropna(subset=[entity_column, start_column, end_column])
                if not temp.empty:
                    temp["_delivery_days"] = (temp[end_column] - temp[start_column]).dt.total_seconds() / 86400.0
                    temp = temp[temp["_delivery_days"] >= 0]
                    rows = []
                    for entity, frame in temp.groupby(entity_column):
                        support = int(frame[order_column].nunique(dropna=True)) if order_column in frame.columns else int(len(frame))
                        if support < 5:
                            continue
                        promised_days = float(frame["_delivery_days"].quantile(0.9))
                        p95_days = float(frame["_delivery_days"].quantile(0.95))
                        median_days = float(frame["_delivery_days"].median())
                        actual_late_rate = None
                        current_promise_days = None
                        if estimated_column in frame.columns:
                            estimated = _coerce_datetime_like(frame[estimated_column])
                            valid = pd.DataFrame({"estimated": estimated, "end": frame[end_column]}).dropna()
                            if not valid.empty:
                                actual_late_rate = float((valid["end"] > valid["estimated"]).mean())
                                current_promise_days = float((valid["estimated"] - frame.loc[valid.index, start_column]).dt.total_seconds().median() / 86400.0)
                        rows.append({
                            entity_column: entity,
                            "recommended_promise_days": promised_days,
                            "p95_delivery_days": p95_days,
                            "median_delivery_days": median_days,
                            "promise_buffer_days": promised_days - median_days,
                            "current_late_rate": actual_late_rate,
                            "current_median_promise_days": current_promise_days,
                            "orders": support,
                        })
                    pdf = pd.DataFrame(rows)
                    if not pdf.empty:
                        pdf["promise_optimization_score"] = pdf["orders"].rank(pct=True) + pdf["recommended_promise_days"].rank(pct=True) * 0.25
                        pdf = pdf.sort_values(["promise_optimization_score", "orders"], ascending=[False, False]).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(pdf.to_dict(orient="records"))
                        result_payload["value"] = float(pdf.iloc[0]["recommended_promise_days"])
                        result_payload["summary"] = {
                            "method": "p90_delivery_duration",
                            "entity_column": entity_column,
                            "promise_policy": "Set promise near the 90th percentile and monitor late-rate movement.",
                        }
        elif operation == "logistics_underperformance_score":
            entity_column = params.get("entity_column")
            order_column = params.get("order_column")
            review_column = params.get("review_column")
            start_column = params.get("start_column")
            end_column = params.get("end_column")
            freight_column = params.get("freight_column")
            top_n = int(params.get("top_n", 10) or 10)
            if entity_column in current.columns and order_column in current.columns:
                needed = _unique_columns([entity_column, order_column, review_column, start_column, end_column, freight_column])
                temp = current[needed].copy()
                if start_column in temp.columns and end_column in temp.columns:
                    temp[start_column] = _coerce_datetime_like(temp[start_column])
                    temp[end_column] = _coerce_datetime_like(temp[end_column])
                    temp["_delay_days"] = (temp[end_column] - temp[start_column]).dt.total_seconds() / 86400.0
                    temp.loc[temp["_delay_days"] < 0, "_delay_days"] = pd.NA
                if review_column in temp.columns:
                    temp[review_column] = pd.to_numeric(temp[review_column], errors="coerce")
                if freight_column in temp.columns:
                    temp[freight_column] = pd.to_numeric(temp[freight_column], errors="coerce")
                rows = []
                for entity, frame in temp.groupby(entity_column):
                    orders = int(frame[order_column].nunique(dropna=True))
                    rows.append({
                        entity_column: entity,
                        "total_orders": orders,
                        "avg_delay_days": float(frame["_delay_days"].mean()) if "_delay_days" in frame.columns and frame["_delay_days"].notna().any() else None,
                        "avg_review": float(frame[review_column].mean()) if review_column in frame.columns and frame[review_column].notna().any() else None,
                        "avg_freight": float(frame[freight_column].mean()) if freight_column in frame.columns and frame[freight_column].notna().any() else None,
                    })
                risk_df = pd.DataFrame(rows)
                if not risk_df.empty:
                    min_orders = _adaptive_support_threshold(risk_df["total_orders"], floor=25, fraction=0.25)
                    filtered = risk_df[risk_df["total_orders"] >= min_orders].copy()
                    if filtered.empty:
                        filtered = risk_df.copy()
                    delay_risk = filtered["avg_delay_days"].rank(pct=True).fillna(0.5) if "avg_delay_days" in filtered else 0.5
                    review_risk = 1.0 - filtered["avg_review"].rank(pct=True).fillna(0.5) if "avg_review" in filtered else 0.5
                    freight_risk = filtered["avg_freight"].rank(pct=True).fillna(0.5) if "avg_freight" in filtered else 0.5
                    support = filtered["total_orders"].rank(pct=True).fillna(0.0)
                    filtered["underperformance_score"] = 0.4 * delay_risk + 0.35 * review_risk + 0.15 * freight_risk + 0.10 * support
                    filtered = filtered.sort_values(["underperformance_score", "total_orders"], ascending=[False, False]).head(top_n)
                    result_payload["rows"] = _sanitize_numbers(filtered.to_dict(orient="records"))
                    result_payload["value"] = float(filtered.iloc[0]["underperformance_score"]) if len(filtered) else None
                    result_payload["summary"] = {"support_threshold": int(min_orders)}
        elif operation == "entity_intervention_score":
            entity_column = params.get("entity_column")
            count_column = params.get("count_column")
            revenue_metric = params.get("revenue_metric")
            review_metric = params.get("review_metric")
            status_column = params.get("status_column")
            delivery_start = params.get("delivery_start_column")
            delivery_end = params.get("delivery_end_column")
            top_n = int(params.get("top_n", 10) or 10)
            if entity_column in current.columns and count_column in current.columns:
                temp = current.copy()
                grouped = temp.groupby(entity_column)
                rows = []
                for entity, frame in grouped:
                    orders = float(frame[count_column].nunique(dropna=True))
                    revenue = float(frame[revenue_metric].sum()) if revenue_metric in frame.columns else 0.0
                    review_mean = float(pd.to_numeric(frame[review_metric], errors="coerce").mean()) if review_metric in frame.columns else None
                    cancel_rate = None
                    if status_column in frame.columns:
                        statuses = frame[status_column].astype(str).str.lower()
                        cancel_rate = float(statuses.str.contains("cancel", na=False).mean())
                    delivery_days = None
                    if delivery_start in frame.columns and delivery_end in frame.columns:
                        start = _coerce_datetime_like(frame[delivery_start])
                        end = _coerce_datetime_like(frame[delivery_end])
                        valid = pd.DataFrame({"start": start, "end": end}).dropna()
                        if not valid.empty:
                            deltas = (valid["end"] - valid["start"]).dt.total_seconds() / 86400.0
                            deltas = deltas[deltas >= 0]
                            if not deltas.empty:
                                delivery_days = float(deltas.mean())
                    rows.append({
                        entity_column: entity,
                        "orders": orders,
                        "revenue": revenue,
                        "review_score": review_mean,
                        "cancel_rate": cancel_rate,
                        "delivery_days": delivery_days,
                    })
                intervention_df = pd.DataFrame(rows)
                if not intervention_df.empty:
                    revenue_rank = intervention_df["revenue"].rank(pct=True).fillna(0.0)
                    order_rank = intervention_df["orders"].rank(pct=True).fillna(0.0)
                    review_risk = 1.0 - intervention_df["review_score"].rank(pct=True).fillna(0.5) if "review_score" in intervention_df else 0.0
                    delivery_risk = intervention_df["delivery_days"].rank(pct=True).fillna(0.5) if "delivery_days" in intervention_df else 0.0
                    cancel_risk = intervention_df["cancel_rate"].rank(pct=True).fillna(0.0) if "cancel_rate" in intervention_df else 0.0
                    intervention_df["intervention_score"] = (
                        0.3 * revenue_rank +
                        0.2 * order_rank +
                        0.25 * review_risk +
                        0.15 * delivery_risk +
                        0.10 * cancel_risk
                    )
                    intervention_df = intervention_df.sort_values("intervention_score", ascending=False).head(top_n)
                    result_payload["rows"] = _sanitize_numbers(intervention_df.to_dict(orient="records"))
                    result_payload["value"] = float(intervention_df.iloc[0]["intervention_score"]) if len(intervention_df) else None
        elif operation == "premium_potential_score":
            entity_column = params.get("entity_column")
            value_metric = params.get("value_metric") or column
            review_metric = params.get("review_metric")
            count_column = params.get("count_column")
            top_n = int(params.get("top_n", 10) or 10)
            if entity_column in current.columns and value_metric in current.columns:
                temp = current.copy()
                rows = []
                for entity, frame in temp.groupby(entity_column):
                    value_score = float(pd.to_numeric(frame[value_metric], errors="coerce").mean())
                    review_score = float(pd.to_numeric(frame[review_metric], errors="coerce").mean()) if review_metric in frame.columns else None
                    order_support = float(frame[count_column].nunique(dropna=True)) if count_column in frame.columns else 0.0
                    rows.append({
                        entity_column: entity,
                        "avg_value": value_score,
                        "review_score": review_score,
                        "order_support": order_support,
                    })
                premium_df = pd.DataFrame(rows).dropna(subset=["avg_value"])
                if not premium_df.empty:
                    support_threshold = _adaptive_support_threshold(premium_df["order_support"], floor=10, fraction=0.2)
                    filtered = premium_df[premium_df["order_support"] >= support_threshold].copy()
                    if filtered.empty:
                        filtered = premium_df[premium_df["order_support"] >= max(5, support_threshold // 2)].copy()
                    if filtered.empty:
                        filtered = premium_df.copy()
                    premium_df = filtered
                    premium_df["premium_score"] = (
                        0.5 * premium_df["avg_value"].rank(pct=True).fillna(0.0) +
                        0.3 * premium_df["review_score"].rank(pct=True).fillna(0.5) +
                        0.2 * premium_df["order_support"].rank(pct=True).fillna(0.0)
                    )
                    premium_df = premium_df.sort_values("premium_score", ascending=False).head(top_n)
                    result_payload["rows"] = _sanitize_numbers(premium_df.to_dict(orient="records"))
                    result_payload["value"] = float(premium_df.iloc[0]["premium_score"]) if len(premium_df) else None
                    result_payload["summary"] = {"support_threshold": int(support_threshold)}
        elif operation == "elasticity_proxy_score":
            entity_column = params.get("entity_column")
            price_column = params.get("price_column") or column
            demand_column = params.get("demand_column")
            time_column = params.get("time_column")
            bucket = params.get("bucket", "month")
            top_n = int(params.get("top_n", 10) or 10)
            if entity_column in current.columns and price_column in current.columns and demand_column in current.columns and time_column in current.columns:
                temp = current[[entity_column, price_column, demand_column, time_column]].copy()
                temp[price_column] = pd.to_numeric(temp[price_column], errors="coerce")
                temp[time_column] = _coerce_datetime_like(temp[time_column])
                temp = temp.dropna(subset=[entity_column, price_column, demand_column, time_column])
                if not temp.empty and pd.api.types.is_datetime64_any_dtype(temp[time_column]):
                    if bucket == "quarter":
                        temp["_period_bucket"] = temp[time_column].dt.to_period("Q").astype(str)
                    elif bucket == "year":
                        temp["_period_bucket"] = temp[time_column].dt.to_period("Y").astype(str)
                    else:
                        temp["_period_bucket"] = temp[time_column].dt.to_period("M").astype(str)
                    grouped = temp.groupby([entity_column, "_period_bucket"]).agg(
                        avg_price=(price_column, "mean"),
                        demand=(demand_column, "nunique"),
                    ).reset_index()
                    rows = []
                    for entity, frame in grouped.groupby(entity_column):
                        if len(frame) < 3 or frame["avg_price"].nunique() < 2 or frame["demand"].nunique() < 2:
                            continue
                        corr = frame["avg_price"].corr(frame["demand"], method="spearman")
                        if pd.notna(corr):
                            rows.append({
                                entity_column: entity,
                                "elasticity_signal": float(-corr),
                                "price_demand_correlation": float(corr),
                                "periods": int(len(frame)),
                            })
                    elasticity_df = pd.DataFrame(rows)
                    if not elasticity_df.empty:
                        elasticity_df = elasticity_df.sort_values("elasticity_signal", ascending=False).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(elasticity_df.to_dict(orient="records"))
                        result_payload["value"] = float(elasticity_df.iloc[0]["elasticity_signal"]) if len(elasticity_df) else None
                        result_payload["method"] = "spearman_time_proxy"
        elif operation == "discount_volume_effect":
            price_column = params.get("price_column") or column
            demand_column = params.get("demand_column")
            time_column = params.get("time_column")
            bucket = params.get("bucket", "month")
            if price_column in current.columns and demand_column in current.columns and time_column in current.columns:
                temp = current[[price_column, demand_column, time_column]].copy()
                temp[price_column] = pd.to_numeric(temp[price_column], errors="coerce")
                temp[time_column] = _coerce_datetime_like(temp[time_column])
                temp = temp.dropna(subset=[price_column, demand_column, time_column])
                if not temp.empty and pd.api.types.is_datetime64_any_dtype(temp[time_column]):
                    if bucket == "quarter":
                        temp["_period_bucket"] = temp[time_column].dt.to_period("Q").astype(str)
                    elif bucket == "year":
                        temp["_period_bucket"] = temp[time_column].dt.to_period("Y").astype(str)
                    else:
                        temp["_period_bucket"] = temp[time_column].dt.to_period("M").astype(str)
                    grouped = temp.groupby("_period_bucket").agg(avg_price=(price_column, "mean"), demand=(demand_column, "nunique")).reset_index()
                    if len(grouped) >= 3:
                        baseline = float(grouped["avg_price"].median())
                        grouped["discount_like"] = grouped["avg_price"] < baseline
                        low_price_demand = float(grouped.loc[grouped["discount_like"], "demand"].mean()) if grouped["discount_like"].any() else None
                        normal_demand = float(grouped.loc[~grouped["discount_like"], "demand"].mean()) if (~grouped["discount_like"]).any() else None
                        corr = grouped["avg_price"].corr(grouped["demand"], method="spearman")
                        result_payload["rows"] = _sanitize_numbers(grouped.to_dict(orient="records"))
                        result_payload["summary"] = {
                            "baseline_price": baseline,
                            "discount_like_period_demand": low_price_demand,
                            "non_discount_period_demand": normal_demand,
                            "price_demand_correlation": float(corr) if pd.notna(corr) else None,
                        }
                        if low_price_demand is not None and normal_demand and normal_demand != 0:
                            result_payload["value"] = float((low_price_demand - normal_demand) / normal_demand)
        elif operation == "price_competition_score":
            entity_column = params.get("entity_column")
            price_column = params.get("price_column") or column
            comparison_column = params.get("comparison_column")
            top_n = int(params.get("top_n", 10) or 10)
            if entity_column in current.columns and price_column in current.columns:
                temp = current[[entity_column, price_column] + ([comparison_column] if comparison_column in current.columns else [])].copy()
                temp[price_column] = pd.to_numeric(temp[price_column], errors="coerce")
                temp = temp.dropna(subset=[entity_column, price_column])
                if not temp.empty:
                    rows = []
                    for entity, frame in temp.groupby(entity_column):
                        avg_price = float(frame[price_column].mean())
                        price_cv = float(frame[price_column].std(ddof=0) / avg_price) if avg_price else 0.0
                        fragmentation = float(frame[comparison_column].nunique(dropna=True)) if comparison_column in frame.columns else float(len(frame))
                        rows.append({
                            entity_column: entity,
                            "avg_price": avg_price,
                            "price_cv": price_cv,
                            "fragmentation": fragmentation,
                        })
                    comp_df = pd.DataFrame(rows)
                    if not comp_df.empty:
                        comp_df["competition_score"] = (
                            0.4 * (1.0 - comp_df["avg_price"].rank(pct=True).fillna(0.5)) +
                            0.35 * comp_df["fragmentation"].rank(pct=True).fillna(0.0) +
                            0.25 * (1.0 - comp_df["price_cv"].rank(pct=True).fillna(0.5))
                        )
                        comp_df = comp_df.sort_values("competition_score", ascending=False).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(comp_df.to_dict(orient="records"))
                        result_payload["value"] = float(comp_df.iloc[0]["competition_score"]) if len(comp_df) else None
        elif operation == "retention_risk_proxy":
            customer_column = params.get("customer_column")
            order_column = params.get("order_column")
            freight_column = params.get("freight_column") or column
            time_column = params.get("time_column")
            if customer_column in current.columns and order_column in current.columns and freight_column in current.columns:
                temp = current[[customer_column, order_column, freight_column] + ([time_column] if time_column in current.columns else [])].copy()
                temp[freight_column] = pd.to_numeric(temp[freight_column], errors="coerce")
                temp = temp.dropna(subset=[customer_column, order_column, freight_column])
                if not temp.empty:
                    grouped = temp.groupby(customer_column).agg(
                        avg_freight=(freight_column, "mean"),
                        order_count=(order_column, pd.Series.nunique),
                    ).reset_index()
                    grouped["repeat_flag"] = (grouped["order_count"] > 1).astype(float)
                    if len(grouped) >= 3 and grouped["avg_freight"].nunique() >= 2:
                        corr = grouped["avg_freight"].corr(grouped["repeat_flag"], method="spearman")
                        grouped["risk_segment"] = pd.qcut(grouped["avg_freight"], q=min(4, grouped["avg_freight"].nunique()), duplicates="drop")
                        segment_rows = (
                            grouped.groupby("risk_segment", observed=False)
                            .agg(avg_freight=("avg_freight", "mean"), repeat_rate=("repeat_flag", "mean"), customers=(customer_column, "count"))
                            .reset_index()
                        )
                        result_payload["rows"] = _sanitize_numbers(segment_rows.to_dict(orient="records"))
                        result_payload["value"] = float(-corr) if pd.notna(corr) else None
                        result_payload["summary"] = {
                            "freight_repeat_correlation": float(corr) if pd.notna(corr) else None,
                            "overall_repeat_rate": float(grouped["repeat_flag"].mean()),
                        }
        elif operation == "logistics_optimization_opportunity":
            entity_column = params.get("entity_column")
            freight_column = params.get("freight_column") or column
            demand_column = params.get("demand_column")
            supply_column = params.get("supply_column")
            top_n = int(params.get("top_n", 10) or 10)
            min_support = int(params.get("min_support", 0) or 0)
            required = [entity_column, freight_column, demand_column]
            if all(col in current.columns for col in required):
                temp_cols = [entity_column, freight_column, demand_column]
                if supply_column in current.columns:
                    temp_cols.append(supply_column)
                temp = current[temp_cols].copy()
                temp[freight_column] = pd.to_numeric(temp[freight_column], errors="coerce")
                temp = temp.dropna(subset=[entity_column, freight_column, demand_column])
                if not temp.empty:
                    rows = []
                    for entity, frame in temp.groupby(entity_column):
                        avg_freight = float(frame[freight_column].mean())
                        demand = float(frame[demand_column].nunique(dropna=True))
                        supply = float(frame[supply_column].nunique(dropna=True)) if supply_column in frame.columns else 0.0
                        rows.append({
                            entity_column: entity,
                            "avg_freight": avg_freight,
                            "demand": demand,
                            "supply_presence": supply,
                        })
                    opt_df = pd.DataFrame(rows)
                    if not opt_df.empty:
                        support_threshold = max(min_support, _adaptive_support_threshold(opt_df["demand"], floor=10, fraction=0.2))
                        supported = opt_df[opt_df["demand"] >= support_threshold].copy()
                        if supported.empty:
                            supported = opt_df[opt_df["demand"] >= max(3, support_threshold // 2)].copy()
                        if supported.empty:
                            supported = opt_df.copy()
                        opt_df = supported
                        opt_df["cost_rank"] = opt_df["avg_freight"].rank(pct=True).fillna(0.0)
                        opt_df["demand_rank"] = opt_df["demand"].rank(pct=True).fillna(0.0)
                        if "supply_presence" in opt_df:
                            opt_df["supply_gap"] = 1.0 - opt_df["supply_presence"].rank(pct=True).fillna(0.5)
                        else:
                            opt_df["supply_gap"] = 0.5
                        opt_df["optimization_score"] = (
                            0.5 * opt_df["cost_rank"] +
                            0.3 * opt_df["demand_rank"] +
                            0.2 * opt_df["supply_gap"]
                        )
                        opt_df = opt_df.sort_values("optimization_score", ascending=False).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(opt_df[[entity_column, "avg_freight", "demand", "supply_presence", "optimization_score"]].to_dict(orient="records"))
                        result_payload["value"] = float(opt_df.iloc[0]["optimization_score"]) if len(opt_df) else None
                        result_payload["summary"] = {
                            "support_threshold": int(support_threshold),
                            "entity_column": entity_column,
                            "scoring": "Ranks supported geographies by high freight burden, demand, and relative local supply gap.",
                        }
        elif operation == "price_band_demand":
            price_column = params.get("price_column") or column
            demand_column = params.get("demand_column")
            bands = int(params.get("bands", 5) or 5)
            top_n = int(params.get("top_n", bands) or bands)
            proxy_mode = params.get("proxy_mode", "order_demand")
            if price_column in current.columns and demand_column in current.columns:
                temp = current[[price_column, demand_column]].copy()
                temp[price_column] = pd.to_numeric(temp[price_column], errors="coerce")
                temp = temp.dropna(subset=[price_column, demand_column])
                if not temp.empty and temp[price_column].nunique() >= 3:
                    q = min(bands, int(temp[price_column].nunique()))
                    temp["_price_band"] = pd.qcut(temp[price_column], q=q, duplicates="drop")
                    grouped = temp.groupby("_price_band").agg(
                        demand=(demand_column, "nunique"),
                        observations=(demand_column, "count"),
                        avg_price=(price_column, "mean"),
                    ).reset_index()
                    grouped["price_band"] = grouped["_price_band"].astype(str)
                    grouped = grouped.drop(columns=["_price_band"]).sort_values(["demand", "observations"], ascending=[False, False]).head(top_n)
                    result_payload["rows"] = _sanitize_numbers(grouped.to_dict(orient="records"))
                    result_payload["value"] = float(grouped.iloc[0]["demand"]) if len(grouped) else None
                    result_payload["summary"] = {
                        "proxy_mode": proxy_mode,
                        "warning": "True conversion data is unavailable; this ranks observed order demand by price band as a proxy.",
                        "bands_used": int(len(grouped)),
                    }
        elif operation == "basket_cooccurrence":
            entity_column = params.get("entity_column") or params.get("item_column")
            order_column = params.get("order_column")
            top_n = int(params.get("top_n", 10) or 10)
            if entity_column in current.columns and order_column in current.columns:
                from itertools import combinations
                import math
                pair_counts = {}
                grouped = current.groupby(order_column)[entity_column].apply(lambda s: sorted(set(s.dropna().astype(str))))
                basket_count = int(len(grouped))
                multi_item_basket_count = 0
                item_counts = {}
                for items in grouped:
                    if len(items) < 2:
                        for item in items:
                            item_counts[item] = item_counts.get(item, 0) + 1
                        continue
                    multi_item_basket_count += 1
                    for item in items:
                        item_counts[item] = item_counts.get(item, 0) + 1
                    for left, right in combinations(items, 2):
                        pair_counts[(left, right)] = pair_counts.get((left, right), 0) + 1
                if pair_counts:
                    recommendation_mode = bool(params.get("recommendation_mode"))
                    min_pair_orders = int(params.get("min_pair_orders") or 0)
                    if not min_pair_orders and recommendation_mode:
                        min_pair_orders = max(10, int(round(basket_count * 0.0001)))
                    min_confidence = float(params.get("min_confidence", 0.0) or 0.0)
                    if not min_confidence and recommendation_mode:
                        min_confidence = 0.05
                    min_lift = float(params.get("min_lift", 0.0) or 0.0)
                    if not min_lift and recommendation_mode:
                        min_lift = 1.0
                    pair_rows = []
                    candidate_pair_count = 0
                    for (left, right), count in pair_counts.items():
                        if min_pair_orders and count < min_pair_orders:
                            continue
                        candidate_pair_count += 1
                        left_count = item_counts.get(left, 0)
                        right_count = item_counts.get(right, 0)
                        pair_support = float(count / basket_count) if basket_count else 0.0
                        left_support = float(left_count / basket_count) if basket_count else 0.0
                        right_support = float(right_count / basket_count) if basket_count else 0.0
                        confidence_a_to_b = float(count / left_count) if left_count else 0.0
                        confidence_b_to_a = float(count / right_count) if right_count else 0.0
                        lift = float(pair_support / (left_support * right_support)) if left_support and right_support else 0.0
                        leverage = pair_support - (left_support * right_support)
                        confidence_options = [
                            ("a_to_b", left, right, confidence_a_to_b, right_support),
                            ("b_to_a", right, left, confidence_b_to_a, left_support),
                        ]
                        direction, antecedent, consequent, directional_confidence, consequent_support = max(
                            confidence_options,
                            key=lambda item: item[3],
                        )
                        if recommendation_mode and (directional_confidence < min_confidence or lift < min_lift):
                            continue
                        conviction = (
                            float((1.0 - consequent_support) / (1.0 - directional_confidence))
                            if directional_confidence < 1.0
                            else None
                        )
                        recommendation_score = lift * pair_support * directional_confidence * math.log1p(count)
                        row = {
                            "item_a": left,
                            "item_b": right,
                            "antecedent": antecedent,
                            "consequent": consequent,
                            "rule_direction": direction,
                            "pair_orders": count,
                            "pair_support": pair_support,
                            "item_a_support": left_support,
                            "item_b_support": right_support,
                            "confidence_a_to_b": confidence_a_to_b,
                            "confidence_b_to_a": confidence_b_to_a,
                            "rule_confidence": directional_confidence,
                            "lift": lift,
                            "leverage": leverage,
                            "conviction": conviction,
                            "recommendation_score": recommendation_score,
                        }
                        pair_rows.append(row)
                    if pair_rows:
                        sort_cols = ["recommendation_score", "lift", "pair_orders"] if recommendation_mode else ["pair_orders", "lift"]
                        pair_df = pd.DataFrame(pair_rows).sort_values(sort_cols, ascending=False).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(pair_df.to_dict(orient="records"))
                        result_payload["value"] = float(pair_df.iloc[0]["pair_orders"]) if len(pair_df) else None
                        result_payload["summary"] = {
                            "basket_count": basket_count,
                            "entity_column": entity_column,
                            "order_column": order_column,
                            "recommendation_mode": recommendation_mode,
                            "min_pair_orders": min_pair_orders,
                            "min_confidence": min_confidence,
                            "min_lift": min_lift,
                            "multi_item_basket_rate": float(multi_item_basket_count / basket_count) if basket_count else 0.0,
                            "candidate_pair_count": int(candidate_pair_count),
                            "raw_pair_count": int(len(pair_counts)),
                            "scoring": "Rows are directional association rules. recommendation_score = lift * pair_support * rule_confidence * log1p(pair_orders).",
                        }
                    elif recommendation_mode and item_counts:
                        popular = pd.DataFrame(
                            [
                                {"item": item, "basket_orders": count}
                                for item, count in item_counts.items()
                            ]
                        ).sort_values("basket_orders", ascending=False).head(top_n)
                        result_payload["rows"] = _sanitize_numbers(popular.to_dict(orient="records"))
                        result_payload["value"] = float(popular.iloc[0]["basket_orders"]) if len(popular) else None
                        result_payload["summary"] = {
                            "basket_count": basket_count,
                            "entity_column": entity_column,
                            "order_column": order_column,
                            "recommendation_mode": True,
                            "fallback_reason": "insufficient_supported_cooccurrence",
                            "multi_item_basket_rate": float(multi_item_basket_count / basket_count) if basket_count else 0.0,
                            "candidate_pair_count": int(candidate_pair_count),
                            "raw_pair_count": int(len(pair_counts)),
                            "scoring": "No supported item-pair rules met recommendation thresholds, so rows rank popular items as a recommendation fallback.",
                        }
                elif params.get("recommendation_mode") and item_counts:
                    popular = pd.DataFrame(
                        [
                            {"item": item, "basket_orders": count}
                            for item, count in item_counts.items()
                        ]
                    ).sort_values("basket_orders", ascending=False).head(top_n)
                    result_payload["rows"] = _sanitize_numbers(popular.to_dict(orient="records"))
                    result_payload["value"] = float(popular.iloc[0]["basket_orders"]) if len(popular) else None
                    result_payload["summary"] = {
                        "basket_count": basket_count,
                        "entity_column": entity_column,
                        "order_column": order_column,
                        "recommendation_mode": True,
                        "fallback_reason": "insufficient_supported_cooccurrence",
                        "multi_item_basket_rate": float(multi_item_basket_count / basket_count) if basket_count else 0.0,
                        "candidate_pair_count": 0,
                        "raw_pair_count": 0,
                        "scoring": "No supported item-pair rules were found, so rows rank popular items as a recommendation fallback.",
                    }
        elif operation == "basket_value_comparison":
            order_column = params.get("order_column")
            entity_column = params.get("entity_column")
            value_column = params.get("value_column") or column
            order_value_method = params.get("order_value_method", "sum")
            if order_column in current.columns and entity_column in current.columns and value_column in current.columns:
                temp = current[[order_column, entity_column, value_column]].copy()
                temp[value_column] = pd.to_numeric(temp[value_column], errors="coerce")
                temp = temp.dropna(subset=[order_column, entity_column, value_column])
                if not temp.empty:
                    if order_value_method == "max":
                        order_value = temp.groupby(order_column)[value_column].max()
                    elif order_value_method == "mean":
                        order_value = temp.groupby(order_column)[value_column].mean()
                    else:
                        order_value = temp.groupby(order_column)[value_column].sum()
                    entity_counts = temp.groupby(order_column)[entity_column].nunique(dropna=True)
                    comparison = pd.DataFrame({"order_value": order_value, "distinct_items": entity_counts}).dropna()
                    if not comparison.empty:
                        comparison["group"] = comparison["distinct_items"].apply(lambda count: "bundled_baskets" if count > 1 else "single_item_baskets")
                        grouped = comparison.groupby("group").agg(mean_value=("order_value", "mean"), count=("order_value", "count"), mean_distinct_items=("distinct_items", "mean")).reset_index()
                        result_payload["rows"] = _sanitize_numbers(grouped.to_dict(orient="records"))
                        bundled = grouped.loc[grouped["group"] == "bundled_baskets", "mean_value"]
                        single = grouped.loc[grouped["group"] == "single_item_baskets", "mean_value"]
                        if not bundled.empty and not single.empty:
                            result_payload["value"] = float(bundled.iloc[0] - single.iloc[0])
                        result_payload["summary"] = {"order_value_method": order_value_method, "entity_column": entity_column, "value_column": value_column}
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
