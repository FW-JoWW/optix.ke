from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from core.insight_validator import validate_insight
from core.recommendation_guard import guard_recommendations
from core.semantic_classifier import classify_relationship
from state.state import AnalystState


def _mentioned_columns(question: str, columns: List[str]) -> List[str]:
    question = question.lower()
    return [col for col in columns if col.lower() in question]


def _relationship_missing_ratio(df: pd.DataFrame | None, columns: List[str]) -> float:
    if df is None or not columns or any(col not in df.columns for col in columns):
        return 0.0
    subset = df[columns].copy()
    if len(subset) == 0:
        return 0.0
    return float(subset.isna().any(axis=1).mean())


def _first_dataframe(*values: Any) -> pd.DataFrame | None:
    for value in values:
        if isinstance(value, pd.DataFrame):
            return value
    return None


def _apply_semantic_guardrails(
    story: Dict[str, Any],
    result: Dict[str, Any],
    state: AnalystState,
) -> Dict[str, Any] | None:
    columns = story.get("columns") or []
    if story.get("column") and story["column"] not in columns:
        columns = [story["column"], *columns]
    if len(columns) < 2:
        return story

    x_column, y_column = columns[:2]
    reference_df = _first_dataframe(
        state.get("cleaned_data"),
        state.get("raw_analysis_dataset"),
        state.get("analysis_dataset"),
        state.get("dataframe"),
    )
    relationship_evidence = result.get("relationship_evidence", {}) or {}
    payload = relationship_evidence or result
    semantic = classify_relationship(
        x_column=x_column,
        y_column=y_column,
        stats_output=payload,
        metadata={
            "dataframe": reference_df,
            "dataset_profile": state.get("dataset_profile", {}) or {},
            "column_registry": state.get("column_registry", {}) or {},
            "business_question": state.get("business_question", ""),
        },
    )
    validity = validate_insight(
        stats_output=payload,
        semantic_output=semantic,
        metadata={"missing_ratio": _relationship_missing_ratio(reference_df, [x_column, y_column])},
    )
    recommendations = guard_recommendations(
        stats_output=payload,
        semantic_output=semantic,
        validation_output=validity,
    )

    if semantic.get("relationship_type") == "duplicate_feature":
        return None

    story["relationship_type"] = semantic.get("relationship_type")
    story["semantic_reasoning"] = semantic
    story["insight_validity"] = validity
    story["recommendation_restrictions"] = recommendations.get("recommendation_restrictions", [])
    story["guarded_recommendation"] = recommendations.get("final_recommendation")
    story["guardrail_allowed_actions"] = recommendations.get("allowed_actions", [])
    story["guardrail_blocked_actions"] = recommendations.get("blocked_actions", [])
    story["guardrail_triggered"] = bool(recommendations.get("guardrail_triggered"))
    return story


def _group_difference_story(result: Dict[str, Any], raw_df: pd.DataFrame) -> Dict[str, Any] | None:
    p_value = result.get("p_value")
    num_col = result.get("numeric_column") or result.get("column")
    cat_col = result.get("categorical_column") or result.get("group_column")

    if p_value is None or num_col not in raw_df.columns or cat_col not in raw_df.columns:
        return None

    safe_df = raw_df[[num_col, cat_col]].copy()
    safe_df[num_col] = pd.to_numeric(safe_df[num_col], errors="coerce")
    safe_df = safe_df.dropna(subset=[num_col, cat_col])

    if safe_df.empty:
        return None

    group_means = safe_df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
    if len(group_means) < 2 or p_value >= 0.05:
        return None

    top_group = str(group_means.index[0])
    bottom_group = str(group_means.index[-1])
    top_value = float(group_means.iloc[0])
    bottom_value = float(group_means.iloc[-1])
    diff = top_value - bottom_value

    return {
        "type": "group_difference",
        "insight": f"{top_group} has higher {num_col} than {bottom_group} by {round(diff, 2)}",
        "column": num_col,
        "group_column": cat_col,
        "top_group": top_group,
        "bottom_group": bottom_group,
        "top_value": top_value,
        "bottom_value": bottom_value,
        "effect_size": diff,
        "direction": f"{top_group} > {bottom_group}",
        "p_value": float(p_value),
        "confidence": "high",
    }


def _correlation_story(result: Dict[str, Any]) -> Dict[str, Any] | None:
    corr = result.get("correlation")
    if corr is None:
        return None

    col1 = result.get("column_1")
    col2 = result.get("column_2")
    direction = "positive" if corr > 0 else "negative"
    abs_corr = abs(float(corr))
    if abs_corr >= 0.7:
        strength = "strong"
        confidence = "high"
    elif abs_corr >= 0.3:
        strength = "moderate"
        confidence = "medium"
    else:
        strength = "weak"
        confidence = "low"

    return {
        "type": "correlation",
        "insight": f"{col1} and {col2} show a {strength} {direction} pattern",
        "columns": [col1, col2],
        "value": float(corr),
        "direction": direction,
        "strength": strength,
        "confidence": confidence,
    }


def _outlier_story(result: Dict[str, Any]) -> Dict[str, Any] | None:
    count = result.get("outlier_count", 0)
    if count <= 0:
        return None

    return {
        "type": "outliers",
        "insight": f"{count} outliers detected in {result.get('column')}",
        "column": result.get("column"),
        "count": int(count),
        "confidence": "medium",
    }


def _summary_stat_stories(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    stories: List[Dict[str, Any]] = []
    for item in result.get("results", []) or []:
        column = item.get("column")
        mean = item.get("mean")
        median = item.get("median")
        min_value = item.get("min")
        max_value = item.get("max")
        if column is None or mean is None:
            continue
        stories.append({
            "type": "summary_numeric",
            "insight": f"{column} typically centers around {round(float(mean), 2)}",
            "column": column,
            "mean": float(mean),
            "median": float(median) if median is not None else None,
            "min": float(min_value) if min_value is not None else None,
            "max": float(max_value) if max_value is not None else None,
            "confidence": "medium",
        })
    return stories


def _direct_computation_stories(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = result.get("results", {})
    rows = payload.get("rows", []) or []
    value = payload.get("value")
    strategy = payload.get("strategy")
    stories: List[Dict[str, Any]] = []

    if payload.get("trend"):
        stories.append({
            "type": "summary_numeric",
            "insight": f"The period trend is {payload.get('trend')}",
            "column": strategy or "trend",
            "mean": float(value) if value is not None else None,
            "median": float(value) if value is not None else None,
            "min": float(value) if value is not None else None,
            "max": float(value) if value is not None else None,
            "confidence": "medium",
        })
        return stories

    if payload.get("top_n") and value is not None:
        stories.append({
            "type": "summary_numeric",
            "insight": f"Top contributors account for about {round(float(value) * 100, 2)}% of the total",
            "column": strategy or "share",
            "mean": float(value),
            "median": float(value),
            "min": float(value),
            "max": float(value),
            "confidence": "medium",
        })
        return stories

    if payload.get("summary", {}).get("ratio") is not None:
        summary = payload.get("summary", {})
        ratio = float(summary.get("ratio", 0.0))
        display_value = float(value) if value is not None else (ratio * 100.0 if summary.get("as_percentage") else ratio)
        stories.append({
            "type": "summary_numeric",
            "insight": f"The cost ratio is about {round(display_value, 2)}{'%' if summary.get('as_percentage') else ''}",
            "column": strategy or "ratio_metric",
            "mean": display_value,
            "median": display_value,
            "min": float(summary.get("numerator_total", 0.0)),
            "max": float(summary.get("denominator_total", 0.0)),
            "confidence": "medium",
        })
        return stories
    if payload.get("summary", {}).get("proxy_mode") == "order_demand" and payload.get("rows"):
        rows = payload.get("rows", [])
        top = rows[0]
        stories.append({
            "type": "summary_numeric",
            "insight": f"The strongest observed order-demand proxy appears in the {top.get('price_band')} price range, but true conversion data is unavailable.",
            "column": strategy or "price_band_demand",
            "mean": float(top.get("demand", 0.0)),
            "median": float(top.get("avg_price", 0.0)),
            "min": float(top.get("observations", 0.0)),
            "max": float(top.get("demand", 0.0)),
            "confidence": "low",
        })
        return stories
    if payload.get("summary", {}).get("correlation") is not None:
        summary = payload.get("summary", {})
        corr = float(summary.get("correlation", 0.0))
        direction = "positive" if corr > 0 else "negative"
        stories.append({
            "type": "relationship",
            "insight": f"The relationship between {summary.get('x_column')} and {summary.get('y_column')} is {direction} with correlation about {round(corr, 3)}",
            "column": strategy or "pairwise_relationship",
            "mean": corr,
            "median": corr,
            "min": float(summary.get("sample_size", 0)),
            "max": float(summary.get("sample_size", 0)),
            "confidence": "medium",
        })
        return stories
    if payload.get("summary", {}).get("mean_delivery_days") is not None:
        summary = payload.get("summary", {})
        stories.append({
            "type": "summary_numeric",
            "insight": f"Average delivery time is about {round(float(summary.get('mean_delivery_days', 0.0)), 2)} days",
            "column": strategy or "delivery_duration",
            "mean": float(summary.get("mean_delivery_days", 0.0)),
            "median": float(summary.get("median_delivery_days", 0.0)),
            "min": float(summary.get("min_delivery_days", 0.0)),
            "max": float(summary.get("max_delivery_days", 0.0)),
            "confidence": "medium",
        })
        return stories
    if payload.get("summary", {}).get("mean_gap_days") is not None:
        summary = payload.get("summary", {})
        stories.append({
            "type": "summary_numeric",
            "insight": f"Average actual-versus-estimated delivery gap is {round(float(summary.get('mean_gap_days', 0.0)), 2)} days",
            "column": strategy or "delivery_gap",
            "mean": float(summary.get("mean_gap_days", 0.0)),
            "median": float(summary.get("median_gap_days", 0.0)),
            "min": float(summary.get("early_share", 0.0)),
            "max": float(summary.get("late_share", 0.0)),
            "confidence": "medium",
        })
        return stories
    if payload.get("summary", {}).get("mode") is not None and payload.get("summary", {}).get("matching_orders") is not None and value is not None:
        summary = payload.get("summary", {})
        mode = summary.get("mode")
        stories.append({
            "type": "summary_numeric",
            "insight": f"About {round(float(value) * 100, 2)}% of orders were delivered {mode}",
            "column": strategy or "delivery_timing_share",
            "mean": float(value),
            "median": float(value),
            "min": float(summary.get("matching_orders", 0)),
            "max": float(summary.get("matching_orders", 0)),
            "confidence": "medium",
        })
        return stories
    if payload.get("summary", {}).get("total_orders") is not None and value is not None:
        summary = payload.get("summary", {})
        mode = summary.get("mode", "status")
        stories.append({
            "type": "summary_numeric",
            "insight": f"About {round(float(value) * 100, 2)}% of orders fall into the {mode} status condition",
            "column": strategy or "status_share",
            "mean": float(value),
            "median": float(value),
            "min": float(summary.get("matching_orders", 0)),
            "max": float(summary.get("total_orders", 0)),
            "confidence": "medium",
        })
        return stories

    if payload.get("summary", {}).get("repeat_entities") is not None and value is not None:
        stories.append({
            "type": "summary_numeric",
            "insight": f"About {round(float(value) * 100, 2)}% of entities are repeat purchasers",
            "column": strategy or "repeat_rate",
            "mean": float(value),
            "median": float(value),
            "min": float(value),
            "max": float(value),
            "confidence": "medium",
        })
        return stories

    if payload.get("summary", {}).get("mean_orders_per_customer") is not None:
        summary = payload.get("summary", {})
        stories.append({
            "type": "summary_numeric",
            "insight": f"Customers place about {round(float(summary.get('mean_orders_per_customer', 0.0)), 2)} orders on average, with repeat behavior around {round(float(summary.get('repeat_rate', 0.0)) * 100, 2)}%",
            "column": strategy or "customer_order_frequency",
            "mean": float(summary.get("mean_orders_per_customer", 0.0)),
            "median": float(summary.get("median_orders_per_customer", 0.0)),
            "min": float(summary.get("repeat_rate", 0.0)),
            "max": float(summary.get("repeat_rate", 0.0)),
            "confidence": "medium",
        })
        return stories

    if payload.get("summary", {}).get("mean_days_between_first_second") is not None:
        summary = payload.get("summary", {})
        stories.append({
            "type": "summary_numeric",
            "insight": f"Repeat purchasers take about {round(float(summary.get('mean_days_between_first_second', 0.0)), 2)} days on average to make a second purchase",
            "column": strategy or "purchase_gap",
            "mean": float(summary.get("mean_days_between_first_second", 0.0)),
            "median": float(summary.get("median_days_between_first_second", 0.0)),
            "min": float(summary.get("median_days_between_first_second", 0.0)),
            "max": float(summary.get("mean_days_between_first_second", 0.0)),
            "confidence": "medium",
        })
        return stories

    if payload.get("summary", {}).get("median_days_since_last_purchase") is not None:
        summary = payload.get("summary", {})
        median_inactive = summary.get("median_days_since_last_purchase") or 0.0
        mean_inactive = summary.get("mean_days_since_last_purchase") or median_inactive
        median_gap = summary.get("median_repeat_gap_days")
        gap_text = ""
        if median_gap is not None:
            gap_text = f"; repeat gaps have a median of {round(float(median_gap), 2)} days"
        stories.append({
            "type": "summary_numeric",
            "insight": f"Median customer inactivity is {round(float(median_inactive), 2)} days{gap_text}",
            "column": "churn_speed_proxy",
            "mean": float(mean_inactive),
            "median": float(median_inactive),
            "min": float(median_gap) if median_gap is not None else float(median_inactive),
            "max": float(summary.get("customers", 0)),
            "confidence": "medium",
        })
        return stories

    if payload.get("summary", {}).get("single_purchase_entities") is not None and value is not None:
        stories.append({
            "type": "summary_numeric",
            "insight": f"About {round(float(value) * 100, 2)}% of customers purchase only once",
            "column": strategy or "single_purchase_share",
            "mean": float(value),
            "median": float(value),
            "min": float(value),
            "max": float(value),
            "confidence": "medium",
        })
        return stories

    if payload.get("summary", {}).get("pattern") is not None:
        summary = payload.get("summary", {})
        pattern = summary.get("pattern")
        if pattern == "many_cheap_items":
            insight = "Order behavior leans toward many lower-priced items rather than a few expensive ones"
        else:
            insight = "Order behavior leans toward fewer higher-priced items rather than many cheap ones"
        stories.append({
            "type": "summary_numeric",
            "insight": insight,
            "column": strategy or "basket_value_pattern",
            "mean": float(summary.get("mean_item_price", 0.0)),
            "median": float(summary.get("median_item_price", 0.0)),
            "min": float(summary.get("mean_items_per_order", 0.0)),
            "max": float(summary.get("median_items_per_order", 0.0)),
            "confidence": "medium",
        })
        return stories

    if payload.get("summary", {}).get("dormant_customers") is not None:
        summary = payload.get("summary", {})
        stories.append({
            "type": "summary_numeric",
            "insight": f"{int(summary.get('dormant_customers', 0))} customers look dormant using a {int(summary.get('dormancy_threshold_days', 0))}-day inactivity threshold",
            "column": strategy or "dormancy_count",
            "mean": float(summary.get("dormancy_rate", 0.0)),
            "median": float(summary.get("dormancy_rate", 0.0)),
            "min": float(summary.get("dormancy_rate", 0.0)),
            "max": float(summary.get("dormancy_rate", 0.0)),
            "confidence": "medium",
        })
        return stories

    summary = payload.get("summary", {}) or {}
    if summary.get("columns_with_missing") is not None and not rows:
        stories.append({
            "type": "data_quality",
            "insight": f"{int(summary.get('columns_with_missing', 0))} columns contain missing values, with {int(summary.get('total_missing_cells', 0))} missing cells overall",
            "column": "missingness",
            "mean": float(summary.get("columns_with_missing", 0)),
            "median": float(summary.get("columns_with_missing", 0)),
            "min": float(summary.get("total_missing_cells", 0)),
            "max": float(summary.get("columns_checked", 0)),
            "confidence": "medium",
        })
        return stories
    if summary.get("duplicate_rows") is not None:
        stories.append({
            "type": "data_quality",
            "insight": f"{int(summary.get('duplicate_rows', 0))} exact duplicate rows were found across {int(summary.get('columns_checked', 0))} checked columns",
            "column": "duplicate_rows",
            "mean": float(summary.get("duplicate_rows", 0)),
            "median": float(summary.get("duplicate_rows", 0)),
            "min": float(summary.get("duplicate_groups", 0)),
            "max": float(summary.get("row_count", 0)),
            "confidence": "medium",
        })
        return stories
    if summary.get("timestamp_columns_checked") is not None:
        stories.append({
            "type": "data_quality",
            "insight": f"{int(summary.get('issue_count', 0))} timestamp consistency issues were found across {int(summary.get('timestamp_columns_checked', 0))} timestamp columns",
            "column": "timestamp_issue_count",
            "mean": float(summary.get("issue_count", 0)),
            "median": float(summary.get("issue_count", 0)),
            "min": 0.0,
            "max": float(summary.get("timestamp_columns_checked", 0)),
            "confidence": "medium",
        })
        return stories
    if summary.get("invalid_count") is not None:
        stories.append({
            "type": "data_quality",
            "insight": f"{int(summary.get('invalid_count', 0))} invalid values were found in {summary.get('column')}",
            "column": "invalid_count",
            "mean": float(summary.get("invalid_count", 0)),
            "median": float(summary.get("invalid_count", 0)),
            "min": float(summary.get("min_observed") or 0.0),
            "max": float(summary.get("non_numeric_count", 0)),
            "confidence": "medium",
        })
        return stories
    if summary.get("rows_checked") is not None and summary.get("issue_count") is not None:
        stories.append({
            "type": "data_quality",
            "insight": f"{int(summary.get('issue_count', 0))} impossible delivery-date records were found",
            "column": "delivery_date_issue_count",
            "mean": float(summary.get("issue_count", 0)),
            "median": float(summary.get("issue_count", 0)),
            "min": 0.0,
            "max": float(summary.get("rows_checked", 0)),
            "confidence": "medium",
        })
        return stories
    if summary.get("variant_groups") is not None:
        stories.append({
            "type": "data_quality",
            "insight": f"{summary.get('column')} has {int(summary.get('blank_or_missing_count', 0))} blank/missing labels and {int(summary.get('variant_groups', 0))} normalized variant groups",
            "column": "label_quality_issue_count",
            "mean": float(summary.get("blank_or_missing_count", 0)),
            "median": float(summary.get("blank_or_missing_count", 0)),
            "min": float(summary.get("variant_groups", 0)),
            "max": float(summary.get("unique_labels", 0)),
            "confidence": "medium",
        })
        return stories

    if rows:
        first = rows[0]
        keys = list(first.keys())
        if {"elasticity_signal", "price_demand_correlation", "periods"}.issubset(first.keys()):
            entity_key = next((key for key in keys if key not in {"elasticity_signal", "price_demand_correlation", "periods"}), "segment")
            top = rows[0]
            stories.append({
                "type": "grouped_numeric",
                "insight": f"{top[entity_key]} shows the strongest price-elasticity signal",
                "column": "elasticity_signal",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top["elasticity_signal"]),
                "effect_size": float(top["price_demand_correlation"]),
                "confidence": "medium",
            })
            return stories
        if {"avg_price", "demand", "discount_like"}.issubset(first.keys()):
            summary = payload.get("summary", {}) or {}
            uplift = value if value is not None else 0.0
            direction = "higher" if uplift >= 0 else "lower"
            stories.append({
                "type": "summary_numeric",
                "insight": f"Lower-price periods show {direction} demand, with an estimated lift of {round(float(uplift) * 100, 2)}%",
                "column": "discount_volume_effect",
                "mean": float(summary.get("price_demand_correlation", 0.0)) if summary.get("price_demand_correlation") is not None else float(uplift),
                "median": float(summary.get("baseline_price", 0.0)) if summary.get("baseline_price") is not None else float(uplift),
                "min": float(summary.get("non_discount_period_demand", 0.0)) if summary.get("non_discount_period_demand") is not None else float(uplift),
                "max": float(summary.get("discount_like_period_demand", 0.0)) if summary.get("discount_like_period_demand") is not None else float(uplift),
                "confidence": "medium",
            })
            return stories
        if {"avg_price", "price_cv", "fragmentation", "competition_score"}.issubset(first.keys()):
            entity_key = next((key for key in keys if key not in {"avg_price", "price_cv", "fragmentation", "competition_score"}), "segment")
            top = rows[0]
            stories.append({
                "type": "grouped_numeric",
                "insight": f"{top[entity_key]} shows the strongest price-competition pressure",
                "column": "competition_score",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top["competition_score"]),
                "effect_size": float(top["fragmentation"]),
                "confidence": "medium",
            })
            return stories
        if {"price_band", "demand"}.issubset(first.keys()):
            ranked = sorted(rows, key=lambda row: row.get("demand", float("-inf")), reverse=True)
            top = ranked[0]
            bottom = ranked[-1]
            stories.append({
                "type": "grouped_numeric",
                "insight": f"{top['price_band']} shows the strongest demand while {bottom['price_band']} trails",
                "column": "demand",
                "group_column": "price_band",
                "top_group": str(top["price_band"]),
                "bottom_group": str(bottom["price_band"]),
                "top_value": float(top["demand"]),
                "bottom_value": float(bottom["demand"]),
                "effect_size": float(top["demand"]) - float(bottom["demand"]),
                "confidence": "medium",
            })
            return stories
        if {"review_segment", "repeat_rate"}.issubset(first.keys()):
            ranked = sorted(rows, key=lambda row: row.get("repeat_rate", float("-inf")), reverse=True)
            top = ranked[0]
            bottom = ranked[-1]
            stories.append({
                "type": "grouped_numeric",
                "insight": f"{top['review_segment']} customers show stronger repeat behavior than {bottom['review_segment']} customers",
                "column": "repeat_rate",
                "group_column": "review_segment",
                "top_group": str(top["review_segment"]),
                "bottom_group": str(bottom["review_segment"]),
                "top_value": float(top["repeat_rate"]),
                "bottom_value": float(bottom["repeat_rate"]),
                "effect_size": float(top["repeat_rate"]) - float(bottom["repeat_rate"]),
                "confidence": "medium",
            })
            return stories
        if {"avg_freight", "repeat_rate", "customers"}.issubset(first.keys()):
            ranked = sorted(rows, key=lambda row: row.get("avg_freight", float("-inf")), reverse=True)
            top = ranked[0]
            bottom = ranked[-1]
            stories.append({
                "type": "grouped_numeric",
                "insight": f"Higher-freight segments show weaker repeat behavior than lower-freight segments",
                "column": "repeat_rate",
                "group_column": "risk_segment",
                "top_group": str(top.get("risk_segment")),
                "bottom_group": str(bottom.get("risk_segment")),
                "top_value": float(top["repeat_rate"]),
                "bottom_value": float(bottom["repeat_rate"]),
                "effect_size": float(bottom["repeat_rate"]) - float(top["repeat_rate"]),
                "confidence": "medium",
            })
            return stories
        if {"avg_freight", "demand", "supply_presence", "optimization_score"}.issubset(first.keys()):
            entity_key = next((key for key in keys if key not in {"avg_freight", "demand", "supply_presence", "optimization_score", "cost_rank", "demand_rank", "supply_gap"}), "segment")
            top = rows[0]
            stories.append({
                "type": "logistics_optimization_opportunity",
                "insight": f"{top[entity_key]} shows the strongest logistics cost-reduction opportunity",
                "column": "optimization_score",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top["optimization_score"]),
                "effect_size": float(top["avg_freight"]),
                "demand": float(top["demand"]),
                "supply_presence": float(top["supply_presence"]),
                "confidence": "medium",
            })
            return stories
        if {"period", "delay_days"}.issubset(first.keys()):
            latest = rows[-1]
            stories.append({
                "type": "summary_numeric",
                "insight": f"Latest supported average delivery delay is {round(float(latest['delay_days']), 2)} days",
                "column": "delay_days",
                "mean": float(latest["delay_days"]),
                "median": float(latest["delay_days"]),
                "min": float(latest["delay_days"]),
                "max": float(latest["delay_days"]),
                "confidence": "medium",
            })
            return stories
        if {"_period_bucket", "issue_rate", "issue_orders", "total_orders"}.issubset(first.keys()):
            top = rows[0]
            stories.append({
                "type": "grouped_numeric",
                "insight": f"{top['_period_bucket']} shows the highest operational issue rate among sufficiently supported periods",
                "column": "smoothed_issue_rate" if "smoothed_issue_rate" in first else "issue_rate",
                "group_column": "_period_bucket",
                "top_group": str(top["_period_bucket"]),
                "top_value": float(top.get("smoothed_issue_rate", top["issue_rate"])),
                "effect_size": float(top.get("smoothed_issue_rate", top["issue_rate"])),
                "confidence": "medium",
            })
            return stories
        if {"rate", "total_orders"}.issubset(first.keys()) and "_period_bucket" in first:
            latest = rows[-1]
            stories.append({
                "type": "summary_numeric",
                "insight": f"Latest sufficiently supported status rate is {round(float(latest.get('smoothed_rate', latest['rate'])) * 100, 2)}%",
                "column": "smoothed_rate" if "smoothed_rate" in latest else "rate",
                "mean": float(latest.get("smoothed_rate", latest["rate"])),
                "median": float(latest.get("smoothed_rate", latest["rate"])),
                "min": float(latest.get("smoothed_rate", latest["rate"])),
                "max": float(latest.get("smoothed_rate", latest["rate"])),
                "confidence": "medium",
            })
            return stories
        if {"item_a", "item_b", "pair_orders"}.issubset(first.keys()):
            top = rows[0]
            recommendation_mode = bool((payload.get("summary") or {}).get("recommendation_mode"))
            lift = float(top.get("lift", 0.0) or 0.0)
            confidence = float(top.get("rule_confidence", 0.0) or 0.0)
            if not confidence:
                confidence = max(float(top.get("confidence_a_to_b", 0.0) or 0.0), float(top.get("confidence_b_to_a", 0.0) or 0.0))
            antecedent = top.get("antecedent", top["item_a"])
            consequent = top.get("consequent", top["item_b"])
            if recommendation_mode:
                action = f"when {antecedent} appears, recommend {consequent} as the strongest association-rule candidate"
            else:
                action = "is the most common basket pair"
            metric_text = f"support {round(float(top.get('pair_support', 0.0)) * 100, 3)}%, confidence {round(confidence * 100, 2)}%, lift {round(lift, 2)}"
            stories.append({
                "type": "basket_cooccurrence",
                "insight": f"{top['item_a']} + {top['item_b']} {action} ({metric_text})",
                "column": "recommendation_score" if recommendation_mode else "pair_orders",
                "group_column": "item_pair",
                "top_group": f"{top['item_a']} + {top['item_b']}",
                "top_value": float(top.get("recommendation_score", top["pair_orders"])),
                "effect_size": lift,
                "support": float(top.get("pair_support", 0.0)),
                "directional_confidence": confidence,
                "antecedent": str(antecedent),
                "consequent": str(consequent),
                "leverage": float(top.get("leverage", 0.0) or 0.0),
                "conviction": top.get("conviction"),
                "recommendation_restrictions": ["observational_cooccurrence_not_causal"] if recommendation_mode else [],
                "confidence": "medium",
            })
            return stories
        if {"item", "basket_orders"}.issubset(first.keys()):
            top = rows[0]
            stories.append({
                "type": "basket_recommendation_fallback",
                "insight": f"{top['item']} is the strongest popularity-based recommendation fallback",
                "column": "basket_orders",
                "group_column": "item",
                "top_group": str(top["item"]),
                "top_value": float(top["basket_orders"]),
                "effect_size": float(top["basket_orders"]),
                "recommendation_restrictions": ["insufficient_supported_cooccurrence", "popularity_proxy_not_personalized"],
                "confidence": "low",
            })
            return stories
        if {"contribution", "share", "rank"}.issubset(first.keys()):
            summary = payload.get("summary", {}) or {}
            top_share = float(summary.get("top_share", payload.get("value") or 0.0))
            top_n = int(summary.get("top_n", len(rows)) or len(rows))
            stories.append({
                "type": "dependency_risk",
                "insight": f"Top {top_n} entities account for {round(top_share * 100, 2)}% of the measured activity",
                "column": "share",
                "group_column": "dependency_rank",
                "top_group": f"top_{top_n}",
                "top_value": top_share,
                "effect_size": float(summary.get("hhi", 0.0) or 0.0),
                "confidence": "medium",
            })
            return stories
        if {"avg_review", "low_review_rate", "crisis_score"}.issubset(first.keys()):
            entity_key = next((key for key in keys if key not in {"avg_review", "low_review_rate", "total_orders", "crisis_score"}), "entity")
            top = rows[0]
            stories.append({
                "type": "review_crisis",
                "insight": f"{top[entity_key]} has the strongest review-crisis signal with {round(float(top['low_review_rate']) * 100, 2)}% low reviews",
                "column": "crisis_score",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top["crisis_score"]),
                "effect_size": float(top["low_review_rate"]),
                "confidence": "medium",
            })
            return stories
        if {"period", "late_rate", "cluster_score"}.issubset(first.keys()):
            top = rows[0]
            stories.append({
                "type": "late_delivery_cluster",
                "insight": f"{top['period']} has the strongest late-delivery cluster signal at {round(float(top['late_rate']) * 100, 2)}% late deliveries",
                "column": "late_rate",
                "group_column": "period",
                "top_group": str(top["period"]),
                "top_value": float(top["late_rate"]),
                "effect_size": float(top["cluster_score"]),
                "confidence": "medium",
            })
            return stories
        if {"forecast_value", "recent_average", "trend_adjustment"}.issubset(first.keys()):
            top = rows[0]
            group_key = next((key for key in keys if key not in {"forecast_period", "forecast_value", "forecast_low", "forecast_high", "recent_average", "trend_adjustment", "seasonal_reference", "backtest_mae", "backtest_mape", "forecast_model", "forecast_reliability", "periods_used"}), "forecast")
            interval_text = ""
            if top.get("forecast_low") is not None and top.get("forecast_high") is not None:
                interval_text = f" with an approximate range of {round(float(top['forecast_low']), 2)} to {round(float(top['forecast_high']), 2)}"
            stories.append({
                "type": "forecast",
                "insight": f"{top.get(group_key, 'Next period')} forecast is {round(float(top['forecast_value']), 2)}{interval_text}",
                "column": "forecast_value",
                "group_column": group_key,
                "top_group": str(top.get(group_key, "next_period")),
                "top_value": float(top["forecast_value"]),
                "effect_size": float(top.get("trend_adjustment", 0.0) or 0.0),
                "forecast_reliability": top.get("forecast_reliability"),
                "backtest_mape": top.get("backtest_mape"),
                "confidence": "medium" if top.get("forecast_reliability") != "low" else "low",
            })
            return stories
        if {"capacity_need", "peak_period_demand"}.issubset(first.keys()):
            entity_key = next((key for key in keys if key not in {"capacity_need", "peak_period_demand", "recent_average_demand", "periods"}), "entity")
            top = rows[0]
            stories.append({
                "type": "capacity_planning",
                "insight": f"{top[entity_key]} has the highest estimated capacity need at {round(float(top['capacity_need']), 2)} orders per planning period",
                "column": "capacity_need",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top["capacity_need"]),
                "effect_size": float(top.get("peak_period_demand", 0.0) or 0.0),
                "confidence": "medium",
            })
            return stories
        if {"strategic_score", "orders"}.issubset(first.keys()):
            entity_key = next((key for key in keys if key not in {"strategic_score", "orders", "revenue", "avg_value", "avg_review", "avg_freight", "demand_score", "revenue_score", "value_score", "review_score_component", "freight_penalty", "evidence_grade", "dominant_signal"}), "entity")
            top = rows[0]
            mode = (payload.get("summary") or {}).get("mode", "growth")
            stories.append({
                "type": "strategic_opportunity",
                "insight": f"{top[entity_key]} has the strongest {mode} opportunity score, led by {top.get('dominant_signal', 'the strongest component')}",
                "column": "strategic_score",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top["strategic_score"]),
                "effect_size": float(top.get("orders", 0.0) or 0.0),
                "evidence_grade": top.get("evidence_grade"),
                "confidence": "medium",
            })
            return stories
        if {"target", "positive_rate", "positive_count"}.issubset(first.keys()):
            top = rows[0]
            stories.append({
                "type": "predictive_target_profile",
                "insight": f"{top['target']} target is {top.get('readiness', 'constructed')} with a positive rate of {round(float(top['positive_rate']) * 100, 2)}%",
                "column": "positive_rate",
                "group_column": "target",
                "top_group": str(top["target"]),
                "top_value": float(top["positive_rate"]),
                "effect_size": float(top.get("positive_count", 0.0) or 0.0),
                "baseline_accuracy": top.get("baseline_accuracy"),
                "class_balance": top.get("class_balance"),
                "confidence": "medium",
            })
            return stories
        if {"lifetime_value", "orders"}.issubset(first.keys()):
            top = rows[0]
            entity_key = next((key for key in keys if key not in {"lifetime_value", "orders", "value_per_order"}), "customer")
            stories.append({
                "type": "ltv_estimate",
                "insight": f"Average lifetime value is {round(float((payload.get('summary') or {}).get('average_lifetime_value', payload.get('value') or 0.0)), 2)}",
                "column": "lifetime_value",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top["lifetime_value"]),
                "effect_size": float(top.get("orders", 0.0) or 0.0),
                "confidence": "medium",
            })
            return stories
        if {"segment_id", "customers", "segment_score"}.issubset(first.keys()):
            top = rows[0]
            stories.append({
                "type": "customer_segmentation",
                "insight": f"{int((payload.get('summary') or {}).get('segments', len(rows)))} customer segments were identified using RFM-style features",
                "column": "segment_score",
                "group_column": "segment_id",
                "top_group": str(top["segment_id"]),
                "top_value": float(top["segment_score"]),
                "effect_size": float(top.get("customers", 0.0) or 0.0),
                "confidence": "medium",
            })
            return stories
        if {"recommended_promise_days", "median_delivery_days", "orders"}.issubset(first.keys()):
            entity_key = next((key for key in keys if key not in {"recommended_promise_days", "p95_delivery_days", "median_delivery_days", "promise_buffer_days", "current_late_rate", "current_median_promise_days", "orders", "promise_optimization_score"}), "segment")
            top = rows[0]
            stories.append({
                "type": "delivery_promise_optimization",
                "insight": f"{top[entity_key]} should use about {round(float(top['recommended_promise_days']), 2)} days as a conservative delivery promise with a {round(float(top.get('promise_buffer_days') or 0.0), 2)} day median buffer",
                "column": "recommended_promise_days",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top["recommended_promise_days"]),
                "effect_size": float(top.get("orders", 0.0) or 0.0),
                "confidence": "medium",
            })
            return stories
        if {"missing_count", "missing_rate", "dtype"}.issubset(first.keys()):
            summary = payload.get("summary", {}) or {}
            top = rows[0]
            stories.append({
                "type": "data_quality",
                "insight": f"{top['column']} has the highest missingness at {round(float(top['missing_rate']) * 100, 2)}%",
                "column": "missing_rate",
                "group_column": "column",
                "top_group": str(top["column"]),
                "top_value": float(top["missing_rate"]),
                "effect_size": float(summary.get("total_missing_cells", top["missing_count"])),
                "confidence": "medium",
            })
            return stories
        if payload.get("summary", {}).get("duplicate_rows") is not None:
            summary = payload.get("summary", {}) or {}
            stories.append({
                "type": "data_quality",
                "insight": f"{int(summary.get('duplicate_rows', 0))} exact duplicate rows were found across {int(summary.get('columns_checked', 0))} checked columns",
                "column": "duplicate_rows",
                "group_column": "row_fingerprint",
                "top_group": "exact_duplicates",
                "top_value": float(summary.get("duplicate_rows", 0)),
                "effect_size": float(summary.get("duplicate_groups", 0)),
                "confidence": "medium",
            })
            return stories
        if payload.get("summary", {}).get("timestamp_columns_checked") is not None:
            summary = payload.get("summary", {}) or {}
            issue_count = int(summary.get("issue_count", 0))
            stories.append({
                "type": "data_quality",
                "insight": f"{issue_count} timestamp consistency issues were found across {int(summary.get('timestamp_columns_checked', 0))} timestamp columns",
                "column": "timestamp_issue_count",
                "group_column": "timestamp_check",
                "top_group": str(rows[0].get("check", "timestamp_checks")) if rows else "timestamp_checks",
                "top_value": float(issue_count),
                "effect_size": float(rows[0].get("issue_count", issue_count)) if rows else float(issue_count),
                "confidence": "medium",
            })
            return stories
        if payload.get("summary", {}).get("invalid_count") is not None:
            summary = payload.get("summary", {}) or {}
            stories.append({
                "type": "data_quality",
                "insight": f"{int(summary.get('invalid_count', 0))} invalid values were found in {summary.get('column')}",
                "column": "invalid_count",
                "group_column": "column",
                "top_group": str(summary.get("column")),
                "top_value": float(summary.get("invalid_count", 0)),
                "effect_size": float(summary.get("min_observed") or 0.0),
                "confidence": "medium",
            })
            return stories
        if payload.get("summary", {}).get("rows_checked") is not None and payload.get("summary", {}).get("issue_count") is not None:
            summary = payload.get("summary", {}) or {}
            stories.append({
                "type": "data_quality",
                "insight": f"{int(summary.get('issue_count', 0))} impossible delivery-date records were found",
                "column": "delivery_date_issue_count",
                "group_column": "delivery_date_check",
                "top_group": str(rows[0].get("issue", "delivery_date_checks")) if rows else "delivery_date_checks",
                "top_value": float(summary.get("issue_count", 0)),
                "effect_size": float(summary.get("rows_checked", 0)),
                "confidence": "medium",
            })
            return stories
        if payload.get("summary", {}).get("variant_groups") is not None:
            summary = payload.get("summary", {}) or {}
            stories.append({
                "type": "data_quality",
                "insight": f"{summary.get('column')} has {int(summary.get('blank_or_missing_count', 0))} blank/missing labels and {int(summary.get('variant_groups', 0))} normalized variant groups",
                "column": "label_quality_issue_count",
                "group_column": "column",
                "top_group": str(summary.get("column")),
                "top_value": float(summary.get("blank_or_missing_count", 0)),
                "effect_size": float(summary.get("variant_groups", 0)),
                "confidence": "medium",
            })
            return stories
        if {"rapid_order_pairs", "min_gap_hours", "anomaly_score"}.issubset(first.keys()):
            entity_key = next((key for key in keys if key not in {"rapid_order_pairs", "min_gap_hours", "median_gap_hours", "latest_rapid_order", "total_orders", "anomaly_score"}), "entity")
            top = rows[0]
            stories.append({
                "type": "anomaly_signal",
                "insight": f"{top[entity_key]} shows the strongest rapid-repeat order signal with a minimum gap of {round(float(top['min_gap_hours']), 2)} hours",
                "column": "anomaly_score",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top["anomaly_score"]),
                "effect_size": float(top["rapid_order_pairs"]),
                "confidence": "medium",
            })
            return stories
        if {"transaction_value", "anomaly_score"}.issubset(first.keys()):
            entity_key = next((key for key in keys if key not in {"transaction_value", "z_score", "iqr_excess", "anomaly_score"}), "transaction")
            top = rows[0]
            stories.append({
                "type": "anomaly_signal",
                "insight": f"{top[entity_key]} has the strongest high-value transaction anomaly signal",
                "column": "anomaly_score",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top["anomaly_score"]),
                "effect_size": float(top["transaction_value"]),
                "confidence": "medium",
            })
            return stories
        if {"low_context_value", "high_signal_value", "mismatch_score"}.issubset(first.keys()):
            top = rows[0]
            entity_key = next((key for key in keys if key not in {"low_context_value", "high_signal_value", "mismatch_score"}), "record")
            stories.append({
                "type": "anomaly_signal",
                "insight": f"{top.get(entity_key, 'A record')} has a strong contextual mismatch: low context value with high signal value",
                "column": "mismatch_score",
                "group_column": entity_key,
                "top_group": str(top.get(entity_key, "record")),
                "top_value": float(top["mismatch_score"]),
                "effect_size": float(top["high_signal_value"]),
                "confidence": "medium",
            })
            return stories
        if {"period", "spike_score", "baseline_mean"}.issubset(first.keys()) and any(key not in {"period", "value", "spike_score", "baseline_mean"} for key in keys):
            entity_key = next((key for key in keys if key not in {"period", "value", "spike_score", "baseline_mean"}), "entity")
            top = rows[0]
            stories.append({
                "type": "anomaly_signal",
                "insight": f"{top[entity_key]} shows the strongest entity-specific spike in {top['period']}",
                "column": "spike_score",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top["spike_score"]),
                "effect_size": float(top["value"]),
                "confidence": "medium",
            })
            return stories
        if {"matching_customers", "duplicate_score"}.issubset(first.keys()):
            top = rows[0]
            stories.append({
                "type": "anomaly_signal",
                "insight": f"A repeated customer behavior fingerprint appears across {int(top['matching_customers'])} customers",
                "column": "duplicate_score",
                "group_column": "behavior_fingerprint",
                "top_group": "shared_behavior_pattern",
                "top_value": float(top["duplicate_score"]),
                "effect_size": float(top["matching_customers"]),
                "confidence": "medium",
            })
            return stories
        if {"review_anomaly_score", "repeated_score_share", "review_count"}.issubset(first.keys()):
            entity_key = next((key for key in keys if key not in {"review_anomaly_score", "repeated_score_share", "review_count", "perfect_review_rate", "low_review_rate", "review_std"}), "entity")
            top = rows[0]
            stories.append({
                "type": "anomaly_signal",
                "insight": f"{top[entity_key]} has the strongest review-pattern anomaly signal with {round(float(top['repeated_score_share']) * 100, 2)}% repeated score concentration",
                "column": "review_anomaly_score",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top["review_anomaly_score"]),
                "effect_size": float(top["repeated_score_share"]),
                "confidence": "medium",
            })
            return stories
        if {"geo_anomaly_score", "total_orders"}.issubset(first.keys()):
            entity_key = next((key for key in keys if key not in {"geo_anomaly_score", "total_orders", "avg_value", "avg_review", "avg_freight"}), "geography")
            top = rows[0]
            stories.append({
                "type": "anomaly_signal",
                "insight": f"{top[entity_key]} has the strongest geographic anomaly signal",
                "column": "geo_anomaly_score",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top["geo_anomaly_score"]),
                "effect_size": float(top["total_orders"]),
                "confidence": "medium",
            })
            return stories
        if {"underperformance_score", "total_orders"}.issubset(first.keys()):
            entity_key = next((key for key in keys if key not in {"underperformance_score", "total_orders", "avg_delay_days", "avg_review", "avg_freight"}), "entity")
            top = rows[0]
            stories.append({
                "type": "operational_risk_score",
                "insight": f"{top[entity_key]} has the strongest operational risk signal",
                "column": "underperformance_score",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top["underperformance_score"]),
                "effect_size": float(top.get("avg_delay_days") or top.get("avg_freight") or 0.0),
                "confidence": "medium",
            })
            return stories
        if {"failure_rate", "affected_orders", "total_orders"}.issubset(first.keys()):
            entity_key = next((key for key in keys if key not in {"failure_rate", "affected_orders", "total_orders"}), "entity")
            top = rows[0]
            stories.append({
                "type": "grouped_numeric",
                "insight": f"{top[entity_key]} has the highest supported failure rate",
                "column": "smoothed_failure_rate" if "smoothed_failure_rate" in first else "failure_rate",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top.get("smoothed_failure_rate", top["failure_rate"])),
                "effect_size": float(top.get("smoothed_failure_rate", top["failure_rate"])),
                "confidence": "medium",
            })
            return stories
        if {"cohort", "repeat_rate", "repeat_customers", "total_customers"}.issubset(first.keys()):
            top = rows[0]
            rank_value = top.get("smoothed_repeat_rate", top["repeat_rate"])
            stories.append({
                "type": "cohort_retention",
                "insight": f"{top['cohort']} has the strongest cohort retention at {round(float(top['repeat_rate']) * 100, 2)}%, with {int(top['repeat_customers'])} repeat customers out of {int(top['total_customers'])}",
                "column": "smoothed_repeat_rate" if "smoothed_repeat_rate" in top else "repeat_rate",
                "group_column": "cohort",
                "top_group": str(top["cohort"]),
                "top_value": float(rank_value),
                "effect_size": float(top["repeat_rate"]),
                "confidence": "medium",
            })
            return stories
        if {"repeat_rate", "repeat_customers", "total_customers"}.issubset(first.keys()):
            group_key = next((key for key in keys if key not in {"repeat_rate", "repeat_customers", "total_customers"}), "segment")
            ranked = sorted(rows, key=lambda row: row.get("repeat_rate", float("-inf")), reverse=True)
            top = ranked[0]
            stories.append({
                "type": "grouped_numeric",
                "insight": f"{top[group_key]} has the strongest repeat behavior",
                "column": "repeat_rate",
                "group_column": group_key,
                "top_group": str(top[group_key]),
                "top_value": float(top["repeat_rate"]),
                "effect_size": float(top["repeat_rate"]),
                "confidence": "medium",
            })
            return stories
        if {"cohort", "long_term_value", "avg_customer_value"}.issubset(first.keys()):
            top = rows[0]
            stories.append({
                "type": "cohort_value",
                "insight": f"{top['cohort']} is the highest long-term value cohort",
                "column": "long_term_value",
                "group_column": "cohort",
                "top_group": str(top["cohort"]),
                "top_value": float(top["long_term_value"]),
                "effect_size": float(top.get("avg_customer_value", 0.0) or 0.0),
                "confidence": "medium",
            })
            return stories
        if {"period", "value", "loyalty_rate"}.issubset(first.keys()):
            valid_loyalty = [row for row in rows if row.get("loyalty_rate") is not None]
            if valid_loyalty:
                latest = valid_loyalty[-1]
                stories.append({
                    "type": "summary_numeric",
                    "insight": f"Latest observed loyalty rate is {round(float(latest['loyalty_rate']) * 100, 2)}%",
                    "column": "loyalty_rate",
                    "mean": float(latest["loyalty_rate"]),
                    "median": float(latest["loyalty_rate"]),
                    "min": float(latest["loyalty_rate"]),
                    "max": float(latest["loyalty_rate"]),
                    "confidence": "medium",
                })
                return stories
        if payload.get("ranking_sort") in {"asc", "desc"} and len(keys) >= 2:
            group_key, value_key = keys[0], keys[1]
            ascending = payload.get("ranking_sort") == "asc"
            top = rows[0]
            step_columns = " ".join(
                str(step.get("column", ""))
                for step in payload.get("steps", [])
                if isinstance(step, dict)
            )
            quality_context = any(
                token in f"{value_key} {step_columns}".lower()
                for token in ["review", "rating", "score", "quality"]
            )
            if ascending:
                if quality_context:
                    insight = f"{top[group_key]} is among the lowest-rated {group_key} values"
                else:
                    insight = f"{top[group_key]} is among the lowest {value_key} {group_key} values"
            else:
                if quality_context:
                    insight = f"{top[group_key]} is among the highest-rated {group_key} values"
                else:
                    insight = f"{top[group_key]} leads on {value_key}"
                if value_key == "delay_days":
                    insight = f"{top[group_key]} shows the heaviest supported delivery delay burden"
            stories.append({
                "type": "grouped_numeric",
                "insight": insight,
                "column": value_key,
                "group_column": group_key,
                "top_group": str(top[group_key]),
                "top_value": float(top[value_key]),
                "effect_size": float(top[value_key]),
                "confidence": "medium",
            })
            return stories
        if {"period", "value", "spike_score"}.issubset(first.keys()):
            top = rows[0]
            stories.append({
                "type": "temporal_spike",
                "insight": f"{top['period']} shows the strongest temporal spike",
                "column": "spike_score",
                "group_column": "period",
                "top_group": str(top["period"]),
                "top_value": float(top["spike_score"]),
                "effect_size": float(top["spike_score"]),
                "confidence": "medium",
            })
            return stories
        if {"event", "event_date", "impact_ratio"}.issubset(first.keys()):
            valid_rows = [row for row in rows if row.get("impact_ratio") is not None]
            if valid_rows:
                top = sorted(valid_rows, key=lambda row: abs(float(row.get("impact_ratio") or 0.0)), reverse=True)[0]
                stories.append({
                    "type": "event_impact",
                    "insight": f"{top['event']} around {top['event_date']} changed the period metric by about {round(float(top['impact_ratio']) * 100, 2)}% versus baseline",
                    "column": "impact_ratio",
                    "group_column": "event_date",
                    "top_group": str(top["event_date"]),
                    "top_value": float(top["impact_ratio"]),
                    "effect_size": float(top["impact_ratio"]),
                    "confidence": "medium",
                })
                return stories
        if {"period", "new_customers"}.issubset(first.keys()):
            latest = rows[-1]
            stories.append({
                "type": "customer_acquisition_trend",
                "insight": f"{latest['period']} added {int(latest['new_customers'])} new customers",
                "column": "new_customers",
                "group_column": "period",
                "top_group": str(latest["period"]),
                "top_value": float(latest["new_customers"]),
                "effect_size": float(latest["new_customers"]),
                "confidence": "medium",
            })
            return stories
        if {"driver", "driver_type", "score", "low_count"}.issubset(first.keys()):
            top = rows[0]
            summary = payload.get("summary", {}) or {}
            threshold = summary.get("threshold")
            threshold_text = f" at or below {round(float(threshold), 2)}" if threshold is not None else ""
            stories.append({
                "type": "low_outcome_driver",
                "insight": f"{top['driver']} is the strongest observed driver of low {summary.get('outcome_column', 'outcome')} cases{threshold_text}",
                "column": summary.get("outcome_column", "outcome"),
                "driver": str(top["driver"]),
                "driver_type": str(top["driver_type"]),
                "top_value": float(top["score"]),
                "effect_size": float(top["score"]),
                "confidence": "medium",
            })
            return stories
        if {"primary_value", "secondary_value", "contrast_score"}.issubset(first.keys()):
            top = rows[0]
            entity_key = next((key for key in keys if key not in {"primary_value", "secondary_value", "contrast_score"}), "segment")
            pattern = payload.get("contrast_pattern")
            if pattern == "high_low":
                insight = f"{top[entity_key]} combines strong volume with comparatively weak revenue"
            elif pattern == "low_high":
                insight = f"{top[entity_key]} combines lower volume with premium pricing"
            else:
                insight = f"{top[entity_key]} best matches the requested volume-versus-value contrast"
            stories.append({
                "type": "grouped_numeric",
                "insight": insight,
                "column": "contrast_score",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top["contrast_score"]),
                "effect_size": float(top["contrast_score"]),
                "confidence": "medium",
            })
            return stories
        elif {"latest_growth_rate", "average_growth_rate"}.issubset(first.keys()):
            entity_key = next((key for key in keys if key not in {"latest_growth_rate", "average_growth_rate", "periods"}), "segment")
            top = rows[0]
            if payload.get("growth_sort") == "asc":
                insight = f"{top[entity_key]} shows the weakest growth pattern"
            else:
                insight = f"{top[entity_key]} shows the strongest recent growth pattern"
            stories.append({
                "type": "grouped_numeric",
                "insight": insight,
                "column": "average_growth_rate",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top["average_growth_rate"]),
                "effect_size": float(top["average_growth_rate"]),
                "confidence": "medium",
            })
            return stories
        elif {"seasonality_score", "peak_month"}.issubset(first.keys()):
            entity_key = next((key for key in keys if key not in {"seasonality_score", "peak_month"}), "segment")
            top = rows[0]
            stories.append({
                "type": "seasonal_demand_signal",
                "insight": f"{top[entity_key]} appears most seasonal, peaking around month {top['peak_month']}",
                "column": "seasonality_score",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top["seasonality_score"]),
                "effect_size": float(top["seasonality_score"]),
                "confidence": "medium",
            })
            return stories
        elif {"top_child_share", "concentration_score", "top_child"}.issubset(first.keys()):
            entity_key = next((key for key in keys if key not in {"top_child_share", "concentration_score", "top_child"}), "segment")
            top = rows[0]
            stories.append({
                "type": "grouped_numeric",
                "insight": f"{top[entity_key]} is highly concentrated in {top['top_child']}",
                "column": "concentration_score",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top["concentration_score"]),
                "effect_size": float(top["top_child_share"]),
                "confidence": "medium",
            })
            return stories
        elif {"group", "mean_value", "count"}.issubset(first.keys()):
            ranked = sorted(rows, key=lambda row: float(row.get("mean_value") or float("-inf")), reverse=True)
            top = ranked[0]
            bottom = ranked[-1]
            stories.append({
                "type": "grouped_numeric",
                "insight": f"{top['group']} shows higher average value than {bottom['group']}",
                "column": "mean_value",
                "group_column": "group",
                "top_group": str(top["group"]),
                "bottom_group": str(bottom["group"]),
                "top_value": float(top["mean_value"]),
                "bottom_value": float(bottom["mean_value"]),
                "effect_size": float(top["mean_value"]) - float(bottom["mean_value"]),
                "confidence": "medium",
            })
            return stories
        elif {"dominant_share", "preference_strength"}.issubset(first.keys()) and (
            "dominant_category_value" in first or "dominant_payment_type" in first
        ):
            dominant_key = "dominant_category_value" if "dominant_category_value" in first else "dominant_payment_type"
            entity_key = next((key for key in keys if key not in {"dominant_category_value", "dominant_payment_type", "dominant_share", "preference_strength", "total_orders"} and not key.endswith("_share")), "entity")
            top = rows[0]
            stories.append({
                "type": "grouped_numeric",
                "insight": f"{top[entity_key]} shows the strongest preference for {top[dominant_key]}",
                "column": "preference_strength",
                "group_column": entity_key,
                "top_group": str(top[entity_key]),
                "top_value": float(top["preference_strength"]),
                "effect_size": float(top["dominant_share"]),
                "confidence": "medium",
            })
            return stories
        elif {"left_value", "right_value"}.issubset(first.keys()):
            stories.append({
                "type": "summary_numeric",
                "insight": "Grouped relationship computed across the resolved segments.",
                "column": "grouped_relationship",
                "mean": float(payload.get("summary", {}).get("correlation") or 0.0),
                "median": float(payload.get("summary", {}).get("correlation") or 0.0),
                "min": float(payload.get("summary", {}).get("correlation") or 0.0),
                "max": float(payload.get("summary", {}).get("correlation") or 0.0),
                "confidence": "medium",
            })
            return stories
        if {"period", "value", "growth_rate"}.issubset(first.keys()):
            valid_growth = [row for row in rows if row.get("growth_rate") is not None]
            if valid_growth:
                latest = valid_growth[-1]
                stories.append({
                    "type": "summary_numeric",
                    "insight": f"Latest period growth is {round(float(latest['growth_rate']) * 100, 2)}%",
                    "column": "growth_rate",
                    "mean": float(latest["growth_rate"]),
                    "median": float(latest["growth_rate"]),
                    "min": float(latest["growth_rate"]),
                    "max": float(latest["growth_rate"]),
                    "confidence": "medium",
                })
        elif len(keys) >= 2:
            group_key, value_key = keys[0], keys[1]
            scored_rows = []
            for row in rows:
                raw_value = row.get(value_key)
                if raw_value is None:
                    continue
                try:
                    numeric_value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                scored_rows.append((row, numeric_value))
            if not scored_rows:
                return stories
            ranked_pairs = sorted(scored_rows, key=lambda item: item[1], reverse=True)
            ascending = payload.get("ranking_sort") == "asc"
            top = ranked_pairs[0][0] if not ascending else ranked_pairs[-1][0]
            bottom = ranked_pairs[-1][0] if not ascending else ranked_pairs[0][0]
            if ascending:
                insight = f"{bottom[group_key]} is among the lowest {value_key} groups and {top[group_key]} among the highest"
                top_group = str(bottom[group_key])
                top_value = float(bottom[value_key])
                bottom_group = str(top[group_key])
                bottom_value = float(top[value_key])
                effect_size = top_value - bottom_value
            else:
                insight = f"{top[group_key]} has the highest {value_key} and {bottom[group_key]} the lowest"
                top_group = str(top[group_key])
                top_value = float(top[value_key])
                bottom_group = str(bottom[group_key])
                bottom_value = float(bottom[value_key])
                effect_size = top_value - bottom_value
            stories.append({
                "type": "grouped_numeric",
                "insight": insight,
                "column": value_key,
                "group_column": group_key,
                "top_group": top_group,
                "bottom_group": bottom_group,
                "top_value": top_value,
                "bottom_value": bottom_value,
                "effect_size": effect_size,
                "confidence": "medium",
            })
    elif value is not None:
        stories.append({
            "type": "summary_numeric",
            "insight": f"The computed {strategy or 'aggregation'} result is {round(float(value), 2)}",
            "column": strategy or "value",
            "mean": float(value),
            "median": float(value),
            "min": float(value),
            "max": float(value),
            "confidence": "medium",
        })

    return stories


def _categorical_stories(
    result: Dict[str, Any],
    question: str,
) -> List[Dict[str, Any]]:
    stories: List[Dict[str, Any]] = []
    categorical_results = result.get("results", {})
    mentioned = _mentioned_columns(question, list(categorical_results.keys()))

    for col, details in categorical_results.items():
        frequency = details.get("frequency", {})
        percentages = details.get("percentages", {})
        mode = details.get("mode")
        rare_categories = details.get("rare_categories", {})
        cross_analysis = details.get("cross_analysis", {})
        numeric_interactions = details.get("numeric_interactions", {})

        if mode is not None and frequency:
            mode_share = percentages.get(str(mode), percentages.get(mode, 0.0))
            stories.append({
                "type": "category_frequency",
                "insight": f"{mode} is the most common value in {col} ({round(mode_share, 2)}%)",
                "column": col,
                "category": str(mode),
                "share": float(mode_share),
                "confidence": "medium",
            })

        if rare_categories:
            labels = list(rare_categories.keys())[:3]
            stories.append({
                "type": "rare_categories",
                "insight": f"{col} contains rare categories: {', '.join(labels)}",
                "column": col,
                "categories": labels,
                "count": len(rare_categories),
                "confidence": "medium",
            })

        for other_col, cross in cross_analysis.items():
            chi_square = cross.get("chi_square", {})
            if chi_square.get("status") == "ok" and chi_square.get("p_value", 1.0) < 0.05:
                stories.append({
                    "type": "categorical_relationship",
                    "insight": f"{col} and {other_col} show a statistically significant relationship",
                    "columns": [col, other_col],
                    "p_value": float(chi_square["p_value"]),
                    "chi2": float(chi_square["chi2"]),
                    "contingency_table": cross.get("contingency_table", {}),
                    "confidence": "high",
                })

        preferred_numeric = _mentioned_columns(question, list(numeric_interactions.keys()))
        numeric_targets = preferred_numeric or list(numeric_interactions.keys())[:1]
        if mentioned and col not in mentioned:
            numeric_targets = []

        for num_col in numeric_targets:
            grouped_stats = numeric_interactions.get(num_col, {})
            if len(grouped_stats) < 2:
                continue

            ranked = sorted(
                grouped_stats.items(),
                key=lambda item: (item[1].get("mean") is not None, item[1].get("mean", float("-inf"))),
                reverse=True,
            )
            if len(ranked) < 2 or ranked[0][1].get("mean") is None or ranked[-1][1].get("mean") is None:
                continue

            top_group, top_stats = ranked[0]
            bottom_group, bottom_stats = ranked[-1]
            diff = float(top_stats["mean"]) - float(bottom_stats["mean"])
            stories.append({
                "type": "grouped_numeric",
                "insight": f"{top_group} has the highest average {num_col} and {bottom_group} the lowest",
                "column": num_col,
                "group_column": col,
                "top_group": str(top_group),
                "bottom_group": str(bottom_group),
                "top_value": float(top_stats["mean"]),
                "bottom_value": float(bottom_stats["mean"]),
                "effect_size": diff,
                "confidence": "medium",
            })

    return stories


def _inferential_story(result: Dict[str, Any], raw_df: pd.DataFrame | None) -> Dict[str, Any] | None:
    analysis_category = result.get("analysis_category")
    hypothesis = result.get("hypothesis_test", {}) or {}
    effect = result.get("effect_size", {}) or {}
    interpretation = result.get("interpretation", {}) or {}
    assumptions = result.get("assumptions", {}) or {}
    decision = hypothesis.get("decision")
    p_value = hypothesis.get("p_value")
    method = result.get("method_selected")
    columns = result.get("columns", [])
    relationship_evidence = result.get("relationship_evidence", {}) or {}
    causal_evidence = result.get("causal_evidence", {}) or relationship_evidence.get("causal_evidence", {}) or {}
    recommended_next_step = result.get("recommended_next_step") or relationship_evidence.get("recommended_next_step")
    bias_risks = result.get("bias_risks", []) or relationship_evidence.get("bias_risks", [])
    confounders = result.get("confounders", []) or relationship_evidence.get("confounders", [])

    if analysis_category == "numeric_relationship" and len(columns) >= 2:
        r_value = effect.get("value")
        if r_value is None:
            r_value = hypothesis.get("test_statistic")
        if r_value is None:
            return None
        direction = "positive" if float(r_value) >= 0 else "negative"
        practical = effect.get("interpretation", "unknown")
        significance = "significant" if decision == "reject_h0" else "not statistically significant"
        return {
            "type": "inferential_relationship",
            "insight": f"{columns[0]} and {columns[1]} show a {practical} {direction} relationship that is {significance}",
            "columns": columns[:2],
            "value": float(r_value),
            "p_value": float(p_value) if p_value is not None else None,
            "method": method,
            "effect_size": effect,
            "assumption_warnings": assumptions.get("warnings", []),
            "confidence": "high" if decision == "reject_h0" else "medium",
            "causal_evidence": causal_evidence,
            "bias_risks": bias_risks,
            "confounders": confounders,
            "recommended_next_step": recommended_next_step,
            "human_summary": result.get("human_summary") or relationship_evidence.get("human_summary"),
        }

    if analysis_category in {"two_group_comparison", "multi_group_comparison"}:
        estimation = result.get("estimation", {}) or {}
        group_stats = estimation.get("group_statistics", {}) or {}
        if len(group_stats) < 2:
            return None
        ranked = sorted(
            group_stats.items(),
            key=lambda item: (item[1].get("mean") is not None, item[1].get("mean", float("-inf"))),
            reverse=True,
        )
        top_group, top_stats = ranked[0]
        bottom_group, bottom_stats = ranked[-1]
        if top_stats.get("mean") is None or bottom_stats.get("mean") is None:
            return None
        practical = effect.get("interpretation", "unknown")
        significance = "significant" if decision == "reject_h0" else "not statistically significant"
        return {
            "type": "inferential_group_difference",
            "insight": f"{top_group} differs most from {bottom_group}; the result is {significance} with a {practical} effect",
            "column": columns[0] if columns else None,
            "group_column": columns[1] if len(columns) > 1 else None,
            "top_group": str(top_group),
            "bottom_group": str(bottom_group),
            "top_value": float(top_stats["mean"]),
            "bottom_value": float(bottom_stats["mean"]),
            "effect_size": effect,
            "p_value": float(p_value) if p_value is not None else None,
            "method": method,
            "assumption_warnings": assumptions.get("warnings", []),
            "confidence": "high" if decision == "reject_h0" else "medium",
            "causal_evidence": causal_evidence,
            "bias_risks": bias_risks,
            "confounders": confounders,
            "recommended_next_step": recommended_next_step,
            "human_summary": result.get("human_summary") or relationship_evidence.get("human_summary"),
        }

    if analysis_category == "categorical_association" and len(columns) >= 2:
        practical = effect.get("interpretation", "unknown")
        significance = "significant" if decision == "reject_h0" else "not statistically significant"
        counts = ((result.get("estimation", {}) or {}).get("counts", {}) or {})
        return {
            "type": "inferential_categorical_association",
            "insight": f"{columns[0]} and {columns[1]} are {significance} with a {practical} association",
            "columns": columns[:2],
            "effect_size": effect,
            "p_value": float(p_value) if p_value is not None else None,
            "method": method,
            "assumption_warnings": assumptions.get("warnings", []),
            "confidence": "high" if decision == "reject_h0" else "medium",
            "contingency_table": counts,
            "causal_evidence": causal_evidence,
            "bias_risks": bias_risks,
            "confounders": confounders,
            "recommended_next_step": recommended_next_step,
            "human_summary": result.get("human_summary") or relationship_evidence.get("human_summary"),
        }

    return None


def _predictive_story(result: Dict[str, Any]) -> Dict[str, Any] | None:
    if result.get("error"):
        return None
    metrics = ((result.get("metrics") or {}).get("values")) or {}
    target = result.get("target_column")
    problem_type = result.get("problem_type")
    confidence = result.get("confidence_level", "low")
    confidence_detail = result.get("confidence", {}) or {}
    if not target or not problem_type:
        return None

    if problem_type == "classification":
        score = metrics.get("f1")
        insight = f"Predictive model for {target} is ready with {result.get('chosen_model')} and F1 around {round(float(score), 3) if score is not None else 'n/a'}"
    elif problem_type == "forecasting":
        score = metrics.get("mape")
        insight = f"Forecasting model for {target} is ready with {result.get('chosen_model')} and MAPE around {round(float(score), 3) if score is not None else 'n/a'}"
    else:
        score = metrics.get("r2")
        insight = f"Predictive model for {target} is ready with {result.get('chosen_model')} and R2 around {round(float(score), 3) if score is not None else 'n/a'}"

    return {
        "type": "predictive_model",
        "insight": insight,
        "column": target,
        "problem_type": problem_type,
        "model_name": result.get("chosen_model"),
        "metrics": metrics,
        "top_drivers": result.get("top_drivers", []),
        "predictions_preview": result.get("predictions_preview", []),
        "limitations": result.get("limitations", []),
        "readiness_warnings": result.get("readiness_warnings", []),
        "validation_summary": result.get("validation_summary", {}),
        "driver_diagnostics": ((result.get("validation_summary") or {}).get("driver_diagnostics")) or {},
        "truthfulness_flags": result.get("truthfulness_flags", []),
        "no_reliable_recommendation": bool(result.get("no_reliable_recommendation")),
        "confidence_assessment": confidence_detail,
        "confidence": confidence,
    }


def _prescriptive_story(result: Dict[str, Any]) -> Dict[str, Any] | None:
    if result.get("error"):
        return None
    actions = result.get("recommended_actions", []) or []
    if not actions:
        return None
    first = actions[0]
    return {
        "type": "prescriptive_action",
        "insight": f"Best next action is {first.get('action')}",
        "column": result.get("based_on_target"),
        "objective": result.get("objective"),
        "estimated_upside": result.get("estimated_upside"),
        "recommended_actions": actions,
        "scenario_summary": result.get("scenario_summary", []),
        "decision_paths": result.get("decision_paths", []),
        "assumptions": result.get("assumptions", []),
        "truthfulness_notes": result.get("truthfulness_notes", []),
        "confidence_assessment": result.get("confidence", {}) or {},
        "operational_confidence_assessment": result.get("operational_confidence", {}) or {},
        "confidence": result.get("confidence_level", "low"),
    }


def evidence_interpreter_node(state: AnalystState) -> AnalystState:
    """
    Interprets raw tool outputs into structured story candidates.
    """
    evidence = state.setdefault("analysis_evidence", {})
    tool_results = evidence.get("tool_results", {})

    if not tool_results:
        print("No tool results found for interpretation.")
        evidence["story_candidates"] = []
        return state

    print("\n=== INTERPRETING STATISTICAL EVIDENCE ===")

    raw_df = state.get("raw_analysis_dataset")
    question = state.get("business_question", "")
    story_candidates: List[Dict[str, Any]] = []

    for key, result in tool_results.items():
        if not isinstance(result, dict):
            continue

        tool_type = result.get("tool")
        story = None

        if tool_type in {"anova", "ttest"} and raw_df is not None:
            story = _group_difference_story(result, raw_df)
            if story:
                story_candidates.append(story)
        elif tool_type == "correlation":
            story = _correlation_story(result)
            if story:
                story_candidates.append(story)
        elif tool_type == "detect_outliers":
            story = _outlier_story(result)
            if story:
                story_candidates.append(story)
        elif tool_type == "summary_statistics":
            story_candidates.extend(_summary_stat_stories(result))
        elif tool_type == "direct_computation":
            story_candidates.extend(_direct_computation_stories(result))
        elif tool_type == "categorical_analysis":
            story_candidates.extend(_categorical_stories(result, question))
        elif tool_type == "inferential_analysis":
            story = _inferential_story(result, raw_df)
            if story:
                guarded_story = _apply_semantic_guardrails(story, result, state)
                if guarded_story:
                    story_candidates.append(guarded_story)
        elif tool_type == "predictive_analysis":
            story = _predictive_story(result)
            if story:
                story_candidates.append(story)
        elif tool_type == "prescriptive_analysis":
            story = _prescriptive_story(result)
            if story:
                story_candidates.append(story)

    evidence["story_candidates"] = story_candidates

    print("\n=== STORY CANDIDATES GENERATED ===")
    for story in story_candidates:
        print("-", story["insight"])

    return state
