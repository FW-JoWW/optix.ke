from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List

from core.analytic_capability import infer_capability_signals
from decision_models import (
    AnalysisOperation,
    AnalysisPlanModel,
    AnalysisAbstractionModel,
    CleaningDecision,
    ComputationPlanModel,
    ComputationStep,
    DecisionEngineOutput,
    DecisionTrace,
)


def _bounded_confidence(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def _info(profile: Dict[str, Any], column: str) -> Dict[str, Any]:
    return (profile.get("columns") or {}).get(column, {})


def _all_columns(profile: Dict[str, Any]) -> List[str]:
    explicit = profile.get("column_names")
    if explicit:
        return explicit
    columns_dict = profile.get("columns") or {}
    if columns_dict:
        return list(columns_dict.keys())
    ordered: List[str] = []
    for group in ("numeric_columns", "categorical_columns", "datetime_columns", "identifier_columns"):
        for column in profile.get(group, []) or []:
            if column not in ordered:
                ordered.append(column)
    unique_counts = profile.get("unique_counts") or {}
    for column in unique_counts.keys():
        if column not in ordered:
            ordered.append(column)
    return ordered


def _role(context: Dict[str, Any], column: str) -> str:
    return (context.get("column_roles") or {}).get(column, "unknown")


def _trace(reason: str, confidence: float, signals: List[str]) -> DecisionTrace:
    return DecisionTrace(
        reason=reason,
        confidence=_bounded_confidence(confidence),
        originating_signals=signals,
    )


def _contains_any(text: str, words: List[str]) -> bool:
    text = (text or "").lower()
    for word in words:
        pattern = rf"(?<!\w){re.escape(word.lower())}(?!\w)"
        if re.search(pattern, text):
            return True
    return False


def _unique_non_empty(values: List[Any]) -> List[Any]:
    seen = set()
    unique: List[Any] = []
    for value in values:
        if not value or value in seen:
            continue
        unique.append(value)
        seen.add(value)
    return unique


COLUMN_PARAMETER_KEYS = {
    "actual_column",
    "child_column",
    "category_column",
    "comparison_column",
    "count_column",
    "customer_column",
    "customer_geo_column",
    "delivery_end_column",
    "delivery_start_column",
    "demand_column",
    "denominator_column",
    "end_column",
    "entity_column",
    "estimated_column",
    "filter_column",
    "freight_column",
    "group_by",
    "group_column",
    "left_column",
    "numerator_column",
    "order_column",
    "parent_column",
    "payment_column",
    "price_column",
    "primary_metric",
    "review_column",
    "review_metric",
    "right_column",
    "secondary_metric",
    "seller_geo_column",
    "start_column",
    "status_column",
    "subtract_column",
    "supply_column",
    "threshold_column",
    "time_column",
    "timestamp_columns",
    "target_columns",
    "value_column",
    "value_metric",
    "within",
}


def _step_parameter_columns(parameters: Dict[str, Any] | None) -> List[str]:
    columns: List[str] = []
    for param_key in COLUMN_PARAMETER_KEYS:
        param_value = (parameters or {}).get(param_key)
        if isinstance(param_value, str):
            columns.append(param_value)
        elif isinstance(param_value, list):
            columns.extend([item for item in param_value if isinstance(item, str)])
    return columns


def _compact_text(value: str) -> str:
    return re.sub(r"[\W_]+", "", str(value or "").lower())


def _matched_profile_values(question: str, profile: Dict[str, Any], column: str | None) -> List[str]:
    if not column:
        return []
    query = (question or "").lower()
    compact_query = _compact_text(query)
    value_patterns = (_info(profile, column).get("value_patterns") or [])
    matches: List[str] = []
    for item in value_patterns:
        raw_value = str(item.get("value", "")).strip()
        if not raw_value:
            continue
        normalized = raw_value.lower()
        phrase = re.sub(r"[_\-]+", " ", normalized)
        compact = _compact_text(normalized)
        if _contains_any(query, [normalized, phrase]) or (compact and compact in compact_query):
            matches.append(normalized)
    return list(dict.fromkeys(matches))


def _categorical_value_contrast_column(
    question: str,
    profile: Dict[str, Any],
    context: Dict[str, Any],
    preferred_columns: List[str] | None = None,
) -> tuple[str | None, List[str]]:
    if not _contains_any(question, ["vs", "versus", "compare", "compared", "prefer", "prefers", "preference", "difference"]):
        return None, []

    preferred = [col for col in preferred_columns or [] if col]
    candidates = list(dict.fromkeys(preferred + _profile_group(profile, "categorical_columns")))
    best_column = None
    best_values: List[str] = []
    best_score = 0.0
    for candidate in candidates:
        if not _categorical_capable(profile, context, candidate):
            continue
        info = _info(profile, candidate)
        if info.get("inferred_type") == "identifier_like":
            continue
        unique_count = int((profile.get("unique_counts") or {}).get(candidate) or info.get("unique_count") or 0)
        if unique_count > 100 and candidate not in preferred:
            continue
        values = _matched_profile_values(question, profile, candidate)
        if len(values) < 2:
            continue
        score = len(values) + (2.0 if candidate in preferred else 0.0)
        if score > best_score:
            best_column = candidate
            best_values = values
            best_score = score
    return best_column, best_values


def _rank_entities_step(
    entity_column: str,
    metric_column: str,
    method: str,
    question: str,
    confidence: float,
    signals: List[str],
    *,
    sort: str = "desc",
    top_n: int = 10,
    intent_type: str = "aggregation",
) -> ComputationStep:
    reason = f"Rank {entity_column} by {method} of {metric_column} using the resolved entity and metric roles."
    return ComputationStep(
        operation="rank_entities",
        column=metric_column,
        parameters={
            "entity_column": entity_column,
            "method": method,
            "top_n": top_n,
            "sort": sort,
            "intent_type": intent_type,
        },
        justification=reason,
        trace=_trace(reason, confidence, signals),
    )


def _segment_order_value_step(
    entity_column: str,
    order_column: str,
    value_column: str,
    confidence: float,
    signals: List[str],
    *,
    sort: str = "desc",
    top_n: int = 10,
) -> ComputationStep:
    reason = f"Compute order-level {value_column} and rank {entity_column} by average order value."
    return ComputationStep(
        operation="segment_order_value",
        column=value_column,
        parameters={
            "entity_column": entity_column,
            "order_column": order_column,
            "value_method": "sum",
            "group_method": "mean",
            "sort": sort,
            "top_n": top_n,
            "intent_type": "aggregation",
        },
        justification=reason,
        trace=_trace(reason, confidence, signals),
    )


def _relative_burden_step(
    entity_column: str,
    numerator_column: str,
    denominator_column: str,
    confidence: float,
    signals: List[str],
    *,
    sort: str = "desc",
    top_n: int = 10,
) -> ComputationStep:
    reason = f"Rank {entity_column} by {numerator_column} relative to {denominator_column}."
    return ComputationStep(
        operation="relative_burden_rank",
        column=numerator_column,
        parameters={
            "entity_column": entity_column,
            "numerator_column": numerator_column,
            "denominator_column": denominator_column,
            "top_n": top_n,
            "sort": sort,
            "intent_type": "comparison",
        },
        justification=reason,
        trace=_trace(reason, confidence, signals),
    )


def _concentration_step(
    parent_column: str,
    child_column: str,
    metric_column: str,
    method: str,
    confidence: float,
    signals: List[str],
    *,
    top_n: int = 10,
) -> ComputationStep:
    reason = f"Measure concentration of {child_column} within {parent_column} using {method} of {metric_column}."
    return ComputationStep(
        operation="concentration_score",
        column=metric_column,
        parameters={
            "parent_column": parent_column,
            "child_column": child_column,
            "method": method,
            "top_n": top_n,
            "intent_type": "composition",
        },
        justification=reason,
        trace=_trace(reason, confidence, signals),
    )


def _best_text_match(query: str, candidates: List[str], required_tokens: List[str] | None = None) -> str | None:
    query_tokens = [token for token in re.findall(r"[a-z0-9_]+", (query or "").lower()) if token]
    best_column = None
    best_score = 0.0
    for candidate in candidates:
        candidate_tokens = [token for token in re.split(r"[_\W]+", str(candidate).lower()) if token]
        if required_tokens and not any(token in candidate_tokens for token in required_tokens):
            continue
        overlap = len(set(query_tokens) & set(candidate_tokens))
        similarity = max((_similarity(candidate, token) for token in query_tokens), default=0.0)
        score = overlap + similarity
        if score > best_score:
            best_score = score
            best_column = candidate
    return best_column


def _profile_group(profile: Dict[str, Any], group_name: str) -> List[str]:
    return list(profile.get(group_name, []) or [])


def _normalized_column_refs(values: List[Any]) -> List[str]:
    normalized: List[str] = []
    for value in values or []:
        if isinstance(value, str):
            normalized.append(value)
        elif isinstance(value, list):
            normalized.extend([item for item in value if isinstance(item, str)])
    return list(dict.fromkeys(normalized))


def _name_contains_any(column: str, tokens: List[str]) -> bool:
    parts = [token for token in re.split(r"[_\W]+", str(column).lower()) if token]
    return any(token in parts for token in tokens)


def _identifier_columns(profile: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
    columns = _all_columns(profile)
    identifiers = []
    for col in columns:
        lower = str(col).lower()
        if _role(context, col) == "identifier" or lower.endswith("_id") or lower == "id":
            identifiers.append(col)
    return identifiers


def _business_time_column(question: str, profile: Dict[str, Any], context: Dict[str, Any], selected_columns: List[str]) -> str | None:
    profile_columns = _all_columns(profile)
    datetime_columns = _profile_group(profile, "datetime_columns")
    candidates = [
        col for col in _all_columns(profile)
        if (_timestamp_like(profile, context, col) or col in datetime_columns)
    ]
    if not candidates:
        candidates = [
            col for col in profile_columns
            if any(token in str(col).lower() for token in ["date", "time", "timestamp", "month", "quarter", "year"])
        ]
    if not candidates:
        return None

    def time_score(column: str) -> float:
        lower = str(column).lower()
        score = 0.0
        if any(token in lower for token in ["purchase", "order"]):
            score += 3.0
        if "approved" in lower:
            score += 2.0
        if any(token in lower for token in ["purchase", "created", "event"]):
            score += 2.0
        if any(token in lower for token in ["timestamp", "date", "time"]):
            score += 1.5
        if any(token in lower for token in ["delivered", "estimated", "shipping", "limit"]):
            score -= 0.8
        lexical = max((_similarity(column, token) for token in re.findall(r"[a-z0-9_]+", (question or "").lower())), default=0.0)
        return score + lexical

    ranked = sorted(candidates, key=time_score, reverse=True)
    event_candidates = [
        col for col in ranked
        if any(token in str(col).lower() for token in ["purchase", "approved", "created", "order"])
        and not any(token in str(col).lower() for token in ["shipping", "limit", "estimated", "delivered"])
    ]
    if event_candidates:
        return event_candidates[0]
    preferred = _best_text_match(question, ranked, ["purchase", "approved", "created", "order", "date", "time", "month", "quarter", "year"])
    return preferred or ranked[0]


def _business_entity_column(question: str, profile: Dict[str, Any], context: Dict[str, Any], entity: str) -> str | None:
    identifiers = _identifier_columns(profile, context)
    required = {
        "order": ["order"],
        "customer": ["customer"],
    }.get(entity, [entity])
    matches = [col for col in identifiers if any(token in re.split(r"[_\W]+", str(col).lower()) for token in required)]
    if not matches:
        return _best_text_match(question, identifiers, required)

    def entity_score(column: str) -> float:
        lower = str(column).lower()
        score = 0.0
        if entity == "order" and lower == "order_id":
            score += 3.0
        if entity == "customer" and "unique" in lower:
            score += 3.0
        if lower.endswith("_id"):
            score += 1.0
        score += max((_similarity(column, token) for token in required), default=0.0)
        return score

    matches.sort(key=entity_score, reverse=True)
    return matches[0]


def _growth_bucket(question: str) -> str:
    if _contains_any(question, ["day", "daily", "per day"]):
        return "day"
    if _contains_any(question, ["week", "weekly", "per week"]):
        return "week"
    if _contains_any(question, ["quarter", "quarterly", "qoq"]):
        return "quarter"
    if _contains_any(question, ["year", "yearly", "annual"]):
        return "year"
    return "month"


def _looks_like_count_question(question: str) -> bool:
    return _contains_any(question, ["how many", "number of", "count", "total orders", "unique customers"])


def _looks_like_share_question(question: str) -> bool:
    return _contains_any(question, ["percentage", "share", "portion", "comes from top"])


def _looks_like_growth_question(question: str) -> bool:
    return _contains_any(question, ["growth rate", "month-over-month", "quarter-over-quarter", "mom", "qoq"])


def _business_segment_column(profile: Dict[str, Any], context: Dict[str, Any], entity: str) -> str | None:
    candidates = _all_columns(profile)
    if not candidates:
        return None

    def score(column: str) -> float:
        lower = str(column).lower()
        value = 0.0
        if entity == "category":
            if "category" in lower:
                value += 4.0
            if "english" in lower:
                value += 1.0
        elif entity == "product":
            if "product" in lower:
                value += 4.0
            if lower == "product_id":
                value += 2.0
        if lower.endswith("_id") and entity != "product":
            value -= 1.0
        return value

    ranked = sorted(candidates, key=score, reverse=True)
    return ranked[0] if score(ranked[0]) > 0 else None


def _business_customer_segment_column(profile: Dict[str, Any], context: Dict[str, Any]) -> str | None:
    candidates = [
        col for col in _all_columns(profile)
        if col in _profile_group(profile, "categorical_columns") or _role(context, col) in {"categorical_feature", "grouping_key"}
    ]
    if not candidates:
        return None

    def score(column: str) -> float:
        lower = str(column).lower()
        value = 0.0
        if "segment" in lower or "cohort" in lower:
            value += 5.0
        if "customer" in lower and any(token in lower for token in ["state", "city", "region", "zip", "prefix"]):
            value += 4.5
        if "state" in lower:
            value += 3.2
        if "region" in lower:
            value += 3.0
        if "city" in lower:
            value += 2.8
        if "zip" in lower or "prefix" in lower:
            value += 2.0
        if lower.endswith("_id"):
            value -= 1.5
        return value

    ranked = sorted(candidates, key=score, reverse=True)
    return ranked[0] if ranked and score(ranked[0]) > 0 else None


def _business_geography_column(profile: Dict[str, Any], context: Dict[str, Any], level: str | None = None, owner: str | None = "customer") -> str | None:
    candidates = _all_columns(profile)
    if not candidates:
        return None

    preferred_levels = [level] if level else ["state", "city", "region", "location", "area", "zip", "prefix"]

    def score(column: str) -> float:
        lower = str(column).lower()
        value = 0.0
        if owner == "customer" and "customer" in lower:
            value += 2.5
        if owner == "seller" and "seller" in lower:
            value += 2.5
        for idx, token in enumerate(preferred_levels):
            if token and token in lower:
                value += 5.0 - (idx * 0.4)
        if "state" in lower:
            value += 1.0
        if "city" in lower:
            value += 0.8
        if "region" in lower:
            value += 0.7
        if any(token in lower for token in ["zip", "prefix"]):
            value += 0.4
        if lower.endswith("_id"):
            value -= 1.5
        return value

    ranked = sorted(candidates, key=score, reverse=True)
    return ranked[0] if ranked and score(ranked[0]) > 0 else None


def _business_entity_metric(profile: Dict[str, Any], context: Dict[str, Any], owner: str) -> str | None:
    identifiers = _identifier_columns(profile, context)
    preferred = []
    for col in identifiers:
        lower = str(col).lower()
        if owner == "seller" and "seller" in lower:
            preferred.append(col)
        if owner == "customer" and "customer" in lower:
            preferred.append(col)
    if preferred:
        preferred.sort(key=lambda col: (str(col).lower().endswith("_id"), "unique" in str(col).lower()), reverse=True)
        return preferred[0]
    return None


def _business_review_metric(profile: Dict[str, Any], context: Dict[str, Any]) -> str | None:
    candidates = [
        col for col in _all_columns(profile)
        if _numeric_capable(profile, context, col)
    ]
    if not candidates:
        return None

    def score(column: str) -> float:
        lower = str(column).lower()
        value = 0.0
        if "review" in lower:
            value += 4.0
        if "score" in lower or "rating" in lower:
            value += 2.5
        return value

    ranked = sorted(candidates, key=score, reverse=True)
    return ranked[0] if ranked and score(ranked[0]) > 0 else None


def build_analysis_abstraction(
    question: str,
    dataset_profile: Dict[str, Any],
    inferred_context: Dict[str, Any],
    relationship_signals: Dict[str, Any],
    user_intent: Dict[str, Any],
    selected_columns: List[str],
) -> AnalysisAbstractionModel:
    signals = infer_capability_signals(question)
    strategies = _determine_strategies(question, user_intent)
    strategy = strategies[0] if strategies else "unknown"
    asks_category = _contains_any(question, ["category", "categories"])
    asks_product = _contains_any(question, ["product", "products", "sku", "item", "items"])
    resolved_roles = user_intent.get("resolved_role_columns", {}) or {}

    time_column = resolved_roles.get("time_column") or _business_time_column(question, dataset_profile, inferred_context, selected_columns)
    order_column = resolved_roles.get("order_column") or _business_entity_column(question, dataset_profile, inferred_context, "order")
    customer_column = resolved_roles.get("customer_column") or _business_entity_column(question, dataset_profile, inferred_context, "customer")
    category_column = _business_segment_column(dataset_profile, inferred_context, "category")
    product_column = _business_segment_column(dataset_profile, inferred_context, "product")
    customer_segment_column = _business_customer_segment_column(dataset_profile, inferred_context)
    customer_geo_column = _business_geography_column(dataset_profile, inferred_context, signals.get("geography_level"), owner="customer")
    seller_geo_column = _business_geography_column(dataset_profile, inferred_context, signals.get("geography_level"), owner="seller")
    geography_column = customer_geo_column or seller_geo_column or _business_geography_column(dataset_profile, inferred_context, None, owner=None)
    seller_column = resolved_roles.get("seller_column") or _business_entity_metric(dataset_profile, inferred_context, "seller")
    payment_type_column = resolved_roles.get("payment_type_column") or _business_payment_type_column(dataset_profile, inferred_context)
    installments_metric = resolved_roles.get("installments_metric") or _business_installments_metric(dataset_profile, inferred_context)
    review_metric = _business_review_metric(dataset_profile, inferred_context)
    purchase_column = resolved_roles.get("purchase_column") or _best_text_match(question + " purchase approved order", _all_columns(dataset_profile), ["purchase", "approved", "order"])
    estimated_delivery_column = resolved_roles.get("estimated_delivery_column") or next(
        (col for col in _profile_group(dataset_profile, "datetime_columns") if "estimated" in str(col).lower()),
        None,
    )
    status_column = resolved_roles.get("status_column") or next(
        (col for col in _profile_group(dataset_profile, "categorical_columns") if "status" in str(col).lower()),
        None,
    )
    focus_dimension = resolved_roles.get("focus_dimension")
    focus_owner = resolved_roles.get("owner")
    if focus_owner == "geography":
        focus_dimension = None
    business_metric_candidates = _preferred_business_metrics(question, dataset_profile, inferred_context, relationship_signals)
    primary_metric = _select_metric_columns(
        question,
        user_intent,
        selected_columns,
        dataset_profile,
        inferred_context,
        relationship_signals,
        "aggregation",
    )
    measure = (business_metric_candidates[:1] or primary_metric[:1])
    dimensions: List[str] = []
    comparison_type = None
    temporal_behavior = None
    statistical_operation = None
    aggregation = None
    presentation_strategy = "scalar"
    family = "general_analytics"
    confidence = 0.74
    notes: List[str] = []

    if focus_dimension and focus_owner not in {"geography", "customer"} and not signals.get("asks_repeat"):
        family = "entity_performance"
        dimensions.append(focus_dimension)
        if signals.get("asks_growth") and time_column:
            dimensions.append(time_column)
            resolved_revenue_metric = resolved_roles.get("revenue_metric")
            measure = [resolved_revenue_metric or business_metric_candidates[0]] if (signals.get("asks_revenue") and (resolved_revenue_metric or business_metric_candidates)) else [order_column or customer_column]
            temporal_behavior = "period_over_period_growth"
            statistical_operation = "segment_growth_rank"
            presentation_strategy = "ranked_time_segment_table"
        elif _contains_any(question, ["high revenue but poor reviews"]) and review_metric:
            resolved_revenue_metric = resolved_roles.get("revenue_metric")
            measure = [col for col in [resolved_revenue_metric or (business_metric_candidates[0] if business_metric_candidates else None), review_metric] if col]
            comparison_type = "revenue_vs_quality_contrast"
            statistical_operation = "segment_contrast"
            presentation_strategy = "contrast_ranking"
        elif _contains_any(question, ["average item price by", "average price by"]) and (resolved_roles.get("price_metric") or resolved_roles.get("revenue_metric")):
            measure = [resolved_roles.get("price_metric") or resolved_roles.get("revenue_metric")]
            aggregation = "mean"
            statistical_operation = "rank_entities"
            presentation_strategy = "ranked_segment_table"
        elif _contains_any(question, ["overpriced relative to reviews", "overpriced"]) and review_metric and (resolved_roles.get("price_metric") or resolved_roles.get("revenue_metric")):
            measure = [col for col in [resolved_roles.get("price_metric") or resolved_roles.get("revenue_metric"), review_metric] if col]
            comparison_type = "value_vs_quality_contrast"
            statistical_operation = "segment_contrast"
            presentation_strategy = "contrast_ranking"
        elif _contains_any(question, ["price elasticity signals", "elasticity"]) and time_column and order_column and (resolved_roles.get("price_metric") or resolved_roles.get("revenue_metric")):
            measure = [col for col in [resolved_roles.get("price_metric") or resolved_roles.get("revenue_metric"), order_column] if col]
            comparison_type = "price_to_demand_relationship"
            temporal_behavior = "price_demand_over_time"
            statistical_operation = "elasticity_proxy_score"
            presentation_strategy = "relationship_summary"
        elif _contains_any(question, ["could support price increases", "price increase", "price increases"]):
            measure = [col for col in [resolved_roles.get("price_metric") or resolved_roles.get("revenue_metric"), review_metric, order_column] if col]
            comparison_type = "premium_potential"
            statistical_operation = "premium_potential_score"
            presentation_strategy = "ranked_segment_table"
        elif _contains_any(question, ["price wars", "price war"]) and (resolved_roles.get("price_metric") or resolved_roles.get("revenue_metric")):
            measure = [col for col in [resolved_roles.get("price_metric") or resolved_roles.get("revenue_metric"), seller_column or focus_dimension] if col]
            comparison_type = "price_competition_pressure"
            statistical_operation = "price_competition_score"
            presentation_strategy = "ranked_segment_table"
        elif _contains_any(question, ["dominate certain categories", "risky dependencies"]) and category_column:
            dimensions.append(category_column)
            resolved_revenue_metric = resolved_roles.get("revenue_metric")
            measure = [resolved_revenue_metric or business_metric_candidates[0]] if (signals.get("asks_revenue") and (resolved_revenue_metric or business_metric_candidates)) else [order_column or customer_column]
            statistical_operation = "concentration_score"
            presentation_strategy = "concentration_table"
        elif _contains_any(question, ["freight cost relative to price", "relative to price", "freight burden", "inefficient to ship"]) and (resolved_roles.get("freight_metric") or _best_text_match(question + " freight shipping", _all_columns(dataset_profile), ["freight", "shipping"])) and (resolved_roles.get("price_metric") or resolved_roles.get("revenue_metric") or _best_text_match(question + " price pricing value revenue", _all_columns(dataset_profile), ["price", "value", "payment", "revenue"])):
            measure = [
                resolved_roles.get("freight_metric") or _best_text_match(question + " freight shipping", _all_columns(dataset_profile), ["freight", "shipping"]),
                resolved_roles.get("price_metric") or resolved_roles.get("revenue_metric") or _best_text_match(question + " price pricing value revenue", _all_columns(dataset_profile), ["price", "value", "payment", "revenue"]),
            ]
            comparison_type = "relative_burden"
            statistical_operation = "relative_burden_rank"
            presentation_strategy = "ranked_segment_table"
        elif _contains_any(question, ["freight expenses", "freight cost", "shipping expenses", "shipping cost", "cost most to deliver", "cost to deliver", "expensive to deliver"]) and (resolved_roles.get("freight_metric") or _best_text_match(question + " freight shipping", _all_columns(dataset_profile), ["freight", "shipping"])):
            measure = [resolved_roles.get("freight_metric") or _best_text_match(question + " freight shipping", _all_columns(dataset_profile), ["freight", "shipping"])]
            aggregation = "sum"
            statistical_operation = "rank_entities"
            presentation_strategy = "ranked_segment_table"
        elif signals.get("asks_delivery"):
            family = "logistics_performance"
            measure = [col for col in [_best_text_match(question + " purchase approved", _all_columns(dataset_profile), ["purchase", "approved", "order"]), _best_text_match(question + " delivered delivery", _all_columns(dataset_profile), ["delivered", "delivery"])] if col]
            aggregation = "elapsed_time_mean"
            statistical_operation = "delay_burden_rank"
            presentation_strategy = "ranked_segment_table"
        elif signals.get("asks_quality") and review_metric:
            measure = [review_metric]
            aggregation = "mean"
            statistical_operation = "rank_entities"
            presentation_strategy = "ranked_segment_table"
        elif _contains_any(question, ["freight cost relative to price", "relative to price"]) and (resolved_roles.get("freight_metric") or _best_text_match(question + " freight shipping", _all_columns(dataset_profile), ["freight", "shipping"])) and (resolved_roles.get("price_metric") or _best_text_match(question + " price pricing", _all_columns(dataset_profile), ["price"])):
            measure = [resolved_roles.get("freight_metric") or _best_text_match(question + " freight shipping", _all_columns(dataset_profile), ["freight", "shipping"]), resolved_roles.get("price_metric") or _best_text_match(question + " price pricing", _all_columns(dataset_profile), ["price"])]
            comparison_type = "relative_burden"
            statistical_operation = "relative_burden_rank"
            presentation_strategy = "ranked_segment_table"
        elif _contains_any(question, ["larger/heavier", "heavier", "larger"]) and _contains_any(question, ["delay", "delays"]):
            measure = [col for col in (resolved_roles.get("size_metrics") or [])[:4]]
            comparison_type = "size_vs_delay_relationship"
            statistical_operation = "derived_delay_relationship"
            presentation_strategy = "relationship_summary"
        elif _contains_any(question, ["bundled in same order", "bundled", "same order"]) and order_column:
            measure = [order_column]
            comparison_type = "cooccurrence"
            statistical_operation = "basket_cooccurrence"
            presentation_strategy = "pair_ranking"
        elif _contains_any(question, ["premium pricing potential", "premium pricing"]):
            measure = [col for col in [resolved_roles.get("price_metric") or resolved_roles.get("revenue_metric"), review_metric, order_column] if col]
            comparison_type = "premium_potential"
            statistical_operation = "premium_potential_score"
            presentation_strategy = "ranked_segment_table"
        elif _contains_any(question, ["underperform despite traffic", "underperform despite orders", "underperform despite traffic/orders"]):
            measure = [col for col in [order_column, resolved_roles.get("revenue_metric") or (business_metric_candidates[0] if business_metric_candidates else None)] if col]
            comparison_type = "demand_vs_value_contrast"
            statistical_operation = "segment_contrast"
            presentation_strategy = "contrast_ranking"
        elif _contains_any(question, ["cancel", "cancelled", "canceled"]) and status_column and order_column:
            measure = [order_column, status_column]
            aggregation = "filtered_distinct_count"
            statistical_operation = "filtered_rank_entities"
            presentation_strategy = "ranked_segment_table"
        elif _contains_any(question, ["need intervention", "intervention"]):
            resolved_revenue_metric = resolved_roles.get("revenue_metric") or _best_value_metric(dataset_profile, inferred_context, relationship_signals)
            measure = [col for col in [order_column, resolved_revenue_metric or (business_metric_candidates[0] if business_metric_candidates else None), review_metric, status_column] if col]
            comparison_type = "multi_signal_risk_priority"
            statistical_operation = "entity_intervention_score"
            presentation_strategy = "priority_table"
        else:
            resolved_revenue_metric = resolved_roles.get("revenue_metric")
            measure = [resolved_revenue_metric or business_metric_candidates[0]] if (signals.get("asks_revenue") and (resolved_revenue_metric or business_metric_candidates)) else [order_column or customer_column]
            aggregation = "sum" if signals.get("asks_revenue") else "distinct_count"
            statistical_operation = "rank_entities"
            presentation_strategy = "ranked_segment_table"

    elif signals.get("asks_geography"):
        family = "geographic_analytics"
        if geography_column:
            dimensions.append(geography_column)
        if signals.get("asks_growth") and time_column:
            growth_measure = (
                [business_metric_candidates[0]]
                if business_metric_candidates
                else [order_column or customer_column]
            )
            measure = [col for col in growth_measure if col]
            dimensions.append(time_column)
            temporal_behavior = "period_over_period_growth"
            statistical_operation = "segment_growth_rank"
            presentation_strategy = "ranked_time_segment_table"
        elif signals.get("asks_demand") and signals.get("asks_supply") and seller_column:
            measure = [order_column or customer_column, seller_column]
            comparison_type = "demand_vs_supply_contrast"
            statistical_operation = "segment_contrast"
            presentation_strategy = "contrast_ranking"
        elif signals.get("asks_price") and order_column:
            location_value_measure = (
                [business_metric_candidates[0]]
                if business_metric_candidates
                else measure[:1]
            )
            measure = [col for col in location_value_measure if col]
            dimensions.append(order_column)
            aggregation = "order_sum_then_location_mean"
            statistical_operation = "segment_order_value"
            presentation_strategy = "ranked_segment_table"
        elif signals.get("asks_quality") and review_metric:
            measure = [review_metric]
            aggregation = "mean"
            statistical_operation = "rank_entities"
            presentation_strategy = "ascending_ranked_segment_table"
        elif _contains_any(question, ["expensive", "cost", "serve", "cost most to deliver", "cost to deliver", "expensive to deliver"]):
            freight_metric = _best_text_match(question + " freight serve shipping", _all_columns(dataset_profile), ["freight", "shipping"])
            if freight_metric:
                measure = [freight_metric]
                aggregation = "mean"
                statistical_operation = "rank_entities"
                presentation_strategy = "ranked_segment_table"
        elif signals.get("asks_delivery"):
            family = "logistics_performance"
            delivered_col = _best_text_match(question + " delivered delivery", _all_columns(dataset_profile), ["delivered", "delivery"])
            purchase_col = _best_text_match(question + " purchase approved order", _all_columns(dataset_profile), ["purchase", "approved", "order"])
            measure = [col for col in [purchase_col, delivered_col] if col]
            aggregation = "elapsed_time_mean"
            statistical_operation = "delay_burden_rank"
            presentation_strategy = "ranked_segment_table"
        elif signals.get("asks_share") or signals.get("asks_coverage"):
            measure = [customer_column] if customer_column else measure
            aggregation = "distinct_count"
            statistical_operation = "rank_entities"
            presentation_strategy = "ranked_segment_table"
        else:
            if signals.get("asks_revenue") and business_metric_candidates:
                measure = [business_metric_candidates[0]]
            elif order_column or customer_column:
                measure = [order_column or customer_column]
            aggregation = "sum" if signals.get("asks_revenue") else "distinct_count"
            statistical_operation = "rank_entities"
            presentation_strategy = "ranked_segment_table"

    elif signals.get("asks_repeat"):
        family = "customer_behavior"
        dimensions.extend([col for col in [customer_column, time_column, customer_segment_column or geography_column] if col][:2])
        if _contains_any(question, ["first and second purchase"]):
            measure = [time_column] if time_column else measure
            temporal_behavior = "purchase_sequence_gap"
            statistical_operation = "purchase_gap"
            presentation_strategy = "distribution_summary"
        elif _contains_any(question, ["dormant"]):
            measure = [time_column] if time_column else measure
            temporal_behavior = "recency_inactivity"
            statistical_operation = "dormancy_count"
            presentation_strategy = "scalar_with_threshold"
        elif _contains_any(question, ["loyal", "return more often"]):
            measure = [order_column] if order_column else measure
            temporal_behavior = "repeat_rate_over_time" if time_column else None
            statistical_operation = "loyalty_trend" if time_column else "review_repeat_comparison"
            presentation_strategy = "time_series_summary" if time_column else "group_comparison"
        elif _contains_any(question, ["buy once", "repeat purchase rate"]):
            measure = [order_column] if order_column else measure
            aggregation = "share"
            statistical_operation = "single_purchase_share" if _contains_any(question, ["buy once"]) else "repeat_rate"
            presentation_strategy = "scalar_rate"
        elif _contains_any(question, ["lifetime value"]):
            measure = [business_metric_candidates[0]] if business_metric_candidates else measure
            aggregation = "customer_sum_then_mean"
            statistical_operation = "customer_lifetime_value_proxy"
            presentation_strategy = "scalar_with_examples"
        else:
            measure = [order_column] if order_column else measure
            aggregation = "distinct_count_per_customer"
            statistical_operation = "customer_order_frequency"
            presentation_strategy = "distribution_summary"

    elif asks_category or asks_product:
        family = "sales_performance"
        dimensions.extend([col for col in [category_column if asks_category else None, product_column if asks_product else None, time_column] if col][:2])
        if signals.get("asks_growth") and time_column:
            temporal_behavior = "segment_growth"
            statistical_operation = "segment_growth_rank"
            presentation_strategy = "ranked_time_segment_table"
        elif signals.get("asks_share"):
            statistical_operation = "concentration_score" if _contains_any(question, ["overdependent"]) else "rank_entities"
            presentation_strategy = "contribution_ranking"
        elif signals.get("asks_contrast"):
            comparison_type = "volume_value_contrast"
            statistical_operation = "segment_contrast"
            presentation_strategy = "contrast_ranking"
        else:
            statistical_operation = "rank_entities"
            presentation_strategy = "ranked_segment_table"
            aggregation = "sum" if signals.get("asks_revenue") else "distinct_count"

    elif customer_column or order_column:
        family = "executive_kpi"
        dimensions.extend([col for col in [time_column, customer_column, order_column] if col][:2])
        if _contains_any(question, ["orders are delivered", "% orders are delivered", "% canceled", "% cancelled", "% unavailable", "invoiced but not delivered"]) and not _contains_any(question, ["delivered early", "delivered late", "% delivered early", "% delivered late", "late", "early"]) and status_column and order_column:
            family = "status_analytics"
            measure = [col for col in [status_column, order_column] if col]
            comparison_type = "status_share"
            statistical_operation = "status_share"
            presentation_strategy = "scalar_rate"
        elif _contains_any(question, ["cancellations increasing"]) and status_column and order_column and time_column:
            family = "status_analytics"
            measure = [col for col in [status_column, order_column, time_column] if col]
            temporal_behavior = "status_trend"
            statistical_operation = "status_rate_trend"
            presentation_strategy = "time_series_summary"
        elif _contains_any(question, ["operational issues"]) and status_column and order_column and time_column:
            family = "status_analytics"
            measure = [col for col in [status_column, order_column, time_column] if col]
            comparison_type = "operational_issue_mix"
            statistical_operation = "operational_issue_score"
            presentation_strategy = "ranked_time_segment_table"
        if _contains_any(question, ["average delivery time", "delivery time"]) and resolved_roles.get("purchase_column") and resolved_roles.get("delivered_column"):
            family = "logistics_performance"
            measure = [col for col in [resolved_roles.get("purchase_column"), resolved_roles.get("delivered_column")] if col]
            comparison_type = "delivery_duration"
            statistical_operation = "delivery_duration_summary"
            presentation_strategy = "scalar"
        elif _contains_any(question, ["estimated vs actual delivery gap", "delivery gap", "estimated vs actual"]) and resolved_roles.get("delivered_column") and estimated_delivery_column:
            family = "logistics_performance"
            measure = [col for col in [resolved_roles.get("delivered_column"), estimated_delivery_column] if col]
            comparison_type = "estimated_actual_gap"
            statistical_operation = "delivery_gap_summary"
            presentation_strategy = "scalar"
        elif _contains_any(question, ["delivered late", "delivered early", "late", "early"]) and resolved_roles.get("delivered_column") and estimated_delivery_column and order_column:
            family = "logistics_performance"
            measure = [col for col in [resolved_roles.get("delivered_column"), estimated_delivery_column, order_column] if col]
            comparison_type = "delivery_timing_share"
            statistical_operation = "delivery_timing_share"
            presentation_strategy = "scalar_rate"
        elif _contains_any(question, ["delays improving over time", "improving over time"]) and resolved_roles.get("purchase_column") and resolved_roles.get("delivered_column") and time_column:
            family = "logistics_performance"
            measure = [col for col in [resolved_roles.get("purchase_column"), resolved_roles.get("delivered_column"), time_column] if col]
            temporal_behavior = "delay_over_time"
            statistical_operation = "delay_trend"
            presentation_strategy = "time_series_summary"
        elif _contains_any(question, ["delivery speed impact ratings", "delivery speed impact", "delivery speed affect ratings"]) and review_metric and resolved_roles.get("purchase_column") and resolved_roles.get("delivered_column"):
            family = "logistics_performance"
            measure = [col for col in [resolved_roles.get("purchase_column"), resolved_roles.get("delivered_column"), review_metric] if col]
            comparison_type = "delay_to_quality_relationship"
            statistical_operation = "delay_quality_relationship"
            presentation_strategy = "relationship_summary"
        elif _contains_any(question, ["shipping distance affect cancellations", "distance affect cancellations", "distance affect cancellation"]) and status_column and resolved_roles.get("seller_geo_column") and resolved_roles.get("customer_geo_column"):
            family = "logistics_performance"
            measure = [col for col in [resolved_roles.get("seller_geo_column"), resolved_roles.get("customer_geo_column"), status_column, order_column] if col]
            comparison_type = "distance_proxy_to_cancellation"
            statistical_operation = "distance_proxy_cancellation_relationship"
            presentation_strategy = "risk_summary"
        if _contains_any(question, ["warehouse redistribution reduce costs", "redistribution reduce costs", "reduce costs"]):
            family = "cost_efficiency"
            measure = [col for col in [resolved_roles.get("freight_metric"), order_column or customer_column, seller_column, geography_column] if col]
            comparison_type = "logistics_reallocation_opportunity"
            statistical_operation = "logistics_optimization_opportunity"
            presentation_strategy = "priority_table"
        if _contains_any(question, ["freight cost as % of revenue", "freight as % of revenue", "freight cost as percentage of revenue"]) and resolved_roles.get("freight_metric") and (resolved_roles.get("revenue_metric") or primary_metric):
            family = "cost_efficiency"
            measure = [col for col in [resolved_roles.get("freight_metric"), resolved_roles.get("revenue_metric") or (primary_metric[0] if primary_metric else None)] if col]
            comparison_type = "ratio_of_totals"
            statistical_operation = "ratio_metric"
            presentation_strategy = "scalar_rate"
        elif _contains_any(question, ["total freight cost"]) and resolved_roles.get("freight_metric"):
            family = "cost_efficiency"
            measure = [resolved_roles.get("freight_metric")]
            statistical_operation = "aggregate"
            aggregation = "sum"
            presentation_strategy = "scalar"
        elif _contains_any(question, ["freight cost reduce review scores", "freight charges reduce review scores", "freight review", "shipping cost reduce review"]) and review_metric and resolved_roles.get("freight_metric"):
            family = "cost_efficiency"
            measure = [resolved_roles.get("freight_metric"), review_metric]
            comparison_type = "cost_to_quality_relationship"
            statistical_operation = "pairwise_relationship"
            presentation_strategy = "relationship_summary"
        elif _contains_any(question, ["causing churn", "cause churn", "freight charges causing churn", "freight causing churn"]) and resolved_roles.get("freight_metric"):
            family = "cost_efficiency"
            measure = [col for col in [resolved_roles.get("freight_metric"), customer_column, order_column, time_column] if col]
            comparison_type = "cost_to_retention_risk"
            statistical_operation = "retention_risk_proxy"
            presentation_strategy = "risk_summary"
        elif _contains_any(question, ["warehouse redistribution reduce costs", "redistribution reduce costs", "reduce costs"]) and resolved_roles.get("freight_metric") and geography_column:
            family = "cost_efficiency"
            measure = [col for col in [resolved_roles.get("freight_metric"), order_column or customer_column, seller_column] if col]
            comparison_type = "logistics_reallocation_opportunity"
            statistical_operation = "logistics_optimization_opportunity"
            presentation_strategy = "priority_table"
        if _contains_any(question, ["higher prices correlate with lower ratings", "correlate"]) and review_metric and (resolved_roles.get("price_metric") or primary_metric):
            measure = [col for col in [resolved_roles.get("price_metric") or (primary_metric[0] if primary_metric else None), review_metric] if col]
            comparison_type = "value_to_quality_relationship"
            statistical_operation = "pairwise_relationship"
            presentation_strategy = "relationship_summary"
        elif _contains_any(question, ["discounts likely drive volume spikes", "discount", "volume spikes"]) and time_column and order_column and (resolved_roles.get("price_metric") or primary_metric):
            measure = [col for col in [resolved_roles.get("price_metric") or (primary_metric[0] if primary_metric else None), order_column] if col]
            temporal_behavior = "price_to_demand_over_time"
            statistical_operation = "discount_volume_effect"
            presentation_strategy = "time_series_summary"
        elif _contains_any(question, ["price ranges convert best", "convert best"]) and (resolved_roles.get("price_metric") or primary_metric) and order_column:
            measure = [col for col in [resolved_roles.get("price_metric") or (primary_metric[0] if primary_metric else None), order_column] if col]
            comparison_type = "price_band_demand"
            statistical_operation = "price_band_demand"
            presentation_strategy = "distribution_summary"
        elif signals.get("asks_growth") and time_column:
            measure = [business_metric_candidates[0]] if business_metric_candidates else measure
            temporal_behavior = "period_over_period_growth"
            statistical_operation = "growth_rate"
            presentation_strategy = "time_series_summary"
        elif signals.get("asks_share"):
            measure = [customer_column or order_column] if (customer_column or order_column) else measure
            statistical_operation = "share_of_total"
            presentation_strategy = "scalar_rate"
        elif _contains_any(question, ["average order value", "aov"]):
            measure = [business_metric_candidates[0]] if business_metric_candidates else measure
            aggregation = "order_sum_then_mean"
            statistical_operation = "segment_order_value" if signals.get("asks_geography") else "aggregate"
            presentation_strategy = "scalar_with_examples"
        else:
            if signals.get("asks_revenue") and business_metric_candidates:
                measure = [business_metric_candidates[0]]
            elif _looks_like_count_question(question):
                measure = [customer_column or order_column] if (customer_column or order_column) else measure
            statistical_operation = "aggregate"
            aggregation = "sum" if signals.get("asks_revenue") else "distinct_count"
            presentation_strategy = "scalar"

    if not measure:
        notes.append("No strongly resolved measure column; fallback computation may be required.")
        confidence -= 0.14

    if not dimensions and geography_column:
        dimensions.append(geography_column)
    if not dimensions and selected_columns:
        dimensions.extend(selected_columns[:1])

    justification = (
        f"Resolved the question into the {family} family with dimensions={dimensions}, "
        f"measures={measure}, aggregation={aggregation or 'unspecified'}, "
        f"comparison={comparison_type or 'none'}, temporal_behavior={temporal_behavior or 'none'}, "
        f"statistical_operation={statistical_operation or 'unspecified'}, presentation={presentation_strategy}."
    )
    if notes:
        justification += " " + " ".join(notes)

    return AnalysisAbstractionModel(
        capability_family=family,
        dimensions=[col for col in dimensions if col],
        measures=[col for col in measure if col],
        aggregation=aggregation,
        comparison_type=comparison_type,
        temporal_behavior=temporal_behavior,
        statistical_operation=statistical_operation,
        presentation_strategy=presentation_strategy,
        confidence_score=_bounded_confidence(confidence),
        justification=justification,
    )


def _base_name(column: str) -> str:
    return re.sub(r"_+\d+$", "", str(column).strip().lower())


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _base_name(left), _base_name(right)).ratio()


def _numeric_capable(profile: Dict[str, Any], context: Dict[str, Any], column: str) -> bool:
    info = _info(profile, column)
    return (
        column in _profile_group(profile, "numeric_columns")
        or info.get("inferred_type") == "numeric"
        or info.get("dtype", "").startswith(("int", "float"))
        or info.get("numeric_like_ratio", 0.0) >= 0.65
        or _role(context, column) in {"numeric_measure", "derived_metric"}
    )


def _categorical_capable(profile: Dict[str, Any], context: Dict[str, Any], column: str) -> bool:
    info = _info(profile, column)
    role = _role(context, column)
    if role in {"numeric_measure", "derived_metric"}:
        return False
    return (
        column in _profile_group(profile, "categorical_columns")
        or column in _profile_group(profile, "datetime_columns")
        or info.get("inferred_type") in {"categorical", "datetime"}
        or role in {"categorical_feature", "grouping_key", "timestamp"}
    )


def _timestamp_like(profile: Dict[str, Any], context: Dict[str, Any], column: str) -> bool:
    info = _info(profile, column)
    if column in _profile_group(profile, "numeric_columns") or _role(context, column) in {"numeric_measure", "derived_metric"}:
        return column in _profile_group(profile, "datetime_columns") or _role(context, column) == "timestamp"
    return (
        column in _profile_group(profile, "datetime_columns")
        or _name_contains_any(column, ["timestamp", "date", "time"])
        or
        info.get("inferred_type") == "datetime"
        or info.get("datetime_like_ratio", 0.0) >= 0.65
        or _role(context, column) == "timestamp"
    )


def _metric_score(
    profile: Dict[str, Any],
    context: Dict[str, Any],
    relationships: Dict[str, Any],
    column: str,
) -> float:
    info = _info(profile, column)
    score = 0.45 * (1.0 - info.get("missing_ratio", 0.0)) + 0.35 * info.get("numeric_like_ratio", 0.0)
    role = _role(context, column)
    if role == "numeric_measure":
        score += 0.15
    elif role == "derived_metric":
        score += 0.05
    if column in relationships.get("derived_columns", []):
        score += 0.05
    lower = str(column).lower()
    if any(token in lower for token in ["price", "revenue", "sales", "payment", "amount", "value", "total", "profit", "cost", "freight"]):
        score += 0.18
    if lower.endswith("_id") or lower in {"id", "order_item_id"}:
        score -= 0.35
    if any(token in lower for token in ["zip", "prefix", "count", "qty", "lenght", "weight", "height", "width"]):
        score -= 0.08
    if "installment" in lower:
        score -= 0.28
    return _bounded_confidence(score)


def _preferred_business_metrics(question: str, profile: Dict[str, Any], context: Dict[str, Any], relationships: Dict[str, Any]) -> List[str]:
    candidates = [
        col for col in _all_columns(profile)
        if _numeric_capable(profile, context, col)
    ]
    if not candidates:
        return []

    tokens = re.findall(r"[a-z0-9_]+", (question or "").lower())

    def score(column: str) -> float:
        lower = str(column).lower()
        value = _metric_score(profile, context, relationships, column)
        if _contains_any(question, ["revenue", "sales", "order value", "aov", "spend", "lifetime value"]):
            if any(token in lower for token in ["payment", "revenue", "sales", "amount"]):
                value += 0.6
            elif "price" in lower:
                value += 0.35
            elif "total" in lower and "freight" not in lower:
                value += 0.25
            if "freight" in lower or "cost" in lower:
                value -= 0.18
        if _contains_any(question, ["profit", "cost", "freight"]):
            if any(token in lower for token in ["profit", "cost", "freight", "price", "payment", "margin"]):
                value += 0.2
        if _contains_any(question, ["quality"]):
            if any(token in lower for token in ["review", "score", "rating", "quality"]):
                value += 0.2
        lexical = max((_similarity(column, token) for token in tokens), default=0.0)
        return value + 0.15 * lexical

    return [item[0] for item in sorted(((col, score(col)) for col in candidates), key=lambda item: item[1], reverse=True)]


def _best_value_metric(profile: Dict[str, Any], context: Dict[str, Any], relationships: Dict[str, Any]) -> str | None:
    candidates = _preferred_business_metrics("revenue sales payment amount value total", profile, context, relationships)
    for col in candidates:
        lower = str(col).lower()
        if any(token in lower for token in ["payment", "revenue", "sales", "amount", "value", "total"]):
            return col
    return candidates[0] if candidates else None


def _business_payment_type_column(profile: Dict[str, Any], context: Dict[str, Any]) -> str | None:
    candidates = []
    for col in _all_columns(profile):
        lower = str(col).lower()
        if "payment_type" in lower or "payment_types" in lower:
            candidates.append(col)
            continue
        if "payment" in lower and ("types" in lower or ("type" in lower and _categorical_capable(profile, context, col))):
            candidates.append(col)
    preferred = [col for col in candidates if any(token in str(col).lower() for token in ["payment_type", "payment_types", "types"])]
    return preferred[0] if preferred else (candidates[0] if candidates else None)


def _business_installments_metric(profile: Dict[str, Any], context: Dict[str, Any]) -> str | None:
    candidates = [
        col for col in _all_columns(profile)
        if "installment" in str(col).lower() and _numeric_capable(profile, context, col)
    ]
    preferred = [col for col in candidates if "max" in str(col).lower()]
    return preferred[0] if preferred else (candidates[0] if candidates else None)


def _family_candidates(
    column: str,
    profile: Dict[str, Any],
    context: Dict[str, Any],
) -> List[str]:
    if not column:
        return []
    family = _base_name(column)
    candidates = []
    for candidate in _all_columns(profile):
        if _base_name(candidate) != family:
            continue
        if _numeric_capable(profile, context, candidate):
            candidates.append(candidate)
    return candidates


def _determine_strategy(question: str, intent: Dict[str, Any]) -> str:
    strategies = _determine_strategies(question, intent)
    return strategies[0] if strategies else "unknown"


def _determine_strategies(question: str, intent: Dict[str, Any]) -> List[str]:
    query = (question or "").lower()
    signals = infer_capability_signals(query)
    explicit = [
        str(item).lower()
        for item in (
            intent.get("analytic_intents")
            or [intent.get("analytic_intent") or intent.get("type")]
        )
        if item
    ]
    ordered: List[str] = []

    def add(strategy: str) -> None:
        if strategy and strategy not in ordered:
            ordered.append(strategy)

    if "predictive" in explicit or _contains_any(query, ["predict", "forecast", "estimate", "project", "risk", "what if", "optimize", "scenario"]):
        add("predictive")
    if "relationship" in explicit or _contains_any(query, ["relationship", "correlation", "correlate", "regression", "cause", "causal", "drive", "drives", "driver", "drivers", "reduce", "reduces", "lower", "lowers", "harsher", "harsh", "elasticity"]) or bool(signals.get("asks_relationship")):
        add("relationship")
    if _contains_any(query, ["larger", "heavier", "delay", "delays"]):
        add("relationship")
    if "comparison" in explicit or _contains_any(query, ["compare", "difference", "affect", "impact", "effect", "overpriced", "price war", "price wars", "price increase", "price increases", "failure rate"]) or bool(signals["asks_contrast"]):
        add("comparison")
    if _contains_any(query, ["freight cost relative to price", "relative to price", "underperform", "premium pricing", "need intervention", "discount", "convert best", "dominate certain categories", "risky dependencies", "dependency", "dependence"]):
        add("comparison")
    if any(item in {"composition", "distribution"} for item in explicit) or bool(signals["asks_share"]) or bool(signals["asks_repeat"]) or bool(signals["asks_coverage"]) or bool(signals.get("asks_basket")):
        add("distribution")
    if _contains_any(query, ["basket", "baskets", "bundle", "bundles", "bundled", "bought together", "commonly bundled", "cross-sell", "cross sell", "same order", "recommend", "recommended", "dominate certain categories", "risky dependencies", "dependency", "dependence", "overdependent"]):
        add("distribution")
    if "data_quality" in explicit or _contains_any(query, ["missing", "null", "duplicate rows", "inconsistent", "invalid", "negative", "impossible", "broken", "data quality"]):
        add("data_quality")
    if any(item in {"profiling", "aggregation", "temporal", "extremes"} for item in explicit) or _contains_any(query, ["average", "mean", "median", "summary", "statistics", "sum", "total", "convert best", "delivered", "unavailable", "invoiced"]) or bool(signals["asks_demand"]) or bool(signals["asks_revenue"]) or bool(signals["asks_price"]) or bool(signals["asks_quality"]) or bool(signals["asks_delivery"]) or bool(signals["asks_geography"]):
        add("aggregation")
    if "outliers" in explicit or _contains_any(query, ["outlier", "outliers", "unusual", "anomaly", "anomalies", "suspicious", "fraud", "fake", "duplicate", "duplicates", "rapid", "rapidly", "excessive"]):
        add("outliers")
    if _contains_any(query, ["cancel", "cancelled", "canceled", "cancellations", "returned", "unavailable", "invoiced"]):
        add("aggregation")
    if bool(signals.get("asks_risk")):
        add("comparison")
    if bool(signals["asks_growth"]) or _contains_any(query, ["seasonal", "seasonality", "operational issues", "cancellations increasing", "increasing"]):
        add("temporal")
    return ordered or ["unknown"]


def _aggregation_method(question: str, default: str = "mean") -> str:
    query = (question or "").lower()
    if _contains_any(query, ["sum", "total"]):
        return "sum"
    if _contains_any(query, ["average", "mean"]):
        return "mean"
    if "median" in query:
        return "median"
    if _contains_any(query, ["min", "minimum", "lowest"]):
        return "min"
    if _contains_any(query, ["max", "maximum", "highest"]):
        return "max"
    return default


def _select_group_columns(
    question: str,
    intent: Dict[str, Any],
    selected_columns: List[str],
    profile: Dict[str, Any],
    context: Dict[str, Any],
    strategy: str,
) -> List[str]:
    selected_columns = _normalized_column_refs(selected_columns)
    strategy_columns = [
            col for col in ((intent.get("intent_columns") or {}).get(strategy) or [])
            if col in _all_columns(profile) and _categorical_capable(profile, context, col)
    ]
    if strategy_columns:
        return strategy_columns

    explicit_group_columns = [
        col for col in (intent.get("group_by_columns") or [])
        if col in _all_columns(profile) and _categorical_capable(profile, context, col)
    ]
    if explicit_group_columns:
        return explicit_group_columns

    group_by = intent.get("group_by")
    if group_by in _all_columns(profile) and _categorical_capable(profile, context, group_by):
        return [group_by]

    candidates = [col for col in selected_columns if _categorical_capable(profile, context, col)]
    if not candidates:
        candidates = [
            col for col in _all_columns(profile)
            if _categorical_capable(profile, context, col)
        ]
    if not candidates:
        return []

    if strategy == "relationship":
        explicit = [col for col in selected_columns if col in candidates]
        if len(explicit) >= 2:
            return explicit
        if len(candidates) >= 2:
            return candidates

    if _contains_any(question, ["per day", "per date", "daily", "by day", "over time", "trend"]):
        ts = [col for col in candidates if _timestamp_like(profile, context, col)]
        if ts:
            return ts[:1]

    if _contains_any(question, ["per", "by"]) and any(_timestamp_like(profile, context, col) for col in candidates):
        ts = [col for col in candidates if _timestamp_like(profile, context, col)]
        return ts[:1]

    timestamp_candidates = [col for col in candidates if _timestamp_like(profile, context, col)]
    if timestamp_candidates:
        return timestamp_candidates[:1]

    return candidates[:1]


def _select_metric_columns(
    question: str,
    intent: Dict[str, Any],
    selected_columns: List[str],
    profile: Dict[str, Any],
    context: Dict[str, Any],
    relationships: Dict[str, Any],
    strategy: str,
) -> List[str]:
    selected_columns = _normalized_column_refs(selected_columns)
    business_metric_candidates = _preferred_business_metrics(question, profile, context, relationships)
    strategy_columns = [
        col for col in ((intent.get("intent_columns") or {}).get(strategy) or [])
        if col in _all_columns(profile) and _numeric_capable(profile, context, col)
    ]
    if strategy_columns:
        if strategy == "aggregation" and _contains_any(question, ["summary", "statistics", "describe"]):
            return strategy_columns
        if strategy in {"relationship", "comparison"}:
            return strategy_columns

    explicit_aggregate_columns = [
        col for col in (intent.get("aggregate_columns") or [])
        if col in _all_columns(profile) and _numeric_capable(profile, context, col)
    ]
    if strategy == "aggregation" and explicit_aggregate_columns and _contains_any(question, ["summary", "statistics", "describe"]):
        return explicit_aggregate_columns

    if strategy == "aggregation" and _contains_any(question, ["revenue", "sales", "profit", "freight", "order value", "aov"]):
        if explicit_aggregate_columns:
            return explicit_aggregate_columns
        if business_metric_candidates:
            return business_metric_candidates[:1]

    aggregate_column = intent.get("aggregate_column")
    if aggregate_column in _all_columns(profile):
        family_candidates = _family_candidates(aggregate_column, profile, context)
        if family_candidates:
            family_candidates.sort(
                key=lambda col: _metric_score(profile, context, relationships, col),
                reverse=True,
            )
            if strategy == "aggregation" and explicit_aggregate_columns and family_candidates[0] in explicit_aggregate_columns:
                return explicit_aggregate_columns
            return [family_candidates[0]]
        if _numeric_capable(profile, context, aggregate_column):
            if strategy == "aggregation" and explicit_aggregate_columns:
                return explicit_aggregate_columns
            return [aggregate_column]

    numeric_candidates = [col for col in selected_columns if _numeric_capable(profile, context, col)]
    if not numeric_candidates:
        numeric_candidates = [
            col for col in _all_columns(profile)
            if _numeric_capable(profile, context, col)
        ]
    if not numeric_candidates:
        return []

    best_per_family: Dict[str, str] = {}
    for col in numeric_candidates:
        family = _base_name(col)
        current = best_per_family.get(family)
        if current is None or _metric_score(profile, context, relationships, col) > _metric_score(profile, context, relationships, current):
            best_per_family[family] = col

    tokens = re.findall(r"[a-zA-Z0-9_]+", (question or "").lower())
    scored = []
    for col in best_per_family.values():
        lexical = max((_similarity(col, token) for token in tokens), default=0.0)
        scored.append((col, _metric_score(profile, context, relationships, col) + 0.2 * lexical))

    scored.sort(key=lambda item: item[1], reverse=True)

    if strategy == "relationship":
        return [item[0] for item in scored]

    if strategy == "comparison":
        explicit_numeric = [col for col in selected_columns if col in [item[0] for item in scored]]
        if explicit_numeric:
            return explicit_numeric
        return [item[0] for item in scored[:2]]

    if strategy == "aggregation":
        explicit_numeric = [col for col in selected_columns if col in [item[0] for item in scored]]
        if _contains_any(question, ["summary", "statistics", "describe"]) and explicit_numeric:
            return explicit_numeric
        if _contains_any(question, ["revenue", "sales", "profit", "freight", "order value", "aov"]) and business_metric_candidates:
            return business_metric_candidates[:1]

    return [scored[0][0]] if scored else []


def _decision_from_action(
    action: Dict[str, Any],
    profile: Dict[str, Any],
    context: Dict[str, Any],
    structural_signals: Dict[str, Any],
    relationships: Dict[str, Any],
    constraints: Dict[str, Any],
) -> CleaningDecision:
    target = action.get("column")
    action_type = action.get("action", "leave_unchanged")
    info = _info(profile, target) if target else {}
    role = _role(context, target)
    reasons: List[str] = []
    confidence = 0.45

    if target in structural_signals.get("high_missing_columns", []):
        confidence += 0.1
        reasons.append("high_missing_columns")
    if target in structural_signals.get("mixed_type_columns", []):
        confidence += 0.1
        reasons.append("mixed_type_columns")
    if role != "unknown":
        confidence += 0.1
        reasons.append(f"role:{role}")

    if action_type == "forward_fill":
        if role in {"timestamp", "grouping_key"} or structural_signals.get("repeated_row_blocks"):
            confidence += 0.2
            reasons.append("sequence_consistency")
        if info.get("missing_ratio", 0.0) > 0.5:
            confidence -= 0.2
    elif action_type == "drop_rows":
        confidence -= 0.05
        if info.get("missing_ratio", 0.0) > constraints.get("max_row_loss_ratio", 0.1):
            confidence -= 0.25
            reasons.append("excessive_data_loss_risk")
    elif action_type == "convert_type":
        if info.get("numeric_like_ratio", 0.0) >= 0.8 or info.get("datetime_like_ratio", 0.0) >= 0.8:
            confidence += 0.2
            reasons.append("coercible_type_signal")
    elif action_type == "recompute_if_possible":
        if target in relationships.get("derived_columns", []):
            confidence += 0.25
            reasons.append("relationship_recomputation")
    elif action_type == "leave_unchanged":
        confidence += 0.25
        reasons.append("transformation_minimization")

    blocked = {
        (item.get("action", {}) or {}).get("column"): item.get("reason")
        for item in constraints.get("blocked_actions", [])
    }
    deferred = False
    if target in blocked:
        confidence = 0.0
        deferred = True
        reasons.append(f"blocked:{blocked[target]}")

    if action_type == "drop_rows" and role in {"identifier", "grouping_key", "timestamp"}:
        confidence = 0.0
        deferred = True
        reasons.append("protected_role")

    if action_type == "forward_fill" and role not in {"timestamp", "grouping_key"} and not structural_signals.get("repeated_row_blocks"):
        confidence -= 0.15

    requires_validation = 0.45 <= confidence < 0.75
    if confidence < 0.45:
        deferred = True
        if action_type != "leave_unchanged":
            action_type = "leave_unchanged"

    justification = (
        f"Selected {action_type} for {target or 'dataset'} from observed structure, context inference, and safety constraints."
        if not deferred
        else f"Deferred transformation on {target or 'dataset'} because the evidence did not justify a safe action."
    )

    return CleaningDecision(
        target=target,
        action_type=action_type,
        confidence_score=_bounded_confidence(confidence),
        justification=justification,
        trace=_trace(justification, confidence, reasons or ["context_inference"]),
        requires_validation=requires_validation,
        deferred=deferred,
    )


def build_cleaning_decisions(
    dataset_profile: Dict[str, Any],
    structural_signals: Dict[str, Any],
    inferred_context: Dict[str, Any],
    relationship_signals: Dict[str, Any],
    constraint_rules: Dict[str, Any],
) -> List[CleaningDecision]:
    # Cleaning is confidence-gated: the context layer may recommend an action,
    # but execution is allowed only after we score the evidence and constraints.
    decisions = [
        _decision_from_action(
            action=item,
            profile=dataset_profile,
            context=inferred_context,
            structural_signals=structural_signals,
            relationships=relationship_signals,
            constraints=constraint_rules,
        )
        for item in inferred_context.get("recommended_actions", [])
    ]

    if not decisions:
        decisions.append(
            CleaningDecision(
                target=None,
                action_type="leave_unchanged",
                confidence_score=1.0,
                justification="No safe transformation was justified from the available signals.",
                trace=_trace(
                    "No safe transformation was justified from the available signals.",
                    1.0,
                    ["transformation_minimization"],
                ),
            )
        )
    return decisions


def build_computation_plan(
    dataset_profile: Dict[str, Any],
    structural_signals: Dict[str, Any],
    inferred_context: Dict[str, Any],
    relationship_signals: Dict[str, Any],
    user_intent: Dict[str, Any],
    selected_columns: List[str],
    analysis_abstraction: AnalysisAbstractionModel | None = None,
) -> ComputationPlanModel:
    # Computation-first planning:
    # 1. resolve the minimal computation needed to answer the question
    # 2. validate grouping/metric choices against structure and sparsity
    # 3. leave tool selection to the later analysis-plan stage
    question = user_intent.get("query", "")
    resolved_roles = user_intent.get("resolved_role_columns", {}) or {}
    purchase_column = resolved_roles.get("purchase_column") or _best_text_match(question + " purchase approved order", _all_columns(dataset_profile), ["purchase", "approved", "order"])
    estimated_delivery_column = resolved_roles.get("estimated_delivery_column") or next(
        (col for col in _profile_group(dataset_profile, "datetime_columns") if "estimated" in str(col).lower()),
        None,
    )
    if "warehouse" in question.lower() or "redistribution" in question.lower():
        freight_metric = resolved_roles.get("freight_metric") or _best_text_match(question + " freight shipping", _all_columns(dataset_profile), ["freight", "shipping"])
        customer_column = _business_entity_column(question, dataset_profile, inferred_context, "customer")
        order_column = _business_entity_column(question, dataset_profile, inferred_context, "order")
        seller_column = _business_entity_metric(dataset_profile, inferred_context, "seller")
        state_geo = _best_text_match(
            "customer state region geography",
            _all_columns(dataset_profile),
            ["customer_state", "state", "region"],
        )
        optimization_geo = state_geo or resolved_roles.get("geography_column") or _business_geography_column(dataset_profile, inferred_context, None, owner=None)
        demand_metric = order_column or customer_column
        if freight_metric and demand_metric and optimization_geo:
            reason = f"Score geographic areas on shipping cost, demand, and local seller presence to surface where redistribution or network rebalancing may reduce cost."
            step = ComputationStep(
                operation="logistics_optimization_opportunity",
                column=freight_metric,
                parameters={
                    "entity_column": optimization_geo,
                    "freight_column": freight_metric,
                    "demand_column": demand_metric,
                    "supply_column": seller_column,
                    "top_n": 10,
                    "min_support": 10,
                    "intent_type": "optimization",
                },
                justification=reason,
                trace=_trace(reason, 0.7, ["logistics_opportunity", "cost_reduction"]),
            )
            return ComputationPlanModel(
                steps=[step],
                confidence_score=0.7,
                justification="Built a generic logistics optimization opportunity plan.",
                deferred=False,
            )
    signals = infer_capability_signals(question)
    strategies = _determine_strategies(question, user_intent)
    primary_strategy = strategies[0] if strategies else "unknown"
    steps: List[ComputationStep] = []
    confidence = 0.75
    notes: List[str] = []

    seen_steps = set()

    def _freeze(value: Any):
        if isinstance(value, dict):
            return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
        if isinstance(value, list):
            return tuple(_freeze(item) for item in value)
        return value

    def add_step(step: ComputationStep) -> None:
        key = (
            step.operation,
            step.column,
            tuple(step.columns),
            _freeze(step.parameters or {}),
        )
        if key in seen_steps:
            return
        seen_steps.add(key)
        steps.append(step)

    planning_terms = [
        "targeted", "expansion", "recruited", "dropped", "launched", "warehouses",
        "forecast", "predict", "recommend", "clustering", "lifetime value", "optimize",
    ]
    if primary_strategy == "unknown" and not _contains_any(question, planning_terms):
        return ComputationPlanModel(
            steps=[],
            confidence_score=0.25,
            justification="The analytical intent could not be mapped to a reliable computation plan.",
            deferred=True,
        )

    predictive_target_terms = [
        "forecast", "next month", "demand", "stock", "capacity", "cancellation", "cancellations",
        "review score risk", "bad reviews", "repeat customers", "churn", "recommend", "segment customers",
        "clustering", "lifetime value", "optimize delivery", "delivery promises",
    ]
    if "predictive" in strategies and not _contains_any(question, predictive_target_terms):
        target = user_intent.get("aggregate_column") or (selected_columns[-1] if selected_columns else None)
        reason = f"Build a supervised or forecasting workflow around {target or 'the resolved target'} because the question is predictive or prescriptive."
        add_step(
            ComputationStep(
                operation="predict_target",
                column=target,
                columns=selected_columns,
                justification=reason,
                trace=_trace(reason, 0.72, ["predictive_intent"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.72,
            justification="Built a predictive workflow plan before model selection, preserving deterministic readiness checks and validation.",
            deferred=False,
        )

    all_group_columns: List[str] = []
    all_metric_columns: List[str] = []

    time_column = _business_time_column(question, dataset_profile, inferred_context, selected_columns)
    order_column = _business_entity_column(question, dataset_profile, inferred_context, "order")
    customer_column = _business_entity_column(question, dataset_profile, inferred_context, "customer")
    category_column = _business_segment_column(dataset_profile, inferred_context, "category")
    product_column = _business_segment_column(dataset_profile, inferred_context, "product")
    customer_segment_column = _business_customer_segment_column(dataset_profile, inferred_context)
    customer_geo_column = _business_geography_column(dataset_profile, inferred_context, signals.get("geography_level"), owner="customer")
    seller_geo_column = _business_geography_column(dataset_profile, inferred_context, signals.get("geography_level"), owner="seller")
    fallback_geo_column = _business_geography_column(dataset_profile, inferred_context, None, owner=None)
    geography_column = customer_geo_column or seller_geo_column or fallback_geo_column
    seller_column = _business_entity_metric(dataset_profile, inferred_context, "seller")
    payment_type_column = resolved_roles.get("payment_type_column") or _business_payment_type_column(dataset_profile, inferred_context)
    installments_metric = resolved_roles.get("installments_metric") or _business_installments_metric(dataset_profile, inferred_context)
    review_metric = _business_review_metric(dataset_profile, inferred_context)
    status_column = resolved_roles.get("status_column") or next(
        (col for col in _profile_group(dataset_profile, "categorical_columns") if "status" in str(col).lower()),
        None,
    )
    focus_dimension = user_intent.get("group_by") or resolved_roles.get("focus_dimension")
    bucket = _growth_bucket(question)
    business_metric_candidates = _preferred_business_metrics(
        question,
        dataset_profile,
        inferred_context,
        relationship_signals,
    )
    primary_metric = _select_metric_columns(
        question,
        user_intent,
        selected_columns,
        dataset_profile,
        inferred_context,
        relationship_signals,
        "aggregation",
    )
    if _contains_any(question, ["revenue", "sales", "order value", "aov"]) and business_metric_candidates:
        primary_metric = business_metric_candidates[:1]

    if analysis_abstraction is not None:
        if analysis_abstraction.dimensions:
            notes.append(f"Abstraction dimensions: {', '.join(analysis_abstraction.dimensions)}.")
        if analysis_abstraction.measures:
            notes.append(f"Abstraction measures: {', '.join(analysis_abstraction.measures)}.")
        confidence = _bounded_confidence((confidence + analysis_abstraction.confidence_score) / 2)

    if focus_dimension and focus_dimension not in _all_columns(dataset_profile):
        focus_dimension = None

    resolved_revenue_metric = resolved_roles.get("revenue_metric")
    preferred_value_metric = (
        resolved_revenue_metric
        or _best_text_match(
            question + " revenue payment sales amount total value",
            _all_columns(dataset_profile),
            ["payment", "revenue", "sales", "value", "amount", "total"],
        )
        or next(
            (
                candidate for candidate in business_metric_candidates
                if "installment" not in str(candidate).lower()
                and not any(token in str(candidate).lower() for token in ["freight", "shipping"])
            ),
            business_metric_candidates[0] if business_metric_candidates else None,
        )
        or (primary_metric[0] if primary_metric else None)
    )

    data_quality_terms = ["missing", "null", "duplicate rows", "inconsistent", "invalid", "negative", "impossible", "broken", "data quality"]
    all_profile_columns = _all_columns(dataset_profile)
    if _contains_any(question, data_quality_terms) or (_contains_any(question, ["outlier", "outliers"]) and _contains_any(question, ["payment", "value", "values"])):
        anchor_column = selected_columns[0] if selected_columns else (order_column or (all_profile_columns[0] if all_profile_columns else None))
        timestamp_columns = _profile_group(dataset_profile, "datetime_columns")
        if _contains_any(question, ["missing", "null"]) and anchor_column:
            reason = "Scan all available columns for missing values and rank missingness by severity."
            add_step(
                ComputationStep(
                    operation="missingness_report",
                    column=anchor_column,
                    parameters={"target_columns": all_profile_columns, "intent_type": "data_quality"},
                    justification=reason,
                    trace=_trace(reason, 0.86, ["data_quality", "missingness", "schema_scan"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.86, justification="Built a reusable missingness diagnostic plan.", deferred=False)
        if _contains_any(question, ["duplicate rows"]) and anchor_column:
            reason = "Check exact duplicate rows across the available schema."
            add_step(
                ComputationStep(
                    operation="duplicate_rows_report",
                    column=anchor_column,
                    parameters={"target_columns": all_profile_columns, "top_n": 10, "intent_type": "data_quality"},
                    justification=reason,
                    trace=_trace(reason, 0.84, ["data_quality", "exact_duplicate_rows"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a reusable exact duplicate-row diagnostic plan.", deferred=False)
        if _contains_any(question, ["inconsistent timestamp", "inconsistent timestamps"]) and timestamp_columns and anchor_column:
            reason = "Validate timestamp parseability and chronological ordering across resolved timestamp columns."
            add_step(
                ComputationStep(
                    operation="timestamp_consistency_report",
                    column=anchor_column,
                    parameters={"timestamp_columns": timestamp_columns, "purchase_column": purchase_column, "delivered_column": resolved_roles.get("delivered_column"), "estimated_column": estimated_delivery_column, "intent_type": "data_quality"},
                    justification=reason,
                    trace=_trace(reason, 0.82, ["data_quality", "timestamp_consistency"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a reusable timestamp consistency diagnostic plan.", deferred=False)
        if _contains_any(question, ["negative"]) and (resolved_roles.get("price_metric") or preferred_value_metric):
            value_column = resolved_roles.get("price_metric") or preferred_value_metric
            reason = f"Check {value_column} for values below zero."
            add_step(
                ComputationStep(
                    operation="numeric_validity_check",
                    column=value_column,
                    parameters={"value_column": value_column, "min_allowed": 0, "allow_zero": True, "top_n": 10, "intent_type": "data_quality"},
                    justification=reason,
                    trace=_trace(reason, 0.84, ["data_quality", "numeric_validity", "non_negative"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a reusable non-negative numeric validity plan.", deferred=False)
        if _contains_any(question, ["invalid freight", "freight values"]) and resolved_roles.get("freight_metric"):
            freight_metric = resolved_roles.get("freight_metric")
            reason = f"Check {freight_metric} for invalid numeric values."
            add_step(
                ComputationStep(
                    operation="numeric_validity_check",
                    column=freight_metric,
                    parameters={"value_column": freight_metric, "min_allowed": 0, "allow_zero": True, "top_n": 10, "intent_type": "data_quality"},
                    justification=reason,
                    trace=_trace(reason, 0.84, ["data_quality", "numeric_validity", "freight"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a reusable freight numeric validity plan.", deferred=False)
        if _contains_any(question, ["impossible delivery", "impossible delivery dates"]) and purchase_column and resolved_roles.get("delivered_column"):
            reason = "Check delivery date chronology for impossible delivery records."
            add_step(
                ComputationStep(
                    operation="delivery_date_validity",
                    column=resolved_roles.get("delivered_column"),
                    parameters={"order_column": order_column, "purchase_column": purchase_column, "delivered_column": resolved_roles.get("delivered_column"), "estimated_column": estimated_delivery_column, "top_n": 10, "intent_type": "data_quality"},
                    justification=reason,
                    trace=_trace(reason, 0.86, ["data_quality", "delivery_date_validity", "chronology"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.86, justification="Built a reusable delivery date validity plan.", deferred=False)
        if _contains_any(question, ["broken", "label", "labels", "category labels"]) and category_column:
            reason = f"Check {category_column} for blank labels and normalized label variants."
            add_step(
                ComputationStep(
                    operation="categorical_label_quality",
                    column=category_column,
                    parameters={"category_column": category_column, "top_n": 10, "intent_type": "data_quality"},
                    justification=reason,
                    trace=_trace(reason, 0.8, ["data_quality", "categorical_label_quality"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a reusable categorical label quality plan.", deferred=False)
        if _contains_any(question, ["outlier", "outliers"]) and (preferred_value_metric or resolved_roles.get("revenue_metric")) and order_column:
            value_column = preferred_value_metric or resolved_roles.get("revenue_metric")
            reason = f"Rank order-level outliers in {value_column} using robust distribution thresholds."
            add_step(
                ComputationStep(
                    operation="transaction_value_outlier_rank",
                    column=value_column,
                    parameters={"order_column": order_column, "value_column": value_column, "top_n": 10, "intent_type": "data_quality"},
                    justification=reason,
                    trace=_trace(reason, 0.84, ["data_quality", "numeric_outliers", "payment_value"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a reusable numeric outlier diagnostic plan.", deferred=False)

    if _contains_any(question, ["forecast", "predict", "estimate", "recommend", "segment customers", "clustering", "optimize delivery"]):
        if time_column and preferred_value_metric and _contains_any(question, ["next month revenue", "revenue"]):
            reason = f"Forecast aggregate {preferred_value_metric} by time bucket using recent trend-adjusted history."
            add_step(
                ComputationStep(
                    operation="aggregate_forecast",
                    column=preferred_value_metric,
                    parameters={"time_column": time_column, "metric_column": preferred_value_metric, "method": "sum", "bucket": "month", "top_n": 1, "intent_type": "predictive"},
                    justification=reason,
                    trace=_trace(reason, 0.78, ["aggregate_forecast", "time_bucket", "revenue_metric"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a reusable aggregate revenue forecast plan.", deferred=False)
        if time_column and category_column and order_column and _contains_any(question, ["demand by category"]):
            reason = f"Forecast category demand from recent distinct-order history by {category_column}."
            add_step(
                ComputationStep(
                    operation="aggregate_forecast",
                    column=order_column,
                    parameters={"time_column": time_column, "metric_column": order_column, "segment_column": category_column, "method": "distinct_count", "bucket": "month", "top_n": 10, "intent_type": "predictive"},
                    justification=reason,
                    trace=_trace(reason, 0.78, ["segment_forecast", "category_demand"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a reusable segmented demand forecast plan.", deferred=False)
        if time_column and category_column and order_column and _contains_any(question, ["holiday stock", "stock needs"]):
            reason = f"Estimate stock needs from category seasonality and peak month demand."
            add_step(
                ComputationStep(
                    operation="segment_seasonality",
                    column=order_column,
                    parameters={"entity_column": category_column, "time_column": time_column, "method": "distinct_count", "top_n": 10, "intent_type": "predictive"},
                    justification=reason,
                    trace=_trace(reason, 0.76, ["stock_need_proxy", "seasonality", "category_demand"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.76, justification="Built a reusable seasonal stock-needs proxy plan.", deferred=False)
        if time_column and seller_column and order_column and _contains_any(question, ["seller capacity", "capacity needs"]):
            reason = f"Estimate seller capacity needs from recent and peak order load by {seller_column}."
            add_step(
                ComputationStep(
                    operation="capacity_need_score",
                    column=order_column,
                    parameters={"entity_column": seller_column, "time_column": time_column, "demand_column": order_column, "bucket": "week", "top_n": 10, "intent_type": "predictive"},
                    justification=reason,
                    trace=_trace(reason, 0.78, ["capacity_need", "seller_load", "time_bucket"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a reusable seller capacity-needs plan.", deferred=False)
        if status_column and order_column and _contains_any(question, ["cancellation", "cancellations"]):
            reason = f"Construct a cancellation risk target from {status_column} before predictive modeling."
            add_step(
                ComputationStep(
                    operation="predictive_target_profile",
                    column=status_column,
                    parameters={"target_type": "cancellation", "order_column": order_column, "status_column": status_column, "time_column": time_column, "intent_type": "predictive"},
                    justification=reason,
                    trace=_trace(reason, 0.78, ["target_engineering", "cancellation_target"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a reusable cancellation target-readiness plan.", deferred=False)
        if review_metric and order_column and _contains_any(question, ["review score risk", "bad reviews", "bad review"]):
            reason = f"Construct a bad-review risk target from {review_metric}."
            add_step(
                ComputationStep(
                    operation="predictive_target_profile",
                    column=review_metric,
                    parameters={"target_type": "bad_review", "order_column": order_column, "review_column": review_metric, "time_column": time_column, "intent_type": "predictive"},
                    justification=reason,
                    trace=_trace(reason, 0.78, ["target_engineering", "review_risk_target"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a reusable review-risk target-readiness plan.", deferred=False)
        if customer_column and order_column and _contains_any(question, ["repeat customers"]):
            reason = f"Construct a repeat-customer target from distinct orders per customer."
            add_step(
                ComputationStep(
                    operation="predictive_target_profile",
                    column=order_column,
                    parameters={"target_type": "repeat_customer", "customer_column": customer_column, "order_column": order_column, "time_column": time_column, "intent_type": "predictive"},
                    justification=reason,
                    trace=_trace(reason, 0.78, ["target_engineering", "repeat_customer_target"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a reusable repeat-customer target-readiness plan.", deferred=False)
        if customer_column and order_column and _contains_any(question, ["churn"]):
            reason = f"Construct a churn/inactivity proxy target from distinct orders per customer."
            add_step(
                ComputationStep(
                    operation="predictive_target_profile",
                    column=order_column,
                    parameters={"target_type": "churn", "customer_column": customer_column, "order_column": order_column, "time_column": time_column, "intent_type": "predictive"},
                    justification=reason,
                    trace=_trace(reason, 0.76, ["target_engineering", "churn_proxy_target"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.76, justification="Built a reusable churn target-readiness proxy plan.", deferred=False)
        if order_column and (product_column or category_column) and _contains_any(question, ["recommend"]):
            item_column = product_column or category_column
            reason = f"Find recommendable item associations using order co-occurrence for {item_column}."
            add_step(
                ComputationStep(
                    operation="basket_cooccurrence",
                    column=item_column,
                    parameters={"order_column": order_column, "item_column": item_column, "top_n": 10, "recommendation_mode": True, "intent_type": "composition"},
                    justification=reason,
                    trace=_trace(reason, 0.8, ["recommendation", "basket_association"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a reusable product recommendation association plan.", deferred=False)
        if customer_column and order_column and _contains_any(question, ["segment customers", "clustering"]):
            reason = f"Create customer segments using recency, frequency, and value features."
            add_step(
                ComputationStep(
                    operation="customer_clustering_segments",
                    column=customer_column,
                    parameters={"customer_column": customer_column, "order_column": order_column, "value_column": preferred_value_metric, "time_column": time_column, "intent_type": "predictive"},
                    justification=reason,
                    trace=_trace(reason, 0.78, ["customer_segmentation", "rfm_features", "clustering_proxy"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a reusable customer segmentation plan.", deferred=False)
        if customer_column and preferred_value_metric and _contains_any(question, ["lifetime value"]):
            reason = f"Estimate customer lifetime value by rolling up {preferred_value_metric} at customer level."
            add_step(
                ComputationStep(
                    operation="customer_ltv_estimate",
                    column=preferred_value_metric,
                    parameters={"customer_column": customer_column, "order_column": order_column, "value_column": preferred_value_metric, "top_n": 10, "intent_type": "predictive"},
                    justification=reason,
                    trace=_trace(reason, 0.8, ["ltv_proxy", "customer_rollup"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a reusable customer lifetime value proxy plan.", deferred=False)
        if customer_segment_column and purchase_column and resolved_roles.get("delivered_column") and _contains_any(question, ["optimize delivery", "delivery promises"]):
            reason = f"Estimate delivery-promise buffers by segment from observed delivery duration quantiles."
            add_step(
                ComputationStep(
                    operation="delivery_promise_optimization",
                    column=resolved_roles.get("delivered_column"),
                    parameters={"entity_column": customer_segment_column, "order_column": order_column, "start_column": purchase_column, "end_column": resolved_roles.get("delivered_column"), "estimated_column": estimated_delivery_column, "top_n": 10, "intent_type": "predictive"},
                    justification=reason,
                    trace=_trace(reason, 0.8, ["delivery_promise", "quantile_buffer", "optimization"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a reusable delivery promise optimization plan.", deferred=False)

    strategic_terms = ["targeted", "expansion", "recruited", "dropped", "premium segments", "launched"]
    if _contains_any(question, strategic_terms):
        if geography_column and order_column and _contains_any(question, ["region", "regions", "targeted"]):
            entity = customer_segment_column or geography_column
            mode = "growth"
        elif category_column and _contains_any(question, ["categories", "category", "expansion", "dropped"]):
            entity = category_column
            mode = "drop" if _contains_any(question, ["dropped", "drop", "low-performing"]) else "growth"
        elif seller_column and _contains_any(question, ["seller", "sellers", "recruited"]):
            entity = seller_column
            mode = "recruit"
        elif (customer_segment_column or geography_column) and _contains_any(question, ["premium", "launched"]):
            entity = customer_segment_column or geography_column
            mode = "premium"
        else:
            entity = focus_dimension
            mode = "growth"
        if entity and order_column:
            reason = f"Score {entity} for strategic {mode} opportunity using demand, value, review, and cost signals."
            add_step(
                ComputationStep(
                    operation="strategic_opportunity_score",
                    column=order_column,
                    parameters={"entity_column": entity, "order_column": order_column, "value_column": preferred_value_metric, "review_column": review_metric, "freight_column": resolved_roles.get("freight_metric"), "mode": mode, "top_n": 10, "intent_type": "comparison"},
                    justification=reason,
                    trace=_trace(reason, 0.8, ["strategic_opportunity", mode, "multi_signal_score"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a reusable strategic opportunity scoring plan.", deferred=False)

    temporal_metric = None
    temporal_method = "sum"
    if signals.get("asks_quality") and review_metric:
        temporal_metric = review_metric
        temporal_method = "mean"
    elif signals.get("asks_revenue") or _contains_any(question, ["sales"]):
        temporal_metric = preferred_value_metric or (primary_metric[0] if primary_metric else None)
        temporal_method = "sum"
    elif order_column:
        temporal_metric = order_column
        temporal_method = "distinct_count"

    if (
        time_column
        and temporal_metric
        and _contains_any(question, ["weekend", "weekday"])
        and _contains_any(question, ["sales", "orders", "demand", "revenue", "value"])
    ):
        reason = f"Derive weekday/weekend from {time_column} and compare {temporal_metric} between the two calendar segments."
        add_step(
            ComputationStep(
                operation="weekday_segment_compare",
                column=temporal_metric,
                parameters={"time_column": time_column, "metric_column": temporal_metric, "entity_column": order_column, "method": temporal_method, "intent_type": "comparison"},
                justification=reason,
                trace=_trace(reason, 0.84, ["calendar_segment", "weekday_weekend", "time_derived_dimension"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a reusable calendar-segment comparison plan.", deferred=False)

    if time_column and temporal_metric and _contains_any(question, ["black friday"]):
        reason = f"Compare {temporal_metric} around the named calendar event against a preceding baseline window."
        add_step(
            ComputationStep(
                operation="event_window_impact",
                column=temporal_metric,
                parameters={"time_column": time_column, "metric_column": temporal_metric, "entity_column": order_column, "method": temporal_method, "event_name": "black friday", "window_days": 7, "intent_type": "temporal"},
                justification=reason,
                trace=_trace(reason, 0.78, ["event_window", "calendar_event", "baseline_contrast"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a reusable event-window impact plan.", deferred=False)

    if time_column and temporal_metric and _contains_any(question, ["holiday", "holidays", "spike", "spikes"]) and not _contains_any(question, ["suspicious", "seller", "sellers"]):
        reason = f"Aggregate {temporal_metric} by calendar period and rank unusually high periods as temporal spikes."
        add_step(
            ComputationStep(
                operation="temporal_spike_detection",
                column=temporal_metric,
                parameters={"time_column": time_column, "metric_column": temporal_metric, "entity_column": order_column, "method": temporal_method, "bucket": "day", "top_n": 10, "intent_type": "temporal"},
                justification=reason,
                trace=_trace(reason, 0.76, ["temporal_spike", "period_baseline", "calendar_pattern"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.76, justification="Built a reusable temporal spike-detection plan.", deferred=False)

    anomaly_terms = ["anomaly", "anomalies", "unusual", "suspicious", "fraud", "fake", "duplicate", "duplicates", "rapid", "rapidly", "excessive"]
    if _contains_any(question, anomaly_terms + ["spike", "spikes"]):
        if customer_column and order_column and time_column and _contains_any(question, ["rapid", "rapidly", "multiple orders"]):
            reason = f"Detect customers with unusually short gaps between distinct orders using {time_column}."
            add_step(
                ComputationStep(
                    operation="rapid_repeat_order_anomaly",
                    column=time_column,
                    parameters={"customer_column": customer_column, "order_column": order_column, "time_column": time_column, "threshold_hours": 24, "top_n": 10, "intent_type": "outliers"},
                    justification=reason,
                    trace=_trace(reason, 0.82, ["anomaly_detection", "customer_order_gap", "time_delta"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a reusable rapid repeat-order anomaly plan.", deferred=False)
        if order_column and preferred_value_metric and _contains_any(question, ["high-value", "high value", "unusual transactions", "transaction", "transactions"]):
            reason = f"Rank unusually high order-level values using robust distribution thresholds."
            add_step(
                ComputationStep(
                    operation="transaction_value_outlier_rank",
                    column=preferred_value_metric,
                    parameters={"order_column": order_column, "value_column": preferred_value_metric, "top_n": 10, "intent_type": "outliers"},
                    justification=reason,
                    trace=_trace(reason, 0.82, ["anomaly_detection", "transaction_value", "iqr_threshold"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a reusable high-value transaction anomaly plan.", deferred=False)
        if installments_metric and (resolved_roles.get("price_metric") or preferred_value_metric) and _contains_any(question, ["installment", "installments", "excessive", "cheap"]):
            price_metric = resolved_roles.get("price_metric") or preferred_value_metric
            reason = f"Flag records where {installments_metric} is high relative to low {price_metric}."
            add_step(
                ComputationStep(
                    operation="contextual_metric_mismatch",
                    column=installments_metric,
                    parameters={"entity_column": product_column or focus_dimension, "order_column": order_column, "low_column": price_metric, "high_column": installments_metric, "top_n": 10, "intent_type": "outliers"},
                    justification=reason,
                    trace=_trace(reason, 0.8, ["anomaly_detection", "contextual_mismatch", "installment_vs_price"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a reusable contextual mismatch anomaly plan.", deferred=False)
        if seller_column and order_column and time_column and _contains_any(question, ["seller", "sellers", "spike", "spikes", "suspicious"]):
            reason = f"Detect entity-specific order spikes for {seller_column} over time."
            add_step(
                ComputationStep(
                    operation="entity_temporal_spike_detection",
                    column=order_column,
                    parameters={"entity_column": seller_column, "time_column": time_column, "metric_column": order_column, "method": "distinct_count", "bucket": "day", "top_n": 10, "intent_type": "outliers"},
                    justification=reason,
                    trace=_trace(reason, 0.82, ["anomaly_detection", "entity_spike", "seller_time_series"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a reusable entity temporal spike anomaly plan.", deferred=False)
        if customer_column and order_column and _contains_any(question, ["duplicate", "duplicates"]):
            fingerprint_columns = [col for col in [customer_segment_column, geography_column, category_column, resolved_roles.get("payment_type_column")] if col]
            reason = f"Group customers by behavioral fingerprints to surface repeated behavior patterns."
            add_step(
                ComputationStep(
                    operation="duplicate_behavior_fingerprint",
                    column=customer_column,
                    parameters={"customer_column": customer_column, "order_column": order_column, "fingerprint_columns": fingerprint_columns, "value_column": preferred_value_metric, "top_n": 10, "intent_type": "outliers"},
                    justification=reason,
                    trace=_trace(reason, 0.76, ["anomaly_detection", "behavior_fingerprint", "duplicate_pattern"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.76, justification="Built a reusable duplicate behavior fingerprint plan.", deferred=False)
        if review_metric and order_column and _contains_any(question, ["fake", "review", "reviews", "rating", "ratings"]):
            entity = seller_column or focus_dimension or customer_column
            if entity:
                reason = f"Score {entity} for abnormal review concentration and extreme-review patterns."
                add_step(
                    ComputationStep(
                        operation="review_pattern_anomaly",
                        column=review_metric,
                        parameters={"entity_column": entity, "review_column": review_metric, "order_column": order_column, "top_n": 10, "intent_type": "outliers"},
                        justification=reason,
                        trace=_trace(reason, 0.78, ["anomaly_detection", "review_pattern", "extreme_concentration"]),
                    )
                )
                return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a reusable review-pattern anomaly plan.", deferred=False)
        if geography_column and order_column and _contains_any(question, ["geographic", "geography", "state", "states", "location", "locations", "region", "regions", "anomaly", "anomalies"]):
            reason = f"Score geographic segments for unusual demand, value, freight, and review patterns."
            add_step(
                ComputationStep(
                    operation="geographic_anomaly_score",
                    column=order_column,
                    parameters={"geography_column": geography_column, "order_column": order_column, "value_column": preferred_value_metric, "review_column": review_metric, "freight_column": resolved_roles.get("freight_metric"), "top_n": 10, "intent_type": "outliers"},
                    justification=reason,
                    trace=_trace(reason, 0.8, ["anomaly_detection", "geography", "multi_metric_outlier"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a reusable geographic anomaly scoring plan.", deferred=False)

    if category_column and time_column and _contains_any(question, ["seasonal category", "category demand", "seasonal demand", "seasonal categories", "seasonality"]):
        seasonal_metric = order_column or temporal_metric
        if seasonal_metric:
            reason = f"Measure how unevenly category demand is distributed across months to identify seasonal categories."
            add_step(
                ComputationStep(
                    operation="segment_seasonality",
                    column=seasonal_metric,
                    parameters={"entity_column": category_column, "time_column": time_column, "bucket": "month", "method": "distinct_count" if seasonal_metric == order_column else temporal_method, "top_n": 10, "intent_type": "temporal"},
                    justification=reason,
                    trace=_trace(reason, 0.8, ["segment_seasonality", "category_demand", "calendar_month"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a reusable seasonal segment-demand plan.", deferred=False)

    if time_column and customer_column and _contains_any(question, ["customer acquisition", "new customers", "acquisition trend"]):
        reason = f"Count first-seen {customer_column} values by period to measure customer acquisition over time."
        add_step(
            ComputationStep(
                operation="customer_acquisition_trend",
                column=customer_column,
                parameters={"customer_column": customer_column, "time_column": time_column, "bucket": _growth_bucket(question), "intent_type": "temporal"},
                justification=reason,
                trace=_trace(reason, 0.86, ["customer_acquisition", "first_seen_entity", "time_series"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.86, justification="Built a reusable first-seen customer acquisition trend plan.", deferred=False)

    if (
        time_column
        and purchase_column
        and resolved_roles.get("delivered_column")
        and signals.get("asks_delivery")
        and _contains_any(question, ["trend", "over time", "performance"])
    ):
        reason = f"Track average delivery duration over time as the delivery-performance trend."
        add_step(
            ComputationStep(
                operation="delay_trend",
                column=resolved_roles.get("delivered_column"),
                parameters={"start_column": purchase_column, "end_column": resolved_roles.get("delivered_column"), "time_column": time_column, "bucket": _growth_bucket(question), "intent_type": "temporal"},
                justification=reason,
                trace=_trace(reason, 0.82, ["delivery_performance_trend", "derived_duration", "time_series"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a reusable delivery-performance trend plan.", deferred=False)

    if (
        time_column
        and temporal_metric
        and (signals.get("asks_growth") or _contains_any(question, ["trend", "pattern", "over time", "daily", "weekly", "monthly"]))
    ):
        bucket = _growth_bucket(question)
        reason = f"Aggregate {temporal_metric} by {bucket} period to produce a KPI time series."
        add_step(
            ComputationStep(
                operation="time_series_metric",
                column=temporal_metric,
                parameters={"time_column": time_column, "metric_column": temporal_metric, "entity_column": order_column, "method": temporal_method, "bucket": bucket, "intent_type": "temporal"},
                justification=reason,
                trace=_trace(reason, 0.84, ["time_series_metric", "resolved_metric_role", f"{bucket}_bucket"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a reusable KPI time-series plan.", deferred=False)

    if order_column and _contains_any(question, ["dependent", "dependency", "dependence", "overdependent"]):
        dependency_entity = seller_column if _contains_any(question, ["seller", "sellers"]) else category_column if _contains_any(question, ["category", "categories"]) else focus_dimension
        dependency_metric = preferred_value_metric if preferred_value_metric else order_column
        dependency_method = "sum" if dependency_metric != order_column else "distinct_count"
        top_n = 10 if _contains_any(question, ["top 10"]) else 5 if _contains_any(question, ["few"]) else 10
        if dependency_entity and dependency_metric:
            reason = f"Measure top-{top_n} contribution share for {dependency_entity} to quantify concentration/dependency risk."
            add_step(
                ComputationStep(
                    operation="top_dependency_share",
                    column=dependency_metric,
                    parameters={"entity_column": dependency_entity, "value_column": dependency_metric, "method": dependency_method, "top_n": top_n, "intent_type": "composition"},
                    justification=reason,
                    trace=_trace(reason, 0.84, ["dependency_risk", "top_share", "concentration"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a reusable top-share dependency risk plan.", deferred=False)

    if (
        geography_column
        and order_column
        and _contains_any(question, ["underperform", "underperformance"])
        and _contains_any(question, ["logistics", "delivery", "freight"])
    ):
        freight_metric = resolved_roles.get("freight_metric") or _best_text_match(question + " freight shipping", _all_columns(dataset_profile), ["freight", "shipping"])
        reason = f"Score {geography_column} on logistics delay, review weakness, freight burden, and support to find underperforming regions."
        add_step(
            ComputationStep(
                operation="logistics_underperformance_score",
                column=order_column,
                parameters={"entity_column": geography_column, "order_column": order_column, "review_column": review_metric, "start_column": purchase_column, "end_column": resolved_roles.get("delivered_column"), "freight_column": freight_metric, "top_n": 10, "intent_type": "comparison"},
                justification=reason,
                trace=_trace(reason, 0.8, ["logistics_underperformance", "multi_signal_risk", "geography"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a reusable logistics underperformance scoring plan.", deferred=False)

    if review_metric and order_column and _contains_any(question, ["review crisis", "review crises", "crisis", "crises"]):
        crisis_entity = geography_column if _contains_any(question, ["where", "region", "regions", "state", "states", "location", "locations"]) else seller_column or focus_dimension
        if crisis_entity:
            reason = f"Rank {crisis_entity} by low-review rate, average review weakness, and support to identify review crisis hotspots."
            add_step(
                ComputationStep(
                    operation="review_crisis_rank",
                    column=review_metric,
                    parameters={"entity_column": crisis_entity, "review_column": review_metric, "order_column": order_column, "threshold": 2.0, "top_n": 10, "intent_type": "comparison"},
                    justification=reason,
                    trace=_trace(reason, 0.8, ["review_crisis", "low_outcome_rate", "support_threshold"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a reusable review crisis hotspot plan.", deferred=False)

    if (
        order_column
        and time_column
        and resolved_roles.get("delivered_column")
        and estimated_delivery_column
        and _contains_any(question, ["late deliveries", "late delivery", "cluster", "clustered", "certain months"])
    ):
        reason = f"Compute late-delivery rates by period to detect temporal clusters of late delivery risk."
        add_step(
            ComputationStep(
                operation="late_delivery_period_cluster",
                column=order_column,
                parameters={"actual_column": resolved_roles.get("delivered_column"), "estimated_column": estimated_delivery_column, "order_column": order_column, "time_column": time_column, "bucket": "month", "intent_type": "temporal"},
                justification=reason,
                trace=_trace(reason, 0.82, ["late_delivery_cluster", "time_bucket", "rate_by_period"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a reusable late-delivery temporal cluster plan.", deferred=False)

    brand_risk_entity = seller_column or (focus_dimension if _contains_any(question, ["seller", "sellers"]) else None) or _best_text_match(question + " seller", _all_columns(dataset_profile), ["seller"])
    if brand_risk_entity and order_column and _contains_any(question, ["brand trust", "damage brand", "damage", "trust"]):
        freight_metric = resolved_roles.get("freight_metric") or _best_text_match(question + " freight shipping", _all_columns(dataset_profile), ["freight", "shipping"])
        reason = f"Score {brand_risk_entity} on review weakness, delivery delay, freight burden, and support to identify brand-trust risk."
        add_step(
            ComputationStep(
                operation="logistics_underperformance_score",
                column=order_column,
                parameters={"entity_column": brand_risk_entity, "order_column": order_column, "review_column": review_metric, "start_column": purchase_column, "end_column": resolved_roles.get("delivered_column"), "freight_column": freight_metric, "top_n": 10, "intent_type": "comparison"},
                justification=reason,
                trace=_trace(reason, 0.78, ["brand_trust_risk", "seller_risk", "multi_signal_risk"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a reusable entity risk scoring plan for brand trust.", deferred=False)

    cohort_terms = ["cohort", "cohorts", "acquired", "acquisition", "retain", "retains", "retention", "bought again", "buy again", "churn", "long term"]
    if customer_column and order_column and time_column and _contains_any(question, cohort_terms):
        if _contains_any(question, ["churn"]):
            reason = f"Estimate churn speed from customer inactivity and repeat-purchase gaps using {time_column}."
            add_step(
                ComputationStep(
                    operation="churn_speed_proxy",
                    column=time_column,
                    parameters={"customer_column": customer_column, "order_column": order_column, "time_column": time_column, "intent_type": "temporal"},
                    justification=reason,
                    trace=_trace(reason, 0.78, ["churn_proxy", "inactivity", "repeat_gap"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a reusable churn-speed proxy plan.", deferred=False)
        if _contains_any(question, ["state", "states", "certain states"]) and (customer_segment_column or geography_column):
            segment = customer_segment_column or geography_column
            reason = f"Compare repeat/retention rate by {segment}."
            add_step(
                ComputationStep(
                    operation="segment_retention_rate",
                    column=order_column,
                    parameters={"customer_column": customer_column, "order_column": order_column, "segment_column": segment, "top_n": 10, "intent_type": "comparison"},
                    justification=reason,
                    trace=_trace(reason, 0.82, ["segment_retention", "customer_identifier", "segment_key"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a reusable segment-retention comparison plan.", deferred=False)
        if _contains_any(question, ["spend", "long term", "lifetime value", "ltv"]) and preferred_value_metric:
            reason = f"Group customers by acquisition cohort and rank cohorts by long-term value."
            add_step(
                ComputationStep(
                    operation="cohort_value_rank",
                    column=preferred_value_metric,
                    parameters={"customer_column": customer_column, "order_column": order_column, "time_column": time_column, "value_column": preferred_value_metric, "bucket": "month", "top_n": 10, "intent_type": "aggregation"},
                    justification=reason,
                    trace=_trace(reason, 0.82, ["cohort_value", "first_seen_customer", "long_term_spend"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a reusable cohort long-term value plan.", deferred=False)
        cohort_filter = "jan" if _contains_any(question, ["jan", "january"]) else ""
        reason = f"Group customers by first purchase cohort and compute repeat-purchase retention."
        add_step(
            ComputationStep(
                operation="cohort_repeat_rate",
                column=time_column,
                parameters={"customer_column": customer_column, "order_column": order_column, "time_column": time_column, "bucket": "month", "cohort_filter": cohort_filter, "top_n": 10, "intent_type": "composition"},
                justification=reason,
                trace=_trace(reason, 0.84, ["cohort_retention", "first_seen_customer", "repeat_purchase"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a reusable acquisition-cohort retention plan.", deferred=False)

    if _looks_like_share_question(question) and _contains_any(question, ["top customers"]) and customer_column and preferred_value_metric:
        reason = f"Compute the share of total {preferred_value_metric} contributed by the top customers."
        add_step(
            ComputationStep(
                operation="share_of_total",
                column=preferred_value_metric,
                parameters={"entity_column": customer_column, "top_n": 10, "intent_type": "composition"},
                justification=reason,
                trace=_trace(reason, 0.84, ["share_of_total", "customer_identifier", "revenue_metric"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.84,
            justification="Built a top-customer share-of-revenue plan before generic ranking so the question is answered as a percentage, not a leaderboard.",
            deferred=False,
        )

    if _contains_any(question, ["profit proxy", "after freight"]):
        freight_column = _best_text_match(question + " freight", _all_columns(dataset_profile), ["freight"])
        if freight_column and preferred_value_metric and freight_column != preferred_value_metric:
            reason = f"Construct a profit proxy by subtracting {freight_column} from {preferred_value_metric} before aggregation."
            add_step(
                ComputationStep(
                    operation="row_expression",
                    column=preferred_value_metric,
                    parameters={"subtract_column": freight_column, "expression_name": "profit_proxy", "intent_type": "aggregation"},
                    justification=reason,
                    trace=_trace(reason, 0.84, ["derived_metric", "profit_proxy"]),
                )
            )
            agg_reason = "Aggregate the derived profit proxy to answer the requested proxy metric."
            add_step(
                ComputationStep(
                    operation="aggregate",
                    column="profit_proxy",
                    parameters={"method": "sum", "intent_type": "aggregation"},
                    justification=agg_reason,
                    trace=_trace(agg_reason, 0.84, ["derived_metric", "aggregation_mapping"]),
                )
            )
            return ComputationPlanModel(
                steps=steps,
                confidence_score=0.84,
                justification="Built a derived-metric plan for profit proxy after freight costs before generic logistics branches can intercept it.",
                deferred=False,
            )

    if (
        review_metric
        and signals.get("asks_quality")
        and _contains_any(question, ["average", "mean", "overall"])
        and not _contains_any(question, ["trend", "over time", "by ", "which ", "highest", "lowest", "poor"])
    ):
        reason = f"Aggregate {review_metric} directly because the question asks for an overall quality score."
        add_step(
            ComputationStep(
                operation="aggregate",
                column=review_metric,
                parameters={"method": "mean", "intent_type": "aggregation"},
                justification=reason,
                trace=_trace(reason, 0.88, ["scalar_quality_metric", "overall_aggregate"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.88,
            justification="Built an overall quality-score aggregate plan instead of introducing an unnecessary grouping dimension.",
            deferred=False,
        )

    if (
        review_metric
        and signals.get("asks_quality")
        and _contains_any(question, ["drive", "drives", "driver", "drivers", "why"])
        and _contains_any(question, ["1-star", "one-star", "low", "poor", "bad", "lowest"])
    ):
        candidate_columns = _unique_non_empty([
            resolved_roles.get("price_metric"),
            resolved_roles.get("freight_metric"),
            preferred_value_metric,
            category_column,
            seller_column,
            geography_column,
            payment_type_column,
            status_column,
        ])
        reason = f"Compare low-{review_metric} records against the baseline distribution across resolved numeric and categorical drivers."
        add_step(
            ComputationStep(
                operation="low_outcome_driver_analysis",
                column=review_metric,
                parameters={
                    "outcome_column": review_metric,
                    "threshold": 1.5 if _contains_any(question, ["1-star", "one-star"]) else None,
                    "candidate_columns": candidate_columns,
                    "start_column": purchase_column,
                    "end_column": resolved_roles.get("delivered_column"),
                    "top_n": 10,
                    "intent_type": "relationship",
                },
                justification=reason,
                trace=_trace(reason, 0.78, ["low_outcome_drivers", "baseline_contrast", "resolved_roles"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.78,
            justification="Built a reusable low-outcome driver analysis plan using baseline contrasts rather than a single unrelated correlation.",
            deferred=False,
        )

    if _contains_any(question, ["average delivery time", "delivery time"]) and purchase_column and resolved_roles.get("delivered_column"):
        reason = f"Measure the average elapsed time between {purchase_column} and {resolved_roles.get('delivered_column')} to answer the delivery-duration question."
        add_step(
            ComputationStep(
                operation="delivery_duration_summary",
                column=resolved_roles.get("delivered_column"),
                parameters={"start_column": purchase_column, "end_column": resolved_roles.get("delivered_column"), "intent_type": "aggregation"},
                justification=reason,
                trace=_trace(reason, 0.84, ["delivery_duration", "derived_time_metric"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a generic delivery-duration summary plan.", deferred=False)

    if (
        "relationship" in strategies
        and signals.get("asks_delivery")
        and _contains_any(question, ["delivery", "deliver", "delivered", "delay", "delays", "late", "slow", "fast"])
        and review_metric
        and purchase_column
        and resolved_roles.get("delivered_column")
    ):
        reason = f"Derive delivery duration from {purchase_column} and {resolved_roles.get('delivered_column')}, then relate it to {review_metric}."
        add_step(
            ComputationStep(
                operation="delay_quality_relationship",
                column=review_metric,
                parameters={"start_column": purchase_column, "end_column": resolved_roles.get("delivered_column"), "review_column": review_metric, "intent_type": "relationship"},
                justification=reason,
                trace=_trace(reason, 0.78, ["derived_duration_relationship", "outcome_metric"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a generic derived delivery-duration relationship plan.", deferred=False)

    if "relationship" in strategies and review_metric and signals.get("asks_quality"):
        relationship_metric = None
        if _contains_any(question, ["freight", "shipping cost", "shipping charges", "cost"]):
            relationship_metric = resolved_roles.get("freight_metric") or _best_text_match(question + " freight shipping cost", _all_columns(dataset_profile), ["freight", "shipping"])
        elif signals.get("asks_price"):
            relationship_metric = resolved_roles.get("price_metric") or resolved_revenue_metric or (primary_metric[0] if primary_metric else None)
        if relationship_metric and relationship_metric != review_metric:
            reason = f"Measure the relationship between resolved metric {relationship_metric} and outcome metric {review_metric}."
            add_step(
                ComputationStep(
                    operation="pairwise_relationship",
                    column=relationship_metric,
                    columns=[relationship_metric, review_metric],
                    parameters={"intent_type": "relationship"},
                    justification=reason,
                    trace=_trace(reason, 0.76, ["metric_outcome_relationship", "resolved_metric_roles"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.76, justification="Built a generic metric-to-outcome relationship plan.", deferred=False)

    explicit_payment_value_intent = _contains_any(
        question,
        ["order value", "aov", "spend", "revenue", "sales", "amount", "value", "gmv"],
    )
    contrast_value_column, contrast_values = _categorical_value_contrast_column(
        question,
        dataset_profile,
        inferred_context,
        [payment_type_column, focus_dimension],
    )
    payment_usage_intent = bool(
        payment_type_column
        and signals.get("asks_payment")
        and (signals.get("asks_ranking") or _contains_any(question, ["used", "usage", "frequency", "common"]))
        and not (signals.get("asks_price") or explicit_payment_value_intent)
    )
    payment_order_value_intent = bool(
        payment_type_column
        and signals.get("asks_payment")
        and (signals.get("asks_price") or explicit_payment_value_intent)
        and (signals.get("asks_ranking") or _contains_any(question, ["generate", "highest", "largest", "best"]))
    )
    installment_spend_comparison_intent = bool(
        installments_metric
        and _contains_any(question, ["installment", "installments"])
        and (signals.get("asks_contrast") or _contains_any(question, ["spend more", "higher spend", "more value", "more revenue"]))
    )
    payment_status_rate_intent = bool(
        payment_type_column
        and status_column
        and order_column
        and signals.get("asks_payment")
        and _contains_any(question, ["cancel", "cancelled", "canceled", "cancellation", "failure rate", "status rate"])
    )
    installment_premium_relationship_intent = bool(
        category_column
        and installments_metric
        and preferred_value_metric
        and _contains_any(question, ["installment", "installments"])
        and (signals.get("asks_price") or _contains_any(question, ["premium", "expensive", "high value"]))
        and "relationship" in strategies
    )
    payment_geo_mix_intent = bool(
        geography_column
        and payment_type_column
        and order_column
        and signals.get("asks_payment")
        and signals.get("asks_geography")
    )
    installment_by_segment_intent = bool(
        category_column
        and installments_metric
        and _contains_any(question, ["installment", "installments"])
        and _contains_any(question, ["average", "mean", "by category", "by product"])
    )

    basket_terms = ["basket", "baskets", "bundle", "bundles", "bundled", "bought together", "commonly bundled", "cross-sell", "cross sell", "same order", "recommend", "recommended"]
    basket_intent = bool(order_column and (signals.get("asks_basket") or _contains_any(question, basket_terms)))
    basket_entity = category_column if _contains_any(question, ["categor", "category", "categories"]) else product_column
    if not basket_entity and focus_dimension and focus_dimension != order_column:
        basket_entity = focus_dimension
    if not basket_entity:
        basket_entity = category_column or product_column

    if basket_intent and _contains_any(question, ["higher aov", "aov", "higher order value", "higher value", "spend more"]) and basket_entity and preferred_value_metric:
        reason = f"Compare order value between baskets with multiple {basket_entity} values and baskets with a single {basket_entity} value."
        value_method = "max" if "payment" in str(preferred_value_metric).lower() else "sum"
        add_step(
            ComputationStep(
                operation="basket_value_comparison",
                column=preferred_value_metric,
                parameters={"order_column": order_column, "entity_column": basket_entity, "value_column": preferred_value_metric, "order_value_method": value_method, "intent_type": "comparison"},
                justification=reason,
                trace=_trace(reason, 0.82, ["basket_value_comparison", "order_level_value", "bundle_size"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a reusable bundled-versus-single basket value comparison plan.", deferred=False)

    if basket_intent and basket_entity:
        reason = f"Count co-occurring {basket_entity} pairs within each {order_column} to surface common bundles and cross-sell candidates."
        recommendation_mode = bool(_contains_any(question, ["recommend", "recommended", "cross-sell", "cross sell"]))
        add_step(
            ComputationStep(
                operation="basket_cooccurrence",
                column=basket_entity,
                parameters={
                    "entity_column": basket_entity,
                    "order_column": order_column,
                    "top_n": 10,
                    "intent_type": "composition",
                    "recommendation_mode": recommendation_mode,
                    "min_confidence": 0.05 if recommendation_mode else 0.0,
                    "min_lift": 1.0 if recommendation_mode else 0.0,
                },
                justification=reason,
                trace=_trace(reason, 0.82, ["basket_cooccurrence", "order_composition", "pair_support"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a reusable basket co-occurrence plan.", deferred=False)

    if payment_usage_intent:
        usage_metric = order_column or payment_type_column
        add_step(
            _rank_entities_step(
                payment_type_column,
                usage_metric,
                "distinct_count",
                question,
                0.84,
                ["payment_usage", "entity_ranking"],
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a payment-method usage ranking plan.", deferred=False)

    if payment_order_value_intent and order_column and preferred_value_metric:
        add_step(
            _segment_order_value_step(
                payment_type_column,
                order_column,
                preferred_value_metric,
                0.82,
                ["payment_order_value", "order_level_metric"],
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a payment-method average-order-value plan.", deferred=False)

    if installment_spend_comparison_intent and preferred_value_metric:
        reason = f"Compare spend between installment-backed and non-installment purchases using {installments_metric} as the threshold signal."
        add_step(
            ComputationStep(
                operation="threshold_value_comparison",
                column=preferred_value_metric,
                parameters={"threshold_column": installments_metric, "threshold": 1, "value_column": preferred_value_metric, "higher_group_label": "installment_users", "lower_group_label": "single_payment_users", "intent_type": "comparison"},
                justification=reason,
                trace=_trace(reason, 0.8, ["threshold_comparison", "installment_behavior"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built an installment-versus-non-installment spend comparison plan.", deferred=False)

    preference_entity_column = customer_column if _contains_any(question, ["customer", "customers"]) else None
    if not preference_entity_column and focus_dimension and focus_dimension != contrast_value_column:
        preference_entity_column = focus_dimension

    if preference_entity_column and contrast_value_column and len(contrast_values) >= 2:
        compared_values = " versus ".join(contrast_values[:2])
        reason = f"Compare {preference_entity_column}-level shares of {contrast_value_column} to identify which entities skew toward {compared_values}."
        add_step(
            ComputationStep(
                operation="categorical_preference_by_entity",
                column=contrast_value_column,
                parameters={"entity_column": preference_entity_column, "category_column": contrast_value_column, "preferred_values": contrast_values[:4], "top_n": 10, "intent_type": "composition"},
                justification=reason,
                trace=_trace(reason, 0.78, ["categorical_preference", "entity_identifier", "matched_category_values"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built an entity-level categorical preference comparison plan.", deferred=False)

    if payment_status_rate_intent:
        reason = f"Compute cancellation rates by {payment_type_column} to test whether payment methods differ in operational failure."
        add_step(
            ComputationStep(
                operation="status_rate_by_entity",
                column=order_column,
                parameters={"entity_column": payment_type_column, "status_column": status_column, "order_column": order_column, "mode": "canceled", "top_n": 10, "intent_type": "comparison"},
                justification=reason,
                trace=_trace(reason, 0.8, ["payment_status_rate", "cancellation_rate"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a payment-method cancellation-rate plan.", deferred=False)

    if installment_premium_relationship_intent:
        reason = f"Relate category-level average installments to category-level average value to test whether installment-heavy categories also look premium."
        add_step(
            ComputationStep(
                operation="grouped_pairwise_relationship",
                column=installments_metric,
                parameters={"entity_column": category_column, "left_column": installments_metric, "left_method": "mean", "right_column": preferred_value_metric, "right_method": "mean", "method": "spearman", "intent_type": "relationship"},
                justification=reason,
                trace=_trace(reason, 0.76, ["grouped_relationship", "installment_premium"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.76, justification="Built a grouped relationship plan between average installments and premium value by category.", deferred=False)

    if payment_geo_mix_intent:
        add_step(
            _concentration_step(
                geography_column,
                payment_type_column,
                order_column,
                "distinct_count",
                0.78,
                ["payment_mix", "geographic_preference"],
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a location-by-payment-type concentration plan.", deferred=False)

    if installment_by_segment_intent:
        reason = f"Average {installments_metric} within each {category_column} to compare installment intensity by category."
        add_step(
            ComputationStep(
                operation="rank_entities",
                column=installments_metric,
                parameters={"entity_column": category_column, "method": "mean", "top_n": 10, "sort": "desc", "intent_type": "aggregation"},
                justification=reason,
                trace=_trace(reason, 0.8, ["installment_average", "category_breakdown"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built an average-installments-by-category plan.", deferred=False)

    if _contains_any(question, ["orders are delivered", "% orders are delivered", "% canceled", "% cancelled", "% unavailable", "invoiced but not delivered"]) and not _contains_any(question, ["delivered early", "delivered late", "% delivered early", "% delivered late", "late", "early"]) and status_column and order_column:
        lower_q = question.lower()
        if "unavailable" in lower_q:
            mode = "unavailable"
        elif "invoiced" in lower_q and "not delivered" in lower_q:
            mode = "invoiced_not_delivered"
        elif "cancel" in lower_q:
            mode = "canceled"
        else:
            mode = "delivered"
        reason = f"Compute the share of orders matching the requested operational status condition."
        add_step(
            ComputationStep(
                operation="status_share",
                column=order_column,
                parameters={"status_column": status_column, "order_column": order_column, "mode": mode, "intent_type": "aggregation"},
                justification=reason,
                trace=_trace(reason, 0.84, ["status_share", mode]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a generic status-share plan.", deferred=False)

    if focus_dimension and _contains_any(question, ["high cancellation rates", "high failure rates"]) and status_column and order_column:
        mode = "canceled" if "cancel" in question.lower() else "failure"
        reason = f"Compute the rate of problematic orders by {focus_dimension} to identify entities with elevated operational failure."
        add_step(
            ComputationStep(
                operation="status_rate_by_entity",
                column=order_column,
                parameters={"entity_column": focus_dimension, "status_column": status_column, "order_column": order_column, "mode": mode, "top_n": 10, "intent_type": "comparison"},
                justification=reason,
                trace=_trace(reason, 0.8, ["status_rate", mode]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a generic status-rate-by-entity plan.", deferred=False)

    if _contains_any(question, ["operational issues"]) and status_column and order_column and time_column:
        reason = f"Score each period on rates of non-delivered operational statuses to identify when operational issues were most severe."
        add_step(
            ComputationStep(
                operation="operational_issue_score",
                column=order_column,
                parameters={"status_column": status_column, "order_column": order_column, "time_column": time_column, "bucket": _growth_bucket(question), "intent_type": "temporal"},
                justification=reason,
                trace=_trace(reason, 0.78, ["operational_issue", "time_bucketed_status"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a generic operational-issue scoring plan.", deferred=False)

    if _contains_any(question, ["cancellations increasing"]) and status_column and order_column and time_column:
        reason = f"Track the cancellation rate over time to determine whether cancellations are increasing."
        add_step(
            ComputationStep(
                operation="status_rate_trend",
                column=order_column,
                parameters={"status_column": status_column, "order_column": order_column, "time_column": time_column, "mode": "canceled", "bucket": _growth_bucket(question), "intent_type": "temporal"},
                justification=reason,
                trace=_trace(reason, 0.8, ["status_trend", "cancellation_rate"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a generic cancellation-rate trend plan.", deferred=False)

    if _contains_any(question, ["estimated vs actual delivery gap", "delivery gap", "estimated vs actual"]) and resolved_roles.get("delivered_column") and estimated_delivery_column:
        reason = f"Measure the average gap between actual delivery and estimated delivery to quantify delivery timing accuracy."
        add_step(
            ComputationStep(
                operation="delivery_gap_summary",
                column=resolved_roles.get("delivered_column"),
                parameters={"actual_column": resolved_roles.get("delivered_column"), "estimated_column": estimated_delivery_column, "intent_type": "comparison"},
                justification=reason,
                trace=_trace(reason, 0.82, ["delivery_gap", "estimated_actual_difference"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a generic estimated-versus-actual delivery gap plan.", deferred=False)

    if _contains_any(question, ["delivered late", "delivered early", "late", "early"]) and resolved_roles.get("delivered_column") and estimated_delivery_column and order_column:
        timing_mode = "late" if "late" in question.lower() else "early"
        reason = f"Compute the share of orders delivered {timing_mode} by comparing actual and estimated delivery dates."
        add_step(
            ComputationStep(
                operation="delivery_timing_share",
                column=order_column,
                parameters={"actual_column": resolved_roles.get("delivered_column"), "estimated_column": estimated_delivery_column, "order_column": order_column, "mode": timing_mode, "intent_type": "aggregation"},
                justification=reason,
                trace=_trace(reason, 0.84, ["delivery_timing", timing_mode]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a generic late/early delivery share plan.", deferred=False)

    if focus_dimension and _contains_any(question, ["experience delays most", "cause most delays", "take longest to deliver"]) and purchase_column and resolved_roles.get("delivered_column"):
        sort_dir = "desc"
        reason = f"Rank {focus_dimension} by average delivery duration to identify which entities contribute most to delays."
        add_step(
            ComputationStep(
                operation="delay_burden_rank",
                column=resolved_roles.get("delivered_column"),
                parameters={"entity_column": focus_dimension, "start_column": purchase_column, "end_column": resolved_roles.get("delivered_column"), "sort": sort_dir, "top_n": 10, "intent_type": "aggregation"},
                justification=reason,
                trace=_trace(reason, 0.82, ["delay_burden", "entity_breakdown"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a generic delay-burden ranking plan.", deferred=False)

    if _contains_any(question, ["delays improving over time", "improving over time"]) and purchase_column and resolved_roles.get("delivered_column") and time_column:
        reason = f"Track average delivery duration over time to determine whether delays are improving or worsening."
        add_step(
            ComputationStep(
                operation="delay_trend",
                column=resolved_roles.get("delivered_column"),
                parameters={"start_column": purchase_column, "end_column": resolved_roles.get("delivered_column"), "time_column": time_column, "bucket": _growth_bucket(question), "intent_type": "temporal"},
                justification=reason,
                trace=_trace(reason, 0.8, ["delay_trend", "time_series_logistics"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a generic delay trend plan.", deferred=False)

    if _contains_any(question, ["delivery speed impact ratings", "delivery speed impact", "delivery speed affect ratings"]) and review_metric and purchase_column and resolved_roles.get("delivered_column"):
        reason = f"Measure the relationship between derived delivery duration and {review_metric} to test whether slower delivery aligns with weaker ratings."
        add_step(
            ComputationStep(
                operation="delay_quality_relationship",
                column=review_metric,
                parameters={"start_column": purchase_column, "end_column": resolved_roles.get("delivered_column"), "review_column": review_metric, "intent_type": "relationship"},
                justification=reason,
                trace=_trace(reason, 0.78, ["delay_quality", "relationship_intent"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a generic delivery-speed-versus-rating relationship plan.", deferred=False)

    if _contains_any(question, ["shipping distance affect cancellations", "distance affect cancellations", "distance affect cancellation"]) and status_column and resolved_roles.get("seller_geo_column") and resolved_roles.get("customer_geo_column") and order_column:
        reason = f"Use seller and customer location prefixes as a distance proxy to test whether larger shipping separation aligns with more cancellations."
        add_step(
            ComputationStep(
                operation="distance_proxy_cancellation_relationship",
                column=order_column,
                parameters={"seller_geo_column": resolved_roles.get("seller_geo_column"), "customer_geo_column": resolved_roles.get("customer_geo_column"), "status_column": status_column, "order_column": order_column, "intent_type": "relationship"},
                justification=reason,
                trace=_trace(reason, 0.62, ["distance_proxy", "cancellation_relationship"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.62, justification="Built a cautious proxy relationship plan for shipping separation and cancellations.", deferred=False)

    if any(token in question.lower() for token in ["warehouse", "redistribution"]):
        freight_metric = resolved_roles.get("freight_metric") or _best_text_match(question + " freight shipping", _all_columns(dataset_profile), ["freight", "shipping"])
        state_geo = _best_text_match(
            "customer state region geography",
            _all_columns(dataset_profile),
            ["customer_state", "state", "region"],
        )
        optimization_geo = state_geo or resolved_roles.get("geography_column") or geography_column or _business_geography_column(dataset_profile, inferred_context, None, owner=None)
        demand_metric = order_column or customer_column
        if freight_metric and demand_metric and optimization_geo:
            reason = f"Score geographic areas on shipping cost, demand, and local seller presence to surface where redistribution or network rebalancing may reduce cost."
            add_step(
                ComputationStep(
                    operation="logistics_optimization_opportunity",
                    column=freight_metric,
                    parameters={
                        "entity_column": optimization_geo,
                        "freight_column": freight_metric,
                        "demand_column": demand_metric,
                        "supply_column": seller_column,
                        "top_n": 10,
                        "min_support": 10,
                        "intent_type": "optimization",
                    },
                    justification=reason,
                    trace=_trace(reason, 0.7, ["logistics_opportunity", "cost_reduction"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.7, justification="Built a generic logistics optimization opportunity plan.", deferred=False)

    if focus_dimension and _contains_any(question, ["relative to", "as % of", "as percentage of", "burden"]):
        numerator_metric = None
        denominator_metric = None
        if _contains_any(question, ["freight", "shipping", "cost"]):
            numerator_metric = resolved_roles.get("freight_metric") or _best_text_match(question + " freight shipping cost", _all_columns(dataset_profile), ["freight", "shipping"])
        if _contains_any(question, ["price", "revenue", "sales", "value"]):
            denominator_metric = (
                resolved_roles.get("price_metric")
                or resolved_roles.get("revenue_metric")
                or _best_text_match(question + " price revenue sales value", _all_columns(dataset_profile), ["price", "revenue", "sales", "value"])
            )
        if numerator_metric and denominator_metric and numerator_metric != denominator_metric:
            add_step(
                _relative_burden_step(
                    focus_dimension,
                    numerator_metric,
                    denominator_metric,
                    0.8,
                    ["relative_metric_burden", "resolved_metric_roles"],
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a generic relative-metric burden ranking plan.", deferred=False)

    if focus_dimension and _contains_any(question, ["freight cost relative to price", "relative to price", "freight burden", "inefficient to ship"]):
        freight_metric = resolved_roles.get("freight_metric") or _best_text_match(question + " freight shipping", _all_columns(dataset_profile), ["freight", "shipping"])
        price_metric = resolved_roles.get("price_metric") or resolved_roles.get("revenue_metric") or _best_text_match(question + " price pricing value revenue", _all_columns(dataset_profile), ["price", "value", "payment", "revenue"])
        if freight_metric and price_metric:
            add_step(
                _relative_burden_step(
                    focus_dimension,
                    freight_metric,
                    price_metric,
                    0.8,
                    ["relative_burden", "cost_to_value"],
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a generic relative-burden ranking plan from the resolved focus dimension.", deferred=False)

    if focus_dimension and _contains_any(question, ["freight expenses", "freight cost", "shipping expenses", "shipping cost", "cost most to deliver", "cost to deliver", "expensive to deliver"]):
        freight_metric = resolved_roles.get("freight_metric") or _best_text_match(question + " freight shipping", _all_columns(dataset_profile), ["freight", "shipping"])
        if freight_metric:
            reason = f"Rank {focus_dimension} by total {freight_metric} to identify which entities contribute most to shipping cost."
            add_step(
                ComputationStep(
                    operation="rank_entities",
                    column=freight_metric,
                    parameters={"entity_column": focus_dimension, "method": "sum", "top_n": 10, "sort": "desc", "intent_type": "aggregation"},
                    justification=reason,
                    trace=_trace(reason, 0.82, ["entity_cost_ranking", "freight_metric"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a generic entity freight-cost ranking plan from the resolved focus dimension.", deferred=False)

    if focus_dimension and _contains_any(question, ["overpriced relative to reviews", "overpriced"]) and review_metric:
        price_metric = resolved_roles.get("price_metric") or resolved_revenue_metric or _best_value_metric(dataset_profile, inferred_context, relationship_signals) or (primary_metric[0] if primary_metric else None)
        if price_metric:
            reason = f"Compare {focus_dimension} on average price and average review quality to surface entities that look expensive relative to customer sentiment."
            add_step(
                ComputationStep(
                    operation="segment_contrast",
                    column=price_metric,
                    parameters={
                        "entity_column": focus_dimension,
                        "primary_metric": price_metric,
                        "primary_method": "mean",
                        "secondary_metric": review_metric,
                        "secondary_method": "mean",
                        "pattern": "high_low",
                        "top_n": 10,
                        "intent_type": "comparison",
                    },
                    justification=reason,
                    trace=_trace(reason, 0.8, ["value_quality_contrast", "review_signal"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a generic price-versus-review contrast plan from the resolved focus dimension.", deferred=False)

    if focus_dimension and _contains_any(question, ["price elasticity signals", "elasticity"]) and time_column and order_column:
        price_metric = resolved_roles.get("price_metric") or resolved_revenue_metric or (primary_metric[0] if primary_metric else None)
        if price_metric:
            reason = f"Estimate whether lower prices tend to coincide with stronger demand over time for each {focus_dimension}."
            add_step(
                ComputationStep(
                    operation="elasticity_proxy_score",
                    column=price_metric,
                    parameters={
                        "entity_column": focus_dimension,
                        "price_column": price_metric,
                        "demand_column": order_column,
                        "time_column": time_column,
                        "bucket": _growth_bucket(question),
                        "top_n": 10,
                        "intent_type": "relationship",
                    },
                    justification=reason,
                    trace=_trace(reason, 0.76, ["elasticity_proxy", "price_demand_time_series"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.76, justification="Built a generic elasticity-proxy plan from the resolved focus dimension.", deferred=False)

    if focus_dimension and _contains_any(question, ["larger/heavier", "heavier", "larger"]) and _contains_any(question, ["delay", "delays"]):
        size_metrics = [col for col in (resolved_roles.get("size_metrics") or []) if col in _all_columns(dataset_profile)]
        delivery_end = resolved_roles.get("delivered_column") or _best_text_match(question + " delivered delivery", _all_columns(dataset_profile), ["delivered", "delivery"])
        delivery_start = resolved_roles.get("purchase_column") or _best_text_match(question + " purchase approved order", _all_columns(dataset_profile), ["purchase", "approved", "order"])
        if size_metrics and delivery_start and delivery_end:
            reason = f"Relate physical size measures to delivery delay to test whether larger or heavier entities are associated with slower fulfillment."
            add_step(
                ComputationStep(
                    operation="derived_delay_relationship",
                    column=size_metrics[0],
                    columns=size_metrics,
                    parameters={
                        "entity_column": focus_dimension,
                        "size_columns": size_metrics,
                        "start_column": delivery_start,
                        "end_column": delivery_end,
                        "top_n": 10,
                        "intent_type": "relationship",
                    },
                    justification=reason,
                    trace=_trace(reason, 0.78, ["derived_relationship", "delivery_delay"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a generic derived relationship plan between size metrics and delivery delay.", deferred=False)

    if focus_dimension and _contains_any(question, ["bundled in same order", "bundled", "same order"]) and order_column:
        reason = f"Count co-occurrence of {focus_dimension} values within the same {order_column} to identify common bundles."
        add_step(
            ComputationStep(
                operation="basket_cooccurrence",
                column=focus_dimension,
                parameters={
                    "entity_column": focus_dimension,
                    "order_column": order_column,
                    "top_n": 10,
                    "intent_type": "composition",
                },
                justification=reason,
                trace=_trace(reason, 0.8, ["basket_cooccurrence", "order_composition"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a generic basket co-occurrence plan from the resolved focus dimension.", deferred=False)

    if focus_dimension and _contains_any(question, ["premium pricing potential", "premium pricing"]):
        value_metric = resolved_roles.get("price_metric") or resolved_revenue_metric or _best_value_metric(dataset_profile, inferred_context, relationship_signals) or (primary_metric[0] if primary_metric else None)
        if value_metric:
            reason = f"Score {focus_dimension} on value, quality, and demand support to estimate which entities have stronger premium pricing potential."
            add_step(
                ComputationStep(
                    operation="premium_potential_score",
                    column=value_metric,
                    parameters={
                        "entity_column": focus_dimension,
                        "value_metric": value_metric,
                        "review_metric": review_metric,
                        "count_column": order_column,
                        "top_n": 10,
                        "intent_type": "comparison",
                    },
                    justification=reason,
                    trace=_trace(reason, 0.76, ["premium_potential", "multi_signal_scoring"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.76, justification="Built a generic premium-potential scoring plan from the resolved focus dimension.", deferred=False)

    if focus_dimension and _contains_any(question, ["could support price increases", "price increase", "price increases"]):
        value_metric = resolved_roles.get("price_metric") or resolved_revenue_metric or _best_value_metric(dataset_profile, inferred_context, relationship_signals) or (primary_metric[0] if primary_metric else None)
        if value_metric:
            reason = f"Score {focus_dimension} on price level, quality support, and demand support to estimate which entities are better positioned for cautious price increases."
            add_step(
                ComputationStep(
                    operation="premium_potential_score",
                    column=value_metric,
                    parameters={
                        "entity_column": focus_dimension,
                        "value_metric": value_metric,
                        "review_metric": review_metric,
                        "count_column": order_column,
                        "top_n": 10,
                        "intent_type": "comparison",
                    },
                    justification=reason,
                    trace=_trace(reason, 0.76, ["premium_potential", "price_increase_support"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.76, justification="Built a generic price-increase support scoring plan from the resolved focus dimension.", deferred=False)

    if focus_dimension and _contains_any(question, ["price wars", "price war"]):
        price_metric = resolved_roles.get("price_metric") or resolved_revenue_metric or (primary_metric[0] if primary_metric else None)
        comparison_column = seller_column if seller_column and seller_column != focus_dimension else order_column
        if price_metric:
            reason = f"Score {focus_dimension} for price competition pressure using low price levels, seller fragmentation, and price dispersion."
            add_step(
                ComputationStep(
                    operation="price_competition_score",
                    column=price_metric,
                    parameters={
                        "entity_column": focus_dimension,
                        "price_column": price_metric,
                        "comparison_column": comparison_column,
                        "top_n": 10,
                        "intent_type": "comparison",
                    },
                    justification=reason,
                    trace=_trace(reason, 0.72, ["price_competition", "fragmentation_signal"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.72, justification="Built a generic price-competition scoring plan from the resolved focus dimension.", deferred=False)

    if focus_dimension and _contains_any(question, ["underperform despite traffic", "underperform despite orders", "underperform despite traffic/orders"]):
        value_metric = resolved_roles.get("revenue_metric") or _best_value_metric(dataset_profile, inferred_context, relationship_signals) or (primary_metric[0] if primary_metric else None)
        if order_column and value_metric:
            reason = f"Compare {focus_dimension} on demand and value to surface entities that attract traffic or orders but translate that into weak commercial value."
            add_step(
                ComputationStep(
                    operation="segment_contrast",
                    column=value_metric,
                    parameters={
                        "entity_column": focus_dimension,
                        "primary_metric": order_column,
                        "primary_method": "distinct_count",
                        "secondary_metric": value_metric,
                        "secondary_method": "sum",
                        "pattern": "high_low",
                        "top_n": 10,
                        "intent_type": "comparison",
                    },
                    justification=reason,
                    trace=_trace(reason, 0.8, ["underperformance_contrast", "demand_value_gap"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a generic underperformance contrast plan from the resolved focus dimension.", deferred=False)

    if _contains_any(question, ["higher prices correlate with lower ratings", "correlate"]) and review_metric:
        price_metric = resolved_roles.get("price_metric") or resolved_revenue_metric or (primary_metric[0] if primary_metric else None)
        if price_metric:
            reason = f"Measure the numeric relationship between {price_metric} and {review_metric} because the question asks whether higher prices align with lower ratings."
            add_step(
                ComputationStep(
                    operation="pairwise_relationship",
                    column=price_metric,
                    columns=[review_metric],
                    parameters={"method": "spearman", "comparison_column": review_metric, "intent_type": "relationship"},
                    justification=reason,
                    trace=_trace(reason, 0.8, ["relationship_intent", "price_quality"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a generic price-versus-rating relationship plan.", deferred=False)

    if _contains_any(question, ["freight cost reduce review scores", "freight charges reduce review scores", "freight review", "shipping cost reduce review"]) and review_metric:
        freight_metric = resolved_roles.get("freight_metric") or _best_text_match(question + " freight shipping", _all_columns(dataset_profile), ["freight", "shipping"])
        if freight_metric:
            reason = f"Measure the numeric relationship between {freight_metric} and {review_metric} to test whether higher shipping charges align with weaker customer sentiment."
            add_step(
                ComputationStep(
                    operation="pairwise_relationship",
                    column=freight_metric,
                    columns=[review_metric],
                    parameters={"method": "spearman", "comparison_column": review_metric, "intent_type": "relationship"},
                    justification=reason,
                    trace=_trace(reason, 0.78, ["relationship_intent", "cost_quality"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a generic freight-versus-review relationship plan.", deferred=False)

    if _contains_any(question, ["freight cost as % of revenue", "freight as % of revenue", "freight cost as percentage of revenue"]):
        freight_metric = resolved_roles.get("freight_metric") or _best_text_match(question + " freight shipping", _all_columns(dataset_profile), ["freight", "shipping"])
        revenue_metric = resolved_roles.get("revenue_metric") or _best_value_metric(dataset_profile, inferred_context, relationship_signals) or (primary_metric[0] if primary_metric else None)
        if freight_metric and revenue_metric:
            reason = f"Divide total {freight_metric} by total {revenue_metric} to quantify shipping cost as a share of revenue."
            add_step(
                ComputationStep(
                    operation="ratio_metric",
                    column=freight_metric,
                    parameters={
                        "numerator_column": freight_metric,
                        "denominator_column": revenue_metric,
                        "numerator_method": "sum",
                        "denominator_method": "sum",
                        "as_percentage": True,
                        "intent_type": "aggregation",
                    },
                    justification=reason,
                    trace=_trace(reason, 0.84, ["ratio_metric", "cost_to_revenue"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a generic cost-to-revenue ratio plan.", deferred=False)

    if _contains_any(question, ["total freight cost"]):
        freight_metric = resolved_roles.get("freight_metric") or _best_text_match(question + " freight shipping", _all_columns(dataset_profile), ["freight", "shipping"])
        if freight_metric:
            reason = f"Sum {freight_metric} because the question asks for the total freight cost."
            add_step(
                ComputationStep(
                    operation="aggregate",
                    column=freight_metric,
                    parameters={"method": "sum", "intent_type": "aggregation"},
                    justification=reason,
                    trace=_trace(reason, 0.86, ["scalar_kpi", "freight_total"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.86, justification="Built a scalar total-freight plan.", deferred=False)

    if _contains_any(question, ["causing churn", "cause churn", "freight charges causing churn", "freight causing churn"]) and customer_column and order_column:
        freight_metric = resolved_roles.get("freight_metric") or _best_text_match(question + " freight shipping", _all_columns(dataset_profile), ["freight", "shipping"])
        if freight_metric:
            reason = f"Compare customer-level freight exposure against repeat-purchase behavior to estimate whether higher shipping charges align with weaker retention."
            add_step(
                ComputationStep(
                    operation="retention_risk_proxy",
                    column=freight_metric,
                    parameters={
                        "customer_column": customer_column,
                        "order_column": order_column,
                        "freight_column": freight_metric,
                        "time_column": time_column,
                        "intent_type": "relationship",
                    },
                    justification=reason,
                    trace=_trace(reason, 0.68, ["retention_proxy", "cost_exposure"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.68, justification="Built a generic freight-to-retention proxy plan.", deferred=False)

    if _contains_any(question, ["discounts likely drive volume spikes", "discount", "volume spikes"]) and time_column and order_column:
        price_metric = resolved_roles.get("price_metric") or resolved_revenue_metric or (primary_metric[0] if primary_metric else None)
        if price_metric:
            reason = f"Estimate whether lower-than-baseline prices tend to coincide with stronger demand over time."
            add_step(
                ComputationStep(
                    operation="discount_volume_effect",
                    column=price_metric,
                    parameters={
                        "price_column": price_metric,
                        "demand_column": order_column,
                        "time_column": time_column,
                        "bucket": _growth_bucket(question),
                        "intent_type": "relationship",
                    },
                    justification=reason,
                    trace=_trace(reason, 0.72, ["discount_proxy", "price_demand_time_series"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.72, justification="Built a generic discount-effect proxy plan from price and demand over time.", deferred=False)

    if _contains_any(question, ["price ranges convert best", "convert best"]) and order_column:
        price_metric = resolved_roles.get("price_metric") or resolved_revenue_metric or (primary_metric[0] if primary_metric else None)
        if price_metric:
            reason = f"Bucket {price_metric} into price ranges and rank those ranges by observed order demand as a cautious proxy because true conversion data is unavailable."
            add_step(
                ComputationStep(
                    operation="price_band_demand",
                    column=price_metric,
                    parameters={
                        "price_column": price_metric,
                        "demand_column": order_column,
                        "bands": 5,
                        "top_n": 5,
                        "proxy_mode": "order_demand",
                        "intent_type": "aggregation",
                    },
                    justification=reason,
                    trace=_trace(reason, 0.68, ["price_banding", "demand_proxy", "conversion_unavailable"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.68, justification="Built a cautious price-band demand-proxy plan because true conversion data is unavailable.", deferred=False)

    resolved_revenue_metric = resolved_roles.get("revenue_metric")

    if focus_dimension and _contains_any(question, ["high revenue but poor reviews"]) and (resolved_revenue_metric or primary_metric) and review_metric:
        revenue_metric = resolved_revenue_metric or primary_metric[0]
        reason = f"Compare {focus_dimension} on both revenue and review quality to surface entities that perform well commercially but weakly on experience."
        add_step(
            ComputationStep(
                operation="segment_contrast",
                column=review_metric,
                parameters={
                    "entity_column": focus_dimension,
                    "primary_metric": revenue_metric,
                    "primary_method": "sum",
                    "secondary_metric": review_metric,
                    "secondary_method": "mean",
                    "pattern": "high_low",
                    "top_n": 10,
                    "intent_type": "comparison",
                },
                justification=reason,
                trace=_trace(reason, 0.8, ["entity_contrast", "multi_metric_reasoning"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a generic entity revenue-versus-quality contrast plan.", deferred=False)

    if focus_dimension and _contains_any(question, ["cancel", "cancelled", "canceled"]) and status_column and order_column:
        reason = f"Rank {focus_dimension} by distinct canceled {order_column} values to identify which entities contribute most to cancellations."
        add_step(
            ComputationStep(
                operation="filtered_rank_entities",
                column=order_column,
                parameters={
                    "entity_column": focus_dimension,
                    "method": "distinct_count",
                    "top_n": 10,
                    "sort": "desc",
                    "intent_type": "aggregation",
                    "filter_column": status_column,
                    "filter_contains": "cancel",
                },
                justification=reason,
                trace=_trace(reason, 0.84, ["status_filtered_ranking", "order_identifier"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a generic status-filtered entity ranking plan from the resolved focus dimension.", deferred=False)

    if category_column and _contains_any(question, ["highest order volume but low revenue", "low volume but premium pricing"]):
        contrast_pattern = "high_low" if _contains_any(question, ["highest order volume but low revenue"]) else "low_high"
        secondary_metric = None
        if _contains_any(question, ["revenue"]) and business_metric_candidates:
            secondary_metric = business_metric_candidates[0]
        elif _contains_any(question, ["pricing", "price", "premium"]) and any("price" in str(col).lower() for col in business_metric_candidates):
            secondary_metric = next(col for col in business_metric_candidates if "price" in str(col).lower())
        elif primary_metric:
            secondary_metric = primary_metric[0]
        if order_column and secondary_metric:
            reason = f"Compare categories on both volume and value metrics to surface the requested contrast pattern."
            add_step(
                ComputationStep(
                    operation="segment_contrast",
                    column=secondary_metric,
                    parameters={
                        "entity_column": category_column,
                        "primary_metric": order_column,
                        "primary_method": "distinct_count",
                        "secondary_metric": secondary_metric,
                        "secondary_method": "mean" if contrast_pattern == "low_high" else "sum",
                        "pattern": contrast_pattern,
                        "top_n": 10,
                        "intent_type": "comparison",
                    },
                    justification=reason,
                    trace=_trace(reason, 0.8, ["segment_contrast", "multi_metric_reasoning"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a dual-metric contrast plan for category performance.", deferred=False)

    if _contains_any(question, ["product categories", "categories"]) and _contains_any(question, ["sell the most", "most sold", "highest order volume"]):
        entity_column = category_column
        if entity_column and order_column:
            reason = f"Rank {entity_column} by distinct {order_column} to identify the categories with the highest selling volume."
            add_step(
                ComputationStep(
                    operation="rank_entities",
                    column=order_column,
                    parameters={"entity_column": entity_column, "method": "distinct_count", "top_n": 10, "sort": "desc", "intent_type": "aggregation"},
                    justification=reason,
                    trace=_trace(reason, 0.84, ["segment_ranking", "demand_metric"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a category demand ranking plan.", deferred=False)

    if category_column and time_column and _contains_any(question, ["categories are growing fastest", "categories are declining"]):
        growth_metric = order_column or (primary_metric[0] if primary_metric else None)
        growth_method = "distinct_count" if growth_metric == order_column else "sum"
        sort_dir = "asc" if _contains_any(question, ["declining"]) else "desc"
        if growth_metric:
            reason = f"Measure period-over-period growth by {category_column} and rank categories in the requested direction."
            add_step(
                ComputationStep(
                    operation="segment_growth_rank",
                    column=growth_metric,
                    parameters={"entity_column": category_column, "time_column": time_column, "bucket": "month", "method": growth_method, "sort": sort_dir, "top_n": 10, "intent_type": "temporal"},
                    justification=reason,
                    trace=_trace(reason, 0.8, ["segment_growth", "temporal_breakdown"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a category growth ranking plan with the requested direction.", deferred=False)

    if product_column and _contains_any(question, ["products are best sellers", "best sellers"]):
        rank_metric = order_column or product_column
        reason = f"Rank {product_column} by distinct {rank_metric} to identify the best-selling products."
        add_step(
            ComputationStep(
                operation="rank_entities",
                column=rank_metric,
                parameters={"entity_column": product_column, "method": "distinct_count", "top_n": 10, "sort": "desc", "intent_type": "aggregation"},
                justification=reason,
                trace=_trace(reason, 0.82, ["segment_ranking", "best_sellers"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a best-selling product ranking plan.", deferred=False)

    if product_column and _contains_any(question, ["products are rarely sold", "rarely sold"]):
        rank_metric = order_column or product_column
        reason = f"Rank {product_column} by distinct {rank_metric} in ascending order to surface rarely sold products."
        add_step(
            ComputationStep(
                operation="rank_entities",
                column=rank_metric,
                parameters={"entity_column": product_column, "method": "distinct_count", "top_n": 10, "sort": "asc", "intent_type": "aggregation"},
                justification=reason,
                trace=_trace(reason, 0.82, ["segment_ranking", "long_tail_products"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a rarely sold product ranking plan.", deferred=False)

    if _contains_any(question, ["customer segments spend most", "segments spend most"]):
        spend_metric = primary_metric[0] if primary_metric else None
        if business_metric_candidates:
            spend_metric = next(
                (
                    candidate for candidate in business_metric_candidates
                    if "installment" not in str(candidate).lower()
                    and any(token in str(candidate).lower() for token in ["payment", "amount", "value", "price", "revenue", "sales", "total"])
                ),
                business_metric_candidates[0],
            )
        broad_customer_geo = _best_text_match(question + " customer state city region geography", _all_columns(dataset_profile), ["customer", "state", "city", "region"])
        segment_column = customer_segment_column or broad_customer_geo or customer_geo_column or geography_column or category_column
        if segment_column and spend_metric:
            reason = f"Rank {segment_column} by total {spend_metric} to identify the highest-spending customer segments."
            add_step(
                ComputationStep(
                    operation="rank_entities",
                    column=spend_metric,
                    parameters={"entity_column": segment_column, "method": "sum", "top_n": 10, "sort": "desc", "intent_type": "aggregation"},
                    justification=reason,
                    trace=_trace(reason, 0.82, ["customer_segment_spend", "revenue_metric"]),
                )
            )
            return ComputationPlanModel(
                steps=steps,
                confidence_score=0.82,
                justification="Built a customer-segment spend ranking plan using the best available segment column.",
                deferred=False,
            )

    if _contains_any(question, ["becoming more loyal over time", "loyal over time"]) and customer_column and order_column and time_column:
        reason = f"Measure repeat-customer share by period to determine whether customer loyalty is strengthening over time."
        add_step(
            ComputationStep(
                operation="loyalty_trend",
                column=time_column,
                parameters={"entity_column": customer_column, "order_column": order_column, "bucket": "month", "intent_type": "temporal"},
                justification=reason,
                trace=_trace(reason, 0.82, ["loyalty_trend", "customer_identifier", "timestamp_resolved"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.82,
            justification="Built a monthly loyalty-trend plan using repeat-customer share over time.",
            deferred=False,
        )

    if _contains_any(question, ["locations have strongest repeat behavior", "strongest repeat behavior"]) and customer_column and order_column:
        broad_customer_geo = _best_text_match(question + " customer state city region geography", _all_columns(dataset_profile), ["customer", "state", "city", "region"])
        segment_column = customer_segment_column or broad_customer_geo or customer_geo_column or geography_column
        if segment_column:
            reason = f"Measure repeat-customer share within each {segment_column} to identify where repeat behavior is strongest."
            add_step(
                ComputationStep(
                    operation="segment_repeat_rate",
                    column=order_column,
                    parameters={"entity_column": customer_column, "group_column": segment_column, "intent_type": "composition"},
                    justification=reason,
                    trace=_trace(reason, 0.82, ["segment_repeat_rate", "customer_identifier", "grouping_key"]),
                )
            )
            return ComputationPlanModel(
                steps=steps,
                confidence_score=0.82,
                justification="Built a location-level repeat-behavior ranking plan.",
                deferred=False,
            )

    if _contains_any(question, ["high-review customers return more often", "return more often"]) and customer_column and order_column and review_metric:
        reason = f"Compare repeat-customer behavior between high-review and lower-review customers using {review_metric}."
        add_step(
            ComputationStep(
                operation="review_repeat_comparison",
                column=review_metric,
                parameters={"entity_column": customer_column, "order_column": order_column, "intent_type": "comparison"},
                justification=reason,
                trace=_trace(reason, 0.8, ["review_repeat_comparison", "customer_identifier", "review_signal"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.8,
            justification="Built a comparison plan between customer review quality and repeat-purchase behavior.",
            deferred=False,
        )

    if focus_dimension and signals.get("asks_geography") and order_column and _contains_any(question, ["average order value", "aov"]):
        value_metric = resolved_revenue_metric or _best_value_metric(dataset_profile, inferred_context, relationship_signals) or (primary_metric[0] if primary_metric else None)
        if value_metric:
            reason = f"Compute order-level totals and rank {focus_dimension} by average order value instead of total revenue."
            add_step(
                ComputationStep(
                    operation="segment_order_value",
                    column=value_metric,
                    parameters={"entity_column": focus_dimension, "order_column": order_column, "value_method": "sum", "group_method": "mean", "sort": "desc", "top_n": 10, "intent_type": "aggregation"},
                    justification=reason,
                    trace=_trace(reason, 0.84, ["geographic_aov", "order_level_metric"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a geographic average-order-value ranking plan from the resolved focus dimension.", deferred=False)

    if (
        focus_dimension
        and signals.get("asks_ranking")
        and signals.get("asks_revenue")
        and not _looks_like_share_question(question)
        and (resolved_revenue_metric or primary_metric)
    ):
        revenue_metric = resolved_revenue_metric or primary_metric[0]
        add_step(
            _rank_entities_step(
                focus_dimension,
                revenue_metric,
                "sum",
                question,
                0.84,
                ["entity_ranking", "revenue_metric"],
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a generic entity revenue ranking plan from the resolved focus dimension.", deferred=False)

    if focus_dimension and signals.get("asks_ranking") and signals.get("asks_demand") and (order_column or customer_column):
        demand_metric = order_column or customer_column
        add_step(
            _rank_entities_step(
                focus_dimension,
                demand_metric,
                "distinct_count",
                question,
                0.84,
                ["entity_ranking", "demand_metric"],
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a generic entity demand ranking plan from the resolved focus dimension.", deferred=False)

    if focus_dimension and signals.get("asks_quality") and review_metric:
        sort_dir = "asc" if _contains_any(question, ["worst", "poor", "lowest"]) else "desc"
        add_step(
            _rank_entities_step(
                focus_dimension,
                review_metric,
                "mean",
                question,
                0.82,
                ["entity_quality", "review_signal"],
                sort=sort_dir,
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a generic entity quality ranking plan from the resolved focus dimension.", deferred=False)

    if focus_dimension and signals.get("asks_geography") and order_column and _contains_any(question, ["average order value", "aov"]):
        value_metric = resolved_revenue_metric or _best_value_metric(dataset_profile, inferred_context, relationship_signals) or (primary_metric[0] if primary_metric else None)
        if value_metric:
            add_step(
                _segment_order_value_step(
                    focus_dimension,
                    order_column,
                    value_metric,
                    0.84,
                    ["geographic_aov", "order_level_metric"],
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a geographic average-order-value ranking plan from the resolved focus dimension.", deferred=False)

    if focus_dimension and signals.get("asks_geography") and _contains_any(question, ["expensive", "cost", "serve", "cost most to deliver", "cost to deliver", "expensive to deliver"]):
        freight_metric = resolved_roles.get("freight_metric") or _best_text_match(question + " freight serve shipping", _all_columns(dataset_profile), ["freight", "shipping"])
        if freight_metric:
            add_step(
                _rank_entities_step(
                    focus_dimension,
                    freight_metric,
                    "mean",
                    question,
                    0.78,
                    ["cost_to_serve", "freight_metric"],
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a geographic cost-to-serve proxy plan from the resolved focus dimension.", deferred=False)

    if focus_dimension and signals.get("asks_delivery"):
        delivered_col = resolved_roles.get("delivered_column") or _best_text_match(question + " delivered delivery", _all_columns(dataset_profile), ["delivered", "delivery"])
        purchase_col = resolved_roles.get("purchase_column") or _best_text_match(question + " purchase approved order", _all_columns(dataset_profile), ["purchase", "approved", "order"])
        if delivered_col and purchase_col:
            sort_dir = "asc" if _contains_any(question, ["fastest", "fast"]) else "desc"
            reason = f"Measure fulfillment duration by {focus_dimension} to compare delivery speed across the resolved entity."
            add_step(
                ComputationStep(
                    operation="delay_burden_rank",
                    column=delivered_col,
                    parameters={"entity_column": focus_dimension, "start_column": purchase_col, "end_column": delivered_col, "sort": sort_dir, "top_n": 10, "intent_type": "aggregation"},
                    justification=reason,
                    trace=_trace(reason, 0.82, ["delivery_duration", "entity_breakdown"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a generic entity delivery-duration ranking plan from the resolved focus dimension.", deferred=False)

    if focus_dimension and _contains_any(question, ["need intervention", "intervention"]) and order_column:
        delivery_end = resolved_roles.get("delivered_column") or _best_text_match(question + " delivered delivery", _all_columns(dataset_profile), ["delivered", "delivery"])
        delivery_start = resolved_roles.get("purchase_column") or _best_text_match(question + " purchase approved order", _all_columns(dataset_profile), ["purchase", "approved", "order"])
        revenue_metric = resolved_revenue_metric or _best_value_metric(dataset_profile, inferred_context, relationship_signals) or next((col for col in business_metric_candidates if any(token in str(col).lower() for token in ["payment", "revenue", "sales", "amount", "value"])), None) or (primary_metric[0] if primary_metric else None)
        reason = f"Score {focus_dimension} across revenue exposure, review quality, delivery speed, and cancellations to prioritize intervention candidates."
        add_step(
            ComputationStep(
                operation="entity_intervention_score",
                column=order_column,
                parameters={
                    "entity_column": focus_dimension,
                    "count_column": order_column,
                    "revenue_metric": revenue_metric,
                    "review_metric": review_metric,
                    "status_column": status_column,
                    "delivery_start_column": delivery_start,
                    "delivery_end_column": delivery_end,
                    "top_n": 10,
                    "intent_type": "comparison",
                },
                justification=reason,
                trace=_trace(reason, 0.76, ["composite_entity_diagnostic", "intervention_priority"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.76, justification="Built a generic composite entity intervention-priority plan.", deferred=False)

    if (
        focus_dimension
        and signals.get("asks_growth")
        and time_column
        and (order_column or customer_column or primary_metric)
        and not _contains_any(
            question,
            [
                "customer growth rate",
                "order growth rate",
                "revenue growth rate",
                "month-over-month",
                "quarter-over-quarter",
            ],
        )
    ):
        growth_metric = primary_metric[0] if primary_metric and signals.get("asks_revenue") else (order_column or customer_column or (primary_metric[0] if primary_metric else None))
        growth_method = "sum" if primary_metric and growth_metric == primary_metric[0] and signals.get("asks_revenue") else "distinct_count"
        if growth_metric:
            reason = f"Measure period-over-period growth by {focus_dimension} and rank entities by their growth performance."
            add_step(
                ComputationStep(
                    operation="segment_growth_rank",
                    column=growth_metric,
                    parameters={"entity_column": focus_dimension, "time_column": time_column, "bucket": bucket, "method": growth_method, "sort": "desc", "top_n": 10, "intent_type": "temporal"},
                    justification=reason,
                    trace=_trace(reason, 0.8, ["entity_growth", "temporal_breakdown"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a generic entity growth ranking plan from the resolved focus dimension.", deferred=False)

    if focus_dimension and _contains_any(question, ["dominate certain categories", "risky dependencies"]) and category_column and (primary_metric or order_column):
        concentration_metric = primary_metric[0] if primary_metric and signals.get("asks_revenue") else (order_column or (primary_metric[0] if primary_metric else None))
        concentration_method = "sum" if primary_metric and concentration_metric == primary_metric[0] and signals.get("asks_revenue") else "distinct_count"
        if concentration_metric:
            reason = f"Measure concentration of {focus_dimension} within each {category_column} to identify dominance and dependency risk."
            add_step(
                ComputationStep(
                    operation="concentration_score",
                    column=concentration_metric,
                    parameters={"parent_column": category_column, "child_column": focus_dimension, "method": concentration_method, "top_n": 10, "intent_type": "composition"},
                    justification=reason,
                    trace=_trace(reason, 0.78, ["concentration", "dependency_risk"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a generic concentration plan from the resolved focus dimension and category context.", deferred=False)

    if _contains_any(question, ["becoming more loyal over time", "loyal over time"]) and customer_column and order_column and time_column:
        reason = f"Measure repeat-customer share by period to determine whether customer loyalty is strengthening over time."
        add_step(
            ComputationStep(
                operation="loyalty_trend",
                column=time_column,
                parameters={"entity_column": customer_column, "order_column": order_column, "bucket": "month", "intent_type": "temporal"},
                justification=reason,
                trace=_trace(reason, 0.82, ["loyalty_trend", "customer_identifier", "timestamp_resolved"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.82,
            justification="Built a monthly loyalty-trend plan using repeat-customer share over time.",
            deferred=False,
        )

    if _contains_any(question, ["over time", "trend", "monthly sales", "monthly revenue", "sales trends", "revenue generated over time"]) and time_column and primary_metric:
        group_reason = f"Group by {time_column} in {bucket} buckets because the question explicitly asks for performance over time."
        add_step(
            ComputationStep(
                operation="group_by",
                column=time_column,
                parameters={"intent_type": "aggregation", "bucket": bucket},
                justification=group_reason,
                trace=_trace(group_reason, 0.86, ["temporal_intent", "timestamp_resolved"]),
            )
        )
        agg_reason = f"Aggregate {primary_metric[0]} within each {bucket} bucket to produce the requested time series."
        add_step(
            ComputationStep(
                operation="aggregate",
                column=primary_metric[0],
                parameters={"method": "sum", "within": time_column, "intent_type": "aggregation"},
                justification=agg_reason,
                trace=_trace(agg_reason, 0.84, ["time_series_aggregation"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.84,
            justification="Built a temporal KPI plan with explicit time bucketing and grouped aggregation.",
            deferred=False,
        )

    if _looks_like_count_question(question):
        if _contains_any(question, ["unique customer", "customers purchased"]) and customer_column:
            reason = f"Count distinct values of {customer_column} because the question asks for unique purchasing customers."
            add_step(
                ComputationStep(
                    operation="distinct_count",
                    column=customer_column,
                    justification=reason,
                    trace=_trace(reason, 0.9, ["distinct_count_intent", "customer_identifier"]),
                )
            )
            return ComputationPlanModel(
                steps=steps,
                confidence_score=0.9,
                justification="Built a distinct-customer count plan from the resolved customer identifier.",
                deferred=False,
            )
        if _contains_any(question, ["total orders", "orders were placed"]) and order_column:
            reason = f"Count distinct values of {order_column} because the question asks for total orders rather than row counts or sums."
            add_step(
                ComputationStep(
                    operation="distinct_count",
                    column=order_column,
                    justification=reason,
                    trace=_trace(reason, 0.9, ["distinct_count_intent", "order_identifier"]),
                )
            )
            return ComputationPlanModel(
                steps=steps,
                confidence_score=0.9,
                justification="Built a distinct-order count plan from the resolved order identifier.",
                deferred=False,
            )

    if _contains_any(question, ["average order value", "aov"]) and primary_metric and order_column and not signals.get("asks_geography"):
        group_reason = f"Group by {order_column} because average order value must be computed at the order level first."
        add_step(
            ComputationStep(
                operation="group_by",
                column=order_column,
                parameters={"intent_type": "aggregation"},
                justification=group_reason,
                trace=_trace(group_reason, 0.88, ["order_level_metric"]),
            )
        )
        agg_reason = f"Sum {primary_metric[0]} within each order to construct order-level revenue."
        add_step(
            ComputationStep(
                operation="aggregate",
                column=primary_metric[0],
                parameters={"method": "sum", "within": order_column, "intent_type": "aggregation"},
                justification=agg_reason,
                trace=_trace(agg_reason, 0.88, ["order_level_metric"]),
            )
        )
        mean_reason = "Average the order-level totals to compute AOV."
        add_step(
            ComputationStep(
                operation="aggregate",
                column=primary_metric[0],
                parameters={"method": "mean", "scope": "group_results", "intent_type": "aggregation"},
                justification=mean_reason,
                trace=_trace(mean_reason, 0.88, ["aov_definition"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.88,
            justification="Built an order-level aggregation plan for average order value instead of averaging merged line rows directly.",
            deferred=False,
        )

    if _contains_any(question, ["average revenue per customer"]) and primary_metric and customer_column:
        group_reason = f"Group by {customer_column} because revenue per customer must be computed at the customer level."
        add_step(
            ComputationStep(
                operation="group_by",
                column=customer_column,
                parameters={"intent_type": "aggregation"},
                justification=group_reason,
                trace=_trace(group_reason, 0.88, ["customer_level_metric"]),
            )
        )
        agg_reason = f"Sum {primary_metric[0]} within each customer to construct customer-level revenue."
        add_step(
            ComputationStep(
                operation="aggregate",
                column=primary_metric[0],
                parameters={"method": "sum", "within": customer_column, "intent_type": "aggregation"},
                justification=agg_reason,
                trace=_trace(agg_reason, 0.88, ["customer_level_metric"]),
            )
        )
        mean_reason = "Average the customer-level totals to compute average revenue per customer."
        add_step(
            ComputationStep(
                operation="aggregate",
                column=primary_metric[0],
                parameters={"method": "mean", "scope": "group_results", "intent_type": "aggregation"},
                justification=mean_reason,
                trace=_trace(mean_reason, 0.88, ["customer_revenue_definition"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.88,
            justification="Built a customer-level aggregation plan for average revenue per customer.",
            deferred=False,
        )

    revenue_share_metric = (
        resolved_revenue_metric
        or _best_text_match(
            question + " revenue payment sales amount total value",
            _all_columns(dataset_profile),
            ["payment", "revenue", "sales", "value", "amount", "total"],
        )
        or next(
            (
                candidate for candidate in business_metric_candidates
                if "installment" not in str(candidate).lower()
                and not any(token in str(candidate).lower() for token in ["freight", "shipping"])
            ),
            business_metric_candidates[0] if business_metric_candidates else None,
        )
        or (primary_metric[0] if primary_metric else None)
    )

    if _looks_like_share_question(question) and revenue_share_metric:
        if _contains_any(question, ["top customers"]) and customer_column:
            reason = f"Compute the share of total {revenue_share_metric} contributed by the top customers."
            add_step(
                ComputationStep(
                    operation="share_of_total",
                    column=revenue_share_metric,
                    parameters={"entity_column": customer_column, "top_n": 10, "intent_type": "composition"},
                    justification=reason,
                    trace=_trace(reason, 0.82, ["share_of_total", "customer_identifier"]),
                )
            )
            return ComputationPlanModel(
                steps=steps,
                confidence_score=0.82,
                justification="Built a top-customer share-of-revenue plan using grouped customer contribution.",
                deferred=False,
            )
        if _contains_any(question, ["repeat purchases"]) and customer_column and order_column:
            reason = f"Measure the share of customers with more than one distinct {order_column}."
            add_step(
                ComputationStep(
                    operation="repeat_rate",
                    column=order_column,
                    parameters={"entity_column": customer_column, "intent_type": "composition"},
                    justification=reason,
                    trace=_trace(reason, 0.86, ["repeat_purchase_rate", "customer_identifier", "order_identifier"]),
                )
            )
            return ComputationPlanModel(
                steps=steps,
                confidence_score=0.86,
                justification="Built a repeat-purchase rate plan using distinct orders per customer.",
                deferred=False,
            )

    if _contains_any(question, ["return to buy again", "return to buy", "buy again", "return again"]) and customer_column and order_column:
        reason = f"Measure distinct {order_column} per {customer_column} to quantify how often customers return to purchase again."
        add_step(
            ComputationStep(
                operation="customer_order_frequency",
                column=order_column,
                parameters={"entity_column": customer_column, "intent_type": "composition"},
                justification=reason,
                trace=_trace(reason, 0.86, ["repeat_purchase_frequency", "customer_identifier", "order_identifier"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.86,
            justification="Built a customer repeat-frequency plan using distinct orders per customer.",
            deferred=False,
        )

    if _contains_any(question, ["repeat purchase rate"]) and customer_column and order_column:
        reason = f"Measure the share of customers with more than one distinct {order_column}."
        add_step(
            ComputationStep(
                operation="repeat_rate",
                column=order_column,
                parameters={"entity_column": customer_column, "intent_type": "composition"},
                justification=reason,
                trace=_trace(reason, 0.88, ["repeat_purchase_rate", "customer_identifier", "order_identifier"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.88,
            justification="Built a repeat-purchase rate plan using distinct orders per customer.",
            deferred=False,
        )

    if _contains_any(question, ["first and second purchase", "between first and second purchase"]) and customer_column and order_column and time_column:
        reason = f"Measure the elapsed time between the first and second distinct {order_column} for each {customer_column}."
        add_step(
            ComputationStep(
                operation="purchase_gap",
                column=time_column,
                parameters={"entity_column": customer_column, "order_column": order_column, "intent_type": "temporal"},
                justification=reason,
                trace=_trace(reason, 0.84, ["purchase_sequence", "customer_identifier", "timestamp_resolved"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.84,
            justification="Built a purchase-sequence gap plan from customer-level first and second purchase timing.",
            deferred=False,
        )

    if _contains_any(question, ["average customer lifetime value proxy", "customer lifetime value proxy", "lifetime value proxy"]) and customer_column:
        ltv_metric = primary_metric[0] if primary_metric else None
        if business_metric_candidates:
            ltv_metric = business_metric_candidates[0]
        if not ltv_metric:
            return ComputationPlanModel(
                steps=[],
                confidence_score=0.3,
                justification="Customer lifetime value proxy requires a reliable revenue-like metric, which could not be resolved.",
                deferred=True,
            )
        group_reason = f"Group by {customer_column} because lifetime value proxy should be computed at the customer level first."
        add_step(
            ComputationStep(
                operation="group_by",
                column=customer_column,
                parameters={"intent_type": "aggregation"},
                justification=group_reason,
                trace=_trace(group_reason, 0.88, ["customer_level_metric"]),
            )
        )
        agg_reason = f"Sum {ltv_metric} within each customer to construct a customer lifetime value proxy."
        add_step(
            ComputationStep(
                operation="aggregate",
                column=ltv_metric,
                parameters={"method": "sum", "within": customer_column, "intent_type": "aggregation"},
                justification=agg_reason,
                trace=_trace(agg_reason, 0.88, ["customer_ltv_proxy"]),
            )
        )
        mean_reason = "Average the customer-level totals to estimate the average customer lifetime value proxy."
        add_step(
            ComputationStep(
                operation="aggregate",
                column=ltv_metric,
                parameters={"method": "mean", "scope": "group_results", "intent_type": "aggregation"},
                justification=mean_reason,
                trace=_trace(mean_reason, 0.88, ["customer_ltv_proxy"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.88,
            justification="Built a customer-level revenue rollup as a lifetime value proxy.",
            deferred=False,
        )

    if _contains_any(question, ["only buy once", "buy once"]) and customer_column and order_column:
        reason = f"Measure the share of customers with exactly one distinct {order_column}."
        add_step(
            ComputationStep(
                operation="single_purchase_share",
                column=order_column,
                parameters={"entity_column": customer_column, "intent_type": "composition"},
                justification=reason,
                trace=_trace(reason, 0.86, ["single_purchase_share", "customer_identifier", "order_identifier"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.86,
            justification="Built a single-purchase customer share plan using distinct orders per customer.",
            deferred=False,
        )

    if _contains_any(question, ["customer segments spend most", "segments spend most"]):
        spend_metric = primary_metric[0] if primary_metric else None
        if business_metric_candidates:
            spend_metric = next(
                (
                    candidate for candidate in business_metric_candidates
                    if "installment" not in str(candidate).lower()
                    and any(token in str(candidate).lower() for token in ["payment", "amount", "value", "price", "revenue", "sales", "total"])
                ),
                business_metric_candidates[0],
            )
        segment_column = customer_segment_column or customer_geo_column or geography_column or category_column
        if segment_column and spend_metric:
            reason = f"Rank {segment_column} by total {spend_metric} to identify the highest-spending customer segments."
            add_step(
                ComputationStep(
                    operation="rank_entities",
                    column=spend_metric,
                    parameters={"entity_column": segment_column, "method": "sum", "top_n": 10, "sort": "desc", "intent_type": "aggregation"},
                    justification=reason,
                    trace=_trace(reason, 0.82, ["customer_segment_spend", "revenue_metric"]),
                )
            )
            return ComputationPlanModel(
                steps=steps,
                confidence_score=0.82,
                justification="Built a customer-segment spend ranking plan using the best available segment column.",
                deferred=False,
            )

    if _contains_any(question, ["many cheap items", "few expensive ones"]) and order_column:
        basket_metric = primary_metric[0] if primary_metric else None
        if business_metric_candidates:
            basket_metric = business_metric_candidates[0]
        if not basket_metric:
            return ComputationPlanModel(
                steps=[],
                confidence_score=0.3,
                justification="Basket-value analysis requires a reliable price or spend metric, which could not be resolved.",
                deferred=True,
            )
        reason = "Compare order-level item counts with average item value to determine whether purchase behavior skews toward many cheap items or few expensive ones."
        add_step(
            ComputationStep(
                operation="basket_value_pattern",
                column=basket_metric,
                parameters={"entity_column": order_column, "count_column": basket_metric, "intent_type": "comparison"},
                justification=reason,
                trace=_trace(reason, 0.78, ["basket_pattern", "order_level_behavior"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.78,
            justification="Built an order-level basket-versus-value pattern plan.",
            deferred=False,
        )

    if _contains_any(question, ["dormant customers"]) and customer_column and time_column:
        reason = f"Measure recency from the latest observed {time_column} to estimate how many customers are currently dormant."
        add_step(
            ComputationStep(
                operation="dormancy_count",
                column=time_column,
                parameters={"entity_column": customer_column, "intent_type": "aggregation"},
                justification=reason,
                trace=_trace(reason, 0.8, ["dormancy", "customer_identifier", "timestamp_resolved"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.8,
            justification="Built a customer dormancy plan from last-purchase recency.",
            deferred=False,
        )

    if _contains_any(question, ["locations have strongest repeat behavior", "strongest repeat behavior"]) and customer_column and order_column:
        segment_column = customer_segment_column or customer_geo_column or geography_column
        if segment_column:
            reason = f"Measure repeat-customer share within each {segment_column} to identify where repeat behavior is strongest."
            add_step(
                ComputationStep(
                    operation="segment_repeat_rate",
                    column=order_column,
                    parameters={"entity_column": customer_column, "group_column": segment_column, "intent_type": "composition"},
                    justification=reason,
                    trace=_trace(reason, 0.82, ["segment_repeat_rate", "customer_identifier", "grouping_key"]),
                )
            )
            return ComputationPlanModel(
                steps=steps,
                confidence_score=0.82,
                justification="Built a location-level repeat-behavior ranking plan.",
                deferred=False,
            )

    if _contains_any(question, ["high-review customers return more often", "return more often"]) and customer_column and order_column and review_metric:
        reason = f"Compare repeat-customer behavior between high-review and lower-review customers using {review_metric}."
        add_step(
            ComputationStep(
                operation="review_repeat_comparison",
                column=review_metric,
                parameters={"entity_column": customer_column, "order_column": order_column, "intent_type": "comparison"},
                justification=reason,
                trace=_trace(reason, 0.8, ["review_repeat_comparison", "customer_identifier", "review_signal"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.8,
            justification="Built a comparison plan between customer review quality and repeat-purchase behavior.",
            deferred=False,
        )

    if _contains_any(question, ["profit proxy", "after freight"]):
        freight_column = _best_text_match(question + " freight", _all_columns(dataset_profile), ["freight"])
        profit_metric = (
            resolved_revenue_metric
            or _best_text_match(
                question + " revenue payment sales amount total value",
                _all_columns(dataset_profile),
                ["payment", "revenue", "sales", "value", "amount", "total"],
            )
            or next(
                (
                    candidate for candidate in business_metric_candidates
                    if "installment" not in str(candidate).lower()
                    and not any(token in str(candidate).lower() for token in ["freight", "shipping"])
                ),
                business_metric_candidates[0] if business_metric_candidates else None,
            )
            or (primary_metric[0] if primary_metric else None)
        )
        if freight_column and profit_metric and freight_column != profit_metric:
            reason = f"Construct a profit proxy by subtracting {freight_column} from {profit_metric} before aggregation."
            add_step(
                ComputationStep(
                    operation="row_expression",
                    column=profit_metric,
                    parameters={"subtract_column": freight_column, "expression_name": "profit_proxy", "intent_type": "aggregation"},
                    justification=reason,
                    trace=_trace(reason, 0.84, ["derived_metric", "profit_proxy"]),
                )
            )
            agg_reason = "Aggregate the derived profit proxy to answer the requested proxy metric."
            add_step(
                ComputationStep(
                    operation="aggregate",
                    column="profit_proxy",
                    parameters={"method": "sum", "intent_type": "aggregation"},
                    justification=agg_reason,
                    trace=_trace(agg_reason, 0.84, ["derived_metric", "aggregation_mapping"]),
                )
            )
            return ComputationPlanModel(
                steps=steps,
                confidence_score=0.84,
                justification="Built a derived-metric plan for profit proxy after freight costs.",
                deferred=False,
            )

    if _looks_like_growth_question(question) and time_column:
        group_reason = f"Group by {time_column} in {bucket} buckets because the question asks for period-over-period growth."
        add_step(
            ComputationStep(
                operation="group_by",
                column=time_column,
                parameters={"intent_type": "temporal", "bucket": bucket},
                justification=group_reason,
                trace=_trace(group_reason, 0.86, ["growth_intent", "timestamp_resolved"]),
            )
        )
        if _contains_any(question, ["customer growth"]) and customer_column:
            growth_reason = f"Count distinct {customer_column} in each {bucket} bucket and compute period-over-period growth."
            add_step(
                ComputationStep(
                    operation="growth_rate",
                    column=customer_column,
                    parameters={"method": "distinct_count", "entity_column": customer_column, "intent_type": "temporal"},
                    justification=growth_reason,
                    trace=_trace(growth_reason, 0.86, ["growth_rate", "customer_identifier"]),
                )
            )
        elif _contains_any(question, ["order growth"]) and order_column:
            growth_reason = f"Count distinct {order_column} in each {bucket} bucket and compute period-over-period growth."
            add_step(
                ComputationStep(
                    operation="growth_rate",
                    column=order_column,
                    parameters={"method": "distinct_count", "entity_column": order_column, "intent_type": "temporal"},
                    justification=growth_reason,
                    trace=_trace(growth_reason, 0.86, ["growth_rate", "order_identifier"]),
                )
            )
        elif primary_metric:
            growth_reason = f"Aggregate {primary_metric[0]} in each {bucket} bucket and compute period-over-period growth."
            add_step(
                ComputationStep(
                    operation="growth_rate",
                    column=primary_metric[0],
                    parameters={"method": "sum", "intent_type": "temporal"},
                    justification=growth_reason,
                    trace=_trace(growth_reason, 0.84, ["growth_rate", "time_series_aggregation"]),
                )
            )
        if _contains_any(question, ["growing", "flat", "declining"]):
            trend_reason = "Classify the resulting period growth pattern as growing, flat, or declining."
            add_step(
                ComputationStep(
                    operation="trend_classification",
                    column=primary_metric[0] if primary_metric else (order_column or customer_column or time_column),
                    parameters={"source": "growth_rows", "intent_type": "temporal"},
                    justification=trend_reason,
                    trace=_trace(trend_reason, 0.8, ["trend_classification"]),
                )
            )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.85,
            justification="Built a temporal growth-rate plan with explicit period bucketing and period-over-period computation.",
            deferred=False,
        )

    period_metric = (
        resolved_revenue_metric
        or _best_text_match(
            question + " revenue payment sales amount total value",
            _all_columns(dataset_profile),
            ["payment", "revenue", "sales", "value", "amount", "total"],
        )
        or next(
            (
                candidate for candidate in business_metric_candidates
                if "installment" not in str(candidate).lower()
                and not any(token in str(candidate).lower() for token in ["freight", "shipping"])
            ),
            business_metric_candidates[0] if business_metric_candidates else None,
        )
        or (primary_metric[0] if primary_metric else None)
    )

    if _contains_any(question, ["best and worst"]) and _contains_any(question, ["month", "months"]) and time_column and period_metric:
        group_reason = f"Group by {time_column} in month buckets because the question asks which months performed best and worst."
        add_step(
            ComputationStep(
                operation="group_by",
                column=time_column,
                parameters={"intent_type": "temporal", "bucket": "month"},
                justification=group_reason,
                trace=_trace(group_reason, 0.85, ["temporal_intent", "extremes"]),
            )
        )
        agg_reason = f"Aggregate {period_metric} within each month to rank monthly performance."
        add_step(
            ComputationStep(
                operation="aggregate",
                column=period_metric,
                parameters={"method": "sum", "within": time_column, "intent_type": "aggregation"},
                justification=agg_reason,
                trace=_trace(agg_reason, 0.84, ["monthly_ranking"]),
            )
        )
        extrema_reason = "Identify the strongest and weakest monthly periods from the grouped monthly totals."
        add_step(
            ComputationStep(
                operation="period_extremes",
                column=period_metric,
                parameters={"top_n": 1, "bottom_n": 1, "intent_type": "temporal"},
                justification=extrema_reason,
                trace=_trace(extrema_reason, 0.84, ["period_extremes", "monthly_ranking"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.84,
            justification="Built a monthly aggregation plan to identify best and worst periods.",
            deferred=False,
        )

    if _contains_any(question, ["growing, flat, or declining", "growing", "flat", "declining"]) and not _contains_any(question, ["categories", "products"]) and time_column and primary_metric:
        group_reason = f"Group by {time_column} in month buckets because the question asks for directional trend classification."
        add_step(
            ComputationStep(
                operation="group_by",
                column=time_column,
                parameters={"intent_type": "temporal", "bucket": "month"},
                justification=group_reason,
                trace=_trace(group_reason, 0.84, ["trend_intent", "timestamp_resolved"]),
            )
        )
        growth_reason = f"Aggregate {primary_metric[0]} by month and compute period-over-period growth before classifying the trend."
        add_step(
            ComputationStep(
                operation="growth_rate",
                column=primary_metric[0],
                parameters={"method": "sum", "intent_type": "temporal"},
                justification=growth_reason,
                trace=_trace(growth_reason, 0.84, ["trend_intent", "growth_rate"]),
            )
        )
        trend_reason = "Classify the average period growth as growing, flat, or declining."
        add_step(
            ComputationStep(
                operation="trend_classification",
                column=primary_metric[0],
                parameters={"source": "growth_rows", "intent_type": "temporal"},
                justification=trend_reason,
                trace=_trace(trend_reason, 0.82, ["trend_classification"]),
            )
        )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.84,
            justification="Built a temporal trend-classification plan from grouped revenue growth.",
            deferred=False,
        )

    if _contains_any(question, ["scaling efficiently", "lower quality"]) and time_column and primary_metric:
        group_reason = f"Group by {time_column} in month buckets to compare commercial growth with quality signals over time."
        add_step(
            ComputationStep(
                operation="group_by",
                column=time_column,
                parameters={"intent_type": "temporal", "bucket": "month"},
                justification=group_reason,
                trace=_trace(group_reason, 0.72, ["composite_business_question", "timestamp_resolved"]),
            )
        )
        add_step(
            ComputationStep(
                operation="aggregate",
                column=primary_metric[0],
                parameters={"method": "sum", "within": time_column, "intent_type": "aggregation"},
                justification="Track revenue over time as the scale signal.",
                trace=_trace("Track revenue over time as the scale signal.", 0.72, ["scale_signal"]),
            )
        )
        quality_metric = _best_text_match(question + " review quality", dataset_profile.get("column_names", []) or [], ["review", "score", "quality"])
        if quality_metric:
            add_step(
                ComputationStep(
                    operation="aggregate",
                    column=quality_metric,
                    parameters={"method": "mean", "within": time_column, "intent_type": "aggregation"},
                    justification="Track average review quality over time as the quality signal.",
                    trace=_trace("Track average review quality over time as the quality signal.", 0.7, ["quality_signal"]),
                )
            )
        return ComputationPlanModel(
            steps=steps,
            confidence_score=0.68,
            justification="Built a composite monthly monitoring plan comparing scale signals with quality signals. This answers the question directionally but still requires executive interpretation.",
            deferred=False,
        )

    if signals.get("asks_geography") and geography_column:
        geo_column = geography_column
        demand_metric = order_column or customer_column
        if signals.get("asks_demand") and signals.get("asks_ranking") and demand_metric:
            method = "distinct_count" if demand_metric in {order_column, customer_column} else "sum"
            reason = f"Rank {geo_column} by the best available demand metric to identify the strongest-performing locations."
            add_step(
                ComputationStep(
                    operation="rank_entities",
                    column=demand_metric,
                    parameters={"entity_column": geo_column, "method": method, "top_n": 10, "sort": "desc", "intent_type": "aggregation"},
                    justification=reason,
                    trace=_trace(reason, 0.84, ["geographic_ranking", "demand_metric"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a geographic demand ranking plan.", deferred=False)

        if signals.get("asks_price") and signals.get("asks_ranking") and primary_metric and order_column:
            reason = f"Compute order-level totals and rank {geo_column} by average order value."
            add_step(
                ComputationStep(
                    operation="segment_order_value",
                    column=primary_metric[0],
                    parameters={"entity_column": geo_column, "order_column": order_column, "value_method": "sum", "group_method": "mean", "sort": "desc", "top_n": 10, "intent_type": "aggregation"},
                    justification=reason,
                    trace=_trace(reason, 0.84, ["geographic_aov", "order_level_metric"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a geographic average-order-value ranking plan.", deferred=False)

        if signals.get("asks_revenue") and signals.get("asks_ranking") and primary_metric:
            reason = f"Rank {geo_column} by total {primary_metric[0]} to identify the highest-revenue locations."
            add_step(
                ComputationStep(
                    operation="rank_entities",
                    column=primary_metric[0],
                    parameters={"entity_column": geo_column, "method": "sum", "top_n": 10, "sort": "desc", "intent_type": "aggregation"},
                    justification=reason,
                    trace=_trace(reason, 0.84, ["geographic_ranking", "revenue_metric"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a geographic revenue ranking plan.", deferred=False)

        if signals.get("asks_quality") and review_metric:
            reason = f"Rank {geo_column} by average {review_metric} in ascending order to surface weaker customer-experience locations."
            add_step(
                ComputationStep(
                    operation="rank_entities",
                    column=review_metric,
                    parameters={"entity_column": geo_column, "method": "mean", "top_n": 10, "sort": "asc", "intent_type": "aggregation"},
                    justification=reason,
                    trace=_trace(reason, 0.82, ["geographic_quality", "review_signal"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a geographic quality ranking plan.", deferred=False)

        if _contains_any(question, ["expensive", "cost", "serve"]) and geography_column:
            freight_metric = _best_text_match(question + " freight serve shipping", _all_columns(dataset_profile), ["freight", "shipping"])
            if freight_metric:
                reason = f"Rank {geo_column} by average {freight_metric} to estimate which locations are more expensive to serve."
                add_step(
                    ComputationStep(
                        operation="rank_entities",
                        column=freight_metric,
                        parameters={"entity_column": geo_column, "method": "mean", "top_n": 10, "sort": "desc", "intent_type": "aggregation"},
                        justification=reason,
                        trace=_trace(reason, 0.78, ["cost_to_serve", "freight_metric"]),
                    )
                )
                return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a geographic cost-to-serve proxy plan.", deferred=False)

        if signals.get("asks_delivery"):
            delivered_col = _best_text_match(question + " delivered delivery", _all_columns(dataset_profile), ["delivered", "delivery"])
            purchase_col = _best_text_match(question + " purchase approved order", _all_columns(dataset_profile), ["purchase", "approved", "order"])
            if delivered_col and purchase_col:
                reason = f"Measure delivery duration by {geo_column} to identify locations with slower delivery performance."
                add_step(
                    ComputationStep(
                        operation="delay_burden_rank",
                        column=delivered_col,
                        parameters={"entity_column": geo_column, "start_column": purchase_col, "end_column": delivered_col, "sort": "desc", "top_n": 10, "intent_type": "aggregation"},
                        justification=reason,
                        trace=_trace(reason, 0.82, ["delivery_duration", "geographic_breakdown"]),
                    )
                )
                return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a geographic delivery-duration ranking plan.", deferred=False)

        if signals.get("asks_demand") and signals.get("asks_supply") and seller_column:
            reason = f"Compare geographic demand with seller presence to surface places where demand is high but supply is relatively limited."
            add_step(
                ComputationStep(
                    operation="segment_contrast",
                    column=seller_column,
                    parameters={
                        "entity_column": geo_column,
                        "primary_metric": order_column or customer_column,
                        "primary_method": "distinct_count",
                        "secondary_metric": seller_column,
                        "secondary_method": "distinct_count",
                        "pattern": "high_low",
                        "top_n": 10,
                        "intent_type": "comparison",
                    },
                    justification=reason,
                    trace=_trace(reason, 0.8, ["demand_supply_contrast", "multi_metric_reasoning"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a geographic demand-versus-seller-presence contrast plan.", deferred=False)

        if signals.get("asks_share") and customer_column:
            reason = f"Rank {geo_column} by distinct {customer_column} to measure where customers are most concentrated."
            add_step(
                ComputationStep(
                    operation="rank_entities",
                    column=customer_column,
                    parameters={"entity_column": geo_column, "method": "distinct_count", "top_n": 10, "sort": "desc", "intent_type": "composition"},
                    justification=reason,
                    trace=_trace(reason, 0.82, ["geographic_concentration", "customer_identifier"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a geographic customer-concentration ranking plan.", deferred=False)

        if signals.get("asks_coverage") and customer_column:
            reason = f"Rank {geo_column} by distinct {customer_column} in ascending order to surface lower-penetration areas."
            add_step(
                ComputationStep(
                    operation="rank_entities",
                    column=customer_column,
                    parameters={"entity_column": geo_column, "method": "distinct_count", "top_n": 10, "sort": "asc", "intent_type": "composition"},
                    justification=reason,
                    trace=_trace(reason, 0.76, ["geographic_coverage", "customer_identifier"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.76, justification="Built a geographic underpenetration heuristic using customer concentration.", deferred=False)

        if signals.get("asks_growth") and time_column and (order_column or customer_column):
            growth_metric = order_column or customer_column
            growth_method = "distinct_count"
            reason = f"Measure period-over-period growth by {geo_column} and rank locations by growth."
            add_step(
                ComputationStep(
                    operation="segment_growth_rank",
                    column=growth_metric,
                    parameters={"entity_column": geo_column, "time_column": time_column, "bucket": "month", "method": growth_method, "sort": "desc", "top_n": 10, "intent_type": "temporal"},
                    justification=reason,
                    trace=_trace(reason, 0.8, ["geographic_growth", "temporal_breakdown"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a geographic growth ranking plan.", deferred=False)


    if category_column and _contains_any(question, ["highest order volume but low revenue", "low volume but premium pricing"]):
        contrast_pattern = "high_low" if _contains_any(question, ["highest order volume but low revenue"]) else "low_high"
        secondary_metric = None
        if _contains_any(question, ["revenue"]) and business_metric_candidates:
            secondary_metric = business_metric_candidates[0]
        elif _contains_any(question, ["pricing", "price", "premium"]) and any("price" in str(col).lower() for col in business_metric_candidates):
            secondary_metric = next(col for col in business_metric_candidates if "price" in str(col).lower())
        elif primary_metric:
            secondary_metric = primary_metric[0]
        if order_column and secondary_metric:
            reason = f"Compare categories on both volume and value metrics to surface the requested contrast pattern."
            add_step(
                ComputationStep(
                    operation="segment_contrast",
                    column=secondary_metric,
                    parameters={
                        "entity_column": category_column,
                        "primary_metric": order_column,
                        "primary_method": "distinct_count",
                        "secondary_metric": secondary_metric,
                        "secondary_method": "mean" if contrast_pattern == "low_high" else "sum",
                        "pattern": contrast_pattern,
                        "top_n": 10,
                        "intent_type": "comparison",
                    },
                    justification=reason,
                    trace=_trace(reason, 0.8, ["segment_contrast", "multi_metric_reasoning"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a dual-metric contrast plan for category performance.", deferred=False)

    if _contains_any(question, ["product categories", "categories"]) and _contains_any(question, ["sell the most", "most sold", "highest order volume"]):
        entity_column = category_column
        if entity_column and order_column:
            reason = f"Rank {entity_column} by distinct {order_column} to identify the categories with the highest selling volume."
            add_step(
                ComputationStep(
                    operation="rank_entities",
                    column=order_column,
                    parameters={"entity_column": entity_column, "method": "distinct_count", "top_n": 10, "sort": "desc", "intent_type": "aggregation"},
                    justification=reason,
                    trace=_trace(reason, 0.86, ["segment_ranking", "volume_metric"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.86, justification="Built a category sales-volume ranking plan.", deferred=False)

    if _contains_any(question, ["product categories", "categories"]) and _contains_any(question, ["generate the most revenue", "most revenue"]):
        entity_column = category_column
        if entity_column and primary_metric:
            reason = f"Rank {entity_column} by total {primary_metric[0]} to identify the highest-revenue categories."
            add_step(
                ComputationStep(
                    operation="rank_entities",
                    column=primary_metric[0],
                    parameters={"entity_column": entity_column, "method": "sum", "top_n": 10, "sort": "desc", "intent_type": "aggregation"},
                    justification=reason,
                    trace=_trace(reason, 0.86, ["segment_ranking", "revenue_metric"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.86, justification="Built a category revenue ranking plan.", deferred=False)

    if category_column and time_column and _contains_any(question, ["categories are growing fastest", "categories are declining"]):
        direction = "desc" if _contains_any(question, ["growing fastest"]) else "asc"
        growth_metric = order_column if not primary_metric else primary_metric[0]
        growth_method = "distinct_count" if growth_metric == order_column else "sum"
        reason = f"Measure period-over-period growth by category and rank categories by the requested direction."
        add_step(
            ComputationStep(
                operation="segment_growth_rank",
                column=growth_metric,
                parameters={
                    "entity_column": category_column,
                    "time_column": time_column,
                    "bucket": "month",
                    "method": growth_method,
                    "sort": direction,
                    "top_n": 10,
                    "intent_type": "temporal",
                },
                justification=reason,
                trace=_trace(reason, 0.8, ["segment_growth", "temporal_breakdown"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a category growth ranking plan.", deferred=False)

    if category_column and time_column and _contains_any(question, ["categories are seasonal", "seasonality"]):
        seasonal_metric = order_column if order_column else (primary_metric[0] if primary_metric else None)
        if seasonal_metric:
            reason = f"Measure how unevenly each category performs across months to estimate seasonality."
            add_step(
                ComputationStep(
                    operation="segment_seasonality",
                    column=seasonal_metric,
                    parameters={
                        "entity_column": category_column,
                        "time_column": time_column,
                        "bucket": "month",
                        "method": "distinct_count" if seasonal_metric == order_column else "sum",
                        "top_n": 10,
                        "intent_type": "temporal",
                    },
                    justification=reason,
                    trace=_trace(reason, 0.76, ["segment_seasonality", "temporal_breakdown"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.76, justification="Built a category seasonality scoring plan.", deferred=False)

    if product_column and _contains_any(question, ["products are best sellers", "best sellers"]):
        rank_metric = order_column or product_column
        reason = f"Rank {product_column} by distinct {rank_metric} to identify best-selling products."
        add_step(
            ComputationStep(
                operation="rank_entities",
                column=rank_metric,
                parameters={"entity_column": product_column, "method": "distinct_count", "top_n": 10, "sort": "desc", "intent_type": "aggregation"},
                justification=reason,
                trace=_trace(reason, 0.84, ["segment_ranking", "product_volume"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.84, justification="Built a best-seller product ranking plan.", deferred=False)

    if product_column and _contains_any(question, ["products are rarely sold", "rarely sold"]):
        rank_metric = order_column or product_column
        reason = f"Rank {product_column} by distinct {rank_metric} in ascending order to surface rarely sold products."
        add_step(
            ComputationStep(
                operation="rank_entities",
                column=rank_metric,
                parameters={"entity_column": product_column, "method": "distinct_count", "top_n": 10, "sort": "asc", "intent_type": "aggregation"},
                justification=reason,
                trace=_trace(reason, 0.82, ["segment_ranking", "long_tail_products"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.82, justification="Built a rarely sold product ranking plan.", deferred=False)

    if category_column and product_column and _contains_any(question, ["overdependent on a few products", "overdependent"]):
        concentration_metric = primary_metric[0] if primary_metric else (order_column or product_column)
        concentration_method = "sum" if primary_metric else "distinct_count"
        reason = f"Measure concentration within each {category_column} to identify dependence on a small number of products."
        add_step(
            ComputationStep(
                operation="concentration_score",
                column=concentration_metric,
                parameters={
                    "parent_column": category_column,
                    "child_column": product_column,
                    "method": concentration_method,
                    "top_n": 10,
                    "intent_type": "composition",
                },
                justification=reason,
                trace=_trace(reason, 0.78, ["concentration_analysis", "mix_dependency"]),
            )
        )
        return ComputationPlanModel(steps=steps, confidence_score=0.78, justification="Built a category concentration plan to detect overdependence on a few products.", deferred=False)

    if _contains_any(question, ["product mix drives total revenue", "product mix"]) and primary_metric:
        entity_column = category_column or product_column
        if entity_column:
            reason = f"Measure contribution by {entity_column} to identify which mix components drive total revenue."
            add_step(
                ComputationStep(
                    operation="rank_entities",
                    column=primary_metric[0],
                    parameters={"entity_column": entity_column, "method": "sum", "top_n": 10, "sort": "desc", "intent_type": "composition"},
                    justification=reason,
                    trace=_trace(reason, 0.8, ["mix_contribution", "revenue_metric"]),
                )
            )
            return ComputationPlanModel(steps=steps, confidence_score=0.8, justification="Built a product-mix contribution ranking plan.", deferred=False)

    for strategy in strategies:
        group_columns = _select_group_columns(
            question,
            user_intent,
            selected_columns,
            dataset_profile,
            inferred_context,
            strategy,
        )
        metric_columns = _select_metric_columns(
            question,
            user_intent,
            selected_columns,
            dataset_profile,
            inferred_context,
            relationship_signals,
            strategy,
        )
        all_group_columns.extend(group_columns)
        all_metric_columns.extend(metric_columns)

        for group_col in group_columns:
            reason = f"Group by {group_col} because it is the best structural match for the requested breakdown."
            add_step(
                ComputationStep(
                    operation="group_by",
                    column=group_col,
                    parameters={"intent_type": strategy},
                    justification=reason,
                    trace=_trace(reason, 0.8, [f"role:{_role(inferred_context, group_col)}"]),
                )
            )

        if strategy == "aggregation":
            agg_method = _aggregation_method(question, default=user_intent.get("aggregation") or "mean")
            if not metric_columns:
                confidence = min(confidence, 0.25)
                notes.append("No reliable metric column was found for aggregation.")
            elif _contains_any(question, ["summary", "statistics", "describe"]):
                for metric in metric_columns:
                    reason = f"Summarize {metric} directly because the user asked for descriptive statistics rather than a single aggregate."
                    add_step(
                        ComputationStep(
                            operation="summarize_metric",
                            column=metric,
                            parameters={"intent_type": strategy},
                            justification=reason,
                            trace=_trace(reason, 0.82, ["summary_statistics_intent"]),
                        )
                    )
            else:
                metric = metric_columns[0]
                group_col = group_columns[0] if group_columns else None
                metric_role = _role(inferred_context, metric)
                if group_col and agg_method == "mean" and metric_role == "derived_metric":
                    reason = f"Summarize {metric} within each {group_col} group before averaging because the metric behaves like a derived measure."
                    add_step(
                        ComputationStep(
                            operation="aggregate",
                            column=metric,
                            parameters={"method": "sum", "within": group_col, "intent_type": strategy},
                            justification=reason,
                            trace=_trace(reason, 0.72, ["derived_metric", "grouped_average"]),
                        )
                    )
                    reason = f"Compute the mean across grouped {metric} totals to answer the requested average per group."
                    add_step(
                        ComputationStep(
                            operation="aggregate",
                            column=metric,
                            parameters={"method": "mean", "scope": "group_results", "intent_type": strategy},
                            justification=reason,
                            trace=_trace(reason, 0.72, ["aggregation_mapping"]),
                        )
                    )
                else:
                    reason = f"Apply {agg_method} to {metric} because this directly answers the requested aggregation."
                    add_step(
                        ComputationStep(
                            operation="aggregate",
                            column=metric,
                            parameters={"method": agg_method, "within": group_col, "intent_type": strategy} if group_col else {"method": agg_method, "intent_type": strategy},
                            justification=reason,
                            trace=_trace(reason, 0.8, ["intent_to_computation"]),
                        )
                    )

        elif strategy == "comparison":
            if metric_columns and group_columns:
                for metric in metric_columns:
                    for group_col in group_columns:
                        reason = f"Compare grouped values of {metric} across {group_col} before considering any inferential method."
                        add_step(
                            ComputationStep(
                                operation="group_compare",
                                column=metric,
                                columns=[metric, group_col],
                                parameters={"group_by": group_col, "aggregate": _aggregation_method(question, "mean"), "intent_type": strategy},
                                justification=reason,
                                trace=_trace(reason, 0.78, ["comparison_intent", "structure_first"]),
                            )
                        )
            else:
                confidence = min(confidence, 0.3)
                notes.append("Comparison requested but the engine could not resolve both a grouping column and a metric column.")

        elif strategy == "relationship":
            if len(metric_columns) >= 2:
                for idx, left in enumerate(metric_columns):
                    for right in metric_columns[idx + 1:]:
                        reason = f"Evaluate the relationship between {left} and {right} because both are valid metric columns for the requested relationship analysis."
                        add_step(
                            ComputationStep(
                                operation="pairwise_relationship",
                                columns=[left, right],
                                parameters={"intent_type": strategy},
                                justification=reason,
                                trace=_trace(reason, 0.78, ["relationship_intent"]),
                            )
                        )
            elif metric_columns and group_columns:
                for metric in metric_columns:
                    for group_col in group_columns:
                        reason = f"The resolved structure indicates a metric-to-group relationship between {metric} and {group_col}, so grouped comparison is the minimal valid computation."
                        add_step(
                            ComputationStep(
                                operation="group_compare",
                                columns=[metric, group_col],
                                parameters={"group_by": group_col, "aggregate": "mean", "intent_type": strategy},
                                justification=reason,
                                trace=_trace(reason, 0.68, ["relationship_intent", "mixed_input_types"]),
                            )
                        )
            elif len(group_columns) >= 2:
                for idx, left in enumerate(group_columns):
                    for right in group_columns[idx + 1:]:
                        reason = f"The resolved structure indicates a categorical association question between {left} and {right}, so association testing is the minimal valid computation."
                        add_step(
                            ComputationStep(
                                operation="categorical_association",
                                columns=[left, right],
                                parameters={"intent_type": strategy},
                                justification=reason,
                                trace=_trace(reason, 0.72, ["relationship_intent", "categorical_inputs"]),
                            )
                        )
            else:
                confidence = min(confidence, 0.3)
                notes.append("Relationship analysis requested but the engine could not resolve structurally valid variables.")

        elif strategy == "distribution":
            target = group_columns[0] if group_columns else (metric_columns[0] if metric_columns else None)
            if target:
                op = "frequency_distribution" if _categorical_capable(dataset_profile, inferred_context, target) else "numeric_distribution"
                reason = f"Compute the distribution of {target} because that directly answers the descriptive question."
                add_step(
                    ComputationStep(
                        operation=op,
                        column=target,
                        parameters={"intent_type": strategy},
                        justification=reason,
                        trace=_trace(reason, 0.8, ["distribution_intent"]),
                    )
                )
            else:
                confidence = min(confidence, 0.3)
                notes.append("Distribution analysis requested but the engine could not resolve a valid target column.")

    deduped_group_columns = list(dict.fromkeys(all_group_columns))
    deduped_metric_columns = list(dict.fromkeys(all_metric_columns))

    if deduped_group_columns and _info(dataset_profile, deduped_group_columns[0]).get("unique_count", 0) < 2:
        confidence -= 0.3
        notes.append("Grouping column does not contain enough distinct values.")
    if deduped_metric_columns and _info(dataset_profile, deduped_metric_columns[0]).get("missing_ratio", 0.0) > 0.8:
        confidence -= 0.3
        notes.append("Metric column is too sparse for reliable computation.")
    if structural_signals.get("primary_structure_confidence", 1.0) < 0.45:
        confidence -= 0.15
        notes.append("Primary dataset structure is ambiguous.")

    confidence = _bounded_confidence(confidence)
    deferred = confidence < 0.45 or not steps
    justification = "Built the computation plan before tool selection using resolved grouping columns, metric columns, and structural validity checks."
    if len(strategies) > 1:
        justification += f" Combined analytical intents were preserved: {', '.join(strategies)}."
    if notes:
        justification += " " + " ".join(notes)

    return ComputationPlanModel(
        steps=steps,
        confidence_score=confidence,
        justification=justification,
        deferred=deferred,
    )


def build_analysis_plan(
    dataset_profile: Dict[str, Any],
    structural_signals: Dict[str, Any],
    inferred_context: Dict[str, Any],
    relationship_signals: Dict[str, Any],
    user_intent: Dict[str, Any],
    candidate_plan: List[Dict[str, Any]],
    selected_columns: List[str],
    computation_plan: ComputationPlanModel,
) -> AnalysisPlanModel:
    # Tool selection is intentionally last. If the computation plan already
    # answers the question deterministically, we keep the plan simple and avoid
    # inferential tests.
    strategies = _determine_strategies(user_intent.get("query", ""), user_intent)
    strategy = strategies[0] if strategies else "unknown"
    confidence = computation_plan.confidence_score

    if computation_plan.deferred:
        return AnalysisPlanModel(
            analytical_strategy=strategy,
            operations=[],
            confidence_score=max(0.2, confidence),
            justification="Analysis planning was deferred because the required computation could not be validated.",
            deferred=True,
        )

    operations: List[AnalysisOperation] = []
    notes: List[str] = []
    computation_dump = [item.model_dump() for item in computation_plan.steps]

    summarize_steps = [step for step in computation_plan.steps if step.operation == "summarize_metric" and step.column]
    if summarize_steps:
        columns = list(dict.fromkeys([step.column for step in summarize_steps if step.column]))
        reason = "Summary statistics are the direct deterministic answer for a descriptive multi-metric request."
        operations.append(
            AnalysisOperation(
                tool="summary_statistics",
                columns=columns,
                computation_refs=[step.operation for step in summarize_steps],
                parameters={},
                justification=reason,
                trace=_trace(reason, confidence, ["computation_first", "summary_statistics_intent"]),
            )
        )

    direct_steps = [
        step for step in computation_plan.steps
        if step.operation in {
            "group_by",
            "aggregate",
            "period_extremes",
            "frequency_distribution",
            "numeric_distribution",
            "distinct_count",
            "row_expression",
            "share_of_total",
            "repeat_rate",
            "missingness_report",
            "duplicate_rows_report",
            "timestamp_consistency_report",
            "numeric_validity_check",
            "delivery_date_validity",
            "categorical_label_quality",
            "customer_order_frequency",
            "purchase_gap",
            "single_purchase_share",
            "cohort_repeat_rate",
            "cohort_value_rank",
            "churn_speed_proxy",
            "segment_retention_rate",
            "basket_value_pattern",
            "threshold_value_comparison",
            "loyalty_trend",
            "dormancy_count",
            "segment_repeat_rate",
            "review_repeat_comparison",
            "categorical_preference_by_entity",
            "payment_preference_by_entity",
            "segment_order_value",
            "grouped_pairwise_relationship",
            "delivery_duration_rank",
            "growth_rate",
            "trend_classification",
            "rank_entities",
            "filtered_rank_entities",
            "relative_burden_rank",
            "low_outcome_driver_analysis",
            "top_dependency_share",
            "review_crisis_rank",
            "late_delivery_period_cluster",
            "aggregate_forecast",
            "capacity_need_score",
            "strategic_opportunity_score",
            "predictive_target_profile",
            "customer_ltv_estimate",
            "customer_clustering_segments",
            "delivery_promise_optimization",
            "logistics_underperformance_score",
            "time_series_metric",
            "weekday_segment_compare",
            "temporal_spike_detection",
            "rapid_repeat_order_anomaly",
            "transaction_value_outlier_rank",
            "contextual_metric_mismatch",
            "entity_temporal_spike_detection",
            "duplicate_behavior_fingerprint",
            "review_pattern_anomaly",
            "geographic_anomaly_score",
            "event_window_impact",
            "customer_acquisition_trend",
            "segment_contrast",
            "derived_delay_relationship",
            "segment_growth_rank",
            "segment_seasonality",
            "concentration_score",
            "entity_intervention_score",
            "premium_potential_score",
            "basket_cooccurrence",
            "basket_value_comparison",
            "elasticity_proxy_score",
            "discount_volume_effect",
            "price_band_demand",
            "price_competition_score",
            "ratio_metric",
            "retention_risk_proxy",
            "logistics_optimization_opportunity",
            "delivery_duration_summary",
            "delivery_gap_summary",
            "delivery_timing_share",
            "delay_burden_rank",
            "delay_trend",
            "delay_quality_relationship",
            "distance_proxy_cancellation_relationship",
            "status_share",
            "status_rate_by_entity",
            "operational_issue_score",
            "status_rate_trend",
        }
    ]
    if direct_steps:
        columns = []
        refs = []
        for step in direct_steps:
            refs.append(step.operation)
            if step.column:
                columns.append(step.column)
            columns.extend(step.columns)
            columns.extend(_step_parameter_columns(step.parameters))
        columns = list(dict.fromkeys([col for col in columns if col]))
        reason = "Direct deterministic computation is sufficient and preferable to statistical testing for this part of the request."
        operations.append(
            AnalysisOperation(
                tool="direct_computation",
                columns=columns,
                computation_refs=refs,
                parameters={"computation_plan": [step.model_dump() for step in direct_steps], "strategy": strategy},
                justification=reason,
                trace=_trace(reason, confidence, ["computation_first", "no_inferential_test_needed"]),
            )
        )
        if strategy == "predictive":
            return AnalysisPlanModel(
                analytical_strategy=strategy,
                operations=operations,
                confidence_score=confidence,
                justification="Used deterministic predictive-planning computations before falling back to generic supervised modeling.",
                deferred=False,
            )

    if strategy == "predictive":
        preserved_operations: List[AnalysisOperation] = []
        for item in candidate_plan:
            tool = item.get("tool")
            if tool not in {"predictive_analysis", "prescriptive_analysis"}:
                continue
            columns = item.get("columns", []) or selected_columns
            reason = "Predictive and prescriptive workflows are handled by the dedicated deterministic modeling engines after readiness validation."
            preserved_operations.append(
                AnalysisOperation(
                    tool=tool,
                    columns=columns,
                    computation_refs=[step.operation for step in computation_plan.steps],
                    parameters=item.get("parameters", {}),
                    justification=reason,
                    trace=_trace(reason, confidence, ["predictive_intent", "deterministic_modeling"]),
                )
            )
        return AnalysisPlanModel(
            analytical_strategy=strategy,
            operations=preserved_operations,
            confidence_score=confidence,
            justification="Preserved predictive and prescriptive operations for dedicated modeling execution.",
            deferred=not preserved_operations,
        )

    for step in computation_plan.steps:
        if step.operation == "pairwise_relationship":
            reason = "The resolved computation requires measuring the relationship between numeric-capable variables."
            operations.append(
                AnalysisOperation(
                    tool="correlation",
                    columns=step.columns[:2],
                    computation_refs=[step.operation],
                    parameters={"computation_plan": computation_dump, "strategy": strategy},
                    justification=reason,
                    trace=_trace(reason, confidence, ["relationship_intent"]),
                )
            )
        elif step.operation == "group_compare":
            metric = step.column or (step.columns[0] if step.columns else None)
            group_col = step.parameters.get("group_by") or (step.columns[-1] if step.columns else None)
            if metric and group_col:
                unique_count = _info(dataset_profile, group_col).get("unique_count", 0)
                intent_type = step.parameters.get("intent_type")
                explicit_inference = _contains_any(user_intent.get("query", ""), ["affect", "effect", "significant", "difference between groups", "independent"])
                if intent_type == "comparison" and not explicit_inference:
                    tool = "direct_computation"
                    reason = "The question asks for comparison, so grouped aggregation is the minimal valid answer and inferential testing is unnecessary."
                else:
                    tool = "ttest" if unique_count == 2 else "anova"
                    reason = "The resolved structure is a metric-to-group relationship, so grouped comparison testing is the valid final step."
                operations.append(
                    AnalysisOperation(
                        tool=tool,
                        columns=[metric, group_col],
                        computation_refs=[step.operation],
                        parameters={"computation_plan": computation_dump, "strategy": strategy},
                        justification=reason,
                        trace=_trace(reason, confidence, ["comparison_intent", "analysis_validity"]),
                    )
                )
            else:
                confidence = _bounded_confidence(confidence - 0.25)
        elif step.operation == "categorical_association":
            reason = "The resolved structure is categorical-versus-categorical, so a chi-square test of independence is the valid final step."
            operations.append(
                AnalysisOperation(
                    tool="chi_square",
                    columns=step.columns[:2],
                    computation_refs=[step.operation],
                    parameters={"computation_plan": computation_dump, "strategy": strategy},
                    justification=reason,
                    trace=_trace(reason, confidence, ["relationship_intent", "categorical_inputs"]),
                )
            )

    if not operations:
        notes.append("No valid tool could be selected after computation-first validation.")

    justification = "Selected tools only after validating the required computation plan and checking structural validity."
    if len(strategies) > 1:
        justification += f" Preserved multiple analytical intents in one coordinated plan: {', '.join(strategies)}."
    if notes:
        justification += " " + " ".join(notes)
    if structural_signals.get("signals"):
        justification += " Structural signals informed fallback and confidence."
    if relationship_signals.get("relationships"):
        justification += " Relationship signals were used to preserve consistency."

    return AnalysisPlanModel(
        analytical_strategy=strategy,
        operations=operations,
        confidence_score=_bounded_confidence(confidence),
        justification=justification,
        deferred=not operations or confidence < 0.45,
    )


def run_decision_engine(
    dataset_profile: Dict[str, Any],
    structural_signals: Dict[str, Any],
    inferred_context: Dict[str, Any],
    relationship_signals: Dict[str, Any],
    user_intent: Dict[str, Any],
    constraint_rules: Dict[str, Any],
    candidate_plan: List[Dict[str, Any]],
    selected_columns: List[str],
) -> DecisionEngineOutput:
    cleaning_decisions = build_cleaning_decisions(
        dataset_profile=dataset_profile,
        structural_signals=structural_signals,
        inferred_context=inferred_context,
        relationship_signals=relationship_signals,
        constraint_rules=constraint_rules,
    )
    analysis_abstraction = build_analysis_abstraction(
        question=user_intent.get("query", ""),
        dataset_profile=dataset_profile,
        inferred_context=inferred_context,
        relationship_signals=relationship_signals,
        user_intent=user_intent,
        selected_columns=selected_columns,
    )
    computation_plan = build_computation_plan(
        dataset_profile=dataset_profile,
        structural_signals=structural_signals,
        inferred_context=inferred_context,
        relationship_signals=relationship_signals,
        user_intent=user_intent,
        selected_columns=selected_columns,
        analysis_abstraction=analysis_abstraction,
    )
    analysis_plan = build_analysis_plan(
        dataset_profile=dataset_profile,
        structural_signals=structural_signals,
        inferred_context=inferred_context,
        relationship_signals=relationship_signals,
        user_intent=user_intent,
        candidate_plan=candidate_plan,
        selected_columns=selected_columns,
        computation_plan=computation_plan,
    )

    notes = []
    if any(item.deferred for item in cleaning_decisions):
        notes.append("Some cleaning actions were deferred because confidence was insufficient or constraints blocked them.")
    if computation_plan.deferred:
        notes.append("The computation plan was deferred because the engine could not validate a minimal reliable computation.")
    if analysis_plan.deferred:
        notes.append("Analysis planning deferred or reduced because no operation met validity and confidence requirements.")
    if any(op.tool == "direct_computation" for op in analysis_plan.operations):
        notes.append("Direct deterministic computation was preferred over inferential testing because it fully answers the intent.")
    notes.append(f"Analysis abstraction family: {analysis_abstraction.capability_family}.")

    return DecisionEngineOutput(
        cleaning_decisions=cleaning_decisions,
        analysis_abstraction=analysis_abstraction,
        computation_plan=computation_plan,
        analysis_plan=analysis_plan,
        decision_notes=notes,
        retry_hints={
            "preferred_fallbacks": [op.fallback_tool for op in analysis_plan.operations if op.fallback_tool],
            "halt_analysis": analysis_plan.deferred and not analysis_plan.operations,
            "clarification_needed": computation_plan.deferred or analysis_plan.deferred,
        },
    )
