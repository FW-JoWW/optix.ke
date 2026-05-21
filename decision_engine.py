from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List

from decision_models import (
    AnalysisOperation,
    AnalysisPlanModel,
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
    preferred = _best_text_match(question, ranked, ["purchase", "order", "date", "time", "month", "quarter", "year"])
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
    candidates = [
        col for col in _all_columns(profile)
        if col in _profile_group(profile, "categorical_columns") or _role(context, col) in {"categorical_feature", "grouping_key"}
    ]
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
        return value

    ranked = sorted(candidates, key=score, reverse=True)
    return ranked[0] if score(ranked[0]) > 0 else None


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
        if _contains_any(question, ["revenue", "sales", "order value", "aov"]):
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

    if "predictive" in explicit or _contains_any(query, ["predict", "forecast", "estimate", "project", "likely", "risk", "what if", "optimize", "scenario"]):
        add("predictive")
    if "relationship" in explicit or _contains_any(query, ["relationship", "correlation", "regression", "cause", "causal", "drive"]):
        add("relationship")
    if "comparison" in explicit or _contains_any(query, ["compare", "difference", "affect", "impact", "effect"]):
        add("comparison")
    if any(item in {"composition", "distribution"} for item in explicit) or _contains_any(query, ["distribution", "frequency", "mode", "cardinality", "rare", "share", "mix", "overdependent"]):
        add("distribution")
    if any(item in {"profiling", "aggregation", "temporal", "extremes"} for item in explicit) or _contains_any(query, ["average", "mean", "median", "summary", "statistics", "sum", "total", "sell", "sold", "best seller", "best sellers", "pricing", "volume", "revenue"]):
        add("aggregation")
    if _contains_any(query, ["trend", "growth", "declining", "seasonal", "seasonality"]):
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
) -> ComputationPlanModel:
    # Computation-first planning:
    # 1. resolve the minimal computation needed to answer the question
    # 2. validate grouping/metric choices against structure and sparsity
    # 3. leave tool selection to the later analysis-plan stage
    question = user_intent.get("query", "")
    strategies = _determine_strategies(question, user_intent)
    primary_strategy = strategies[0] if strategies else "unknown"
    steps: List[ComputationStep] = []
    confidence = 0.75
    notes: List[str] = []

    seen_steps = set()

    def add_step(step: ComputationStep) -> None:
        key = (
            step.operation,
            step.column,
            tuple(step.columns),
            tuple(sorted((step.parameters or {}).items())),
        )
        if key in seen_steps:
            return
        seen_steps.add(key)
        steps.append(step)

    if primary_strategy == "unknown":
        return ComputationPlanModel(
            steps=[],
            confidence_score=0.25,
            justification="The analytical intent could not be mapped to a reliable computation plan.",
            deferred=True,
        )

    if "predictive" in strategies:
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

    if _contains_any(question, ["average order value", "aov"]) and primary_metric and order_column:
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

    if _looks_like_share_question(question) and primary_metric:
        if _contains_any(question, ["top customers"]) and customer_column:
            reason = f"Compute the share of total {primary_metric[0]} contributed by the top customers."
            add_step(
                ComputationStep(
                    operation="share_of_total",
                    column=primary_metric[0],
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

    if _contains_any(question, ["profit proxy", "after freight"]) and primary_metric:
        freight_column = _best_text_match(question + " freight", _all_columns(dataset_profile), ["freight"])
        if freight_column:
            reason = f"Construct a profit proxy by subtracting {freight_column} from {primary_metric[0]} before aggregation."
            add_step(
                ComputationStep(
                    operation="row_expression",
                    column=primary_metric[0],
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

    if _contains_any(question, ["best and worst"]) and _contains_any(question, ["month", "months"]) and time_column and primary_metric:
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
        agg_reason = f"Aggregate {primary_metric[0]} within each month to rank monthly performance."
        add_step(
            ComputationStep(
                operation="aggregate",
                column=primary_metric[0],
                parameters={"method": "sum", "within": time_column, "intent_type": "aggregation"},
                justification=agg_reason,
                trace=_trace(agg_reason, 0.84, ["monthly_ranking"]),
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
            "frequency_distribution",
            "numeric_distribution",
            "distinct_count",
            "row_expression",
            "share_of_total",
            "repeat_rate",
            "growth_rate",
            "trend_classification",
            "rank_entities",
            "segment_contrast",
            "segment_growth_rank",
            "segment_seasonality",
            "concentration_score",
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
            for param_key in ("group_by", "within", "entity_column", "subtract_column", "parent_column", "child_column", "time_column", "primary_metric", "secondary_metric"):
                param_value = (step.parameters or {}).get(param_key)
                if isinstance(param_value, str):
                    columns.append(param_value)
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
    computation_plan = build_computation_plan(
        dataset_profile=dataset_profile,
        structural_signals=structural_signals,
        inferred_context=inferred_context,
        relationship_signals=relationship_signals,
        user_intent=user_intent,
        selected_columns=selected_columns,
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

    return DecisionEngineOutput(
        cleaning_decisions=cleaning_decisions,
        computation_plan=computation_plan,
        analysis_plan=analysis_plan,
        decision_notes=notes,
        retry_hints={
            "preferred_fallbacks": [op.fallback_tool for op in analysis_plan.operations if op.fallback_tool],
            "halt_analysis": analysis_plan.deferred and not analysis_plan.operations,
            "clarification_needed": computation_plan.deferred or analysis_plan.deferred,
        },
    )
