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
    return any(word in text for word in words)


def _base_name(column: str) -> str:
    return re.sub(r"_+\d+$", "", str(column).strip().lower())


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _base_name(left), _base_name(right)).ratio()


def _numeric_capable(profile: Dict[str, Any], context: Dict[str, Any], column: str) -> bool:
    info = _info(profile, column)
    return (
        info.get("inferred_type") == "numeric"
        or info.get("numeric_like_ratio", 0.0) >= 0.65
        or _role(context, column) in {"numeric_measure", "derived_metric"}
    )


def _categorical_capable(profile: Dict[str, Any], context: Dict[str, Any], column: str) -> bool:
    info = _info(profile, column)
    role = _role(context, column)
    if role in {"numeric_measure", "derived_metric"}:
        return False
    return info.get("inferred_type") in {"categorical", "datetime"} or role in {"categorical_feature", "grouping_key", "timestamp"}


def _timestamp_like(profile: Dict[str, Any], context: Dict[str, Any], column: str) -> bool:
    info = _info(profile, column)
    return (
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
    return _bounded_confidence(score)


def _family_candidates(
    column: str,
    profile: Dict[str, Any],
    context: Dict[str, Any],
) -> List[str]:
    if not column:
        return []
    family = _base_name(column)
    candidates = []
    for candidate in profile.get("column_names", []) or []:
        if _base_name(candidate) != family:
            continue
        if _numeric_capable(profile, context, candidate):
            candidates.append(candidate)
    return candidates


def _determine_strategy(question: str, intent: Dict[str, Any]) -> str:
    query = (question or "").lower()
    analytic = str(intent.get("analytic_intent") or intent.get("type") or "").lower()
    if analytic == "relationship" or _contains_any(query, ["relationship", "correlation", "regression", "cause", "causal", "drive"]):
        return "relationship"
    if analytic == "comparison" or _contains_any(query, ["compare", "difference", "affect", "impact", "effect"]):
        return "comparison"
    if analytic in {"composition", "distribution"} or _contains_any(query, ["distribution", "frequency", "mode", "cardinality", "rare"]):
        return "distribution"
    if analytic in {"profiling", "aggregation"} or _contains_any(query, ["average", "mean", "median", "summary", "statistics", "sum", "total"]):
        return "aggregation"
    return "unknown"


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
    group_by = intent.get("group_by")
    if group_by in (profile.get("column_names") or []) and _categorical_capable(profile, context, group_by):
        return [group_by]

    candidates = [col for col in selected_columns if _categorical_capable(profile, context, col)]
    if not candidates:
        candidates = [
            col for col in (profile.get("column_names") or [])
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
    aggregate_column = intent.get("aggregate_column")
    if aggregate_column in (profile.get("column_names") or []):
        family_candidates = _family_candidates(aggregate_column, profile, context)
        if family_candidates:
            family_candidates.sort(
                key=lambda col: _metric_score(profile, context, relationships, col),
                reverse=True,
            )
            return [family_candidates[0]]
        if _numeric_capable(profile, context, aggregate_column):
            return [aggregate_column]

    numeric_candidates = [col for col in selected_columns if _numeric_capable(profile, context, col)]
    if not numeric_candidates:
        numeric_candidates = [
            col for col in (profile.get("column_names") or [])
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
    strategy = _determine_strategy(question, user_intent)
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
    steps: List[ComputationStep] = []
    confidence = 0.75
    notes: List[str] = []

    if strategy == "unknown":
        return ComputationPlanModel(
            steps=[],
            confidence_score=0.25,
            justification="The analytical intent could not be mapped to a reliable computation plan.",
            deferred=True,
        )

    for group_col in group_columns:
        reason = f"Group by {group_col} because it is the best structural match for the requested breakdown."
        steps.append(
            ComputationStep(
                operation="group_by",
                column=group_col,
                justification=reason,
                trace=_trace(reason, 0.8, [f"role:{_role(inferred_context, group_col)}"]),
            )
        )

    if strategy == "aggregation":
        agg_method = _aggregation_method(question, default=user_intent.get("aggregation") or "mean")
        if not metric_columns:
            confidence = 0.25
            notes.append("No reliable metric column was found for aggregation.")
        else:
            metric = metric_columns[0]
            group_col = group_columns[0] if group_columns else None
            metric_role = _role(inferred_context, metric)
            if group_col and agg_method == "mean" and metric_role == "derived_metric":
                reason = f"Summarize {metric} within each {group_col} group before averaging because the metric behaves like a derived measure."
                steps.append(
                    ComputationStep(
                        operation="aggregate",
                        column=metric,
                        parameters={"method": "sum", "within": group_col},
                        justification=reason,
                        trace=_trace(reason, 0.72, ["derived_metric", "grouped_average"]),
                    )
                )
                reason = f"Compute the mean across grouped {metric} totals to answer the requested average per group."
                steps.append(
                    ComputationStep(
                        operation="aggregate",
                        column=metric,
                        parameters={"method": "mean", "scope": "group_results"},
                        justification=reason,
                        trace=_trace(reason, 0.72, ["aggregation_mapping"]),
                    )
                )
            else:
                reason = f"Apply {agg_method} to {metric} because this directly answers the requested aggregation."
                steps.append(
                    ComputationStep(
                        operation="aggregate",
                        column=metric,
                        parameters={"method": agg_method, "within": group_col} if group_col else {"method": agg_method},
                        justification=reason,
                        trace=_trace(reason, 0.8, ["intent_to_computation"]),
                    )
                )

    elif strategy == "comparison":
        if metric_columns and group_columns:
            metric = metric_columns[0]
            group_col = group_columns[0]
            reason = f"Compare grouped values of {metric} across {group_col} before considering any inferential method."
            steps.append(
                ComputationStep(
                    operation="group_compare",
                    column=metric,
                    columns=[metric, group_col],
                    parameters={"group_by": group_col, "aggregate": _aggregation_method(question, "mean")},
                    justification=reason,
                    trace=_trace(reason, 0.78, ["comparison_intent", "structure_first"]),
                )
            )
        else:
            confidence = 0.3
            notes.append("Comparison requested but the engine could not resolve both a grouping column and a metric column.")

    elif strategy == "relationship":
        if len(metric_columns) >= 2:
            for idx, left in enumerate(metric_columns):
                for right in metric_columns[idx + 1:]:
                    reason = f"Evaluate the relationship between {left} and {right} because both are valid metric columns for the requested relationship analysis."
                    steps.append(
                        ComputationStep(
                            operation="pairwise_relationship",
                            columns=[left, right],
                            justification=reason,
                            trace=_trace(reason, 0.78, ["relationship_intent"]),
                        )
                    )
        elif metric_columns and group_columns:
            for metric in metric_columns:
                for group_col in group_columns:
                    reason = f"The resolved structure indicates a metric-to-group relationship between {metric} and {group_col}, so grouped comparison is the minimal valid computation."
                    steps.append(
                        ComputationStep(
                            operation="group_compare",
                            columns=[metric, group_col],
                            parameters={"group_by": group_col, "aggregate": "mean"},
                            justification=reason,
                            trace=_trace(reason, 0.68, ["relationship_intent", "mixed_input_types"]),
                        )
                    )
        elif len(group_columns) >= 2:
            for idx, left in enumerate(group_columns):
                for right in group_columns[idx + 1:]:
                    reason = f"The resolved structure indicates a categorical association question between {left} and {right}, so association testing is the minimal valid computation."
                    steps.append(
                        ComputationStep(
                            operation="categorical_association",
                            columns=[left, right],
                            justification=reason,
                            trace=_trace(reason, 0.72, ["relationship_intent", "categorical_inputs"]),
                        )
                    )
        else:
            confidence = 0.3
            notes.append("Relationship analysis requested but the engine could not resolve structurally valid variables.")

    elif strategy == "distribution":
        target = group_columns[0] if group_columns else (metric_columns[0] if metric_columns else None)
        if target:
            op = "frequency_distribution" if _categorical_capable(dataset_profile, inferred_context, target) else "numeric_distribution"
            reason = f"Compute the distribution of {target} because that directly answers the descriptive question."
            steps.append(
                ComputationStep(
                    operation=op,
                    column=target,
                    justification=reason,
                    trace=_trace(reason, 0.8, ["distribution_intent"]),
                )
            )
        else:
            confidence = 0.3
            notes.append("Distribution analysis requested but the engine could not resolve a valid target column.")

    if group_columns and _info(dataset_profile, group_columns[0]).get("unique_count", 0) < 2:
        confidence -= 0.3
        notes.append("Grouping column does not contain enough distinct values.")
    if metric_columns and _info(dataset_profile, metric_columns[0]).get("missing_ratio", 0.0) > 0.8:
        confidence -= 0.3
        notes.append("Metric column is too sparse for reliable computation.")
    if structural_signals.get("primary_structure_confidence", 1.0) < 0.45:
        confidence -= 0.15
        notes.append("Primary dataset structure is ambiguous.")

    confidence = _bounded_confidence(confidence)
    deferred = confidence < 0.45 or not steps
    justification = "Built the computation plan before tool selection using resolved grouping columns, metric columns, and structural validity checks."
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
    strategy = _determine_strategy(user_intent.get("query", ""), user_intent)
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

    if strategy in {"aggregation", "distribution"}:
        columns = []
        refs = []
        for step in computation_plan.steps:
            refs.append(step.operation)
            if step.column:
                columns.append(step.column)
            columns.extend(step.columns)
        columns = list(dict.fromkeys([col for col in columns if col]))
        reason = "Direct deterministic computation is sufficient and preferable to statistical testing for this request."
        operations.append(
            AnalysisOperation(
                tool="direct_computation",
                columns=columns,
                computation_refs=refs,
                parameters={"computation_plan": [step.model_dump() for step in computation_plan.steps], "strategy": strategy},
                justification=reason,
                trace=_trace(reason, confidence, ["computation_first", "no_inferential_test_needed"]),
            )
        )
        return AnalysisPlanModel(
            analytical_strategy=strategy,
            operations=operations,
            confidence_score=confidence,
            justification="Selected direct computation because the requested answer is an aggregation or distribution, not an inferential test.",
        )

    if strategy == "comparison":
        step = next((item for item in computation_plan.steps if item.operation == "group_compare"), None)
        if step is not None:
            metric = step.column or (step.columns[0] if step.columns else None)
            group_col = step.parameters.get("group_by") or (step.columns[-1] if step.columns else None)
            if metric and group_col:
                unique_count = _info(dataset_profile, group_col).get("unique_count", 0)
                explicit_inference = _contains_any(user_intent.get("query", ""), ["affect", "effect", "significant", "difference between groups", "independent"])
                if explicit_inference:
                    tool = "ttest" if unique_count == 2 else "anova"
                    reason = "Inferential language in the question justifies a group-comparison test after grouped computation planning."
                else:
                    tool = "direct_computation"
                    reason = "The question asks for comparison, so grouped aggregation is the minimal valid answer and inferential testing is unnecessary."
                operations.append(
                    AnalysisOperation(
                        tool=tool,
                        columns=[metric, group_col],
                        computation_refs=[step.operation],
                        parameters={"computation_plan": [item.model_dump() for item in computation_plan.steps], "strategy": strategy},
                        justification=reason,
                        trace=_trace(reason, confidence, ["comparison_intent", "analysis_validity"]),
                    )
                )
            else:
                confidence = _bounded_confidence(confidence - 0.25)

    elif strategy == "relationship":
        computation_dump = [item.model_dump() for item in computation_plan.steps]
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
                metric = step.columns[0] if step.columns else None
                group_col = step.parameters.get("group_by") or (step.columns[-1] if step.columns else None)
                if metric and group_col:
                    unique_count = _info(dataset_profile, group_col).get("unique_count", 0)
                    tool = "ttest" if unique_count == 2 else "anova"
                    reason = "The resolved structure is a metric-to-group relationship, so grouped comparison testing is the valid final step."
                    operations.append(
                        AnalysisOperation(
                            tool=tool,
                            columns=[metric, group_col],
                            computation_refs=[step.operation],
                            parameters={"computation_plan": computation_dump, "strategy": strategy},
                            justification=reason,
                            trace=_trace(reason, confidence, ["relationship_intent", "mixed_input_types"]),
                        )
                    )
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
