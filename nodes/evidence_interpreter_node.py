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

    if rows:
        first = rows[0]
        keys = list(first.keys())
        if len(keys) >= 2:
            group_key, value_key = keys[0], keys[1]
            ranked = sorted(rows, key=lambda row: row.get(value_key, float("-inf")), reverse=True)
            top = ranked[0]
            bottom = ranked[-1]
            stories.append({
                "type": "grouped_numeric",
                "insight": f"{top[group_key]} has the highest {value_key} and {bottom[group_key]} the lowest",
                "column": value_key,
                "group_column": group_key,
                "top_group": str(top[group_key]),
                "bottom_group": str(bottom[group_key]),
                "top_value": float(top[value_key]),
                "bottom_value": float(bottom[value_key]),
                "effect_size": float(top[value_key]) - float(bottom[value_key]),
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

    evidence["story_candidates"] = story_candidates

    print("\n=== STORY CANDIDATES GENERATED ===")
    for story in story_candidates:
        print("-", story["insight"])

    return state
