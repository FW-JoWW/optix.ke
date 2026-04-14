from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from state.state import AnalystState


def _mentioned_columns(question: str, columns: List[str]) -> List[str]:
    question = question.lower()
    return [col for col in columns if col.lower() in question]


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
        elif tool_type == "categorical_analysis":
            story_candidates.extend(_categorical_stories(result, question))

    evidence["story_candidates"] = story_candidates

    print("\n=== STORY CANDIDATES GENERATED ===")
    for story in story_candidates:
        print("-", story["insight"])

    return state
