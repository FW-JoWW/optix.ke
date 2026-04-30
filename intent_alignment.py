from __future__ import annotations

from typing import Any, Dict, List


def validate_analysis_plan_against_intent(
    question: str,
    intent: Dict[str, Any],
    plan: List[Dict[str, Any]],
    selected_columns: List[str],
) -> List[Dict[str, Any]]:
    question = (question or "").lower()
    selected_set = set(selected_columns or [])

    allowed_tools = set()
    if any(word in question for word in ["summary", "statistics", "average", "mean", "median"]):
        allowed_tools.add("summary_statistics")
    if any(word in question for word in ["outlier", "anomaly", "unusual"]):
        allowed_tools.add("detect_outliers")
    if any(word in question for word in ["relationship", "correlation", "cause", "causal", "drive"]):
        allowed_tools.add("correlation")
    if any(word in question for word in ["compare", "difference", "affect", "impact", "effect"]):
        allowed_tools.update({"ttest", "anova"})
    if any(word in question for word in ["distribution", "frequency", "cardinality", "rare", "mode", "category"]):
        allowed_tools.add("categorical_analysis")

    if not allowed_tools:
        allowed_tools = {item.get("tool") for item in plan}

    validated: List[Dict[str, Any]] = []
    seen = set()
    for item in plan:
        tool = item.get("tool")
        columns = [col for col in item.get("columns", []) if not selected_set or col in selected_set]
        if tool not in allowed_tools or not columns:
            continue
        key = (tool, tuple(columns))
        if key in seen:
            continue
        seen.add(key)
        validated.append({"tool": tool, "columns": columns})

    return validated
