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
        allowed_tools.add("direct_computation")
    if any(word in question for word in ["outlier", "anomaly", "unusual"]):
        allowed_tools.add("detect_outliers")
    if any(word in question for word in ["relationship", "correlation", "cause", "causal", "drive"]):
        allowed_tools.update({"correlation", "chi_square", "ttest", "anova"})
    if any(word in question for word in ["compare", "difference", "affect", "impact", "effect"]):
        allowed_tools.update({"ttest", "anova", "direct_computation"})
    if any(word in question for word in ["trend", "over time", "monthly", "quarter", "growth", "growing", "declining", "flat", "percentage", "share", "how many", "count", "profit proxy", "top customers", "repeat purchases"]):
        allowed_tools.add("direct_computation")
    if any(word in question for word in ["distribution", "frequency", "cardinality", "rare", "mode", "category"]):
        allowed_tools.add("categorical_analysis")
    if any(word in question for word in ["predict", "forecast", "estimate", "project", "likely", "risk", "score"]):
        allowed_tools.add("predictive_analysis")
    if any(word in question for word in ["optimize", "optimization", "recommend", "allocate", "allocation", "what if", "scenario", "reorder", "capacity", "pricing"]):
        allowed_tools.update({"predictive_analysis", "prescriptive_analysis"})

    if not allowed_tools:
        allowed_tools = {item.get("tool") for item in plan}

    validated: List[Dict[str, Any]] = []
    seen = set()
    for item in plan:
        tool = item.get("tool")
        if tool in {"predictive_analysis", "prescriptive_analysis"}:
            columns = item.get("columns", [])
        else:
            columns = [col for col in item.get("columns", []) if not selected_set or col in selected_set]
        if tool not in allowed_tools or not columns:
            continue
        key = (tool, tuple(columns))
        if key in seen:
            continue
        seen.add(key)
        validated.append({"tool": tool, "columns": columns, "parameters": item.get("parameters", {})})

    return validated
