from typing import List, Dict, Any

def llm_decide_tools(business_question: str, dataset_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns a list of tools + columns to run based on simple heuristics.
    """
    plan = []

    # Simple heuristics — can expand later with LLM reasoning
    if "correlation" in business_question.lower() or "spend" in business_question.lower():
        plan.append({"tool": "correlation", "columns": ["revenue", "ad_spend"]})

    if "channel" in business_question.lower():
        plan.append({"tool": "ttest", "columns": ["revenue", "channel"]})

    plan.append({"tool": "detect_outliers", "columns": ["revenue", "ad_spend"]})

    return plan

def llm_interpret_tool_results(tool_results):

    insights = []

    # -----------------------
    # Correlation
    # -----------------------
    if "correlation" in tool_results:

        corr = tool_results["correlation"]["correlation"]
        col1 = tool_results["correlation"]["column_1"]
        col2 = tool_results["correlation"]["column_2"]

        insights.append(
            f"{col1} vs {col2} shows correlation of {corr:.2f}."
        )

    # -----------------------
    # T TEST
    # -----------------------
    if "ttest" in tool_results:

        p_value = tool_results["ttest"]["p_value"]
        column = tool_results["ttest"]["column"]
        group = tool_results["ttest"]["group_column"]

        if p_value < 0.05:
            insights.append(
                f"{column} differs significantly across {group} groups (p={p_value:.3f})."
            )
        else:
            insights.append(
                f"{column} shows no significant difference across {group} groups (p={p_value:.3f})."
            )

    # -----------------------
    # OUTLIERS
    # -----------------------
    if "detect_outliers" in tool_results:

        outlier_count = tool_results["detect_outliers"]["outlier_count"]
        column = tool_results["detect_outliers"]["column"]

        if outlier_count > 0:
            insights.append(
                f"{column} contains {outlier_count} potential outlier value(s)."
            )
        else:
            insights.append(
                f"No outliers detected in {column}."
            )

    return insights

