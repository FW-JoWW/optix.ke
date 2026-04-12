# nodes/cleaning_strategy_planner_node.py
import json

import pandas as pd

from state.state import AnalystState


def _has_numeric_like_columns(df: pd.DataFrame) -> bool:
    """
    Detect whether the raw dataset contains numeric or numeric-like columns that
    should be normalized before later cleaning and analysis steps.
    """
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            return True

        cleaned = (
            series.astype(str)
            .str.replace(r"[^\d,.\-]", "", regex=True)
            .replace("", pd.NA)
        )
        coerced = pd.to_numeric(cleaned, errors="coerce")
        if coerced.notna().mean() >= 0.8:
            return True

    return False


def cleaning_strategy_planner_node(state: AnalystState) -> AnalystState:
    """
    Generates a robust cleaning plan:
    - Adds global cleaning steps first
    - Maps detected issues to execution actions
    - Ensures deterministic ordering and deduplication
    """
    if "data_quality_issues" not in state:
        raise ValueError("No data_quality_issues found in state.")

    issues = state.get("data_quality_issues", {}).get("issues", [])
    df = state.get("dataframe")
    cleaning_plan = []

    # Always normalize numeric-like columns before issue-specific actions.
    if df is not None and _has_numeric_like_columns(df):
        cleaning_plan.append({
            "column": None,
            "action": "numeric_cleaning",
            "severity": "high",
            "explanation": "Normalize numeric columns before other operations",
        })

    cleaning_plan.append({
        "column": None,
        "action": "remove_duplicates",
        "severity": "medium",
        "explanation": "Remove duplicate rows",
    })

    for issue in issues:
        action = issue.get("recommended_action")
        if not action:
            continue

        cleaning_plan.append({
            "column": issue.get("column"),
            "action": action,
            "severity": issue.get("severity", "medium"),
            "explanation": issue.get("explanation", ""),
        })

    seen = set()
    unique_plan = []
    for step in cleaning_plan:
        key = (step["action"], step["column"])
        if key not in seen:
            unique_plan.append(step)
            seen.add(key)

    severity_order = {"high": 0, "medium": 1, "low": 2}
    global_actions = [s for s in unique_plan if s["column"] is None]
    column_actions = [s for s in unique_plan if s["column"] is not None]
    column_actions.sort(key=lambda x: severity_order.get(x["severity"], 3))

    final_plan = global_actions + column_actions
    state["cleaning_plan"] = final_plan

    print("\n=== CLEANING PLAN GENERATED ===")
    print(json.dumps(final_plan, indent=2))

    return state
