# nodes/cleaning_strategy_planner_node.py
from state.state import AnalystState
import json


def cleaning_strategy_planner_node(state: AnalystState) -> AnalystState:
    """
    Generates a robust cleaning plan:
    - Adds global cleaning steps (numeric cleaning, duplicates)
    - Maps data quality issues to actions
    - Ensures proper execution order
    """

    if "data_quality_issues" not in state:
        raise ValueError("No data_quality_issues found in state.")

    issues = state.get("data_quality_issues", {}).get("issues", [])
    cleaning_plan = []

    # ----------------------------------------
    # STEP 1: GLOBAL ACTIONS (ALWAYS FIRST)
    # ----------------------------------------

    # Numeric cleaning → always needed if numeric columns exist
    dataset_profile = state.get("dataset_profile", {})
    if dataset_profile.get("numeric_columns"):
        cleaning_plan.append({
            "column": None,
            "action": "numeric_cleaning",
            "severity": "high",
            "explanation": "Normalize numeric columns before other operations"
        })

    # Remove duplicates → optional but safe default
    cleaning_plan.append({
        "column": None,
        "action": "remove_duplicates",
        "severity": "medium",
        "explanation": "Remove duplicate rows"
    })

    # ----------------------------------------
    # STEP 2: ISSUE-BASED ACTIONS
    # ----------------------------------------
    for issue in issues:
        col = issue.get("column")
        action = issue.get("recommended_action")
        severity = issue.get("severity", "medium")
        explanation = issue.get("explanation", "")

        # Skip invalid actions
        if not action:
            continue

        plan_step = {
            "column": col,
            "action": action,
            "severity": severity,
            "explanation": explanation
        }

        cleaning_plan.append(plan_step)

    # ----------------------------------------
    # STEP 3: NORMALIZATION / DEDUPLICATION
    # ----------------------------------------
    seen = set()
    unique_plan = []

    for step in cleaning_plan:
        key = (step["action"], step["column"])
        if key not in seen:
            unique_plan.append(step)
            seen.add(key)

    # ----------------------------------------
    # STEP 4: SORT BY SEVERITY (BUT KEEP GLOBAL FIRST)
    # ----------------------------------------
    severity_order = {"high": 0, "medium": 1, "low": 2}

    global_actions = [s for s in unique_plan if s["column"] is None]
    column_actions = [s for s in unique_plan if s["column"] is not None]

    column_actions.sort(key=lambda x: severity_order.get(x["severity"], 3))

    final_plan = global_actions + column_actions

    # ----------------------------------------
    # SAVE TO STATE
    # ----------------------------------------
    state["cleaning_plan"] = cleaning_plan

    print("\n=== CLEANING PLAN GENERATED ===")
    print(json.dumps(cleaning_plan, indent=2))

    return state

'''# nodes/cleaning_strategy_planner_node.py
from state.state import AnalystState
import json

def cleaning_strategy_planner_node(state: AnalystState) -> AnalystState:
    """
    Generates a detailed cleaning plan based on the detected data quality issues.
    Each issue in state['data_quality_issues'] is mapped to a cleaning action.
    """

    if "data_quality_issues" not in state:
        raise ValueError("No data_quality_issues found in state.")

    issues = state["data_quality_issues"]["issues"]
    cleaning_plan = []

    for issue in issues:
        col = issue.get("column")
        action = issue.get("recommended_action")
        severity = issue.get("severity")
        explanation = issue.get("explanation")

        plan_step = {
            "column": col,
            "action": action,
            "severity": severity,
            "explanation": explanation
        }

        # Optional prioritization: high severity issues first
        cleaning_plan.append(plan_step)

    # Sort plan by severity
    severity_order = {"high": 0, "medium": 1, "low": 2}
    cleaning_plan.sort(key=lambda x: severity_order.get(x["severity"], 3))

    # Save plan to state
    state["cleaning_plan"] = cleaning_plan

    print("\n=== CLEANING PLAN GENERATED ===")
    print(json.dumps(cleaning_plan, indent=2))

    return state
'''