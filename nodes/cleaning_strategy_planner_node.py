# nodes/cleaning_strategy_planner_node.py
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