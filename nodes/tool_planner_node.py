from state.state import AnalystState


def tool_planner_node(state: AnalystState) -> AnalystState:

    analysis_plan = state.get("analysis_plan", [])

    tool_plan = []

    # Convert analysis steps into executable tools
    for step in analysis_plan:

        tool = step.get("tool")
        columns = step.get("columns", [])

        if tool in [
            "correlation",
            "anova",
            "ttest",
            "detect_outliers",
            "regression",
            "summary_statistics"
        ]:

            tool_plan.append({
                "tool": tool,
                "columns": columns
            })

    state["tool_plan"] = tool_plan

    print("\n=== TOOL PLAN ===")
    for t in tool_plan:
        print(t)

    return state

