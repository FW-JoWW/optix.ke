from tools.anova_tool import anova_tool
from tools.categorical_analysis_tool import categorical_analysis_tool
from tools.correlation_tool import correlation_tool
from tools.outlier_tool import outlier_tool
from tools.regression_tool import regression_tool
from tools.summary_statistics_tool import summary_statistics_tool
from tools.ttest_tool import ttest_tool
from state.state import AnalystState

TOOL_MAPPING = {
    "correlation": correlation_tool,
    "ttest": ttest_tool,
    "detect_outliers": outlier_tool,
    "summary_statistics": summary_statistics_tool,
    "regression": regression_tool,
    "anova": anova_tool,
    "categorical_analysis": categorical_analysis_tool,
}


def tool_executor_node(state: AnalystState) -> AnalystState:
    """
    Executes analysis tools decided by the planner.
    """
    print("STATE KEYS BEFORE TOOL EXECUTION:", list(state.keys()))

    evidence = state.setdefault("analysis_evidence", {})

    df = None
    if state.get("analysis_dataset") is not None:
        df = state["analysis_dataset"]
    elif state.get("cleaned_data") is not None:
        df = state["cleaned_data"]
    elif state.get("dataframe") is not None:
        df = state["dataframe"]

    if df is None:
        print("WARNING: No dataset available for tools.")
        return state

    tool_plan = evidence.get("analysis_plan", [])
    if not tool_plan:
        print("No statistical tools required for this analysis.")
        evidence["tool_results"] = {}
        return state

    tool_results = {}

    for task in tool_plan:
        tool_name = task["tool"]
        columns = task.get("columns", [])
        tool_func = TOOL_MAPPING.get(tool_name)

        if not tool_func:
            print(f"Skipping unknown tool: {tool_name}")
            continue

        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            print(f"Skipping {tool_name}, missing columns: {missing_cols}")
            continue

        try:
            print(f"\nRunning {tool_name} on {columns}")

            if tool_name == "anova":
                result = tool_func(df, columns[0], columns[1])
            elif tool_name in ["ttest", "correlation", "detect_outliers", "summary_statistics"]:
                result = tool_func(df, columns)
            elif tool_name == "regression":
                result = tool_func(df, x_col=columns[0], y_col=columns[1])
            elif tool_name == "categorical_analysis":
                result = tool_func(
                    df,
                    columns=columns,
                    config=state.get("categorical_analysis_config"),
                )
            else:
                print(f"Unsupported tool format: {tool_name}")
                continue

            if result is not None:
                key = f"{tool_name}_{'_'.join(columns)}" if columns else tool_name
                tool_results[key] = result

        except Exception as e:
            tool_results[f"{tool_name}_error"] = str(e)

    evidence["tool_results"] = tool_results

    print("\n=== TOOL RESULTS ===")
    print(tool_results)

    return state
