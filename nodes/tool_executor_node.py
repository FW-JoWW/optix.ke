from tools.correlation_tool import correlation_tool
from tools.ttest_tool import ttest_tool
from tools.outlier_tool import outlier_tool
from tools.summary_statistics_tool import summary_statistics_tool
from tools.regression_tool import regression_tool
from tools.anova_tool import anova_tool
from state.state import AnalystState

TOOL_MAPPING = {
    "correlation": correlation_tool,
    "ttest": ttest_tool,
    "detect_outliers": outlier_tool,
    "summary_statistics": summary_statistics_tool,
    "regression": regression_tool,
    "anova": anova_tool
}

def tool_executor_node(state: AnalystState) -> AnalystState:
    """
    Executes statistical tools decided by the tool planner.
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
        #state["tool_results"] = {}
        return state

    tool_plan = evidence.get("analysis_plan", [])

    # If nothing to execute, skip
    if not tool_plan:
        print("No statistical tools required for this analysis.")
        evidence["tool_results"] = {}
        return state
    
    tool_results = {}
    #if "tool_result" not in state:
        #state["tool_results"] = {}

    for task in tool_plan:
        tool_name = task["tool"]
        columns = task.get("columns", [])

        tool_func = TOOL_MAPPING.get(tool_name)

        if not tool_func:
            print(f"Skipping unknown tool: {tool_name}")
            continue
        
        # Validate missing columns exist
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            print(f"Skipping {tool_name}, missing columns: {missing_cols}")
            continue

        try:
            print(f"\nRunning {tool_name} on {columns}")
            
            if tool_name in ["anova", "ttest", "correlation"]:
                result = tool_func(df, columns[0], columns[1])

            elif tool_name == "regression":
                result = tool_func(df, x_col=columns[0], y_col=columns[1])

            elif tool_name == "detect_outliers":
                result = tool_func(df, columns[0])

            elif tool_name == "summary_statistics":
                result = tool_func(df, columns)

            else:
                print(f"⚠️ Unsupported tool format: {tool_name}")
                continue

            #result = tool_func(df, columns)
            if result is not None:
               key = f"{tool_name}_{'_'.join(columns)}"
               tool_results[key] = result

        except Exception as e:
            tool_results[f"{tool_name}_error"] = str(e)    

    evidence["tool_results"] = tool_results

    print("\n=== TOOL RESULTS ===")
    print(tool_results)

    return state

