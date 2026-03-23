from tools.correlation_tool import correlation_tool
from tools.ttest_tool import ttest_tool
from tools.outlier_tool import outlier_tool
from tools.regression_tool import regression_tool
from state.state import AnalystState


def tool_executor(state: AnalystState) -> AnalystState:

    df = state.get("dataframe")
    tool_plan = state.get("tool_plan", [])

    tool_results = {}

    for step in tool_plan:

        tool_name = step["tool"]
        columns = step["columns"]

        if tool_name == "correlation":

            result = correlation_tool(df, columns)
            tool_results["correlation"] = result

        elif tool_name == "ttest":

            result = ttest_tool(df, columns)
            tool_results["ttest"] = result

        elif tool_name == "detect_outliers":

            result = outlier_tool(df, columns)
            tool_results["detect_outliers"] = result

        elif tool_name == "regression":

            result = regression_tool(df, columns)
            tool_results["regression"] = result

    #state["tool_results"] = tool_results
    key = f"{tool_name}_{'_'.join(columns)}"
    state["tool_results"][key] = result

    print("\n=== TOOL RESULTS ===")
    print(tool_results)

    return state

