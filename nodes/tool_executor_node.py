from __future__ import annotations

from analysis_engine import execute_analysis_plan
from state.state import AnalystState


def tool_executor_node(state: AnalystState) -> AnalystState:
    print("STATE KEYS BEFORE TOOL EXECUTION:", list(state.keys()))

    evidence = state.setdefault("analysis_evidence", {})
    df = state.get("analysis_dataset")
    if df is None:
        df = state.get("cleaned_data")
    if df is None:
        df = state.get("dataframe")

    if df is None:
        print("WARNING: No dataset available for tools.")
        return state

    tool_plan = evidence.get("analysis_plan", [])
    if not tool_plan:
        print("No statistical tools required for this analysis.")
        evidence["tool_results"] = {}
        return state

    tool_results = execute_analysis_plan(
        df=df,
        plan=tool_plan,
        config=state.get("categorical_analysis_config"),
    )
    evidence["tool_results"] = tool_results

    print("\n=== TOOL RESULTS ===")
    print(tool_results)

    return state
