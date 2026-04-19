from __future__ import annotations

from cleaning_executor import execute_cleaning_actions
from state.state import AnalystState


def cleaning_execution_node(state: AnalystState) -> AnalystState:
    df = state.get("dataframe")
    if df is None:
        raise ValueError("No dataframe found in state.")

    plan = state.get("cleaning_plan", [])
    dataset_profile = (
        state.get("dataset_profile")
        or state.get("analysis_evidence", {}).get("preclean_profile_json", {})
    )
    relationships = (state.get("relationship_signals") or {}).get("relationships", [])

    result = execute_cleaning_actions(
        df=df,
        actions=plan,
        dataset_profile=dataset_profile,
        relationships=relationships,
    )

    state["cleaned_data"] = result["cleaned_df"]
    state.setdefault("analysis_evidence", {})
    state["analysis_evidence"]["cleaning_execution_log"] = result["execution_log"]

    print("\n=== CLEANING EXECUTION COMPLETE ===")
    print(f"Cleaned dataset shape: {state['cleaned_data'].shape}")

    return state
