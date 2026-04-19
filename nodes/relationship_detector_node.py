from __future__ import annotations

from relationship_detector import detect_relationships
from state.state import AnalystState


def relationship_detector_node(state: AnalystState) -> AnalystState:
    df = state.get("cleaned_data")
    if df is None:
        df = state.get("dataframe")

    dataset_profile = state.get("analysis_evidence", {}).get("dataset_profile_json")
    if df is None or not dataset_profile:
        state["relationship_signals"] = {"relationships": [], "derived_columns": []}
        return state

    result = detect_relationships(df, dataset_profile)
    state["relationship_signals"] = result
    state.setdefault("analysis_evidence", {})
    state["analysis_evidence"]["relationship_signals"] = result

    print("\n=== RELATIONSHIP DETECTION ===")
    print(result)
    return state
