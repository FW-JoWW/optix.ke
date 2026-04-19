from __future__ import annotations

from data_profiling import profile_dataset
from state.state import AnalystState


def dataset_profiler_node(state: AnalystState) -> AnalystState:
    df = state.get("cleaned_data")
    if df is None:
        df = state.get("dataframe")

    if df is None:
        raise ValueError("No dataframe found in state.")

    profile = profile_dataset(df)
    columns = profile.get("columns", {})

    dataset_profile = {
        "row_count": profile.get("row_count", 0),
        "column_count": profile.get("column_count", 0),
        "numeric_columns": [col for col, info in columns.items() if info.get("inferred_type") == "numeric"],
        "categorical_columns": [col for col, info in columns.items() if info.get("inferred_type") == "categorical"],
        "datetime_columns": [col for col, info in columns.items() if info.get("inferred_type") == "datetime"],
        "identifier_columns": [col for col, info in columns.items() if info.get("inferred_type") == "identifier_like"],
        "missing_values": {col: info.get("missing_count", 0) for col, info in columns.items()},
        "unique_counts": {col: info.get("unique_count", 0) for col, info in columns.items()},
        "column_statistics": {
            col: (
                info.get("distribution_summary")
                or {"unique_count": info.get("unique_count", 0), "unique_ratio": info.get("unique_ratio", 0.0)}
            )
            for col, info in columns.items()
        },
        "columns": columns,
        "pattern_detection": profile.get("pattern_detection", {}),
        "sample_rows": profile.get("sample_rows", []),
    }

    state["dataset_profile"] = dataset_profile
    state.setdefault("analysis_evidence", {})
    state["analysis_evidence"]["dataset_profile_json"] = profile

    print("\n=== DATASET PROFILE ===")
    print(dataset_profile)

    return state
