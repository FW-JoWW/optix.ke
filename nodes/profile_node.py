from state.state import AnalystState
from tools.dataset_profiler import profile_dataset

def profiling_node(state: AnalystState) -> AnalystState:
    """
    LangGraph node that profiles the dataset.
    """

    state = profile_dataset(state)

    profile = state.get("dataset_profile")

    print("\n=== DATASET PROFILE ===")
    print(f"Rows: {profile['rows']}")
    print(f"Columns: {profile['columns']}")
    print(f"Numeric Columns: {profile['numeric_columns']}")
    print(f"Categorical Columns: {profile['categorical_columns']}")

    return state

