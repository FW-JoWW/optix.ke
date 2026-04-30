from state.state import AnalystState
from tools.load_data import load_csv, load_excel, load_sql_table
from utils.structure_normalizer import choose_best_dataframe
import pandas as pd

def data_intake_node(state: AnalystState) -> AnalystState:
    """
    LangGraph Data Intake Node.
    Detects dataset type and loads it into the state.
    """
    # If dataset is already loaded, skip file/sql loading
    if state.get("dataframe") is not None:
        dataset_path = state.get("dataset_path")
        existing_df = state["dataframe"]
        normalized = None
        try:
            if dataset_path and dataset_path.lower().endswith(".csv"):
                raw_df = pd.read_csv(dataset_path, header=None, low_memory=False)
                normalized = choose_best_dataframe(existing_df, raw_df)
            elif dataset_path and dataset_path.lower().endswith((".xls", ".xlsx")):
                raw_df = pd.read_excel(dataset_path, header=None)
                normalized = choose_best_dataframe(existing_df, raw_df)
        except Exception:
            normalized = None

        if normalized is not None:
            state["dataframe"] = normalized.dataframe
            state.setdefault("analysis_evidence", {})
            state["analysis_evidence"]["structure_normalization"] = {
                "applied": normalized.applied,
                "strategy": normalized.strategy,
                "details": normalized.details,
            }
            if normalized.applied:
                print(f"Dataset normalized using {normalized.strategy}: {normalized.details}")
            else:
                print("Dataset already in state, no structural normalization applied.")
        else:
            print("Dataset already in state, skipping intake load.")
        return state
    # load dataset from file
    dataset_path = state.get("dataset_path")

    if dataset_path is None:
        raise ValueError("No dataset_path provided in state.")

    # Determine loader by file extension or SQL path format
    if dataset_path.lower().endswith(".csv"):
        state = load_csv(state, dataset_path)

    elif dataset_path.lower().endswith((".xls", ".xlsx")):
        state = load_excel(state, dataset_path)

    elif dataset_path.startswith("SQL:"):
        # Expected format in state: "SQL:table_name"
        table_name = dataset_path.split("SQL:")[1]
        connection_string = state.get("sql_connection_string")
        if not connection_string:
            raise ValueError("SQL connection string not provided in state.")
        state = load_sql_table(state, connection_string, table_name)

    else:
        raise ValueError("Unsupported dataset type.")

    # Optional: print dataset info for testing
    df = state.get("dataframe")
    print(f"Loaded dataset with shape: {df.shape}")

    return state

