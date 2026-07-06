import pandas as pd
from state.state import AnalystState

def profile_dataset(state: AnalystState) -> AnalystState:
    """
    Generates a basic dataset profile and stores it in the state.
    """

    df = state.get("dataframe")
    if df is None:
        df = state.get("dataset")

    if df is None:
        raise ValueError("No dataframe found in state.")

    profile = {}

    # shape
    profile["rows"] = df.shape[0]
    profile["columns"] = df.shape[1]

    # column types
    profile["column_types"] = df.dtypes.astype(str).to_dict()

    # numeric vs categorical
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["str", "category"]).columns.tolist()

    profile["numeric_columns"] = numeric_cols
    profile["categorical_columns"] = categorical_cols
    profile["row_count"] = profile["rows"]
    profile["column_count"] = profile["columns"]

    # missing values
    profile["missing_values"] = df.isnull().sum().to_dict()

    # numeric statistics
    if numeric_cols:
        profile["summary_stats"] = df[numeric_cols].describe().to_dict()

    state["dataset_profile"] = profile
    state["analysis_evidence"] = state.get("analysis_evidence", {})
    state["analysis_evidence"]["dataset_profile_json"] = profile

    return state

