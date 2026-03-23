from state.state import AnalystState
import pandas as pd

def eda_node(state: AnalystState) -> AnalystState:
    """
    Performs descriptive data analysis (EDA).
    Generates summary statistics and dataset insights.
    Stores all results under state["analysis_evidence"].
    """

    df = state.get("analysis_dataset") or state.get("cleaned_data")
    if df is None:
        raise ValueError("No dataset available for EDA")

    print("\n=== RUNNING DESCRIPTIVE ANALYSIS ===")

    # Initialize analysis_evidence if not present
    state.setdefault("analysis_evidence", {})

    # Dataset overview
    state["analysis_evidence"]["dataset_overview"] = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict()
    }

    # Numeric summary
    numeric_cols = df.select_dtypes(include=["int64", "float64"])
    state["analysis_evidence"]["numeric_summary"] = (
        numeric_cols.describe().to_dict() if not numeric_cols.empty else {}
    )

    # Categorical summary
    categorical_cols = df.select_dtypes(include=["object", "category"])
    cat_summary = {col: df[col].value_counts().to_dict() for col in categorical_cols.columns}
    state["analysis_evidence"]["categorical_summary"] = cat_summary

    # Correlations
    if len(numeric_cols.columns) >= 2:
        state["analysis_evidence"]["correlations"] = numeric_cols.corr().to_dict()

    print("EDA Completed")
    return state

'''import pandas as pd
from state.state import AnalystState


def eda_node(state: AnalystState) -> AnalystState:
    """
    Performs descriptive data analysis (EDA).
    Generates summary statistics and dataset insights.
    """

    df = state.get("analysis_dataset") or state.get("cleaned_data")

    if df is None:
        raise ValueError("No dataset available for EDA")

    print("\n=== RUNNING DESCRIPTIVE ANALYSIS ===")

    eda_results = {}

    # Dataset shape
    eda_results["shape"] = df.shape

    # Column types
    eda_results["dtypes"] = df.dtypes.astype(str).to_dict()

    # Missing values
    eda_results["missing_values"] = df.isnull().sum().to_dict()

    # Numeric summary
    numeric_cols = df.select_dtypes(include=["int64", "float64"])

    if not numeric_cols.empty:
        eda_results["numeric_summary"] = numeric_cols.describe().to_dict()

    # Categorical summary
    categorical_cols = df.select_dtypes(include=["object", "category"])

    cat_summary = {}

    for col in categorical_cols.columns:
        cat_summary[col] = df[col].value_counts().to_dict()

    eda_results["categorical_summary"] = cat_summary

    # Correlation matrix
    if len(numeric_cols.columns) >= 2:
        eda_results["correlations"] = numeric_cols.corr().to_dict()

    state["eda_results"] = eda_results

    print("EDA Completed")

    return state

'''