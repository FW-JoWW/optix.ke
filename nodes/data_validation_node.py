from state.state import AnalystState
import pandas as pd
import numpy as np
from typing import Dict, Any

def data_validation_node(state: AnalystState) -> AnalystState:
    """
    Validates the dataset and updates the AnalystState with findings.
    Performs:
    - Missing value check
    - Duplicate row check
    - Column type detection
    - Basic summary statistics
    - Outlier detection (numeric columns)
    """

    df: pd.DataFrame | None = state.get("cleaned_data")
    if df is None:
        df = state.get("dataframe")

    if df is None:
        state["data_validation"] = {"error": "No dataset provided."}
        return state

    validation: Dict[str, Any] = {}

    # Column types
    validation["column_types"] = df.dtypes.apply(lambda x: str(x)).to_dict()

    # Missing values
    validation["missing_values"] = df.isnull().sum().to_dict()

    # Duplicate rows
    validation["duplicates"] = int(df.duplicated().sum())

    # Basic descriptive stats for numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    validation["summary_statistics"] = df[numeric_cols].describe().to_dict() if numeric_cols else {}

    # Outlier detection using IQR for numeric columns
    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].tolist()
    validation["outliers"] = outliers

    # Optional warnings
    warnings = []
    if any(v > 0 for v in validation["missing_values"].values()):
        warnings.append("Dataset contains missing values.")
    if validation["duplicates"] > 0:
        warnings.append(f"{validation['duplicates']} duplicate rows detected.")
    if len(numeric_cols) == 0:
        warnings.append("No numeric columns detected; some analysis may be limited.")
    validation["warnings"] = warnings

    # Update state
    state["data_validation"] = validation

    # Optional: ask for clarification if major issues exist
    state["clarification_questions"] = []
    if any(v > 0 for v in validation["missing_values"].values()):
        state["clarification_questions"].append(
            "Some columns have missing values. Should we drop rows, fill missing, or leave as is?"
        )

    if validation["duplicates"] > 0:
        state["clarification_questions"].append(
            "Duplicate rows detected. Should we remove them?"
        )

    return state
