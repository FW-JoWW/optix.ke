# nodes/cleaning_execution_node.py
from state.state import AnalystState
import pandas as pd
import numpy as np
import json

def cleaning_execution_node(state: AnalystState) -> AnalystState:
    """
    Executes the cleaning plan on the dataset.
    Applies actions defined in state['cleaning_plan'].
    Saves the cleaned dataframe back to state['cleaned_data'].
    """

    df = state.get("dataframe")
    if df is None:
        raise ValueError("No dataframe found in state.")

    if "cleaning_plan" not in state:
        raise ValueError("No cleaning plan found in state.")

    df = df.copy()
    plan = state["cleaning_plan"]

    for step in plan:
        col = step.get("column")
        action = step.get("action")

        # -----------------------------
        # Remove duplicates (dataset-level)
        # -----------------------------
        if action == "remove_duplicates":
            df = df.drop_duplicates()

        # -----------------------------
        # Impute missing values
        # -----------------------------
        elif action == "impute" and col in df.columns:
            if df[col].dtype in ["float64", "int64"]:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])

        # -----------------------------
        # Drop constant columns
        # -----------------------------
        elif action == "drop_column" and col in df.columns:
            df = df.drop(columns=[col])

        # -----------------------------
        # Handle numeric stored as object
        # -----------------------------
        elif action == "convert_to_numeric" and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # -----------------------------
        # High cardinality categorical columns
        # -----------------------------
        elif action == "consider_encoding_or_drop" and col in df.columns:
            # Simple approach: leave it for analysis, optionally encode later
            pass

        # -----------------------------
        # Outliers
        # -----------------------------
        elif action == "investigate_or_cap" and col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df[col] = np.where(df[col] < lower, lower, df[col])
                df[col] = np.where(df[col] > upper, upper, df[col])

    # Save cleaned dataset
    state["cleaned_data"] = df

    print("\n=== CLEANING EXECUTION COMPLETE ===")
    print(f"Cleaned dataset shape: {df.shape}")

    return state