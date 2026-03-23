# nodes/numeric_cleaning_node.py

import pandas as pd
from state.state import AnalystState
import re

def numeric_cleaning_node(state: AnalystState) -> AnalystState:
    """
    Robust numeric cleaning for analysis_dataset.
    Handles:
        - $ signs, commas, spaces
        - European decimals (e.g., '49,6' → 49.6)
        - Units and text mixed in numeric fields
        - Ensures proper float/int dtype
    Updates state['analysis_dataset'] and logs cleaned columns.
    """

    df = state.get("analysis_dataset")
    if df is None:
        raise ValueError("analysis_dataset not found in state")

    df = df.copy()

    dataset_profile = state.get("dataset_profile", {})
    column_registry = state.get("column_registry", {})

    selected_columns = state.get("selected_columns", [])
    numeric_cols =[
        col for col in dataset_profile.get("numeric_columns", [])
        if col in selected_columns
    ] 
    cleaned_columns = []

    if not state.get("apply_numeric_cleaning", True):
        print("Numeric cleaning skipped.")
        return state

    for col in numeric_cols:
        if col not in df.columns:
            continue

        # Skip identifiers
        if column_registry.get(col, {}).get("semantic_role") == "identifier":
            continue

        # If already numeric, no cleaning needed
        if pd.api.types.is_numeric_dtype(df[col]):
            cleaned_columns.append(col)
            continue
        
        # Only clean object/string dtype
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            cleaned = []
            for val in df[col]:
                if pd.isna(val):
                    cleaned.append(None)
                    continue

                s = str(val).strip()

                # Remove currency symbols, spaces, units
                s = re.sub(r'[^\d,.\-]', '', s)

                # Handle European decimals: if there is a comma with no dot, treat comma as decimal
                if ',' in s and '.' not in s:
                    s = s.replace(',', '.')

                # Remove thousands separator if dot exists before comma(e.g., '1.234,56' → '1234.56')
                if re.match(r'^\d{1,3}(\.\d{3})+,\d+$', s):
                    s = s.replace('.', '')
                    s = s.replace(',', '.')

                # Final numeric conversion
                try:
                    num = float(s)
                except:
                    num = None

                cleaned.append(num)

            df[col] = pd.Series(cleaned)
            cleaned_columns.append(col)
            print(f"[CLEANED NUMERIC] {col}")

    # Update state
    state["analysis_dataset"] = df
    state.setdefault("analysis_evidence", {})
    state["analysis_evidence"]["numeric_cleaning"] = {
        "cleaned_columns": cleaned_columns
    }

    print("\n=== NUMERIC CLEANING COMPLETE ===")
    print("Cleaned columns:", cleaned_columns)

    return state

