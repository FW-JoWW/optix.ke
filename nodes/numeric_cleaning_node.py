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
    warnings = []

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
            before_na = df[col].isna().sum()
            cleaned = []
            for val in df[col]:
                if pd.isna(val):
                    cleaned.append(None)
                    continue

                s = str(val).strip()

                # Remove currency symbols, spaces, units
                s = re.sub(r'[^\d,.\-]', '', s)

                # Case 1: European format "1.234,56"
                if re.match(r'^\d{1,3}(\.\d{3})+,\d+$', s):
                    s = s.replace('.', '')
                    s = s.replace(',', '.')
                
                # Case 2: Thousands format "7,495"
                elif re.match(r'^\d{1,3}(,\d{3})+$', s):
                    s = s.replace(',', '')
                
                # Case 3: Decimal comma "49,6"
                elif ',' in s and '.' not in s:
                    # Only treat as decimal if NOT thousands pattern
                    s = s.replace(',', '.')

                # Final numeric conversion
                try:
                    num = float(s)
                except:
                    num = None

                cleaned.append(num)

            df[col] = pd.Series(cleaned, index=df.index)
            after_na = df[col].isna().sum()
            cleaned_columns.append(col)
            print(f"[CLEANED NUMERIC] {col}")

            # Validation 
            if after_na > before_na:
                warnings.append(f"{col}: NaN increased ({before_na} → {after_na})")

            # Detect suspicious scaling
            col_mean = df[col].mean()
            col_max = df[col].max()
            
            if pd.notna(col_mean) and pd.notna(col_max):
                if col_mean != 0:
                    ratio = col_max / col_mean
            
                    # Very skewed or tiny values
                    if ratio > 1000:
                        warnings.append(f"{col}: highly skewed distribution (max/mean > 1000)")
            
                    if col_mean < 1:
                        warnings.append(f"{col}: mean very small → possible scaling issue")
            
            '''if col == "price":
                if df[col].mean() < 100:  # suspicious for price dataset
                    print(f"[WARNING] {col} values look scaled incorrectly")'''
    # 🔍 DEBUG (keep this for now)
    print("\n=== NUMERIC DEBUG SUMMARY ===")
    for col in cleaned_columns[:3]:  # limit to avoid spam
        if col in df.columns:
            print(f"\n[{col}]")
            print(df[col].describe())
            print(df[col].head(5))
    '''if "price" in df.columns:
        print("\nDEBUG PRICE AFTER CLEANING:")
        print(df["price"].describe())
        print(df["price"].head(10))
    '''
    # Update state
    state["analysis_dataset"] = df
    state.setdefault("analysis_evidence", {})
    state["analysis_evidence"]["numeric_cleaning"] = {
        "cleaned_columns": cleaned_columns,
        "warnings": warnings
    }

    print("\n=== NUMERIC CLEANING COMPLETE ===")
    print("Cleaned columns:", cleaned_columns)
    if warnings:
        print("\n===NUMERIC WARNINGS ===")
        for w in warnings:
            print("-", w)

    return state

