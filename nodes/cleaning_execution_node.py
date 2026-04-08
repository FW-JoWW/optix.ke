# nodes/cleaning_execution_node.py
from state.state import AnalystState
import pandas as pd
import numpy as np
import re

def apply_numeric_cleaning(df, state):
    dataset_profile = state.get("dataset_profile", {})
    column_registry = state.get("column_registry", {})
    #selected_columns = state.get("selected_columns", [])

    '''numeric_cols = [
        col for col in dataset_profile.get("numeric_columns", [])
        if col in selected_columns
    ]'''
    numeric_cols = dataset_profile.get("numeric_columns", [])

    cleaned_columns = []
    warnings = []

    for col in numeric_cols:
        if col not in df.columns:
            continue

        # Skip identifiers
        if column_registry.get(col, {}).get("semantic_role") == "identifier":
            continue

        # Already numeric -> skip
        if pd.api.types.is_numeric_dtype(df[col]):
            cleaned_columns.append(col)
            continue

        # Clean string/object numeric columns
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            before_na = df[col].isna().sum()
            cleaned = []

            for val in df[col]:
                if pd.isna(val):
                    cleaned.append(None)
                    continue

                s = str(val).strip()
                s = re.sub(r'[^\d,.\-]', '', s)

                # European format
                if re.match(r'^\d{1,3}(\.\d{3})+,\d+$', s):
                    s = s.replace('.', '').replace(',', '.')
                # Thousand format
                elif re.match(r'^\d{1,3}(,\d{3})+$', s):
                    s = s.replace(',', '')
                # Decimal comma
                elif ',' in s and '.' not in s:
                    s = s.replace(',', '.')

                try:
                    num = float(s)
                except:
                    num = None

                cleaned.append(num)

            df[col] = pd.Series(cleaned, index=df.index)
            
            after_na = df[col].isna().sum()
            cleaned_columns.append(col)

            if after_na > before_na:
                warnings.append(f"{col}: NaN increased ({before_na} → {after_na})")

    return df, cleaned_columns, warnings


def cleaning_execution_node(state: AnalystState) -> AnalystState:
    df = state.get("dataframe")
    if df is None:
        raise ValueError("No dataframe found in state.")

    if "cleaning_plan" not in state:
        raise ValueError("No cleaning plan found in state.")

    df = df.copy()
    plan = state["cleaning_plan"]

    #state.setdefault("analysis_evidence", {})

    # -----------------------------
    # STEP 1: Basic cleaning first
    # -----------------------------
    for step in plan:
        col = step.get("column")
        action = step.get("action")

        # ------------------
        # NUMERIC CLEANING
        # ------------------
        if action == "numeric_cleaning":
            df, cleaned_cols, warnings = apply_numeric_cleaning(df, state)
        
            state["analysis_evidence"]["numeric_cleaning"] = {
                "cleaned_columns": cleaned_cols,
                "warnings": warnings
            }
        
        # -------------------
        # REMOVE DUPLICATES
        # -------------------
        elif action == "remove_duplicates":
            df = df.drop_duplicates()

        # -----------------
        # IMPUT MISSING 
        # -----------------
        elif action == "impute" and col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])

        # --------------
        # DROP COLUMN
        # --------------
        elif action == "drop_column" and col in df.columns:
            df = df.drop(columns=[col])

        # ----------------------------
        # SIMPLE NUMERIC CONVERSION
        # ----------------------------
        elif action == "convert_to_numeric" and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # -------------------------
        # OUTLIER CAPPING
        # -------------------------
        elif action == "investigate_or_cap" and col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df[col] = np.clip(df[col], lower, upper)

    # ----------------------------------
    # POST-CLEANING VALIDATION
    # ----------------------------------
    for col in df.select_dtypes(include=[np.number]).columns:
        col_mean = df[col].mean()
        col_max = df[col].max()

        if pd.notna(col_mean) and pd.notna(col_max) and col_mean != 0:
            ratio = col_max / col_mean

            if ratio > 1000:
                state["analysis_evidence"].setdefault("scaling_warnings", []).append(
                    f"{col}: highly skewed distribution"
                )

            if col_mean < 1:
                state["analysis_evidence"].setdefault("scaling_warnings", []).append(
                    f"{col}: mean very small → possible scaling issue"
                )
    
    # -----------------------------
    # FINAL OUTPUT
    # -----------------------------
    state["cleaned_data"] = df

    print("\n=== CLEANING EXECUTION COMPLETE ===")
    print(f"Cleaned dataset shape: {df.shape}")

    return state

'''# nodes/cleaning_execution_node.py
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

    return state'''