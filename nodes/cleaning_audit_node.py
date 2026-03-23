# nodes/cleaning_audit_node.py
from state.state import AnalystState
import pandas as pd
import numpy as np
import json

def cleaning_audit_node(state: AnalystState) -> AnalystState:
    """
    Verifies the cleaning execution results.
    Checks if the planned actions were applied and logs remaining issues.
    Saves a summary in state['cleaning_audit'].
    """

    df = state.get("cleaned_data")
    plan = state.get("cleaning_plan")

    if df is None:
        raise ValueError("No cleaned_data found in state.")

    if plan is None:
        raise ValueError("No cleaning plan found in state.")

    audit_results = []

    for step in plan:
        col = step.get("column")
        action = step.get("action")
        status = "not_checked"

        # -----------------------------
        # Check missing value imputation
        # -----------------------------
        if action == "impute" and col in df.columns:
            remaining_missing = df[col].isnull().sum()
            status = f"{remaining_missing} missing values remaining" if remaining_missing > 0 else "OK"

        # -----------------------------
        # Check duplicates removal
        # -----------------------------
        elif action == "remove_duplicates":
            duplicates = df.duplicated().sum()
            status = f"{duplicates} duplicates remaining" if duplicates > 0 else "OK"

        # -----------------------------
        # Check column drop
        # -----------------------------
        elif action == "drop_column":
            status = "Dropped" if col not in df.columns else "Still exists"

        # -----------------------------
        # Check numeric conversion
        # -----------------------------
        elif action == "convert_to_numeric" and col in df.columns:
            non_numeric = df[col].apply(lambda x: not pd.api.types.is_number(x) if x is not None else False).sum()
            status = f"{non_numeric} non-numeric values remaining" if non_numeric > 0 else "OK"

        # -----------------------------
        # Check outliers capping
        # -----------------------------
        elif action == "investigate_or_cap" and col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            status = f"{len(outliers)} outliers remaining" if len(outliers) > 0 else "OK"

        audit_results.append({
            "column": col,
            "action": action,
            "status": status
        })

    state["cleaning_audit"] = audit_results

    print("\n=== CLEANING AUDIT RESULTS ===")
    print(json.dumps(audit_results, indent=2))

    return state