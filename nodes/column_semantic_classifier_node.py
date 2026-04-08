# nodes/column_semantic_classifier_node.py
from state.state import AnalystState
import pandas as pd
import numpy as np

# Optional: import LLM helper if you want column-level semantic reasoning
# from nodes.llm_utils import llm_infer_column_role  

def column_semantic_classifier_node(state: AnalystState) -> AnalystState:
    """
    Fully generated Column Semantic Classifier Node.
    Classifies each column into semantic roles and assigns cleaning rules.
    Stores results in state['column_registry'].
    """

    if "cleaned_data" in state and state["cleaned_data"] is not None:
        state["active_dataset"] = "cleaned_data"
    else:
        state["active_dataset"] = "dataframe"
    '''df = state.get("cleaned_data")
    if df is None:
        df = state.get("dataframe")'''
    df = state.get(state.get("active_dataset"))
        
    profile = state.get("dataset_profile")

    if df is None:
        raise ValueError("No dataframe found in state for semantic classification.")

    if profile is None:
        raise ValueError("Dataset profile missing in state.")

    column_registry = {}

    for col in df.columns:

        col_info = {}

        # -----------------------------
        # 1️⃣ Base type from dataset profile
        # -----------------------------
        if col in profile.get("numeric_columns", []):
            col_info["type"] = "numeric"
        elif col in profile.get("categorical_columns", []):
            col_info["type"] = "categorical"
        elif col in profile.get("datetime_columns", []):
            col_info["type"] = "datetime"
        else:
            col_info["type"] = "unknown"

        # -----------------------------
        # 2️⃣ Semantic role via heuristics + distribution
        # -----------------------------
        col_lower = col.lower()
        semantic_role = "irrelevant"

        if "id" in col_lower:
            semantic_role = "identifier"
        elif "name" in col_lower or "first" in col_lower or "last" in col_lower:
            semantic_role = "personal_attribute"
        elif "date" in col_lower or col_info["type"] == "datetime":
            semantic_role = "datetime"
        elif col in profile.get("identifier_columns", []):
            semantic_role = "identifier"
        elif col_info["type"] == "numeric":
                semantic_role = "numeric_measure"
        elif col_info["type"] == "categorical":
            semantic_role = "categorical_feature"

        # Optional: LLM-assisted disambiguation for tricky columns
        # semantic_role = llm_infer_column_role(col, df[col].sample(min(len(df), 20)).tolist(), semantic_role)

        col_info["semantic_role"] = semantic_role

        # -----------------------------
        # 3️⃣ Assign cleaning rules
        # -----------------------------
        rules = []
        if semantic_role == "identifier":
            rules = ["must_be_unique", "never_impute"]
        elif semantic_role == "numeric_measure":
            rules = ["detect_outliers", "impute_missing"]
        elif semantic_role == "categorical_feature":
            rules = ["standardize_categories", "fix_capitalization", "remove_whitespace"]
        elif semantic_role == "datetime":
            rules = ["standardize_format", "check_chronology"]
        elif semantic_role == "personal_attribute":
            rules = ["trim_whitespace", "handle_privacy"]
        else:
            rules = ["ignore"]

        col_info["cleaning_rules"] = rules

        # -----------------------------
        # 4️⃣ Save info to registry
        # -----------------------------
        column_registry[col] = col_info

    state["column_registry"] = column_registry

    # -----------------------------
    # Print summary
    # -----------------------------
    print("\n=== COLUMN SEMANTIC CLASSIFICATION ===")
    for col, info in column_registry.items():
        print(f"{col}: {info['semantic_role']} → {info['cleaning_rules']}")

    return state