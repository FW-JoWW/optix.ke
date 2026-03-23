# nodes/row_filter_node.py

import pandas as pd
import re
from difflib import get_close_matches
from state.state import AnalystState


def parse_number(value: str):
    """Convert numbers with k/m to actual floats."""
    value = value.lower().strip()
    if "k" in value:
        return float(value.replace("k", "")) * 1_000
    if "m" in value:
        return float(value.replace("m", "")) * 1_000_000
    
    return float(value)


def fuzzy_match(term, choices):
    """Return closest match for a categorical term."""
    matches = get_close_matches(term.lower(),[str(c).lower() for c in choices], n=1, cutoff=0.6)
    return matches[0] if matches else None


def apply_filters(df, group, column_registry, filters_applied):
    """Apply categorical and numeric filter to dataframe group"""
    temp_df = df.copy()

    # -----------------------------
    # 1. CATEGORICAL FILTERS
    # -----------------------------
    for col, meta in column_registry.items():
        if col not in temp_df.columns:
            continue

        if meta.get("semantic_role") == "categorical_feature":
            values = temp_df[col].dropna().astype(str).str.lower().unique()
            for word in group.split():
                match = fuzzy_match(word, values)
                if match:
                    temp_df = temp_df[temp_df[col].str.lower() == match]
                    filters_applied.append(f"{col} ≈ {match}")
                    print(f"[FILTER] {col} ≈ {match}")
                    break

    # -----------------------------
    # 2. BETWEEN (FIXED)
    # -----------------------------
    between_pattern = r'(\w+)_between_([\d\.kKmM]+)_([\d\.kKmM]+)'
    for col, low, high in re.findall(between_pattern, group):
        if col in temp_df.columns:
            low_val = parse_number(low)
            high_val = parse_number(high)
            temp_df = temp_df[(temp_df[col] >= low_val) & (temp_df[col] <= high_val)]
            filters_applied.append(f"{col} between {low_val}-{high_val}")
            print(f"[FILTER] {col} between {low_val}-{high_val}")
   
   
    # -----------------------------
    # 3. STANDARD OPERATORS
    # -----------------------------
    op_pattern = r'(\w+)\s*(>=|<=|>|<|=)\s*([\d\.kKmM]+)'
    op_matches = re.findall(op_pattern, group)

    for col, op, val in op_matches:
        if col in temp_df.columns:
            val = parse_number(val)

            if op == ">":
                temp_df = temp_df[temp_df[col] > val]
            elif op == "<":
                temp_df = temp_df[temp_df[col] < val]
            elif op == ">=":
                temp_df = temp_df[temp_df[col] >= val]
            elif op == "<=":
                temp_df = temp_df[temp_df[col] <= val]
            elif op == "=":
                temp_df = temp_df[temp_df[col] == val]

            filters_applied.append(f"{col} {op} {val}")
            print(f"[FILTER] {col} {op} {val}")

    return temp_df


def row_filter_node(state: AnalystState) -> AnalystState:
    """Main node to filter rows based on business question."""
    df = state.get("analysis_dataset")
    question = state.get("business_question", "").lower()
    filter_keywords = ["under", "over", "between", "above", "below", "less", "greater", "where"]
    column_registry = state.get("column_registry", {})
    available_columns = set(df.columns)
    filtered_registry = {
        col: meta for col, meta in column_registry.items()
        if col in available_columns
    }
    
    if not any(k in question for k in filter_keywords):
        print("No filtering intent detected — skipping row filtering.")
        return state
          
    if df is None:
        raise ValueError("analysis_dataset not found in state")

    df = df.copy()
    filters_applied = []

    # ------------------
    # Normalize question
    # ------------------
    # Step 1: 
    question = re.sub(r"\bwith\b", "and", question)
    question = re.sub(
        r'(\w+)\s+between\s+([\d\.kKmM]+)\s+and\s+([\d\.kKmM]+)',
        r'\1_between_\2_\3',
        question
    )
    

    # -----------------------------
    # LOGICAL GROUPING FIX
    # -----------------------------
    # Step 2: split by AND first (higher priority)
    and_groups = re.split(r"\s+and\s+", question)
    result_df = df.copy()

    for and_group in and_groups:
        # Inside each AND group, handle OR
        or_groups = re.split(r"\s+or\s+", and_group)

        temp_results = []

        for group in or_groups:
            '''group = re.sub(
                r'between\s+([\d\.kKmM]+)\s+and\s+([\d\.kKmM]+)',
                r'between_\1_\2', 
                group
            )'''
            # Apply filter per OR-group
            filtered = apply_filters(df, group, filtered_registry, filters_applied)
            temp_results.append(filtered)

        # OR → union
        or_result = pd.concat(temp_results).drop_duplicates()

        # AND → intersection
        result_df = result_df[
            result_df.index.isin(or_result.index)
        ]

    if result_df.empty:
        print("[WARNING] Filtering resulted in empty dataset")

    # -----------------------------
    # UPDATE STATE
    # -----------------------------
    state["analysis_dataset"] = result_df

    state.setdefault("analysis_evidence", {})
    state["analysis_evidence"]["row_filtering"] = {
        "filters_applied": filters_applied,
        "remaining_rows": len(result_df)
    }

    print("\n=== ROW FILTERING COMPLETE ===")
    print("Filters applied:", filters_applied)
    print("Remaining rows:", len(result_df))

    return state

