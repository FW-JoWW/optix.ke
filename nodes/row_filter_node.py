# nodes/row_filter_node.py
import pandas as pd
from state.state import AnalystState

def is_number(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
    
# -------------------
#  AST EVALUATOR
# -------------------
def evaluate_condition(df, node):
    col = node["column"]
    op = node["operator"]
    val = node["value"]

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset")
        #return pd.Series([False] * len(df), index=df.index)
    
    series = df[col]

    if not op:
        raise ValueError(f"Missing oprator in node: {node}")

    # handle string columns case-insensitively
    if pd.api.types.is_string_dtype(series) and isinstance(val, str):
        col_values = series.astype(str).str.strip().str.lower()
        val = val.strip().lower()
        if op == "equals":
            exact_match = col_values == val
            partial_match = col_values.str.contains(val, na=False)
            return exact_match | partial_match
            #return col_values == val
        elif op == "contains":
            return col_values.str.contains(val, na=False)
    
    # -------------------------
    # NUMERIC SAFE OPERATIONS
    # -------------------------

    is_series_numeric = pd.api.types.is_numeric_dtype(series)
    is_val_numeric = is_number(val) or (
        isinstance(val, (list, tuple)) and all(is_number(v) for v in val)
    )

    # ❌ If operation requires numeric but data is not numeric → skip safely
    numeric_ops = {">", "<", ">=", "<=", "between"}

    if op in numeric_ops:
        if not is_series_numeric or not is_val_numeric:
            # return all False instead of crashing
            return pd.Series([False] * len(df), index=df.index)

        numeric_series = pd.to_numeric(series, errors="coerce")

        if op == "between":
            low, high = val
            low, high = float(low), float(high)
            if low > high:
                low, high = high, low
            return (numeric_series >= low) & (numeric_series <= high)

        val = float(val)

        if op == ">":
            return numeric_series > val
        elif op == "<":
            return numeric_series < val
        elif op == ">=":
            return numeric_series >= val
        elif op == "<=":
            return numeric_series <= val

    # -------------------------
    # NON-NUMERIC SAFE OPS
    # -------------------------

    if op == "equals":
        if is_series_numeric and is_val_numeric:
            return pd.to_numeric(series, errors="coerce") == float(val)
        else:
            return series.astype(str).str.lower() == str(val).lower()

    elif op in ["!=", "not equals"]:
        if is_series_numeric and is_val_numeric:
            return pd.to_numeric(series, errors="coerce") != float(val)
        else:
            return series.astype(str).str.lower() != str(val).lower()

    elif op == "contains":
        return series.astype(str).str.lower().str.contains(str(val).lower(), na=False)

    # fallback
    return pd.Series([False] * len(df), index=df.index)

    '''# numeric or compatible ops
    numeric_series = pd.to_numeric(series, errors="coerce")
    if op == "between":
        low, high = val
        if low > high:
            low, high = high, low
        return(numeric_series >= low) & (numeric_series <= high)
    elif op == ">":
        return numeric_series > val
    elif op == "<":
        return numeric_series < val
    elif op == "equals":
         return numeric_series == val
    elif op == "<=":
        return numeric_series <= val
    elif op == ">=":
        return numeric_series >= val
    elif op == "!=":
        return numeric_series != val if pd.api.types.is_numeric_dtype(series) else series.astype(str).str.lower() != str(val).lower()
    elif op == "not equals":
        return numeric_series != val if pd.api.types.is_numeric_dtype(series) else series.astype(str).str.lower() != str(val).lower()
    #fallback: mark all Fales if numeric conversion fails
    #return pd.Series([False] * len(df), index=df.index)
    raise ValueError(f"Unhandled condition: {node}")'''
    
# --------------------
# AST/LOGIC EVALUATOR
# --------------------
def evaluate_ast(df, node):
    if node is None:
        return pd.Series([True] * len(df), index=df.index)

    # ---- BASE CONDITION ----
    if node["type"] == "condition":
        return evaluate_condition(df, node)
    
    if node["type"] == "logic":
        conditions = node.get("conditions", [])
        if not conditions:
            return pd.Series([True] * len(df), index=df.index)
        
        masks = [evaluate_ast(df, c) for c in conditions]

        if node["operator"] == "and":
            result = masks[0]
            for m in masks[1:]:
                result = result & m
            return result

        elif node["operator"] == "or":
            result = masks[0]
            for m in masks[1:]:
                result = result | m
            return result
        
    return pd.Series([True] * len(df), index=df.index)


def is_filter_only(intent):
    return (
        intent.get("aggregation") is None and
        intent.get("group_by") is None
    )

# ------------
# MAIN NODE
# ------------

def row_filter_node(state: AnalystState) -> AnalystState:
    """Applies filters strictly from parsed intent (no guessing)."""

    if "cleaned_data" in state and state["cleaned_data"] is not None:
        state["active_dataset"] = "cleaned_data"
    else:
        state["active_dataset"] = "dataframe"
    
    df = state.get(state.get("active_dataset"))
        
    
    if df is None:
        raise ValueError("Dataset not found in state")

    df = df.copy()
    state.setdefault("analysis_evidence", {})

    intent = state.get("intent", {})
    ast = intent.get("ast")
    filters = intent.get("filters", [])
    
    filters_applied = []

    '''if not ast and filters:
        # fallback → build simple AND AST
        ast = {
            "type": "logic",
            "operator": "and",
            "conditions": [
                {
                    "type": "condition",
                    "column": f["column"],
                    "operator": f["operator"],
                    "value": f["value"]
                }
                for f in filters if f.get("type") != "logic"
            ]
        }'''

    if not ast:
        print("[INFO] No AST found - skipping filtering")
        state["analysis_dataset"] = df
        return state
    
    # --------------------
    # APPLY AST FILTERS
    # --------------------
    
    if not ast:
        print("[INFO] No AST found - skipping filtering")
        state["analysis_dataset"] = df
        return state
    
    # -------------------------
    # APPLY AST FILTERS
    # -------------------------
    mask = evaluate_ast(df, ast)
    print("\n[AST FILTER APPLIED]")
    if mask is None:
        print("[WARNING] AST evaluation failed")
        state["analysis_dataset"] = df
        return state
    
    mask = mask.fillna(False)
    df = df[mask]
    
    # collect applied filters for evidence
    for f in filters:
        if f.get("type") == "logic":
            filters_applied.append(f"{f['operator']} logic group")
        else:
            filters_applied.append(f"{f['column']} {f['operator']} {f['value']}")

    if df.empty:
        print("[WARNING] No rows matched filters.")
        empty_df = df.iloc[0:0].copy()
        state["analysis_dataset"] = empty_df
        #state.setdefault("analysis_evidence", {})
        state["analysis_evidence"]["row_filtering"] = {
            "filters_applied": filters_applied,
            "remaining_rows": 0,
            "error": "No rows matched the filters",
            "failed_failters": filters_applied
        }
        # optional: stop pipeline
        state["stop_execution"] = True

        return state

    # ---- update state ----
    state["analysis_dataset"] = df
   
    # -----------------------------
    # GROUPING + AGGREGATION
    # -----------------------------
    group_col = intent.get("group_by")
    aggregation = intent.get("aggregation")
    agg_column = intent.get("aggregate_column")
    
    grouped = None

    if group_col and group_col in df.columns:
        
        # ---- COUNT ----
        if not aggregation:
            grouped = (
                df.groupby(group_col)
                .size()
                .sort_values(ascending=False)
                .reset_index(name="count")
            )
    
            state["analysis_evidence"]["grouped_summary"] = grouped.to_dict("records")
        
            print("\n=== GROUPED SUMMARY ===")
            print(grouped.head(10))
        
        # -------- AGGREGATION --------
        elif agg_column and agg_column in df.columns:
            agg_map = {
                "mean": "mean",
                "sum": "sum",
                "min": "min",
                "max": "max"
            }
            if aggregation not in agg_map:
                print(f"[WARNING] Unsupported aggregation: {aggregation}")
            else:
                grouped = (
                    df.groupby(group_col)[agg_column]
                    .agg(agg_map[aggregation])
                    .reset_index()
                    .sort_values(by=agg_column, ascending=False)
                )

                state["analysis_evidence"]["grouped_summary"] = grouped.to_dict("records")
                state["analysis_evidence"]["aggregation"] = {
                    "type": aggregation,
                    "column": agg_column
                }
            
                print(f"\n=== GROUPED SUMMARY ({aggregation.upper()}) ===")
                print(grouped.head(10))

    print("\n[DEBUG] Unique values after filter:")
    if "make" in df.columns:
        print(df["make"].unique()[:10])


    # ------------------
    # UPDATE STATE
    # ------------------

    state["analysis_evidence"]["row_filtering"] = {
        "filters_applied": filters_applied,
        "remaining_rows": len(df)
    }

    '''# --------------------------------------
    # SHORT-CIRCUIT FOR AGGREGATION QUERIES
    # --------------------------------------
    if intent.get("aggregation") and group_col and grouped is not None:
        print("\n[INFO] Aggregation query detected - skipping statistical tools")
        state["analysis_evidence"]["final_answer"] = grouped.to_dict("records")
        state["skip_statistical_analysis"] = True
        return state'''

    print("\n=== ROW FILTERING COMPLETE ===")
    print("Remaining rows:", len(df))

    return state

