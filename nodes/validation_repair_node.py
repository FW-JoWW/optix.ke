#nodes/validation_repair_node.py
from state.state import AnalystState
from utils.value_resolver import resolve_value

NUMERIC_SYNONYMS = {
    "under": "<",
    "below": "<",
    "less than": "<",
    "over": ">",
    "above": ">",
    "more than": ">",
    "not equals": "!=",
    "!=": "!=",
    "<=": "<=",
    ">=": ">="
}

def validation_repair_node(state: AnalystState) -> AnalystState:
    """
    Advanced validation & repair:
    - Maps numeric synonyms
    - Detects numeric & categorical contradictions
    - Suggests soft repairs
    - Logs all issues for analyst trace
    """
    filters = state.get("intent", {}).get("filters", [])
    dataset_profile = state.get("dataset_profile", {})
    numeric_cols = dataset_profile.get("numeric_columns", [])
    categorical_cols = dataset_profile.get("categorical_columns", [])

    df = state.get("dataframe")
    
    repaired_filters = []
    issues = []

    # ------------------------
    # 1. MAP NUMERIC SYNONYMS
    # ------------------------
    for f in filters:
        op = f.get("operator", "")
        if op in NUMERIC_SYNONYMS:
            f["operator"] = NUMERIC_SYNONYMS[op]
            issues.append(f"Mapped synonym '{op}' → '{f['operator']}'")
        repaired_filters.append(f)

    # ------------------------
    # 2. FIX WRONG COLUMNS
    # ------------------------
    for f in repaired_filters:
        col = f["column"]
        if col not in numeric_cols + categorical_cols and df is not None:
            matched = False
            # Try matching categorical first
            for cat_col in categorical_cols:
                sample_value = df[cat_col].astype(str).str.strip().str.lower().unique()
                if str(col).strip().lower() in sample_value:
                    f["column"] = cat_col
                    issues.append(f"Re-mapped value '{col}' → column '{cat_col}'")
                    matched = True
                    break
            if not matched:
                issues.append(f"Unresolved column '{col}' (ignored)")

    # ------------------------
    # 2.5 RESOLVE CATEGORICAL VALUES (FUZZY MATCH)
    # ------------------------
    for f in repaired_filters:
        col, val = f["column"], f["value"]

        if col in categorical_cols and df is not None:
            possible_values = df[col].dropna().unique()
            resolved = resolve_value(val, possible_values)

            if resolved != val:
                issues.append(f"Resolved '{val}' → '{resolved}' in column '{col}'")

            f["value"] = resolved

    # ------------------------
    # 3. DETECT NUMERIC CONTRADICTIONS
    # ------------------------
    numeric_constraints = {}
    for f in repaired_filters:
        col, op, val = f["column"], f["operator"], f["value"]
        if col in numeric_cols:
            numeric_constraints.setdefault(col, {"min": None, "max": None})
            if op == ">":
                numeric_constraints[col]["min"] = max(val, numeric_constraints[col]["min"] or val)
            elif op == "<":
                numeric_constraints[col]["max"] = min(val, numeric_constraints[col]["max"] or val)
            elif op == ">=":
                numeric_constraints[col]["min"] = max(val, numeric_constraints[col]["min"] or val)
            elif op == "<=":
                numeric_constraints[col]["max"] = min(val, numeric_constraints[col]["max"] or val)
    
    # Check numeric contradictions
    for col, bounds in numeric_constraints.items():
        if bounds["min"] is not None and bounds["max"] is not None:
            if bounds["min"] > bounds["max"]:
                issues.append(f"Contradiction in '{col}': min {bounds['min']} > max {bounds['max']}")
                # Soft repair: swap or suggest nearest valid range
                state["intent"]["filters"] = []
                state["halt_pipeline"] = True
                state["intent"]["ast"] = None
                state["error"] = "Contradictory numeric filters"

                print("\n[VALIDATION ISSUES V2]:", issues)
                return state

    # ------------------------
    # 4. DETECT CATEGORICAL CONTRADICTIONS
    # ------------------------
    categorical_values = {}
    
    '''for f in repaired_filters:
        col = f["column"]
        if col not in numeric_cols + categorical_cols:
            for cat_col in categorical_cols:
                if str(col).lower() in cat_col.lower():
                    f["column"] = cat_col
                    issues.append(f"Re-mapped column '{col}' → '{cat_col}'")
                    break'''
    for f in repaired_filters:
        col, val = f["column"], f["value"]
        if col in categorical_cols and f.get("operator") in ["equals", "="]:
            #if col not in categorical_values:
            categorical_values.setdefault(col, set()).add(val)
            
    
    for col, vals in categorical_values.items():
        if len(vals) > 1:
            issues.append(f"Multiple categorical values in '{col}'→ converted to OR")
            # Soft repair: convert multiple equals into OR
            repaired_filters = [
                f for f in repaired_filters
                if f["column"] != col
            ]            
            # Add OR filter
            repaired_filters.append({
                "type": "logic",
                "operator": "or",
                "conditions": [{"type": "condition", "column": col, "operator": "equals", "value": v} for v in vals]
            })

    # ----------------------------
    # REBUILD AST
    # ----------------------------
    def rebuild_ast(filters):
        logic_nodes = []
        conditions = []
        for f in filters:
            if f.get("type") == "logic":
                logic_nodes.append(f) # preserve or blocks
            else:
                conditions.append({
                        "type": "condition",
                        "column": f["column"],
                        "operator": f["operator"],
                        "value": f["value"]
                    })
        #If both exists add them proprely
        nodes = []

        # preserve logic nodes
        nodes.extend(logic_nodes)
        
        # add standalone conditions
        nodes.extend(conditions)

        if not nodes:
            return None
        
        if len(nodes) == 1:
                return nodes[0]
                #state["intent"]["ast"] = conditions[0]
        return {
            "type": "logic",
            "operator": "and",
            "conditions": nodes
        }
    
    # APPLY
    state["intent"]["filters"] = repaired_filters
    state["intent"]["ast"] = rebuild_ast(repaired_filters)
            

    # ------------------------
    # 6. SAVE REPAIRED FILTERS
    # ------------------------
    state["intent"]["filters"] = repaired_filters
    state["validation_issues"] = issues

    print("\n[VALIDATION ISSUES V2]:", issues)

    return state

'''from state.state import AnalystState


def validation_repair_node(state: AnalystState) -> AnalystState:
    """
    Validates and repairs filters before execution.
    - Fixes obvious mistakes
    - Detects contradictions
    - Prevents empty dataset crashes
    """

    filters = state.get("intent", {}).get("filters", [])
    dataset_profile = state.get("dataset_profile", {})
    numeric_cols = dataset_profile.get("numeric_columns", [])
    categorical_cols = dataset_profile.get("categorical_columns", [])

    repaired_filters = []
    issues = []

    # ------------------------
    # 1. FIX WRONG COLUMNS (e.g. "audi" → brand)
    # ------------------------
    for f in filters:
        col = f["column"]
        # If column not valid, try to map to categorical
        if col not in numeric_cols + categorical_cols:
            # Try mapping value into categorical column
            for cat_col in categorical_cols:
                f["column"] = cat_col
                repaired_filters.append(f)
                issues.append(f"Re-mapped column '{col}' → '{cat_col}'")
                break
        else:
            repaired_filters.append(f)

    # ------------------------
    # 2. DETECT CONTRADICTIONS
    # ------------------------
    numeric_constraints = {}

    for f in repaired_filters:
        if f["column"] in numeric_cols:
            col = f["column"]
            op = f["operator"]
            val = f["value"]

            if col not in numeric_constraints:
                numeric_constraints[col] = {"min": None, "max": None}

            if op == ">":
                numeric_constraints[col]["min"] = max(
                    val, numeric_constraints[col]["min"] or val
                )
            elif op == "<":
                numeric_constraints[col]["max"] = min(
                    val, numeric_constraints[col]["max"] or val
                )

    # Check contradictions
    for col, bounds in numeric_constraints.items():
        if bounds["min"] and bounds["max"]:
            if bounds["min"] > bounds["max"]:
                issues.append(
                    f"Contradiction detected in '{col}': {bounds['min']} > {bounds['max']}"
                )
                state["error"] = "Contradictory filters detected"
                state["intent"]["filters"] = []
                return state

    # ------------------------
    # 3. SAVE REPAIRED FILTERS
    # ------------------------
    state["intent"]["filters"] = repaired_filters
    state["validation_issues"] = issues

    print("\n[VALIDATION ISSUES]:", issues)

    return state'''