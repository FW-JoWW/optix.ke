#nodes/validation_repair_node.py
from state.state import AnalystState
from utils.value_resolver import resolve_value
import numpy as np

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

def validate_ast(node, df, numeric_cols, categorical_cols, issues):
    if not node:
        return None

    # CONDITION NODE
    if node["type"] == "condition":
        col = node.get("column")
        val = node.get("value")
        op = node.get("operator")
        confidence = node.get("confidence", 0.7)

        # Validate operator
        VALID_OPERATORS = {"equals", "!=", ">", "<", ">=", "<=", "between", "contains"}
        if op not in VALID_OPERATORS:
            issues.append(f"Invalid operator '{op}' in column '{col}'")
            return None

        # Invalid column
        if col not in numeric_cols + categorical_cols:
            if confidence < 0.85:
                issues.append(f"Drop low-confidence invalid column '{col}'")
                return None
            else:
                issues.append(f"Kept high-confidence unknown column '{col}' (soft accept)")

        # Type validation
        '''if col in numeric_cols and not isinstance(val, (int, float, list)):
            issues.append(f"Invalid numeric value for '{col}'")
            return None'''

        if col in categorical_cols and not isinstance(val, str):
            issues.append(f"Invalid categorical value for '{col}'")
            return None

        return node

    # LOGIC NODE
    elif node["type"] == "logic":
        cleaned_children = []

        for child in node.get("conditions", []):
            validated_child = validate_ast(child, df, numeric_cols, categorical_cols, issues)
            if validated_child:
                cleaned_children.append(validated_child)

        if not cleaned_children:
            return None

        if len(cleaned_children) == 1:
            return cleaned_children[0]

        return {
            "type": "logic",
            "operator": node["operator"],
            "conditions": cleaned_children
        }

    return None

def validation_repair_node(state: AnalystState) -> AnalystState:
    """
    Advanced validation & repair:
    - Maps numeric synonyms
    - Detects numeric & categorical contradictions
    - Suggests soft repairs
    - Logs all issues for analyst trace
    """
    #filters = state.get("intent", {}).get("filters", [])
    ast = state.get("intent", {}).get("ast")
    dataset_profile = state.get("dataset_profile", {})
    numeric_cols = dataset_profile.get("numeric_columns", [])
    categorical_cols = dataset_profile.get("categorical_columns", [])

    df = state.get(state.get("active_dataset", "dataframe"))
    
    #repaired_filters = []
    issues = []

    def get_column_stats(df, col):
        series = df[col].dropna()
        if len(series) == 0:
            return None

        if not np.issubdtype(series.dtype, np.number):
            return None

        return {
            "min": float(series.min()),
            "max": float(series.max()),
            "q1": float(series.quantile(0.25)),
            "median": float(series.quantile(0.5)),
            "q3": float(series.quantile(0.75))
        }

    # ------------------------
    # AST NORMALIZATION PASS
    # ------------------------
    def normalize_ast(node):
        if not node:
            return None

        if node["type"] == "condition":
            col = node.get("column")
            op = node.get("operator")
            val = node.get("value")

            new_node = {
                "type": "condition",
                "column": col,
                "operator": op,
                "value": val,
                "confidence": 0.7
            }
            # ------------------------
            # SEMANTIC VALUE RESOLUTION
            # ------------------------
            if df is not None and col in numeric_cols and col in df.columns:

                stats = get_column_stats(df, col)

                if stats and isinstance(val, str):

                    v = val.lower()

                    # LOW RANGE (cheap, low, affordable)
                    if v in ["cheap", "affordable", "low", "budget"]:
                        node["operator"] = "<="
                        node["value"] = stats["q1"]
                        node["confidence"] = 0.85
                        issues.append(f"Semantic: '{val}' -> Q1 threshold for '{col}'")

                    # HIGH RANGE (expensive, premium)
                    elif v in ["expensive", "premium", "high", "luxury"]:
                        node["operator"] = ">="
                        node["value"] = stats["q3"]
                        node["confidence"] = 0.85
                        issues.append(f"Semantic: '{val}' -> Q3 threshold for '{col}'")

                    # MID RANGE
                    elif v in ["average", "mid", "normal"]:
                        node["operator"] = "between"
                        node["value"] = [stats["q1"], stats["q3"]]
                        node["confidence"] = 0.8
                        issues.append(f"Semantic: '{val}' -> IQR range for '{col}'")
                        
            # Map numeric synonyms
            if op in NUMERIC_SYNONYMS:
                # High confidence if explicitly stated numeric constraint
                if op in [">", "<", ">=", "<=", "between"]:
                    new_node["confidence"] = 0.9
                new_node["operator"] = NUMERIC_SYNONYMS[op]
                issues.append(f"Mapped synonym '{op}' -> '{node['operator']}'")

            # Resolve categorical values
            if col in categorical_cols and df is not None:
                possible_values = df[col].dropna().unique()
                resolved = resolve_value(val, possible_values)
                new_node["confidence"] = 0.8

                if resolved != val:
                    issues.append(f"Resolved '{val}' -> '{resolved}' in '{col}'")

                new_node["value"] = resolved

            return new_node

        elif node["type"] == "logic":
            return {
                "type": "logic",
                "operator": node.get("operator"),
                "conditions": [
                    normalize_ast(child)
                    for child in node.get("conditions", [])
                    if child
                ]
            }
        
            '''node["conditions"] = [
                normalize_ast(child)
                for child in node.get("conditions", [])
                if child
            ]
            return node

        return node'''
    
    # ------------------------
    # APPLY AST PIPELINE
    # ------------------------

    # 1. Normalize AST
    normalized_ast = normalize_ast(ast)
    if normalized_ast is None:
        state["intent"]["ast"] = None
        state["intent"]["filters"] = []
        state["validation_issues"] = issues
        return state

    # 2. Validate AST
    validated_ast = validate_ast(
        normalized_ast,
        df,
        numeric_cols,
        categorical_cols,
        issues
    )

    def sort_by_confidence(node):
        if not node:
            return None

        if node["type"] == "logic":
            node["conditions"] = sorted(
                [sort_by_confidence(c) for c in node["conditions"] if c],
                key=lambda x: x.get("confidence", 0.7) if x else 0,
                reverse=True
            )
        return node

    validated_ast = sort_by_confidence(validated_ast)
    
    state["intent"]["ast"] = validated_ast

    # 3. Re-extract filters (READ-ONLY)
    from nodes.intent_parser_node import extract_filters
    if validate_ast:
        state["intent"]["filters"] = extract_filters(validated_ast)
    else:
        state["intent"]["filters"] = []

    state["validation_issues"] = issues

    print("\n[VALIDATION ISSUES V3]:", issues)

    return state

    """# ------------------------
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
    #state["intent"]["ast"] = rebuild_ast(repaired_filters)
          

    # ------------------------
    # 6. SAVE REPAIRED FILTERS
    # ------------------------
    state["intent"]["filters"] = repaired_filters
    state["validation_issues"] = issues

    print("\n[VALIDATION ISSUES V2]:", issues)

    return state"""

'''# nodes/validation_repair_node.py
from state.state import AnalystState


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
