# nodes/intent_perser_node.py
import re
from state.state import AnalystState
from nodes.llm_reasoning_node import llm_reasoning_node
from utils.semantic_mapper import map_semantic_filters

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


def map_numeric_synonyms(query: str):
    for word, op in NUMERIC_SYNONYMS.items():
        query = re.sub(rf"\b{word}\b", op, query, flags=re.IGNORECASE)
    return query

# ------------------
# HELPER
# ------------------
def normalize(text: str):
    return text.lower().replace("_", "").replace(" ", "")

def parse_number(value: str):
    value = value.lower().strip()
    if "k" in value:
        return float(value.replace("k", "")) * 1000
    if "m" in value:
        return float(value.replace("m", "")) * 1_000_000
    return float(value)

# ------------------------
# CONDITION BUILDERS
# ------------------------

# ---- NUMERIC PARSING ----

def build_numeric_conditions(query: str, df):
    """
    Build numeric conditions from query after mapping numeric synonyms
    Handles >, <, between, and ignores negation words like 'not'
    """
    conditions = []
    # Make sure numeric synonyms are ready mapped (under -> <, over -> >, etc.)
    query_mapped = map_numeric_synonyms(query)

    patterns = [
        (r'([\w_]+)\s*>\s*([\d\.kKmM]+)', '>'),
        (r'([\w_]+)\s*<\s*([\d\.kKmM]+)', '<'),
        (r'([\w_]+)\s+between\s+([\d\.kKmM]+)\s+&&\s+([\d\.kKmM]+)', 'between')
    ]
    # VALIDATE COLUMN BEFORE ADDING
    valid_columns = []
    if df is not None:
        valid_columns = [c.lower() for c in df.columns]

    for pattern, op in patterns:
        matches = re.findall(pattern, query_mapped)
        for match in matches:
            if op == "between":
                col, low, high = match
                if df is not None and col.lower() not in valid_columns:
                    continue  # SKIP invalid column               
                conditions.append({
                    "type": "condition",
                    "column": col,
                    "operator": "between",
                    "value": (parse_number(low), parse_number(high))
                })
            else:
                col, val = match
                if df is not None and col.lower() not in valid_columns:
                    continue  # SKIP invalid column
                conditions.append({
                    "type": "condition",
                    "column": col,
                    "operator": op,
                    "value": parse_number(val)
                })        
            
    return conditions

# ---- CATEGORICAL PARSING ----

def build_categorical_conditions(query: str, df):
    conditions = []
    if df is None:
        return conditions

    categorical_columns = df.select_dtypes(include=["object", "string"]).columns
    query_lower = query.lower()

    for col in categorical_columns:
        values = df[col].dropna().unique()
        tokens = query_lower.split()
        matched_vals = [v for v in values if str(v).lower() in tokens]
        if matched_vals:
            if len(matched_vals) == 1:
                conditions.append({
                    "type": "condition",
                    "column": col,
                    "operator": "equals",
                    "value": matched_vals[0]
                })
            else:
                # Multiple matches -> ORlogic
                conditions.append({
                    "type": "logic",
                    "operator": "or",
                    "conditions": [
                        {
                            "type": "condition",
                            "column": col,
                            "operator": "equals",
                            "value": v
                        }
                        for v in matched_vals
                    ]
                })

    return conditions

# ------------------
# NAGGATION SUPPORT
# ------------------

def detect_negation(query: str) -> bool:
    """
    Detects whether a query contains negation.
    Only triggers on explicit negation words and ignores numeric comparative words.
    """
    query = query.lower()
    # Explicit negation words
    NAGATION_WORDS = [
        r'\bnot\b',
        r'\bno\b',
        r'\bnever\b',
        r'\bnon\b',
        r'\bwithout\b',
        r'\bexclude\b',
        r'\bexcluding\b',
        r'\bexcept\b'
    ]

    # If any negation word is found, return True
    for word in NAGATION_WORDS:
        if re.search(word, query):
            return True

    return False

def apply_negation_to_condition(cond):
    if cond["operator"] == "equals":
        cond["operator"] = "!="
    elif cond["operator"] == ">":
        cond["operator"] = "<="
    elif cond["operator"] == "<":
        cond["operator"] = ">="
    return cond

def apply_negation(node):
    if node["type"] == "condition":
        return apply_negation_to_condition(node)
    elif node["type"] == "logic":
        return {
            "type": "logic",
            "operator": node["operator"],
            "conditions": [apply_negation(c) for c in node["conditions"]]
        }
    return node

# ----------------------------
# AST BUILDING
# ----------------------------
def build_ast(query: str, df):
    query = query.strip()
    query = map_numeric_synonyms(query)

    # PROTECT BETWEEN
    query = re.sub(
        r'between\s+([\d\.kKmM]+)\s+(and|&&)\s+([\d\.kKmM]+)',
        lambda m: f"between {m.group(1)} && {m.group(3)}",
        query,
        flags=re.IGNORECASE
    )

    # ---- HANDLE AND / WITH ----  
    and_parts = re.split(r'\s+(and|with)\s+', query)
    
    # remove the actual words "and"/"with"
    and_parts = [p for p in and_parts if p not in ("and", "with")]
    
    if len(and_parts) > 1:
        nodes = [build_ast(p, df) for p in and_parts if p.strip()]
        nodes = [n for n in nodes if n]

        if len(nodes) == 1:
            return nodes[0]
    
        ast_node = {
            "type": "logic",
            "operator": "and",
            "conditions": nodes
        }

        return ast_node

    # ---- HANDLES OR ----
    or_parts = re.split(r'\s+or\s+', query)
    
    if len(or_parts) > 1:
        nodes = [build_ast(p, df) for p in or_parts if p.strip()]
        nodes = [n for n in nodes if n]

        if not nodes:
            return None
    
        if len(nodes) == 1:
            return nodes[0]
    
        ast_node = {
            "type": "logic",
            "operator": "or",
            "conditions": nodes
        }
        
        return ast_node
           
    is_negation = detect_negation(query)

    # ---- BASE CONDITIONS ----     
    conds = []
    conds.extend(build_numeric_conditions(query, df))
    conds.extend(build_categorical_conditions(query, df))
    
    if not conds:
        return None
    
    ast_node = None
    if is_negation:
        #conds = [apply_negation_to_condition(c) for c in conds if c["type"] == "condition"]
        for c in conds:
            if c.get("type") == "condition":
                apply_negation_to_condition(c)

    if len(conds) == 1:
        ast_node = conds[0]
    
    else:
        ast_node = {
            "type": "logic",
            "operator": "and",
            "conditions": conds
        }
    
    return ast_node
    

# -----------------------
# FLAT FILTER EXTRACTOR
# -----------------------
def extract_filters(node):
    if node is None:
        return None

    # Handle leaf nodes(individual conditions)
    if node["type"] == "condition":
        return {
            "type": "condition",
            "column": node.get("column"),
            "operator": node.get("operator"),
            "value": node.get("value")
        }
        
    # Handle logic nodes (AND/OR groups)
    elif node["type"] == "logic":
        return {
            "type": "logic",
            "operator": node.get("operator"),
            # This rebuilds the 'conditions' list by calling this function on each child
            "conditions": [extract_filters(child) for child in node.get("conditions", [])]
        }

    return node

def build_final_ast(processed_node):
    # If there's nothing there, return None
    if not processed_node:
        return None

    # If it's already a dictionary (the tree), just return it.
    if isinstance(processed_node, dict):
        return processed_node

    # If for some reason you still have a list, wrap it in an AND
    if isinstance(processed_node, list):
        if len(processed_node) == 1:
            return processed_node[0]
        return {
            "type": "logic",
            "operator": "and",
            "conditions": processed_node
        }

# ------------------------
# MAIN NODE
# ------------------------

def intent_parser_node(state: AnalystState) -> AnalystState:
    query = state.get("business_question", "").lower()
    
    # Select dataset
    if "cleaned_data" in state and state["cleaned_data"] is not None:
        state["active_dataset"] = "cleaned_data"
    else:
        state["active_dataset"] = "dataframe"
    df = state.get(state.get("active_dataset"))
    
    
    # --------------
    # AST BUILDING
    # --------------
    is_negation = detect_negation(query)
    ast = build_ast(query, df)
    #filters = extract_filters(ast) #if ast else {}
    #final_filters = build_final_ast(filters)
    
    # ------------------------
    # SEMANTIC ENRICHMENT
    # ------------------------
    if df is not None:
        numeric_cols = state.get("dataset_profile", {}).get("numeric_columns", [])
        semantic_filters = map_semantic_filters(query, df, numeric_cols)
        #filters.extend(semantic_filters)
        if semantic_filters:
            semantic_ast = {
                "type": "logic",
                "operator": "and",
                "condition": semantic_filters
            }
            if ast:
                ast = {
                    "type": "logic",
                    "operator": "and",
                    "conditions": [ast, semantic_filters]
                }
            else:
                ast = semantic_filters
        if is_negation and ast:
            ast = apply_negation(ast)
        '''if is_negation:
            for f in filters:
                if f.get("type") == "condition":
                    apply_negation_to_condition(f)'''

    # If numeric intent exists but no valid numeric filters → defer to LLM
    if df is not None:
        numeric_cols = state.get("dataset_profile", {}).get("numeric_columns", [])
        
        #has_numeric_hint = any(sym in query for sym in [">", "<", "between"])
        #has_numeric_filter = any(f["column"] in numeric_cols for f in filters)
    
        '''if has_numeric_hint and not has_numeric_filter:
            filters = []  # wipe bad parse → let LLM fix
            ast = None'''
    
    if ast is None: #and not filters:
        pass
        #raise ValueError("Failed to parse query into valid intent.")

    # -----------------------
    # INITIAL INTENT
    # -----------------------
    intent = {
        "type": "ast",
        "ast": ast,
        "filters": ast,#filters,
        "group_by": None,
        "aggregation": None,
        "aggregate_column": None
    }
    # ------------------------
    # AGGREGATION 
    # ------------------------
    
    aggregation_keywords = {
        "mean": ["average", "mean", "avg"],
        "sum": ["total", "sum"],
        "max": ["max", "maximum", "highest"],
        "min": ["min", "minimum", "lowest"]
    }
    
    if df is not None:
        numeric_columns = state.get("dataset_profile", {}).get("numeric_columns", [])
        for agg_type, keywords in aggregation_keywords.items():
            if any(word in query for word in keywords):
                intent["aggregation"] = agg_type
                for col in numeric_columns:
                    if normalize(col) in normalize(query):
                        intent["aggregate_column"] = col
                        break
    
                break
    
    # ------------------------
    # GROUP BY )
    # ------------------------
    
    if df is not None:
        categorical_columns = state.get("dataset_profile", {}).get("categorical_columns", [])
        for col in categorical_columns:
            if normalize(col) in normalize(query):
                intent["group_by"] = col
                break
    
    state["intent"] = intent
    
    # -----------------
    # LLM REASONING
    # -----------------
    original_ast = state["intent"].get("ast")
    state = llm_reasoning_node(state)

    if state["intent"].get("ast") is None:
        state["intent"]["ast"] = original_ast
    # ---- REBUILD AST AFTER LLM ----
    state["intent"]["ast"] = build_final_ast(state["intent"]["filters"])

    # ------------------------
    # TYPE
    # ------------------------
    
    if intent["group_by"] and not intent["aggregation"]:
        intent["type"] = "group"
    elif intent["aggregation"]:
        intent["type"] = "aggregation"
    elif ast:
        intent["type"] = "filter"
    else:
        intent["type"] = "exploration"
        
    '''if state["intent"]["filters"]:
        state["intent"]["type"] = "filter"
    elif state["intent"]["aggregation"]:
        state["intent"]["type"] = "aggregation"'''

    

    print("\n=== INTENT PARSER COMPLETE ===")
    print("Filters after llm reasoning:", state["intent"]["filters"])

    return state

