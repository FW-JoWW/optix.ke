# nodes/intent_perser_node.py
import re
from state.state import AnalystState
from nodes.llm_reasoning_node import llm_reasoning_node
from utils.semantic_mapper import map_semantic_filters

# --- NUMERIC SYNONYMS ---
NUMERIC_SYNONYMS = {
    "under": "<",
    "below": "<",
    "less than": "<",
    "over": ">",
    "above": ">",
    "more than": ">",
    "not equals": "!=",
    "newer than": ">",
    "older than": "<",
    "after": ">",
    "before": "<",
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

def classify_analytic_intent(query: str):
    """
    Classifies high-level analytic intent from user query.
    """

    query = query.lower()

    intent_map = {
        "comparison": ["compare", "vs", "versus", "difference", "against"],
        "temporal": ["trend", "over time", "per day", "per month", "growth", "decline"],
        "composition": ["breakdown", "distribution", "percentage", "share", "portion"],
        "relationship": ["correlation", "relationship", "impact", "effect", "influence"],
        "extremes": ["top", "bottom", "highest", "lowest", "max", "min"],
        "profiling": ["average", "mean", "median", "summary", "stats"],
        "investigative": ["why", "drill down", "details", "explain"],
        "predictive": ["forecast", "predict", "what if", "estimate"]
    }

    for intent, keywords in intent_map.items():
        if any(word in query for word in keywords):
            return intent

    return "unknown"

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
        (r'([\w_]+)\s+(newer than|after)\s+([\d]{4})', '>'),
        (r'([\w_]+)\s+(older than|before)\s+([\d]{4})', '<'),
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

def convert_reasoning_to_ast(reasoning, df):
    if not reasoning:
        return None

    constraints = reasoning.get("constraints", [])
    logic_op = reasoning.get("logic", "and")

    nodes = []

    for c in constraints:
        node = {
            "type": "condition",
            "column": c.get("field"),
            "operator": c.get("operator"),
            "value": c.get("value"),
            "confidence": 0.8 if c.get("confidence") == "high" else 0.6
        }
        nodes.append(node)

    if len(nodes) == 1:
        return nodes[0]

    return {
        "type": "logic",
        "operator": logic_op if logic_op in ["and", "or"] else "and",
        "conditions": nodes
    }

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
    """
    Flattens AST into a list of filters while preserving OR blocks.
    """
    if node is None:
        return []

    # Leaf condition
    if node["type"] == "condition":
        return [node]

    filters = []

    if node["type"] == "logic":
        if node["operator"] == "or":
            # PRESERVE OR BLOCK as a single unit
            filters.append(node)
        else:
            # Flatten AND
            for child in node.get("conditions", []):
                filters.extend(extract_filters(child))

    return filters

'''def extract_filters(node):
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

    return node'''

def build_final_ast(filters):
    """
    Rebuilds AST from flat filters while preserving OR logic blocks.
    """
    if not filters:
        return None

    logic_nodes = []
    conditions = []

    for f in filters:
        if f.get("type") == "logic":
            logic_nodes.append(f)  # keep OR blocks intact
        else:
            conditions.append({
                "type": "condition",
                "column": f["column"],
                "operator": f["operator"],
                "value": f["value"]
            })

    nodes = []
    nodes.extend(logic_nodes)
    nodes.extend(conditions)

    if not nodes:
        return None

    if len(nodes) == 1:
        return nodes[0]

    return {
        "type": "logic",
        "operator": "and",
        "conditions": nodes
    }
'''def build_final_ast(processed_node):
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
        }'''

# ------------------------
# INTENT DETECTION 
# ------------------------
def detect_intents(query: str):
    query = query.lower()

    intents = []

    # --- COMPARISON ---
    if any(word in query for word in ["compare", "vs", "versus", "difference", "between"]):
        intents.append({"type": "comparison", "confidence": 0.8})

    # --- TEMPORAL ---
    if any(word in query for word in ["over time", "trend", "monthly", "yearly", "daily", "growth"]):
        intents.append({"type": "temporal", "confidence": 0.75})

    # --- EXTREMES ---
    if any(word in query for word in ["top", "highest", "lowest", "max", "min"]):
        intents.append({"type": "extremes", "confidence": 0.8})

    # --- COMPOSITION ---
    if any(word in query for word in ["percentage", "ratio", "breakdown", "share"]):
        intents.append({"type": "composition", "confidence": 0.75})

    # --- PROFILING ---
    if any(word in query for word in ["average", "mean", "distribution", "median"]):
        intents.append({"type": "profiling", "confidence": 0.7})

    return intents
# ------------------------
# INTENT → OPERATIONS (STEP 3)
# ------------------------
def map_intents_to_operations(intents):
    ops = []

    mapping = {
        "comparison": ["groupby", "aggregate"],
        "temporal": ["time_series"],
        "extremes": ["sort", "limit"],
        "composition": ["ratio"],
        "profiling": ["describe"]
    }

    for intent in intents:
        intent_type = intent.get("type")
        if intent_type in mapping:
            ops.extend(mapping[intent_type])

    # remove duplicates while preserving order
    seen = set()
    final_ops = []
    for op in ops:
        if op not in seen:
            seen.add(op)
            final_ops.append(op)

    return final_ops

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

    reasoning = state.get("llm_reasoning")

    if reasoning:
        ast = convert_reasoning_to_ast(reasoning, df)
    else:
        ast = build_ast(query, df)
    
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
                "conditions": semantic_filters
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
    analytic_intent = classify_analytic_intent(query)
    filters = extract_filters(ast) 

    intent = {
        "type": "ast",
        "analytic_intent": analytic_intent,
        "ast": ast,
        "filters": filters,
        "group_by": None,
        "aggregation": None,
        "aggregate_column": None,

        "intents": [],
        "operations_hint": [],
        "output_mode": None,
        "confidence": None
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

        # If no aggregation keyword found, infer from analytic intent
        if not intent["aggregation"]:
            intent_type = intent.get("analytic_intent")
            if intent_type in ["comparison", "extremes", "profiling"]:
                # Default to mean for comparison/profiling
                intent["aggregation"] = "mean"
                if numeric_columns:
                    intent["aggregate_column"] = numeric_columns[0]
            elif intent_type == "temporal":
                # For trends, default to sum over time
                intent["aggregation"] = "sum"
                if numeric_columns:
                    intent["aggregate_column"] = numeric_columns[0]

    # ------------------------
    # GROUP BY )
    # ------------------------
    
    if df is not None:
        categorical_columns = state.get("dataset_profile", {}).get("categorical_columns", [])
        for col in categorical_columns:
            if normalize(col) in normalize(query):
                intent["group_by"] = col
                break

        # If no group_by detected, infer from analytic intent
        if not intent["group_by"]:
            if intent_type in ["comparison", "composition"]:
                if categorical_columns:
                    intent["group_by"] = categorical_columns[0]
    
    '''# --- ENSURE NEGATION APPLIED TO FILTERS ---
    if filters:
        for f in filters:
            if f.get("type") == "condition" and is_negation:
                apply_negation_to_condition(f)
    
    # --- MERGE SEMANTIC FILTERS INTO AST ---
    if df is not None and filters:
        if ast is None:
            ast = {
                "type": "logic",
                "operator": "and",
                "conditions": filters
            }
        else:
            # Combine existing AST with filters
            ast = {
                "type": "logic",
                "operator": "and",
                "conditions": [ast] + filters
            }'''
    
    state["intent"] = intent
    
    # ------------------------
    # APPLY INTENT DETECTION
    # ------------------------
    detected_intents = detect_intents(query)
    state["intent"]["intents"] = detected_intents

    # ------------------------
    # APPLY OPERATION MAPPING
    # ------------------------
    operations_hint = map_intents_to_operations(detected_intents)
    state["intent"]["operations_hint"] = operations_hint

    # -----------------
    # LLM REASONING
    # -----------------

    should_call_llm = False

    # Case 1: AST failed completely
    if ast is None:
        should_call_llm = True

    # Case 2: AST exists but is too weak (no conditions)
    elif ast and ast.get("type") == "logic" and not ast.get("conditions"):
        should_call_llm = True

    # Case 3: No filters extracted
    elif not filters:
        should_call_llm = True

    if should_call_llm:
        state = llm_reasoning_node(state)
    else:
        print("[INFO] Skipping LLM — symbolic parsing sufficient")

    
    # Re-extract filters ONLY (AST stays untouched)
    if state["intent"].get("ast"):
        state["intent"]["filters"] = extract_filters(state["intent"]["ast"])
    else:
        state["intent"]["filters"] = []
    
    '''original_ast = state["intent"].get("ast")
    state = llm_reasoning_node(state)

    if state["intent"].get("ast") is None:
        state["intent"]["ast"] = original_ast'''
    # ---- REBUILD AST AFTER LLM ----
    #state["intent"]["ast"] = build_final_ast(state["intent"]["filters"])

    # ------------------------
    # TYPE
    # ------------------------
    
    if intent["aggregation"]:
        intent["type"] = "aggregation"
    elif ast:
        intent["type"] = "filter"
    else:
        intent["type"] = "exploration"
    intent["analysis_type"] = intent.get("analytic_intent", "unknown")


    print("\n=== INTENT PARSER COMPLETE ===")
    print("Filters after llm reasoning:", state["intent"]["filters"])

    return state

