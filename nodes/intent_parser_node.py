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

def extract_mentioned_columns(query: str, columns: list[str]):
    normalized_query = normalize(query)
    mentioned = []
    for col in columns:
        if normalize(col) in normalized_query:
            mentioned.append(col)
    return mentioned

def get_numeric_columns(df):
    if df is None:
        return []
    return [c for c in df.select_dtypes(include="number").columns if c in df.columns]

def get_temporal_numeric_columns(df):
    numeric_columns = get_numeric_columns(df)
    temporal_keywords = {"year", "date", "time", "month", "day", "week", "quarter"}
    temporal_columns = [
        col for col in numeric_columns
        if any(keyword in col.lower() for keyword in temporal_keywords)
    ]

    if temporal_columns:
        return temporal_columns

    inferred = []
    for col in numeric_columns:
        series = df[col].dropna()
        if series.empty:
            continue
        if series.between(1900, 2100).mean() >= 0.8:
            inferred.append(col)
    return inferred

def classify_analytic_intent(query: str):
    """
    Classifies high-level analytic intent from user query.
    """

    query = query.lower()

    intent_map = {
        "comparison": ["compare", "vs", "versus", "difference", "against"],
        "temporal": ["trend", "over time", "per day", "per month", "growth", "decline"],
        "composition": ["breakdown", "distribution", "percentage", "share", "portion"],
        "relationship": ["correlation", "relationship", "impact", "effect", "influence", "affect"],
        "extremes": ["top", "bottom", "highest", "lowest", "max", "min"],
        "profiling": ["average", "mean", "median", "summary", "stats", "statistics"],
        "outliers": ["outlier", "outliers", "unusual", "anomaly", "anomalies"],
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

def infer_numeric_column(query: str, df, raw_value: str | None = None, temporal_only: bool = False):
    if df is None:
        return None

    numeric_columns = get_temporal_numeric_columns(df) if temporal_only else get_numeric_columns(df)
    if not numeric_columns:
        return None

    query_lower = query.lower()
    normalized_columns = {c.lower(): c for c in numeric_columns}

    for key, original in normalized_columns.items():
        if key in query_lower:
            return original

    if len(numeric_columns) == 1:
        return numeric_columns[0]

    if raw_value:
        value_lower = raw_value.lower()
        digits = re.sub(r"[^\d]", "", value_lower)
        if temporal_only and len(digits) == 4:
            year_val = int(digits)
            candidate_columns = []
            for col in numeric_columns:
                series = df[col].dropna()
                if series.empty:
                    continue
                within_range = series.between(year_val - 20, year_val + 20).mean()
                if within_range >= 0.5:
                    candidate_columns.append(col)
            if len(candidate_columns) == 1:
                return candidate_columns[0]

    return None

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
    query_mapped = query

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
                    inferred_col = infer_numeric_column(query, df, raw_value=high)
                    if not inferred_col:
                        continue
                    col = inferred_col
                conditions.append({
                    "type": "condition",
                    "column": col,
                    "operator": "between",
                    "value": (parse_number(low), parse_number(high))
                })
            else:
                if len(match) == 3:
                    col, _, val = match
                else:
                    col, val = match
                if df is not None and col.lower() not in valid_columns:
                    inferred_col = infer_numeric_column(query, df, raw_value=val)
                    if not inferred_col:
                        continue
                    col = inferred_col
                conditions.append({
                    "type": "condition",
                    "column": col,
                    "operator": op,
                    "value": parse_number(val)
                })        

    implicit_patterns = [
        (r'\b(?:under|below|less than)\s*([\d\.kKmM]+)\b', '<', False),
        (r'\b(?:over|above|more than)\s*([\d\.kKmM]+)\b', '>', False),
        (r'(^|\s)<\s*([\d\.kKmM]+)\b', '<', False),
        (r'(^|\s)>\s*([\d\.kKmM]+)\b', '>', False),
        (r'\b(?:newer than|after)\s*([\d]{4})\b', '>', True),
        (r'\b(?:older than|before)\s*([\d]{4})\b', '<', True),
        (r'\bbetween\s+([\d\.kKmM]+)\s+&&\s+([\d\.kKmM]+)\b', 'between', None)
    ]

    existing_keys = {
        (c["column"], c["operator"], str(c["value"]))
        for c in conditions
    }

    for pattern, op, temporal_only in implicit_patterns:
        matches = re.findall(pattern, query, flags=re.IGNORECASE)
        for match in matches:
            if op == "between":
                low, high = match
                col = infer_numeric_column(query, df, raw_value=high)
                if not col:
                    continue
                candidate = {
                    "type": "condition",
                    "column": col,
                    "operator": "between",
                    "value": (parse_number(low), parse_number(high))
                }
            else:
                if isinstance(match, tuple):
                    raw_value = match[-1]
                else:
                    raw_value = match
                col = infer_numeric_column(
                    query,
                    df,
                    raw_value=raw_value,
                    temporal_only=bool(temporal_only)
                )
                if not col:
                    continue
                candidate = {
                    "type": "condition",
                    "column": col,
                    "operator": op,
                    "value": parse_number(raw_value)
                }

            key = (candidate["column"], candidate["operator"], str(candidate["value"]))
            if key not in existing_keys:
                conditions.append(candidate)
                existing_keys.add(key)

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

def strip_negation_words(query: str) -> str:
    """
    Removes negation words from query AFTER detection,
    so they don't interfere with parsing.
    """
    NEGATION_WORDS = [
        r'\bnot\b',
        r'\bno\b',
        r'\bnever\b',
        r'\bwithout\b',
        r'\bexclude\b',
        r'\bexcluding\b',
        r'\bexcept\b'
    ]

    for pattern in NEGATION_WORDS:
        query = re.sub(pattern, '', query, flags=re.IGNORECASE)

    return query.strip()

def extract_negation_scopes(query: str):
    """
    Finds phrases where negation applies.
    Example:
        "not red" → ["red"]
        "exclude toyota" → ["toyota"]
    """
    patterns = [
        r'not\s+(\w+)',
        r'exclude\s+(\w+)',
        r'excluding\s+(\w+)',
        r'without\s+(\w+)'
    ]

    negated_terms = []

    for pattern in patterns:
        matches = re.findall(pattern, query, flags=re.IGNORECASE)
        negated_terms.extend(matches)

    return negated_terms

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
        conf_map = {
            "high": 0.9,
            "medium": 0.75,
            "low": 0.6
        }
        node = {
            "type": "condition",
            "column": c.get("field"),
            "operator": c.get("operator"),
            "value": c.get("value"),
            "confidence": conf_map.get(c.get("confidence"), 0.7)
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

        if not nodes:
            return None

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

    negated_terms = extract_negation_scopes(query)
    query = strip_negation_words(query)

    # ---- BASE CONDITIONS ----     
    conds = []
    conds.extend(build_numeric_conditions(query, df))
    conds.extend(build_categorical_conditions(query, df))

    for cond in conds:
        val = str(cond.get("value", "")).lower()
        col = str(cond.get("column", "")).lower()

        # If value OR column matches a negated term → flip
        if any(term in val or term in col for term in negated_terms):
            apply_negation_to_condition(cond)

    if not conds:
        return None
    
    ast_node = None
    '''if is_negation:
        #conds = [apply_negation_to_condition(c) for c in conds if c["type"] == "condition"]
        for c in conds:
            if c.get("type") == "condition":
                c = apply_negation_to_condition(c)'''

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

def merge_ast_nodes(base_ast, extra_ast):
    if base_ast and extra_ast:
        return {
            "type": "logic",
            "operator": "and",
            "conditions": [base_ast, extra_ast]
        }
    return extra_ast or base_ast

def estimate_filter_confidence(node):
    if node is None:
        return 0.0

    if node["type"] == "logic":
        child_scores = [estimate_filter_confidence(child) for child in node.get("conditions", [])]
        return min(child_scores) if child_scores else 0.0

    if node.get("confidence") is not None:
        return node["confidence"]

    if node.get("source") == "semantic":
        return 0.65

    if node.get("operator") in {">", "<", ">=", "<=", "between"}:
        return 0.9

    if node.get("operator") in {"equals", "!=", "contains"}:
        return 0.8

    return 0.7

def has_numeric_constraint(filters, numeric_columns):
    return any(
        f.get("type") == "condition" and f.get("column") in numeric_columns
        for f in filters
    )

def query_has_unresolved_numeric_phrase(query: str):
    comparator_patterns = [
        r"\b(?:under|below|less than|over|above|more than)\s*[\d\.kKmM]+\b",
        r"(^|\s)[<>]\s*[\d\.kKmM]+\b",
        r"\bbetween\s+[\d\.kKmM]+\s+(?:and|&&)\s+[\d\.kKmM]+\b"
    ]
    return any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in comparator_patterns)

def query_has_semantic_magnitude(query: str):
    words = ["cheap", "affordable", "budget", "expensive", "premium", "high", "low", "luxury"]
    return any(re.search(rf"\b{word}\b", query, flags=re.IGNORECASE) for word in words)

def should_call_llm_for_intent(state: AnalystState, ast, filters):
    if not state.get("enable_llm_reasoning", True):
        return False

    if ast is None or not filters:
        return True

    if state.get("intent", {}).get("low_confidence"):
        return True

    query = state.get("business_question", "").lower()
    numeric_columns = state.get("dataset_profile", {}).get("numeric_columns", [])
    ambiguous_terms = [
        "bimmer",
        "bimmers",
        "bmws",
        "cheap",
        "expensive",
        "affordable",
        "premium"
    ]

    if query_has_unresolved_numeric_phrase(query) and not has_numeric_constraint(filters, numeric_columns):
        return True

    if query_has_semantic_magnitude(query) and not filters:
        return True

    return any(term in query for term in ambiguous_terms)
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
    if any(word in query for word in ["compare", "vs", "versus", "difference", "between", "affect"]):
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
    if any(word in query for word in ["average", "mean", "distribution", "median", "summary", "statistics"]):
        intents.append({"type": "profiling", "confidence": 0.7})

    # --- OUTLIER DETECTION ---
    if any(word in query for word in ["outlier", "outliers", "unusual", "anomaly", "anomalies"]):
        intents.append({"type": "outliers", "confidence": 0.8})

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
    original_query = state.get("business_question", "")
    query = original_query.lower()
    
    # Select dataset
    if "cleaned_data" in state and state["cleaned_data"] is not None:
        state["active_dataset"] = "cleaned_data"
    else:
        state["active_dataset"] = "dataframe"
    df = state.get(state.get("active_dataset"))
    
    if "intent" not in state:
        state["intent"] = {}

    state["llm_reasoning"] = None

    analytic_intent = classify_analytic_intent(query)
    ast = build_ast(query, df)
    selected_columns = []
    mentioned_numeric = []
    mentioned_categorical = []

    if df is not None:
        all_columns = list(df.columns)
        selected_columns = extract_mentioned_columns(query, all_columns)
        numeric_columns = state.get("dataset_profile", {}).get("numeric_columns", [])
        categorical_columns = state.get("dataset_profile", {}).get("categorical_columns", [])
        mentioned_numeric = [col for col in selected_columns if col in numeric_columns]
        mentioned_categorical = [col for col in selected_columns if col in categorical_columns]

        effect_words = ["affect", "effect", "influence"]
        compare_words = ["compare", "difference", "better", "vs", "versus"]
        if (
            any(word in query for word in effect_words)
            and len(mentioned_numeric) == 1
            and len(mentioned_categorical) == 1
            and not any(word in query for word in compare_words)
        ):
            state["original_business_question"] = original_query
            query = f"{query} compare {mentioned_numeric[0]} by {mentioned_categorical[0]}"
            state["business_question"] = query
            analytic_intent = "comparison"

    if df is not None:
        numeric_cols = state.get("dataset_profile", {}).get("numeric_columns", [])
        semantic_filters = map_semantic_filters(query, df, numeric_cols)
        if semantic_filters:
            semantic_ast = {
                "type": "logic",
                "operator": "and",
                "conditions": semantic_filters
            }
            ast = merge_ast_nodes(ast, semantic_ast)

    filters = extract_filters(ast)
    low_confidence = any(
        estimate_filter_confidence(f) < 0.75 for f in filters
    )

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
        "confidence": None,
        "low_confidence": low_confidence
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
    intent_type = intent.get("analytic_intent")
    
    if df is not None:
        numeric_columns = state.get("dataset_profile", {}).get("numeric_columns", [])
        for agg_type, keywords in aggregation_keywords.items():
            if any(word in query for word in keywords):
                intent["aggregation"] = agg_type
                if mentioned_numeric:
                    intent["aggregate_column"] = mentioned_numeric[0]
                else:
                    for col in numeric_columns:
                        if normalize(col) in normalize(query):
                            intent["aggregate_column"] = col
                            break
                break

        # If no aggregation keyword found, infer from analytic intent
        if not intent["aggregation"]:
            if intent_type in ["comparison", "extremes", "profiling"]:
                # Default to mean for comparison/profiling
                intent["aggregation"] = "mean"
                if mentioned_numeric:
                    intent["aggregate_column"] = mentioned_numeric[0]
                elif numeric_columns:
                    intent["aggregate_column"] = numeric_columns[0]
            elif intent_type == "temporal":
                # For trends, default to sum over time
                intent["aggregation"] = "sum"
                if mentioned_numeric:
                    intent["aggregate_column"] = mentioned_numeric[0]
                elif numeric_columns:
                    intent["aggregate_column"] = numeric_columns[0]

    # ------------------------
    # GROUP BY )
    # ------------------------
    
    if df is not None:
        categorical_columns = state.get("dataset_profile", {}).get("categorical_columns", [])
        if mentioned_categorical:
            intent["group_by"] = mentioned_categorical[0]
        else:
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
    if selected_columns:
        state["selected_columns"] = selected_columns

    # ------------------------
    # APPLY OPERATION MAPPING
    # ------------------------
    operations_hint = map_intents_to_operations(detected_intents)
    state["intent"]["operations_hint"] = operations_hint

    if should_call_llm_for_intent(state, ast, filters):
        state = llm_reasoning_node(state)
        reasoning = state.get("llm_reasoning")
        llm_ast = convert_reasoning_to_ast(reasoning, df)
        if llm_ast:
            state["intent"]["ast"] = merge_ast_nodes(state["intent"].get("ast"), llm_ast)
    else:
        print("\n[INFO] Skipping LLM - symbolic parsing sufficient")

    if state["intent"].get("ast"):
        state["intent"]["filters"] = extract_filters(state["intent"]["ast"])
    else:
        state["intent"]["filters"] = []

    state["intent"]["low_confidence"] = any(
        estimate_filter_confidence(f) < 0.75 for f in state["intent"]["filters"]
    )

    # ------------------------
    # TYPE
    # ------------------------
    
    if intent["aggregation"]:
        intent["type"] = "aggregation"
    elif state["intent"].get("ast"):
        intent["type"] = "filter"
    else:
        intent["type"] = "exploration"
    intent["analysis_type"] = intent.get("analytic_intent", "unknown")


    print("\n=== INTENT PARSER COMPLETE ===")
    print("Filters after llm reasoning:", state["intent"]["filters"])

    return state

