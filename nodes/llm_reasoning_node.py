# nodes/llm_reasoning_node.py
import json
import re
from state.state import AnalystState
from openai import OpenAI  
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------
# HELPER: SAFE JSON EXTRACTOR
# -----------------------------
def extract_json(text: str):
    """
    Extracts JSON array from LLM output even if wrapped in text.
    """
    try:
        return json.loads(text)
    except:
        pass

    # Try to extract JSON list using regex
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return []

    return []

# -------------------
# NOMALIZE FILTERS
# -------------------
def normalize_filter(f):
    if not isinstance(f, dict):
        return None

    col = f.get("column")
    op = f.get("operator")
    val = f.get("value")

    if not col or not op:
        return None

    OP_MAP = {
        "=": "equals",
        "==": "equals",
        "not equals": "!=",
        "!=": "!=",
    }

    op = OP_MAP.get(op, op)

    return {
        "type": "condition",
        "column": col,
        "operator": op,
        "value": val
    }

VALID_OPERATORS = {
    "equals", ">", "<", ">=", "<=", "between", "!=", "contains"
}

# ------------------------
# TYPE VALIDATION
# ------------------------
def validate_value_type(col, val, df):
    if col not in df.columns:
        return False
    
    if val is None:
        return False
    
    series = df[col]

    # numeric column → value must be numeric
    if pd.api.types.is_numeric_dtype(series):
        try:
            if isinstance(val, list):
                return all(isinstance(v, (int, float)) for v in val)
            return isinstance(val, (int, float))
        except:
            return False

    # categorical column → value must be string
    if pd.api.types.is_string_dtype(series):
        if isinstance(val, str):
            return True
        return False

    return False

# ------------------------
# CONFLICT RESOLUTION
# ------------------------
def is_conflicting(existing_filters, new_filter):
    for f in existing_filters:
        if f["column"] == new_filter["column"]:
            # same column numeric conflict
            if f["operator"] == new_filter["operator"]:
                return True
    return False


# ------------------------
# LLM WRAPPER
# ------------------------
def llm_reasoning_node(state: AnalystState) -> AnalystState:
    """
    Enhances AST filters using LLM reasoning.
    Only produces additional filters in strict schema.
    """
    query = state.get("business_question", "").strip()
    ast_filters = state.get("intent", {}).get("filters", [])
    df = state.get(state.get("active_dataset"))

    # Early exit if no dataset
    if df is None:
        raise ValueError("No dataset available for LLM reasoning.")

    dataset_columns = {
        "numeric": state.get("dataset_profile", {}).get("numeric_columns", []),
        "categorical": state.get("dataset_profile", {}).get("categorical_columns", [])
    }

    allowed_columns = dataset_columns["numeric"] + dataset_columns["categorical"]

    # Build LLM prompt
    prompt = f"""
    You are a reasoning engine for data analysis.
    Your job is to interpret user intent before any filtering or execution happens.
    Do NOT think in database terms. Think in meaning first, then structure. 
    User query: "{query}"
    Current AST: {state.get("intent", {}).get("ast")}
    Available numeric columns: {dataset_columns['numeric']}
    Available categorical columns: {dataset_columns['categorical']}

    your task:
    - Identify ONLY missing or implied conditions NOT already in the AST
    - DO NOT repeat existing filters.
    - DO NOT invent columns.
    - DO NOT restructure the AST.
    - Only suggest NEW conditions

    Rules:
    - Numeric columns → value must be numbers
    - Categorical columns → values must be strings
    - "between" must return [low, high] as numbers

    Output MUST be a JSON object:

    {{
        "entities": [],
        "intent_type": "filter | comparison | temporal | aggregation | mixed",
        "constraints": [
            {{
            "field": "column name OR semantic concept",
            "operator": "equals | > | < | between | contains",
            "value": "raw value or list",
            "confidence": "high | medium | low",
            "semantic": true | false
            }}
        ],
        "logic": "and | or | mixed",
        "group_by": [],
        "aggregation": {{
            "type": "mean | sum | max | min | none",
            "target": null
        }}
    }}

    If no new filters → return []

    NO explanation. ONLY JSON.
    """

    # Call LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0  # deterministic
    )

    # Parse LLM output
    #try:
    llm_text = response.choices[0].message.content.strip()
    print("\n[DEBUG] Raw LLM output:", llm_text)

    #---- SAFE PARSE ----
    llm_filters = extract_json(llm_text)
    
    state["llm_reasoning"] = {
        "raw": llm_text,
        "parsed": llm_filters
    }

    if not isinstance(llm_filters, list):
        print("[WARNING] LLM output is not a list. Ignoring.")
        llm_filters = []
    
    # sanity check each suggested filter
    validated_filters = []
 
    for f in llm_filters:
        f = normalize_filter(f)
        if not f:
            continue

        if not isinstance(f, dict):
            continue

        # LLM must ONLY return conditions
        if "column" not in f:
            continue

        col = f["column"]
        op = f["operator"]
        val = f["value"]

        if col not in allowed_columns:
            continue

        if op not in VALID_OPERATORS:
            continue

        if not validate_value_type(col, val, df):
            continue

        validated_filters.append(f)
        
        '''# --- OPERATOR NORMALIZATION ---
        OPERATOR_MAP = {
            "<=": "<=",
            ">=": ">=",
            "not equals": "!=",
            "!=": "!="
        }

        if not isinstance(f, dict):
            continue
        col = f.get("column")
        raw_op = f.get("operator")
        val = f.get("value")

        # normalize operator
        op = OPERATOR_MAP.get(raw_op, raw_op)

        if not isinstance(f, dict):
            continue

        col = f.get("column")
        op = f.get("operator")
        val = f.get("value")

        print("[DEBUG] Allowed columns:", allowed_columns)

        # --- VALIDATION ---
        if not validate_value_type(col, val, df):
            print(f"[WARNING] Type mismatch for column {col}")
            continue
        
        if col not in allowed_columns:
            print(f"[WARNING] Ignoring unknown column from LLM: {col}")
            continue

        #if col in dataset_columns["numeric"] + dataset_columns["categorical"]:
        if op not in ["equals", ">", "<", "<=", ">=", "between", "contains", "!="]:
            print(f"[WARNING] Ignoring invalid operator from LLM: {op}")
            continue

        validated_filters.append({
            "type": "condition",
            "column": col,
            "operator": op,
            "value": val
        })'''
    
    # ---- MERGE FILTERS ----
    existing_columns = {f.get("column") for f in ast_filters}
    existing_set = {(f.get("column"), f.get("operator"), str(f.get("value"))) for f in ast_filters}

    for f in validated_filters:
        # If f is a logic node (like OR), merge it separately
        '''if f.get("type") == "logic":
            ast_filters.append(f)
            continue'''
        key = (f["column"], f["operator"], str(f["value"]))
        if key not in existing_set: #and f["column"] not in existing_columns:
            ast_filters.append(f)
            
#except Exception as e:
    
    state["intent"]["filters"] = ast_filters

    print("\n[INFO] LLM reasoning filters merged:", validated_filters)

    return state