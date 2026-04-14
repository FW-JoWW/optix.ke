import json
import os
import re

import pandas as pd
from dotenv import load_dotenv

from state.state import AnalystState
from utils.openai_runtime import get_openai_client
from utils.value_resolver import resolve_value

load_dotenv()

VALID_OPERATORS = {
    "equals", ">", "<", ">=", "<=", "between", "!=", "contains"
}


def extract_json_object(text: str):
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}

    return {}


def normalize_filter(constraint):
    if not isinstance(constraint, dict):
        return None

    field = constraint.get("field")
    operator = constraint.get("operator")
    value = constraint.get("value")

    if not field or not operator:
        return None

    op_map = {
        "=": "equals",
        "==": "equals",
        "not equals": "!=",
    }
    operator = op_map.get(operator, operator)

    if operator not in VALID_OPERATORS:
        return None

    return {
        "type": "condition",
        "column": field,
        "operator": operator,
        "value": value,
    }


def validate_value_type(col, val, df):
    if col not in df.columns or val is None:
        return False

    series = df[col]

    if pd.api.types.is_numeric_dtype(series):
        if isinstance(val, list):
            return len(val) == 2 and all(isinstance(v, (int, float)) for v in val)
        return isinstance(val, (int, float))

    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        return isinstance(val, str)

    return False


def get_column_hints(df, numeric_columns, categorical_columns):
    numeric_ranges = {}
    for col in numeric_columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(
            df[col].astype(str).str.replace(r"[^\d\.\-]", "", regex=True),
            errors="coerce"
        ).dropna()
        if series.empty:
            continue
        numeric_ranges[col] = {
            "min": float(series.min()),
            "max": float(series.max())
        }

    categorical_samples = {}
    for col in categorical_columns:
        if col not in df.columns:
            continue
        values = df[col].dropna().astype(str).unique().tolist()
        categorical_samples[col] = values[:10]

    return numeric_ranges, categorical_samples


def normalize_reasoning(reasoning, df, numeric_columns, categorical_columns):
    if not isinstance(reasoning, dict):
        return None

    allowed_columns = set(numeric_columns + categorical_columns)
    constraints = reasoning.get("constraints", [])
    if not isinstance(constraints, list):
        constraints = []

    normalized_constraints = []
    seen = set()

    for item in constraints:
        if not isinstance(item, dict):
            continue

        field = item.get("field")
        operator = item.get("operator")
        value = item.get("value")
        confidence = item.get("confidence", "low")

        if field not in allowed_columns:
            continue

        if field in categorical_columns and isinstance(value, str):
            value = resolve_value(value, df[field].dropna().unique())

        candidate = {
            "field": field,
            "operator": operator,
            "value": value,
            "confidence": confidence if confidence in {"high", "medium", "low"} else "low"
        }

        normalized_filter = normalize_filter(candidate)
        if not normalized_filter:
            continue

        if not validate_value_type(field, normalized_filter["value"], df):
            continue

        key = (
            normalized_filter["column"],
            normalized_filter["operator"],
            str(normalized_filter["value"])
        )
        if key in seen:
            continue
        seen.add(key)
        normalized_constraints.append(candidate)

    logic = reasoning.get("logic", "and")
    if logic not in {"and", "or"}:
        logic = "and"

    intent_type = reasoning.get("intent_type", "unknown")
    if not isinstance(intent_type, str):
        intent_type = "unknown"

    return {
        "entities": reasoning.get("entities", []),
        "intent_type": intent_type,
        "constraints": normalized_constraints,
        "logic": logic,
        "group_by": reasoning.get("group_by", []),
        "aggregation": reasoning.get("aggregation", {"type": "none", "target": None})
    }


def llm_reasoning_node(state: AnalystState) -> AnalystState:
    """
    Uses the LLM only as a schema-constrained fallback for unresolved intent.
    """
    if state.get("disable_llm_reasoning"):
        state["llm_reasoning"] = None
        state["llm_reasoning_status"] = "disabled"
        print("\n[INFO] LLM reasoning disabled by state flag")
        return state

    client = get_openai_client()
    if client is None:
        state["llm_reasoning"] = None
        state["llm_reasoning_status"] = "unavailable: OPENAI_API_KEY not set"
        print("\n[INFO] OPENAI_API_KEY not set - skipping LLM reasoning")
        return state

    query = state.get("business_question", "").strip()
    df = state.get(state.get("active_dataset"))
    if df is None:
        raise ValueError("No dataset available for LLM reasoning.")

    ast_filters = state.get("intent", {}).get("filters", [])
    dataset_profile = state.get("dataset_profile", {})
    numeric_columns = dataset_profile.get("numeric_columns", [])
    categorical_columns = dataset_profile.get("categorical_columns", [])
    numeric_ranges, categorical_samples = get_column_hints(
        df,
        numeric_columns,
        categorical_columns
    )

    prompt = f"""
You are a schema-grounded reasoning engine for data analysis.
Reason strictly from the user query, the current AST, and the provided schema.
Prefer abstaining over guessing.

User query: "{query}"
Current AST: {state.get("intent", {}).get("ast")}
Existing filters: {ast_filters}
Available numeric columns: {numeric_columns}
Numeric ranges: {numeric_ranges}
Available categorical columns: {categorical_columns}
Categorical sample values: {categorical_samples}

Rules:
- Use only exact column names from the provided schema.
- Do not invent columns, values, entities, or business meaning.
- Add only missing constraints that are clearly supported by the query and schema.
- If a phrase is ambiguous across multiple columns, do not guess.
- Use only operators from: equals, !=, >, <, >=, <=, between, contains.
- Numeric values must be JSON numbers.
- Categorical values must be JSON strings.
- "between" values must be [low, high].
- If nothing can be added safely, return an empty constraints list.
- Output JSON only. No prose.

Return exactly one JSON object in this shape:
{{
  "entities": [],
  "intent_type": "filter | comparison | temporal | aggregation | mixed | unknown",
  "constraints": [
    {{
      "field": "exact column name",
      "operator": "equals | != | > | < | >= | <= | between | contains",
      "value": 0,
      "confidence": "high | medium | low"
    }}
  ],
  "logic": "and | or",
  "group_by": [],
  "aggregation": {{
    "type": "mean | sum | max | min | none",
    "target": null
  }}
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        state["llm_reasoning"] = None
        state["llm_reasoning_status"] = f"unavailable: {e}"
        print(f"\n[INFO] LLM reasoning unavailable - skipping: {e}")
        return state

    llm_text = response.choices[0].message.content.strip()
    print("\n[DEBUG] Raw LLM output:", llm_text)

    parsed = extract_json_object(llm_text)
    normalized = normalize_reasoning(
        parsed,
        df,
        numeric_columns,
        categorical_columns
    )

    if not normalized:
        state["llm_reasoning"] = None
        state["llm_reasoning_status"] = "invalid_output"
        print("[WARNING] Invalid LLM reasoning output")
        return state

    state["llm_reasoning"] = normalized
    state["llm_reasoning_status"] = "live_llm"
    print("\n[INFO] Validated LLM reasoning:", normalized)
    return state
