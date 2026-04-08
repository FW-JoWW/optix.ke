# nodes/column_selection_node.py
from state.state import AnalystState
from utils.semantic_matcher import semantic_column_match

def column_selection_node(state: AnalystState) -> AnalystState:

    column_registry = state.get("column_registry")
    df = state.get("analysis_dataset")

    if column_registry is None:
        raise ValueError("Column registry missing")

    if df is None:
        raise ValueError("Filtered dataset missing")

    question = state.get("business_question", "")
    intent = state.get("intent", {})

    candidate_columns = list(column_registry.keys())

    matched_columns = semantic_column_match(
        question,
        candidate_columns,
        threshold=0.4
    )

    intent_columns = [f["column"] for f in intent.get("filters", [])]

    if intent_columns:
        selected_columns = list(set(intent_columns + matched_columns))
    else:
        selected_columns = matched_columns

    if not selected_columns:
        selected_columns = df.columns.tolist()[:5]

    selected_columns = [c for c in selected_columns if c in df.columns]

    if not selected_columns:
        raise ValueError("No valid colimns selected after filtering")

    analysis_df = df[selected_columns].copy()

    state["selected_columns"] = selected_columns
    state["analysis_dataset"] = analysis_df

    print("\n=== COLUMN SELECTION ===")
    print(selected_columns)

    return state


"""# nodes/column_selection_node.py
from state.state import AnalystState
from utils.semantic_matcher import semantic_column_match
import pandas as pd

def column_selection_node(state: AnalystState) -> AnalystState:
    '''
    Selects columns for analysis using semantic roles and
    embedding-based matching with the business question.
    Works for any dataset.
    '''

    # Load column registry and dataset
    column_registry = state.get("column_registry")
    df = state.get("cleaned_data") or state.get("dataframe")

    if column_registry is None:
        raise ValueError("Column registry not found. Run column_semantic_classifier first.")

    if df is None:
        raise ValueError("Dataset not found.")

    question = state.get("business_question", "")

    # Filter columns by semantic roles
    allowed_roles = ["numeric_measure", "categorical_feature", "datetime"]
    candidate_columns = [
        col for col, meta in column_registry.items()
        if meta.get("semantic_role") in allowed_roles
    ]

    if not candidate_columns:
        raise ValueError("No valid columns found for analysis.")

    # Use semantic matching to prioritize columns for this question
    intent = state.get("intent", {})
    
    # prioritize columns from intent
    intent_columns = [f["column"] for f in intent.get("filters", [])]
    if intent.get("group_by"):
        intent_columns.append(intent["group_by"])
        
    #matched_columns = list(set(intent_columns + candidate_columns))
    matched_columns = semantic_column_match(question, candidate_columns, threshold=0.4)

    # Further classify columns by type for analysis heuristics
    numeric_cols = [col for col in matched_columns if column_registry[col]["semantic_role"] == "numeric_measure"]
    categorical_cols = [col for col in matched_columns if column_registry[col]["semantic_role"] == "categorical_feature"]
    datetime_cols = [col for col in matched_columns if column_registry[col]["semantic_role"] == "datetime"]

    # Heuristic: determine columns based on question intent
    relationship_keywords = ["relationship", "correlation", "impact", "effect", "influence"]
    comparison_keywords = ["compare", "difference", "by", "between"]
    trend_keywords = ["trend", "over time", "growth", "change"]

    selected_columns = []

    question_lower = question.lower()
    if any(k in question_lower for k in relationship_keywords):
        selected_columns = numeric_cols[:4]
    elif any(k in question_lower for k in comparison_keywords):
        selected_columns = categorical_cols[:2] + numeric_cols[:2]
    elif any(k in question_lower for k in trend_keywords):
        selected_columns = datetime_cols[:1] + numeric_cols[:2]
    else:
        selected_columns = list(set(
            intent_columns +
            numeric_cols[:3] + 
            categorical_cols[:2]
        ))
        #selected_columns = matched_columns[:4]

    
    # Force include columns explicitly mentioned in question
    '''forced_columns = [
        col for col in df.columns
        if col.lower() in question_lower
    ]'''

    # selected_columns = list(set(selected_columns + forced_columns))

    # -------------------------
    # FORCE INCLUDE INTENT COLUMNS
    # -------------------------
    intent = state.get("intent", {})
    
    needed_cols = []

    for f in intent.get("filters", []):
        needed_cols.append(f["column"])
    
    if intent.get("group_by"):
        needed_cols.append(intent["group_by"])
     
    if intent.get("aggregate_column"):
        needed_cols.append(intent["aggregate_column"])

    needed_cols = [c for c in needed_cols if c in df.columns]
    
    selected_columns = list(set(selected_columns + needed_cols))
    '''# Ensure columns exist in the dataframe
    selected_columns = [col for col in selected_columns if col in df.columns]'''

    '''# Store original dataset for filter recovery
    if "analysis_dataset_original" not in state:
        state["analysis_dataset_original"] = df.copy()'''
    
    # Remove invalid columns
    if not needed_cols:
        raise ValueError("No valid selected after validation.")
    
    # Build the analysis dataset
    analysis_df = df[selected_columns].copy()

    # Optional: drop columns with >80% missing values
    filtered_columns = [col for col in analysis_df.columns if analysis_df[col].notna().sum() > len(analysis_df) * 0.2]
    analysis_df = analysis_df[filtered_columns]
    selected_columns = filtered_columns

    # Update state
    state["selected_columns"] = selected_columns #analysis_df.reset_index(drop=True)
    state["analysis_dataset"] = analysis_df.copy()
    state["raw_analysis_dataset"] = analysis_df.copy()

    # Logging
    print("\n=== COLUMN SELECTION ===")
    print(f"Selected columns for analysis: {selected_columns}")
    print(f"Analysis dataset shape: {analysis_df.shape}")

    return state

"""