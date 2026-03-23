#nodes/analysis_planner_node
from state.state import AnalystState
from typing import List, Dict


def analysis_planner_node(state: AnalystState) -> AnalystState:
    """
    Determines which analyses to perform based on:
    - business question
    - selected dataset
    - existing dataset profile
    - sementic column registry
    """

    question = state.get("business_question", "").lower()

    #df = state.get("analysis_dataset")
    df = None
    if state.get("analysis_dataset") is not None:
        df = state["analysis_dataset"]
    elif state.get("cleaned_data") is not None:
        df = state["cleaned_data"]    
    elif state.get("dataframe") is not None:
        df = state["dataframe"]


    if df is None:
        raise ValueError("No analysis dataset found.")

    evidence = state.setdefault("analysis_evidence", {})
    #eda = state.get("eda_results", {})

    #columns = [c.lower() for c in df.columns]
    dataset_profile = state.get("dataset_profile", {})
    column_registry = state.get("column_registry", {})
    #numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    #categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    selected_columns = state.get("selected_columns", list(df.columns))
    
    #numeric_cols = dataset_profile.get("numeric_columns", [])
    #categorical_cols = dataset_profile.get("categorical_columns", [])

    #remove identifiers from analysis
    numeric_cols = [
        c for c in dataset_profile.get("numeric_columns", [])
        if c in selected_columns
        and column_registry.get(c, {}).get("semantic_role") != "identifier"
    ]
    categorical_cols = [
        c for c in dataset_profile.get("categorical_columns", [])
        if c in selected_columns
        and column_registry.get(c, {}).get("semantic_role") != "identifier"
    ]

    unique_counts = {col: df[col].nunique() for col in df.columns}

    plan: List[Dict] = []

    # ------------------------------------------
    # Detect columns mentioned inthe question
    # ------------------------------------------

    mentioned_columns = [
        col for col in selected_columns
        if col.lower() in question
    ]    
    mentioned_numeric = [c for c in mentioned_columns if c in numeric_cols]
    mentioned_categorical = [c for c in mentioned_columns if c in categorical_cols]

    # -------------------------
    # Relationship / effect
    # -------------------------

    if any(word in question for word in ["relationship", "correlation", "affect", "impact"]):

        #numeric_mentioned = [c for c in mentioned_columns if c in numeric_cols]

        if len(mentioned_numeric) >= 2:
            col1, col2 = mentioned_numeric[:2]
            plan.append({
                "tool": "correlation",
                "columns": [col1, col2]
            })
        '''else:
            # fallback
            if len(numeric_cols) >= 2:    
               col1, col2 = numeric_cols[:2]
            else:
                col1 = col2 = None   
            if col1 and col2:   
                plan.append({
                    "tool": "correlation",
                    "columns": [col1, col2]
                })

                plan.append({
                    "tool": "regression",
                    "columns": [col1, col2]
                })'''

    # -------------------------
    # Group comparison
    # -------------------------

    if any(word in question for word in ["compare", "difference", "better"]):
        
        for num in numeric_cols:
            for cat in categorical_cols:
                n_unique = unique_counts.get(cat, 0)
                if n_unique ==2:
                    tool = "ttest"
                elif n_unique > 2:
                    tool = "anova"
                else:
                    continue
                plan.append({
                    "tool": tool,
                    "columns": [num, cat]
                })
        
        '''if mentioned_numeric and mentioned_catehorical:
            num = mentioned_numeric[0]
            cat = mentioned_catehorical[0]
            n_unique = unique_counts.get(cat, 0)
            if n_unique == 2:
                tool = "ttest"
            elif n_unique > 2:
                tool = "anova"    
            else:
                tool = None

            if tool:
                plan.append({
                    "tool": tool,
                    "columns": [num, cat]
                })    
        
        elif numeric_cols and categorical_cols:
            num = numeric_cols[0]
            cat = categorical_cols[0]
            n_unique = unique_counts.get(cat, 0)
            if n_unique == 2:
                tool = "ttest"
            elif n_unique > 2:
                tool = "anova"
            else:
                tool = None

            if tool:
                plan.append({
                    "tool": tool,
                    "columns": [num, cat]
                })'''
        
    # -------------------------
    # Outliers
    # -------------------------

    if any(word in question for word in ["outlier", "unusual", "anomaly"]):
        for num in numeric_cols:
            plan.append({
                "tool": "detect_outliers",
                "columns": [num]
            })

    # -------------------------
    # Summary statistics
    # -------------------------

    if any(word in question for word in ["average", "mean", "summary", "describe"]):
        if numeric_cols:
            plan.append({
                "tool": "summary_statistics",
                "columns": numeric_cols
            })

    # -------------------------
    # Fallback
    # -------------------------

    if not plan and len(numeric_cols) >= 2:
        plan.append({
            "tool": "correlation",
            "columns": numeric_cols[:2]
        })

    # Remove duplicates
    seen = set()
    unique_plan = []

    for item in plan:
        key = (item["tool"], tuple(item["columns"]))
        if key not in seen:
            unique_plan.append(item)
            seen.add(key)

    evidence["analysis_plan"] = unique_plan

    print("\n=== ANALYSIS PLAN ===")
    print("Dataset shape:", df.shape)
    for p in unique_plan:
        print("-", p)

    return state
