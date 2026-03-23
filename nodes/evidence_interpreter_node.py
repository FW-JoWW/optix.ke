from state.state import AnalystState
import pandas as pd

# ✅ IMPORT YOUR GUARDS
from guards.validation_layer import (
    run_full_validation,
    compute_confidence
)
from engine.reasoning_orchestrator import reasoning_orchestrator

def evidence_interpreter_node(state: AnalystState) -> AnalystState:
    """
    Interprets raw statistical tool outputs and converts them into
    structured, validated story candidates.
    """

    evidence = state.setdefault("analysis_evidence", {})
    tool_results = evidence.get("tool_results", {})

    if not tool_results:
        print("No tool results found for interpretation.")
        evidence["story_candidates"] = []
        return state

    print("\n=== INTERPRETING STATISTICAL EVIDENCE ===")

    story_candidates = []
    df = state.get("analysis_dataset")

    raw_df = state.get("raw_analysis_dataset")

    for key, result in tool_results.items():

        if result.get("tool") != "anova":
            continue

        p = result.get("p_value")
        f_stat = result.get("f_statistic")
        num_col = result.get("numeric_column")
        cat_col = result.get("categorical_column")

        if raw_df is None or p is None:
            continue

        safe_df = raw_df.copy()
        safe_df[num_col] = pd.to_numeric(safe_df[num_col], errors="coerce")
        safe_df = safe_df.dropna(subset=[num_col, cat_col])

        if safe_df.empty:
            continue

        group_means = (
            safe_df.groupby(cat_col)[num_col]
            .mean()
            .sort_values(ascending=False)
        )

        if len(group_means) < 2:
            continue

        top_group = group_means.index[0]
        bottom_group = group_means.index[-1]

        top_value = float(group_means.iloc[0])
        bottom_value = float(group_means.iloc[-1])

        diff = top_value - bottom_value

        mean_val = safe_df[num_col].mean()
        relative_effect = abs(diff) / mean_val if mean_val != 0 else 0

        # -----------------------------
        # CORE FILTER (statistical truth)
        # -----------------------------
        if not (p < 0.05 and relative_effect > 0.1):
            continue        

        # -----------------------------
        # COMPUTE CONFIDENCE (SYSTEM)
        # -----------------------------
        computed_conf = compute_confidence(p, relative_effect)

        # -----------------------------
        # CRITIC OUTPUT (placeholder for now)
        # -----------------------------
        critic_output = {
            "issues": []
        }


        # -----------------------------
        # VALIDATION PIPELINE
        # -----------------------------
        
        def get_category_values(df, categorical_columns):
            category_values = {}
        
            for col in categorical_columns:
                if col not in df.columns:
                            continue
                
                category_values[col] = df[col].dropna().unique().tolist()
        
            return category_values
        
        #print(f"[RESULT STATUS] {result['status']}")
    
        # -----------------------------
        # MULTI-TEMPLATE INSIGHT ENGINE
        # -----------------------------

        def generate_templates(top_group, bottom_group, num_col, diff, cat_col):
            diff = round(diff, 2)
            return [
                f"{top_group} has higher {num_col} than {bottom_group} by {diff}",
                #f"{num_col} is higher for {top_group} compared to {bottom_group}",
                #f"{top_group} leads in {num_col} among {cat_col}",
                f"{bottom_group} has lower {num_col} than {top_group} by {diff}"
            ]


        def get_category_values(df, categorical_columns):
            category_values = {}

            for col in categorical_columns:
                if col not in df.columns:
                   continue
                category_values[col] = df[col].dropna().unique().tolist()

            return category_values
        
        # Generate candidate insights
        templates = generate_templates(top_group, bottom_group, num_col, diff, cat_col)
        
        # Prepare category values ONCE
        category_values = get_category_values(
            raw_df,
            safe_df.columns.tolist()
        )    
        
        accepted_output = None
        
        for template_text in templates:

            print(f"\n[TESTING TEMPLATE] {template_text}")
        
            llm_output = {
                "insight": template_text,
                "evidence_used": [key],
                "assumptions": [],
                "uncertainties": [],
                "confidence": computed_conf
            }
        
            computed_conf = compute_confidence(p, relative_effect)
        
            critic_output = {"issues": []}

            # -------------
            # RETRY LOOP
            # -------------
            result = reasoning_orchestrator(
                candidate=llm_output,
                validation_fn=lambda x: run_full_validation(
                    output=x,
                    tool_results=tool_results,
                    dataset_columns=safe_df.columns.tolist(),
                    category_values=category_values,
                    computed_conf=computed_conf,
                    critic_output=critic_output
                )
            )

            print(f"[RESULT STATUS] {result['status']}")

            if result["status"] == "accepted":
                accepted_output = result["output"]
                break # Stop at first valid insight
            else:
                print(f"[TEMPLATE FAILED] {template_text}")

        #if nothing worked reject
        if not accepted_output:
            print(f"[REJECTED AFTER RETRIES] {top_group} vs {bottom_group}")
            continue

        # -----------------------------
        # ACCEPTED STORY
        # -----------------------------
        story_candidates.append({
            **accepted_output,
            "type": "group_difference",
            "column": num_col,
            "group_column": cat_col,
            "top_group": str(top_group),
            "bottom_group": str(bottom_group),
            "top_value": top_value,
            "bottom_value": bottom_value,
            "effect_size": diff,
            "direction": f"{top_group} > {bottom_group}",
            "p_value": p,
            "f_statistic": f_stat
        })

    evidence["story_candidates"] = story_candidates

    print("\n=== STORY CANDIDATES (VALIDATED) ===")
    for s in story_candidates:
        print("-", s["insight"])

    return state

'''from state.state import AnalystState
import pandas as pd


def evidence_interpreter_node(state: AnalystState) -> AnalystState:
    """
    Interprets raw statistical tool outputs and converts them into
    structured, meaningful story candidates.
    """

    evidence = state.setdefault("analysis_evidence", {})
    tool_results = evidence.get("tool_results", {})

    if not tool_results:
        print("No tool results found for interpretation.")
        evidence["story_candidates"] = []
        return state

    print("\n=== INTERPRETING STATISTICAL EVIDENCE ===")

    story_candidates = []

    df = state.get("analysis_dataset")

    for key, result in tool_results.items():

        tool_type = result.get("tool")

        # =============================
        # ANOVA 
        # =============================
        if tool_type == "anova":

            p = result.get("p_value")
            f_stat = result.get("f_statistic")
            num_col = result.get("numeric_column")
            cat_col = result.get("categorical_column")

            raw_df = state.get("raw_analysis_dataset")
            
            if raw_df is not None and p is not None:

                # ✅ FORCE NUMERIC SAFELY
                safe_df = raw_df.copy()
                safe_df[num_col] = pd.to_numeric(safe_df[num_col], errors="coerce")
                
                # Drop rows where conversion failed
                safe_df = safe_df.dropna(subset=[num_col, cat_col])
                
                if safe_df.empty:
                    continue
                
                group_means = (
                    raw_df.groupby(cat_col)[num_col]
                    .mean()
                    .sort_values(ascending=False)
                )

                if len(group_means) < 2:
                    continue

                top_group = group_means.index[0]
                bottom_group = group_means.index[-1]

                top_value = float(group_means.iloc[0])
                bottom_value = float(group_means.iloc[-1])

                diff = top_value - bottom_value

                # Calculate relative effect size (importance)
                mean_val = safe_df[num_col].mean()
                relative_effect = abs(diff) / mean_val if mean_val != 0 else 0

                # STRICT relevance filter (THIS is the key fix)
                if p < 0.05 and relative_effect > 0.1:
                #if p < 0.05:

                    story_candidates.append({
                        "type": "group_difference",

                        # CORE INSIGHT
                        "insight": f"{top_group} has higher {num_col} than {bottom_group} by {round(diff,2)}",

                        # STRUCTURED FIELDS
                        "column": num_col,
                        "group_column": cat_col,
                        "top_group": str(top_group),
                        "bottom_group": str(bottom_group),
                        "top_value": top_value,
                        "bottom_value": bottom_value,
                        "effect_size": diff,
                        "direction": f"{top_group} > {bottom_group}",

                        # STATISTICAL CONTEXT
                        "p_value": p,
                        "f_statistic": f_stat,

                        # INTERPRETATION LAYERS
                        "confidence": "high",
                        "impact": "high" if abs(diff) > 10 else "moderate",
                        "business_relevance": "high"
                    })

        # =============================
        # Correlation 
        # =============================
        elif tool_type == "correlation":

            corr = result.get("correlation")

            if corr is not None and abs(corr) >= 0.5:

                direction = "positive" if corr > 0 else "negative"

                story_candidates.append({
                    "type": "correlation",

                    "insight": f"{result.get('column_1')} and {result.get('column_2')} have a {direction} relationship",

                    "columns": [
                        result.get("column_1"),
                        result.get("column_2")
                    ],
                    "value": corr,
                    "direction": direction,
                    "strength": "strong" if abs(corr) > 0.7 else "moderate",

                    "confidence": "high" if abs(corr) > 0.7 else "medium",
                    "business_relevance": "medium"
                })

        # =============================
        # Regression 
        # =============================
        elif tool_type == "regression":

            r2 = result.get("r_squared")

            if r2 is not None and r2 >= 0.5:

                story_candidates.append({
                    "type": "regression",

                    "insight": f"{result.get('x_column')} predicts {result.get('y_column')} with R² of {round(r2,2)}",

                    "x_column": result.get("x_column"),
                    "y_column": result.get("y_column"),
                    "r_squared": r2,

                    "strength": "strong" if r2 > 0.7 else "moderate",
                    "confidence": "high" if r2 > 0.7 else "medium",
                    "business_relevance": "medium"
                })

        # =============================
        # T-test 
        # =============================
        elif tool_type == "ttest":

            p = result.get("p_value")

            if p is not None and p < 0.05:

                story_candidates.append({
                    "type": "t_test",

                    "insight": f"{result.get('categorical_column')} significantly impacts {result.get('numeric_column')}",

                    "column": result.get("numeric_column"),
                    "group_column": result.get("categorical_column"),
                    "p_value": p,

                    "confidence": "high",
                    "business_relevance": "high"
                })

        # =============================
        # Outliers
        # =============================
        elif tool_type == "detect_outliers":

            count = result.get("outlier_count", 0)

            if count > 0:

                story_candidates.append({
                    "type": "outliers",

                    "insight": f"{count} outliers detected in {result.get('column')}",

                    "column": result.get("column"),
                    "count": count,

                    "confidence": "medium",
                    "business_relevance": "low"
                })

        # =============================
        # Summary statistics
        # =============================
        elif tool_type == "summary_statistics":

            summary = result.get("summary", {})

            for col, stats in summary.items():

                mean = stats.get("mean")
                std = stats.get("std")

                if std and abs(mean) > 2 * std:

                    story_candidates.append({
                        "type": "numeric_anomaly",

                        "insight": f"{col} shows abnormal distribution (mean deviates strongly)",

                        "column": col,
                        "mean": mean,
                        "std_dev": std,

                        "confidence": "medium",
                        "business_relevance": "low"
                    })

    evidence["story_candidates"] = story_candidates

    print("\n=== STORY CANDIDATES GENERATED ===")
    for s in story_candidates:
        print("-", s["insight"])

    return state

'''