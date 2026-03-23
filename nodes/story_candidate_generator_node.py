from state.state import AnalystState

def story_candidate_generator_node(state: AnalystState) -> AnalystState:
    """
    Converts raw tool results into structured story candidates.
    Filters weak patterns and formats for LLM synthesis.
    """
    evidence = state.setdefault("analysis_evidence", {})
    tool_results = state.get("tool_results", {})
    story_candidates = []

    for key, result in tool_results.items():
        # Skip error results
        if "error" in key.lower():
            continue

        # Correlations
        if result.get("tool") == "correlation":
            corr = result.get("correlation")
            if abs(corr) >= 0.5:
                story_candidates.append({
                    "type": "correlation",
                    "columns": [result.get("column_1"), result.get("column_2")],
                    "value": corr
                })

        # T-tests
        elif result.get("type") == "t_test" or result.get("test") == "Independent t-test":
            p = result.get("p_value")
            if p is not None and p < 0.05:
                story_candidates.append({
                    "type": "t_test",
                    "column": result.get("column") or result.get("value_column"),
                    "group_column": result.get("group_column"),
                    "p_value": p
                })

        # Outliers
        elif result.get("type") == "outlier_detection" or result.get("tool") == "detect_outliers":
            if result.get("outlier_count", 0) > 0:
                story_candidates.append({
                    "type": "outliers",
                    "column": result.get("column"),
                    "count": result.get("outlier_count")
                })

        # Regression
        elif result.get("tool") == "regression":
            r2 = result.get("r_squared")
            if r2 >= 0.5:
                story_candidates.append({
                    "type": "regression",
                    "x_column": result.get("x_column"),
                    "y_column": result.get("y_column"),
                    "r_squared": r2
                })

    evidence["story_candidates"] = story_candidates
    print("\n=== STORY CANDIDATES GENERATED ===")
    print(story_candidates)

    return state