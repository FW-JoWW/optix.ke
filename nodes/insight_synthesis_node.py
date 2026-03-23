from state.state import AnalystState
from typing import Dict, Any

def insight_synthesis_node(state: AnalystState) -> AnalystState:
    """
    Converts statistical results and EDA into readable insights.
    Updates AnalystState with:
    - insights: List of sentences explaining results
    - clarification_questions: List of questions if more info is needed
    """

    insights = state.get("insights", [])
    questions = state.get("clarification_questions", [])

    stat_results: Dict[str, Any] = state.get("statistical_results", {})
    eda_results: Dict[str, Any] = state.get("eda_results", {})

    # Numeric correlations
    numeric_corr = stat_results.get("numeric_correlations", {})
    for k, v in numeric_corr.items():
        corr = v.get("correlation_coefficient", 0)
        p_val = v.get("p_value", 1)
        if p_val < 0.05:
            insights.append(f"{k} shows a significant correlation (r={corr:.2f}, p={p_val:.3f}).")
        else:
            insights.append(f"{k} shows no significant correlation (r={corr:.2f}, p={p_val:.3f}).")

    # Numeric by categorical
    num_cat = stat_results.get("numeric_by_categorical", {})
    for k, v in num_cat.items():
        test_name = v.get("test", "")
        p_val = v.get("p_value", 1)
        if p_val < 0.05:
            insights.append(f"{k} differs significantly across groups ({test_name}, p={p_val:.3f}).")
        else:
            insights.append(f"{k} shows no significant difference across groups ({test_name}, p={p_val:.3f}).")

    # EDA-based insights
    eda_insights = eda_results.get("insights", [])
    insights.extend(eda_insights)

    # Clarifying questions if data is missing or unusual
    dataset = state.get("dataframe")
    if dataset is not None and not insights:
        insights.append(f"Dataset has {dataset.shape[0]} row and {dataset.shape[1]} columns.")
        if dataset.shape[0] < 10:
            questions.append("Dataset id small; results may be unreliable.")
    '''if dataset is None or dataset.empty:
        questions.append("No dataset provided. Can you provide the dataset?")
'''
    if not numeric_corr and not num_cat and dataset is not None:
        questions.append("Could you specify which numeric variables you want to analyze?")
        
    
    # Update state
    state["insights"] = insights
    state["clarification_questions"] = questions

    return state

