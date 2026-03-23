from state.state import AnalystState
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any

def statistical_analysis_node(state: AnalystState) -> AnalystState:
    """
    Performs statistical tests automatically based on dataset.
    Stores results under state["analysis_evidence"]["statistical_tests"].
    """

    df: pd.DataFrame = state.get("analysis_dataset")
    if df is None:
        state.setdefault("analysis_evidence", {})
        state["analysis_evidence"]["statistical_tests"] = {"error": "No dataset provided."}
        return state

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    results: Dict[str, Any] = {}

    # Numeric correlations
    if len(numeric_cols) >= 2:
        correlations = {}
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr_coef, p_val = stats.pearsonr(df[col1].fillna(0), df[col2].fillna(0))
                correlations[f"{col1} vs {col2}"] = {
                    "test": "Pearson correlation",
                    "correlation_coefficient": corr_coef,
                    "p_value": p_val
                }
        results["numeric_correlations"] = correlations

    # Numeric by categorical (t-test or ANOVA)
    if numeric_cols and categorical_cols:
        cat_numeric_tests = {}
        for cat_col in categorical_cols:
            categories = df[cat_col].dropna().unique()
            for num_col in numeric_cols:
                if len(categories) == 2:
                    group1 = df[df[cat_col] == categories[0]][num_col].dropna()
                    group2 = df[df[cat_col] == categories[1]][num_col].dropna()
                    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
                    cat_numeric_tests[f"{num_col} by {cat_col}"] = {
                        "test": "Independent t-test",
                        "t_statistic": t_stat,
                        "p_value": p_val
                    }
                elif len(categories) > 2:
                    groups = [df[df[cat_col] == cat][num_col].dropna() for cat in categories]
                    f_stat, p_val = stats.f_oneway(*groups)
                    cat_numeric_tests[f"{num_col} by {cat_col}"] = {
                        "test": "ANOVA",
                        "f_statistic": f_stat,
                        "p_value": p_val
                    }
        results["numeric_by_categorical"] = cat_numeric_tests

    # Store results in analysis_evidence
    state.setdefault("analysis_evidence", {})
    state["analysis_evidence"]["statistical_tests"] = results

    print("Statistical Analysis Completed")
    return state

'''from state.state import AnalystState
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any

def statistical_analysis_node(state: AnalystState) -> AnalystState:
    """
    Performs statistical tests automatically based on dataset and business question.
    Updates AnalystState with:
    - statistical_results: p-values, test type, assumptions
    - recommended test
    """

    if state.get("dataset") is None:
        state["statistical_results"] = {"error": "No dataset provided."}
        return state

    df: pd.DataFrame = state["dataset"]
    results: Dict[str, Any] = {}

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # Determine relationships based on business question
    # For simplicity, assume we analyze numeric vs numeric correlations
    # and numeric vs categorical differences
    if len(numeric_cols) >= 2:
        correlations = {}
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr_coef, p_val = stats.pearsonr(df[col1].fillna(0), df[col2].fillna(0))
                correlations[f"{col1} vs {col2}"] = {
                    "test": "Pearson correlation",
                    "correlation_coefficient": corr_coef,
                    "p_value": p_val
                }
        results["numeric_correlations"] = correlations

    if numeric_cols and categorical_cols:
        cat_numeric_tests = {}
        for cat_col in categorical_cols:
            categories = df[cat_col].dropna().unique()
            for num_col in numeric_cols:
                if len(categories) == 2:
                    # Two groups → t-test
                    group1 = df[df[cat_col] == categories[0]][num_col].dropna()
                    group2 = df[df[cat_col] == categories[1]][num_col].dropna()
                    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
                    cat_numeric_tests[f"{num_col} by {cat_col}"] = {
                        "test": "Independent t-test",
                        "t_statistic": t_stat,
                        "p_value": p_val
                    }
                elif len(categories) > 2:
                    # More than two groups → ANOVA
                    groups = [df[df[cat_col] == cat][num_col].dropna() for cat in categories]
                    f_stat, p_val = stats.f_oneway(*groups)
                    cat_numeric_tests[f"{num_col} by {cat_col}"] = {
                        "test": "ANOVA",
                        "f_statistic": f_stat,
                        "p_value": p_val
                    }
        results["numeric_by_categorical"] = cat_numeric_tests

    # Update state
    state["statistical_results"] = results

    return state

'''