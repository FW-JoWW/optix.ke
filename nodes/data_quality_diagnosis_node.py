# nodes/data_quality_diagnosis_node.py
from state.state import AnalystState
from utils.issue_detector import detect_issues
import json

# Placeholder for your LLM API
def llm_reasoning(detected_issues: dict, dataset_summary: dict) -> dict:
    """
    Runs LLM reasoning to assign explanations and confirm cleaning actions.
    Here we simulate with deterministic explanations for now.
    Replace this with your actual LLM call.
    """
    enriched_issues = []

    for issue in detected_issues["detected_issues"]:
        col = issue.get("column")
        issue_type = issue.get("issue_type")
        severity = issue.get("severity")

        explanation = ""
        recommended_action = ""

        # Example deterministic explanations (can be replaced with LLM)
        if issue_type == "missing_values":
            explanation = f"{issue['missing_count']} missing values in column {col}"
            recommended_action = "impute"
        elif issue_type == "duplicate_rows":
            explanation = f"{issue['duplicate_count']} duplicate rows in dataset"
            recommended_action = "remove_duplicates"
        elif issue_type == "outliers":
            explanation = f"{issue['outlier_count']} outliers detected in column {col}"
            recommended_action = "investigate_or_cap"
        elif issue_type == "constant_column":
            explanation = f"Column {col} contains a single unique value"
            recommended_action = "drop_column"
        elif issue_type == "high_cardinality":
            explanation = f"Column {col} has high cardinality (>80% unique)"
            recommended_action = "consider_encoding_or_drop"
        elif issue_type == "numeric_as_object":
            explanation = f"Column {col} is numeric but stored as object"
            recommended_action = "convert_to_numeric"

        enriched_issues.append({
            "column": col,
            "issue_type": issue_type,
            "severity": severity,
            "explanation": explanation,
            "recommended_action": recommended_action
        })

    return {"issues": enriched_issues}

def data_quality_diagnosis_node(state: AnalystState) -> AnalystState:
    """
    Hybrid Data Quality Diagnosis Node:
    1. Runs deterministic rules from issue_detector
    2. Runs LLM reasoning to enrich issues
    3. Saves structured output to state
    """

    df = state.get("dataframe")
    if df is None:
        raise ValueError("No dataframe found in state.")

    print("\n=== DATA QUALITY DIAGNOSIS NODE ===")

    # Step 1: Rule-based detection
    detected_issues = detect_issues(df)

    # Step 2: Dataset summary for LLM context
    dataset_summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist()
    }

    # Step 3: LLM reasoning
    structured_issues = llm_reasoning(detected_issues, dataset_summary)

    # Step 4: Save to state
    state["data_quality_issues"] = structured_issues

    print(json.dumps(structured_issues, indent=2))

    return state