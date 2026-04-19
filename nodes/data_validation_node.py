from __future__ import annotations

from state.state import AnalystState
from validation import validate_cleaning


def data_validation_node(state: AnalystState) -> AnalystState:
    before_df = state.get("dataframe")
    after_df = state.get("cleaned_data")

    if before_df is None and after_df is None:
        state["data_validation"] = {"error": "No dataset provided."}
        return state

    if before_df is None:
        before_df = after_df
    if after_df is None:
        after_df = before_df

    validation_result = validate_cleaning(before_df, after_df)
    state["data_validation"] = validation_result
    state["cleaning_validation"] = validation_result
    state["clarification_questions"] = []

    if validation_result.get("row_loss_ratio", 0.0) > 0.1:
        state["clarification_questions"].append(
            "Cleaning changed the row count materially. Should we keep this cleaned version?"
        )
    if validation_result.get("anomalies"):
        state["clarification_questions"].append(
            "Validation detected anomalies after cleaning. Should the workflow continue with the cleaned dataset?"
        )

    return state
