# nodes/data_quality_diagnosis_node.py
import json

from ambiguity_detector import detect_ambiguity
from context_inference import infer_context
from data_profiling import profile_dataset
from structural_signal_extraction import extract_structural_signals
from state.state import AnalystState
from utils.cleaning_recommender import recommend_cleaning_issues
from utils.issue_detector import detect_issues

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

    profile = profile_dataset(df)
    ambiguity = detect_ambiguity(profile)
    structural_signals = extract_structural_signals(profile)
    llm_reasoning_allowed = bool(state.get("enable_llm_reasoning", True) and not state.get("disable_llm_reasoning"))
    if profile.get("column_count", 0) > 40:
        llm_reasoning_allowed = False

    # Step 1: Rule-based detection
    detected_issues = detect_issues(df)

    # Step 2: constrained reasoning over column context
    structured_issues = recommend_cleaning_issues(
        detected_issues,
        df,
        base_profiles=profile.get("columns", {}),
    )
    context = infer_context(
        dataset_profile=profile,
        ambiguity_report=ambiguity,
        sample_rows=profile.get("sample_rows", []),
        structural_signals=structural_signals,
        llm_enabled=llm_reasoning_allowed,
    )

    # Step 3: Save to state
    state["data_quality_issues"] = structured_issues
    state["profile_ambiguity"] = ambiguity
    state["structural_signals"] = structural_signals
    state["context_inference"] = context
    state.setdefault("analysis_evidence", {})
    state["analysis_evidence"]["cleaning_reasoning_status"] = structured_issues.get("cleaning_reasoning_status")
    state["analysis_evidence"]["cleaning_column_profiles"] = structured_issues.get("column_profiles", {})
    state["analysis_evidence"]["preclean_profile_json"] = profile
    state["analysis_evidence"]["profile_ambiguity"] = ambiguity
    state["analysis_evidence"]["structural_signals"] = structural_signals
    state["analysis_evidence"]["context_inference"] = context

    print(
        json.dumps(
            {
                "issue_count": len(structured_issues.get("issues", [])),
                "cleaning_reasoning_status": structured_issues.get("cleaning_reasoning_status"),
                "requires_reasoning": ambiguity.get("requires_reasoning"),
                "ambiguity_reasons": ambiguity.get("reasons", [])[:5],
                "structural_signal_count": len(structural_signals.get("signals", [])),
                "dataset_structure": context.get("dataset_structure"),
                "context_reasoning_status": context.get("reasoning_status"),
            },
            indent=2,
        )
    )

    return state
