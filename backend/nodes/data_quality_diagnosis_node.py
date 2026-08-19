# nodes/data_quality_diagnosis_node.py
import json

from backend.ambiguity_detector import detect_ambiguity
from backend.core.context_inference import infer_context
from backend.analytics.data_profiling import profile_dataset
from backend.core.execution_state import record_evidence_provenance, record_execution_event
from backend.analytics.structural_signal_extraction import extract_structural_signals
from backend.state.state import AnalystState
from backend.utils.cleaning_recommender import recommend_cleaning_issues
from backend.utils.issue_detector import detect_issues

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
    record_execution_event(
        state,
        phase="data_quality",
        message="Checking dataset shape, quality signals, and evidence scope.",
        progress=15,
        operation="data_profile",
        evidence_scope=profile.get("evidence_scope") or profile.get("profiling_mode") or "exact",
    )
    ambiguity = detect_ambiguity(profile)
    structural_signals = extract_structural_signals(profile)
    llm_reasoning_allowed = bool(state.get("enable_llm_reasoning", True) and not state.get("disable_llm_reasoning"))

    # Step 1: Rule-based detection
    record_execution_event(
        state,
        phase="data_quality",
        message="Detecting data quality issues with deterministic checks.",
        progress=30,
        operation="issue_detection",
        evidence_scope="exact" if profile.get("evidence_scope") == "exact" else "sampled",
    )
    detected_issues = detect_issues(df)

    # Step 2: constrained reasoning over column context
    record_execution_event(
        state,
        phase="data_quality",
        message="Interpreting quality findings and building cleaning recommendations.",
        progress=55,
        operation="cleaning_recommendation",
    )
    structured_issues = recommend_cleaning_issues(
        detected_issues,
        df,
        base_profiles=profile.get("columns", {}),
        allow_llm=llm_reasoning_allowed,
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
    record_evidence_provenance(
        state,
        "dataset_profile",
        scope=profile.get("evidence_scope") or profile.get("profiling_mode") or "exact",
        source="profile_dataset",
        verified=profile.get("evidence_scope") == "exact",
        method=profile.get("provenance", {}).get("method"),
    )
    record_evidence_provenance(
        state,
        "issue_detection",
        scope=detected_issues.get("evidence_scope") or "exact",
        source="detect_issues",
        verified=detected_issues.get("evidence_scope") == "exact",
        method=detected_issues.get("provenance", {}).get("method"),
    )
    record_evidence_provenance(
        state,
        "cleaning_recommendations",
        scope=structured_issues.get("evidence_scope") if isinstance(structured_issues, dict) else "exact",
        source="recommend_cleaning_issues",
        verified=(structured_issues.get("evidence_scope") if isinstance(structured_issues, dict) else "exact") == "exact",
        method=(structured_issues.get("provenance") or {}).get("method") if isinstance(structured_issues, dict) else None,
    )
    record_execution_event(
        state,
        phase="data_quality",
        message="Data quality analysis complete.",
        progress=100,
        operation="data_quality_complete",
        evidence_scope=profile.get("evidence_scope") or "exact",
        status="complete",
    )

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


