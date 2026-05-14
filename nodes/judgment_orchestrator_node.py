from __future__ import annotations

from judgment_orchestrator import JudgmentOrchestrator
from state.state import AnalystState


def judgment_orchestrator_node(state: AnalystState) -> AnalystState:
    evidence = state.setdefault("analysis_evidence", {})
    orchestrator = JudgmentOrchestrator(
        stats_output=evidence.get("tool_results", {}),
        semantic_output={"top_stories": evidence.get("top_stories", [])},
        causal_output={"top_stories": evidence.get("top_stories", [])},
        quality_output={
            "data_validation": state.get("data_validation"),
            "cleaning_validation": state.get("cleaning_validation"),
        },
        story_candidates=evidence.get("top_stories", []) or evidence.get("story_candidates", []),
        decision_candidates=evidence.get("decision_recommendations", []),
        report_context={
            "decision_priority_ranking": evidence.get("decision_priority_ranking", []),
            "decision_recommended_first": evidence.get("decision_recommended_first"),
        },
        domain_context={
            "business_question": state.get("business_question", ""),
            "selected_columns": state.get("selected_columns", []),
        },
    )
    judgment = orchestrator.finalize()
    evidence["judgment_summary"] = judgment

    if not judgment["actionability"]:
        evidence["decision_priority_ranking"] = []
        evidence["decision_recommended_first"] = None

    print("\n=== JUDGMENT ORCHESTRATOR COMPLETE ===")
    print(
        {
            "final_truth_state": judgment["final_truth_state"],
            "global_confidence": judgment["global_confidence"],
            "actionability": judgment["actionability"],
            "recommended_first_action": judgment["recommended_first_action"],
        }
    )
    return state
