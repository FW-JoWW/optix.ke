from state.state import AnalystState


def initialize_analysis_evidence_node(state: AnalystState) -> AnalystState:
    """
    Initializes the analysis_evidence structure used by all
    downstream analytical nodes.

    This prevents missing-key errors and guarantees a consistent
    evidence structure across the pipeline.
    """

    print("\n=== INITIALIZING ANALYSIS EVIDENCE STRUCTURE ===")

    if "analysis_evidence" not in state or state["analysis_evidence"] is None:

        state["analysis_evidence"] = {
            "analysis_plan": [],
            "tool_results": {},
            "visualizations": [],
            "story_candidates": [],
            "top_stories": [],
            "llm_insights": "",
            "clarification_questions": []
        }

    else:
        # Ensure missing keys are added if partial state exists
        defaults = {
            "analysis_plan": [],
            "tool_results": {},
            "visualizations": [],
            "story_candidates": [],
            "top_stories": [],
            "llm_insights": "",
            "clarification_questions": []
        }

        for key, value in defaults.items():
            state["analysis_evidence"].setdefault(key, value)

    print("Analysis evidence structure ready.")

    return state