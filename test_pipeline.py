import pandas as pd

from nodes.data_intake_node import data_intake_node
from nodes.dataset_profiler_node import dataset_profiler_node
from nodes.column_semantic_classifier_node import column_semantic_classifier_node
from nodes.intent_parser_node import intent_parser_node
from nodes.column_selection_node import column_selection_node

# NEW NODES
from nodes.numeric_cleaning_node import numeric_cleaning_node
from nodes.row_filter_node import row_filter_node
from nodes.initialize_analysis_evidence_node import initialize_analysis_evidence_node
from nodes.analysis_planner_node import analysis_planner_node
from nodes.tool_executor_node import tool_executor_node
from nodes.visualization_generator_node import visualization_generator_node
from nodes.evidence_interpreter_node import evidence_interpreter_node
from nodes.story_scoring_engine_node import story_scoring_engine_node
from nodes.llm_insight_synthesizer_node import llm_insight_synthesizer_node


def run_pipeline():

    state = {}

    # -----------------------------
    # dataset path
    # -----------------------------
    state["dataset_path"] = "data/data_set.csv"

    # -----------------------------
    # business question
    # -----------------------------
    state["business_question"] = " bmw or audi"

    print("\n==============================")
    print("RUNNING ANALYST PIPELINE TEST")
    print("==============================")

    # -----------------------------
    # DATA PREP + SELECTION
    # -----------------------------

    state = data_intake_node(state)
    state = dataset_profiler_node(state)
    state = column_semantic_classifier_node(state)
    state = intent_parser_node(state)
    #state = column_selection_node(state)

    # -----------------------------
    # ANALYTICS ENGINE
    # -----------------------------   
    state = initialize_analysis_evidence_node(state)
    #state = numeric_cleaning_node(state)
    state = row_filter_node(state)
    # stop if filtering failed
    if state.get("halt_pipeline"):
        print("\n PIPELINE STOPPED:")
        print(state.get("error"))
        return state
    
    state = column_selection_node(state)
    #state = numeric_cleaning_node(state)
    state = analysis_planner_node(state)
    # If filter-only -> skip heavy analysis
    if state.get("intent", {}).get("type") == "filter":
        from nodes.output_mode_node import output_mode_node
        state = output_mode_node(state)
        return state
    state = tool_executor_node(state)
    state = evidence_interpreter_node(state)
    state = story_scoring_engine_node(state)
    state = visualization_generator_node(state)
    state = llm_insight_synthesizer_node(state)

    # -----------------------------
    # OUTPUT DEBUG
    # -----------------------------

    print("\n==============================")
    print("ANALYSIS EVIDENCE")
    print("==============================")

    evidence = state.get("analysis_evidence", {})

    print("\nAnalysis Plan:")
    print(evidence.get("analysis_plan"))

    print("\nTool Results:")
    print(evidence.get("tool_results"))

    print("\nStory Candidates:")
    print(evidence.get("story_candidates"))

    print("\nTop Stories:")
    print(evidence.get("top_stories"))

    print("\nVisualizations:")
    for viz in evidence.get("visualizations", []):
        print(viz)

    print("\nLLM Insights:")
    print(evidence.get("llm_insights"))

    print("\nClarification Questions:")
    print(evidence.get("clarification_questions"))

    return state


if __name__ == "__main__":
    run_pipeline()