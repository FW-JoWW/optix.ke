# nodes/analyst_graph.py
from langgraph.graph import StateGraph, END
from state.state import AnalystState

# Import nodes
from nodes.data_intake_node import data_intake_node
from nodes.data_quality_diagnosis_node import data_quality_diagnosis_node
from nodes.cleaning_strategy_planner_node import cleaning_strategy_planner_node
from nodes.cleaning_execution_node import cleaning_execution_node
from nodes.cleaning_audit_node import cleaning_audit_node
from nodes.data_validation_node import data_validation_node
from nodes.dataset_profiler_node import dataset_profiler_node
from nodes.column_semantic_classifier_node import column_semantic_classifier_node
from nodes.relationship_detector_node import relationship_detector_node
from nodes.intent_parser_node import intent_parser_node
from nodes.row_filter_node import row_filter_node
from nodes.column_selection_node import column_selection_node
from nodes.initialize_analysis_evidence_node import initialize_analysis_evidence_node
from nodes.categorical_analysis_node import categorical_analysis_node
from nodes.output_mode_node import output_mode_node
from nodes.analysis_planner_node import analysis_planner_node
from nodes.interaction_node import interaction_node
from nodes.tool_executor_node import tool_executor_node
from nodes.eda_node import eda_node
from nodes.evidence_interpreter_node import evidence_interpreter_node
from nodes.story_scoring_engine_node import story_scoring_engine_node
from nodes.visualization_generator_node import visualization_generator_node
from nodes.llm_insight_synthesizer_node import llm_insight_synthesizer_node
from nodes.decision_engine_node import decision_engine_node
from nodes.report_node import report_node

# Build graph
builder = StateGraph(AnalystState)

# Add nodes
builder.add_node("data_intake", data_intake_node)
builder.add_node("data_quality_diagnosis", data_quality_diagnosis_node)
builder.add_node("cleaning_strategy_planner", cleaning_strategy_planner_node)
builder.add_node("cleaning_execution", cleaning_execution_node)
builder.add_node("cleaning_audit", cleaning_audit_node)
builder.add_node("data_validation", data_validation_node)

builder.add_node("dataset_profiler", dataset_profiler_node)
builder.add_node("column_semantic_classifier", column_semantic_classifier_node)
builder.add_node("relationship_detector", relationship_detector_node)

builder.add_node("intent_parser", intent_parser_node)
builder.add_node("row_filter", row_filter_node)
builder.add_node("column_selection", column_selection_node)
builder.add_node("initialize_analysis_evidence", initialize_analysis_evidence_node)
builder.add_node("categorical_analysis", categorical_analysis_node)

builder.add_node("output_mode", output_mode_node)

builder.add_node("analysis_planner", analysis_planner_node)
builder.add_node("interaction", interaction_node)
#builder.add_node("eda", eda_node)
builder.add_node("tool_executor", tool_executor_node)
builder.add_node("evidence_interpreter", evidence_interpreter_node)
builder.add_node("story_scoring", story_scoring_engine_node)
builder.add_node("visualization", visualization_generator_node)
builder.add_node("llm_insight_synthesizer", llm_insight_synthesizer_node)
builder.add_node("decision_engine", decision_engine_node)
builder.add_node("report", report_node)

# Set entry point
builder.set_entry_point("data_intake")

# Connect nodes
builder.add_edge("data_intake", "data_quality_diagnosis")
builder.add_edge("data_quality_diagnosis", "cleaning_strategy_planner")
builder.add_edge("cleaning_strategy_planner", "cleaning_execution")
builder.add_edge("cleaning_execution", "cleaning_audit")
builder.add_edge("cleaning_audit", "data_validation")

builder.add_edge("data_validation", "dataset_profiler")
builder.add_edge("dataset_profiler", "column_semantic_classifier")
builder.add_edge("column_semantic_classifier", "relationship_detector")
builder.add_edge("relationship_detector", "intent_parser")

builder.add_edge("intent_parser", "row_filter")
builder.add_edge("row_filter", "column_selection")
builder.add_edge("column_selection", "initialize_analysis_evidence")

def route_after_intent(state: AnalystState):
    intent = state.get("intent", {})
    intent_type = intent.get("type")

    if not intent:
        return "analysis_planner"

    if intent.get("wants_analysis"):
        return "analysis_planner"

    if intent_type == "filter" and not intent.get("aggregation"):
        return "output_mode"

    if intent_type in [None, "unknown", "exploration"]:
        return "analysis_planner"

    return "analysis_planner"

builder.add_edge("initialize_analysis_evidence", "categorical_analysis")

builder.add_conditional_edges(
    "categorical_analysis", 
    route_after_intent,
    {
       "output_mode": "output_mode",
       "analysis_planner": "analysis_planner",
    }
)

builder.add_edge("output_mode", END)

#builder.add_edge("initialize_analysis_evidence", "analysis_planner")

builder.add_edge("analysis_planner", "interaction")


def route_after_interaction(state: AnalystState):
    if state.get("awaiting_user"):
        return "end"
    return "tool_executor"


builder.add_conditional_edges(
    "interaction",
    route_after_interaction,
    {
        "tool_executor": "tool_executor",
        "end": END,
    }
)

builder.add_edge("tool_executor", "evidence_interpreter")
builder.add_edge("evidence_interpreter", "story_scoring")
builder.add_edge("story_scoring", "visualization")
builder.add_edge("visualization", "llm_insight_synthesizer")
builder.add_edge("llm_insight_synthesizer", "decision_engine")
builder.add_edge("decision_engine", "report")
builder.add_edge("report", END)

# Compile graph
graph = builder.compile()

