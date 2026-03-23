# analyst_graph.py
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
from nodes.column_selection_node import column_selection_node
from nodes.initialize_analysis_evidence_node import initialize_analysis_evidence_node
from nodes.analysis_planner_node import analysis_planner_node
from nodes.interaction_node import interaction_node
from nodes.tool_executor_node import tool_executor_node
from nodes.visualization_generator_node import visualization_generator_node
from nodes.evidence_interpreter_node import evidence_interpreter_node
from nodes.story_scoring_engine_node import story_scoring_engine_node
from nodes.eda_node import eda_node
from nodes.statistical_analysis_node import statistical_analysis_node
from nodes.insight_generator_node import insight_generator_node
from nodes.insight_synthesis_node import insight_synthesis_node
from nodes.llm_insight_synthesizer_node import llm_insight_synthesizer_node
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
builder.add_node("column_selection", column_selection_node)
builder.add_node("initialize_analysis_evidence", initialize_analysis_evidence_node)
builder.add_node("analysis_planner", analysis_planner_node)
builder.add_node("interaction", interaction_node)
builder.add_node("tool_executor", tool_executor_node)
builder.add_node("visualization", visualization_generator_node)
builder.add_node("evidence_interpreter", evidence_interpreter_node)
builder.add_node("story_scoring", story_scoring_engine_node)
#builder.add_node("eda", eda_node)
#builder.add_node("statistical_analysis", statistical_analysis_node)
#builder.add_node("insight_generator", insight_generator_node)
#builder.add_node("insight_synthesis", insight_synthesis_node)
builder.add_node("llm_insight_synthesizer", llm_insight_synthesizer_node)
builder.add_node("report", report_node)

# Set entry point
builder.set_entry_point("data_intake")

# Connect nodes
#builder.add_edge("data_intake", "data_quality")
#builder.add_edge("data_quality", "data_validation")
builder.add_edge("data_intake", "data_quality_diagnosis")
builder.add_edge("data_quality_diagnosis", "cleaning_strategy_planner")
builder.add_edge("cleaning_strategy_planner", "cleaning_execution")
builder.add_edge("cleaning_execution", "cleaning_audit")
builder.add_edge("cleaning_audit", "data_validation")
builder.add_edge("data_validation", "dataset_profiler")
builder.add_edge("dataset_profiler", "column_semantic_classifier")
builder.add_edge("column_semantic_classifier", "column_selection")
builder.add_edge("column_selection", "initialize_analysis_evidence")
builder.add_edge("initialize_analysis_evidence", "analysis_planner")
builder.add_edge("analysis_planner", "interaction")
builder.add_edge("interaction", "tool_executer")
builder.add_edge("tool_executor", "evidence_interpreter")
builder.add_edge("evidence_interpreter", "story_scoring")
builder.add_edge("story_scoring", "visualization")
builder.add_edge("visualization", "llm_insight_synthesizer")
builder.add_edge("llm_insight_synthesizer", "report")
builder.add_edge("report", END)

# Compile graph
graph = builder.compile()

