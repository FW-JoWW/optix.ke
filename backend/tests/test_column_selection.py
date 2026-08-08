from backend.state.state import AnalystState
from backend.nodes.data_intake_node import data_intake_node
from backend.nodes.profile_node import profiling_node
from backend.nodes.column_selection_node import column_selection_node

state: AnalystState = {
    "business_question": "Why did revenue decrease despite higher ad spending?",
    "dataset_path": "backend/data/sales.csv",
    "dataset_profile": None,
    "dataframe": None,
    "relevant_columns": None,
    "data_validation": None,
    "eda_results": None,
    "statistical_results": None,
    "insights": None,
    "clarification_questions": [],
    "final_report": None,
    "cleaned_data": None,
    "stat_test": None
}

state = data_intake_node(state)
state = profiling_node(state)
state = column_selection_node(state)

print("\nFinal relevant columns:")
print(state["relevant_columns"])

