from state.state import AnalystState
from nodes.data_intake_node import data_intake_node

# Example for CSV
state: AnalystState = {
    "business_question": "How did revenue perform this month?",
    "dataset_path": "data/sales.csv",
    "dataset_profile": None,
    "dataframe": None,
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

print(state["dataframe"].head())