from state.state import AnalystState
from nodes.data_intake_node import data_intake_node
from nodes.profile_node import profiling_node

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

# Step 1: load dataset
state = data_intake_node(state)

# Step 2: profile dataset
state = profiling_node(state)

print("\nFULL PROFILE OBJECT:")
print(state["dataset_profile"])