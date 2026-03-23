import pandas as pd
import pprint

from state.state import AnalystState
from graph.analyst_graph import graph


print("\n===== DATA ANALYST AGENT =====\n")

# Ask user for business question
question = input("Enter your business question:\n> ")

# Ask execution mode
mode = input("\nChoose mode (autonomous / guided / collaborative):\n> ").strip().lower()

# Load dataset
dataset_path = "data/data_set.csv"
df = pd.read_csv(dataset_path)


# Initial agent state
state: AnalystState = {
    "business_question": question,
    "dataset_path": dataset_path,
    "dataframe": df,
    "mode": mode
}


print("\n[Agent] Starting analysis...\n")


# Run LangGraph workflow
final_state = graph.invoke(state)


print("\n===== FINAL INSIGHTS =====")
pprint.pprint(final_state.get("insights"))

print("\n===== CLARIFICATION QUESTIONS =====")
pprint.pprint(final_state.get("clarification_questions"))

print("\n===== LLM INSIGHTS =====")
pprint.pprint(final_state.get("llm_insights"))

print("\n===== FINAL REPORT =====")
pprint.pprint(final_state.get("final_report"))

