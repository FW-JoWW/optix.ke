import os
import pprint

import pandas as pd

from graph.analyst_graph import graph
from state.state import AnalystState


def load_default_dataframe(dataset_path: str) -> pd.DataFrame:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return pd.read_csv(dataset_path)


print("\n===== DATA ANALYST AGENT =====\n")

question = input("Enter your business question:\n> ").strip()
mode = input("\nChoose mode (autonomous / guided / collaborative):\n> ").strip().lower() or "autonomous"

dataset_path = "data/data_set.csv"
df = load_default_dataframe(dataset_path)

state: AnalystState = {
    "business_question": question,
    "dataset_path": dataset_path,
    "dataframe": df,
    "mode": mode,
    "enable_llm_reasoning": False,
    "disable_llm_reasoning": True,
    "analysis_evidence": {},
}

print("\n[Agent] Starting analysis...\n")

final_state = graph.invoke(state)
evidence = final_state.get("analysis_evidence", {})

print("\n===== ANALYSIS PLAN =====")
pprint.pprint(evidence.get("analysis_plan"))

print("\n===== TOOL RESULTS =====")
pprint.pprint(evidence.get("tool_results"))

print("\n===== STORY CANDIDATES =====")
pprint.pprint(evidence.get("story_candidates"))

print("\n===== TOP STORIES =====")
pprint.pprint(evidence.get("top_stories"))

print("\n===== CLARIFICATION QUESTIONS =====")
pprint.pprint(final_state.get("clarification_questions") or evidence.get("clarification_questions"))

print("\n===== LLM INSIGHTS =====")
pprint.pprint(final_state.get("llm_insights") or evidence.get("llm_insights"))

print("\n===== FINAL REPORT =====")
print(final_state.get("final_report", "No report generated"))
