import pandas as pd

from nodes.analysis_planner_node import analysis_planner_node
from nodes.evidence_interpreter_node import evidence_interpreter_node
from nodes.initialize_analysis_evidence_node import initialize_analysis_evidence_node
from nodes.intent_parser_node import intent_parser_node
from nodes.story_scoring_engine_node import story_scoring_engine_node
from nodes.tool_executor_node import tool_executor_node
from nodes.validation_repair_node import validation_repair_node


DATASET_PATH = "data/Car Dataset 1945-2020.csv"


def build_dataset_profile(df: pd.DataFrame) -> dict:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = [col for col in df.columns if col not in numeric_columns]
    columns = {}
    for col in df.columns:
        series = df[col]
        inferred_type = "numeric" if col in numeric_columns else "categorical"
        columns[col] = {
            "inferred_type": inferred_type,
            "missing_ratio": float(series.isna().mean()),
            "unique_count": int(series.nunique(dropna=True)),
            "numeric_like_ratio": 1.0 if col in numeric_columns else 0.0,
            "datetime_like_ratio": 0.0,
        }
    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": [],
        "identifier_columns": [],
        "unique_counts": {col: int(df[col].nunique(dropna=True)) for col in df.columns},
        "columns": columns,
    }


def build_column_registry(df: pd.DataFrame, dataset_profile: dict) -> dict:
    registry = {}
    for col in df.columns:
        if col in dataset_profile["numeric_columns"]:
            semantic_role = "numeric_measure"
        else:
            semantic_role = "categorical_feature"
        registry[col] = {
            "type": dataset_profile["columns"][col]["inferred_type"],
            "semantic_role": semantic_role,
        }
    return registry


def create_state(df: pd.DataFrame, query: str) -> dict:
    dataset_profile = build_dataset_profile(df)
    return {
        "business_question": query,
        "dataset_path": DATASET_PATH,
        "dataframe": df,
        "cleaned_data": df.copy(),
        "analysis_dataset": df.copy(),
        "raw_analysis_dataset": df.copy(),
        "mode": "autonomous",
        "enable_llm_reasoning": False,
        "disable_llm_reasoning": True,
        "disable_semantic_matcher": True,
        "analysis_evidence": {},
        "intent": {},
        "dataset_profile": dataset_profile,
        "column_registry": build_column_registry(df, dataset_profile),
        "context_inference": {},
        "relationship_signals": {},
        "structural_signals": {},
        "cleaning_constraints": {},
    }


def run_query(query: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("QUERY:", query)

    state = create_state(df, query)
    state = intent_parser_node(state)
    state = validation_repair_node(state)
    state = initialize_analysis_evidence_node(state)
    state = analysis_planner_node(state)
    state = tool_executor_node(state)
    state = evidence_interpreter_node(state)
    state = story_scoring_engine_node(state)

    evidence = state.get("analysis_evidence", {})
    print("\n[COMPUTATION PLAN]")
    print(evidence.get("computation_plan"))
    print("\n[ANALYSIS PLAN]")
    print(evidence.get("analysis_plan"))
    print("\n[TOOL RESULTS]")
    print(evidence.get("tool_results"))
    print("\n[TOP STORIES]")
    print(evidence.get("top_stories"))


if __name__ == "__main__":
    dataframe = pd.read_csv(DATASET_PATH, low_memory=False)
    queries = [
        "what is the relationship between engine_hp and max_power_kw",
        "does rating_name affect engine_hp",
        "what is the relationship between transmission and drive_wheels",
    ]

    for query in queries:
        run_query(query, dataframe)
