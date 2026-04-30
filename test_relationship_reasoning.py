import math
from typing import Dict

import numpy as np
import pandas as pd

from nodes.analysis_planner_node import analysis_planner_node
from nodes.evidence_interpreter_node import evidence_interpreter_node
from nodes.initialize_analysis_evidence_node import initialize_analysis_evidence_node
from nodes.intent_parser_node import intent_parser_node
from nodes.story_scoring_engine_node import story_scoring_engine_node
from nodes.tool_executor_node import tool_executor_node
from nodes.validation_repair_node import validation_repair_node


def build_synthetic_causal_dataframe(n_days: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    seasonality = np.sin(np.arange(n_days) / 12.0) * 20
    promo = (np.arange(n_days) % 14 < 4).astype(int)
    ads = 40 + seasonality * 0.7 + promo * 12 + rng.normal(0, 3, n_days)
    lagged_ads = np.roll(ads, 2)
    lagged_ads[:2] = ads[:2]
    sales = 120 + seasonality * 1.2 + promo * 20 + lagged_ads * 1.8 + rng.normal(0, 8, n_days)
    region = np.where(np.arange(n_days) % 3 == 0, "North", np.where(np.arange(n_days) % 3 == 1, "South", "West"))
    cohort = np.where(promo == 1, "campaign", "baseline")
    return pd.DataFrame(
        {
            "date": dates,
            "ads": ads,
            "sales": sales,
            "seasonality_index": seasonality,
            "promotion_flag": promo,
            "region": region,
            "cohort": cohort,
        }
    )


def build_dataset_profile(df: pd.DataFrame) -> dict:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    datetime_columns = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    categorical_columns = [col for col in df.columns if col not in numeric_columns and col not in datetime_columns]
    columns: Dict[str, dict] = {}
    for col in df.columns:
        series = df[col]
        if col in numeric_columns:
            inferred_type = "numeric"
            numeric_like_ratio = 1.0
            datetime_like_ratio = 0.0
        elif col in datetime_columns:
            inferred_type = "datetime"
            numeric_like_ratio = 0.0
            datetime_like_ratio = 1.0
        else:
            inferred_type = "categorical"
            numeric_like_ratio = 0.0
            datetime_like_ratio = 0.0
        columns[col] = {
            "inferred_type": inferred_type,
            "missing_ratio": float(series.isna().mean()),
            "unique_count": int(series.nunique(dropna=True)),
            "numeric_like_ratio": numeric_like_ratio,
            "datetime_like_ratio": datetime_like_ratio,
        }
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "identifier_columns": [],
        "unique_counts": {col: int(df[col].nunique(dropna=True)) for col in df.columns},
        "columns": columns,
    }


def build_column_registry(df: pd.DataFrame, dataset_profile: dict) -> dict:
    registry = {}
    for col in df.columns:
        if col in dataset_profile["numeric_columns"]:
            semantic_role = "numeric_measure"
        elif col in dataset_profile["datetime_columns"]:
            semantic_role = "timestamp"
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
        "dataset_path": "synthetic://causal_relationship",
        "dataframe": df.copy(),
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
        "context_inference": {"column_roles": {col: meta["semantic_role"] for col, meta in build_column_registry(df, dataset_profile).items()}},
        "relationship_signals": {"relationships": [], "derived_columns": []},
        "structural_signals": {},
        "cleaning_constraints": {},
        "selected_columns": [],
    }


def run_query(query: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("QUERY:", query)
    state = create_state(df, query)
    state = intent_parser_node(state)
    state = validation_repair_node(state)
    state["analysis_dataset"] = df.copy()
    state["raw_analysis_dataset"] = df.copy()
    state = initialize_analysis_evidence_node(state)
    state = analysis_planner_node(state)
    state = tool_executor_node(state)
    state = evidence_interpreter_node(state)
    state = story_scoring_engine_node(state)

    evidence = state.get("analysis_evidence", {})
    print("\n[ANALYSIS PLAN]")
    print(evidence.get("analysis_plan"))
    print("\n[TOOL RESULTS]")
    print(evidence.get("tool_results"))
    print("\n[TOP STORIES]")
    print(evidence.get("top_stories"))


if __name__ == "__main__":
    dataframe = build_synthetic_causal_dataframe()
    queries = [
        "did ads cause sales growth",
        "what is the relationship between ads and sales",
    ]
    for query in queries:
        run_query(query, dataframe)
