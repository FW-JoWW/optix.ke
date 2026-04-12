from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from state.state import AnalystState
from tools.categorical_analysis import (
    CategoricalAnalysisConfig,
    analyze_categorical_columns,
    categorical_analysis_config_to_dict,
    detect_categorical_columns,
)


def categorical_analysis_node(state: AnalystState) -> AnalystState:
    """
    Runs categorical analysis as an additive pipeline step.
    Results are stored under state["analysis_evidence"]["categorical_analysis"].
    """
    df: pd.DataFrame | None = state.get("analysis_dataset")
    if df is None:
        df = state.get("cleaned_data")
    if df is None:
        df = state.get("dataframe")

    evidence = state.setdefault("analysis_evidence", {})

    if df is None:
        evidence["categorical_analysis"] = {"error": "No dataset available."}
        return state

    raw_config: Dict[str, Any] = state.get("categorical_analysis_config", {}) or {}
    config = CategoricalAnalysisConfig(**{
        key: value
        for key, value in raw_config.items()
        if key in CategoricalAnalysisConfig.__dataclass_fields__
    })

    dataset_profile = state.get("dataset_profile", {}) or {}
    selected_columns = state.get("selected_columns", []) or list(df.columns)
    numeric_columns = [
        col for col in dataset_profile.get("numeric_columns", [])
        if col in df.columns and col in selected_columns
    ]
    categorical_columns = [
        col for col in dataset_profile.get("categorical_columns", [])
        if col in df.columns and col in selected_columns
    ]

    if not categorical_columns:
        evidence["categorical_analysis"] = {}
        evidence["categorical_analysis_meta"] = {
            "config": categorical_analysis_config_to_dict(config),
            "categorical_columns": [],
            "numeric_columns": numeric_columns,
        }
        print("\n=== CATEGORICAL ANALYSIS COMPLETE ===")
        print("Categorical columns analyzed: []")
        return state

    results = analyze_categorical_columns(
        df,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        config=config,
    )

    evidence["categorical_analysis"] = results
    evidence["categorical_analysis_meta"] = {
        "config": categorical_analysis_config_to_dict(config),
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
    }

    print("\n=== CATEGORICAL ANALYSIS COMPLETE ===")
    print("Categorical columns analyzed:", categorical_columns)

    return state
