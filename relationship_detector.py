from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _pairwise_correlation(df: pd.DataFrame, numeric_columns: List[str]) -> List[Dict[str, Any]]:
    relationships: List[Dict[str, Any]] = []

    for left, right in combinations(numeric_columns, 2):
        subset = df[[left, right]].dropna()
        if len(subset) < 3:
            continue
        corr = subset[left].corr(subset[right])
        if pd.isna(corr):
            continue
        if abs(float(corr)) >= 0.7:
            relationships.append(
                {
                    "type": "correlation",
                    "source_columns": [left, right],
                    "strength": round(float(corr), 4),
                }
            )
    return relationships


def _derived_relationships(df: pd.DataFrame, numeric_columns: List[str]) -> List[Dict[str, Any]]:
    relationships: List[Dict[str, Any]] = []

    for target, left, right in combinations(numeric_columns, 3):
        subset = df[[target, left, right]].dropna()
        if len(subset) < 3:
            continue

        multiplied = subset[left] * subset[right]
        added = subset[left] + subset[right]

        for operator, candidate in (("multiply", multiplied), ("add", added)):
            if candidate.std(ddof=0) == 0:
                continue
            corr = candidate.corr(subset[target])
            if pd.notna(corr) and abs(float(corr)) >= 0.97:
                relationships.append(
                    {
                        "type": "derived_metric",
                        "target_column": target,
                        "source_columns": [left, right],
                        "operator": operator,
                        "strength": round(float(corr), 4),
                    }
                )

    return relationships


def detect_relationships(df: pd.DataFrame, dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
    numeric_columns = [
        col for col, info in dataset_profile.get("columns", {}).items()
        if info.get("inferred_type") == "numeric"
    ]
    numeric_df = df.copy()
    for col in numeric_columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

    correlations = _pairwise_correlation(numeric_df, numeric_columns)
    derived = _derived_relationships(numeric_df, numeric_columns)

    return {
        "relationships": correlations + derived,
        "derived_columns": [item["target_column"] for item in derived],
    }
