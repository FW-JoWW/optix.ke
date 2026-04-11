from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from tools.categorical_analysis import (
    CategoricalAnalysisConfig,
    analyze_categorical_columns,
    detect_categorical_columns,
)


def categorical_analysis_tool(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_config = CategoricalAnalysisConfig(**{
        key: value
        for key, value in (config or {}).items()
        if key in CategoricalAnalysisConfig.__dataclass_fields__
    })

    detected_categorical = detect_categorical_columns(df, resolved_config)
    target_columns = columns or detected_categorical
    target_categorical = [col for col in target_columns if col in detected_categorical]
    numeric_columns = [col for col in df.select_dtypes(include="number").columns if col not in target_categorical]

    return {
        "tool": "categorical_analysis",
        "columns": target_categorical,
        "results": analyze_categorical_columns(
            df,
            numeric_columns=numeric_columns,
            categorical_columns=target_categorical,
            config=resolved_config,
        ),
    }
