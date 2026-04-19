from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from data_profiling import profile_dataset


def validate_cleaning(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
) -> Dict[str, Any]:
    before_profile = profile_dataset(before_df)
    after_profile = profile_dataset(after_df)

    row_change = len(after_df) - len(before_df)
    row_loss_ratio = abs(row_change) / max(len(before_df), 1)
    null_before = before_df.isna().sum().to_dict()
    null_after = after_df.isna().sum().to_dict()

    distribution_shifts = {}
    for col, info in before_profile.get("columns", {}).items():
        if col not in after_df.columns:
            continue
        before_summary = info.get("distribution_summary") or {}
        after_summary = after_profile.get("columns", {}).get(col, {}).get("distribution_summary") or {}
        if before_summary and after_summary:
            distribution_shifts[col] = {
                "mean_delta": round(float(after_summary.get("mean", 0.0) - before_summary.get("mean", 0.0)), 4),
                "std_delta": round(float(after_summary.get("std", 0.0) - before_summary.get("std", 0.0)), 4),
            }

    anomalies = []
    if row_loss_ratio > 0.1:
        anomalies.append("large_row_count_change")
    if list(before_df.columns) != list(after_df.columns):
        anomalies.append("schema_changed")

    return {
        "row_count_before": int(len(before_df)),
        "row_count_after": int(len(after_df)),
        "row_count_delta": int(row_change),
        "row_loss_ratio": round(float(row_loss_ratio), 4),
        "null_value_consistency": {
            col: {
                "before": int(null_before.get(col, 0)),
                "after": int(null_after.get(col, 0)),
            }
            for col in after_df.columns
        },
        "distribution_shifts": distribution_shifts,
        "schema_stable": list(before_df.columns) == list(after_df.columns),
        "anomalies": anomalies,
        "after_profile": after_profile,
    }
