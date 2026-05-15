from __future__ import annotations

import numpy as np
import pandas as pd

from predictive.runtime import apply_runtime_optimizations, determine_runtime_mode


def test_runtime_mode_detects_exploratory_from_question() -> None:
    mode = determine_runtime_mode({"business_question": "run a quick exploratory forecast"})
    assert mode == "exploratory"
    print("RUNTIME MODE OK:", mode)


def test_runtime_optimization_samples_and_prunes_high_cardinality_columns() -> None:
    rows = 25050
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "target": rng.normal(100, 5, rows),
            "stable_numeric": rng.normal(20, 2, rows),
            "quasi_id": [f"id_{idx}" for idx in range(rows)],
        }
    )
    optimized, report = apply_runtime_optimizations(
        df=df,
        target_column="target",
        date_column="date",
        runtime_mode="exploratory",
    )
    assert report["sampled"] is True
    assert len(optimized) == 20000
    assert "quasi_id" in report["dropped_columns"]
    assert "target" in optimized.columns
    print("RUNTIME OPTIMIZATION OK:", report)


if __name__ == "__main__":
    test_runtime_mode_detects_exploratory_from_question()
    test_runtime_optimization_samples_and_prunes_high_cardinality_columns()
    print("All runtime optimization tests passed.")
