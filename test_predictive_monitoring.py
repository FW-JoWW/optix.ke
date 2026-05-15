from __future__ import annotations

import pandas as pd

from predictive.monitoring import detect_data_drift, performance_decay_monitor


def test_detect_data_drift_flags_material_distribution_change() -> None:
    train = pd.DataFrame(
        {
            "feature_a": [10, 11, 12, 10, 11, 12],
            "feature_b": ["east", "east", "west", "east", "west", "west"],
        }
    )
    score = pd.DataFrame(
        {
            "feature_a": [30, 32, 31, 29, 34, 33],
            "feature_b": ["north", "north", "north", "west", "north", "north"],
        }
    )
    drift = detect_data_drift(train, score, ["feature_a", "feature_b"])
    assert drift["average_drift_score"] > 0.35
    assert drift["health_score"] < 70
    assert drift["warnings"]
    print("DRIFT DETECTION OK:", drift["average_drift_score"], drift["warnings"])


def test_performance_decay_monitor_flags_regression_decay() -> None:
    decay = performance_decay_monitor(
        metrics={"r2": 0.31},
        baseline_metrics={"r2": 0.52},
        problem_type="regression",
    )
    assert decay["healthy"] is False
    assert decay["warnings"]
    print("PERFORMANCE DECAY OK:", decay)


if __name__ == "__main__":
    test_detect_data_drift_flags_material_distribution_change()
    test_performance_decay_monitor_flags_regression_decay()
    print("All predictive monitoring tests passed.")
