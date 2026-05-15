from __future__ import annotations

import json
from pathlib import Path

from predictive import recommendation_memory


def test_memory_store_and_calibration() -> None:
    original_path = recommendation_memory.MEMORY_PATH
    temp_path = Path("data/test_recommendation_memory.json")
    if temp_path.exists():
        temp_path.unlink()
    recommendation_memory.MEMORY_PATH = temp_path
    try:
        recommendation_memory.store_recommendation_snapshot(
            {
                "target": "revenue",
                "predicted_uplift": 0.20,
                "realized_uplift": 0.12,
                "recommended_action": "Shift spend to high ROI segment.",
            }
        )
        recommendation_memory.store_recommendation_snapshot(
            {
                "target": "revenue",
                "predicted_uplift": 0.15,
                "realized_uplift": 0.10,
                "recommended_action": "Test tighter pricing guardrails.",
            }
        )
        history = recommendation_memory.load_memory()
        calibration = recommendation_memory.calibrate_from_memory("revenue")
        assert len(history) == 2
        assert calibration["evidence_count"] == 2
        assert calibration["memory_adjustment"] < 0.05
        assert "calibrated" in calibration["note"].lower()
        print("RECOMMENDATION MEMORY OK:", json.dumps(calibration, indent=2))
    finally:
        recommendation_memory.MEMORY_PATH = original_path
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    test_memory_store_and_calibration()
    print("All recommendation memory tests passed.")
