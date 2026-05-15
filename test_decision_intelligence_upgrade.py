from __future__ import annotations

from predictive.confidence import calibrate_confidence
from prescriptive.prescriptive_engine import run_prescriptive_analysis
from prescriptive.scenario_simulator import simulate_scenarios


def _predictive_payload() -> dict:
    return {
        "tool": "predictive_analysis",
        "problem_type": "regression",
        "target_column": "payment_value_total",
        "chosen_model": "random_forest_regressor",
        "top_drivers": [
            {"feature": "price", "importance": 0.6676},
            {"feature": "freight_value", "importance": 0.1330},
            {"feature": "seller_state", "importance": 0.071},
            {"feature": "review_score_avg", "importance": 0.052},
        ],
        "metrics": {"values": {"mae": 29.4, "rmse": 65.0, "r2": 0.94}},
        "confidence_level": "high",
        "confidence": {
            "score": 89,
            "label": "high",
            "explanation": "Large sample size and strong validation support the prediction, but operations still require caution.",
            "factors": {},
        },
        "validation_summary": {
            "driver_diagnostics": {
                "top_driver_share": 0.6676,
                "driver_concentration_score": 0.48,
                "signal_diversity": "low",
                "warnings": ["Predictive signal is heavily dominated by one feature."],
            }
        },
        "truthfulness_flags": [
            "Feature dominance is strong, so intervention confidence should stay below predictive fit confidence."
        ],
        "no_reliable_recommendation": False,
    }


def test_recommendation_diversity() -> None:
    scenarios = simulate_scenarios(_predictive_payload(), objective="maximize")
    actions = [item["action"] for item in scenarios]
    assert len(set(actions)) >= 3
    assert scenarios[0]["scenario"] == "growth_option"
    assert scenarios[1]["scenario"] == "safest_option"
    print("RECOMMENDATION DIVERSITY OK:", actions)


def test_feature_dominance_downgrades_confidence() -> None:
    confidence = calibrate_confidence(
        problem_type="regression",
        sample_size=1200,
        test_metrics={"r2": 0.94, "mae": 10.0, "rmse": 14.0},
        baseline_metrics={"r2": 0.40, "mae": 22.0, "rmse": 29.0},
        cv_summary={"mean": {"r2": 0.89}, "std": {"r2": 0.03}},
        diagnostics={"overfit_gap": 0.04, "residual_skew": 0.2, "relative_interval_width": 0.6},
        readiness_warnings=[],
        weak_model=False,
        drift_summary={"average_drift_score": 0.18},
        feature_diagnostics={
            "top_driver_share": 0.72,
            "driver_concentration_score": 0.55,
        },
        scenario_uncertainty={"spread_ratio": 0.9},
    )
    assert confidence["score"] < 90
    assert "dominant driver" in confidence["explanation"].lower()
    print("CONFIDENCE DOWNGRADE OK:", confidence)


def test_causal_safety_and_reliability() -> None:
    result = run_prescriptive_analysis(
        _predictive_payload(),
        "predict payment_value_total and recommend optimization actions",
    )
    first = result["recommended_actions"][0]
    assert first["safety_grade"] in {"guarded", "controlled"}
    assert first["monitoring_kpis"]
    assert first["downside_risks"]
    assert first["failure_conditions"]
    assert result["operational_confidence"]["score"] <= result["confidence"]["score"]
    print("CAUSAL SAFETY OK:", first["safety_grade"], result["operational_confidence"])


def test_narrative_duplication_prevention() -> None:
    scenarios = simulate_scenarios(_predictive_payload(), objective="maximize")
    action_signatures = [item["action"].split(".")[0].strip().lower() for item in scenarios]
    assert len(set(action_signatures)) == len(action_signatures)
    print("NARRATIVE DEDUPLICATION OK:", action_signatures)


if __name__ == "__main__":
    test_recommendation_diversity()
    test_feature_dominance_downgrades_confidence()
    test_causal_safety_and_reliability()
    test_narrative_duplication_prevention()
    print("All decision intelligence upgrade tests passed.")
