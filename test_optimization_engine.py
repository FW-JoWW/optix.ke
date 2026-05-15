from __future__ import annotations

from prescriptive.optimization_engine import optimize_actions, parse_constraints


def test_parse_constraints_extracts_common_limits() -> None:
    constraints = parse_constraints(
        "maximize revenue under budget 5000 with max price increase 2 and low risk"
    )
    assert constraints["budget_cap"] == 5000.0
    assert constraints["max_price_increase_pct"] == 2.0
    assert constraints["risk_tolerance"] == "low"
    print("CONSTRAINT PARSING OK:", constraints)


def test_optimize_actions_penalizes_high_risk_and_constraint_violations() -> None:
    predictive_result = {
        "confidence": {"score": 76},
        "truthfulness_flags": ["Use only as a directional signal."],
    }
    scenarios = [
        {
            "scenario": "high_risk_high_reward_option",
            "action": "Test a 3% price increase on the strongest segment.",
            "estimated_effect": 0.18,
            "estimated_range": {"low": 0.08, "high": 0.24},
            "risk_level": "high",
            "feasibility_level": "medium",
            "affected_segments": ["segment_a"],
        },
        {
            "scenario": "balanced_option",
            "action": "Shift 12% budget from low ROI segment A to segment B.",
            "estimated_effect": 0.12,
            "estimated_range": {"low": 0.05, "high": 0.16},
            "risk_level": "medium",
            "feasibility_level": "high",
            "affected_segments": ["segment_a", "segment_b"],
        },
    ]
    optimization = optimize_actions(
        predictive_result,
        scenarios,
        "maximize revenue with low risk and max price increase 2",
    )
    ranked = optimization["ranked_actions"]
    assert ranked
    assert ranked[0]["scenario"] == "balanced_option"
    assert any("risk tolerance" in reason.lower() for reason in ranked[1]["constraint_explanations"])
    assert any("price move exceeds" in reason.lower() for reason in ranked[1]["constraint_explanations"])
    print("OPTIMIZATION ENGINE OK:", ranked[0], ranked[1])


if __name__ == "__main__":
    test_parse_constraints_extracts_common_limits()
    test_optimize_actions_penalizes_high_risk_and_constraint_violations()
    print("All optimization engine tests passed.")
