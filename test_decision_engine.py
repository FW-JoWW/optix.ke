from __future__ import annotations

from nodes.decision_engine_node import decision_engine_node


def build_state(stories):
    return {
        "business_question": "test",
        "analysis_evidence": {"top_stories": stories},
    }


def test_unit_conversion_no_action() -> None:
    story = {
        "type": "inferential_relationship",
        "insight": "engine_hp and max_power_kw move together",
        "columns": ["engine_hp", "max_power_kw"],
        "relationship_type": "unit_conversion",
        "insight_validity": {"valid": False, "severity": "medium", "missing_ratio": 0.0},
        "causal_evidence": {"grade": "LOW", "score": 10},
        "recommendation_restrictions": ["insight_invalid"],
        "bias_risks": [],
    }
    state = decision_engine_node(build_state([story]))
    top = state["analysis_evidence"]["decision_recommendations"][0]
    assert top["action_type"] == "no_action"
    assert top["priority"]["priority_score"] == 0
    assert top["recommended_action"] == "No action required"
    assert state["analysis_evidence"]["decision_priority_ranking"] == []
    print("CASE 1 OK:", top)


def test_low_causal_experiment_cap() -> None:
    story = {
        "type": "inferential_relationship",
        "insight": "ads and revenue are associated",
        "columns": ["ads", "revenue"],
        "relationship_type": "economic",
        "insight_validity": {"valid": True, "severity": "high", "missing_ratio": 0.0},
        "causal_evidence": {"grade": "LOW", "score": 35},
        "effect_size": {"value": 0.7},
        "value": 0.7,
        "bias_risks": ["confounding"],
        "recommendation_restrictions": ["causal_low_exploratory_only"],
    }
    state = decision_engine_node(build_state([story]))
    top = state["analysis_evidence"]["decision_priority_ranking"][0]
    assert top["action_type"] == "experiment"
    assert top["priority"]["priority_score"] <= 40
    print("CASE 2 OK:", top)


def test_strong_causal_strategic() -> None:
    story = {
        "type": "inferential_relationship",
        "insight": "proven feature increases conversion",
        "columns": ["feature_flag", "conversion_rate"],
        "relationship_type": "behavioral",
        "insight_validity": {"valid": True, "severity": "low", "missing_ratio": 0.0},
        "causal_evidence": {"grade": "STRONG", "score": 92},
        "effect_size": {"value": 0.65},
        "value": 0.65,
        "bias_risks": [],
        "recommendation_restrictions": [],
    }
    state = decision_engine_node(build_state([story]))
    top = state["analysis_evidence"]["decision_priority_ranking"][0]
    assert top["action_type"] == "strategic"
    assert top["priority"]["priority_level"] in {"high", "critical"}
    print("CASE 3 OK:", top)


def test_predictive_story_stays_investigative_but_usable() -> None:
    story = {
        "type": "predictive_model",
        "insight": "predictive model for revenue is ready",
        "column": "revenue",
        "problem_type": "regression",
        "confidence": "high",
        "metrics": {"r2": 0.86, "mae": 15.2, "rmse": 21.1},
        "top_drivers": [{"feature": "marketing_spend", "importance": 0.44}],
        "readiness_warnings": [],
    }
    state = decision_engine_node(build_state([story]))
    top = state["analysis_evidence"]["decision_priority_ranking"][0]
    assert top["action_type"] == "investigation"
    assert top["recommended_action"].lower().startswith("use the model")
    assert top["confidence_in_action"] >= 60
    print("CASE 4 OK:", top)


def test_prescriptive_story_becomes_ranked_optimization() -> None:
    story = {
        "type": "prescriptive_action",
        "insight": "best next action is reallocate budget",
        "column": "revenue",
        "confidence": "moderate",
        "estimated_upside": 12000.0,
        "recommended_actions": [
            {"action": "Reallocate budget toward high-return channels."},
            {"action": "Pilot the reallocation for one quarter."},
        ],
        "scenario_summary": [{"scenario": "increase marketing_spend by 5%", "estimated_effect": 8000.0}],
    }
    state = decision_engine_node(build_state([story]))
    top = state["analysis_evidence"]["decision_priority_ranking"][0]
    assert top["action_type"] == "optimization"
    assert "reallocate budget" in top["recommended_action"].lower()
    assert top["priority"]["priority_level"] in {"medium", "high", "critical"}
    print("CASE 5 OK:", top)


if __name__ == "__main__":
    test_unit_conversion_no_action()
    test_low_causal_experiment_cap()
    test_strong_causal_strategic()
    test_predictive_story_stays_investigative_but_usable()
    test_prescriptive_story_becomes_ranked_optimization()
    print("All decision engine tests passed.")
