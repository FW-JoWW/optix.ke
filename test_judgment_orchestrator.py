from __future__ import annotations

from judgment_orchestrator import JudgmentOrchestrator


def test_unit_conversion_no_action() -> None:
    story = {
        "insight": "engine_hp and max_power_kw move together",
        "score": 0.09,
        "relationship_type": "unit_conversion",
        "insight_validity": {"valid": False, "severity": "medium", "missing_ratio": 0.0},
        "causal_evidence": {"grade": "LOW", "score": 48},
        "bias_risks": [],
    }
    decision = {
        "action_type": "no_action",
        "recommended_action": "No action required",
        "confidence_in_action": 5,
        "possible_actions": ["ignore", "data validation"],
    }
    judgment = JudgmentOrchestrator(
        story_candidates=[story],
        decision_candidates=[decision],
        report_context={"decision_priority_ranking": []},
    ).finalize()
    assert judgment["final_truth_state"] == "deterministic_relationship"
    assert judgment["actionability"] is False
    assert judgment["recommended_first_action"] is None
    print("CASE 1 OK:", judgment)


def test_invalid_insight_blocks_recommendation() -> None:
    story = {
        "insight": "invalid but strong sounding relationship",
        "score": 0.8,
        "relationship_type": "unknown",
        "insight_validity": {"valid": False, "severity": "high", "missing_ratio": 0.0},
        "causal_evidence": {"grade": "MODERATE", "score": 60},
        "bias_risks": [],
    }
    decision = {
        "action_type": "strategic",
        "recommended_action": "full rollout",
        "confidence_in_action": 80,
        "possible_actions": ["full rollout"],
    }
    judgment = JudgmentOrchestrator(
        story_candidates=[story],
        decision_candidates=[decision],
        report_context={"decision_priority_ranking": [decision]},
    ).finalize()
    assert judgment["actionability"] is False
    assert "invalid_insight_but_action_recommended" in judgment["contradictions_found"]
    print("CASE 2 OK:", judgment)


def test_high_missingness_lowers_confidence() -> None:
    story = {
        "insight": "missing data relationship",
        "score": 0.85,
        "relationship_type": "economic",
        "confidence": "high",
        "insight_validity": {"valid": True, "severity": "medium", "missing_ratio": 0.52},
        "causal_evidence": {"grade": "MODERATE", "score": 70},
        "bias_risks": ["missingness_bias"],
    }
    decision = {
        "action_type": "optimization",
        "recommended_action": "controlled rollout",
        "confidence_in_action": 82,
        "possible_actions": ["controlled rollout"],
    }
    judgment = JudgmentOrchestrator(
        story_candidates=[story],
        decision_candidates=[decision],
        report_context={"decision_priority_ranking": [decision]},
    ).finalize()
    assert judgment["global_confidence"] < 70
    assert "high_missingness_but_high_story_confidence" in judgment["contradictions_found"]
    print("CASE 3 OK:", judgment)


def test_contradictory_outputs_suppressed() -> None:
    story = {
        "insight": "deterministic relation",
        "score": 0.9,
        "relationship_type": "unit_conversion",
        "insight_validity": {"valid": False, "severity": "medium", "missing_ratio": 0.0},
        "causal_evidence": {"grade": "LOW", "score": 50},
        "bias_risks": [],
    }
    decision = {
        "action_type": "experiment",
        "recommended_action": "run A/B test",
        "confidence_in_action": 55,
        "possible_actions": ["run A/B test"],
    }
    judgment = JudgmentOrchestrator(
        story_candidates=[story],
        decision_candidates=[decision],
        report_context={"decision_priority_ranking": [decision]},
    ).finalize()
    assert "deterministic_relationship_but_strategy_suggested" in judgment["contradictions_found"] or "deterministic_relationship_but_experiment_suggested" in judgment["contradictions_found"]
    assert "causal_recommendations" in judgment["suppressed_modules"]
    print("CASE 4 OK:", judgment)


def test_valid_causal_evidence_passes_through() -> None:
    story = {
        "insight": "proven feature increases conversion",
        "score": 0.95,
        "relationship_type": "behavioral",
        "insight_validity": {"valid": True, "severity": "low", "missing_ratio": 0.02},
        "causal_evidence": {"grade": "STRONG", "score": 92},
        "bias_risks": [],
    }
    decision = {
        "action_type": "strategic",
        "recommended_action": "full rollout",
        "confidence_in_action": 92,
        "possible_actions": ["full rollout"],
    }
    judgment = JudgmentOrchestrator(
        story_candidates=[story],
        decision_candidates=[decision],
        report_context={"decision_priority_ranking": [decision]},
    ).finalize()
    assert judgment["actionability"] is True
    assert judgment["recommended_first_action"] == "full rollout"
    print("CASE 5 OK:", judgment)


def test_prescriptive_story_becomes_dominant_judgment() -> None:
    predictive_story = {
        "type": "predictive_model",
        "insight": "predictive model for demand is ready",
        "score": 0.88,
        "confidence": "moderate",
        "readiness_warnings": [{"severity": "medium"}],
        "insight_validity": {"valid": True, "severity": "low", "missing_ratio": 0.04},
    }
    prescriptive_story = {
        "type": "prescriptive_action",
        "insight": "best next action is increase stock on top segments",
        "score": 0.83,
        "confidence": "moderate",
        "estimated_upside": 9000.0,
        "insight_validity": {"valid": True, "severity": "low", "missing_ratio": 0.04},
    }
    decision = {
        "action_type": "optimization",
        "recommended_action": "Increase stock on top segments.",
        "confidence_in_action": 72,
        "possible_actions": ["Increase stock on top segments."],
    }
    judgment = JudgmentOrchestrator(
        story_candidates=[predictive_story, prescriptive_story],
        decision_candidates=[decision],
        report_context={"decision_priority_ranking": [decision]},
    ).finalize()
    assert judgment["final_truth_state"] == "prescriptive_recommendation"
    assert judgment["actionability"] is True
    assert judgment["recommended_first_action"] == "Increase stock on top segments."
    print("CASE 6 OK:", judgment)


if __name__ == "__main__":
    test_unit_conversion_no_action()
    test_invalid_insight_blocks_recommendation()
    test_high_missingness_lowers_confidence()
    test_contradictory_outputs_suppressed()
    test_valid_causal_evidence_passes_through()
    test_prescriptive_story_becomes_dominant_judgment()
    print("All judgment orchestrator tests passed.")
