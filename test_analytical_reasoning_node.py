from __future__ import annotations

from nodes.analytical_reasoning_node import analytical_reasoning_node
from state.state import AnalystState


def test_analytical_reasoning_node_builds_structured_reasoning() -> None:
    state: AnalystState = {
        "business_question": "Reduce churn in premium customers",
        "mode": "autonomous",
        "awaiting_user": False,
        "question_for_user": "",
        "user_response": "",
        "enable_llm_reasoning": False,
        "disable_llm_reasoning": True,
        "disable_semantic_matcher": True,
        "analysis_evidence": {
            "top_stories": [
                {
                    "type": "correlation",
                    "insight": "revenue and churn show a strong negative pattern",
                    "columns": ["revenue", "churn"],
                    "value": -0.82,
                    "score": 0.91,
                    "confidence": "high",
                    "insight_validity": {"valid": True, "severity": "low", "missing_ratio": 0.02},
                    "causal_evidence": {"grade": "LOW", "score": 52, "rationale": ["observational association only"]},
                    "bias_risks": ["missingness_bias"],
                    "recommendation_restrictions": ["observational_only"],
                }
            ],
            "decision_priority_ranking": [
                {
                    "story_signature": "correlation|revenue|churn",
                    "recommended_action": "Launch a retention pilot for premium customers",
                    "action_type": "optimization",
                    "priority": {"priority_score": 82, "priority_level": "high"},
                    "impact_assessment": {"impact_level": "high", "estimated_direction": "positive"},
                    "confidence_in_action": 76,
                    "monitoring_kpis": ["churn_rate", "repeat_purchase_rate"],
                }
            ],
            "decision_recommended_first": {
                "recommended_action": "Launch a retention pilot for premium customers"
            },
            "judgment_summary": {
                "dominant_reasoning": "A predictive signal is available for planning, but it should be used as decision support rather than interpreted as causal evidence.",
                "global_confidence": 79,
                "actionability": True,
                "contradictions_found": [],
                "recommended_first_action": "Launch a retention pilot for premium customers",
            },
            "llm_insight_details": [
                {
                    "related_story_signature": "correlation|revenue|churn",
                    "plain_english": "revenue appears to move inversely with churn",
                    "business_implication": "This suggests the relationship is useful for planning but not for proving causality.",
                    "recommended_action": "Use the observed relationship as a planning signal.",
                    "limitations": "The data does not establish a root cause."
                }
            ],
        },
    }

    final_state = analytical_reasoning_node(state)
    reasoning = final_state["analytical_reasoning"]

    assert reasoning["sections"]["findings"]
    assert reasoning["sections"]["root_cause"]
    assert reasoning["sections"]["recommendations"]
    assert reasoning["confidence"]["level"] in {"low", "medium", "high"}
    assert reasoning["traceability"]["primary_story_signature"] == "correlation|revenue|churn"


if __name__ == "__main__":
    test_analytical_reasoning_node_builds_structured_reasoning()
    print("Analytical reasoning node test passed.")
