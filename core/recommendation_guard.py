from __future__ import annotations

from typing import Any, Dict, List


LOW_CAUSAL_ALLOWED = [
    "investigate further",
    "run experiment",
    "collect more data",
]
LOW_CAUSAL_BLOCKED = [
    "pricing strategy",
    "marketing decisions",
    "product changes",
    "operational changes",
]


def guard_recommendations(
    stats_output: Dict[str, Any],
    semantic_output: Dict[str, Any],
    validation_output: Dict[str, Any],
) -> Dict[str, Any]:
    relationship_type = semantic_output.get("relationship_type", "unknown")
    causal_grade = ((stats_output.get("causal_evidence") or {}).get("grade")) or "LOW"

    restrictions: List[str] = []
    allowed_actions: List[str] = []
    blocked_actions: List[str] = []
    final_recommendation = ""
    guardrail_triggered = False

    if not validation_output.get("valid", True):
        guardrail_triggered = True
        restrictions.append("insight_invalid")
        if relationship_type == "unit_conversion":
            allowed_actions = ["data validation note", "no action required"]
            final_recommendation = "No business action is warranted; this is a mathematically dependent relationship."
        else:
            allowed_actions = []
            final_recommendation = "No actionable recommendation due to insufficient or non-meaningful relationship."
        return {
            "allowed_actions": allowed_actions,
            "blocked_actions": blocked_actions,
            "recommendation_restrictions": restrictions,
            "final_recommendation": final_recommendation,
            "guardrail_triggered": guardrail_triggered,
        }

    if relationship_type == "unit_conversion":
        guardrail_triggered = True
        restrictions.append("unit_conversion_only")
        allowed_actions = ["data validation note", "no action required"]
        final_recommendation = "Treat this as a data consistency check, not an independent business insight."
        return {
            "allowed_actions": allowed_actions,
            "blocked_actions": blocked_actions,
            "recommendation_restrictions": restrictions,
            "final_recommendation": final_recommendation,
            "guardrail_triggered": guardrail_triggered,
        }

    if causal_grade == "LOW":
        guardrail_triggered = True
        restrictions.append("causal_low_exploratory_only")
        allowed_actions = LOW_CAUSAL_ALLOWED[:]
        blocked_actions = LOW_CAUSAL_BLOCKED[:]
        final_recommendation = "Use this relationship to guide follow-up investigation, experimentation, or data collection only."
    else:
        allowed_actions = [
            "monitor the signal",
            "segment reporting",
            "design a targeted follow-up analysis",
        ]
        final_recommendation = "Use the relationship carefully in planning, while keeping the listed limitations visible."

    return {
        "allowed_actions": allowed_actions,
        "blocked_actions": blocked_actions,
        "recommendation_restrictions": restrictions,
        "final_recommendation": final_recommendation,
        "guardrail_triggered": guardrail_triggered,
    }
