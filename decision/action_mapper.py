from __future__ import annotations

from typing import Any, Dict


def map_actions(
    relationship_type: str,
    causal_evidence: Dict[str, Any] | None,
    insight_validity: Dict[str, Any] | None,
    business_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    del business_context
    validity = insight_validity or {}
    causal_grade = ((causal_evidence or {}).get("grade")) or "LOW"

    if not validity.get("valid", True):
        return {
            "action_type": "no_action",
            "possible_actions": ["ignore", "data validation"],
        }

    if relationship_type in {"unit_conversion", "duplicate_feature"}:
        return {
            "action_type": "no_action",
            "possible_actions": ["ignore", "data validation"],
        }

    if causal_grade == "LOW":
        return {
            "action_type": "experiment",
            "possible_actions": ["run A/B test", "run holdout test", "collect more data"],
        }

    if causal_grade == "MODERATE":
        return {
            "action_type": "optimization",
            "possible_actions": ["controlled rollout", "limited deployment", "monitor impact"],
        }

    if causal_grade == "STRONG":
        return {
            "action_type": "strategic",
            "possible_actions": ["full rollout", "policy change", "resource allocation"],
        }

    return {
        "action_type": "investigation",
        "possible_actions": ["investigate further", "review with domain owner"],
    }
