from __future__ import annotations

from typing import Any, Dict, List

from core.decision_contracts import DecisionRecord, ImpactAssessment, PriorityAssessment
from decision.action_mapper import map_actions
from decision.decision_scorer import score_decision
from decision.impact_estimator import estimate_impact
from decision.prioritizer import prioritize_decisions
from state.state import AnalystState


def _story_signature(story: Dict[str, Any]) -> str:
    columns = story.get("columns") or []
    if story.get("column"):
        columns = [story["column"], *columns]
    if story.get("group_column"):
        columns = [*columns, story["group_column"]]
    return f"{story.get('type', 'story')}|{'|'.join(str(col) for col in columns if col)}"


def _decision_summary(action_type: str) -> str:
    if action_type == "no_action":
        return "Detected relationship is not actionable. No decision required."
    if action_type == "experiment":
        return "Relationship observed but causal evidence is weak. Recommend running controlled experiment before taking action."
    if action_type == "optimization":
        return "Moderate evidence suggests potential improvement. Recommend controlled rollout and monitoring."
    if action_type == "strategic":
        return "Strong causal evidence supports action. Recommend full implementation."
    return "Further investigation is recommended before acting."


def _recommended_action(action_type: str, possible_actions: List[str]) -> str:
    if action_type == "no_action":
        return "No action required"
    return possible_actions[0] if possible_actions else "Investigate further"


def decision_engine_node(state: AnalystState) -> AnalystState:
    evidence = state.setdefault("analysis_evidence", {})
    top_stories = evidence.get("top_stories", []) or []
    if not top_stories:
        evidence["decision_recommendations"] = []
        evidence["decision_priority_ranking"] = []
        return state

    decisions: List[Dict[str, Any]] = []
    for story in top_stories:
        semantic = story.get("semantic_reasoning") or {}
        validity = story.get("insight_validity") or {"valid": True, "severity": "low"}
        action_mapping = map_actions(
            relationship_type=story.get("relationship_type", "unknown"),
            causal_evidence=story.get("causal_evidence") or {},
            insight_validity=validity,
            business_context={"question": state.get("business_question", "")},
        )
        impact = estimate_impact(story, action_mapping)
        priority = score_decision(story, impact)
        summary = _decision_summary(action_mapping["action_type"])
        recommended = _recommended_action(action_mapping["action_type"], action_mapping["possible_actions"])

        confidence_in_action = int(round(min(
            100,
            ((priority["priority_score"] * 0.5) + (((story.get("causal_evidence") or {}).get("score") or 0) * 0.5))
        )))

        record = DecisionRecord(
            story_signature=_story_signature(story),
            insight=story.get("insight", ""),
            action_type=action_mapping["action_type"],
            possible_actions=action_mapping["possible_actions"],
            impact_assessment=ImpactAssessment(**impact),
            priority=PriorityAssessment(**priority),
            decision_summary=summary,
            recommended_action=recommended,
            confidence_in_action=confidence_in_action,
            relationship_type=story.get("relationship_type"),
            validity=validity.get("valid"),
            recommendation_restrictions=story.get("recommendation_restrictions", []),
        )
        decisions.append(record.model_dump())

    ranked = prioritize_decisions(decisions)
    evidence["decision_recommendations"] = decisions
    evidence["decision_priority_ranking"] = ranked
    evidence["decision_recommended_first"] = ranked[0] if ranked else None

    print("\n=== DECISION ENGINE COMPLETE ===")
    for item in ranked[:5]:
        priority = item.get("priority") or {}
        print(
            f"{item.get('action_type')} | score={priority.get('priority_score')} | "
            f"recommended={item.get('recommended_action')}"
        )

    return state
