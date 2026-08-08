from __future__ import annotations

from typing import Any, Dict, List


DETERMINISTIC_TYPES = {"unit_conversion", "duplicate_feature", "formula_dependency", "derived"}
SAFE_NON_ACTIONS = {"ignore", "data validation", "monitoring", "documentation", "no action required"}
HIGH_IMPACT_ACTIONS = {
    "pricing strategy",
    "marketing strategy",
    "product changes",
    "run a/b test",
    "run holdout test",
    "full rollout",
    "policy change",
    "resource allocation",
}


class JudgmentOrchestrator:
    """
    Final governance layer that reconciles semantic, causal, quality, and
    decision outputs into one coherent analyst judgment.
    """

    def __init__(
        self,
        *,
        stats_output: Dict[str, Any] | None = None,
        semantic_output: Dict[str, Any] | None = None,
        causal_output: Dict[str, Any] | None = None,
        quality_output: Dict[str, Any] | None = None,
        story_candidates: List[Dict[str, Any]] | None = None,
        decision_candidates: List[Dict[str, Any]] | None = None,
        report_context: Dict[str, Any] | None = None,
        domain_context: Dict[str, Any] | None = None,
    ) -> None:
        self.stats_output = stats_output or {}
        self.semantic_output = semantic_output or {}
        self.causal_output = causal_output or {}
        self.quality_output = quality_output or {}
        self.story_candidates = story_candidates or []
        self.decision_candidates = decision_candidates or []
        self.report_context = report_context or {}
        self.domain_context = domain_context or {}
        self.primary_story = self._pick_primary_story()

    def _pick_primary_story(self) -> Dict[str, Any]:
        if not self.story_candidates:
            return {}
        prescriptive = [story for story in self.story_candidates if story.get("type") == "prescriptive_action" and float(story.get("score", 0.0)) >= 0.6]
        if prescriptive:
            return max(prescriptive, key=lambda story: float(story.get("score", 0.0)))
        return max(self.story_candidates, key=lambda story: float(story.get("score", 0.0)))

    def evaluate_evidence(self) -> Dict[str, Any]:
        story = self.primary_story
        relationship_type = story.get("relationship_type", "unknown")
        validity = story.get("insight_validity") or {}
        causal = story.get("causal_evidence") or {}
        causal_grade = causal.get("grade", "LOW")
        story_type = story.get("type")

        final_truth_state = "no_reliable_finding"
        suppressed_modules: List[str] = []
        notes: List[str] = []

        if relationship_type in DETERMINISTIC_TYPES:
            final_truth_state = "deterministic_relationship"
            suppressed_modules = [
                "confounder_analysis",
                "causal_recommendations",
                "strategy_recommendations",
            ]
            notes.append("Deterministic semantic classification outranks statistical strength.")
        elif validity and not validity.get("valid", True):
            final_truth_state = "invalid_relationship"
            suppressed_modules = ["strategy_recommendations"]
            notes.append("Insight validity gate failed, so action-oriented conclusions are suppressed.")
        elif story_type == "prescriptive_action":
            final_truth_state = "prescriptive_recommendation"
            notes.append("Prescriptive output is being treated as the dominant operational recommendation layer.")
        elif story_type == "predictive_model":
            final_truth_state = "predictive_signal"
            notes.append("Predictive output is being treated as decision support rather than causal proof.")
        elif story_type in {"summary_numeric", "category_frequency", "rare_categories", "grouped_numeric", "group_difference", "inferential_group_difference", "categorical_relationship", "inferential_categorical_association"}:
            final_truth_state = "descriptive_finding"
            notes.append("Descriptive evidence is being treated as baseline insight, not as causal or experimental proof.")
        elif causal_grade == "STRONG":
            final_truth_state = "causally_actionable_relationship"
            notes.append("Strong causal evidence outranks weaker interpretive layers.")
        elif causal_grade == "MODERATE":
            final_truth_state = "conditionally_actionable_relationship"
            notes.append("Moderate causal evidence supports controlled action but not unconstrained strategy.")
        elif story:
            final_truth_state = "statistical_relationship"
            notes.append("Relationship remains observational and should be interpreted cautiously.")

        return {
            "final_truth_state": final_truth_state,
            "suppressed_modules": suppressed_modules,
            "notes": notes,
        }

    def detect_contradictions(self, evaluated: Dict[str, Any]) -> List[str]:
        contradictions: List[str] = []
        story = self.primary_story
        validity = story.get("insight_validity") or {}
        relationship_type = story.get("relationship_type", "unknown")
        missing_ratio = float(validity.get("missing_ratio", 0.0) or 0.0)

        for decision in self.decision_candidates:
            action_type = decision.get("action_type")
            recommended_action = str(decision.get("recommended_action", "")).lower()
            possible_actions = [str(item).lower() for item in decision.get("possible_actions", [])]
            if validity and not validity.get("valid", True) and action_type != "no_action":
                contradictions.append("invalid_insight_but_action_recommended")
            if relationship_type in DETERMINISTIC_TYPES and (
                action_type not in {None, "no_action"} or any(action in HIGH_IMPACT_ACTIONS for action in possible_actions)
            ):
                contradictions.append("deterministic_relationship_but_strategy_suggested")
            if relationship_type in DETERMINISTIC_TYPES and "a/b test" in recommended_action:
                contradictions.append("deterministic_relationship_but_experiment_suggested")
            if not validity.get("valid", True) and recommended_action not in {"", "no action required"}:
                contradictions.append("invalid_insight_has_nontrivial_recommendation")

        if missing_ratio > 0.2:
            if story.get("confidence") == "high":
                contradictions.append("high_missingness_but_high_story_confidence")
            for decision in self.decision_candidates:
                if int(decision.get("confidence_in_action", 0)) >= 70:
                    contradictions.append("high_missingness_but_high_action_confidence")

        return sorted(set(contradictions))

    def harmonize_confidence(self, evaluated: Dict[str, Any], contradictions: List[str]) -> int:
        story = self.primary_story
        validity = story.get("insight_validity") or {}
        causal = story.get("causal_evidence") or {}
        confidence_assessment = (
            story.get("operational_confidence_assessment")
            or story.get("confidence_assessment")
            or {}
        )

        base = int(confidence_assessment.get("score", 0) or 0)
        if base == 0:
            base = int(causal.get("score", 0) or 0)
        if base == 0 and story:
            base = int(round(float(story.get("score", 0.0)) * 100))
        if base == 0 and self.decision_candidates:
            base = int(self.decision_candidates[0].get("confidence_in_action", 0) or 0)

        relationship_type = story.get("relationship_type", "unknown")
        missing_ratio = float(validity.get("missing_ratio", 0.0) or 0.0)
        bias_count = len(story.get("bias_risks", []) or [])
        readiness_warnings = story.get("readiness_warnings", []) or []

        if relationship_type in DETERMINISTIC_TYPES:
            base = min(base, 20)
        if story.get("type") == "prescriptive_action":
            if confidence_assessment.get("score") is None:
                base = max(base, int(round(float(story.get("score", 0.0)) * 100)))
        elif story.get("type") == "predictive_model":
            if confidence_assessment.get("score") is None:
                base = max(base, int(round(float(story.get("score", 0.0)) * 100)))
        if validity and not validity.get("valid", True):
            base = min(base, 15)
        if missing_ratio > 0.2:
            base -= 20
        base -= sum(15 if item.get("severity") == "high" else 8 for item in readiness_warnings)
        base -= min(bias_count * 6, 18)
        base -= min(len(contradictions) * 12, 36)

        severity = validity.get("severity", "low")
        if severity == "high":
            base -= 10
        elif severity == "medium":
            base -= 5

        return max(0, min(int(base), 100))

    def determine_actionability(self, evaluated: Dict[str, Any], contradictions: List[str], confidence: int) -> Dict[str, Any]:
        story = self.primary_story
        validity = story.get("insight_validity") or {}
        relationship_type = story.get("relationship_type", "unknown")
        story_type = story.get("type")

        actionability = True
        allowed_actions: List[str] = []
        blocked_actions: List[str] = []

        if relationship_type in DETERMINISTIC_TYPES or not validity.get("valid", True):
            actionability = False
            allowed_actions = ["data validation", "monitoring", "documentation", "no action"]
            blocked_actions = ["pricing strategy", "marketing strategy", "product changes", "experiments"]
        elif contradictions:
            actionability = False
            allowed_actions = ["investigate further", "data validation", "monitoring"]
            blocked_actions = ["pricing strategy", "marketing strategy", "product changes"]
        elif story_type == "predictive_model" and confidence >= 45:
            allowed_actions = ["planning", "monitoring", "validate model stability"]
        elif story_type == "prescriptive_action" and confidence >= 50:
            allowed_actions = ["controlled rollout", "monitor impact", "resource reallocation"]
        elif confidence < 35:
            actionability = False
            allowed_actions = ["investigate further", "collect more data"]
            blocked_actions = ["pricing strategy", "marketing strategy", "product changes"]

        return {
            "actionability": actionability,
            "allowed_actions": allowed_actions,
            "blocked_actions": blocked_actions,
        }

    def prioritize_narrative(self, evaluated: Dict[str, Any], contradictions: List[str], actionability_payload: Dict[str, Any]) -> Dict[str, Any]:
        story = self.primary_story
        relationship_type = story.get("relationship_type", "unknown")
        validity = story.get("insight_validity") or {}
        causal_grade = ((story.get("causal_evidence") or {}).get("grade")) or "LOW"
        story_type = story.get("type")

        dominant_reasoning = "No reliable analytical judgment could be formed."
        narrative_priority: List[str] = []
        notes: List[str] = []

        if relationship_type in DETERMINISTIC_TYPES:
            dominant_reasoning = "This is a mathematically dependent relationship useful for validation only."
            narrative_priority = [
                "deterministic semantic truth",
                "data integrity interpretation",
            ]
        elif not validity.get("valid", True):
            dominant_reasoning = "This finding is not reliable enough for decision-making."
            narrative_priority = [
                "validity failure",
                "quality limitations",
            ]
        elif story_type == "prescriptive_action":
            dominant_reasoning = "A model-informed optimization path is available, but it should be rolled out in a controlled way rather than treated as causal proof."
            narrative_priority = [
                "prescriptive recommendation",
                "controlled operational rollout",
                "monitoring discipline",
            ]
        elif story_type == "predictive_model":
            dominant_reasoning = "A predictive signal is available for planning, but it should be used as decision support rather than interpreted as causal evidence."
            narrative_priority = [
                "predictive decision support",
                "model monitoring",
                "causal caution",
            ]
        elif story_type in {"summary_numeric", "category_frequency", "rare_categories", "grouped_numeric", "group_difference", "inferential_group_difference", "categorical_relationship", "inferential_categorical_association"}:
            dominant_reasoning = "This is a descriptive finding that helps with monitoring and segmentation, but it is not by itself a causal basis for strategy."
            narrative_priority = [
                "descriptive baseline",
                "segment monitoring",
                "follow-up analysis",
            ]
        elif not actionability_payload.get("actionability", True):
            dominant_reasoning = "The current evidence is too conflicted or weak to justify action."
            narrative_priority = [
                "contradiction resolution",
                "quality constraints",
                "causal caution",
            ]
        elif causal_grade == "STRONG":
            dominant_reasoning = "Evidence supports a decision-ready causal interpretation."
            narrative_priority = [
                "experimental or strong causal evidence",
                "operational action",
            ]
        elif causal_grade == "MODERATE":
            dominant_reasoning = "Evidence supports controlled action, but not unconstrained rollout."
            narrative_priority = [
                "moderate causal evidence",
                "controlled optimization",
            ]
        elif story:
            dominant_reasoning = "A relationship is present, but it remains observational and should guide experiments, not strategy."
            narrative_priority = [
                "observational relationship",
                "causal caution",
            ]

        if contradictions:
            notes.append("Conflicting lower-priority outputs were suppressed to preserve one coherent analyst judgment.")

        return {
            "dominant_reasoning": dominant_reasoning,
            "narrative_priority": narrative_priority,
            "judgment_notes": notes,
        }

    def finalize(self) -> Dict[str, Any]:
        evaluated = self.evaluate_evidence()
        contradictions = self.detect_contradictions(evaluated)
        global_confidence = self.harmonize_confidence(evaluated, contradictions)
        actionability_payload = self.determine_actionability(evaluated, contradictions, global_confidence)
        narrative_payload = self.prioritize_narrative(evaluated, contradictions, actionability_payload)

        recommended_first_action = None
        if actionability_payload["actionability"]:
            recommended_first_action = None
            ranked = self.report_context.get("decision_priority_ranking") or []
            if ranked:
                recommended_first_action = ranked[0].get("recommended_action")
        elif "data validation" in actionability_payload["allowed_actions"]:
            recommended_first_action = None

        return {
            "final_truth_state": evaluated["final_truth_state"],
            "global_confidence": global_confidence,
            "actionability": actionability_payload["actionability"],
            "allowed_actions": actionability_payload["allowed_actions"],
            "blocked_actions": actionability_payload["blocked_actions"],
            "suppressed_modules": evaluated["suppressed_modules"],
            "dominant_reasoning": narrative_payload["dominant_reasoning"],
            "contradictions_found": contradictions,
            "narrative_priority": narrative_payload["narrative_priority"],
            "recommended_first_action": recommended_first_action,
            "judgment_notes": [*evaluated["notes"], *narrative_payload["judgment_notes"]],
        }
