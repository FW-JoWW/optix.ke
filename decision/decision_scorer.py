from __future__ import annotations

from typing import Any, Dict


IMPACT_MAP = {
    "none": 0,
    "uncertain": 25,
    "low": 40,
    "medium": 65,
    "high": 90,
}


def score_decision(
    story: Dict[str, Any],
    impact_assessment: Dict[str, Any],
) -> Dict[str, Any]:
    validity = story.get("insight_validity") or {}
    if not validity.get("valid", True):
        return {"priority_score": 0, "priority_level": "low"}

    impact_score = IMPACT_MAP.get(impact_assessment.get("impact_level", "low"), 0)
    causal_score = int(((story.get("causal_evidence") or {}).get("score")) or 0)
    bias_count = len(story.get("bias_risks", []) or [])
    missing_ratio = float(validity.get("missing_ratio", 0.0) or 0.0)

    quality_score = 100
    quality_score -= min(bias_count * 12, 36)
    quality_score -= 20 if missing_ratio > 0.2 else 0
    severity = validity.get("severity", "low")
    if severity == "high":
        quality_score -= 20
    elif severity == "medium":
        quality_score -= 10
    quality_score = max(0, quality_score)

    priority_score = int(round((0.45 * impact_score) + (0.35 * causal_score) + (0.20 * quality_score)))

    causal_grade = ((story.get("causal_evidence") or {}).get("grade")) or "LOW"
    if causal_grade == "LOW":
        priority_score = min(priority_score, 40)

    priority_score = max(0, min(priority_score, 100))

    if priority_score >= 85:
        level = "critical"
    elif priority_score >= 65:
        level = "high"
    elif priority_score >= 35:
        level = "medium"
    else:
        level = "low"

    return {
        "priority_score": priority_score,
        "priority_level": level,
    }
