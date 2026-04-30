from __future__ import annotations

from typing import Any, Dict


def _effect_value(story: Dict[str, Any]) -> float:
    effect = story.get("effect_size") or {}
    value = effect.get("value", story.get("value", 0.0))
    try:
        return abs(float(value or 0.0))
    except Exception:
        return 0.0


def _sample_size(story: Dict[str, Any]) -> int:
    sample_size = (((story.get("estimation") or {}).get("sample_size")) if isinstance(story.get("estimation"), dict) else None)
    if sample_size is None:
        sample_size = (((story.get("semantic_reasoning") or {}).get("details") or {}).get("sample_size"))
    causal = story.get("causal_evidence") or {}
    if sample_size is None:
        sample_size = (((causal.get("rubric") or {}).get("sample_size", 0)) * 10) if causal else 0
    try:
        return int(sample_size or 0)
    except Exception:
        return 0


def estimate_impact(
    story: Dict[str, Any],
    action_mapping: Dict[str, Any],
) -> Dict[str, Any]:
    validity = story.get("insight_validity") or {}
    if not validity.get("valid", True):
        return {
            "impact_level": "none",
            "estimated_direction": "unclear",
            "confidence_adjusted_impact": 0.0,
        }

    effect_value = _effect_value(story)
    causal_grade = ((story.get("causal_evidence") or {}).get("grade")) or "LOW"
    bias_count = len(story.get("bias_risks", []) or [])
    missing_ratio = float(validity.get("missing_ratio", 0.0) or 0.0)
    sample_size = _sample_size(story)

    direction = "unclear"
    raw_value = story.get("value")
    if raw_value is not None:
        try:
            direction = "positive" if float(raw_value) >= 0 else "negative"
        except Exception:
            direction = "unclear"

    confidence = 0.25
    if effect_value >= 0.5:
        confidence += 0.35
    elif effect_value >= 0.3:
        confidence += 0.22
    elif effect_value >= 0.1:
        confidence += 0.1

    if causal_grade == "STRONG":
        confidence += 0.25
    elif causal_grade == "MODERATE":
        confidence += 0.15
    elif causal_grade == "LOW":
        confidence -= 0.05

    if sample_size >= 1000:
        confidence += 0.1
    elif sample_size >= 100:
        confidence += 0.05
    elif sample_size < 30:
        confidence -= 0.08

    confidence -= min(bias_count * 0.08, 0.24)
    if missing_ratio > 0.2:
        confidence -= 0.1

    confidence = max(0.0, min(confidence, 1.0))

    if action_mapping.get("action_type") == "no_action":
        impact_level = "none"
    elif effect_value >= 0.5 and causal_grade == "LOW":
        impact_level = "uncertain"
    elif effect_value >= 0.5 and causal_grade == "STRONG":
        impact_level = "high"
    elif effect_value >= 0.3 and causal_grade in {"MODERATE", "STRONG"}:
        impact_level = "medium"
    elif effect_value >= 0.1:
        impact_level = "low"
    else:
        impact_level = "low"

    return {
        "impact_level": impact_level,
        "estimated_direction": direction,
        "confidence_adjusted_impact": round(confidence, 4),
    }
