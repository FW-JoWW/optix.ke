from __future__ import annotations

import re
from typing import Any, Dict, List

from llm.guarded_reasoning import validate_explanation_text


def _replace_section(text: str, section_name: str, replacement_lines: List[str]) -> str:
    pattern = rf"({re.escape(section_name)}:\s*)(.*?)(?=\n[A-Z][A-Z ]+:\s|\Z)"
    replacement_body = "\n".join(f"- {line}" for line in replacement_lines) if replacement_lines else "- None"
    if re.search(pattern, text, flags=re.DOTALL):
        return re.sub(pattern, rf"\1{replacement_body}\n", text, flags=re.DOTALL)
    return text.rstrip() + f"\n\n{section_name}:\n{replacement_body}"


def _sanitize_causal_phrases(text: str) -> str:
    replacements = {
        r"\bcauses\b": "is associated with",
        r"\bcaused\b": "was associated with",
        r"\bcause\b": "association",
        r"\bdrives\b": "is associated with",
        r"\bdrove\b": "was associated with",
        r"\bled to\b": "was associated with",
        r"\bresults in\b": "is associated with",
    }
    sanitized = text
    for pattern, replacement in replacements.items():
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return sanitized


def filter_llm_output(
    raw_text: str,
    payload: Dict[str, Any],
    semantic_output: Dict[str, Any],
    validation_output: Dict[str, Any],
    recommendation_output: Dict[str, Any],
) -> Dict[str, Any]:
    relationship_type = semantic_output.get("relationship_type", "unknown")
    causal_grade = ((payload.get("causal_evidence") or {}).get("grade")) or "LOW"
    guardrail_triggered = bool(recommendation_output.get("guardrail_triggered"))
    issues: List[str] = []

    if relationship_type == "unit_conversion":
        forced = (
            "INSIGHT:\n"
            "- This relationship is expected due to mathematical or unit conversion dependency and does not represent an independent business insight.\n\n"
            "BUSINESS IMPLICATION:\n"
            "- This result is useful for data consistency checking, not for strategic decision-making.\n\n"
            "RECOMMENDATIONS:\n"
            "- No action required.\n\n"
            "LIMITATIONS:\n"
            "- The relationship is derived rather than behaviorally or economically informative."
        )
        return {
            "text": forced,
            "guardrail_triggered": True,
            "issues": ["unit_conversion_override"],
        }

    if not validation_output.get("valid", True):
        forced = (
            "INSIGHT:\n"
            "- Detected relationship is not meaningful for decision-making.\n\n"
            "BUSINESS IMPLICATION:\n"
            f"- {validation_output.get('reason', 'The evidence is not reliable enough for action.')}\n\n"
            "RECOMMENDATIONS:\n"
            "- No actionable recommendation due to insufficient or non-meaningful relationship.\n\n"
            "LIMITATIONS:\n"
            "- Treat this result as informational only until better evidence is available."
        )
        return {
            "text": forced,
            "guardrail_triggered": True,
            "issues": ["invalid_insight_override"],
        }

    filtered = raw_text
    if causal_grade != "STRONG":
        sanitized = _sanitize_causal_phrases(filtered)
        if sanitized != filtered:
            issues.append("causal_language_sanitized")
            guardrail_triggered = True
        filtered = sanitized

    if recommendation_output.get("recommendation_restrictions"):
        blocked_tokens = ("pricing", "marketing", "product", "operational")
        recommendation_lines = recommendation_output.get("allowed_actions") or [recommendation_output.get("final_recommendation", "Investigate further.")]
        if any(token in filtered.lower() for token in blocked_tokens):
            filtered = _replace_section(filtered, "RECOMMENDATIONS", recommendation_lines)
            issues.append("recommendations_restricted")
            guardrail_triggered = True

    validation = validate_explanation_text(filtered, payload, causal_grade)
    filtered = validation["text"]
    if validation["issues"]:
        issues.extend(validation["issues"])
        guardrail_triggered = True

    return {
        "text": filtered,
        "guardrail_triggered": guardrail_triggered,
        "issues": issues,
    }
