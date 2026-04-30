from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


CAUSAL_VERBS = ("caused", "causes", "cause", "drives", "driven by", "led to", "produced")


def allowed_numeric_tokens(payload: Dict[str, Any]) -> set[str]:
    tokens: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, (int, float)):
            tokens.add(str(round(float(value), 4)))
            tokens.add(str(int(value)) if float(value).is_integer() else str(value))
        elif isinstance(value, str):
            for token in re.findall(r"-?\d+(?:\.\d+)?", value):
                tokens.add(token)

    visit(payload)
    return tokens


def sanitize_causal_language(text: str, causal_grade: str) -> str:
    if causal_grade == "STRONG":
        return text
    sanitized = text
    for phrase in CAUSAL_VERBS:
        sanitized = re.sub(rf"\b{re.escape(phrase)}\b", "is associated with", sanitized, flags=re.IGNORECASE)
    return sanitized


def numeric_consistency_issues(text: str, payload: Dict[str, Any]) -> List[str]:
    allowed = allowed_numeric_tokens(payload)
    found = re.findall(r"-?\d+(?:\.\d+)?", text)
    issues: List[str] = []
    for token in found:
        if token not in allowed:
            issues.append(f"Unexpected numeric token in LLM output: {token}")
    return issues


def validate_explanation_text(text: str, payload: Dict[str, Any], causal_grade: str) -> Dict[str, Any]:
    sanitized = sanitize_causal_language(text, causal_grade)
    issues = numeric_consistency_issues(sanitized, payload)
    return {
        "text": sanitized,
        "issues": issues,
        "valid": not issues,
    }
