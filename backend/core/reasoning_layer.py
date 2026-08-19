from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Tuple

from backend.core.analytic_capability import infer_capability_signals
from backend.utils.openai_runtime import get_openai_client


PROMPT_VERSION = "1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _numeric_tokens(value: Any) -> set[str]:
    payload = _canonical_json(value)
    tokens = set(re.findall(r"-?\d+(?:\.\d+)?", payload))
    return tokens


def reasoning_cache_key(decision_object: Dict[str, Any], prompt_version: str = PROMPT_VERSION) -> str:
    digest = hashlib.sha256(_canonical_json({"prompt_version": prompt_version, "decision": decision_object}).encode("utf-8")).hexdigest()
    return digest


def build_reasoning_prompt(decision_object: Dict[str, Any]) -> str:
    return f"""
You are a senior data analyst explaining a deterministic decision object.
You must not invent statistics, confidence, evidence, assumptions, or recommendations.
Use only the fields in the decision object.
If the object is missing an answer, say it is not available.
Return JSON only.

Decision object:
{json.dumps(decision_object, ensure_ascii=True, indent=2)}

Return exactly:
{{
  "summary": "short analyst-facing explanation",
  "why_this_recommendation": "why the recommendation was selected",
  "why_not_alternative": "why a plausible alternative was not selected",
  "confidence": {{
    "score": 0,
    "level": "high | medium | low",
    "factors": ["deterministic evidence that supports the decision"],
    "reducing_factors": ["deterministic evidence that weakens the decision"]
  }},
  "assumptions": ["only assumptions already present in the decision object"],
  "impact_if_accepted": "what happens if the recommendation is accepted",
  "impact_if_modified": "what happens if the recommendation is modified",
  "alternatives": [
    {{
      "name": "alternative name",
      "reason": "why it is not the primary choice"
    }}
  ]
}}
""".strip()


def _fallback_reasoning(decision_object: Dict[str, Any]) -> Dict[str, Any]:
    confidence = decision_object.get("confidence") or {}
    alternatives = decision_object.get("alternatives") or []
    primary_alt = alternatives[0] if alternatives else {}
    evidence = decision_object.get("evidence") or []
    return {
        "summary": f"{decision_object.get('recommendation', 'Continue')} at {decision_object.get('stage', 'the current stage')} because the deterministic evidence supports it.",
        "why_this_recommendation": f"The deterministic engine selected this because {evidence[0] if evidence else 'it was the safest available option.'}",
        "why_not_alternative": f"An alternative such as {primary_alt.get('name', 'Modify')} was not selected because {primary_alt.get('reason', 'it is less suitable for the current evidence profile.')}",
        "confidence": {
            "score": confidence.get("score", 0),
            "level": confidence.get("level", "low"),
            "factors": list(confidence.get("factors") or []),
            "reducing_factors": list(confidence.get("reducing_factors") or []),
        },
        "assumptions": list(decision_object.get("assumptions") or []),
        "impact_if_accepted": "; ".join(decision_object.get("impact") or ["The current deterministic path continues."]),
        "impact_if_modified": "Modifying the decision will change the downstream path and may require recomputation.",
        "alternatives": alternatives[:3],
    }


def _validate_reasoning_payload(payload: Dict[str, Any], decision_object: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False

    required_keys = {
        "summary",
        "why_this_recommendation",
        "why_not_alternative",
        "confidence",
        "assumptions",
        "impact_if_accepted",
        "impact_if_modified",
        "alternatives",
    }
    if not required_keys.issubset(payload.keys()):
        return False

    confidence = payload.get("confidence")
    decision_confidence = decision_object.get("confidence") or {}
    if not isinstance(confidence, dict):
        return False
    if confidence.get("score") != decision_confidence.get("score"):
        return False
    if str(confidence.get("level", "")).lower() != str(decision_confidence.get("level", "")).lower():
        return False

    allowed_numbers = _numeric_tokens(decision_object)
    for field in ("summary", "why_this_recommendation", "why_not_alternative", "impact_if_accepted", "impact_if_modified"):
        text = str(payload.get(field, ""))
        for token in re.findall(r"-?\d+(?:\.\d+)?", text):
            if token not in allowed_numbers:
                return False

    for item in confidence.get("factors", []) + confidence.get("reducing_factors", []):
        if not isinstance(item, str):
            return False

    for alt in payload.get("alternatives", []):
        if not isinstance(alt, dict):
            return False
        if not alt.get("name") or not alt.get("reason"):
            return False

    return True


def explain_decision(
    decision_object: Dict[str, Any],
    cache: Dict[str, Any] | None = None,
    allow_llm: bool = True,
) -> Tuple[Dict[str, Any], str]:
    cache = cache if cache is not None else {}
    cache_key = reasoning_cache_key(decision_object)
    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        return cached, "cache_hit"

    if not allow_llm:
        fallback = _fallback_reasoning(decision_object)
        cache[cache_key] = fallback
        return fallback, "deterministic_fallback"

    client = get_openai_client("reasoning")
    if client is None:
        fallback = _fallback_reasoning(decision_object)
        cache[cache_key] = fallback
        return fallback, "deterministic_fallback"

    prompt = build_reasoning_prompt(decision_object)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        parsed = json.loads(content) if content.strip() else {}
    except Exception:
        fallback = _fallback_reasoning(decision_object)
        cache[cache_key] = fallback
        return fallback, "deterministic_fallback"

    if not _validate_reasoning_payload(parsed, decision_object):
        fallback = _fallback_reasoning(decision_object)
        cache[cache_key] = fallback
        return fallback, "validation_fallback"

    cache[cache_key] = parsed
    return parsed, "live_llm"


def format_reasoning_explanation(reasoning: Dict[str, Any]) -> List[str]:
    if not reasoning:
        return ["No reasoning explanation was available."]
    confidence = reasoning.get("confidence") or {}
    lines = [
        str(reasoning.get("summary", "")).strip(),
        str(reasoning.get("why_this_recommendation", "")).strip(),
        str(reasoning.get("why_not_alternative", "")).strip(),
    ]
    confidence_line = f"Confidence: {confidence.get('score', 'unknown')}% ({confidence.get('level', 'unknown')})"
    lines.append(confidence_line)
    factors = confidence.get("factors") or []
    reducing = confidence.get("reducing_factors") or []
    if factors:
        lines.append(f"Confidence factors: {', '.join(map(str, factors[:4]))}")
    if reducing:
        lines.append(f"Confidence reducers: {', '.join(map(str, reducing[:4]))}")
    assumptions = reasoning.get("assumptions") or []
    if assumptions:
        lines.append(f"Assumptions: {', '.join(map(str, assumptions[:4]))}")
    impact = reasoning.get("impact_if_accepted")
    if impact:
        lines.append(f"Impact if accepted: {impact}")
    modified = reasoning.get("impact_if_modified")
    if modified:
        lines.append(f"Impact if modified: {modified}")
    alternatives = reasoning.get("alternatives") or []
    for alt in alternatives[:3]:
        if isinstance(alt, dict):
            lines.append(f"Alternative: {alt.get('name')}: {alt.get('reason')}")
    return [line for line in lines if line]


def _match_capabilities(request_text: str, capability_profile: Dict[str, Any] | None) -> Dict[str, Any]:
    signals = infer_capability_signals(request_text)
    profile = capability_profile or {}
    capability_keywords = profile.get("capability_keywords") or {}
    supported_capabilities = list(profile.get("supported_capabilities") or [])
    matched_capabilities: List[str] = []
    unsupported_capabilities: List[str] = []

    for capability, keywords in capability_keywords.items():
        if any(keyword in (request_text or "").lower() for keyword in keywords):
            matched_capabilities.append(capability)
            if capability not in supported_capabilities:
                unsupported_capabilities.append(capability)

    return {
        "signals": signals,
        "matched_capabilities": matched_capabilities,
        "unsupported_capabilities": unsupported_capabilities,
        "supported_capabilities": supported_capabilities,
    }


def interpret_modification_request(
    request_text: str,
    decision_object: Dict[str, Any],
    capability_profile: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    text = (request_text or "").strip().lower()
    alternatives = decision_object.get("alternatives") or []
    recommendation = str(decision_object.get("recommendation", "continue")).lower()
    capability_match = _match_capabilities(text, capability_profile)
    request_mode = "neutral"
    if any(token in text for token in ["instead of", "replace", "swap", "rather than"]):
        request_mode = "substitution"
    elif any(token in text for token in ["add", "include", "also", "append", "plus", "along with"]):
        request_mode = "additive"
    elif any(token in text for token in ["remove", "drop", "exclude", "without", "skip", "omit"]):
        request_mode = "removal"
    question_words = ("what ", "why ", "which ", "how ", "can ", "could ", "should ", "would ", "is ", "are ", "do ", "does ")
    request_kind = "question" if "?" in text or text.startswith(question_words) else "modification"

    keywords = {
        "robust": ["kruskal", "robust", "nonparametric", "stronger"],
        "median": ["median", "skew", "outlier"],
        "mean": ["mean", "average"],
        "mode": ["mode", "most frequent"],
        "keep": ["keep", "preserve", "retain"],
        "remove": ["remove", "drop", "exclude"],
    }

    matches: List[Dict[str, Any]] = []
    for alt in alternatives:
        if not isinstance(alt, dict):
            continue
        alt_text = f"{alt.get('name', '')} {alt.get('reason', '')}".lower()
        score = 0
        for keyword, tokens in keywords.items():
            if any(token in text and token in alt_text for token in tokens):
                score += 1
        if score:
            matches.append({"name": alt.get("name"), "reason": alt.get("reason"), "score": score})

    matches.sort(key=lambda item: item.get("score", 0), reverse=True)
    needs_clarification = not matches and len(text.split()) <= 3
    supported_matches = [item for item in matches if item.get("name") and item.get("name") != "Modify"]
    support_status = "supported" if supported_matches else "partial" if matches else "unsupported"
    if capability_match["matched_capabilities"] and not capability_match["unsupported_capabilities"]:
        support_status = "supported"
    elif capability_match["matched_capabilities"] and capability_match["unsupported_capabilities"]:
        support_status = "partial"

    best_match = matches[0] if matches else {}
    confidence_score = 85 if support_status == "supported" else 68 if support_status == "partial" else 45
    if needs_clarification:
        confidence_score = 35
    if best_match.get("score"):
        confidence_score += min(15, int(best_match["score"]) * 5)
    confidence_score = max(0, min(100, confidence_score))
    confidence_level = "high" if confidence_score >= 80 else "medium" if confidence_score >= 55 else "low"

    explanation_parts: List[str] = []
    if capability_match["matched_capabilities"]:
        explanation_parts.append(
            f"I mapped your request to {', '.join(capability_match['matched_capabilities'][:4])} within this stage."
        )
    if best_match:
        explanation_parts.append(f"The closest stage option is {best_match.get('name')} because {best_match.get('reason')}.")
    if not best_match and capability_match["matched_capabilities"]:
        explanation_parts.append("The request fits the stage capability, but no named alternative was a strong textual match.")
    if not capability_match["matched_capabilities"]:
        explanation_parts.append("The request did not match a supported capability for this stage.")
    if request_mode == "additive":
        explanation_parts.append("This request adds to the current plan rather than replacing it.")
    elif request_mode == "substitution":
        explanation_parts.append("This request asks for a replacement, so I compare it against the current recommendation.")
    elif request_mode == "removal":
        explanation_parts.append("This request removes part of the current plan, so I check whether the stage can still stay valid.")

    if support_status == "supported":
        response_type = "suggest"
    elif support_status == "partial" and not needs_clarification:
        response_type = "suggest"
    else:
        response_type = "clarify" if needs_clarification else "fallback"
    if request_kind == "question":
        response_type = "question"

    return {
        "stage": decision_object.get("stage"),
        "original_request": request_text,
        "request_kind": request_kind,
        "signals": capability_match["signals"],
        "matched_capabilities": capability_match["matched_capabilities"],
        "unsupported_capabilities": capability_match["unsupported_capabilities"],
        "best_matches": matches[:3],
        "best_match": best_match,
        "support_status": support_status,
        "request_mode": request_mode,
        "needs_clarification": needs_clarification,
        "fallback_recommendation": decision_object.get("recommendation") or recommendation,
        "response_type": response_type,
        "confidence": {
            "score": confidence_score,
            "level": confidence_level,
            "factors": [
                f"Decision recommendation remains {decision_object.get('recommendation', 'continue')}.",
                *(f"Matched capability: {item}" for item in capability_match["matched_capabilities"][:3]),
            ],
            "reducing_factors": [
                *(f"Unsupported capability: {item}" for item in capability_match["unsupported_capabilities"][:3]),
            ],
        },
        "explanation": " ".join(explanation_parts).strip(),
        "why_keep_current": f"The current recommendation is still preferred because {decision_object.get('evidence', ['it remains the strongest supported choice'])[0] if decision_object.get('evidence') else 'it remains the strongest supported choice.'}",
        "why_modify": f"The requested change is only useful when it maps cleanly to the current stage capabilities.",
    }


