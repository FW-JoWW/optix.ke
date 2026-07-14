from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Tuple

from utils.openai_runtime import get_openai_client


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


def explain_decision(decision_object: Dict[str, Any], cache: Dict[str, Any] | None = None) -> Tuple[Dict[str, Any], str]:
    cache = cache if cache is not None else {}
    cache_key = reasoning_cache_key(decision_object)
    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        return cached, "cache_hit"

    client = get_openai_client()
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


def interpret_modification_request(request_text: str, decision_object: Dict[str, Any]) -> Dict[str, Any]:
    text = (request_text or "").strip().lower()
    alternatives = decision_object.get("alternatives") or []
    recommendation = str(decision_object.get("recommendation", "continue")).lower()

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

    return {
        "stage": decision_object.get("stage"),
        "original_request": request_text,
        "best_matches": matches[:3],
        "needs_clarification": needs_clarification,
        "fallback_recommendation": decision_object.get("recommendation") or recommendation,
        "response_type": "clarify" if needs_clarification else "suggest",
    }
