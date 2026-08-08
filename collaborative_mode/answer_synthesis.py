from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Sequence

from .confidence_diagnostics import (
    build_confidence_diagnostics_sections,
    build_investigation_decision_bundle,
    build_investigation_decision_sections,
    render_confidence_diagnostics_report,
    render_investigation_decision_report,
)
from .narration import humanize_text, suggestion_impact_percent, format_suggestion_line
from utils.openai_runtime import get_openai_client


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _unique_lines(lines: Sequence[Any]) -> List[str]:
    seen: set[str] = set()
    unique: List[str] = []
    for line in lines:
        text = _normalize_text(line)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(text)
    return unique


def _question_relevant_lines(question: str, lines: Sequence[Any]) -> List[str]:
    question_terms = _tokens(question)
    if not question_terms:
        return _unique_lines(lines)
    relevant: List[str] = []
    for line in _unique_lines(lines):
        line_terms = _tokens(line)
        if line_terms & question_terms:
            relevant.append(line)
            continue
        lowered = line.lower()
        if lowered.startswith(("no ", "the evidence", "the analysis", "the answer", "if yes", "if partial", "if no")):
            relevant.append(line)
    return relevant


def _tokens(value: Any) -> set[str]:
    text = _normalize_text(value).lower()
    tokens = set()
    for token in text.split():
        token = token.strip(".,:;!?()[]{}<>\"'")
        if len(token) >= 4:
            tokens.add(token)
    return tokens


def _overlap_score(*values: Any) -> int:
    if not values:
        return 0
    base = _tokens(values[0])
    if not base:
        return 0
    overlap: set[str] = set()
    for value in values[1:]:
        overlap |= _tokens(value)
    return len(base & overlap)


def _numeric_tokens(value: Any) -> set[str]:
    return set(re.findall(r"-?\d+(?:\.\d+)?", _normalize_text(value)))


def _collect_numeric_tokens(*values: Any) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        tokens |= _numeric_tokens(value)
    return tokens


def _confidence_label(value: Any) -> str:
    if isinstance(value, (int, float)):
        score = float(value)
        if score >= 75:
            return "High"
        if score >= 45:
            return "Moderate"
        return "Low"
    if value:
        return str(value).strip().capitalize()
    return "Unknown"


def _item_text(item: Dict[str, Any] | Any) -> str:
    if not isinstance(item, dict):
        return _normalize_text(item)
    candidates = [
        item.get("statement"),
        item.get("finding"),
        item.get("summary"),
        item.get("insight"),
        item.get("narrative"),
        item.get("report_excerpt"),
        item.get("task_finding"),
        item.get("current_understanding"),
        item.get("business_interpretation"),
    ]
    text = " ".join(_normalize_text(value) for value in candidates if _normalize_text(value))
    if text:
        return text
    return _normalize_text(item.get("title") or item.get("request") or item.get("description"))


def _evidence_bucket(question: str, text: str) -> str:
    overlap = _overlap_score(question, text)
    if overlap >= 2:
        return "direct"
    if overlap >= 1:
        return "indirect"
    return "supporting"


def _collect_evidence_items(evidence: Dict[str, Any], state: Dict[str, Any]) -> List[Dict[str, Any]]:
    session = evidence.get("collaborative_session") or state.get("collaborative_session") or {}
    evidence_store = session.get("evidence_store") or evidence.get("collaborative_evidence_store") or {}
    items: List[Dict[str, Any]] = []

    for record in evidence_store.values():
        if isinstance(record, dict):
            items.append({
                "source": record.get("task_source") or record.get("evidence_id") or "evidence_store",
                "text": _item_text(record),
                "confidence": record.get("confidence"),
                "type": record.get("evidence_type") or "evidence_store",
                "raw": record,
            })

    for story in evidence.get("top_stories") or []:
        if isinstance(story, dict):
            items.append({
                "source": story.get("type") or "top_story",
                "text": _item_text(story),
                "confidence": story.get("confidence"),
                "type": "story",
                "raw": story,
            })

    judgment = evidence.get("judgment_summary") or {}
    if judgment:
        items.append({
            "source": "judgment_summary",
            "text": _item_text({
                "statement": judgment.get("summary"),
                "finding": judgment.get("dominant_reasoning"),
                "report_excerpt": judgment.get("summary"),
            }),
            "confidence": judgment.get("global_confidence"),
            "type": "judgment",
            "raw": judgment,
        })

    return [item for item in items if _normalize_text(item.get("text"))]


def _derive_missing_evidence(
    *,
    question: str,
    evidence_items: Sequence[Dict[str, Any]],
    hypotheses: Sequence[Dict[str, Any]],
    knowledge_gaps: Sequence[Any],
    memory: Dict[str, Any],
) -> List[str]:
    missing: List[str] = []
    question_terms = sorted({token for token in _tokens(question) if len(token) >= 4})
    direct_hits = [item for item in evidence_items if _evidence_bucket(question, item.get("text", "")) == "direct"]
    if not direct_hits:
        missing.append("No direct evidence has yet been captured for the original business question.")
    if question_terms and not any(term in _normalize_text(" ".join(item.get("text", "") for item in evidence_items)).lower() for term in question_terms):
        missing.append(
            f"Evidence that speaks directly to {', '.join(question_terms[:4])} is still missing."
        )
    if not any(str(item.get("status", "")).lower() == "supported" for item in hypotheses):
        missing.append("No hypothesis has yet been confirmed strongly enough to treat as the leading explanation.")
    if knowledge_gaps:
        for gap in knowledge_gaps[:3]:
            gap_text = _normalize_text(gap)
            if gap_text:
                missing.append(gap_text)
    if memory.get("last_failure"):
        failure = memory.get("last_failure") or {}
        failure_text = _normalize_text(failure.get("reason") or failure.get("message"))
        if failure_text:
            missing.append(failure_text)
    if not evidence_items:
        missing.append("The investigation has not recorded enough supporting evidence yet to distinguish fact from inference.")
    return list(dict.fromkeys(missing))


def _extract_gap_candidates(evidence: Dict[str, Any], state: Dict[str, Any]) -> List[str]:
    gap_candidates: List[str] = []
    session = evidence.get("collaborative_session") or state.get("collaborative_session") or {}
    memory = session.get("investigation_memory") or state.get("collaborative_memory") or {}
    for value in (
        evidence.get("clarification_questions"),
        memory.get("knowledge_gaps"),
        memory.get("remaining_questions"),
        memory.get("open_questions"),
    ):
        if isinstance(value, list):
            gap_candidates.extend([_normalize_text(item) for item in value if _normalize_text(item)])
        elif value:
            text = _normalize_text(value)
            if text:
                gap_candidates.append(text)
    return list(dict.fromkeys(gap_candidates))


def _rank_next_investigations(
    suggestions: Sequence[Dict[str, Any]],
    question: str,
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for index, suggestion in enumerate(suggestions):
        if not isinstance(suggestion, dict):
            continue
        candidate = dict(suggestion)
        impact = suggestion_impact_percent(candidate)
        title = _normalize_text(candidate.get("title") or candidate.get("request") or "Next investigation")
        request = _normalize_text(candidate.get("request") or candidate.get("description") or title)
        candidate["impact_percent"] = impact
        candidate["_question_relevance"] = _overlap_score(question, title, request)
        candidate["_confidence_score"] = candidate.get("confidence") if isinstance(candidate.get("confidence"), (int, float)) else 0
        candidate["_order"] = index
        ranked.append(candidate)
    ranked.sort(
        key=lambda item: (
            -(item.get("_question_relevance") or 0),
            -(item.get("impact_percent") or 0),
            -(item.get("_confidence_score") or 0),
            item.get("_order", 0),
        )
    )
    return ranked


def _confidence_bundle(
    *,
    status: str,
    evidence_score: int,
    interpretation_score: int,
    business_score: int,
    recommendation_score: int,
) -> Dict[str, Any]:
    return {
        "question": "Do I have enough evidence to answer?",
        "status": status,
        "evidence": {"score": evidence_score, "label": _confidence_label(evidence_score)},
        "interpretation": {"score": interpretation_score, "label": _confidence_label(interpretation_score)},
        "business": {"score": business_score, "label": _confidence_label(business_score)},
        "recommendation": {"score": recommendation_score, "label": _confidence_label(recommendation_score)},
    }


def _build_semantic_answer_prompt(
    *,
    business_question: str,
    deterministic_answer: str,
    answer_status: str,
    evidence_items: Sequence[Dict[str, Any]],
    missing_evidence: Sequence[str],
    confidence_bundle: Dict[str, Any],
    current_understanding: str,
    business_interpretation: str,
) -> str:
    evidence_lines = []
    for item in list(evidence_items)[:8]:
        source = _normalize_text(item.get("source") or item.get("type") or "evidence")
        text = _normalize_text(item.get("text") or "")
        confidence = _normalize_text(item.get("confidence") or "unknown")
        if text:
            evidence_lines.append(f"- {source} | confidence={confidence} | {text}")

    missing_lines = [f"- {line}" for line in list(missing_evidence)[:5] if _normalize_text(line)]
    return f"""
You are a senior data analyst converting evidence into an answer to the user's question.
You must answer the question using only the evidence below.
Do not invent numbers, categories, causes, or business context.
If the evidence does not directly answer the question, say so plainly and explain what evidence is missing.
If the evidence is only a proxy, explain why it is a proxy and not the exact answer.
If the evidence is enough, state the answer directly and briefly explain why.

Business question:
{business_question}

Deterministic answer candidate:
{deterministic_answer}

Deterministic answer status:
{answer_status}

Current understanding:
{current_understanding}

Current business interpretation:
{business_interpretation}

Confidence bundle:
{json.dumps(confidence_bundle, ensure_ascii=True, indent=2)}

Evidence items:
{chr(10).join(evidence_lines) if evidence_lines else "- None"}

Known missing evidence:
{chr(10).join(missing_lines) if missing_lines else "- None"}

Return JSON only in exactly this shape:
{{
  "answer_status": "direct | partial | insufficient",
  "direct_answer": "concise business answer or a statement that the question cannot yet be answered",
  "business_interpretation": "why the evidence does or does not answer the question",
  "supporting_evidence_summary": ["short evidence-based bullets"],
  "observed_facts": ["short fact-based bullets"],
  "analytical_interpretation": ["short interpretation bullets"],
  "key_assumptions": ["assumptions or missing information"],
  "remaining_uncertainty": ["what still blocks a final answer"],
  "recommended_next_investigation": ["the most useful next step"],
  "reasoning": "short explanation of how the evidence maps to the question"
}}
""".strip()


def _validate_semantic_answer_payload(payload: Dict[str, Any], allowed_numbers: set[str]) -> bool:
    if not isinstance(payload, dict):
        return False

    required_keys = {
        "answer_status",
        "direct_answer",
        "business_interpretation",
        "supporting_evidence_summary",
        "observed_facts",
        "analytical_interpretation",
        "key_assumptions",
        "remaining_uncertainty",
        "recommended_next_investigation",
        "reasoning",
    }
    if not required_keys.issubset(payload.keys()):
        return False

    if str(payload.get("answer_status", "")).lower() not in {"direct", "partial", "insufficient"}:
        return False

    for field in (
        "direct_answer",
        "business_interpretation",
        "reasoning",
    ):
        text = _normalize_text(payload.get(field))
        if not text:
            return False
        if allowed_numbers:
            for token in _numeric_tokens(text):
                if token not in allowed_numbers:
                    return False

    for field in (
        "supporting_evidence_summary",
        "observed_facts",
        "analytical_interpretation",
        "key_assumptions",
        "remaining_uncertainty",
        "recommended_next_investigation",
    ):
        value = payload.get(field)
        if not isinstance(value, list):
            return False
        for item in value:
            if not _normalize_text(item):
                return False
            if allowed_numbers:
                for token in _numeric_tokens(item):
                    if token not in allowed_numbers:
                        return False

    return True


def _parse_json_object(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _semantic_answer_fallback(
    *,
    business_question: str,
    deterministic_answer: str,
    answer_status: str,
    evidence_items: Sequence[Dict[str, Any]],
    missing_evidence: Sequence[str],
    confidence_bundle: Dict[str, Any],
    current_understanding: str,
    business_interpretation: str,
) -> Dict[str, Any]:
    if answer_status == "partial":
        business_answer = (
            f"The best available answer is {deterministic_answer}."
            if deterministic_answer
            else "The answer is still provisional."
        )
    else:
        business_answer = (
            f"The current evidence does not yet answer {humanize_text(business_question)}."
            if business_question
            else "The current evidence does not yet support a direct answer."
        )
    return {
        "answer_status": answer_status if answer_status in {"direct", "partial", "insufficient"} else "partial",
        "direct_answer": business_answer,
        "business_interpretation": business_interpretation,
        "supporting_evidence_summary": [
            f"Direct evidence: {item['text']}" for item in list(evidence_items)[:3] if _normalize_text(item.get("text"))
        ] or ["No direct evidence has yet been captured for the original business question."],
        "observed_facts": [
            _normalize_text(item.get("text")) for item in list(evidence_items)[:3] if _normalize_text(item.get("text"))
        ] or ["No observed fact directly resolves the question yet."],
        "analytical_interpretation": [
            "The evidence base is still being translated into a direct answer for the business question."
        ],
        "key_assumptions": list(missing_evidence[:4]) or ["No direct evidence has yet been captured for the original business question."],
        "remaining_uncertainty": list(missing_evidence[:4]) or ["No direct evidence has yet been captured for the original business question."],
        "recommended_next_investigation": [
            "Collect evidence that directly measures the business concept named in the question."
        ],
        "reasoning": (
            f"The evidence suggests {current_understanding} but the business interpretation still does not directly answer the question."
        ),
    }


def _semantic_synthesize_answer(
    *,
    business_question: str,
    deterministic_answer: str,
    answer_status: str,
    evidence_items: Sequence[Dict[str, Any]],
    missing_evidence: Sequence[str],
    confidence_bundle: Dict[str, Any],
    current_understanding: str,
    business_interpretation: str,
) -> tuple[Dict[str, Any], str]:
    client = get_openai_client()
    if client is None:
        return (
            _semantic_answer_fallback(
                business_question=business_question,
                deterministic_answer=deterministic_answer,
                answer_status=answer_status,
                evidence_items=evidence_items,
                missing_evidence=missing_evidence,
                confidence_bundle=confidence_bundle,
                current_understanding=current_understanding,
                business_interpretation=business_interpretation,
            ),
            "deterministic_fallback",
        )

    prompt = _build_semantic_answer_prompt(
        business_question=business_question,
        deterministic_answer=deterministic_answer,
        answer_status=answer_status,
        evidence_items=evidence_items,
        missing_evidence=missing_evidence,
        confidence_bundle=confidence_bundle,
        current_understanding=current_understanding,
        business_interpretation=business_interpretation,
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = _parse_json_object(content)
    except Exception:
        return (
            _semantic_answer_fallback(
                business_question=business_question,
                deterministic_answer=deterministic_answer,
                answer_status=answer_status,
                evidence_items=evidence_items,
                missing_evidence=missing_evidence,
                confidence_bundle=confidence_bundle,
                current_understanding=current_understanding,
                business_interpretation=business_interpretation,
            ),
            "deterministic_fallback",
        )

    allowed_numbers = _collect_numeric_tokens(
        business_question,
        deterministic_answer,
        current_understanding,
        business_interpretation,
        missing_evidence,
        " ".join(_normalize_text(item.get("text")) for item in evidence_items),
    )
    if not _validate_semantic_answer_payload(parsed, allowed_numbers):
        return (
            _semantic_answer_fallback(
                business_question=business_question,
                deterministic_answer=deterministic_answer,
                answer_status=answer_status,
                evidence_items=evidence_items,
                missing_evidence=missing_evidence,
                confidence_bundle=confidence_bundle,
                current_understanding=current_understanding,
                business_interpretation=business_interpretation,
            ),
            "validation_fallback",
        )

    return parsed, "live_llm"


def synthesize_answer(
    *,
    business_question: str,
    evidence: Dict[str, Any],
    hypotheses: Sequence[Dict[str, Any]] | None = None,
    current_understanding: Any = None,
    confidence: Any = None,
    knowledge_gaps: Sequence[Any] | None = None,
    investigation_memory: Dict[str, Any] | None = None,
    dataframe: Any = None,
) -> Dict[str, Any]:
    hypotheses = list(hypotheses or [])
    knowledge_gaps = list(knowledge_gaps or [])
    investigation_memory = dict(investigation_memory or {})
    evidence_items = _collect_evidence_items(evidence, {"collaborative_session": evidence.get("collaborative_session"), "collaborative_memory": investigation_memory})
    direct_items = [item for item in evidence_items if _evidence_bucket(business_question, item["text"]) == "direct"]
    indirect_items = [item for item in evidence_items if _evidence_bucket(business_question, item["text"]) == "indirect"]
    supporting_items = [item for item in evidence_items if _evidence_bucket(business_question, item["text"]) == "supporting"]
    conflicting_items = []
    for hypothesis in hypotheses:
        if str(hypothesis.get("status", "")).lower() == "rejected":
            text = _item_text(hypothesis)
            if text:
                conflicting_items.append({"source": "hypothesis", "text": text, "raw": hypothesis})
    judgment = evidence.get("judgment_summary") or {}
    if judgment.get("contradictions_found"):
        conflicting_items.append({
            "source": "judgment_summary",
            "text": _normalize_text(judgment.get("summary") or judgment.get("dominant_reasoning") or "The judgment summary recorded contradictions."),
            "raw": judgment,
        })

    best_answer = (investigation_memory.get("best_answer") or {}).get("answer")
    current_text = _normalize_text(
        best_answer
        or current_understanding
        or investigation_memory.get("current_understanding")
        or judgment.get("summary")
        or (evidence_items[0]["text"] if evidence_items else "")
    )
    directness = _overlap_score(business_question, current_text)
    evidence_score = min(100, 20 + len(direct_items) * 25 + len(indirect_items) * 12 + len(supporting_items) * 6 - len(conflicting_items) * 15)
    if directness >= 3:
        evidence_score = min(100, evidence_score + 12)
    if not evidence_items:
        evidence_score = min(evidence_score, 20)

    interpretation_score = min(100, 25 + directness * 18 + (10 if current_text else 0) - len(conflicting_items) * 10)
    if isinstance(confidence, (int, float)):
        interpretation_score = min(100, round((interpretation_score * 0.6) + (float(confidence) * 0.4)))

    business_score = min(100, round((evidence_score * 0.5) + (interpretation_score * 0.5)))
    if any(str(item.get("status", "")).lower() == "supported" for item in hypotheses):
        business_score = min(100, business_score + 8)
    if conflicting_items:
        business_score = max(0, business_score - min(20, len(conflicting_items) * 5))

    recommendation_score = 40
    suggestions = evidence.get("ai_suggestions") or evidence.get("collaborative_suggestions") or []
    ranked_suggestions = _rank_next_investigations(suggestions, business_question)
    if ranked_suggestions:
        recommendation_score = min(100, 45 + int(ranked_suggestions[0].get("impact_percent") or 0) // 2)

    if evidence_score >= 70 and not conflicting_items:
        sufficiency_status = "yes"
    elif evidence_score >= 40 or direct_items or indirect_items:
        sufficiency_status = "partial"
    else:
        sufficiency_status = "no"

    direct_answer_core = current_text or "The investigation is still developing."
    if sufficiency_status == "yes":
        direct_answer = humanize_text(direct_answer_core, dataframe=dataframe)
        answer_position = "direct"
    elif sufficiency_status == "partial":
        assumptions_text = "; ".join(_derive_missing_evidence(
            question=business_question,
            evidence_items=evidence_items,
            hypotheses=hypotheses,
            knowledge_gaps=knowledge_gaps,
            memory=investigation_memory,
        )[:2])
        direct_answer = (
            f"The best available answer is {humanize_text(direct_answer_core, dataframe=dataframe)}."
            + (f" That answer remains provisional and depends on these assumptions: {assumptions_text}." if assumptions_text else " That answer remains provisional.")
        )
        answer_position = "closest_defensible"
    else:
        direct_answer = (
            f"The investigation cannot yet give an exact answer to {humanize_text(business_question, dataframe=dataframe)}. "
            f"The strongest current direction is {humanize_text(direct_answer_core or 'that the evidence is still too thin to resolve the question cleanly', dataframe=dataframe)}."
        )
        answer_position = "needs_more_evidence"

    supporting_evidence_summary: List[str] = []
    for item in direct_items[:3]:
        supporting_evidence_summary.append(f"Direct evidence: {humanize_text(item['text'], dataframe=dataframe)}")
    for item in indirect_items[:2]:
        supporting_evidence_summary.append(f"Indirect evidence: {humanize_text(item['text'], dataframe=dataframe)}")
    for item in supporting_items[:2]:
        supporting_evidence_summary.append(f"Supporting evidence: {humanize_text(item['text'], dataframe=dataframe)}")
    if conflicting_items:
        for item in conflicting_items[:2]:
            supporting_evidence_summary.append(f"Conflicting evidence: {humanize_text(item['text'], dataframe=dataframe)}")
    if not supporting_evidence_summary:
        supporting_evidence_summary.append("No structured evidence was available to support a defensible answer yet.")

    facts: List[str] = []
    for item in direct_items[:3]:
        facts.append(humanize_text(item["text"], dataframe=dataframe))
    if not facts:
        facts.append("No observed fact directly resolves the question yet.")

    analytical_interpretation: List[str] = []
    if direct_items:
        analytical_interpretation.append(
            f"The strongest evidence points toward {humanize_text(direct_answer_core, dataframe=dataframe)}."
        )
    if indirect_items:
        analytical_interpretation.append(
            "Some evidence is directional rather than conclusive, so it should be treated as support rather than proof."
        )
    if conflicting_items:
        analytical_interpretation.append(
            "There is still at least one competing explanation, which keeps the conclusion from being fully closed."
        )
    if not analytical_interpretation:
        analytical_interpretation.append("The analysis is still building the evidence base needed to make a firm interpretation.")

    if sufficiency_status == "yes":
        business_interpretation = (
            f"The evidence is sufficient to answer the business question with confidence: {humanize_text(direct_answer_core, dataframe=dataframe)}."
        )
    elif sufficiency_status == "partial":
        business_interpretation = (
            f"The evidence is enough to form a provisional business answer, but not enough to treat it as final: {humanize_text(direct_answer_core, dataframe=dataframe)}."
        )
    else:
        business_interpretation = (
            "The analysis is not yet sufficient for a final business answer, so the current interpretation should be treated as directional only."
        )

    missing_evidence = _derive_missing_evidence(
        question=business_question,
        evidence_items=evidence_items,
        hypotheses=hypotheses,
        knowledge_gaps=knowledge_gaps,
        memory=investigation_memory,
    )
    if sufficiency_status == "yes":
        remaining_uncertainty = [
            "The conclusion is defensible, but it should still be monitored against new evidence or business rule changes.",
        ]
    elif missing_evidence:
        remaining_uncertainty = missing_evidence
    else:
        remaining_uncertainty = [
            "The exact answer still needs more direct evidence or a business-defined validation rule.",
        ]

    if ranked_suggestions:
        best_next = ranked_suggestions[0]
        recommended_next = [
            format_suggestion_line(best_next, index=1, dataframe=dataframe),
        ]
        if len(ranked_suggestions) > 1:
            recommended_next.append(
                "Other options remain available, but this one appears to offer the clearest next gain in certainty."
            )
    else:
        recommended_next = [
            "No distinct follow-up has been ranked yet; the next step should clarify the missing evidence or validate the current interpretation.",
        ]

    evidence_confidence = min(100, evidence_score)
    interpretation_confidence = min(100, interpretation_score)
    business_conclusion_confidence = min(100, business_score)
    recommendation_confidence = min(100, recommendation_score)

    confidence_bundle = _confidence_bundle(
        status=sufficiency_status,
        evidence_score=evidence_confidence,
        interpretation_score=interpretation_confidence,
        business_score=business_conclusion_confidence,
        recommendation_score=recommendation_confidence,
    )

    answer_payload = {
        "confidence": confidence_bundle,
        "evidence_breakdown": {
            "direct": direct_items,
            "indirect": indirect_items,
            "supporting": supporting_items,
            "conflicting": conflicting_items,
            "missing": missing_evidence,
        },
        "answer_position": answer_position,
        "key_assumptions": missing_evidence[:4] if sufficiency_status != "yes" else [
            "The answer remains contingent on the current evidence base staying stable.",
        ],
        "remaining_uncertainty": remaining_uncertainty[:4],
        "recommended_next_investigation": recommended_next[:3],
    }

    semantic_reasoning_status = "deterministic_only"
    semantic_answer = None
    needs_semantic_synthesis = sufficiency_status != "yes" or directness < 2 or answer_position != "direct"
    if needs_semantic_synthesis:
        semantic_answer, semantic_reasoning_status = _semantic_synthesize_answer(
            business_question=business_question,
            deterministic_answer=direct_answer_core,
            answer_status=sufficiency_status,
            evidence_items=evidence_items,
            missing_evidence=missing_evidence,
            confidence_bundle=confidence_bundle,
            current_understanding=current_text,
            business_interpretation=business_interpretation,
        )
        if semantic_answer:
            semantic_status = _normalize_text(semantic_answer.get("answer_status") or sufficiency_status).lower()
            if semantic_status in {"direct", "partial", "insufficient"}:
                answer_position = {
                    "direct": "direct",
                    "partial": "closest_defensible",
                    "insufficient": "needs_more_evidence",
                }[semantic_status]
            direct_answer = _normalize_text(semantic_answer.get("direct_answer") or direct_answer)
            business_interpretation = _normalize_text(semantic_answer.get("business_interpretation") or business_interpretation)
            supporting_evidence_summary = _unique_lines(semantic_answer.get("supporting_evidence_summary") or supporting_evidence_summary)
            facts = _unique_lines(semantic_answer.get("observed_facts") or facts)
            analytical_interpretation = _unique_lines(semantic_answer.get("analytical_interpretation") or analytical_interpretation)
            missing_evidence_override = _unique_lines(semantic_answer.get("key_assumptions") or [])
            if missing_evidence_override:
                missing_evidence = missing_evidence_override
            remaining_uncertainty_override = _unique_lines(semantic_answer.get("remaining_uncertainty") or [])
            if remaining_uncertainty_override:
                remaining_uncertainty = remaining_uncertainty_override
            recommended_override = _unique_lines(semantic_answer.get("recommended_next_investigation") or [])
            if recommended_override:
                recommended_next = recommended_override[:3]
            semantic_reasoning = _normalize_text(semantic_answer.get("reasoning") or "")
        else:
            semantic_reasoning = ""
    else:
        semantic_reasoning = ""

    answer_payload.update(
        {
            "answer_position": answer_position,
            "direct_answer": direct_answer,
            "business_interpretation": business_interpretation,
            "reasoning": semantic_reasoning if needs_semantic_synthesis else "",
            "key_assumptions": missing_evidence[:4] if sufficiency_status != "yes" else [
                "The answer remains contingent on the current evidence base staying stable.",
            ],
            "remaining_uncertainty": remaining_uncertainty[:4],
            "recommended_next_investigation": recommended_next[:3],
            "semantic_reasoning_status": semantic_reasoning_status,
        }
    )

    decision_bundle = build_investigation_decision_bundle(
        business_question=business_question,
        evidence=evidence,
        answer=answer_payload,
        hypotheses=hypotheses,
        investigation_memory=investigation_memory,
        collaborative_mode=bool(evidence.get("collaborative_session")),
        dataframe=dataframe,
    )
    diagnostics = decision_bundle["confidence_diagnostics"]
    diagnostics_sections = decision_bundle["confidence_diagnostics_sections"]
    decision = decision_bundle["investigation_decision"]
    decision_sections = decision_bundle["investigation_decision_sections"]
    if diagnostics.get("recommendation_alignment", {}).get("should_stop"):
        recommended_next = [decision.get("recommended_next_step") or diagnostics.get("evidence_sufficiency", {}).get("stopping_recommendation") or "Finish the investigation."]

    return {
        "business_question": business_question,
        "direct_answer": direct_answer,
        "best_available_answer": humanize_text(direct_answer_core, dataframe=dataframe) if direct_answer_core else direct_answer,
        "answer_position": answer_position,
        "supporting_evidence_summary": supporting_evidence_summary,
        "observed_facts": facts,
        "analytical_interpretation": analytical_interpretation,
        "business_interpretation": business_interpretation,
        "key_assumptions": missing_evidence[:4] if sufficiency_status != "yes" else [
            "The answer remains contingent on the current evidence base staying stable.",
        ],
        "remaining_uncertainty": remaining_uncertainty[:4],
        "recommended_next_investigation": recommended_next[:3],
        "evidence_breakdown": {
            "direct": direct_items,
            "indirect": indirect_items,
            "supporting": supporting_items,
            "conflicting": conflicting_items,
            "missing": missing_evidence,
        },
        "confidence": confidence_bundle,
        "confidence_diagnostics": diagnostics,
        "confidence_diagnostics_sections": diagnostics_sections,
        "investigation_decision": decision,
        "investigation_decision_sections": decision_sections,
        "semantic_reasoning_status": semantic_reasoning_status,
        "semantic_reasoning": semantic_reasoning if needs_semantic_synthesis else "",
    }


def build_answer_synthesis_sections(answer: Dict[str, Any]) -> Dict[str, List[str]]:
    if not answer:
        return {}

    question = _normalize_text(answer.get("business_question"))
    sufficiency = answer.get("confidence") or {}
    status = _normalize_text(sufficiency.get("status") or "").lower()
    status_line = {
        "yes": "Yes",
        "partial": "Partially",
        "no": "No",
    }.get(status, _confidence_label(status or sufficiency.get("evidence", {}).get("score")))

    sections = {
        "Direct Answer": [answer.get("direct_answer") or answer.get("best_available_answer") or "No defensible answer was produced."],
        "Evidence Sufficiency Check": [
            f"Do I have enough evidence to answer? {status_line}.",
            _normalize_text(
                " ".join(
                    [
                        f"Evidence confidence: {sufficiency.get('evidence', {}).get('label', 'Unknown')} ({sufficiency.get('evidence', {}).get('score', 'n/a')}).",
                        f"Interpretation confidence: {sufficiency.get('interpretation', {}).get('label', 'Unknown')} ({sufficiency.get('interpretation', {}).get('score', 'n/a')}).",
                        f"Business conclusion confidence: {sufficiency.get('business', {}).get('label', 'Unknown')} ({sufficiency.get('business', {}).get('score', 'n/a')}).",
                        f"Recommendation confidence: {sufficiency.get('recommendation', {}).get('label', 'Unknown')} ({sufficiency.get('recommendation', {}).get('score', 'n/a')}).",
                    ]
                )
            ),
            _normalize_text(
                " ".join(
                    [
                        "If yes, the answer is stated directly.",
                        "If partial, the answer is provisional and assumptions are named.",
                        "If no, the report names the exact evidence that is still missing.",
                    ]
                )
            ),
        ],
        "Supporting Evidence Summary": _question_relevant_lines(question, answer.get("supporting_evidence_summary") or []),
        "Observed Facts": _question_relevant_lines(question, answer.get("observed_facts") or []),
        "Analytical Interpretation": _question_relevant_lines(question, answer.get("analytical_interpretation") or []),
        "Business Interpretation": _unique_lines([answer.get("business_interpretation")] if answer.get("business_interpretation") else []),
        "Semantic Interpretation": _unique_lines([answer.get("semantic_reasoning")] if answer.get("semantic_reasoning") else []),
        "Key Assumptions": _unique_lines(answer.get("key_assumptions") or []),
        "Remaining Uncertainty": _unique_lines(answer.get("remaining_uncertainty") or []),
        "Recommended Next Investigation": _unique_lines(answer.get("recommended_next_investigation") or []),
    }
    diagnostics = answer.get("confidence_diagnostics") or {}
    diagnostics_sections = answer.get("confidence_diagnostics_sections") or {}
    decision = answer.get("investigation_decision") or {}
    decision_sections = answer.get("investigation_decision_sections") or {}
    if not diagnostics_sections and diagnostics:
        diagnostics_sections = build_confidence_diagnostics_sections(diagnostics)
    if diagnostics_sections:
        sections.update(diagnostics_sections)
    if not decision_sections and decision:
        decision_sections = build_investigation_decision_sections(decision)
    if decision_sections:
        sections.update(decision_sections)
    return {key: [line for line in value if _normalize_text(line)] for key, value in sections.items() if any(_normalize_text(line) for line in value)}


def render_answer_synthesis_report(answer: Dict[str, Any]) -> str:
    sections = build_answer_synthesis_sections(answer)
    if not sections:
        return ""
    lines: List[str] = ["================ ANSWER SYNTHESIS ================"]
    for section_title, body_lines in sections.items():
        lines.append("")
        lines.append(section_title)
        for line in body_lines:
            lines.append(f"- {line}")
    return "\n".join(lines)
