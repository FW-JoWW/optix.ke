from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from state.state import AnalystState

from .answer_synthesis import build_answer_synthesis_sections, synthesize_answer
from .narration import format_suggestion_line, humanize_text, suggestion_impact_percent


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _best_answer_text(memory: Dict[str, Any], session: Dict[str, Any], fallback: str = "") -> str:
    session_memory = {}
    if isinstance(session, dict):
        session_memory = session.get("investigation_memory") or session.get("collaborative_memory") or {}
    anchored_answer = _normalize_text((memory.get("best_answer") or {}).get("answer"))
    if anchored_answer:
        return anchored_answer
    session_anchored_answer = _normalize_text((session_memory.get("best_answer") or {}).get("answer"))
    if session_anchored_answer:
        return session_anchored_answer
    for value in [
        memory.get("current_understanding"),
        session_memory.get("current_understanding"),
        session.get("current_understanding") if isinstance(session, dict) else "",
        fallback,
    ]:
        text = _normalize_text(value)
        if text:
            return text
    return ""


def _non_empty_lines(lines: Sequence[str]) -> List[str]:
    return [line for line in lines if _normalize_text(line)]


def _unique_non_empty_lines(lines: Sequence[str]) -> List[str]:
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


def _question_relevant_lines(question: str, lines: Sequence[str]) -> List[str]:
    question_terms = _tokens(question)
    if not question_terms:
        return _unique_non_empty_lines(lines)
    relevant: List[str] = []
    for line in _unique_non_empty_lines(lines):
        lowered = line.lower()
        if _tokens(line) & question_terms:
            relevant.append(line)
            continue
        if lowered.startswith(("no ", "the evidence", "the analysis", "the answer", "if yes", "if partial", "if no")):
            relevant.append(line)
    return relevant


def _tokens(value: Any) -> set[str]:
    text = _normalize_text(value).lower()
    return {token for token in text.split() if len(token) >= 4}


def _overlap_score(left: Any, right: Any) -> int:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    return len(left_tokens & right_tokens)


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


def _confidence_reason(value: Any, evidence_count: int, gap_count: int, directness: int) -> str:
    reasons: List[str] = []
    if evidence_count:
        reasons.append(f"{evidence_count} evidence item(s) were captured")
    if directness >= 2:
        reasons.append("the current evidence speaks directly to the original question")
    elif directness == 1:
        reasons.append("the evidence is directionally relevant but not yet fully conclusive")
    else:
        reasons.append("the evidence still only partially addresses the original question")
    if gap_count:
        reasons.append(f"{gap_count} analytical gap(s) remain")
    if isinstance(value, (int, float)):
        if float(value) >= 75:
            reasons.append("the evidence is internally consistent")
        elif float(value) < 45:
            reasons.append("the evidence remains narrow or mixed")
    return "; ".join(reasons) if reasons else "The confidence assessment is based on the current evidence quality."


def _latest_checkpoint(session: Dict[str, Any]) -> Dict[str, Any]:
    checkpoints = session.get("checkpoint_summaries") or []
    if isinstance(checkpoints, list) and checkpoints:
        latest = checkpoints[-1]
        if isinstance(latest, dict):
            return latest
    return {}


def _all_checkpoints(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    checkpoints = session.get("checkpoint_summaries") or []
    if isinstance(checkpoints, list):
        return [item for item in checkpoints if isinstance(item, dict)]
    return []


def _summary_lines(summary: Dict[str, Any], preferred_keys: Sequence[str]) -> List[str]:
    lines: List[str] = []
    for key in preferred_keys:
        value = summary.get(key)
        if not value:
            continue
        if isinstance(value, list):
            for item in value[:2]:
                text = _normalize_text(item)
                if text:
                    lines.append(text)
        else:
            text = _normalize_text(value)
            if text:
                lines.append(text)
    return lines


def _direct_answer(question: str, answer: str, directness: int) -> str:
    question_text = humanize_text(question)
    answer_text = humanize_text(answer)
    if directness >= 2 and answer_text:
        return answer_text
    if answer_text:
        return (
            f"The investigation has not fully answered {question_text}. "
            f"The strongest evidence currently points to: {answer_text}."
        )
    return (
        f"The investigation has not yet answered {question_text}. "
        "The available evidence is still too limited to give a defensible direct answer."
    )


def _evidence_lines(evidence_items: Sequence[Dict[str, Any]], top_stories: Sequence[Dict[str, Any]], dataframe: Any = None) -> List[str]:
    lines: List[str] = []
    for item in list(evidence_items)[:4]:
        statement = humanize_text(item.get("statement") or item.get("finding") or item.get("summary") or "", dataframe=dataframe)
        if not statement:
            continue
        confidence = item.get("confidence")
        lines.append(f"{statement} ({_confidence_label(confidence)} confidence)" if confidence is not None else statement)
    if lines:
        return lines
    for story in list(top_stories)[:3]:
        statement = humanize_text(story.get("insight") or "", dataframe=dataframe)
        if statement:
            confidence = story.get("confidence")
            lines.append(f"{statement} ({_confidence_label(confidence)} confidence)" if confidence is not None else statement)
    return lines or ["No strong supporting evidence has been captured yet."]


def _question_contribution(original_question: str, task_focus: str, answer: str, directness: int) -> str:
    if directness >= 2:
        return "This task directly strengthens the answer to the original business question."
    if directness == 1:
        return "This task contributes indirectly by narrowing the evidence around the original business question."
    if task_focus:
        return (
            f"This task contributes indirectly because it clarifies {task_focus}, "
            "which may influence the original question but does not answer it on its own."
        )
    return "This task contributes indirectly by reducing uncertainty around the original business question."


def _gap_lines(
    original_question: str,
    answer: str,
    evidence_items: Sequence[Dict[str, Any]],
    hypotheses: Sequence[Dict[str, Any]],
    next_steps: Sequence[Dict[str, Any]],
    last_failure: Any,
) -> List[str]:
    gaps: List[str] = []
    if not evidence_items:
        gaps.append("No direct evidence has been collected yet.")
    if not any(str(item.get("status", "")).lower() == "supported" for item in hypotheses):
        gaps.append("No hypothesis has yet been strongly confirmed.")
    if not next_steps:
        gaps.append("No follow-up investigation has been queued.")
    if isinstance(last_failure, dict) and last_failure:
        failure_reason = _normalize_text(last_failure.get("reason") or last_failure.get("message"))
        if failure_reason:
            gaps.append(failure_reason)
    if _overlap_score(original_question, answer) == 0:
        gaps.append("The current answer is adjacent to the question but does not directly resolve it.")
    return gaps or ["No major analytical gap is currently visible, although the conclusion still needs confirmation."]


def _progress_lines(
    original_question: str,
    answer: str,
    evidence_items: Sequence[Dict[str, Any]],
    hypotheses: Sequence[Dict[str, Any]],
    next_steps: Sequence[Dict[str, Any]],
    checkpoints: Sequence[Dict[str, Any]],
    confidence: Any,
) -> List[str]:
    directness = _overlap_score(original_question, answer)
    question_understanding = "High" if directness >= 2 else "Moderate" if directness == 1 else "Low"
    evidence_collected = "High" if len(evidence_items) >= 4 else "Moderate" if len(evidence_items) >= 2 else "Low"
    root_cause = "High" if any(str(item.get("status", "")).lower() == "supported" for item in hypotheses) else "Moderate" if len(hypotheses) >= 2 else "Low"
    business_confidence = _confidence_label(confidence)

    remaining = 100
    remaining -= 25 if question_understanding == "High" else 10 if question_understanding == "Moderate" else 0
    remaining -= 25 if evidence_collected == "High" else 10 if evidence_collected == "Moderate" else 0
    remaining -= 25 if root_cause == "High" else 10 if root_cause == "Moderate" else 0
    remaining -= 10 if business_confidence == "High" else 5 if business_confidence == "Moderate" else 0
    if next_steps:
        remaining -= 10
    remaining = max(5, min(95, remaining))

    lines = [
        f"Question understanding: {question_understanding}",
        f"Evidence collected: {evidence_collected}",
        f"Root-cause understanding: {root_cause}",
        f"Business confidence: {business_confidence}",
        f"Remaining investigation: approximately {remaining}%",
    ]
    if checkpoints:
        lines.append(f"Checkpoint continuity: {len(checkpoints)} checkpoint(s) have been accumulated so far.")
    return lines


def _planning_candidates(suggestions: Sequence[Dict[str, Any]], original_question: str, dataframe: Any = None) -> List[Tuple[str, str, str, str, str, str, int | None, int, int, int]]:
    ranked = [item for item in suggestions if isinstance(item, dict)]
    candidates: List[Tuple[str, str, str, str, str, str, int | None, int, int, int]] = []
    for suggestion in ranked:
        title = humanize_text(suggestion.get("title") or "Next investigation", dataframe=dataframe)
        request = humanize_text(suggestion.get("request") or suggestion.get("description") or "Review the next step.", dataframe=dataframe)
        impact = suggestion_impact_percent(suggestion)
        question_relevance = _overlap_score(original_question, title) + _overlap_score(original_question, request)
        uncertainty_reduction = 70 if any(term in _normalize_text(request).lower() for term in ["challenge", "falsify", "test", "verify"]) else 60 if any(term in _normalize_text(request).lower() for term in ["compare", "contrast"]) else 55
        business_value = min(100, 40 + question_relevance * 15)
        analytical_confidence = int(suggestion.get("confidence", 50)) if isinstance(suggestion.get("confidence"), (int, float)) else 50
        candidates.append(
            (
                title,
                request,
                humanize_text(suggestion.get("reason") or suggestion.get("justification") or "This path follows directly from the current evidence.", dataframe=dataframe),
                humanize_text(suggestion.get("builds_on") or suggestion.get("continuation") or "It extends the current evidence rather than restarting the analysis.", dataframe=dataframe),
                humanize_text(suggestion.get("confidence_gain") or "Expected to improve confidence by testing the remaining uncertainty.", dataframe=dataframe),
                humanize_text(suggestion.get("business_value") or suggestion.get("value") or "Expected to clarify the business implication.", dataframe=dataframe),
                impact,
                question_relevance,
                uncertainty_reduction,
                business_value + analytical_confidence // 2,
            )
        )
    candidates.sort(key=lambda item: ((item[1] and item[7]) or 0, item[8], item[9], item[6] or 0), reverse=True)
    return candidates


def _planning_lines(suggestions: Sequence[Dict[str, Any]], original_question: str, dataframe: Any = None) -> Tuple[List[str], str]:
    candidates = _planning_candidates(suggestions, original_question, dataframe=dataframe)
    if not candidates:
        return ["No distinct follow-up is currently suggested; the investigation is ready for consolidation or closure."], "No alternative path currently offers a clear advantage."

    lines: List[str] = []
    best = candidates[0]
    for index, (title, request, why, build, confidence_gain, value, impact, relevance, uncertainty, business_score) in enumerate(candidates[:3], start=1):
        impact_text = f"{impact}%" if impact is not None else "unknown"
        lines.append(f"{index}. {title}")
        lines.append(f"   What it investigates: {request}")
        lines.append(f"   Why it is valuable: {why}")
        lines.append(f"   How it builds on current findings: {build}")
        lines.append(f"   Expected analytical impact: {impact_text}")
        lines.append(f"   Expected confidence improvement: {confidence_gain}")
        lines.append(f"   Expected business value: {value}")
    best_title, best_request, best_why, best_build, best_confidence_gain, best_value, best_impact, best_relevance, best_uncertainty, best_business_score = best
    best_impact_text = f"{best_impact}%" if best_impact is not None else "unknown"
    justification = (
        f"The best next step is {best_title}, because it most directly reduces the uncertainty around the original business question. "
        f"It investigates {best_request.lower()} and is preferred over the other options because it has the strongest combination of question relevance, uncertainty reduction, and business value ({best_impact_text} expected impact)."
    )
    if best_why:
        justification += f" {best_why}"
    if best_build:
        justification += f" {best_build}"
    if best_confidence_gain:
        justification += f" {best_confidence_gain}"
    if best_value:
        justification += f" Business value signal: {best_value}."
    return lines, justification


def _section_map_from_lines(section_items: Sequence[Tuple[str, Sequence[str]]]) -> Dict[str, List[str]]:
    return {title: _non_empty_lines(list(lines)) for title, lines in section_items if _non_empty_lines(list(lines))}


def _integrity_lines(integrity: Dict[str, Any] | None) -> List[str]:
    if not integrity:
        return []
    lines = [
        f"Question relevance: {integrity.get('question_relevance', {}).get('level', 'Unknown')} ({integrity.get('question_relevance', {}).get('score', 'n/a')})",
        f"Investigation continuity: {integrity.get('continuity', {}).get('level', 'Unknown')} ({integrity.get('continuity', {}).get('score', 'n/a')})",
        f"Information gain: {integrity.get('information_gain', {}).get('level', 'Unknown')} ({integrity.get('information_gain', {}).get('score', 'n/a')})",
        f"Analytical validity: {integrity.get('analytical_validity', {}).get('level', 'Unknown')} ({integrity.get('analytical_validity', {}).get('score', 'n/a')})",
        f"Overall integrity: {integrity.get('overall', {}).get('level', 'Unknown')} ({integrity.get('overall', {}).get('score', 'n/a')})",
        f"Decision: {'promoted to current understanding' if integrity.get('should_promote') else 'retained as supporting evidence'}",
    ]
    reason = _normalize_text(integrity.get("reason"))
    if reason:
        lines.append(reason)
    return lines


def _traceability_lines(traceability: Dict[str, Any] | None) -> List[str]:
    if not traceability:
        return []
    answered = [humanize_text(item) for item in (traceability.get("questions_answered") or []) if _normalize_text(item)]
    remaining = [humanize_text(item) for item in (traceability.get("questions_remaining") or []) if _normalize_text(item)]
    lines = [
        f"Original question: {humanize_text(traceability.get('original_question') or '')}",
        f"Current hypothesis: {humanize_text(traceability.get('current_hypothesis') or '')}",
        f"Purpose of this task: {humanize_text(traceability.get('purpose_of_task') or '')}",
        f"Expected contribution: {humanize_text(traceability.get('expected_contribution') or '')}",
        f"Actual contribution: {humanize_text(traceability.get('actual_contribution') or '')}",
        f"Question relevance: {traceability.get('question_relevance', {}).get('level', 'Unknown')}",
        f"Questions answered: {', '.join(answered) or 'None'}",
        f"Questions remaining: {', '.join(remaining) or 'None'}",
    ]
    return lines


def compose_checkpoint_narrative(
    checkpoint: Dict[str, Any],
    dataframe: Any = None,
    original_question: str | None = None,
    session: Dict[str, Any] | None = None,
) -> List[str]:
    if not isinstance(checkpoint, dict) or not checkpoint:
        return []

    session = session or {}
    original_question = original_question or session.get("original_question") or checkpoint.get("task_request") or checkpoint.get("task_title") or "the investigation"
    integrity = checkpoint.get("integrity") or {}
    traceability = checkpoint.get("traceability") or {}
    task_focus = humanize_text(checkpoint.get("task_request") or checkpoint.get("task_title") or "the investigation", dataframe=dataframe)
    task_finding = humanize_text(
        checkpoint.get("task_finding")
        or checkpoint.get("current_understanding")
        or checkpoint.get("narrative")
        or checkpoint.get("analysis_story")
        or checkpoint.get("report_excerpt")
        or "The investigation has not yet reached a stable conclusion.",
        dataframe=dataframe,
    )
    promoted_understanding = humanize_text(
        checkpoint.get("current_understanding")
        or session.get("current_understanding")
        or checkpoint.get("narrative")
        or checkpoint.get("analysis_story")
        or "The investigation has not yet reached a stable conclusion.",
        dataframe=dataframe,
    )
    direct_source = task_finding if integrity.get("should_promote", True) else promoted_understanding
    directness = _overlap_score(original_question, direct_source)
    answer = humanize_text(direct_source, dataframe=dataframe)
    evidence_lines = _evidence_lines([checkpoint] if checkpoint.get("statement") else [], [], dataframe=dataframe)
    if checkpoint.get("analysis_steps"):
        step_lines = _non_empty_lines([humanize_text(line, dataframe=dataframe) for line in checkpoint.get("analysis_steps")[:4]])
        if step_lines:
            evidence_lines = step_lines
    if not evidence_lines:
        evidence_lines = ["No strong supporting evidence has been recorded for this checkpoint."]
    if checkpoint.get("task_finding"):
        evidence_lines.insert(0, f"Task finding: {task_finding}")

    gap_lines = _gap_lines(
        original_question=original_question,
        answer=answer,
        evidence_items=[checkpoint] if checkpoint.get("statement") else [],
        hypotheses=[],
        next_steps=checkpoint.get("next_investigations") or [],
        last_failure=checkpoint.get("failure_reason") or checkpoint.get("failure_message") or "",
    )

    synthesis = synthesize_answer(
        business_question=original_question,
        evidence={
            "top_stories": [checkpoint] if checkpoint.get("task_finding") else [],
            "judgment_summary": {
                "summary": checkpoint.get("current_understanding") or checkpoint.get("narrative"),
                "global_confidence": checkpoint.get("confidence"),
            },
            "collaborative_session": session,
        },
        hypotheses=[],
        current_understanding=checkpoint.get("current_understanding") or checkpoint.get("task_finding") or checkpoint.get("narrative"),
        confidence=checkpoint.get("confidence"),
        knowledge_gaps=gap_lines,
        investigation_memory=session.get("investigation_memory") or {},
        dataframe=dataframe,
    )
    synthesis_sections = build_answer_synthesis_sections(synthesis)

    planning_lines, justification = _planning_lines(checkpoint.get("next_investigations") or [], original_question, dataframe=dataframe)
    if checkpoint.get("next_investigations") and isinstance(checkpoint.get("next_investigations"), list) and checkpoint.get("next_investigations"):
        planning_lines = planning_lines[:9]
    confidence_value = checkpoint.get("confidence")
    current_focus = task_focus
    contribution = _question_contribution(original_question, current_focus, answer, directness)

    progress_lines = _progress_lines(
        original_question=original_question,
        answer=answer,
        evidence_items=evidence_lines,
        hypotheses=[],
        next_steps=checkpoint.get("next_investigations") or [],
        checkpoints=[checkpoint],
        confidence=confidence_value,
    )
    focus_lines = [
        f"Original question: {humanize_text(original_question, dataframe=dataframe)}",
        f"Current investigation focus: {current_focus}",
        f"Current understanding: {promoted_understanding}",
        f"Contribution to original question: {contribution}",
        f"Reason for current investigation: {humanize_text(checkpoint.get('task_request') or checkpoint.get('task_title') or 'This task was selected to reduce uncertainty around the original question.', dataframe=dataframe)}",
    ]
    if not integrity.get("should_promote", True):
        focus_lines.append("Integrity outcome: this finding was retained as supporting evidence and did not replace the current understanding.")
    if directness == 0:
        focus_lines.append("This task is only indirectly related to the original question and should be treated as supporting context rather than a final answer.")

    hypothesis_lines = [
        f"Questions answered: {'High' if directness >= 2 else 'Moderate' if directness == 1 else 'Low'}",
        f"Questions remaining: {'Moderate' if gap_lines else 'Low'}",
        f"Current best explanation: {answer}",
        "Alternative explanations remain plausible until the evidence is stronger.",
    ]
    if gap_lines:
        hypothesis_lines.append(f"Remaining uncertainty: {gap_lines[0]}")

    awaiting_lines = [
        "This checkpoint exists to help the next analyst decide whether to deepen the current line of inquiry, challenge the finding, or close the investigation.",
        "The next step should be chosen based on how much it reduces uncertainty about the original business question.",
    ]
    if justification:
        awaiting_lines.append(f"Why the next step matters: {justification}")

    sections = _section_map_from_lines(
        [
            ("Direct Answer", synthesis_sections.get("Direct Answer") or [ _direct_answer(original_question, answer, directness) ]),
            ("Evidence Sufficiency Check", synthesis_sections.get("Evidence Sufficiency Check", [])),
            ("Supporting Evidence Summary", synthesis_sections.get("Supporting Evidence Summary", [])),
            ("Observed Facts", synthesis_sections.get("Observed Facts", [])),
            ("Analytical Interpretation", synthesis_sections.get("Analytical Interpretation", [])),
            ("Business Interpretation", synthesis_sections.get("Business Interpretation", [])),
            ("Key Assumptions", synthesis_sections.get("Key Assumptions", [])),
            ("Remaining Uncertainty", synthesis_sections.get("Remaining Uncertainty", [])),
            ("Recommended Next Investigation", synthesis_sections.get("Recommended Next Investigation", [])),
            ("Original Question", [humanize_text(original_question, dataframe=dataframe)]),
            ("Current Investigation Focus", focus_lines),
            ("What Has Been Established", evidence_lines),
            ("Contribution to Original Question", [contribution]),
            ("Questions Remaining", gap_lines),
            ("Hypothesis Position", hypothesis_lines),
            ("Investigation Progress", progress_lines),
            ("Confidence Assessment", [
                f"Confidence: {_confidence_label(confidence_value)}",
                _confidence_reason(confidence_value, len(evidence_lines), len(gap_lines), directness),
            ]),
            ("Investigation Integrity", _integrity_lines(integrity)),
            ("Question Traceability", _traceability_lines(traceability)),
            ("Investigation Planning", planning_lines),
            ("Awaiting Analyst Decision", awaiting_lines),
        ]
    )

    output_order = [
        "Direct Answer",
        "Original Question",
        "Current Investigation Focus",
        "What Has Been Established",
        "Contribution to Original Question",
        "Questions Remaining",
        "Hypothesis Position",
        "Investigation Progress",
        "Confidence Assessment",
        "Investigation Integrity",
        "Question Traceability",
        "Investigation Planning",
        "Awaiting Analyst Decision",
    ]
    lines: List[str] = []
    for section in output_order:
        body = sections.get(section) or []
        if not body:
            continue
        lines.append(section)
        lines.extend(f"- {item}" for item in body)
    return _non_empty_lines(lines)


def _guided_sections(state: AnalystState, evidence: Dict[str, Any]) -> Dict[str, List[str]]:
    guided_summaries = evidence.get("guided_checkpoint_summaries") or state.get("guided_checkpoint_summaries") or {}
    if not isinstance(guided_summaries, dict) or not guided_summaries:
        return {}

    dataframe = state.get("dataframe")
    question = humanize_text(state.get("business_question") or "the business question")
    judgment = evidence.get("judgment_summary") or {}
    top_stories = evidence.get("top_stories") or []
    llm_insights = evidence.get("llm_insights") or state.get("llm_insights") or ""
    recommended_action = (
        (evidence.get("decision_recommended_first") or {}).get("recommended_action")
        or judgment.get("recommended_first_action")
        or "Continue with the current evidence."
    )

    evidence_lines: List[str] = []
    timeline: List[str] = []
    for label, key in [
        ("Data preparation", "data_preparation"),
        ("Business understanding", "business_understanding"),
        ("Analysis strategy", "analysis_strategy"),
        ("Result review", "result_review"),
    ]:
        summary = guided_summaries.get(key) or {}
        if not isinstance(summary, dict) or not summary:
            continue
        recap = _summary_lines(summary, ["Analyst interpretation", "What happened", "Primary variables", "Why this method was selected", "Recommendation", "What I recommend"])
        if recap:
            evidence_lines.append(f"{label}: {recap[0]}")
            if len(recap) > 1:
                evidence_lines.append(f"{label} detail: {recap[1]}")
        timeline.append(label)

    if not evidence_lines:
        evidence_lines = [humanize_text(llm_insights)] if llm_insights else ["The guided checkpoints have not yet yielded a stable narrative."]

    current_understanding = humanize_text(
        _best_answer_text({}, state)
        or (judgment.get("summary") if isinstance(judgment, dict) else None)
        or (judgment.get("dominant_reasoning") if isinstance(judgment, dict) else None)
        or (top_stories[0].get("insight") if top_stories else None)
        or llm_insights
        or "The investigation is still developing."
    )
    directness = _overlap_score(question, current_understanding)
    answer = _direct_answer(question, current_understanding, directness)
    progress_lines = [
        f"Question understanding: {'High' if directness >= 2 else 'Moderate' if directness == 1 else 'Low'}",
        f"Evidence collected: {'Moderate' if len(evidence_lines) >= 2 else 'Low'}",
        f"Root-cause understanding: {'Moderate' if any('because' in line.lower() for line in evidence_lines) else 'Low'}",
        f"Business confidence: {_confidence_label(judgment.get('global_confidence'))}",
        "Remaining investigation: approximately 45%",
    ]
    planning_lines, justification = _planning_lines(
        [
            {
                "title": recommended_action,
                "request": recommended_action,
                "reason": current_understanding,
                "business_value": "It keeps the investigation aligned with the original question.",
                "confidence_gain": "Expected to strengthen the conclusion.",
                "impact_percent": 80,
                "confidence": judgment.get("global_confidence") or 50,
            }
        ],
        question,
        dataframe=None,
    )
    gap_lines = [
        "The investigation still needs either stronger evidence, a more precise business definition, or a clearer alternative explanation.",
    ]
    if not recommended_action or recommended_action == "Continue with the current evidence.":
        gap_lines.append("No clearly superior follow-up has been identified yet.")

    synthesis = synthesize_answer(
        business_question=question,
        evidence=evidence,
        hypotheses=[],
        current_understanding=current_understanding,
        confidence=judgment.get("global_confidence"),
        knowledge_gaps=gap_lines,
        investigation_memory=state.get("collaborative_memory") or {},
        dataframe=dataframe,
    )
    synthesis_sections = build_answer_synthesis_sections(synthesis)

    sections = _section_map_from_lines(
        [
            ("Direct Answer", synthesis_sections.get("Direct Answer") or [answer]),
            ("Evidence Sufficiency Check", synthesis_sections.get("Evidence Sufficiency Check", [])),
            ("Supporting Evidence Summary", synthesis_sections.get("Supporting Evidence Summary", [])),
            ("Observed Facts", synthesis_sections.get("Observed Facts", [])),
            ("Analytical Interpretation", synthesis_sections.get("Analytical Interpretation", [])),
            ("Business Interpretation", synthesis_sections.get("Business Interpretation", [])),
            ("Key Assumptions", synthesis_sections.get("Key Assumptions", [])),
            ("Remaining Uncertainty", synthesis_sections.get("Remaining Uncertainty", [])),
            ("Recommended Next Investigation", synthesis_sections.get("Recommended Next Investigation", [])),
            ("Original Question", [question]),
            ("Current Investigation Focus", [
                f"The current focus is the question: {question}.",
                f"Contribution to original question: { _question_contribution(question, current_understanding, answer, directness) }",
                f"Reason for current investigation: The guided workflow is using the current checkpoint to reduce uncertainty around the original question.",
            ]),
            ("What Has Been Established", evidence_lines),
            ("Contribution to Original Question", [
                _question_contribution(question, current_understanding, answer, directness),
            ]),
            ("Questions Remaining", gap_lines),
            ("Hypothesis Position", [
                f"Current best explanation: {current_understanding}",
                "Alternative explanations remain plausible until the evidence is stronger.",
            ]),
            ("Investigation Progress", progress_lines),
            ("Confidence Assessment", [
                f"Confidence: {_confidence_label(judgment.get('global_confidence'))}",
                _confidence_reason(judgment.get("global_confidence"), len(evidence_lines), len(gap_lines), directness),
            ]),
            ("Investigation Planning", planning_lines),
            ("Awaiting Analyst Decision", [
                "The guided checkpoint is ready for the analyst to accept the current direction, refine it, or ask for a different line of investigation.",
                f"Checkpoint timeline so far: {', '.join(timeline)}." if timeline else "",
                f"Why this recommendation is preferred: {justification}" if justification else "",
            ]),
        ]
    )
    return sections


def _collaborative_sections(state: AnalystState, evidence: Dict[str, Any]) -> Dict[str, List[str]]:
    session = evidence.get("collaborative_session") or state.get("collaborative_session") or {}
    if not session:
        return {}

    dataframe = state.get("dataframe")
    memory = session.get("investigation_memory") or state.get("collaborative_memory") or {}
    evidence_store = session.get("evidence_store") or {}
    top_stories = evidence.get("top_stories") or []
    suggestions = session.get("ai_suggestions") or state.get("collaborative_suggestions") or []
    checkpoints = _all_checkpoints(session or state.get("collaborative_session") or {})
    last_failure = session.get("last_failure") or state.get("collaborative_last_failure") or memory.get("last_failure") or {}
    hypotheses = session.get("hypotheses") or {}
    decision_log = session.get("decision_log") or state.get("collaborative_decision_log") or []

    question = humanize_text(session.get("original_question") or state.get("business_question") or "the business question", dataframe=dataframe)
    objective = humanize_text(session.get("objective") or question, dataframe=dataframe)
    current_understanding = humanize_text(
        _best_answer_text(memory, session)
        or (top_stories[0].get("insight") if top_stories else None)
        or (evidence.get("judgment_summary") or {}).get("dominant_reasoning")
        or (evidence.get("judgment_summary") or {}).get("summary")
        or "The investigation is still developing.",
        dataframe=dataframe,
    )
    directness = _overlap_score(question, current_understanding)
    direct_answer = _direct_answer(question, current_understanding, directness)

    supporting_items = list(evidence_store.values()) or top_stories
    evidence_lines = _evidence_lines(supporting_items, top_stories, dataframe=dataframe)
    if decision_log:
        evidence_lines.append(f"Decision log entries: {len(decision_log)}")

    supported = [item for item in hypotheses.values() if str(item.get("status", "")).lower() == "supported"]
    rejected = [item for item in hypotheses.values() if str(item.get("status", "")).lower() == "rejected"]
    inconclusive = [item for item in hypotheses.values() if str(item.get("status", "")).lower() == "inconclusive"]
    gap_lines = _gap_lines(
        original_question=question,
        answer=current_understanding,
        evidence_items=supporting_items,
        hypotheses=list(hypotheses.values()),
        next_steps=suggestions,
        last_failure=last_failure,
    )
    if rejected:
        gap_lines.append(f"{len(rejected)} hypothesis/hypotheses have been rejected and need not be revisited unless new evidence emerges.")
    if inconclusive:
        gap_lines.append(f"{len(inconclusive)} hypothesis/hypotheses remain unresolved.")

    planning_lines, justification = _planning_lines(suggestions, question, dataframe=dataframe)
    progress_lines = _progress_lines(
        original_question=question,
        answer=current_understanding,
        evidence_items=supporting_items,
        hypotheses=list(hypotheses.values()),
        next_steps=suggestions,
        checkpoints=checkpoints,
        confidence=memory.get("confidence"),
    )
    confidence_lines = [
        f"Confidence: {_confidence_label(memory.get('confidence') or (top_stories[0].get('confidence') if top_stories else None))}",
        _confidence_reason(memory.get("confidence") or (top_stories[0].get("confidence") if top_stories else None), len(supporting_items), len(gap_lines), directness),
    ]
    if checkpoints:
        confidence_lines.append(f"{len(checkpoints)} checkpoint(s) have strengthened the traceability of the investigation.")

    synthesis = synthesize_answer(
        business_question=question,
        evidence=evidence,
        hypotheses=list(hypotheses.values()),
        current_understanding=current_understanding,
        confidence=memory.get("confidence") or (top_stories[0].get("confidence") if top_stories else None),
        knowledge_gaps=gap_lines,
        investigation_memory=memory,
        dataframe=dataframe,
    )
    synthesis_sections = build_answer_synthesis_sections(synthesis)

    return _section_map_from_lines(
        [
            ("Direct Answer", synthesis_sections.get("Direct Answer") or [direct_answer]),
            ("Evidence Sufficiency Check", synthesis_sections.get("Evidence Sufficiency Check", [])),
            ("Supporting Evidence Summary", synthesis_sections.get("Supporting Evidence Summary", [])),
            ("Observed Facts", synthesis_sections.get("Observed Facts", [])),
            ("Analytical Interpretation", synthesis_sections.get("Analytical Interpretation", [])),
            ("Business Interpretation", synthesis_sections.get("Business Interpretation", [])),
            ("Key Assumptions", synthesis_sections.get("Key Assumptions", [])),
            ("Remaining Uncertainty", synthesis_sections.get("Remaining Uncertainty", [])),
            ("Recommended Next Investigation", synthesis_sections.get("Recommended Next Investigation", [])),
            ("Original Question", [question]),
            ("Current Investigation Focus", [
                f"Working objective: {objective}",
                f"Contribution to original question: { _question_contribution(question, objective, current_understanding, directness) }",
                f"Reason for current investigation: This task is part of a longer investigation into the original business question, not an isolated analysis.",
            ]),
            ("What Has Been Established", evidence_lines),
            ("Contribution to Original Question", [
                _question_contribution(question, objective, current_understanding, directness),
            ]),
            ("Questions Remaining", gap_lines),
            ("Hypothesis Position", [
                f"Questions answered: {'High' if directness >= 2 else 'Moderate' if directness == 1 else 'Low'}",
                f"Questions remaining: {'Moderate' if gap_lines else 'Low'}",
                f"Current best explanation: {current_understanding}",
                "Alternative explanations remain plausible until the evidence is stronger.",
                f"Supported hypotheses: {len(supported)}." if supported else "",
                f"Rejected hypotheses: {len(rejected)}." if rejected else "",
                f"Inconclusive hypotheses: {len(inconclusive)}." if inconclusive else "",
            ]),
            ("Investigation Progress", progress_lines),
            ("Confidence Assessment", confidence_lines),
            ("Investigation Planning", planning_lines),
            ("Awaiting Analyst Decision", [
                "This checkpoint exists to confirm what has been established and decide the next analytical move.",
                "The next step should be chosen based on how much uncertainty it removes from the original business question.",
                f"Why the next step matters: {justification}" if justification else "",
            ]),
        ]
    )


def compose_analyst_sections(state: AnalystState, evidence: Dict[str, Any]) -> Dict[str, List[str]]:
    if state.get("mode") == "guided" and not (evidence.get("collaborative_session") or state.get("collaborative_session")):
        return _guided_sections(state, evidence)
    if evidence.get("collaborative_session") or state.get("collaborative_session"):
        return _collaborative_sections(state, evidence)
    return {}


def _final_timeline_lines(state: AnalystState, evidence: Dict[str, Any]) -> List[str]:
    session = evidence.get("collaborative_session") or state.get("collaborative_session") or {}
    checkpoints = _all_checkpoints(session)
    if not checkpoints:
        return ["No step-by-step timeline was available."]
    question = humanize_text(session.get("original_question") or state.get("business_question") or "")
    lines: List[str] = []
    for index, checkpoint in enumerate(checkpoints[:5], start=1):
        integrity = checkpoint.get("integrity") or {}
        relevance = int((integrity.get("question_relevance") or {}).get("score") or 0)
        task_request = humanize_text(checkpoint.get("task_request") or checkpoint.get("traceability", {}).get("purpose_of_task") or "")
        task_title = humanize_text(checkpoint.get("task_title") or "")
        checkpoint_text = " ".join(
            part for part in [task_request, task_title, humanize_text(checkpoint.get("current_understanding") or checkpoint.get("narrative") or "")] if part
        )
        direct_relevance = _overlap_score(question, checkpoint_text)
        if index > 1 and direct_relevance < 1:
            continue
        if relevance and relevance < 50 and direct_relevance < 1:
            continue
        title = task_title or task_request or f"Checkpoint {index}"
        summary = humanize_text(checkpoint.get("current_understanding") or checkpoint.get("narrative") or "No conclusion captured.")
        line = f"{index}. {title}: {summary}"
        if _tokens(line) & _tokens(question):
            lines.append(line)
        elif relevance >= 50 and direct_relevance >= 1:
            lines.append(line)
    return _unique_non_empty_lines(lines) or ["No step-by-step timeline was available."]


def compose_final_analyst_sections(state: AnalystState, evidence: Dict[str, Any]) -> Dict[str, List[str]]:
    session = evidence.get("collaborative_session") or state.get("collaborative_session") or {}
    if not session and state.get("mode") != "guided":
        return {}

    dataframe = state.get("dataframe")
    memory = session.get("investigation_memory") or state.get("collaborative_memory") or {}
    question = humanize_text(session.get("original_question") or state.get("business_question") or "the business question", dataframe=dataframe)
    current_understanding = humanize_text(
        _best_answer_text(memory, session)
        or (evidence.get("judgment_summary") or {}).get("summary")
        or (evidence.get("judgment_summary") or {}).get("dominant_reasoning")
        or (evidence.get("top_stories") or [{}])[0].get("insight")
        or state.get("llm_insights")
        or "The investigation is still developing.",
        dataframe=dataframe,
    )
    top_stories = evidence.get("top_stories") or []
    evidence_store = session.get("evidence_store") or {}
    suggestions = session.get("ai_suggestions") or state.get("collaborative_suggestions") or []
    confidence_value = (evidence.get("judgment_summary") or {}).get("global_confidence")
    direct_answer = _direct_answer(question, current_understanding, _overlap_score(question, current_understanding))
    supporting_items = list(evidence_store.values()) or top_stories
    supported_lines = _question_relevant_lines(question, _evidence_lines(supporting_items, top_stories, dataframe=dataframe))
    hypothesis_values = list((session.get("hypotheses") or {}).values())
    tested_lines = [
        humanize_text(item.get("hypothesis") or item.get("summary") or item.get("statement") or "", dataframe=dataframe)
        for item in hypothesis_values
        if isinstance(item, dict) and _normalize_text(item.get("hypothesis") or item.get("summary") or item.get("statement"))
    ]
    tested_lines = _question_relevant_lines(question, [line for line in tested_lines if line])
    if not tested_lines:
        tested_lines = ["The investigation has not yet surfaced a clearly named hypothesis trail."]

    alt_lines = [
        humanize_text(item.get("hypothesis") or item.get("summary") or item.get("statement") or "", dataframe=dataframe)
        for item in hypothesis_values
        if isinstance(item, dict) and str(item.get("status", "")).lower() == "rejected"
    ]
    alt_lines = _question_relevant_lines(question, [line for line in alt_lines if line])
    if not alt_lines:
        alt_lines = ["No strong alternative explanation has been established yet."]

    gap_lines = _question_relevant_lines(question, _gap_lines(question, current_understanding, supporting_items, hypothesis_values, suggestions, session.get("last_failure") or {}))
    planning_lines, justification = _planning_lines(suggestions, question, dataframe=dataframe)
    planning_lines = _question_relevant_lines(question, planning_lines)
    timeline_lines = _final_timeline_lines(state, evidence)
    integrity = memory.get("last_integrity") or ((session.get("checkpoint_summaries") or [{}])[-1].get("integrity") if session.get("checkpoint_summaries") else {})
    traceability = memory.get("last_traceability") or ((session.get("checkpoint_summaries") or [{}])[-1].get("traceability") if session.get("checkpoint_summaries") else {})
    synthesis = synthesize_answer(
        business_question=question,
        evidence=evidence,
        hypotheses=hypothesis_values,
        current_understanding=current_understanding,
        confidence=confidence_value,
        knowledge_gaps=gap_lines,
        investigation_memory=memory,
        dataframe=dataframe,
    )
    synthesis["supporting_evidence_summary"] = _question_relevant_lines(question, synthesis.get("supporting_evidence_summary") or [])
    synthesis["observed_facts"] = _question_relevant_lines(question, synthesis.get("observed_facts") or [])
    synthesis["analytical_interpretation"] = _question_relevant_lines(question, synthesis.get("analytical_interpretation") or [])
    synthesis["recommended_next_investigation"] = _question_relevant_lines(question, synthesis.get("recommended_next_investigation") or [])
    synthesis_sections = build_answer_synthesis_sections(synthesis)

    answer_lines = [
        f"Executive answer: {synthesis.get('direct_answer') or direct_answer}",
        f"Business question: {question}",
        f"Final Conclusion: {synthesis.get('best_available_answer') or current_understanding}",
    ]
    if _overlap_score(question, current_understanding) == 0:
        answer_lines.append("The investigation does not yet fully resolve the original business question.")

    return _section_map_from_lines(
        [
            ("Direct Answer", synthesis_sections.get("Direct Answer") or [synthesis.get("direct_answer") or direct_answer]),
            ("Evidence Sufficiency Check", synthesis_sections.get("Evidence Sufficiency Check", [])),
            ("Supporting Evidence Summary", synthesis_sections.get("Supporting Evidence Summary", [])),
            ("Observed Facts", synthesis_sections.get("Observed Facts", [])),
            ("Analytical Interpretation", synthesis_sections.get("Analytical Interpretation", [])),
            ("Business Interpretation", synthesis_sections.get("Business Interpretation", [])),
            ("Key Assumptions", synthesis_sections.get("Key Assumptions", [])),
            ("Remaining Uncertainty", synthesis_sections.get("Remaining Uncertainty", [])),
            ("Recommended Next Investigation", synthesis_sections.get("Recommended Next Investigation", [])),
            ("Executive Answer", answer_lines),
            ("Business Question", [question]),
            ("How Understanding Evolved", timeline_lines),
            ("Key Supporting Evidence", supported_lines),
            ("Hypotheses Tested", tested_lines),
            ("Alternative Explanations Considered", alt_lines),
            ("Remaining Uncertainty", gap_lines),
            ("Investigation Integrity", _integrity_lines(integrity)),
            ("Question Traceability", _traceability_lines(traceability)),
            ("Recommended Business Actions", [
                humanize_text((evidence.get("decision_recommended_first") or {}).get("recommended_action") or (evidence.get("judgment_summary") or {}).get("recommended_first_action") or "No immediate business action is established.", dataframe=dataframe),
                "The recommended business action should stay tied to the evidence rather than the internal workflow.",
            ]),
            ("Recommended Future Investigations", _unique_non_empty_lines(planning_lines[:9] + ([f"Why this is preferred: {justification}"] if justification else []))),
            ("Investigation Timeline", timeline_lines),
            ("Overall Confidence", [
                f"Confidence: {_confidence_label(confidence_value)}",
                _confidence_reason(confidence_value, len(supporting_items), len(gap_lines), _overlap_score(question, current_understanding)),
            ]),
        ]
    )


def render_report_sections(title: str, sections: Dict[str, List[str]]) -> str:
    lines: List[str] = [title]
    for section_title, body_lines in sections.items():
        if not body_lines:
            continue
        lines.append("")
        lines.append(section_title)
        for line in body_lines:
            lines.append(f"- {line}")
    return "\n".join(lines)


def render_analyst_report(state: AnalystState, evidence: Dict[str, Any]) -> str:
    session = evidence.get("collaborative_session") or state.get("collaborative_session") or {}
    completed = _normalize_text(session.get("current_status") or state.get("current_status") or "")
    has_guided_context = bool(evidence.get("guided_checkpoint_summaries") or state.get("guided_checkpoint_summaries"))
    is_final = (
        completed.lower() == "completed"
        or bool(session.get("final_executive_report"))
        or bool(state.get("final_report"))
        or (state.get("mode") == "guided" and has_guided_context)
    )
    sections = compose_final_analyst_sections(state, evidence) if is_final else compose_analyst_sections(state, evidence)
    if not sections:
        return ""
    title = "================ EXECUTIVE ANSWER ================" if is_final else "================ ANALYST VIEW ================"
    return render_report_sections(title, sections)
