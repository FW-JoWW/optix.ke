from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from io import StringIO
from contextlib import redirect_stdout
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from backend.graph.analyst_graph import graph
from backend.nodes.report_node import report_node
from backend.state.state import AnalystState

from .answer_synthesis import synthesize_answer
from .models import EvidenceRecord, HypothesisRecord, InvestigationSession
from .integrity import build_traceability_record, evaluate_investigation_integrity, evaluate_task_request_relevance, update_best_answer_anchor
from .narration import humanize_columns, humanize_text, suggestion_impact_percent
from .task_manager import TaskManager


def _best_answer_text(memory: Dict[str, Any], fallback: str = "") -> str:
    return (
        (memory.get("best_answer") or {}).get("answer")
        or memory.get("current_understanding")
        or fallback
    )


@dataclass
class CollaborativeRunResult:
    final_state: AnalystState
    session: Dict[str, Any]
    desk: Dict[str, Any]
    task_outputs: Dict[str, Dict[str, Any]]


def _utc_session_id() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("inv-%Y%m%d-%H%M%S")


def create_investigation_session(question: str) -> InvestigationSession:
    return InvestigationSession(
        investigation_id=_utc_session_id(),
        original_question=question,
        investigation_title=question,
        objective=question,
        current_status="active",
        investigation_memory={
            "original_question": question,
            "accepted_assumptions": [],
            "rejected_hypotheses": [],
            "previous_findings": [],
            "task_references": [],
        },
    )


def _build_task_state(base_state: AnalystState, session: InvestigationSession, task_request: str) -> AnalystState:
    state: AnalystState = deepcopy(base_state)
    state["business_question"] = task_request or base_state.get("business_question", "")
    state["mode"] = "autonomous"
    state["awaiting_user"] = False
    state["question_for_user"] = ""
    state["user_response"] = ""
    state["collaborative_session"] = session.to_dict()
    state["collaborative_tasks"] = [task.to_dict() for task in session.tasks.values()]
    state["collaborative_task_graph"] = deepcopy(session.task_graph)
    state["collaborative_queue"] = [session.tasks[task_id].to_dict() for task_id in session.queued_tasks]
    state["collaborative_evidence_store"] = {
        evidence_id: record.to_dict() for evidence_id, record in session.evidence_store.items()
    }
    state["collaborative_memory"] = deepcopy(session.investigation_memory)
    state["collaborative_hypotheses"] = [hypothesis.to_dict() for hypothesis in session.hypotheses.values()]
    state["collaborative_decision_log"] = list(session.decision_log)
    state["collaborative_progressive_narrative"] = list(session.progressive_narrative)
    state["collaborative_checkpoint_summaries"] = list(session.checkpoint_summaries)
    state["collaborative_suggestions"] = list(session.ai_suggestions)
    state["collaborative_task_comparisons"] = list(session.task_comparisons)
    state.setdefault("analysis_evidence", {})
    return state


def _short_text(value: Any, limit: int = 220) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _format_analysis_step(step: Any) -> str:
    if isinstance(step, dict):
        parts: List[str] = []
        tool = step.get("tool") or step.get("analysis") or step.get("method") or step.get("type")
        if tool:
            parts.append(str(tool))
        columns = step.get("columns") or step.get("selected_columns") or step.get("features") or []
        if columns:
            parts.append(f"columns={list(columns)}")
        target = step.get("target") or step.get("target_column")
        if target:
            parts.append(f"target={target}")
        description = step.get("description") or step.get("purpose") or step.get("reason")
        if description:
            parts.append(str(description))
        return "; ".join(parts) if parts else _short_text(step)
    return _short_text(step)


def _analysis_story(task_request: str, final_state: AnalystState, summary: Dict[str, Any]) -> tuple[str, List[str], List[str]]:
    evidence = final_state.get("analysis_evidence", {}) or {}
    judgment = evidence.get("judgment_summary", {}) or {}
    top_story = _first_story(evidence)
    plan = list(summary.get("analysis_plan") or evidence.get("analysis_plan") or final_state.get("analysis_plan") or [])
    tool_results = evidence.get("tool_results") or {}
    dataframe = final_state.get("dataframe")
    selected_columns = humanize_columns(summary.get("selected_columns") or final_state.get("selected_columns") or [], dataframe=dataframe)

    analysis_steps = [humanize_text(_format_analysis_step(step), dataframe=dataframe) for step in plan[:5]]
    if not analysis_steps:
        analysis_steps = ["No explicit analysis steps were recorded, but the task still ran through the analytical pipeline."]

    tool_notes: List[str] = []
    if isinstance(tool_results, dict):
        for key, value in list(tool_results.items())[:5]:
            if isinstance(value, dict):
                note_parts = [str(value.get("tool") or value.get("type") or key)]
                if value.get("summary"):
                    note_parts.append(humanize_text(_short_text(value.get("summary"), 120), dataframe=dataframe))
                elif value.get("insight"):
                    note_parts.append(humanize_text(_short_text(value.get("insight"), 120), dataframe=dataframe))
                elif value.get("message"):
                    note_parts.append(humanize_text(_short_text(value.get("message"), 120), dataframe=dataframe))
                tool_notes.append(": ".join(note_parts))
            else:
                tool_notes.append(humanize_text(f"{key}: {type(value).__name__}", dataframe=dataframe))
    elif tool_results:
        tool_notes.append(humanize_text(_short_text(tool_results, 180), dataframe=dataframe))
    if not tool_notes:
        tool_notes = ["No tool result summary was available."]

    conclusion = humanize_text(top_story.get("insight") or judgment.get("summary") or summary.get("current_understanding") or task_request, dataframe=dataframe)
    confidence = summary.get("confidence")
    confidence_label = "unknown"
    if isinstance(confidence, (int, float)):
        if float(confidence) >= 70:
            confidence_label = "strong"
        elif float(confidence) >= 45:
            confidence_label = "moderate"
        else:
            confidence_label = "weak"
    elif confidence is not None:
        confidence_label = str(confidence)

    story_lines = []
    readable_request = humanize_text(task_request, dataframe=dataframe)
    if selected_columns:
        focus_text = ", ".join(selected_columns)
        story_lines.append(
            f"We centered the analysis on {focus_text} because the request was really asking how {readable_request} behaves when these fields change, and these are the most direct places to look for the pattern."
        )
    else:
        story_lines.append(
            f"We started with the full analytical context because {readable_request} was broad enough that the strongest signal could come from more than one part of the dataset, so narrowing too early could have missed the real driver."
        )
    story_lines.append(
        f"We then moved through these analysis steps: {' -> '.join(analysis_steps)}. Each step was used to test a different angle of the same question, so the conclusion came from a sequence of checks rather than a single glance."
    )
    story_lines.append(
        f"Those steps produced these signals and intermediate results: {'; '.join(tool_notes)}. This is the evidence trail that tells us what the data actually supported."
    )
    story_lines.append(
        f"Taken together, that evidence supports this conclusion: {conclusion}. In practical terms, this is the part of the answer that is most worth carrying forward into the next decision or follow-up investigation."
    )
    if judgment.get("contradictions_found"):
        story_lines.append("We also checked for contradictions, so the conclusion was tested against competing evidence instead of being accepted at face value.")
    if confidence is not None:
        story_lines.append(
            f"That puts the result at {confidence_label} confidence, which means it is useful for decision-making but still should be treated as evidence rather than a final verdict."
        )
    else:
        story_lines.append("Confidence was not explicitly quantified, so the result should be treated as provisional evidence that deserves follow-up.")
    if selected_columns:
        story_lines.append(
            "Why these fields matter: they are the strongest levers available in the current dataset for explaining the pattern, so they help separate a real signal from a coincidence."
        )

    return " ".join(story_lines), analysis_steps, tool_notes


def _first_story(evidence: Dict[str, Any]) -> Dict[str, Any]:
    stories = evidence.get("top_stories") or []
    return stories[0] if stories else {}


def _confidence_from_state(final_state: AnalystState) -> Any:
    evidence = final_state.get("analysis_evidence", {}) or {}
    judgment = evidence.get("judgment_summary", {}) or {}
    reasoning = final_state.get("analytical_reasoning", {}) or evidence.get("analytical_reasoning", {}) or {}
    if judgment.get("global_confidence") is not None:
        return judgment.get("global_confidence")
    confidence = (reasoning.get("confidence") or {}).get("score")
    if confidence is not None:
        return confidence
    first_story = _first_story(evidence)
    return first_story.get("confidence")


def _summarize_task_result(task_request: str, final_state: AnalystState, task_id: str, version: int) -> Dict[str, Any]:
    evidence = final_state.get("analysis_evidence", {}) or {}
    judgment = evidence.get("judgment_summary", {}) or {}
    top_story = _first_story(evidence)
    report = final_state.get("final_report") or ""
    narrative = top_story.get("insight") or judgment.get("summary") or _short_text(report.splitlines()[0] if report else task_request)
    confidence = _confidence_from_state(final_state)
    analysis_story, analysis_steps, tool_notes = _analysis_story(task_request, final_state, {
        "current_understanding": top_story.get("insight") or judgment.get("summary") or task_request,
        "confidence": confidence,
        "analysis_plan": list(evidence.get("analysis_plan") or final_state.get("analysis_plan") or []),
        "selected_columns": list(final_state.get("selected_columns") or []),
    })
    summary = {
        "task_id": task_id,
        "version": version,
        "request": task_request,
        "task_finding": top_story.get("insight") or judgment.get("summary") or task_request,
        "current_understanding": top_story.get("insight") or judgment.get("summary") or task_request,
        "narrative": narrative,
        "analysis_story": analysis_story,
        "analysis_steps": analysis_steps,
        "analysis_signals": tool_notes,
        "confidence": confidence,
        "selected_columns": list(final_state.get("selected_columns") or []),
        "analysis_plan": list(evidence.get("analysis_plan") or final_state.get("analysis_plan") or []),
        "tool_results": list((evidence.get("tool_results") or {}).keys()),
        "visualizations": len(evidence.get("visualizations") or []),
        "report_excerpt": _short_text(report, 350),
        "status": judgment.get("result_status") or judgment.get("summary") or "completed",
    }
    return summary


def _build_evidence_record(task_id: str, final_state: AnalystState, summary: Dict[str, Any]) -> EvidenceRecord:
    evidence = final_state.get("analysis_evidence", {}) or {}
    judgment = evidence.get("judgment_summary", {}) or {}
    top_story = _first_story(evidence)
    supporting_visualizations = []
    for visual in evidence.get("visualizations") or []:
        if isinstance(visual, dict):
            supporting_visualizations.append(visual.get("title") or visual.get("path") or visual.get("type") or visual)
        else:
            supporting_visualizations.append(visual)
    quality_score = 0.0
    confidence = _confidence_from_state(final_state)
    try:
        if confidence is not None:
            quality_score = float(confidence)
    except Exception:
        quality_score = float(top_story.get("score", 0.0) or 0.0)

    return EvidenceRecord(
        evidence_id=f"{task_id}-evidence",
        task_source=task_id,
        evidence_type="task_result",
        statement=top_story.get("insight") or judgment.get("summary") or summary.get("current_understanding") or "Task completed.",
        statistical_support={
            "judgment_summary": judgment,
            "top_story": top_story,
            "analysis_plan": evidence.get("analysis_plan") or final_state.get("analysis_plan") or [],
            "tool_results": list((evidence.get("tool_results") or {}).keys()),
        },
        confidence=confidence,
        method=_short_text(" | ".join(map(str, summary.get("analysis_plan") or [])), 180),
        supporting_visualizations=supporting_visualizations,
        dependencies=list(summary.get("selected_columns") or []),
        quality_score=quality_score,
        metadata={
            "business_question": final_state.get("business_question"),
            "report_excerpt": summary.get("report_excerpt"),
            "version": summary.get("version"),
        },
    )


def _derive_hypothesis(
    task_request: str,
    final_state: AnalystState,
    evidence_record: EvidenceRecord,
    integrity: Dict[str, Any] | None = None,
) -> HypothesisRecord:
    evidence = final_state.get("analysis_evidence", {}) or {}
    judgment = evidence.get("judgment_summary", {}) or {}
    top_story = _first_story(evidence)
    confidence = _confidence_from_state(final_state)
    confidence_value = float(confidence) if isinstance(confidence, (int, float)) else None
    integrity = dict(integrity or {})
    promoted = bool(integrity.get("should_promote"))
    question_relevance = int((integrity.get("question_relevance") or {}).get("score") or 0)
    continuity_score = int((integrity.get("continuity") or {}).get("score") or 0)
    if promoted and confidence_value is not None and confidence_value >= 70 and question_relevance >= 45 and continuity_score >= 35:
        status = "supported"
    elif confidence_value is not None and confidence_value < 45:
        status = "inconclusive"
    else:
        status = "inconclusive"
    if not promoted:
        status = "inconclusive"
    if judgment.get("contradictions_found"):
        status = "rejected" if confidence_value is not None and confidence_value < 60 else "inconclusive"
    hypothesis_text = top_story.get("insight") or judgment.get("summary") or task_request
    notes = []
    if judgment.get("contradictions_found"):
        notes.append("Contradictions were surfaced by the analytical pipeline.")
    if integrity:
        notes.append(
            f"Integrity gate: relevance={integrity.get('question_relevance', {}).get('level', 'unknown')}, "
            f"continuity={integrity.get('continuity', {}).get('level', 'unknown')}, "
            f"information_gain={integrity.get('information_gain', {}).get('level', 'unknown')}, "
            f"validity={integrity.get('analytical_validity', {}).get('level', 'unknown')}."
        )
        if not promoted:
            notes.append("The result was retained as supporting evidence because it did not sufficiently advance the main investigation.")
    if final_state.get("llm_reasoning_status"):
        notes.append(str(final_state.get("llm_reasoning_status")))
    return HypothesisRecord(
        hypothesis=hypothesis_text,
        status=status,
        confidence=confidence,
        supporting_evidence=[evidence_record.evidence_id],
        conflicting_evidence=[],
        notes=notes,
    )


def _suggest_next_investigations(session: InvestigationSession, final_state: AnalystState, task_id: str) -> List[Dict[str, Any]]:
    original_question = final_state.get("business_question") or session.original_question or ""
    suggestions: List[Dict[str, Any]] = []
    evidence = final_state.get("analysis_evidence", {}) or {}
    top_story = _first_story(evidence)
    confidence = _confidence_from_state(final_state)
    current_understanding = _best_answer_text(session.investigation_memory, session.original_question or original_question)
    current_hypothesis = current_understanding
    executed_requests = {
        str(item.get("request") or "").strip().lower()
        for item in (session.investigation_memory.get("task_summaries") or {}).values()
        if isinstance(item, dict) and str(item.get("request") or "").strip()
    }
    executed_requests |= {
        str(task.request or "").strip().lower()
        for task in session.tasks.values()
        if str(task.request or "").strip()
    }
    has_executed_challenge = any(
        any(term in request for term in ("challenge", "test", "verify", "falsify"))
        for request in executed_requests
    )
    previous_next = []
    if session.checkpoint_summaries:
        previous_next = list(session.checkpoint_summaries[-1].get("next_investigations") or [])

    def _signature(item: Dict[str, Any]) -> tuple[str, str]:
        return (
            str(item.get("title") or "").strip().lower(),
            str(item.get("request") or item.get("description") or "").strip().lower(),
        )

    def _confidence_score(value: Any) -> int:
        if isinstance(value, (int, float)):
            numeric = float(value)
            return int(numeric * 100 if numeric <= 1 else numeric)
        if isinstance(value, dict):
            score = value.get("score")
            if isinstance(score, (int, float)):
                numeric = float(score)
                return int(numeric * 100 if numeric <= 1 else numeric)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"high", "strong", "very high"}:
                return 85
            if lowered in {"moderate", "medium"}:
                return 60
            if lowered in {"low", "weak"}:
                return 35
        return 50

    def _term_overlap_score(*texts: Any) -> int:
        tokens: set[str] = set()
        for value in texts:
            for token in str(value or "").lower().split():
                token = token.strip(".,:;!?()[]{}<>\"'")
                if len(token) >= 4:
                    tokens.add(token)
        question_tokens = {
            token.strip(".,:;!?()[]{}<>\"'")
            for token in original_question.lower().split()
            if len(token.strip(".,:;!?()[]{}<>\"'")) >= 4
        }
        return len(tokens & question_tokens)

    def _rank_suggestion(item: Dict[str, Any]) -> Dict[str, Any]:
        candidate = dict(item)
        title = str(candidate.get("title") or "")
        request = str(candidate.get("request") or candidate.get("description") or "")
        reason = str(candidate.get("reason") or candidate.get("justification") or "")
        integrity = evaluate_task_request_relevance(
            original_question=original_question,
            task_request=request,
            current_understanding=current_understanding,
            current_hypothesis=current_hypothesis,
            prior_findings=session.investigation_memory.get("previous_findings") or [],
        )
        relevance = min(100, max(integrity["question_alignment"]["score"], 20 + _term_overlap_score(title, request, reason) * 18))
        if any(term in (title + " " + request + " " + reason).lower() for term in ["challenge", "falsify", "test", "verify"]):
            uncertainty_reduction = 85
        elif any(term in (title + " " + request + " " + reason).lower() for term in ["compare", "contrast"]):
            uncertainty_reduction = 75
        elif any(term in (title + " " + request + " " + reason).lower() for term in ["refine", "rerun", "re-run", "repeat"]):
            uncertainty_reduction = 70
        else:
            uncertainty_reduction = 60
        hypothesis_coverage = min(100, 35 + integrity["continuity"]["score"] // 2 + (10 if integrity["allowed"] else 0))
        business_value = min(100, 35 + relevance // 2 + (10 if top_story.get("insight") else 0))
        analytical_confidence = _confidence_score(candidate.get("confidence", confidence))
        recommendation_score = round(
            (0.35 * relevance)
            + (0.20 * hypothesis_coverage)
            + (0.20 * uncertainty_reduction)
            + (0.15 * business_value)
            + (0.10 * analytical_confidence)
        )
        candidate["question_relevance"] = relevance
        candidate["request_alignment"] = int(integrity["question_alignment"]["score"])
        candidate["hypothesis_coverage"] = hypothesis_coverage
        candidate["uncertainty_reduction"] = uncertainty_reduction
        candidate["business_value"] = business_value
        candidate["analytical_confidence"] = analytical_confidence
        candidate["integrity"] = integrity
        candidate["continuity_score"] = integrity["continuity"]["score"]
        candidate["integrity_score"] = integrity["score"]
        candidate["recommendation_score"] = max(0, min(100, recommendation_score))
        return candidate

    def _apply_impact(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        base_impact = suggestion_impact_percent({"confidence": confidence}) or 50
        impact_offsets = [0, -10, -20]
        enriched: List[Dict[str, Any]] = []
        for index, item in enumerate(items[:3]):
            candidate = _rank_suggestion(item)
            request_key = str(candidate.get("request") or candidate.get("description") or candidate.get("title") or "").strip().lower()
            if request_key and request_key in executed_requests:
                continue
            candidate_text = " ".join(
                str(candidate.get(field) or "")
                for field in ("title", "request", "description", "reason", "justification")
            ).lower()
            if has_executed_challenge and any(term in candidate_text for term in ("challenge", "test", "verify", "falsify")) and candidate.get("request_alignment", 0) < 50:
                continue
            if candidate.get("question_relevance", 0) < 35 and not any(term in candidate_text for term in ("compare", "challenge", "refine", "rerun", "repeat")):
                continue
            if not candidate["integrity"].get("allowed") and candidate["integrity"]["score"] < 45:
                continue
            candidate["impact_percent"] = max(
                0,
                min(
                    100,
                    round(
                        0.35 * base_impact
                        + 0.30 * candidate["question_relevance"]
                        + 0.15 * candidate["hypothesis_coverage"]
                        + 0.20 * candidate["uncertainty_reduction"]
                        + 0.15 * candidate["business_value"]
                        + impact_offsets[index],
                    ),
                ),
            )
            candidate["confidence"] = candidate.get("confidence", confidence)
            enriched.append(candidate)
        enriched.sort(
            key=lambda item: (
                item.get("recommendation_score", 0),
                item.get("impact_percent", 0),
                item.get("question_relevance", 0),
                item.get("hypothesis_coverage", 0),
                item.get("uncertainty_reduction", 0),
            ),
            reverse=True,
        )
        return enriched

    if any(term in original_question.lower() for term in ["region", "geo", "geographic", "location"]):
        suggestions.append(
            {
                "title": "Investigate regional differences",
                "request": "Compare the outcome by region and test whether regional segmentation changes the conclusion.",
                "depends_on": [task_id],
                "confidence": confidence,
                "source_task_id": task_id,
            }
        )

    if any(term in original_question.lower() for term in ["customer", "segment", "group", "cohort"]):
        suggestions.append(
            {
                "title": "Compare customer segments",
                "request": "Compare the most important customer segments and inspect whether the main driver changes across groups.",
                "depends_on": [task_id],
                "confidence": confidence,
                "source_task_id": task_id,
            }
        )

    if top_story.get("insight"):
        suggestions.append(
            {
                "title": "Challenge the leading finding",
                "request": f"Challenge this finding: {top_story.get('insight')}",
                "depends_on": [task_id],
                "confidence": confidence,
                "source_task_id": task_id,
            }
        )

    if not suggestions:
        suggestions.append(
            {
                "title": "Refine the current task",
                "request": "Re-run the current task with a narrower scope or an alternative hypothesis.",
                "depends_on": [task_id],
                "confidence": confidence,
                "source_task_id": task_id,
            }
        )

    suggestions = _apply_impact(suggestions)

    current_signatures = {_signature(item) for item in suggestions}
    previous_signatures = {_signature(item) for item in previous_next}
    if current_signatures and previous_signatures and current_signatures.issubset(previous_signatures):
        return [
            {
                "title": "No further distinct investigation",
                "request": "The strongest follow-up from the previous checkpoint is still the right one, so there is no new distinct next step to add. Consolidate the result or finish the investigation before starting a new question.",
                "depends_on": [task_id],
                "confidence": confidence,
                "impact_percent": 0,
                "source_task_id": task_id,
                "terminal": True,
            }
        ]

    unique_suggestions: List[Dict[str, Any]] = []
    seen_signatures: set[tuple[str, str]] = set()
    for suggestion in suggestions:
        signature = _signature(suggestion)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique_suggestions.append(suggestion)

    return unique_suggestions[:3]


def _build_desk_view(session: InvestigationSession) -> Dict[str, Any]:
    completed = [session.tasks[task_id].to_dict() for task_id in session.completed_tasks if task_id in session.tasks]
    running = [session.tasks[task_id].to_dict() for task_id in session.running_tasks if task_id in session.tasks]
    queued = [session.tasks[task_id].to_dict() for task_id in session.queued_tasks if task_id in session.tasks]
    failed = [session.tasks[task_id].to_dict() for task_id in session.failed_tasks if task_id in session.tasks]
    decision = session.investigation_memory.get("investigation_decision") or {}
    return {
        "investigation_id": session.investigation_id,
        "original_question": session.original_question,
        "current_status": session.current_status,
        "completed_tasks": completed,
        "running_tasks": running,
        "queued_tasks": queued,
        "failed_tasks": failed,
        "current_understanding": _best_answer_text(session.investigation_memory, session.original_question),
        "last_failure": session.investigation_memory.get("last_failure"),
        "evidence_summary": [record.statement for record in session.evidence_store.values()],
        "current_hypotheses": [hypothesis.to_dict() for hypothesis in session.hypotheses.values()],
        "ai_suggested_next_investigations": list(session.ai_suggestions),
        "human_actions": [
            "new investigation",
            "refine task",
            "compare results",
            "challenge finding",
            "accept AI suggestion",
            "finish investigation",
        ],
        "progressive_narrative": list(session.progressive_narrative),
        "task_graph": deepcopy(session.task_graph),
        "decision_log": list(session.decision_log),
        "investigation_memory": deepcopy(session.investigation_memory),
        "current_decision": decision.get("decision"),
        "investigation_decision": decision,
        "question_for_user": session.investigation_memory.get("question_for_user"),
    }


def _inject_collaborative_context(state: AnalystState, session: InvestigationSession) -> AnalystState:
    state["collaborative_session"] = session.to_dict()
    state["collaborative_tasks"] = [task.to_dict() for task in session.tasks.values()]
    state["collaborative_task_graph"] = deepcopy(session.task_graph)
    state["collaborative_queue"] = [session.tasks[task_id].to_dict() for task_id in session.queued_tasks if task_id in session.tasks]
    state["collaborative_evidence_store"] = {
        evidence_id: record.to_dict() for evidence_id, record in session.evidence_store.items()
    }
    state["collaborative_memory"] = deepcopy(session.investigation_memory)
    state["collaborative_hypotheses"] = [hypothesis.to_dict() for hypothesis in session.hypotheses.values()]
    state["collaborative_decision_log"] = list(session.decision_log)
    state["collaborative_progressive_narrative"] = list(session.progressive_narrative)
    state["collaborative_checkpoint_summaries"] = list(session.checkpoint_summaries)
    state["collaborative_suggestions"] = list(session.ai_suggestions)
    state["collaborative_task_comparisons"] = list(session.task_comparisons)
    return state


def _default_initial_tasks(question: str) -> List[Dict[str, Any]]:
    return [
        {
            "title": "Primary investigation",
            "request": question,
        },
    ]


def run_collaborative_investigation(
    question: str,
    responses: Sequence[str] | None = None,
    dataset_path: str | None = None,
    dataframe: Any | None = None,
    initial_tasks: Sequence[Dict[str, Any] | str] | None = None,
    build_final_report: bool = True,
) -> CollaborativeRunResult:
    """
    Run an investigation by dispatching each task through the existing analytical pipeline.

    The collaborative layer manages session state, task lineage, evidence, and synthesis.
    The analytical graph itself remains unchanged and is invoked in autonomous mode for
    each task.
    """
    from backend.scripts.guided_mode_harness import build_guided_sample_dataframe

    if dataframe is not None:
        df = dataframe
    elif dataset_path:
        df = pd.read_csv(dataset_path, low_memory=False)
    else:
        df = build_guided_sample_dataframe()

    base_state: AnalystState = {
        "business_question": question,
        "dataset_path": dataset_path,
        "dataframe": df,
        "mode": "collaborative",
        "enable_llm_reasoning": False,
        "disable_llm_reasoning": True,
        "disable_semantic_matcher": True,
        "analysis_evidence": {},
    }

    session = create_investigation_session(question)
    manager = TaskManager(session)

    if not initial_tasks:
        initial_tasks = _default_initial_tasks(question)

    for task_spec in initial_tasks:
        if isinstance(task_spec, str):
            manager.enqueue_request(task_spec)
            continue
        manager.enqueue_request(
            request=str(task_spec.get("request") or question),
            title=str(task_spec.get("title") or task_spec.get("request") or question),
            dependencies=task_spec.get("dependencies") or [],
            parent_task_id=task_spec.get("parent_task_id"),
            metadata=task_spec.get("metadata") or {},
        )

    task_outputs: Dict[str, Dict[str, Any]] = {}
    last_successful_state: AnalystState | None = None

    while True:
        task = manager.dequeue_next_task()
        if task is None:
            break

        task_state = _build_task_state(base_state, session, task.request)
        try:
            with redirect_stdout(StringIO()):
                final_state = graph.invoke(task_state)
        except Exception as exc:  # pragma: no cover - defensive safety net
            manager.mark_failed(task.task_id, str(exc), metadata={"request": task.request})
            session.progressive_narrative.append(f"{task.title} failed: {exc}")
            continue

        summary = _summarize_task_result(task.request, final_state, task.task_id, task.version)
        evidence = _build_evidence_record(task.task_id, final_state, summary)
        manager.mark_completed(task.task_id, final_state, evidence, summary)
        current_hypothesis = _best_answer_text(session.investigation_memory, session.original_question)
        answer_synthesis = synthesize_answer(
            business_question=task.request,
            evidence=final_state.get("analysis_evidence", {}) or {},
            hypotheses=[],
            current_understanding=summary.get("current_understanding") or summary.get("task_finding") or task.request,
            confidence=summary.get("confidence"),
            knowledge_gaps=session.investigation_memory.get("knowledge_gaps") or [],
            investigation_memory=session.investigation_memory,
            dataframe=base_state.get("dataframe"),
        )
        decision = (
            final_state.get("investigation_decision")
            or final_state.get("analysis_evidence", {}).get("investigation_decision")
            or answer_synthesis.get("investigation_decision")
            or {}
        )
        integrity = evaluate_investigation_integrity(
            original_question=session.original_question,
            task_request=task.request,
            summary=summary,
            current_hypothesis=current_hypothesis,
            prior_findings=session.investigation_memory.get("previous_findings") or [],
            dataframe=base_state.get("dataframe"),
        )
        summary["integrity"] = integrity
        summary["traceability"] = build_traceability_record(integrity)
        summary["investigation_decision"] = decision
        summary["current_understanding"] = integrity["promoted_understanding"] if integrity["should_promote"] else (session.investigation_memory.get("current_understanding") or session.original_question)
        summary["best_answer"] = update_best_answer_anchor(
            session.investigation_memory,
            original_question=session.original_question,
            summary=summary,
            integrity=integrity,
            task_id=task.task_id,
            task_title=task.title,
        )
        if not integrity["should_promote"]:
            summary["narrative"] = f"Supporting evidence only: {summary.get('narrative') or summary['task_finding']}"
        summary["task_finding"] = summary.get("task_finding") or summary["current_understanding"]
        summary["integrity_status"] = integrity["overall"]["level"]
        hypothesis = _derive_hypothesis(task.request, final_state, evidence, integrity=integrity)
        hypothesis.notes.append(
            f"Integrity: question relevance={integrity['question_relevance']['level']}, continuity={integrity['continuity']['level']}, information gain={integrity['information_gain']['level']}, validity={integrity['analytical_validity']['level']}."
        )
        if not integrity["should_promote"]:
            hypothesis.status = "inconclusive"
        session.hypotheses[f"{task.task_id}:v{task.version}"] = hypothesis
        session.progressive_narrative.append(summary["narrative"])
        session.investigation_memory["last_integrity"] = integrity
        session.investigation_memory["last_traceability"] = summary["traceability"]
        session.investigation_memory["investigation_decision"] = decision
        session.investigation_memory["current_decision"] = decision.get("decision")
        session.investigation_memory["question_for_user"] = decision.get("question_for_user") or session.investigation_memory.get("question_for_user")
        session.investigation_memory.setdefault("integrity_history", []).append(
            {
                "task_id": task.task_id,
                "task_request": task.request,
                "traceability": summary["traceability"],
                "integrity": integrity,
                "decision": decision,
            }
        )
        if integrity["should_promote"]:
            session.investigation_memory["current_understanding"] = summary["current_understanding"]
        else:
            session.investigation_memory.setdefault("supporting_findings", []).append(summary["task_finding"])
            if "current_understanding" not in session.investigation_memory:
                session.investigation_memory["current_understanding"] = session.original_question
        session.investigation_memory.setdefault("previous_findings", []).append(summary["task_finding"])
        session.investigation_memory.setdefault("task_references", []).append(task.task_id)
        next_suggestions = _suggest_next_investigations(session, final_state, task.task_id)
        session.ai_suggestions = next_suggestions
        session.checkpoint_summaries.append(
            {
                "checkpoint_id": f"checkpoint-{len(session.checkpoint_summaries) + 1}",
                "task_id": task.task_id,
                "task_title": task.title,
                "task_request": task.request,
                "version": task.version,
                "status": summary.get("status", "completed"),
                "current_understanding": summary.get("current_understanding"),
                "narrative": summary.get("narrative"),
                "confidence": summary.get("confidence"),
                "integrity": integrity,
                "traceability": summary["traceability"],
                "selected_columns": list(summary.get("selected_columns") or []),
                "analysis_plan": list(summary.get("analysis_plan") or []),
                "tool_results": list(summary.get("tool_results") or []),
                "visualizations": summary.get("visualizations"),
                "report_excerpt": summary.get("report_excerpt"),
                "timestamp": summary.get("timestamp"),
                "next_investigations": list(next_suggestions),
            }
        )
        if len(session.completed_tasks) >= 2:
            comparison = manager.compare_tasks(session.completed_tasks[-2], session.completed_tasks[-1])
            session.investigation_memory.setdefault("comparison_history", []).append(comparison)
        task_outputs[task.task_id] = summary
        last_successful_state = final_state

    session.current_status = "completed" if session.completed_tasks else "failed"

    if last_successful_state is None:
        final_state = deepcopy(base_state)
        final_state["analysis_evidence"] = {
            "collaborative_session": session.to_dict(),
            "final_output": [
                "No collaborative tasks completed successfully.",
                "The investigation preserved the session and can be resumed.",
            ],
        }
        final_state["final_report"] = "\n".join(final_state["analysis_evidence"]["final_output"])
        session.final_executive_report = final_state["final_report"]
        desk = _build_desk_view(session)
        return CollaborativeRunResult(
            final_state=final_state,
            session=session.to_dict(),
            desk=desk,
            task_outputs=task_outputs,
        )

    final_state = deepcopy(last_successful_state)
    _inject_collaborative_context(final_state, session)
    final_state["analysis_evidence"]["collaborative_session"] = session.to_dict()
    final_state["analysis_evidence"]["collaborative_task_outputs"] = dict(task_outputs)
    final_state["analysis_evidence"]["collaborative_desk"] = _build_desk_view(session)
    final_state["analysis_evidence"]["investigation_decision"] = session.investigation_memory.get("investigation_decision") or {}
    final_state["investigation_decision"] = session.investigation_memory.get("investigation_decision") or {}

    if build_final_report:
        final_state = report_node(final_state)
        session.final_executive_report = final_state.get("final_report")
    else:
        report_lines = [
            "================ COLLABORATIVE INVESTIGATION ================",
            f"Investigation ID: {session.investigation_id}",
            f"Original question: {session.original_question}",
            f"Completed tasks: {', '.join(session.completed_tasks) if session.completed_tasks else 'None'}",
            f"Current understanding: {_best_answer_text(session.investigation_memory, 'None')}",
            "Task findings:",
        ]
        for task_id in session.completed_tasks:
            task_summary = task_outputs.get(task_id, {})
            report_lines.append(
                f"- {task_id}: {task_summary.get('current_understanding') or task_summary.get('narrative') or 'Task completed.'}"
            )
        report_lines.extend(
            [
                "Evidence store:",
                f"- {len(session.evidence_store)} evidence items",
                "Hypotheses:",
                f"- {len(session.hypotheses)} hypotheses tracked",
            ]
        )
        final_state["final_report"] = "\n".join(report_lines)
        session.final_executive_report = final_state["final_report"]

    final_state["collaborative_final_report"] = final_state.get("final_report")

    return CollaborativeRunResult(
        final_state=final_state,
        session=session.to_dict(),
        desk=_build_desk_view(session),
        task_outputs=task_outputs,
    )


