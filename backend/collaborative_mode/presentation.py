from __future__ import annotations

from typing import Any, Dict, List

from .narrative_composer import render_analyst_report
from .narration import format_suggestion_line, humanize_text, suggestion_impact_percent


def _non_empty_lines(lines: List[str]) -> List[str]:
    return [line for line in lines if isinstance(line, str) and line.strip()]


def _best_answer(session: Dict[str, Any]) -> Any:
    memory = session.get("investigation_memory") or {}
    return (
        (memory.get("best_answer") or {}).get("answer")
        or memory.get("current_understanding")
        or session.get("current_understanding")
    )


def _format_suggestion(suggestion: Dict[str, Any], dataframe: Any = None) -> str:
    title = humanize_text(suggestion.get("title") or "Next investigation", dataframe=dataframe)
    request = humanize_text(suggestion.get("request") or suggestion.get("description") or "No request provided.", dataframe=dataframe)
    impact = suggestion_impact_percent(suggestion)
    impact_text = f"{impact}%" if impact is not None else "unknown"
    return f"{title}: {request} (impact {impact_text})"


def render_collaborative_analyst_view(result: Any) -> str:
    final_state = getattr(result, "final_state", {}) or {}
    session = getattr(result, "session", {}) or {}
    desk = getattr(result, "desk", {}) or {}
    evidence = final_state.get("analysis_evidence", {}) or {}
    dataframe = final_state.get("dataframe")

    question = humanize_text(session.get("original_question") or final_state.get("business_question") or "the investigation", dataframe=dataframe)
    current_understanding = humanize_text(
        _best_answer(session)
        or evidence.get("current_understanding")
        or "The investigation is still developing.",
        dataframe=dataframe,
    )
    status = session.get("current_status") or final_state.get("mode") or "unknown"
    completed = session.get("completed_tasks") or []
    evidence_count = len(session.get("evidence_store", {}) or {})
    hypotheses_count = len(session.get("hypotheses", {}) or {})
    suggestions = session.get("ai_suggestions") or []
    next_step = _format_suggestion(suggestions[0], dataframe=dataframe) if suggestions else "No further suggestion is currently available."
    final_report = final_state.get("final_report") or ""
    decision = session.get("investigation_memory", {}).get("investigation_decision") or final_state.get("investigation_decision") or {}
    decision_text = decision.get("decision") or "pending"
    decision_reason = decision.get("recommended_next_step") or "No decision recommendation is currently available."

    lines = [
        "===== COLLABORATIVE ANALYST VIEW =====",
        f"Business question: {question}",
        f"Current answer: {current_understanding}",
        f"Status: {status}",
        f"Completed tasks: {len(completed)}",
        f"Evidence items captured: {evidence_count}",
        f"Hypotheses tracked: {hypotheses_count}",
        f"Next decision: {next_step}",
        f"Investigation decision: {decision_text}",
        f"Decision guidance: {decision_reason}",
        f"Desk status: {humanize_text(desk.get('current_status') or 'unknown', dataframe=dataframe)}",
    ]
    if final_report:
        lines.extend(["", "Final report preview:", render_analyst_report(final_state, evidence) or final_report])
    return "\n".join(_non_empty_lines(lines))


def render_debug_collaborative_view(result: Any) -> str:
    final_state = getattr(result, "final_state", {}) or {}
    session = getattr(result, "session", {}) or {}
    desk = getattr(result, "desk", {}) or {}
    evidence = final_state.get("analysis_evidence", {}) or {}
    lines = [
        "===== DEBUG VIEW =====",
        f"Investigation ID: {session.get('investigation_id')}",
        f"Session keys: {sorted(session.keys())}",
        f"Desk keys: {sorted(desk.keys())}",
        f"Evidence keys: {sorted(evidence.keys())}",
        f"Final report available: {bool(final_state.get('final_report'))}",
    ]
    return "\n".join(lines)


def render_collaborative_desk_view(session: Dict[str, Any], dataframe: Any = None) -> str:
    memory = session.get("investigation_memory") or {}
    suggestions = session.get("ai_suggestions") or []
    decision = memory.get("investigation_decision") or session.get("investigation_decision") or {}
    current_understanding = humanize_text(
        (memory.get("best_answer") or {}).get("answer")
        or memory.get("current_understanding")
        or session.get("current_understanding")
        or "The investigation is still developing.",
        dataframe=dataframe,
    )
    question = humanize_text(session.get("original_question") or session.get("objective") or "the investigation", dataframe=dataframe)
    next_step = _format_suggestion(suggestions[0], dataframe=dataframe) if suggestions else "No next step has been suggested yet."
    evidence_count = len(session.get("evidence_store", {}) or {})
    task_count = len(session.get("tasks", {}) or {})
    status = humanize_text(session.get("current_status") or "unknown", dataframe=dataframe)

    lines = [
        "Investigation Desk",
        f"Question: {question}",
        f"Current understanding: {current_understanding}",
        f"Status: {status}",
        f"Decision: {decision.get('decision') or 'pending'}",
        f"Recommended next step: {humanize_text(decision.get('recommended_next_step') or 'Awaiting decision.', dataframe=dataframe)}",
        f"Tasks reviewed: {task_count}",
        f"Evidence items: {evidence_count}",
        f"Next best step: {next_step}",
        "Pause rule: the investigation keeps moving until the decision layer asks for human guidance or you choose a human action that needs input.",
    ]
    if decision.get("remaining_uncertainties"):
        lines.append(
            "Remaining uncertainties: "
            + "; ".join(humanize_text(item, dataframe=dataframe) for item in decision.get("remaining_uncertainties", [])[:3])
        )
    if suggestions:
        lines.append("Choose an action or ask a capability question:")
        for index, suggestion in enumerate(sorted(suggestions, key=lambda item: suggestion_impact_percent(item) or 0, reverse=True)[:3], start=1):
            lines.append(format_suggestion_line(suggestion, index=index, dataframe=dataframe))
    return "\n".join(_non_empty_lines(lines))


def render_collaborative_handoff_view(session: Dict[str, Any], dataframe: Any = None) -> str:
    memory = session.get("investigation_memory") or {}
    suggestions = session.get("ai_suggestions") or []
    decision = memory.get("investigation_decision") or session.get("investigation_decision") or {}
    checkpoints = session.get("checkpoint_summaries") or []
    completed = session.get("completed_tasks") or []
    question = humanize_text(session.get("original_question") or session.get("objective") or "the investigation", dataframe=dataframe)
    current_understanding = humanize_text(
        (memory.get("best_answer") or {}).get("answer")
        or memory.get("current_understanding")
        or session.get("current_understanding")
        or "The investigation is still developing.",
        dataframe=dataframe,
    )
    reasoning = [humanize_text(item, dataframe=dataframe) for item in (decision.get("reasoning") or [])[:3]]
    uncertainties = [humanize_text(item, dataframe=dataframe) for item in (decision.get("remaining_uncertainties") or [])[:3]]
    next_actions = sorted(suggestions, key=lambda item: suggestion_impact_percent(item) or 0, reverse=True)[:3]
    known_lines = [
        f"Question: {question}",
        f"What we know: {current_understanding}",
        f"Completed tasks: {len(completed)}",
        f"Latest checkpoint: {humanize_text(checkpoints[-1].get('task_title') if checkpoints else 'None yet', dataframe=dataframe)}",
    ]
    if reasoning:
        known_lines.append("Why human input is needed now: " + " ".join(reasoning))
    else:
        known_lines.append(
            "Why human input is needed now: the decision layer needs analyst judgment to choose the next branch or confirm that the current answer is sufficient."
        )
    if uncertainties:
        known_lines.append("What is still uncertain: " + "; ".join(uncertainties))
    else:
        known_lines.append("What is still uncertain: no major uncertainty was recorded, but the workflow still needs a human choice to proceed.")
    if next_actions:
        known_lines.append("Best next actions:")
        for index, suggestion in enumerate(next_actions, start=1):
            known_lines.append(f" - {format_suggestion_line(suggestion, index=index, dataframe=dataframe)}")
    else:
        known_lines.append("Best next actions: choose a new investigation, refine the current task, or finish the investigation if the answer is sufficient.")
    known_lines.append("You can also ask a capability question if you want a quick review before choosing the next action.")
    return "\n".join(_non_empty_lines(["===== ANALYST HANDOFF =====", *known_lines]))


