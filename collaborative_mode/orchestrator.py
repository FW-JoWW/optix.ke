from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from io import StringIO
from contextlib import redirect_stdout
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from graph.analyst_graph import graph
from nodes.report_node import report_node
from state.state import AnalystState

from .models import EvidenceRecord, HypothesisRecord, InvestigationSession
from .task_manager import TaskManager


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
    state["collaborative_suggestions"] = list(session.ai_suggestions)
    state["collaborative_task_comparisons"] = list(session.task_comparisons)
    state.setdefault("analysis_evidence", {})
    return state


def _short_text(value: Any, limit: int = 220) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


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
    summary = {
        "task_id": task_id,
        "version": version,
        "request": task_request,
        "current_understanding": top_story.get("insight") or judgment.get("summary") or task_request,
        "narrative": narrative,
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


def _derive_hypothesis(task_request: str, final_state: AnalystState, evidence_record: EvidenceRecord) -> HypothesisRecord:
    evidence = final_state.get("analysis_evidence", {}) or {}
    judgment = evidence.get("judgment_summary", {}) or {}
    top_story = _first_story(evidence)
    confidence = _confidence_from_state(final_state)
    confidence_value = float(confidence) if isinstance(confidence, (int, float)) else None
    if confidence_value is not None and confidence_value >= 70:
        status = "supported"
    elif confidence_value is not None and confidence_value < 45:
        status = "inconclusive"
    else:
        status = "inconclusive"
    if judgment.get("contradictions_found"):
        status = "rejected" if confidence_value is not None and confidence_value < 60 else "inconclusive"
    hypothesis_text = top_story.get("insight") or judgment.get("summary") or task_request
    notes = []
    if judgment.get("contradictions_found"):
        notes.append("Contradictions were surfaced by the analytical pipeline.")
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
    question = (final_state.get("business_question") or session.original_question or "").lower()
    suggestions: List[Dict[str, Any]] = []
    evidence = final_state.get("analysis_evidence", {}) or {}
    top_story = _first_story(evidence)

    if any(term in question for term in ["region", "geo", "geographic", "location"]):
        suggestions.append(
            {
                "title": "Investigate regional differences",
                "request": "Compare the outcome by region and test whether regional segmentation changes the conclusion.",
                "depends_on": [task_id],
            }
        )

    if any(term in question for term in ["customer", "segment", "group", "cohort"]):
        suggestions.append(
            {
                "title": "Compare customer segments",
                "request": "Compare the most important customer segments and inspect whether the main driver changes across groups.",
                "depends_on": [task_id],
            }
        )

    if top_story.get("insight"):
        suggestions.append(
            {
                "title": "Challenge the leading finding",
                "request": f"Challenge this finding: {top_story.get('insight')}",
                "depends_on": [task_id],
            }
        )

    if not suggestions:
        suggestions.append(
            {
                "title": "Refine the current task",
                "request": "Re-run the current task with a narrower scope or an alternative hypothesis.",
                "depends_on": [task_id],
            }
        )

    return suggestions[:3]


def _build_desk_view(session: InvestigationSession) -> Dict[str, Any]:
    completed = [session.tasks[task_id].to_dict() for task_id in session.completed_tasks if task_id in session.tasks]
    running = [session.tasks[task_id].to_dict() for task_id in session.running_tasks if task_id in session.tasks]
    queued = [session.tasks[task_id].to_dict() for task_id in session.queued_tasks if task_id in session.tasks]
    return {
        "investigation_id": session.investigation_id,
        "original_question": session.original_question,
        "current_status": session.current_status,
        "completed_tasks": completed,
        "running_tasks": running,
        "queued_tasks": queued,
        "current_understanding": session.investigation_memory.get("current_understanding"),
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
    state["collaborative_suggestions"] = list(session.ai_suggestions)
    state["collaborative_task_comparisons"] = list(session.task_comparisons)
    return state


def run_collaborative_investigation(
    question: str,
    responses: Sequence[str] | None = None,
    dataset_path: str | None = None,
    dataframe: Any | None = None,
    initial_tasks: Sequence[Dict[str, Any] | str] | None = None,
) -> CollaborativeRunResult:
    """
    Run an investigation by dispatching each task through the existing analytical pipeline.

    The collaborative layer manages session state, task lineage, evidence, and synthesis.
    The analytical graph itself remains unchanged and is invoked in autonomous mode for
    each task.
    """
    from scripts.guided_mode_harness import build_guided_sample_dataframe

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
        initial_tasks = [
            {
                "title": "Primary investigation",
                "request": question,
            }
        ]

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
        hypothesis = _derive_hypothesis(task.request, final_state, evidence)
        session.hypotheses[f"{task.task_id}:v{task.version}"] = hypothesis
        session.progressive_narrative.append(summary["narrative"])
        session.investigation_memory["current_understanding"] = summary["current_understanding"]
        session.investigation_memory.setdefault("previous_findings", []).append(summary["current_understanding"])
        session.investigation_memory.setdefault("task_references", []).append(task.task_id)
        session.ai_suggestions = _suggest_next_investigations(session, final_state, task.task_id)
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
    final_state = report_node(final_state)
    session.final_executive_report = final_state.get("final_report")
    final_state["collaborative_final_report"] = final_state.get("final_report")

    return CollaborativeRunResult(
        final_state=final_state,
        session=session.to_dict(),
        desk=_build_desk_view(session),
        task_outputs=task_outputs,
    )
