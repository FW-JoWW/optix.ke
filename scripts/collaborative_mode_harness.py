from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from collaborative_mode.orchestrator import CollaborativeRunResult, run_collaborative_investigation
from scripts.guided_mode_harness import build_guided_sample_dataframe


@dataclass
class CollaborativeHarnessResult:
    final_state: Dict[str, Any]
    session: Dict[str, Any]
    desk: Dict[str, Any]
    task_outputs: Dict[str, Dict[str, Any]]


def default_collaborative_responses() -> List[str]:
    return [
        "new investigation",
        "compare results",
        "challenge finding",
        "finish investigation",
    ]


def run_collaborative_workflow(
    question: str,
    responses: Sequence[str] | None = None,
    dataset_path: str | None = None,
    dataframe: Any | None = None,
    initial_tasks: Sequence[Dict[str, Any] | str] | None = None,
) -> CollaborativeHarnessResult:
    """
    Script-friendly collaborative harness.

    The responses are accepted for compatibility with the guided harness style.
    The collaborative orchestrator currently uses a deterministic task queue and
    keeps the session state and evidence store reusable for tests.
    """
    result: CollaborativeRunResult = run_collaborative_investigation(
        question=question,
        responses=responses,
        dataset_path=dataset_path,
        dataframe=dataframe if dataframe is not None else build_guided_sample_dataframe(),
        initial_tasks=initial_tasks,
    )
    return CollaborativeHarnessResult(
        final_state=result.final_state,
        session=result.session,
        desk=result.desk,
        task_outputs=result.task_outputs,
    )


def summarize_collaborative_result(result: CollaborativeHarnessResult) -> Dict[str, Any]:
    final_state = result.final_state
    evidence = final_state.get("analysis_evidence", {}) or {}
    session = result.session
    return {
        "investigation_id": session.get("investigation_id"),
        "current_status": session.get("current_status"),
        "completed_tasks": session.get("completed_tasks", []),
        "running_tasks": session.get("running_tasks", []),
        "queued_tasks": session.get("queued_tasks", []),
        "final_report_available": bool(final_state.get("final_report")),
        "evidence_count": len(session.get("evidence_store", {})),
        "hypothesis_count": len(session.get("hypotheses", {})),
        "ai_suggestions": session.get("ai_suggestions", []),
        "current_understanding": session.get("investigation_memory", {}).get("current_understanding"),
        "task_outputs": result.task_outputs,
        "final_output": evidence.get("final_output"),
    }
