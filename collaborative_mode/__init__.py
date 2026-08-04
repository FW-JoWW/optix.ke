from __future__ import annotations

from .models import (
    CollaborativeTask,
    EvidenceRecord,
    HypothesisRecord,
    InvestigationSession,
)
from .task_manager import TaskManager

__all__ = [
    "CollaborativeTask",
    "EvidenceRecord",
    "HypothesisRecord",
    "InvestigationSession",
    "CollaborativeSessionController",
    "CollaborativeRunResult",
    "TaskManager",
    "run_interactive_collaborative_session",
    "run_collaborative_investigation",
]


def __getattr__(name: str):
    if name in {"CollaborativeSessionController", "run_interactive_collaborative_session"}:
        from .session_runner import CollaborativeSessionController, run_interactive_collaborative_session

        globals().update(
            {
                "CollaborativeSessionController": CollaborativeSessionController,
                "run_interactive_collaborative_session": run_interactive_collaborative_session,
            }
        )
        return globals()[name]

    if name in {"CollaborativeRunResult", "run_collaborative_investigation"}:
        from .orchestrator import CollaborativeRunResult, run_collaborative_investigation

        globals().update(
            {
                "CollaborativeRunResult": CollaborativeRunResult,
                "run_collaborative_investigation": run_collaborative_investigation,
            }
        )
        return globals()[name]

    raise AttributeError(f"module 'collaborative_mode' has no attribute {name!r}")
