from __future__ import annotations

from typing import Any, Dict, List, Optional


_INVESTIGATION_REGISTRY: Dict[str, Any] = {}


def register_investigation(controller: Any) -> None:
    session = getattr(controller, "session", None)
    investigation_id = getattr(session, "investigation_id", None)
    if investigation_id:
        _INVESTIGATION_REGISTRY[investigation_id] = controller


def unregister_investigation(investigation_id: str) -> None:
    _INVESTIGATION_REGISTRY.pop(investigation_id, None)


def get_investigation(investigation_id: str) -> Any | None:
    return _INVESTIGATION_REGISTRY.get(investigation_id)


def list_investigations(include_closed: bool = True) -> List[Dict[str, Any]]:
    investigations: List[Dict[str, Any]] = []
    for controller in _INVESTIGATION_REGISTRY.values():
        session = getattr(controller, "session", None)
        if session is None:
            continue
        if not include_closed and getattr(session, "current_status", None) in {"completed", "archived"}:
            continue
        investigations.append(
            {
                "investigation_id": session.investigation_id,
                "title": getattr(session, "investigation_title", None) or session.original_question,
                "objective": getattr(session, "objective", None) or session.original_question,
                "status": getattr(session, "current_status", None),
                "completed_tasks": len(getattr(session, "completed_tasks", []) or []),
                "queued_tasks": len(getattr(session, "queued_tasks", []) or []),
                "current_understanding": getattr(session, "investigation_memory", {}).get("current_understanding"),
                "archived_at": getattr(session, "archived_at", None),
                "closed_at": getattr(session, "closed_at", None),
            }
        )
    return investigations



