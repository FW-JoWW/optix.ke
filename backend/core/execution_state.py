from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_execution_evidence(state: Dict[str, Any]) -> Dict[str, Any]:
    evidence = state.setdefault("analysis_evidence", {})
    evidence.setdefault("execution_trace", [])
    evidence.setdefault("evidence_provenance", {})
    return evidence


def record_execution_event(
    state: Dict[str, Any],
    *,
    phase: str,
    message: str,
    progress: int | None = None,
    operation: str | None = None,
    status: str | None = None,
    evidence_scope: str | None = None,
    waiting_for: str | None = None,
    details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    evidence = ensure_execution_evidence(state)
    event: Dict[str, Any] = {
        "timestamp": utc_now(),
        "phase": phase,
        "message": message,
    }
    if progress is not None:
        event["progress"] = max(0, min(100, int(progress)))
    if operation:
        event["operation"] = operation
    if status:
        event["status"] = status
    if evidence_scope:
        event["evidence_scope"] = evidence_scope
    if waiting_for:
        event["waiting_for"] = waiting_for
    if details:
        event["details"] = details

    trace: List[Dict[str, Any]] = evidence.setdefault("execution_trace", [])
    trace.append(event)
    state["workflow_status"] = {
        "phase": phase,
        "message": message,
        "progress": event.get("progress", state.get("workflow_status", {}).get("progress", 0)),
        "current_operation": operation or state.get("workflow_status", {}).get("current_operation"),
        "status": status or state.get("workflow_status", {}).get("status") or "running",
        "updated_at": event["timestamp"],
    }
    if waiting_for:
        state["workflow_status"]["waiting_for"] = waiting_for
    if evidence_scope:
        state["workflow_status"]["evidence_scope"] = evidence_scope
    if details:
        state["workflow_status"]["details"] = details

    live_hook = state.get("_live_status_hook")
    if callable(live_hook):
        try:
            live_hook(state, event)
        except Exception:
            pass
    return event


def record_evidence_provenance(
    state: Dict[str, Any],
    key: str,
    *,
    scope: str,
    source: str,
    verified: bool,
    method: str | None = None,
    notes: str | None = None,
) -> Dict[str, Any]:
    evidence = ensure_execution_evidence(state)
    provenance = evidence.setdefault("evidence_provenance", {})
    payload: Dict[str, Any] = {
        "scope": scope,
        "source": source,
        "verified": verified,
    }
    if method:
        payload["method"] = method
    if notes:
        payload["notes"] = notes
    provenance[key] = payload
    return payload
