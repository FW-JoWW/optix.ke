from __future__ import annotations

import json
import base64
import mimetypes
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable
from urllib.parse import unquote, urlparse

import pandas as pd

from backend.collaborative_mode.registry import get_investigation as get_live_collaborative_investigation

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_DIR = ROOT_DIR / "backend" / "data"
DEFAULT_HOST = os.getenv("DAA_API_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("DAA_API_PORT", "8787"))

_INVESTIGATIONS: dict[str, dict[str, Any]] = {}
_INVESTIGATION_RECORDS: dict[str, dict[str, Any]] = {}
_INVESTIGATION_ORDER: list[str] = []
_CACHE_LOCK = threading.Lock()
_DATASET_CACHE: dict[str, dict[str, Any]] = {}
_ACTION_WORKERS: dict[str, threading.Thread] = {}


_WORKFLOW_PHASES = {
    "autonomous": [
        ("preparing", "Preparing the dataset", 12),
        ("analyzing", "Running the analytical workflow", 38),
        ("synthesizing", "Synthesizing findings", 64),
        ("finalizing", "Finalizing the answer", 86),
    ],
    "guided": [
        ("preparing", "Preparing the guided workflow", 10),
        ("checking", "Working through guided checkpoints", 34),
        ("reviewing", "Reviewing evidence and responses", 62),
        ("finalizing", "Assembling the guided report", 85),
    ],
    "collaborative": [
        ("preparing", "Preparing collaborative tasks", 10),
        ("coordinating", "Coordinating tasks and hypotheses", 32),
        ("synthesizing", "Synthesizing shared evidence", 60),
        ("finalizing", "Finalizing the collaborative answer", 84),
    ],
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_investigation_id() -> str:
    return f"inv-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}-{uuid.uuid4().hex[:6]}"


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{int(size)} B"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if value is pd.NA:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        preview = value.head(12).copy()
        preview = preview.where(pd.notna(preview), None)
        return {
            "kind": "dataframe",
            "row_count": int(len(value)),
            "column_count": int(len(value.columns)),
            "columns": [str(col) for col in value.columns],
            "preview_rows": preview.to_dict(orient="records"),
        }
    if isinstance(value, pd.Series):
        return _jsonable(value.to_list())
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _resolve_dataset_path(dataset_path: str | None) -> Path:
    if not dataset_path:
        raise FileNotFoundError("No dataset path was provided.")

    candidate = Path(dataset_path)
    if not candidate.is_absolute():
        candidate = (ROOT_DIR / candidate).resolve()
    else:
        candidate = candidate.resolve()

    data_root = DATA_DIR.resolve()
    if data_root not in candidate.parents and candidate != data_root:
        raise ValueError("Dataset must live inside backend/data.")

    if not candidate.exists():
        raise FileNotFoundError(f"Dataset not found: {candidate}")
    return candidate


def _read_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported dataset type: {suffix}")


def _dataset_metadata(path: Path) -> dict[str, Any]:
    cache_key = str(path.resolve())
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    suffix = path.suffix.lower()
    metadata: dict[str, Any] = {
        "name": path.name,
        "path": str(path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "source_type": "csv" if suffix == ".csv" else "excel" if suffix in {".xlsx", ".xls"} else suffix.lstrip("."),
        "size_bytes": path.stat().st_size,
        "size_label": _human_size(path.stat().st_size),
        "last_modified": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
    }

    try:
        if suffix == ".csv":
            columns = list(pd.read_csv(path, nrows=0).columns)
        elif suffix in {".xlsx", ".xls"}:
            columns = list(pd.read_excel(path, nrows=0).columns)
        else:
            columns = []
        metadata.update(
            {
                "row_count": None,
                "column_count": len(columns) or None,
                "columns": [str(column) for column in columns[:16]],
            }
        )
    except Exception as exc:
        metadata["metadata_warning"] = f"Could not read detailed schema: {exc}"

    _DATASET_CACHE[cache_key] = metadata
    return metadata


def _catalog_datasets() -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    for path in sorted(DATA_DIR.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            continue
        datasets.append(_dataset_metadata(path))
    return datasets


def _default_dataset_path() -> str | None:
    catalog = _catalog_datasets()
    preferred = next((item for item in catalog if item["name"] == "olist_merged_dataset.csv"), None)
    if preferred:
        return preferred["path"]
    return catalog[0]["path"] if catalog else None


def _investigation_summary(record: dict[str, Any]) -> dict[str, Any]:
    evidence = record.get("analysis_evidence", {}) or {}
    answer = evidence.get("answer_synthesis", {}) or {}
    judgment = evidence.get("judgment_summary", {}) or {}
    reports = record.get("report_package", {}) or evidence.get("report_package", {}) or {}
    top_stories = evidence.get("top_stories") or []
    dataset = record.get("dataset") or {}
    confidence = (
        answer.get("confidence", {}).get("overall", {}).get("label")
        or judgment.get("global_confidence")
        or "Unknown"
    )
    status = record.get("status") or ("completed" if record.get("final_report") else "running")
    return {
        "id": record.get("id"),
        "question": record.get("business_question") or record.get("question") or "Untitled investigation",
        "mode": record.get("mode") or "autonomous",
        "status": status,
        "client_request_id": record.get("client_request_id"),
        "dataset": {
            "name": dataset.get("name") or Path(record.get("dataset_path") or "").name,
            "path": record.get("dataset_path"),
            "row_count": (record.get("dataset_profile") or {}).get("row_count") or dataset.get("row_count"),
            "column_count": (record.get("dataset_profile") or {}).get("column_count") or dataset.get("column_count"),
            "source_type": dataset.get("source_type"),
        },
        "confidence": confidence,
        "finding_count": len(top_stories),
        "has_report": bool(record.get("final_report")),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "answer": _safe_text(answer.get("direct_answer") or answer.get("best_available_answer") or judgment.get("summary") or record.get("final_report")),
        "workflow_status": _jsonable(record.get("workflow_status") or {}),
        "reports": {
            "analyst": bool(record.get("analyst_report")),
            "business": bool(record.get("business_report")),
            "executive": bool(record.get("executive_report")),
        },
    }


def _store_investigation(record: dict[str, Any]) -> dict[str, Any]:
    normalized = _coerce_analysis_payload(dict(record))
    sanitized = _public_investigation(normalized)
    investigation_id = sanitized["id"]
    with _CACHE_LOCK:
        _INVESTIGATION_RECORDS[investigation_id] = _jsonable(normalized)
        _INVESTIGATIONS[investigation_id] = sanitized
        if investigation_id in _INVESTIGATION_ORDER:
            _INVESTIGATION_ORDER.remove(investigation_id)
        _INVESTIGATION_ORDER.insert(0, investigation_id)
        _INVESTIGATION_ORDER[:] = _INVESTIGATION_ORDER[:50]
    return sanitized


def _update_investigation(investigation_id: str, **updates: Any) -> None:
    with _CACHE_LOCK:
        existing_record = dict(_INVESTIGATION_RECORDS.get(investigation_id) or {})
        existing_public = dict(_INVESTIGATIONS.get(investigation_id) or {})
        if not existing_record and not existing_public:
            return
        existing_record.update(updates)
        _INVESTIGATION_RECORDS[investigation_id] = existing_record
        _INVESTIGATIONS[investigation_id] = _public_investigation(existing_record)


def _get_investigation_record(investigation_id: str) -> dict[str, Any] | None:
    with _CACHE_LOCK:
        record = _INVESTIGATION_RECORDS.get(investigation_id)
        if record is not None:
            return dict(record)
        public = _INVESTIGATIONS.get(investigation_id)
        if public is not None:
            return dict(public)
    return None


def _workflow_status(mode: str, progress_index: int, total_phases: int, completed: bool = False) -> dict[str, Any]:
    phases = _WORKFLOW_PHASES.get(mode, _WORKFLOW_PHASES["autonomous"])
    phase_key, message, target = phases[min(progress_index, len(phases) - 1)]
    if completed:
        return {"phase": "completed", "message": "Investigation complete", "progress": 100, "current_operation": "final synthesis"}
    drift = min(12, progress_index * 4)
    return {
        "phase": phase_key,
        "message": message,
        "progress": min(94, target + drift),
        "total_phases": total_phases,
        "current_operation": message,
    }


def _list_investigations() -> list[dict[str, Any]]:
    with _CACHE_LOCK:
        ordered = [_INVESTIGATIONS[item] for item in _INVESTIGATION_ORDER if item in _INVESTIGATIONS]
    return [_investigation_summary(item) for item in ordered]


def _trim_tool_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"summary": _safe_text(value)}
    keys = (
        "tool",
        "type",
        "summary",
        "insight",
        "message",
        "direct_answer",
        "business_interpretation",
        "recommended_next_step",
        "confidence",
        "caption",
        "title",
        "file_path",
        "path",
    )
    result = {key: _jsonable(value.get(key)) for key in keys if value.get(key) is not None}
    if not result:
        result["summary"] = _safe_text(value)
    return result


def _trim_story(story: Any) -> dict[str, Any]:
    if not isinstance(story, dict):
        return {"summary": _safe_text(story)}
    return {
        "headline": _safe_text(story.get("headline") or story.get("insight") or story.get("summary") or story.get("finding")),
        "insight": _safe_text(story.get("insight") or story.get("summary") or story.get("headline")),
        "summary": _safe_text(story.get("summary") or story.get("detail") or story.get("explanation")),
        "business_implication": _safe_text(story.get("business_implication") or story.get("recommendation") or story.get("interpretation")),
        "confidence": _safe_text(story.get("confidence") or story.get("score") or ""),
        "relationship_type": _safe_text(story.get("relationship_type") or story.get("type") or "supporting"),
        "method_used": _safe_text(story.get("method_used") or story.get("method") or story.get("analysis") or story.get("tool") or "Analytical evidence"),
        "supporting_evidence": _trim_sequence(story.get("supporting_evidence") or story.get("evidence") or [], limit=4),
        "limitations": _trim_sequence(story.get("limitations") or story.get("warnings") or [], limit=3),
    }


def _trim_decision(decision: Any) -> dict[str, Any]:
    if not isinstance(decision, dict):
        return {"summary": _safe_text(decision)}
    return {
        "recommended_action": _safe_text(decision.get("recommended_action") or decision.get("action") or decision.get("title")),
        "decision_summary": _safe_text(decision.get("decision_summary") or decision.get("summary") or decision.get("description")),
        "impact_assessment": _jsonable(decision.get("impact_assessment") or {}),
        "confidence_in_action": _safe_text(decision.get("confidence_in_action") or decision.get("confidence") or ""),
        "priority": _jsonable(decision.get("priority") or {}),
        "recommendation_restrictions": _trim_sequence(decision.get("recommendation_restrictions") or [], limit=3),
    }


def _trim_mapping(value: Any, limit: int = 8) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    trimmed: dict[str, Any] = {}
    for index, (key, item) in enumerate(value.items()):
        if index >= limit:
            break
        trimmed[str(key)] = _jsonable(item)
    return trimmed


def _trim_sequence(value: Any, limit: int = 8) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        return []
    return [_jsonable(item) for item in list(value)[:limit]]


def _trim_visualization(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
      return {"summary": _safe_text(value)}
    result = {
        "type": _safe_text(value.get("type") or value.get("chart_type") or ""),
        "title": _safe_text(value.get("title") or value.get("caption") or value.get("label") or ""),
        "caption": _safe_text(value.get("caption") or value.get("summary") or ""),
        "file_path": _safe_text(value.get("file_path") or value.get("path") or ""),
        "priority": _safe_text(value.get("priority") or ""),
        "based_on": _jsonable(value.get("based_on") or {}),
    }
    file_path = result.get("file_path") or ""
    if file_path:
        chart_path = Path(file_path)
        if not chart_path.is_absolute():
            chart_path = ROOT_DIR / file_path
        if not chart_path.exists():
            chart_path = ROOT_DIR / "backend" / "charts" / Path(file_path).name
        try:
            if chart_path.exists() and chart_path.is_file():
                mime_type = mimetypes.guess_type(chart_path.name)[0] or "image/png"
                encoded = base64.b64encode(chart_path.read_bytes()).decode("ascii")
                result["data_url"] = f"data:{mime_type};base64,{encoded}"
        except Exception:
            pass
    return result


def _story_from_text(
    headline: str,
    insight: str,
    *,
    summary: str = "",
    confidence: Any = "",
    relationship_type: str = "supporting",
    method_used: str = "Analytical evidence",
) -> dict[str, Any]:
    return {
        "headline": _safe_text(headline),
        "insight": _safe_text(insight),
        "summary": _safe_text(summary or insight),
        "business_implication": _safe_text(summary or insight),
        "confidence": _safe_text(confidence),
        "relationship_type": _safe_text(relationship_type),
        "method_used": _safe_text(method_used),
        "supporting_evidence": [],
        "limitations": [],
    }


def _story_from_task_output(task_id: str, task_output: Any) -> dict[str, Any] | None:
    if not isinstance(task_output, dict):
        task_output = {"summary": task_output}
    insight = (
        task_output.get("current_understanding")
        or task_output.get("task_finding")
        or task_output.get("narrative")
        or task_output.get("report_excerpt")
        or task_output.get("summary")
        or task_output.get("task_request")
        or task_id
    )
    summary = task_output.get("narrative") or task_output.get("task_finding") or task_output.get("summary") or insight
    confidence = task_output.get("confidence") or task_output.get("integrity_status") or ""
    return _story_from_text(
        task_output.get("task_title") or task_id,
        insight,
        summary=summary,
        confidence=confidence,
        relationship_type=task_output.get("status") or "collaborative",
        method_used="Collaborative task synthesis",
    )


def _coerce_analysis_payload(record: dict[str, Any], *, task_outputs: dict[str, Any] | None = None) -> dict[str, Any]:
    evidence = dict(record.get("analysis_evidence") or {})
    answer = dict(evidence.get("answer_synthesis") or record.get("answer_synthesis") or {})
    judgment = dict(evidence.get("judgment_summary") or record.get("judgment_summary") or {})
    report_package = dict(evidence.get("report_package") or record.get("report_package") or {})
    top_stories = list(evidence.get("top_stories") or record.get("top_stories") or [])
    decision_recommendations = list(evidence.get("decision_recommendations") or record.get("decision_recommendations") or [])
    tool_results = dict(evidence.get("tool_results") or record.get("tool_results") or {})
    visualizations = list(evidence.get("visualizations") or record.get("visualizations") or [])

    if task_outputs:
        evidence["collaborative_task_outputs"] = _jsonable(task_outputs)
        if not tool_results:
            tool_results = {str(task_id): _jsonable(summary) for task_id, summary in task_outputs.items()}
        task_stories = [
            story
            for story in (
                _story_from_task_output(task_id, summary)
                for task_id, summary in task_outputs.items()
            )
            if story
        ]
        top_stories = task_stories + top_stories
        if task_stories and not decision_recommendations:
            decision_recommendations = [
                {
                    "recommended_action": story.get("headline") or "Review collaborative task",
                    "decision_summary": story.get("summary") or story.get("insight") or "Collaborative task completed.",
                    "confidence_in_action": story.get("confidence") or "Moderate",
                    "impact_assessment": {"impact_summary": story.get("business_implication") or ""},
                    "priority": {"label": "Medium"},
                    "recommendation_restrictions": [],
                }
                for story in task_stories[:3]
            ]

    if not top_stories:
        fallback_text = (
            answer.get("direct_answer")
            or answer.get("best_available_answer")
            or judgment.get("summary")
            or judgment.get("dominant_reasoning")
            or record.get("final_report")
            or ""
        )
        if fallback_text:
            top_stories.append(
                _story_from_text(
                    answer.get("answer_position") or "Primary result",
                    fallback_text,
                    summary=answer.get("business_interpretation") or judgment.get("summary") or fallback_text,
                    confidence=answer.get("confidence", {}).get("overall", {}).get("label") if isinstance(answer.get("confidence"), dict) else judgment.get("global_confidence") or "",
                    relationship_type="supporting",
                    method_used="Final synthesis",
                )
            )

    if not answer:
        best_answer = (
            top_stories[0].get("insight")
            if top_stories
            else record.get("final_report")
            or judgment.get("summary")
            or ""
        )
        answer = {
            "direct_answer": _safe_text(best_answer),
            "best_available_answer": _safe_text(best_answer),
            "business_interpretation": _safe_text(judgment.get("summary") or best_answer),
            "supporting_evidence_summary": [top_stories[0].get("summary") or top_stories[0].get("insight")] if top_stories else [],
            "observed_facts": [top_stories[0].get("insight")] if top_stories else [],
            "analytical_interpretation": [top_stories[0].get("summary") or top_stories[0].get("insight")] if top_stories else [],
            "remaining_uncertainty": [],
            "recommended_next_investigation": [decision_recommendations[0].get("decision_summary")] if decision_recommendations else [],
            "answer_position": answer.get("answer_position") if isinstance(answer, dict) else "unknown",
            "confidence": {
                "overall": {
                    "label": judgment.get("global_confidence") or "Unknown",
                    "score": judgment.get("global_confidence"),
                    "reason": judgment.get("summary") or "",
                }
            },
        }

    if not report_package:
        report_package = {
            "analyst_report": record.get("analyst_report") or "",
            "business_report": record.get("business_report") or _safe_text(answer.get("business_interpretation") or answer.get("direct_answer") or record.get("final_report") or ""),
            "executive_report": record.get("executive_report") or _safe_text(record.get("final_report") or answer.get("direct_answer") or ""),
            "master_report": record.get("master_report") or _safe_text(record.get("final_report") or answer.get("direct_answer") or ""),
            "traceability": _jsonable(judgment),
        }

    if not visualizations and top_stories:
        visualizations = list(evidence.get("visualizations") or record.get("visualizations") or [])

    evidence["answer_synthesis"] = _jsonable(answer)
    evidence["judgment_summary"] = _jsonable(judgment)
    evidence["top_stories"] = [_jsonable(story) for story in top_stories[:6]]
    evidence["decision_recommendations"] = [_jsonable(item) for item in decision_recommendations[:6]]
    evidence["tool_results"] = _jsonable(tool_results)
    evidence["visualizations"] = [_jsonable(item) for item in visualizations[:6]]
    evidence["report_package"] = _jsonable(report_package)

    record["analysis_evidence"] = evidence
    record["answer_synthesis"] = answer
    record["judgment_summary"] = judgment
    record["report_package"] = report_package
    record["top_stories"] = top_stories[:6]
    record["decision_recommendations"] = decision_recommendations[:6]
    record["tool_results"] = tool_results
    record["visualizations"] = visualizations[:6]
    if not record.get("analyst_report"):
        record["analyst_report"] = _safe_text(report_package.get("analyst_report") or "")
    if not record.get("business_report"):
        record["business_report"] = _safe_text(report_package.get("business_report") or "")
    if not record.get("executive_report"):
        record["executive_report"] = _safe_text(report_package.get("executive_report") or "")
    if not record.get("master_report"):
        record["master_report"] = _safe_text(report_package.get("master_report") or "")
    return record


def _make_live_status_hook(record_id: str):
    def _hook(state: dict[str, Any], event: dict[str, Any]) -> None:
        snapshot = _coerce_analysis_payload(_jsonable(state))
        snapshot["id"] = record_id
        snapshot.setdefault("created_at", state.get("created_at") or event.get("timestamp"))
        snapshot["updated_at"] = event.get("timestamp") or _utc_now()
        snapshot["status"] = state.get("status") or ("awaiting_user" if state.get("awaiting_user") else "running")
        snapshot["workflow_status"] = _jsonable(state.get("workflow_status") or {})
        _update_investigation(record_id, **snapshot)

    return _hook


def _public_investigation(record: dict[str, Any]) -> dict[str, Any]:
    record = _coerce_analysis_payload(dict(record))
    evidence = record.get("analysis_evidence") or {}
    report_package = evidence.get("report_package") or record.get("report_package") or {}
    tool_results = evidence.get("tool_results") or record.get("tool_results") or {}
    collaborative_session = evidence.get("collaborative_session") or record.get("collaborative_session") or {}
    report_bundle = {
        "analyst_report": _safe_text(record.get("analyst_report") or report_package.get("analyst_report") or ""),
        "business_report": _safe_text(record.get("business_report") or report_package.get("business_report") or ""),
        "executive_report": _safe_text(record.get("executive_report") or report_package.get("executive_report") or ""),
        "master_report": _safe_text(record.get("master_report") or report_package.get("master_report") or ""),
    }
    return {
        "id": record.get("id"),
        "question": record.get("business_question") or record.get("question") or "Untitled investigation",
        "business_question": record.get("business_question") or record.get("question"),
        "mode": record.get("mode") or "autonomous",
        "status": record.get("status") or ("completed" if record.get("final_report") else "running"),
        "answer": _safe_text(
            evidence.get("answer_synthesis", {}).get("direct_answer")
            or record.get("answer")
            or record.get("final_report")
            or record.get("business_report")
            or record.get("executive_report")
        ),
        "dataset_path": record.get("dataset_path"),
        "client_request_id": record.get("client_request_id"),
        "dataset": record.get("dataset") or {},
        "dataset_profile": record.get("dataset_profile") or {},
        "analysis_plan": record.get("analysis_plan") or evidence.get("analysis_plan") or [],
        "analysis_evidence": {
            "answer_synthesis": _jsonable(evidence.get("answer_synthesis") or record.get("answer_synthesis") or {}),
            "judgment_summary": _jsonable(evidence.get("judgment_summary") or record.get("judgment_summary") or {}),
            "top_stories": [_trim_story(story) for story in _trim_sequence(evidence.get("top_stories") or record.get("top_stories") or [], limit=6)],
            "decision_recommendations": [_trim_decision(item) for item in _trim_sequence(evidence.get("decision_recommendations") or record.get("decision_recommendations") or [], limit=6)],
            "tool_results": {str(key): _trim_tool_result(value) for key, value in list(tool_results.items())[:6]},
            "visualizations": [_trim_visualization(item) for item in _trim_sequence(evidence.get("visualizations") or record.get("visualizations") or [], limit=6)],
            "report_package": report_bundle,
            "guided_version_snapshots": _trim_mapping(evidence.get("guided_version_snapshots") or record.get("guided_version_snapshots") or {}, limit=8),
            "guided_checkpoint_summaries": _trim_mapping(evidence.get("guided_checkpoint_summaries") or record.get("guided_checkpoint_summaries") or {}, limit=8),
            "execution_trace": _trim_sequence(evidence.get("execution_trace") or record.get("execution_trace") or [], limit=20),
            "evidence_provenance": _jsonable(evidence.get("evidence_provenance") or record.get("evidence_provenance") or {}),
            "collaborative_session": {
                "tasks": _trim_sequence(collaborative_session.get("tasks") if isinstance(collaborative_session, dict) else [], limit=8),
                "hypotheses": _trim_sequence(collaborative_session.get("hypotheses") if isinstance(collaborative_session, dict) else [], limit=8),
                "evidence_store": _trim_mapping(collaborative_session.get("evidence_store") if isinstance(collaborative_session, dict) else {}, limit=8),
            },
            "collaborative_evidence_store": _trim_mapping(evidence.get("collaborative_evidence_store") or record.get("collaborative_evidence_store") or {}, limit=8),
        },
        "guided_decision_log": _trim_sequence(record.get("guided_decision_log") or evidence.get("guided_decision_log") or [], limit=8),
        "collaborative_decision_log": _trim_sequence(record.get("collaborative_decision_log") or evidence.get("collaborative_decision_log") or [], limit=8),
        "final_report": record.get("final_report") or "",
        **report_bundle,
        "answer_synthesis_report": record.get("answer_synthesis_report") or evidence.get("answer_synthesis_report") or "",
        "investigation_decision_report": record.get("investigation_decision_report") or evidence.get("investigation_decision_report") or "",
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "selected_columns": _trim_sequence(record.get("selected_columns") or [], limit=16),
        "visualizations": _trim_sequence(record.get("visualizations") or [], limit=6),
        "awaiting_user": record.get("awaiting_user") or False,
        "workflow_status": _jsonable(record.get("workflow_status") or {}),
        "raw": {},
    }


def _load_bootstrap() -> dict[str, Any]:
    datasets = _catalog_datasets()
    return {
        "app": {
            "name": "Data Analyst Agent",
            "tagline": "Investigation-centered analytics workspace",
            "version": "frontend-bridge",
        },
        "modes": [
            {"id": "autonomous", "label": "Autonomous", "description": "Agent investigates independently and presents the conclusion."},
            {"id": "guided", "label": "Guided", "description": "Agent pauses at checkpoints for review, modification, or continuation."},
            {"id": "collaborative", "label": "Collaborative", "description": "Human and agent investigate together through tasks and evidence."},
        ],
        "datasets": datasets,
        "recentInvestigations": _list_investigations(),
        "suggestedQuestions": [
            "Why did sales decline last quarter?",
            "Which region has the strongest growth?",
            "What is driving delivery delays?",
        ],
        "defaultDatasetPath": _default_dataset_path(),
        "serverTime": _utc_now(),
    }


def _run_autonomous(question: str, dataset_path: str, mode: str, *, record_id: str | None = None) -> dict[str, Any]:
    from backend.graph.analyst_graph import graph

    df = _read_dataframe(_resolve_dataset_path(dataset_path))
    state: dict[str, Any] = {
        "business_question": question,
        "dataset_path": dataset_path,
        "dataframe": df,
        "mode": mode,
        "enable_llm_reasoning": True,
        "disable_llm_reasoning": False,
        "disable_semantic_matcher": True,
        "analysis_evidence": {},
        "fast_finalization": False,
    }
    if record_id:
        state["id"] = record_id
        state["investigation_id"] = record_id
        state["_live_status_hook"] = _make_live_status_hook(record_id)
    final_state = graph.invoke(state)
    return _coerce_analysis_payload(_jsonable(final_state))


def _run_guided(question: str, dataset_path: str, responses: list[str] | None, *, record_id: str | None = None) -> dict[str, Any]:
    from backend.scripts.guided_mode_harness import default_guided_responses, run_guided_workflow

    df = _read_dataframe(_resolve_dataset_path(dataset_path))
    guided_responses = list(responses or default_guided_responses())
    if len(guided_responses) < 12:
        guided_responses.extend(["continue"] * (12 - len(guided_responses)))
    result = run_guided_workflow(
        question=question,
        responses=guided_responses,
        dataset_path=dataset_path,
        dataframe=df,
        fast_finalization=False,
        record_id=record_id,
        live_status_hook=_make_live_status_hook(record_id) if record_id else None,
    )
    return _coerce_analysis_payload(_jsonable(result.final_state))


def _run_collaborative(
    question: str,
    dataset_path: str,
    responses: list[str] | None,
    initial_tasks: list[Any] | None,
    *,
    record_id: str | None = None,
) -> dict[str, Any]:
    from backend.scripts.collaborative_mode_harness import default_collaborative_responses, run_collaborative_workflow

    df = _read_dataframe(_resolve_dataset_path(dataset_path))
    collaborative_responses = list(responses or [])
    if not collaborative_responses:
        collaborative_responses = default_collaborative_responses()
    while len(collaborative_responses) < 8:
        collaborative_responses.extend(default_collaborative_responses())
    result = run_collaborative_workflow(
        question=question,
        dataset_path=dataset_path,
        dataframe=df,
        responses=collaborative_responses,
        initial_tasks=initial_tasks,
        build_final_report=True,
        fast_finalization=False,
        record_id=record_id,
        live_status_hook=_make_live_status_hook(record_id) if record_id else None,
    )
    return _coerce_analysis_payload(_jsonable(result.final_state), task_outputs=result.task_outputs)


def _run_investigation_sync(payload: dict[str, Any], record_id: str | None = None) -> dict[str, Any]:
    question = _safe_text(payload.get("question") or payload.get("business_question"))
    if not question:
        raise ValueError("A business question is required.")

    mode = _safe_text(payload.get("mode") or "autonomous").lower()
    dataset_path = _safe_text(payload.get("datasetPath") or payload.get("dataset_path") or _default_dataset_path())
    if not dataset_path:
        raise ValueError("No dataset is available.")

    created_at = _utc_now()
    if mode == "guided":
        final_state = _run_guided(question, dataset_path, payload.get("guidedResponses") or payload.get("responses"), record_id=record_id)
    elif mode == "collaborative":
        final_state = _run_collaborative(
            question,
            dataset_path,
            payload.get("collaborativeResponses") or payload.get("responses"),
            payload.get("initialTasks") or payload.get("initial_tasks"),
            record_id=record_id,
        )
    else:
        mode = "autonomous"
        final_state = _run_autonomous(question, dataset_path, mode, record_id=record_id)

    dataset_meta = _dataset_metadata(_resolve_dataset_path(dataset_path))
    record_id = _safe_text(record_id or final_state.get("id") or final_state.get("investigation_id") or _new_investigation_id())
    final_state = _coerce_analysis_payload(final_state)
    final_state.update(
        {
            "id": record_id,
            "question": question,
            "mode": mode,
            "client_request_id": _safe_text(payload.get("clientRequestId") or payload.get("client_request_id") or ""),
            "dataset_path": dataset_path,
            "dataset": dataset_meta,
            "created_at": created_at,
            "updated_at": _utc_now(),
            "status": "completed" if final_state.get("final_report") else ("awaiting_user" if final_state.get("awaiting_user") else "running"),
        }
    )
    _store_investigation(final_state)
    return final_state


def _extract_initial_tasks_from_record(record: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if not record:
        return None
    evidence = record.get("analysis_evidence") or {}
    collaborative_session = evidence.get("collaborative_session") or record.get("collaborative_session") or {}
    tasks = collaborative_session.get("tasks") if isinstance(collaborative_session, dict) else []
    initial_tasks: list[dict[str, Any]] = []
    for index, task in enumerate(tasks or []):
        if not isinstance(task, dict):
            continue
        request = _safe_text(task.get("request") or task.get("title") or task.get("summary") or "")
        if not request:
            continue
        initial_tasks.append(
            {
                "title": _safe_text(task.get("title") or request or f"Task {index + 1}"),
                "request": request,
                "dependencies": task.get("dependencies") or [],
                "parent_task_id": task.get("parent_task_id"),
                "metadata": task.get("metadata") or {},
            }
        )
    return initial_tasks or None


def _continue_investigation_action(investigation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    record = _get_investigation_record(investigation_id)
    if not record:
        raise KeyError("Investigation not found.")

    stored_mode = _safe_text(record.get("mode") or payload.get("mode") or "autonomous").lower()
    question = _safe_text(payload.get("question") or record.get("business_question") or record.get("question"))
    dataset_path = _safe_text(payload.get("datasetPath") or payload.get("dataset_path") or record.get("dataset_path") or _default_dataset_path())
    action = _safe_text(payload.get("action") or "").strip().lower()
    details = _safe_text(payload.get("details") or "")

    if stored_mode == "guided":
        final_state = _run_guided(
            question or _safe_text(record.get("question") or record.get("business_question")),
            dataset_path,
            payload.get("guidedResponses") or payload.get("responses"),
        )
        final_state["id"] = investigation_id
        final_state["client_request_id"] = _safe_text(payload.get("clientRequestId") or payload.get("client_request_id") or investigation_id)
        final_state["mode"] = "guided"
        final_state["question"] = question or _safe_text(record.get("question") or record.get("business_question"))
        final_state["business_question"] = final_state["question"]
        final_state["dataset_path"] = dataset_path
        final_state["dataset"] = _dataset_metadata(_resolve_dataset_path(dataset_path))
        final_state["status"] = "completed" if final_state.get("final_report") else ("awaiting_user" if final_state.get("awaiting_user") else "running")
        _store_investigation(final_state)
        return {"investigation": _public_investigation(final_state), "summary": _investigation_summary(final_state)}

    if stored_mode == "collaborative":
        live_controller = get_live_collaborative_investigation(investigation_id)
        if live_controller is not None:
            action_result = live_controller.apply_action(action or "queue", details)
            if action_result.get("run_next"):
                if live_controller.queue_paused:
                    snapshot = live_controller._snapshot()
                else:
                    snapshot = live_controller.process_next_task()
            elif live_controller.finished:
                snapshot = live_controller.finalize()
            else:
                snapshot = live_controller._snapshot()
            final_state = snapshot.final_state
            final_state["id"] = investigation_id
            final_state["client_request_id"] = _safe_text(payload.get("clientRequestId") or payload.get("client_request_id") or investigation_id)
            final_state["mode"] = "collaborative"
            final_state["question"] = question or _safe_text(record.get("question") or record.get("business_question"))
            final_state["business_question"] = final_state["question"]
            final_state["dataset_path"] = dataset_path
            final_state["dataset"] = _dataset_metadata(_resolve_dataset_path(dataset_path))
            final_state["status"] = "completed" if final_state.get("final_report") else ("awaiting_user" if final_state.get("awaiting_user") else "running")
            _store_investigation(final_state)
            return {
                "investigation": _public_investigation(final_state),
                "summary": _investigation_summary(final_state),
                "action_result": action_result,
            }

        initial_tasks = payload.get("initialTasks") or payload.get("initial_tasks") or _extract_initial_tasks_from_record(record)
        from backend.collaborative_mode.orchestrator import run_collaborative_investigation

        final_state = run_collaborative_investigation(
            question=question or _safe_text(record.get("question") or record.get("business_question")),
            dataset_path=dataset_path,
            responses=payload.get("collaborativeResponses") or payload.get("responses"),
            initial_tasks=initial_tasks,
            build_final_report=False,
        ).final_state
        final_state["id"] = investigation_id
        final_state["client_request_id"] = _safe_text(payload.get("clientRequestId") or payload.get("client_request_id") or investigation_id)
        final_state["mode"] = "collaborative"
        final_state["question"] = question or _safe_text(record.get("question") or record.get("business_question"))
        final_state["business_question"] = final_state["question"]
        final_state["dataset_path"] = dataset_path
        final_state["dataset"] = _dataset_metadata(_resolve_dataset_path(dataset_path))
        final_state["status"] = "completed" if final_state.get("final_report") else ("awaiting_user" if final_state.get("awaiting_user") else "running")
        _store_investigation(final_state)
        return {"investigation": _public_investigation(final_state), "summary": _investigation_summary(final_state)}

    raise ValueError(f"Investigation {investigation_id} is not in a resumable mode.")


def _build_running_investigation(payload: dict[str, Any], record_id: str) -> dict[str, Any]:
    question = _safe_text(payload.get("question") or payload.get("business_question") or "Untitled investigation")
    mode = _safe_text(payload.get("mode") or "autonomous").lower() or "autonomous"
    dataset_path = _safe_text(payload.get("datasetPath") or payload.get("dataset_path") or _default_dataset_path() or "")
    dataset = _dataset_metadata(_resolve_dataset_path(dataset_path)) if dataset_path else {}
    initial_tasks = payload.get("initialTasks") or payload.get("initial_tasks") or []
    collaborative_session = {
        "tasks": _trim_sequence(initial_tasks, limit=8) if mode == "collaborative" else [],
        "hypotheses": [],
        "evidence_store": {},
    }
    return {
        "id": record_id,
        "question": question,
        "business_question": question,
        "mode": mode,
        "client_request_id": _safe_text(payload.get("clientRequestId") or payload.get("client_request_id") or ""),
        "dataset_path": dataset_path,
        "dataset": dataset,
        "dataset_profile": {},
        "analysis_plan": [],
        "analysis_evidence": {
            "collaborative_session": collaborative_session,
            "collaborative_evidence_store": {},
        },
        "collaborative_tasks": collaborative_session["tasks"],
        "collaborative_queue": collaborative_session["tasks"],
        "collaborative_hypotheses": [],
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "status": "running",
        "workflow_status": _workflow_status(mode, 0, 4),
        "awaiting_user": False,
        "final_report": "",
    }


def _run_investigation_worker(payload: dict[str, Any], record_id: str) -> None:
    done = threading.Event()
    result_box: dict[str, Any] = {}
    error_box: dict[str, Any] = {}

    def run_graph() -> None:
        try:
            result_box["final_state"] = _run_investigation_sync(payload, record_id=record_id)
        except Exception as exc:
            error_box["error"] = exc
        finally:
            done.set()

    graph_thread = threading.Thread(target=run_graph, daemon=True)
    graph_thread.start()

    try:
        while not done.wait(3.0):
            current = _get_investigation_record(record_id)
            if current:
                _update_investigation(
                    record_id,
                    updated_at=_utc_now(),
                    workflow_status=current.get("workflow_status") or {},
                )
        graph_thread.join()
        if "error" in error_box:
            raise error_box["error"]
        final_state = result_box.get("final_state") or {}
        final_state.setdefault("workflow_status", {})
        final_state["workflow_status"].update(
            {
                "phase": "completed",
                "message": "Investigation complete",
                "progress": 100,
                "current_operation": "final synthesis",
                "status": "completed",
            }
        )
        final_state["status"] = "completed" if final_state.get("final_report") else final_state.get("status") or "running"
        final_state["updated_at"] = _utc_now()
        _store_investigation(final_state)
    except Exception as exc:
        _update_investigation(
            record_id,
            status="failed",
            updated_at=_utc_now(),
            workflow_status={"phase": "failed", "message": _safe_text(exc) or "Investigation failed", "progress": 0},
            final_report="",
        )


def _start_investigation_job(payload: dict[str, Any]) -> dict[str, Any]:
    record_id = _safe_text(
        payload.get("clientRequestId")
        or payload.get("client_request_id")
        or _new_investigation_id()
    )
    running = _build_running_investigation(payload, record_id)
    _store_investigation(running)
    worker = threading.Thread(target=_run_investigation_worker, args=(dict(payload), record_id), daemon=True)
    worker.start()
    return running


def _mark_action_processing(investigation_id: str, action: str, details: str) -> dict[str, Any] | None:
    current = _get_investigation_record(investigation_id)
    if not current:
        return None
    _update_investigation(
        investigation_id,
        action_state="processing",
        active_action=action,
        active_action_details=details,
        updated_at=_utc_now(),
        workflow_status={
            "phase": "processing",
            "message": f"Accepted {action or 'action'}. Processing in the background.",
            "progress": 58,
        },
        status="running",
    )
    return _get_investigation_record(investigation_id)


def _run_action_worker(investigation_id: str, payload: dict[str, Any]) -> None:
    try:
        _continue_investigation_action(investigation_id, payload)
    except Exception as exc:
        _update_investigation(
            investigation_id,
            action_state="failed",
            updated_at=_utc_now(),
            status="failed",
            workflow_status={"phase": "failed", "message": _safe_text(exc) or "Action failed", "progress": 0},
        )
    finally:
        with _CACHE_LOCK:
            _ACTION_WORKERS.pop(investigation_id, None)


def _dispatch_action_continuation(investigation_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    snapshot = _mark_action_processing(investigation_id, _safe_text(payload.get("action") or "action"), _safe_text(payload.get("details") or ""))
    if snapshot is None:
        return None
    with _CACHE_LOCK:
        existing_worker = _ACTION_WORKERS.get(investigation_id)
        if existing_worker and existing_worker.is_alive():
            return snapshot
        worker = threading.Thread(target=_run_action_worker, args=(investigation_id, dict(payload)), daemon=True)
        _ACTION_WORKERS[investigation_id] = worker
        worker.start()
    return snapshot


class DataAnalystAPIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _write_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        if not raw.strip():
            return {}
        return json.loads(raw)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path.startswith("/charts/") or path.startswith("/backend/charts/"):
            chart_name = unquote(Path(path).name)
            chart_path = ROOT_DIR / "backend" / "charts" / chart_name
            if not chart_path.exists() or not chart_path.is_file():
                return self._write_json({"error": "Chart not found."}, status=HTTPStatus.NOT_FOUND)
            data = chart_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(chart_path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/api/health":
            return self._write_json({"ok": True, "time": _utc_now()})
        if path == "/api/bootstrap":
            return self._write_json(_load_bootstrap())
        if path == "/api/datasets":
            return self._write_json({"datasets": _catalog_datasets()})
        if path == "/api/investigations":
            return self._write_json({"investigations": _list_investigations()})
        match = re.match(r"^/?api/investigations/([^/]+)/workspace/?$", parsed.path.strip())
        if match:
            investigation_id = match.group(1)
            with _CACHE_LOCK:
                investigation = _INVESTIGATIONS.get(investigation_id)
            if not investigation:
                return self._write_json({"error": "Investigation not found."}, status=HTTPStatus.NOT_FOUND)
            return self._write_json({"investigation": _public_investigation(investigation)})
        match = re.match(r"^/?api/investigations/([^/]+)/?$", parsed.path.strip())
        if match:
            investigation_id = match.group(1)
            with _CACHE_LOCK:
                investigation = _INVESTIGATIONS.get(investigation_id)
            if not investigation:
                return self._write_json({"error": "Investigation not found."}, status=HTTPStatus.NOT_FOUND)
            return self._write_json({"investigation": _public_investigation(investigation)})

        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        match = re.match(r"^/?api/investigations/([^/]+)/cancel/?$", parsed.path.strip())
        if match:
            investigation_id = match.group(1)
            with _CACHE_LOCK:
                investigation = _INVESTIGATIONS.get(investigation_id)
                if not investigation:
                    return self._write_json({"error": "Investigation not found."}, status=HTTPStatus.NOT_FOUND)
                investigation = dict(investigation)
                investigation["status"] = "cancelled"
                investigation["updated_at"] = _utc_now()
                _INVESTIGATIONS[investigation_id] = investigation
            return self._write_json({"investigation": _public_investigation(investigation), "summary": _investigation_summary(investigation)})

        match = re.match(r"^/?api/investigations/([^/]+)/action/?$", parsed.path.strip())
        if match:
            investigation_id = match.group(1)
            payload = self._read_json()
            try:
                snapshot = _dispatch_action_continuation(investigation_id, payload)
                if snapshot is None:
                    return self._write_json({"error": "Investigation not found."}, status=HTTPStatus.NOT_FOUND)
                return self._write_json(
                    {
                        "accepted": True,
                        "processing": True,
                        "investigation": _public_investigation(snapshot),
                        "summary": _investigation_summary(snapshot),
                    },
                    status=HTTPStatus.ACCEPTED,
                )
            except KeyError:
                return self._write_json({"error": "Investigation not found."}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:
                return self._write_json(
                    {
                        "error": _safe_text(exc),
                        "message": "The backend could not continue the investigation.",
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )

        if path != "/api/investigations":
            return self._write_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

        try:
            payload = self._read_json()
            investigation = _start_investigation_job(payload)
            return self._write_json({"investigation": _public_investigation(investigation), "summary": _investigation_summary(investigation)})
        except Exception as exc:
            return self._write_json(
                {
                    "error": _safe_text(exc),
                    "message": "The backend could not complete the investigation.",
                },
                status=HTTPStatus.BAD_REQUEST,
            )


def main() -> None:
    server = ThreadingHTTPServer((DEFAULT_HOST, DEFAULT_PORT), DataAnalystAPIHandler)
    print(f"Data Analyst API listening on http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    print(f"Serving frontend from {FRONTEND_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
