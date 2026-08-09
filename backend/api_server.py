from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable
from urllib.parse import urlparse

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_DIR = ROOT_DIR / "backend" / "data"
DEFAULT_HOST = os.getenv("DAA_API_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("DAA_API_PORT", "8787"))

_INVESTIGATIONS: dict[str, dict[str, Any]] = {}
_INVESTIGATION_ORDER: list[str] = []
_CACHE_LOCK = threading.Lock()
_DATASET_CACHE: dict[str, dict[str, Any]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        "reports": {
            "analyst": bool(record.get("analyst_report")),
            "business": bool(record.get("business_report")),
            "executive": bool(record.get("executive_report")),
        },
    }


def _store_investigation(record: dict[str, Any]) -> dict[str, Any]:
    investigation_id = record["id"]
    with _CACHE_LOCK:
        _INVESTIGATIONS[investigation_id] = record
        if investigation_id in _INVESTIGATION_ORDER:
            _INVESTIGATION_ORDER.remove(investigation_id)
        _INVESTIGATION_ORDER.insert(0, investigation_id)
        _INVESTIGATION_ORDER[:] = _INVESTIGATION_ORDER[:50]
    return record


def _list_investigations() -> list[dict[str, Any]]:
    with _CACHE_LOCK:
        ordered = [_INVESTIGATIONS[item] for item in _INVESTIGATION_ORDER if item in _INVESTIGATIONS]
    return [_investigation_summary(item) for item in ordered]


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


def _run_autonomous(question: str, dataset_path: str, mode: str) -> dict[str, Any]:
    from backend.graph.analyst_graph import graph

    df = _read_dataframe(_resolve_dataset_path(dataset_path))
    state: dict[str, Any] = {
        "business_question": question,
        "dataset_path": dataset_path,
        "dataframe": df,
        "mode": mode,
        "enable_llm_reasoning": True,
        "disable_llm_reasoning": False,
        "disable_semantic_matcher": False,
        "analysis_evidence": {},
    }
    final_state = graph.invoke(state)
    return _jsonable(final_state)


def _run_guided(question: str, dataset_path: str, responses: list[str] | None) -> dict[str, Any]:
    from backend.scripts.guided_mode_harness import default_guided_responses, run_guided_workflow

    df = _read_dataframe(_resolve_dataset_path(dataset_path))
    guided_responses = list(responses or default_guided_responses())
    if len(guided_responses) < 4:
        guided_responses.extend(["continue"] * (4 - len(guided_responses)))
    result = run_guided_workflow(
        question=question,
        responses=guided_responses,
        dataset_path=dataset_path,
        dataframe=df,
    )
    return _jsonable(result.final_state)


def _run_collaborative(
    question: str,
    dataset_path: str,
    responses: list[str] | None,
    initial_tasks: list[Any] | None,
) -> dict[str, Any]:
    from backend.collaborative_mode import run_interactive_collaborative_session
    from backend.scripts.collaborative_mode_harness import demo_collaborative_responses

    df = _read_dataframe(_resolve_dataset_path(dataset_path))
    collaborative_responses = list(responses or [])
    if not collaborative_responses:
        collaborative_responses = demo_collaborative_responses()
    if len(collaborative_responses) < 8:
        collaborative_responses.extend(demo_collaborative_responses())
    result = run_interactive_collaborative_session(
        question=question,
        dataset_path=dataset_path,
        dataframe=df,
        responses=collaborative_responses,
        initial_tasks=initial_tasks,
        build_final_report=True,
    )
    return _jsonable(result.final_state)


def _run_investigation(payload: dict[str, Any]) -> dict[str, Any]:
    question = _safe_text(payload.get("question") or payload.get("business_question"))
    if not question:
        raise ValueError("A business question is required.")

    mode = _safe_text(payload.get("mode") or "autonomous").lower()
    dataset_path = _safe_text(payload.get("datasetPath") or payload.get("dataset_path") or _default_dataset_path())
    if not dataset_path:
        raise ValueError("No dataset is available.")

    created_at = _utc_now()
    if mode == "guided":
        final_state = _run_guided(question, dataset_path, payload.get("guidedResponses") or payload.get("responses"))
    elif mode == "collaborative":
        final_state = _run_collaborative(
            question,
            dataset_path,
            payload.get("collaborativeResponses") or payload.get("responses"),
            payload.get("initialTasks") or payload.get("initial_tasks"),
        )
    else:
        mode = "autonomous"
        final_state = _run_autonomous(question, dataset_path, mode)

    dataset_meta = _dataset_metadata(_resolve_dataset_path(dataset_path))
    record_id = _safe_text(final_state.get("id") or final_state.get("investigation_id") or f"inv-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}")
    final_state.update(
        {
            "id": record_id,
            "question": question,
            "mode": mode,
            "dataset_path": dataset_path,
            "dataset": dataset_meta,
            "created_at": created_at,
            "updated_at": _utc_now(),
            "status": "completed" if final_state.get("final_report") else ("awaiting_user" if final_state.get("awaiting_user") else "running"),
        }
    )
    _store_investigation(final_state)
    return final_state


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

        if path == "/api/health":
            return self._write_json({"ok": True, "time": _utc_now()})
        if path == "/api/bootstrap":
            return self._write_json(_load_bootstrap())
        if path == "/api/datasets":
            return self._write_json({"datasets": _catalog_datasets()})
        if path == "/api/investigations":
            return self._write_json({"investigations": _list_investigations()})
        if path.startswith("/api/investigations/"):
            investigation_id = path.split("/", 3)[-1]
            with _CACHE_LOCK:
                investigation = _INVESTIGATIONS.get(investigation_id)
            if not investigation:
                return self._write_json({"error": "Investigation not found."}, status=HTTPStatus.NOT_FOUND)
            return self._write_json({"investigation": investigation})

        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path != "/api/investigations":
            return self._write_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

        try:
            payload = self._read_json()
            investigation = _run_investigation(payload)
            return self._write_json({"investigation": investigation, "summary": _investigation_summary(investigation)})
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
