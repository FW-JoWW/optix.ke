from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Sequence

import pandas as pd
from pydantic import BaseModel, Field


SourceType = Literal["csv", "json", "excel", "sql", "records", "dataframe"]


class IngestionResult(BaseModel):
    source_type: SourceType
    row_count: int
    column_names: list[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


def _ensure_dataframe(payload: Any) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        return pd.DataFrame(payload)
    raise TypeError("Unsupported in-memory payload type for ingestion.")


def ingest_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, IngestionResult]:
    raw = df.copy()
    return raw, IngestionResult(
        source_type="dataframe",
        row_count=int(len(raw)),
        column_names=list(raw.columns),
    )


def ingest_csv(path: str | Path, **read_kwargs: Any) -> tuple[pd.DataFrame, IngestionResult]:
    raw = pd.read_csv(path, **read_kwargs)
    return raw, IngestionResult(
        source_type="csv",
        row_count=int(len(raw)),
        column_names=list(raw.columns),
        metadata={"path": str(path)},
    )


def ingest_json(path_or_payload: str | Path | dict | list, **read_kwargs: Any) -> tuple[pd.DataFrame, IngestionResult]:
    if isinstance(path_or_payload, (str, Path)):
        raw = pd.read_json(path_or_payload, **read_kwargs)
        metadata = {"path": str(path_or_payload)}
    else:
        raw = _ensure_dataframe(path_or_payload)
        metadata = {"input_type": type(path_or_payload).__name__}
    return raw, IngestionResult(
        source_type="json",
        row_count=int(len(raw)),
        column_names=list(raw.columns),
        metadata=metadata,
    )


def ingest_excel(path: str | Path, **read_kwargs: Any) -> tuple[pd.DataFrame, IngestionResult]:
    raw = pd.read_excel(path, **read_kwargs)
    return raw, IngestionResult(
        source_type="excel",
        row_count=int(len(raw)),
        column_names=list(raw.columns),
        metadata={"path": str(path)},
    )


def ingest_sql_result(rows: Sequence[dict[str, Any]] | pd.DataFrame) -> tuple[pd.DataFrame, IngestionResult]:
    raw = _ensure_dataframe(rows)
    return raw, IngestionResult(
        source_type="sql",
        row_count=int(len(raw)),
        column_names=list(raw.columns),
        metadata={"input_type": type(rows).__name__},
    )


def ingest_records(records: Iterable[dict[str, Any]] | dict[str, list[Any]]) -> tuple[pd.DataFrame, IngestionResult]:
    raw = _ensure_dataframe(records)
    return raw, IngestionResult(
        source_type="records",
        row_count=int(len(raw)),
        column_names=list(raw.columns),
        metadata={"input_type": type(records).__name__},
    )


def ingest_source(source: Any, source_type: SourceType | None = None, **kwargs: Any) -> tuple[pd.DataFrame, IngestionResult]:
    if isinstance(source, pd.DataFrame):
        return ingest_dataframe(source)

    if source_type is None and isinstance(source, (str, Path)):
        suffix = Path(source).suffix.lower()
        if suffix == ".csv":
            source_type = "csv"
        elif suffix in {".json"}:
            source_type = "json"
        elif suffix in {".xls", ".xlsx"}:
            source_type = "excel"

    if source_type == "csv":
        return ingest_csv(source, **kwargs)
    if source_type == "json":
        return ingest_json(source, **kwargs)
    if source_type == "excel":
        return ingest_excel(source, **kwargs)
    if source_type == "sql":
        return ingest_sql_result(source)
    if source_type == "records":
        return ingest_records(source)

    raise ValueError("Could not determine a supported ingestion source type.")
