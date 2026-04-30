from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel, Field

from normalization.schema_engine import DatasetSchema, SchemaField


class StandardizationReport(BaseModel):
    transformed_fields: Dict[str, List[str]] = Field(default_factory=dict)
    unmapped_source_columns: List[str] = Field(default_factory=list)


def _normalize_string(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_bool(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().lower()
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f"}
    if text in truthy:
        return True
    if text in falsy:
        return False
    return pd.NA


def _standardize_series(series: pd.Series, field: SchemaField) -> tuple[pd.Series, List[str]]:
    actions: List[str] = []

    if field.field_type == "string":
        standardized = series.map(_normalize_string)
        actions.extend(["strip_whitespace", "lowercase", "collapse_spaces"])
    elif field.field_type == "datetime":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            standardized = pd.to_datetime(series, errors="coerce", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        standardized = standardized.where(pd.notna(standardized), pd.NA)
        actions.extend(["parse_datetime", "normalize_iso_utc"])
    elif field.field_type == "float":
        standardized = pd.to_numeric(series, errors="coerce")
        actions.append("coerce_float")
    elif field.field_type == "int":
        standardized = pd.to_numeric(series, errors="coerce").round()
        standardized = standardized.astype("Int64")
        actions.extend(["coerce_int", "round_numeric"])
    elif field.field_type == "bool":
        standardized = series.map(_normalize_bool).astype("boolean")
        actions.append("normalize_boolean")
    else:
        standardized = series.copy()

    category_map = {
        _normalize_string(key): _normalize_string(value)
        for key, value in field.constraints.category_map.items()
    }
    if category_map and field.field_type == "string":
        standardized = standardized.map(lambda v: category_map.get(v, v) if pd.notna(v) else v)
        actions.append("normalize_categories")

    return standardized, actions


def standardize_dataframe(
    raw_df: pd.DataFrame,
    schema: DatasetSchema,
    field_to_source: Dict[str, str],
) -> tuple[pd.DataFrame, StandardizationReport]:
    standardized_columns: Dict[str, pd.Series] = {}
    transformed_fields: Dict[str, List[str]] = {}

    for field in schema.fields:
        source_column = field_to_source.get(field.name)
        if source_column is None or source_column not in raw_df.columns:
            standardized_columns[field.name] = pd.Series([pd.NA] * len(raw_df), dtype="object")
            transformed_fields[field.name] = ["unmapped_field"]
            continue
        standardized_series, actions = _standardize_series(raw_df[source_column], field)
        standardized_columns[field.name] = standardized_series
        transformed_fields[field.name] = actions

    standardized_df = pd.DataFrame(standardized_columns, index=raw_df.index)
    report = StandardizationReport(
        transformed_fields=transformed_fields,
        unmapped_source_columns=[col for col in raw_df.columns if col not in field_to_source.values()],
    )
    return standardized_df, report
