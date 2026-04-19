from __future__ import annotations

import re
from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel, Field

from normalization.schema_engine import DatasetSchema, SchemaField


class RowValidationResult(BaseModel):
    row_index: int
    errors: List[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    total_rows: int
    valid_rows: int
    invalid_rows: int
    field_errors: Dict[str, int] = Field(default_factory=dict)
    row_errors: List[RowValidationResult] = Field(default_factory=list)


def _is_missing(value: Any) -> bool:
    return pd.isna(value) or value == ""


def _type_valid(field: SchemaField, value: Any) -> bool:
    if _is_missing(value):
        return True
    if field.field_type == "string":
        return isinstance(value, str)
    if field.field_type == "datetime":
        return isinstance(value, str) and value.endswith("Z")
    if field.field_type == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if field.field_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if field.field_type == "bool":
        return isinstance(value, bool)
    return False


def validate_dataframe(
    standardized_df: pd.DataFrame,
    schema: DatasetSchema,
) -> tuple[pd.DataFrame, pd.DataFrame, ValidationReport]:
    field_errors: Dict[str, int] = {}
    row_errors: List[RowValidationResult] = []
    uniqueness_cache: Dict[str, set[Any]] = {
        field.name: set()
        for field in schema.fields
        if field.constraints.unique
    }

    valid_mask: List[bool] = []

    for idx, row in standardized_df.iterrows():
        errors: List[str] = []

        for field in schema.fields:
            value = row.get(field.name, pd.NA)

            if field.required and _is_missing(value):
                errors.append(f"{field.name}:required")
                field_errors[field.name] = field_errors.get(field.name, 0) + 1
                continue

            if _is_missing(value):
                if not field.constraints.nullable and field.required:
                    errors.append(f"{field.name}:null_not_allowed")
                    field_errors[field.name] = field_errors.get(field.name, 0) + 1
                continue

            if not _type_valid(field, value):
                errors.append(f"{field.name}:type_invalid")
                field_errors[field.name] = field_errors.get(field.name, 0) + 1
                continue

            if field.field_type in {"int", "float"}:
                numeric_value = float(value)
                if field.constraints.min_value is not None and numeric_value < field.constraints.min_value:
                    errors.append(f"{field.name}:below_min")
                    field_errors[field.name] = field_errors.get(field.name, 0) + 1
                if field.constraints.max_value is not None and numeric_value > field.constraints.max_value:
                    errors.append(f"{field.name}:above_max")
                    field_errors[field.name] = field_errors.get(field.name, 0) + 1

            if field.constraints.allowed_values and str(value) not in field.constraints.allowed_values:
                errors.append(f"{field.name}:not_allowed")
                field_errors[field.name] = field_errors.get(field.name, 0) + 1

            if field.constraints.regex and not re.match(field.constraints.regex, str(value)):
                errors.append(f"{field.name}:regex_mismatch")
                field_errors[field.name] = field_errors.get(field.name, 0) + 1

            if field.constraints.unique:
                cache = uniqueness_cache[field.name]
                if value in cache:
                    errors.append(f"{field.name}:duplicate_unique")
                    field_errors[field.name] = field_errors.get(field.name, 0) + 1
                else:
                    cache.add(value)

        row_errors.append(RowValidationResult(row_index=int(idx), errors=errors))
        valid_mask.append(not errors)

    clean_df = standardized_df.loc[valid_mask].reset_index(drop=True)
    invalid_df = standardized_df.loc[[not item for item in valid_mask]].copy()
    invalid_df["__errors__"] = [item.errors for item in row_errors if item.errors]
    invalid_df = invalid_df.reset_index(drop=True)

    report = ValidationReport(
        total_rows=int(len(standardized_df)),
        valid_rows=int(sum(valid_mask)),
        invalid_rows=int(len(standardized_df) - sum(valid_mask)),
        field_errors=field_errors,
        row_errors=[item for item in row_errors if item.errors],
    )
    return clean_df, invalid_df, report
