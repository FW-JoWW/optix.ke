from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from normalization.column_mapping_engine import MappingReport
from normalization.pre_analysis_gate import PreAnalysisReport
from normalization.schema_engine import DatasetSchema
from normalization.validation_engine import ValidationReport


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    serializable = df.astype(object).where(pd.notna(df), None)
    return serializable.to_dict(orient="records")


def build_output_payload(
    clean_data: pd.DataFrame,
    invalid_data: pd.DataFrame,
    schema: DatasetSchema,
    mapping_report: MappingReport,
    validation_report: ValidationReport,
    pre_analysis_report: PreAnalysisReport,
) -> Dict[str, Any]:
    return {
        "clean_data": _df_to_records(clean_data),
        "invalid_data": _df_to_records(invalid_data),
        "schema": schema.as_contract(),
        "mapping_report": mapping_report.model_dump(),
        "validation_report": validation_report.model_dump(),
        "pre_analysis_report": pre_analysis_report.model_dump(),
    }
