from __future__ import annotations

from typing import List

import pandas as pd
from pydantic import BaseModel, Field

from normalization.schema_engine import DatasetSchema


class PreAnalysisRisk(BaseModel):
    risk_type: str
    affected_fields: List[str] = Field(default_factory=list)
    severity: str
    recommendation: str


class PreAnalysisReport(BaseModel):
    warnings: List[PreAnalysisRisk] = Field(default_factory=list)


def _recommendation_for(risk_type: str) -> str:
    mapping = {
        "duplicates_affecting_aggregates": "Review duplicated rows before aggregate analysis.",
        "missing_time_continuity": "Check time-series completeness before trend or period-over-period analysis.",
        "inconsistent_category_grouping": "Standardize category labels before group-based comparisons.",
        "abnormal_distribution": "Inspect outliers and skew before relying on mean-based metrics.",
    }
    return mapping[risk_type]


def _severity(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def run_pre_analysis_gate(clean_df: pd.DataFrame, schema: DatasetSchema) -> PreAnalysisReport:
    warnings: List[PreAnalysisRisk] = []

    if clean_df.duplicated().any():
        ratio = float(clean_df.duplicated().mean())
        warnings.append(
            PreAnalysisRisk(
                risk_type="duplicates_affecting_aggregates",
                affected_fields=list(clean_df.columns),
                severity=_severity(min(1.0, ratio * 4)),
                recommendation=_recommendation_for("duplicates_affecting_aggregates"),
            )
        )

    datetime_fields = [field.name for field in schema.fields if field.field_type == "datetime" and field.name in clean_df.columns]
    for field_name in datetime_fields:
        series = pd.to_datetime(clean_df[field_name], errors="coerce", utc=True).dropna().sort_values()
        if len(series) < 3:
            continue
        diffs = series.diff().dropna()
        if diffs.empty:
            continue
        expected_step = diffs.mode().iloc[0]
        if pd.isna(expected_step) or expected_step <= pd.Timedelta(0):
            continue
        full_range = pd.date_range(series.min(), series.max(), freq=expected_step)
        if len(full_range) > len(series):
            missing_ratio = float((len(full_range) - len(series)) / max(len(full_range), 1))
            warnings.append(
                PreAnalysisRisk(
                    risk_type="missing_time_continuity",
                    affected_fields=[field_name],
                    severity=_severity(missing_ratio),
                    recommendation=_recommendation_for("missing_time_continuity"),
                )
            )

    string_fields = [field.name for field in schema.fields if field.field_type == "string" and field.name in clean_df.columns]
    for field_name in string_fields:
        raw = clean_df[field_name].dropna().astype(str)
        if raw.empty:
            continue
        normalized = raw.str.strip().str.lower()
        if normalized.nunique() < raw.nunique():
            ratio = float((raw.nunique() - normalized.nunique()) / max(raw.nunique(), 1))
            warnings.append(
                PreAnalysisRisk(
                    risk_type="inconsistent_category_grouping",
                    affected_fields=[field_name],
                    severity=_severity(ratio),
                    recommendation=_recommendation_for("inconsistent_category_grouping"),
                )
            )

    numeric_fields = [field.name for field in schema.fields if field.field_type in {"int", "float"} and field.name in clean_df.columns]
    for field_name in numeric_fields:
        series = pd.to_numeric(clean_df[field_name], errors="coerce").dropna()
        if len(series) < 8:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        outlier_ratio = float(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).mean())
        if outlier_ratio > 0:
            warnings.append(
                PreAnalysisRisk(
                    risk_type="abnormal_distribution",
                    affected_fields=[field_name],
                    severity=_severity(outlier_ratio * 2),
                    recommendation=_recommendation_for("abnormal_distribution"),
                )
            )

    return PreAnalysisReport(warnings=warnings)
