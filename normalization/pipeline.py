from __future__ import annotations

from typing import Any, Dict, Optional

from normalization.column_mapping_engine import build_mapping_report
from normalization.ingestion_layer import ingest_source
from normalization.output_layer import build_output_payload
from normalization.pre_analysis_gate import run_pre_analysis_gate
from normalization.schema_engine import DatasetSchema, default_schema_registry
from normalization.standardization_engine import standardize_dataframe
from normalization.validation_engine import validate_dataframe


class NormalizationPipeline:
    def __init__(
        self,
        llm_mapping_enabled: bool = False,
        synonym_dictionary: Optional[Dict[str, list[str]]] = None,
    ) -> None:
        self.llm_mapping_enabled = llm_mapping_enabled
        self.synonym_dictionary = synonym_dictionary or {}

    def run(
        self,
        source: Any,
        schema: DatasetSchema | str,
        source_type: Optional[str] = None,
        **ingestion_kwargs: Any,
    ) -> Dict[str, Any]:
        resolved_schema = default_schema_registry.get(schema) if isinstance(schema, str) else schema
        raw_df, _ = ingest_source(source, source_type=source_type, **ingestion_kwargs)
        mapping_report = build_mapping_report(
            raw_df=raw_df,
            schema=resolved_schema,
            synonym_dictionary=self.synonym_dictionary,
            llm_enabled=self.llm_mapping_enabled,
        )
        standardized_df, _ = standardize_dataframe(raw_df, resolved_schema, mapping_report.field_to_source)
        clean_df, invalid_df, validation_report = validate_dataframe(standardized_df, resolved_schema)
        pre_analysis_report = run_pre_analysis_gate(clean_df, resolved_schema)
        return build_output_payload(
            clean_data=clean_df,
            invalid_data=invalid_df,
            schema=resolved_schema,
            mapping_report=mapping_report,
            validation_report=validation_report,
            pre_analysis_report=pre_analysis_report,
        )


def run_normalization_pipeline(
    source: Any,
    schema: DatasetSchema | str,
    source_type: Optional[str] = None,
    llm_mapping_enabled: bool = False,
    synonym_dictionary: Optional[Dict[str, list[str]]] = None,
    **ingestion_kwargs: Any,
) -> Dict[str, Any]:
    pipeline = NormalizationPipeline(
        llm_mapping_enabled=llm_mapping_enabled,
        synonym_dictionary=synonym_dictionary,
    )
    return pipeline.run(source=source, schema=schema, source_type=source_type, **ingestion_kwargs)
