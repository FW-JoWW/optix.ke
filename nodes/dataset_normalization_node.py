from __future__ import annotations

from normalization.pipeline import run_normalization_pipeline
from normalization.schema_engine import default_schema_registry
from state.state import AnalystState


def dataset_normalization_node(state: AnalystState) -> AnalystState:
    source = state.get("dataframe")
    if source is None:
        dataset_path = state.get("dataset_path")
        if dataset_path is None:
            raise ValueError("No dataframe or dataset_path available for normalization.")
        source = dataset_path

    schema = state.get("normalization_schema")
    schema_name = state.get("normalization_schema_name")
    if schema is None and schema_name is None:
        raise ValueError("Normalization schema is required.")
    if schema is None:
        schema = default_schema_registry.get(schema_name)

    result = run_normalization_pipeline(
        source=source,
        schema=schema,
        source_type=state.get("normalization_source_type"),
        llm_mapping_enabled=bool(
            state.get("enable_llm_reasoning", False)
            and not state.get("disable_llm_reasoning", False)
        ),
        synonym_dictionary=state.get("normalization_synonyms"),
    )

    state["normalization_output"] = result
    return state
