from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from normalization.schema_engine import DatasetSchema, SchemaField
from utils.openai_runtime import get_openai_client


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


class MappingSuggestion(BaseModel):
    schema_field: str
    source_column: str
    confidence: float = Field(ge=0.0, le=1.0)
    method: str


class MappingReport(BaseModel):
    field_to_source: Dict[str, str] = Field(default_factory=dict)
    source_to_field: Dict[str, str] = Field(default_factory=dict)
    unmapped_schema_fields: List[str] = Field(default_factory=list)
    unmapped_source_columns: List[str] = Field(default_factory=list)
    suggestions: List[MappingSuggestion] = Field(default_factory=list)


def _deterministic_candidates(field: SchemaField, source_columns: List[str], synonyms: Dict[str, List[str]]) -> List[MappingSuggestion]:
    candidates: List[MappingSuggestion] = []
    normalized_source = {_normalize(col): col for col in source_columns}
    field_names = [field.name] + field.synonyms + synonyms.get(field.name, [])

    for alias in field_names:
        normalized_alias = _normalize(alias)
        if normalized_alias in normalized_source:
            candidates.append(
                MappingSuggestion(
                    schema_field=field.name,
                    source_column=normalized_source[normalized_alias],
                    confidence=0.99,
                    method="deterministic_exact",
                )
            )

    if candidates:
        return candidates

    for col in source_columns:
        similarity = max(
            SequenceMatcher(None, _normalize(alias), _normalize(col)).ratio()
            for alias in field_names
        )
        if similarity >= 0.84:
            candidates.append(
                MappingSuggestion(
                    schema_field=field.name,
                    source_column=col,
                    confidence=round(float(similarity), 4),
                    method="deterministic_fuzzy",
                )
            )
    candidates.sort(key=lambda item: item.confidence, reverse=True)
    return candidates


def _extract_json_object(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _llm_mapping_suggestions(
    source_columns: List[str],
    unmapped_fields: List[str],
    llm_enabled: bool,
) -> List[MappingSuggestion]:
    if not llm_enabled or not unmapped_fields:
        return []

    client = get_openai_client()
    if client is None:
        return []

    prompt = f"""
You are a constrained column-mapping assistant.
Map raw source columns to schema fields using only the exact candidates provided.
Do not invent fields or columns. If uncertain, return no mapping for that field.

Schema fields to resolve:
{json.dumps(unmapped_fields, ensure_ascii=True)}

Available source columns:
{json.dumps(source_columns, ensure_ascii=True)}

Return JSON only in this shape:
{{
  "suggestions": [
    {{
      "schema_field": "exact schema field",
      "source_column": "exact source column",
      "confidence": 0.0
    }}
  ]
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception:
        return []

    parsed = _extract_json_object(response.choices[0].message.content or "")
    suggestions: List[MappingSuggestion] = []
    for item in parsed.get("suggestions", []):
        if not isinstance(item, dict):
            continue
        schema_field = item.get("schema_field")
        source_column = item.get("source_column")
        confidence = item.get("confidence")
        if schema_field not in unmapped_fields or source_column not in source_columns:
            continue
        if not isinstance(confidence, (int, float)):
            continue
        suggestions.append(
            MappingSuggestion(
                schema_field=schema_field,
                source_column=source_column,
                confidence=round(float(confidence), 4),
                method="llm_candidate",
            )
        )
    suggestions.sort(key=lambda item: item.confidence, reverse=True)
    return suggestions


def build_mapping_report(
    raw_df: pd.DataFrame,
    schema: DatasetSchema,
    synonym_dictionary: Optional[Dict[str, List[str]]] = None,
    llm_enabled: bool = False,
) -> MappingReport:
    synonym_dictionary = synonym_dictionary or {}
    source_columns = list(raw_df.columns)
    field_to_source: Dict[str, str] = {}
    source_to_field: Dict[str, str] = {}
    suggestions: List[MappingSuggestion] = []

    for field in schema.fields:
        candidates = _deterministic_candidates(field, source_columns, synonym_dictionary)
        if not candidates:
            continue
        candidate = candidates[0]
        if candidate.source_column in source_to_field:
            continue
        field_to_source[field.name] = candidate.source_column
        source_to_field[candidate.source_column] = field.name
        suggestions.append(candidate)

    unmapped_fields = [field.name for field in schema.fields if field.name not in field_to_source]
    llm_suggestions = _llm_mapping_suggestions(source_columns, unmapped_fields, llm_enabled)
    for candidate in llm_suggestions:
        if candidate.schema_field in field_to_source or candidate.source_column in source_to_field:
            continue
        field_to_source[candidate.schema_field] = candidate.source_column
        source_to_field[candidate.source_column] = candidate.schema_field
        suggestions.append(candidate)

    return MappingReport(
        field_to_source=field_to_source,
        source_to_field=source_to_field,
        unmapped_schema_fields=[field.name for field in schema.fields if field.name not in field_to_source],
        unmapped_source_columns=[col for col in source_columns if col not in source_to_field],
        suggestions=suggestions,
    )
