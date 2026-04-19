from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


FieldType = Literal["int", "float", "string", "datetime", "bool"]


class FieldConstraints(BaseModel):
    nullable: bool = True
    unique: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: List[str] = Field(default_factory=list)
    regex: Optional[str] = None
    category_map: Dict[str, str] = Field(default_factory=dict)


class SchemaField(BaseModel):
    name: str
    field_type: FieldType
    required: bool = False
    synonyms: List[str] = Field(default_factory=list)
    constraints: FieldConstraints = Field(default_factory=FieldConstraints)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Schema field names must be non-empty.")
        return value


class DatasetSchema(BaseModel):
    name: str
    version: str = "1.0"
    fields: List[SchemaField]

    @field_validator("fields")
    @classmethod
    def _validate_unique_fields(cls, value: List[SchemaField]) -> List[SchemaField]:
        seen = set()
        for field in value:
            lowered = field.name.lower()
            if lowered in seen:
                raise ValueError(f"Duplicate schema field detected: {field.name}")
            seen.add(lowered)
        return value

    def get_field(self, name: str) -> Optional[SchemaField]:
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def field_names(self) -> List[str]:
        return [field.name for field in self.fields]

    def as_contract(self) -> Dict[str, dict]:
        return {
            field.name: {
                "type": field.field_type,
                "required": field.required,
                "synonyms": field.synonyms,
                "constraints": field.constraints.model_dump(),
            }
            for field in self.fields
        }


class SchemaRegistry:
    def __init__(self) -> None:
        self._schemas: Dict[str, DatasetSchema] = {}

    def register(self, schema: DatasetSchema) -> DatasetSchema:
        self._schemas[schema.name] = schema
        return schema

    def get(self, name: str) -> DatasetSchema:
        if name not in self._schemas:
            raise KeyError(f"Schema '{name}' is not registered.")
        return self._schemas[name]

    def has(self, name: str) -> bool:
        return name in self._schemas

    def list(self) -> List[str]:
        return sorted(self._schemas.keys())


default_schema_registry = SchemaRegistry()
