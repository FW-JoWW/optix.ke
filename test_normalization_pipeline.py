from normalization.pipeline import run_normalization_pipeline
from normalization.schema_engine import DatasetSchema, FieldConstraints, SchemaField


schema = DatasetSchema(
    name="generic_transactions",
    fields=[
        SchemaField(
            name="record_id",
            field_type="string",
            required=True,
            constraints=FieldConstraints(unique=True, nullable=False),
            synonyms=["id", "identifier"],
        ),
        SchemaField(
            name="event_time",
            field_type="datetime",
            required=True,
            constraints=FieldConstraints(nullable=False),
            synonyms=["timestamp", "time", "event date"],
        ),
        SchemaField(
            name="amount",
            field_type="float",
            required=True,
            constraints=FieldConstraints(min_value=0, nullable=False),
            synonyms=["total", "value"],
        ),
        SchemaField(
            name="status",
            field_type="string",
            required=False,
            constraints=FieldConstraints(category_map={"Complete": "complete", "Completed": "complete"}),
        ),
        SchemaField(
            name="is_active",
            field_type="bool",
            required=False,
            synonyms=["active"],
        ),
    ],
)

raw_records = [
    {"ID": "A-001", "Time": "2026-04-01 10:00:00+03:00", "Total": "14.50", "Status": "Complete", "Active": "Yes"},
    {"ID": "A-001", "Time": "2026-04-01 10:00:00+03:00", "Total": "14.50", "Status": "Completed", "Active": "Yes"},
    {"ID": "A-002", "Time": "2026-04-03 11:15:00", "Total": "-7", "Status": " pending ", "Active": "No"},
    {"ID": None, "Time": "bad-date", "Total": "oops", "Status": "Complete", "Active": "unknown"},
]

result = run_normalization_pipeline(
    source=raw_records,
    source_type="records",
    schema=schema,
    synonym_dictionary={"record_id": ["record id"], "event_time": ["date"]},
)

print("CLEAN DATA")
print(result["clean_data"])
print("\nINVALID DATA")
print(result["invalid_data"])
print("\nMAPPING REPORT")
print(result["mapping_report"])
print("\nVALIDATION REPORT")
print(result["validation_report"])
print("\nPRE-ANALYSIS REPORT")
print(result["pre_analysis_report"])
