from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import pandas as pd

from normalization.pipeline import run_normalization_pipeline
from normalization.schema_engine import DatasetSchema, FieldConstraints, SchemaField


DATASET_PATH = Path("data/Car Dataset 1945-2020.csv")
OUTPUT_DIR = Path("data/normalization_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_name(name: str) -> str:
    value = name.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def infer_field_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"

    sample = series.dropna().astype(str).head(500)
    if sample.empty:
        return "string"

    numeric = pd.to_numeric(sample, errors="coerce")
    if numeric.notna().mean() >= 0.95:
        if (numeric.dropna() % 1 == 0).mean() >= 0.95:
            return "int"
        return "float"

    datetime_signal = sample.str.contains(r"[-/:TtZz]", regex=True).mean()
    if datetime_signal >= 0.5:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            datetime_like = pd.to_datetime(sample, errors="coerce", utc=True)
        if datetime_like.notna().mean() >= 0.95:
            return "datetime"

    normalized_values = {value.strip().lower() for value in sample}
    bool_like = {"true", "false", "yes", "no", "y", "n", "1", "0", "t", "f"}
    if normalized_values and normalized_values.issubset(bool_like):
        return "bool"

    return "string"


def build_schema_from_dataframe(df: pd.DataFrame) -> DatasetSchema:
    fields = []
    for column in df.columns:
        normalized = normalize_name(column)
        field_type = infer_field_type(df[column])
        constraints = FieldConstraints(nullable=True)
        if normalized == "id_trim":
            constraints.unique = True
            constraints.nullable = False
        required = normalized == "id_trim"
        fields.append(
            SchemaField(
                name=normalized,
                field_type=field_type,  # type: ignore[arg-type]
                required=required,
                synonyms=[column],
                constraints=constraints,
            )
        )
    return DatasetSchema(name="car_dataset_1945_2020", fields=fields)


def main() -> None:
    preview_df = pd.read_csv(DATASET_PATH, low_memory=False)
    schema = build_schema_from_dataframe(preview_df)

    result = run_normalization_pipeline(
        source=str(DATASET_PATH),
        source_type="csv",
        schema=schema,
        synonym_dictionary={field.name: field.synonyms for field in schema.fields},
        low_memory=False,
    )

    clean_df = pd.DataFrame(result["clean_data"])
    invalid_df = pd.DataFrame(result["invalid_data"])

    clean_path = OUTPUT_DIR / "car_dataset_1945_2020_clean.csv"
    invalid_path = OUTPUT_DIR / "car_dataset_1945_2020_invalid.csv"
    report_path = OUTPUT_DIR / "car_dataset_1945_2020_reports.json"

    clean_df.to_csv(clean_path, index=False)
    invalid_df.to_csv(invalid_path, index=False)
    report_payload = {
        "schema": result["schema"],
        "mapping_report": result["mapping_report"],
        "validation_report": result["validation_report"],
        "pre_analysis_report": result["pre_analysis_report"],
        "output_files": {
            "clean_data": str(clean_path),
            "invalid_data": str(invalid_path),
            "reports": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print("NORMALIZATION COMPLETE")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Rows in clean dataset: {len(clean_df)}")
    print(f"Rows in invalid dataset: {len(invalid_df)}")
    print(f"Clean dataset saved to: {clean_path}")
    print(f"Invalid dataset saved to: {invalid_path}")
    print(f"Reports saved to: {report_path}")

    print("\nMAPPING REPORT SUMMARY")
    print(
        {
            "mapped_fields": len(result["mapping_report"]["field_to_source"]),
            "unmapped_schema_fields": len(result["mapping_report"]["unmapped_schema_fields"]),
            "unmapped_source_columns": len(result["mapping_report"]["unmapped_source_columns"]),
        }
    )
    print("\nVALIDATION REPORT SUMMARY")
    print(result["validation_report"])
    print("\nPRE-ANALYSIS REPORT")
    print(result["pre_analysis_report"])


if __name__ == "__main__":
    main()
