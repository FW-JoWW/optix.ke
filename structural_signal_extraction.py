from __future__ import annotations

from typing import Any, Dict, List


def extract_structural_signals(dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
    patterns = dataset_profile.get("pattern_detection", {})
    columns = dataset_profile.get("columns", {})
    repeated_blocks = patterns.get("repeated_row_blocks", [])
    similarity = patterns.get("column_similarity", [])
    sparsity = patterns.get("sparsity_patterns", {})

    high_missing_columns = [
        col for col, info in columns.items()
        if info.get("missing_ratio", 0.0) >= 0.35
    ]
    mixed_type_columns = [
        col for col, info in columns.items()
        if 0.2 <= info.get("numeric_like_ratio", 0.0) <= 0.8
        or 0.2 <= info.get("datetime_like_ratio", 0.0) <= 0.8
    ]

    signals: List[str] = []
    if repeated_blocks:
        signals.append("repeated_row_blocks")
    if similarity:
        signals.append("similar_column_names")
    if len(sparsity.get("row_pattern_examples", [])) > 3:
        signals.append("inconsistent_sparsity_patterns")
    if high_missing_columns:
        signals.append("high_missing_columns")
    if mixed_type_columns:
        signals.append("mixed_type_columns")

    return {
        "signals": signals,
        "repeated_row_blocks": repeated_blocks,
        "similar_columns": similarity,
        "high_missing_columns": high_missing_columns,
        "mixed_type_columns": mixed_type_columns,
        "row_pattern_examples": sparsity.get("row_pattern_examples", []),
        "primary_structure_confidence": round(
            max(0.0, 1.0 - 0.12 * len(signals)),
            4,
        ),
    }
