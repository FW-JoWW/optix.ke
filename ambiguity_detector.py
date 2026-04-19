from __future__ import annotations

from typing import Any, Dict, List


def detect_ambiguity(
    dataset_profile: Dict[str, Any],
    missing_ratio_threshold: float = 0.35,
    mixed_ratio_lower: float = 0.2,
    mixed_ratio_upper: float = 0.8,
) -> Dict[str, Any]:
    reasons: List[str] = []
    columns = dataset_profile.get("columns", {})
    patterns = dataset_profile.get("pattern_detection", {})

    for col, info in columns.items():
        if info.get("missing_ratio", 0.0) >= missing_ratio_threshold:
            reasons.append(f"{col}: high_missing_ratio")

        numeric_ratio = info.get("numeric_like_ratio", 0.0)
        datetime_ratio = info.get("datetime_like_ratio", 0.0)
        if mixed_ratio_lower <= numeric_ratio <= mixed_ratio_upper:
            reasons.append(f"{col}: mixed_numeric_types")
        if mixed_ratio_lower <= datetime_ratio <= mixed_ratio_upper:
            reasons.append(f"{col}: mixed_datetime_patterns")

    if patterns.get("repeated_row_blocks"):
        reasons.append("repeated_row_blocks_detected")

    if patterns.get("column_similarity"):
        reasons.append("similar_or_duplicate_column_names_detected")

    row_patterns = patterns.get("sparsity_patterns", {}).get("row_pattern_examples", [])
    if len(row_patterns) > 3:
        reasons.append("inconsistent_row_sparsity_patterns")

    if not columns:
        reasons.append("no_obvious_primary_structure")

    return {
        "requires_reasoning": bool(reasons),
        "reasons": reasons,
    }
