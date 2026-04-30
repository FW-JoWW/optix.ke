from __future__ import annotations

from typing import Any, Dict, List


def validate_insight(
    stats_output: Dict[str, Any],
    semantic_output: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = metadata or {}
    relationship_type = semantic_output.get("relationship_type", "unknown")
    warnings: List[str] = []
    restrictions: List[str] = []
    valid = True
    reason = "Relationship passes semantic screening."
    severity = "low"

    coefficient = None
    stats_payload = stats_output.get("stats", {}) or {}
    coefficient = stats_payload.get("coefficient")
    if coefficient is None:
        coefficient = ((stats_output.get("effect_size") or {}).get("value"))
    abs_corr = abs(float(coefficient)) if coefficient is not None else 0.0

    causal_grade = ((stats_output.get("causal_evidence") or {}).get("grade")) or "LOW"
    confounders = stats_output.get("confounders", []) or []
    confounder_count = len(confounders)
    partial = (stats_output.get("partial_correlation") or {}) if isinstance(stats_output, dict) else {}
    controls_applied = bool(partial.get("computed"))
    missing_ratio = float(metadata.get("missing_ratio", 0.0) or 0.0)

    if relationship_type in {"unit_conversion", "duplicate_feature"}:
        valid = False
        reason = "Derived relationship, not actionable."
        severity = "high" if relationship_type == "duplicate_feature" else "medium"
        restrictions.append("no_business_action")

    if abs_corr > 0.8 and causal_grade == "LOW":
        warnings.append("Strong association without causal support; downstream recommendations must stay exploratory.")
        restrictions.append("restrict_to_experiments")
        severity = "medium" if severity == "low" else severity

    if confounder_count > 3 and not controls_applied:
        warnings.append("Multiple plausible confounders were detected without any control adjustment.")
        reason = "Observed relationship is vulnerable to confounding."
        severity = "high"

    if missing_ratio > 0.2:
        warnings.append("Missing data exceeds 20% for the relationship subset.")
        if severity == "low":
            severity = "medium"

    return {
        "valid": valid,
        "reason": reason,
        "severity": severity,
        "warnings": warnings,
        "restriction_flags": sorted(set(restrictions)),
        "confounders_count": confounder_count,
        "controls_applied": controls_applied,
        "missing_ratio": round(missing_ratio, 4),
    }
