from __future__ import annotations

from typing import Any, Dict, List


def grade_causal_evidence(
    stats_result: Dict[str, Any],
    temporal_signal: Dict[str, Any],
    confounders: List[Dict[str, Any]],
    segment_correlations: List[Dict[str, Any]],
    bias_result: Dict[str, Any],
    nonlinear_signal: Dict[str, Any],
    question: str,
) -> Dict[str, Any]:
    rubric = {
        "temporal_precedence": 0,
        "confounder_control": 0,
        "sample_size": 0,
        "consistency_across_segments": 0,
        "robustness": 0,
        "experimental_evidence": 0,
        "effect_stability": 0,
    }
    rationale: List[str] = []
    sample_size = int(stats_result.get("sample_size") or 0)
    p_value = stats_result.get("p_value")
    coefficient = abs(float(stats_result.get("coefficient") or 0.0))

    if temporal_signal.get("lag_direction") == "x_precedes_y":
        rubric["temporal_precedence"] = 20
        rationale.append("Observed temporal ordering is consistent with the proposed direction.")
    elif temporal_signal.get("lag_direction") == "same_time":
        rubric["temporal_precedence"] = 8
        rationale.append("Variables move together, but temporal precedence is not established.")
    elif temporal_signal.get("lag_direction") == "y_precedes_x":
        rubric["temporal_precedence"] = 0
        rationale.append("The apparent driver may occur after the outcome, so reverse causality is plausible.")

    if not confounders:
        rubric["confounder_control"] = 15
        rationale.append("No strong confounders were detected in the available candidates.")
    elif len(confounders) <= 2:
        rubric["confounder_control"] = 3
        rationale.append("A small number of plausible confounders remain, so causal interpretation is limited.")
    else:
        rationale.append("Several plausible confounders remain uncontrolled.")

    if sample_size >= 500:
        rubric["sample_size"] = 15
    elif sample_size >= 100:
        rubric["sample_size"] = 10
    elif sample_size >= 30:
        rubric["sample_size"] = 5
    if sample_size < 30:
        rationale.append("Sample size is too small for confident causal interpretation.")

    strong_segments = [item for item in segment_correlations if item.get("sample_size", 0) >= 20 and item.get("coefficient") is not None]
    if strong_segments:
        same_sign = len({1 if float(item["coefficient"]) > 0 else -1 for item in strong_segments if abs(float(item["coefficient"])) >= 0.1}) <= 1
        if same_sign:
            rubric["consistency_across_segments"] = 15
        else:
            rubric["consistency_across_segments"] = 4
            rationale.append("Relationship direction changes across segments.")
    else:
        rubric["consistency_across_segments"] = 5

    if p_value is not None and p_value < 0.05 and coefficient >= 0.2:
        rubric["robustness"] = 15
    elif p_value is not None and p_value < 0.1:
        rubric["robustness"] = 8
    else:
        rationale.append("The observed relationship is weak or statistically unstable.")

    rubric["experimental_evidence"] = 0
    rationale.append("No randomized or quasi-experimental evidence is present in the current workflow output.")

    if not bias_result.get("bias_risks") and not nonlinear_signal.get("warning") and not confounders:
        rubric["effect_stability"] = 20
    elif len(bias_result.get("bias_risks", [])) <= 1 and len(confounders) <= 1:
        rubric["effect_stability"] = 8
    else:
        rubric["effect_stability"] = 3
        rationale.append("Bias or instability warnings reduce confidence in causal interpretation.")

    total = int(sum(rubric.values()))
    if total >= 90:
        grade = "STRONG"
    elif total >= 50:
        grade = "MODERATE"
    else:
        grade = "LOW"

    if "cause" not in (question or "").lower() and grade == "STRONG":
        rationale.append("Evidence is strong for consistent influence, but causal language should still be matched to the user question.")

    return {"grade": grade, "score": total, "rubric": rubric, "rationale": rationale}
