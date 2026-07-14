from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def _confidence_bucket(score: int) -> str:
    if score >= 80:
        return "high"
    if score >= 55:
        return "medium"
    return "low"


def _summary_to_lines(values: Any, limit: int = 4) -> List[str]:
    if not values:
        return []
    if isinstance(values, str):
        return [values]
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    lines: List[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            lines.append(text)
        if len(lines) >= limit:
            break
    return lines


def _current_dataframe(state: Dict[str, Any]) -> pd.DataFrame | None:
    df = state.get("cleaned_data")
    if df is None:
        df = state.get("analysis_dataset")
    if df is None:
        df = state.get("dataframe")
    return df


def _confidence_from_validation(validation: Dict[str, Any] | None, base: int = 85) -> Dict[str, Any]:
    validation = validation or {}
    score = float(base)
    factors: List[str] = []
    reducing: List[str] = []

    if validation.get("schema_stable") is True:
        factors.append("Schema stayed stable")
    elif validation.get("schema_stable") is False:
        reducing.append("Schema changed during validation")
        score -= 12

    row_loss_ratio = float(validation.get("row_loss_ratio", 0.0) or 0.0)
    if row_loss_ratio == 0:
        factors.append("No row loss")
    elif row_loss_ratio > 0.1:
        reducing.append("Row loss was materially high")
        score -= 15
    elif row_loss_ratio > 0:
        reducing.append("Some row loss occurred")
        score -= 8

    anomalies = validation.get("anomalies") or []
    if anomalies:
        reducing.append(f"Validation anomalies: {', '.join(map(str, anomalies[:3]))}")
        score -= min(20, len(anomalies) * 5)

    score = max(0, min(100, int(round(score))))
    return {
        "score": score,
        "level": _confidence_bucket(score),
        "factors": factors or ["Confidence is based on the current deterministic evidence."],
        "reducing_factors": reducing,
    }


def build_data_preparation_object(state: Dict[str, Any]) -> Dict[str, Any]:
    evidence = state.setdefault("analysis_evidence", {})
    profile = state.get("dataset_profile") or evidence.get("dataset_profile_json") or {}
    validation = state.get("cleaning_validation") or state.get("data_validation") or {}
    issues = list((state.get("data_quality_issues") or {}).get("issues", []))
    cleaning_plan = list(state.get("cleaning_plan", []))
    top_issue = issues[0] if issues else {}
    recommended = "Continue"
    alternatives = [
        {"name": "Modify", "reason": "Lets the analyst request a different cleaning treatment."},
        {"name": "Cancel", "reason": "Stops before the cleaned dataset is reused downstream."},
    ]
    if top_issue:
        recommended = str(top_issue.get("recommended_action", "Continue"))
        alternatives.insert(0, {
            "name": str(top_issue.get("suggested_action", recommended)),
            "reason": str(top_issue.get("explanation", "This is the current deterministic recommendation.")),
        })

    return {
        "decision_id": f"data_preparation:v{state.get('guided_checkpoint_versions', {}).get('data_preparation', 1)}",
        "decision_type": "cleaning",
        "stage": "data_preparation",
        "title": "Cleaning checkpoint",
        "recommendation": recommended,
        "evidence": [
            f"Rows: {profile.get('row_count', 'unknown')}",
            f"Columns: {profile.get('column_count', 'unknown')}",
            f"Missing values: {int(sum(float(info.get('missing_ratio', 0.0)) * profile.get('row_count', 0) for info in (profile.get('columns') or {}).values())) if profile.get('columns') else 'unknown'}",
            f"Top issue: {top_issue.get('issue_type', 'none')}",
        ],
        "reason_codes": [
            str(top_issue.get("issue_type", "no_major_issue")),
            f"severity:{top_issue.get('severity', 'unknown')}",
            "profile_based",
        ],
        "confidence": _confidence_from_validation(validation, base=95 if not issues else 88),
        "alternatives": alternatives[:3],
        "assumptions": [
            "The deterministic cleaning engine already produced the candidate plan.",
            "The explanation should not invent new cleaning actions.",
        ],
        "impact": [
            "A modification may require cleaning recomputation and downstream revalidation.",
            "Continue keeps the current cleaned dataset version.",
        ],
        "dependencies": [
            "dataset_profile",
            "data_quality_issues",
            "cleaning_plan",
            "cleaning_validation",
        ],
        "metadata": {
            "issue_count": len(issues),
            "cleaning_step_count": len(cleaning_plan),
            "validation": validation,
        },
    }


def build_business_understanding_object(state: Dict[str, Any]) -> Dict[str, Any]:
    intent = state.get("intent") or {}
    selected = list(state.get("selected_columns") or intent.get("selected_columns") or [])
    df = _current_dataframe(state)
    all_columns = list(df.columns) if df is not None else []
    rejected = [col for col in all_columns if col not in selected]
    confidence = intent.get("confidence")
    confidence_score = int(round(float(confidence) * 100)) if isinstance(confidence, (int, float)) else 75
    if confidence_score > 100:
        confidence_score = 100
    if confidence_score < 0:
        confidence_score = 0

    primary = selected[:3]
    supporting = selected[3:8]
    recommendation = "Continue"
    alternatives = [
        {"name": "Modify", "reason": "Refines the variable scope before analysis continues."},
        {"name": "Cancel", "reason": "Stops if the current variable set is not the right framing."},
    ]

    return {
        "decision_id": f"business_understanding:v{state.get('guided_checkpoint_versions', {}).get('business_understanding', 1)}",
        "decision_type": "variable_selection",
        "stage": "business_understanding",
        "title": "Variable selection checkpoint",
        "recommendation": recommendation,
        "evidence": [
            f"Primary variables: {', '.join(primary) if primary else 'none'}",
            f"Supporting variables: {', '.join(supporting) if supporting else 'none'}",
            f"Rejected variables: {', '.join(rejected) if rejected else 'none'}",
            f"Analytical intent: {intent.get('analytic_intent') or intent.get('type') or 'unknown'}",
        ],
        "reason_codes": [
            str(intent.get("analytic_intent") or intent.get("type") or "unknown"),
            "intent_alignment",
            "column_registry",
        ],
        "confidence": {
            "score": confidence_score,
            "level": _confidence_bucket(confidence_score),
            "factors": [
                "The current variable set matches the resolved intent.",
                "Column roles and semantic hints were available.",
            ],
            "reducing_factors": [] if selected else ["No variables were selected."],
        },
        "alternatives": alternatives,
        "assumptions": [
            "The current selection is based on the resolved analytical intent.",
            "Excluding columns does not remove a needed business signal.",
        ],
        "impact": [
            "Changing the selection will alter the planner, statistical tests, charts, and report.",
            "Continue preserves the current evidence path.",
        ],
        "dependencies": [
            "intent",
            "column_registry",
            "selected_columns",
        ],
        "metadata": {
            "selected_count": len(selected),
            "rejected_count": len(rejected),
        },
    }


def build_analysis_strategy_object(state: Dict[str, Any]) -> Dict[str, Any]:
    evidence = state.setdefault("analysis_evidence", {})
    plan = evidence.get("analysis_plan") or state.get("analysis_plan") or []
    computation = evidence.get("computation_plan") or {}
    intent = state.get("intent") or {}
    confidence = computation.get("confidence_score")
    if not isinstance(confidence, (int, float)):
        confidence = intent.get("confidence") if isinstance(intent.get("confidence"), (int, float)) else 0.74
    confidence_score = int(round(float(confidence) * 100)) if float(confidence) <= 1 else int(round(float(confidence)))
    if confidence_score > 100:
        confidence_score = 100
    if confidence_score < 0:
        confidence_score = 0

    steps = [
        f"{item.get('tool')} on {', '.join(item.get('columns', [])) or item.get('column', 'n/a')}"
        for item in plan[:5]
        if isinstance(item, dict)
    ]
    assumptions = [
        str(item.get("justification", "No justification provided."))
        for item in computation.get("steps", [])[:5]
        if isinstance(item, dict)
    ]
    if not assumptions:
        assumptions = ["No explicit assumption checks were available."]

    return {
        "decision_id": f"analysis_strategy:v{state.get('guided_checkpoint_versions', {}).get('analysis_strategy', 1)}",
        "decision_type": "analysis_plan",
        "stage": "analysis_strategy",
        "title": "Analysis strategy checkpoint",
        "recommendation": "Continue",
        "evidence": steps or ["No analysis plan was generated."],
        "reason_codes": [
            str(intent.get("analytic_intent") or intent.get("type") or "unknown"),
            "planner_output",
            "assumption_checks",
        ],
        "confidence": {
            "score": confidence_score,
            "level": _confidence_bucket(confidence_score),
            "factors": [
                "The plan is aligned with the resolved analytical intent.",
                "The deterministic planner already validated the available signals.",
            ],
            "reducing_factors": [] if steps else ["No analysis steps were produced."],
        },
        "alternatives": [
            {"name": "Modify", "reason": "Changes the analysis method or target columns before execution."},
            {"name": "Cancel", "reason": "Stops before tool execution begins."},
        ],
        "assumptions": assumptions,
        "impact": [
            "Changing the strategy affects statistical tests, evidence synthesis, visualizations, and the report.",
            "Continue preserves the current plan and execution order.",
        ],
        "dependencies": [
            "analysis_plan",
            "computation_plan",
            "intent",
        ],
        "metadata": {
            "plan_size": len(plan),
            "computation_step_count": len(computation.get("steps", []) or []),
        },
    }


def build_result_review_object(state: Dict[str, Any]) -> Dict[str, Any]:
    evidence = state.setdefault("analysis_evidence", {})
    top_stories = evidence.get("top_stories", []) or []
    judgment = evidence.get("judgment_summary", {}) or {}
    charts = evidence.get("visualizations", []) or []
    primary = top_stories[0] if top_stories else {}
    confidence = primary.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = judgment.get("global_confidence")
    if isinstance(confidence, (int, float)) and confidence <= 1:
        confidence_score = int(round(float(confidence) * 100))
    elif isinstance(confidence, (int, float)):
        confidence_score = int(round(float(confidence)))
    else:
        confidence_score = 65

    return {
        "decision_id": f"result_review:v{state.get('guided_checkpoint_versions', {}).get('result_review', 1)}",
        "decision_type": "result_review",
        "stage": "result_review",
        "title": "Results review checkpoint",
        "recommendation": "Continue",
        "evidence": [
            primary.get("insight", "No primary story was identified."),
            judgment.get("dominant_reasoning", "No judgment was available."),
        ],
        "reason_codes": [
            str(primary.get("type", "story")),
            str(judgment.get("dominant_reasoning", "unknown")),
        ],
        "confidence": {
            "score": confidence_score,
            "level": _confidence_bucket(confidence_score),
            "factors": [
                "The story ranking already consolidated the strongest evidence.",
                "Visualization output was generated from the accepted story set.",
            ],
            "reducing_factors": [] if top_stories else ["No ranked stories were available."],
        },
        "alternatives": [
            {"name": "Modify", "reason": "Lets you change how charts or presentation are rendered."},
            {"name": "Cancel", "reason": "Stops before the report is finalized."},
        ],
        "assumptions": [
            "The top-ranked story reflects the current evidence balance.",
            "The visual output should follow the deterministic story ranking.",
        ],
        "impact": [
            "Changing the visual review may alter charts and final wording.",
            "Continue preserves the current reporting direction.",
        ],
        "dependencies": [
            "top_stories",
            "visualizations",
            "judgment_summary",
        ],
        "metadata": {
            "visualization_count": len(charts),
            "top_story_count": len(top_stories),
        },
    }


def build_final_reasoning_object(state: Dict[str, Any]) -> Dict[str, Any]:
    evidence = state.setdefault("analysis_evidence", {})
    judgment = evidence.get("judgment_summary", {}) or {}
    top_stories = evidence.get("top_stories", []) or []
    primary = top_stories[0] if top_stories else {}
    confidence = judgment.get("global_confidence")
    if not isinstance(confidence, (int, float)):
        confidence = primary.get("confidence", 60)
    confidence_score = int(round(float(confidence))) if float(confidence) > 1 else int(round(float(confidence) * 100))
    return {
        "decision_id": "final_report",
        "decision_type": "reporting",
        "stage": "report",
        "title": "Final report reasoning",
        "recommendation": judgment.get("recommended_first_action") or "Continue",
        "evidence": [
            judgment.get("dominant_reasoning", "No dominant reasoning was identified."),
            primary.get("insight", "No primary story was available."),
        ],
        "reason_codes": [
            str(judgment.get("dominant_reasoning", "unknown")),
            str(judgment.get("actionability", "unknown")),
        ],
        "confidence": {
            "score": max(0, min(100, confidence_score)),
            "level": _confidence_bucket(max(0, min(100, confidence_score))),
            "factors": [
                "The final report inherits the deterministic judgment summary.",
                "The strongest story evidence was already ranked upstream.",
            ],
            "reducing_factors": [] if top_stories else ["No top story was available."],
        },
        "alternatives": [
            {"name": "Investigate further", "reason": "Use when the evidence is informative but not yet decision-ready."},
            {"name": "Collect more data", "reason": "Use when the current sample or quality is insufficient."},
        ],
        "assumptions": [
            "The deterministic pipeline already computed the current final recommendation.",
            "No additional analytical facts should be invented in the explanation layer.",
        ],
        "impact": [
            "The report summarizes the accepted deterministic path.",
            "Changing it would require rerunning the upstream pipeline stages.",
        ],
        "dependencies": [
            "judgment_summary",
            "top_stories",
            "decision_priority_ranking",
        ],
        "metadata": {
            "business_question": state.get("business_question"),
        },
    }


def build_reasoning_objects(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    objects = [
        build_data_preparation_object(state),
        build_business_understanding_object(state),
        build_analysis_strategy_object(state),
        build_result_review_object(state),
        build_final_reasoning_object(state),
    ]
    return objects
