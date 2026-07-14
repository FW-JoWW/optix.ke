from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Tuple

import pandas as pd

from cleaning_executor import execute_cleaning_actions
from core.reasoning_layer import explain_decision, format_reasoning_explanation, interpret_modification_request
from core.guided_versions import capture_guided_stage_snapshot, diff_guided_stage_snapshots, restore_guided_stage_snapshot
from core.reasoning_objects import (
    build_analysis_strategy_object,
    build_business_understanding_object,
    build_data_preparation_object,
    build_result_review_object,
)
from nodes.cleaning_audit_node import cleaning_audit_node
from nodes.data_validation_node import data_validation_node
from nodes.visualization_generator_node import visualization_generator_node
from state.state import AnalystState
from data_profiling import profile_dataset


@dataclass
class ModificationOutcome:
    applied: bool
    notes: List[str]
    unsupported: List[str]
    fallback_reason: str | None = None
    restored_version: int | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_guided_state(state: AnalystState) -> Dict[str, Any]:
    evidence = state.setdefault("analysis_evidence", {})
    state.setdefault("guided_decision_log", [])
    state.setdefault("guided_checkpoint_versions", {})
    state.setdefault("guided_version_snapshots", {})
    state.setdefault("guided_checkpoint_summaries", {})
    state.setdefault("guided_mode", {})
    state.setdefault("guided_visualization_preferences", {})
    evidence.setdefault("guided_decision_log", state["guided_decision_log"])
    evidence.setdefault("guided_version_snapshots", state["guided_version_snapshots"])
    evidence.setdefault("guided_checkpoint_summaries", state["guided_checkpoint_summaries"])
    return evidence


def _stage_version(state: AnalystState, stage: str, increment: bool = False) -> int:
    versions = state.setdefault("guided_checkpoint_versions", {})
    current = int(versions.get(stage, 0))
    if current <= 0:
        current = 1
    if increment:
        current += 1
    versions[stage] = current
    return current


def _append_log(
    state: AnalystState,
    stage: str,
    ai_recommendation: str,
    user_decision: str,
    version: int,
    reason: str | None = None,
    details: Dict[str, Any] | None = None,
) -> None:
    log = state.setdefault("guided_decision_log", [])
    entry = {
        "stage": stage,
        "ai_recommendation": ai_recommendation,
        "user_decision": user_decision,
        "timestamp": _utc_now(),
        "version": version,
    }
    if reason:
        entry["reason_for_modification"] = reason
    if details:
        entry["details"] = details
    log.append(entry)
    state.setdefault("analysis_evidence", {})["guided_decision_log"] = log
    state["guided_mode"]["last_checkpoint"] = entry
    state["human_in_loop"] = {
        "mode": "guided",
        "action": user_decision,
        "details": {
            "stage": stage,
            "version": version,
            **(details or {}),
        },
    }


def _record_checkpoint_summary(state: AnalystState, stage: str, summary: Dict[str, List[str] | str]) -> None:
    evidence = state.setdefault("analysis_evidence", {})
    summaries = evidence.setdefault("guided_checkpoint_summaries", state.setdefault("guided_checkpoint_summaries", {}))
    summaries[stage] = summary


def _selected_columns_summary(state: AnalystState) -> Tuple[List[str], List[str], List[str]]:
    df = _resolve_dataframe(state)
    all_columns = list(df.columns) if df is not None else []
    selected = list(state.get("selected_columns") or [])
    rejected = [col for col in all_columns if col not in selected]
    return all_columns, selected, rejected


def _current_df(state: AnalystState):
    return _resolve_dataframe(state)


def _extract_named_columns(text: str, columns: List[str]) -> List[str]:
    lower = text.lower()
    ordered: List[Tuple[int, str]] = []
    for column in columns:
        match = re.search(rf"(?<!\w){re.escape(column.lower())}(?!\w)", lower)
        if match:
            ordered.append((match.start(), column))
    ordered.sort(key=lambda item: item[0])
    return [column for _, column in ordered]


def _print_checkpoint(title: str, sections: Dict[str, List[str] | str]) -> None:
    print(f"\n=== {title} ===")
    for heading, body in sections.items():
        print(f"\n{heading}")
        if isinstance(body, str):
            print(body if body else "None")
        else:
            if not body:
                print("None")
            else:
                for line in body:
                    print(f"- {line}")


def _prompt_user(stage_label: str) -> str:
    return input(
        f"\nActions for {stage_label} (continue / modify / cancel): "
    ).strip().lower()


def _prompt_modification(stage_label: str, examples: str) -> str:
    return input(
        f"Describe the modification for {stage_label}.\n{examples}\n> "
    ).strip()


def _print_modification_review(stage_label: str, outcome: ModificationOutcome) -> None:
    print(f"\n[Agent] Modification review for {stage_label}:")
    if outcome.applied:
        if outcome.notes:
            print("[Agent] Applied changes:")
            for note in outcome.notes:
                print(f"- {note}")
        else:
            print("[Agent] No explicit supported changes were applied.")
    else:
        print(f"[Agent] I could not fully apply the requested {stage_label} change.")
        if outcome.fallback_reason:
            print(f"[Agent] Reason: {outcome.fallback_reason}")
        if outcome.unsupported:
            print("[Agent] Unsupported:", ", ".join(outcome.unsupported))
        print("[Agent] The original recommendation remains in place for the unsupported parts.")
    if outcome.applied and outcome.unsupported:
        print("[Agent] Some requested parts were left unchanged.")
        print("[Agent] Unsupported:", ", ".join(outcome.unsupported))


def _prompt_modification_approval(stage_label: str) -> str:
    while True:
        decision = input(
            f"\nApprove these {stage_label} changes? (continue / modify / cancel): "
        ).strip().lower()
        if decision in {"continue", "modify", "cancel"}:
            return decision
        print("[Agent] Please enter continue, modify, or cancel.")


def _format_count(items: List[Any]) -> str:
    if not items:
        return "None"
    return ", ".join(str(item) for item in items[:8])


def _resolve_dataframe(state: AnalystState) -> pd.DataFrame | None:
    df = state.get("analysis_dataset")
    if df is None:
        df = state.get("cleaned_data")
    if df is None:
        df = state.get("dataframe")
    return df


def _resolve_profile(state: AnalystState) -> Dict[str, Any]:
    evidence = state.setdefault("analysis_evidence", {})
    profile = state.get("dataset_profile") or evidence.get("dataset_profile_json")
    df = _resolve_dataframe(state)
    if (not profile) and df is not None:
        profile = profile_dataset(df)
        state["dataset_profile"] = profile
        evidence["dataset_profile_json"] = profile
    return profile or {}


def _reasoning_cache(state: AnalystState) -> Dict[str, Any]:
    evidence = state.setdefault("analysis_evidence", {})
    return evidence.setdefault("reasoning_cache", {})


def _explain_stage_decision(state: AnalystState, decision_object: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any], str]:
    reasoning, status = explain_decision(decision_object, cache=_reasoning_cache(state))
    return format_reasoning_explanation(reasoning), reasoning, status


def _profile_groups(profile: Dict[str, Any]) -> Dict[str, List[str]]:
    columns = profile.get("columns", {}) or {}
    numeric = [col for col, info in columns.items() if info.get("inferred_type") == "numeric"]
    categorical = [col for col, info in columns.items() if info.get("inferred_type") == "categorical"]
    datetime = [col for col, info in columns.items() if info.get("inferred_type") == "datetime"]
    identifiers = [col for col, info in columns.items() if info.get("inferred_type") == "identifier_like"]
    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime,
        "identifier_like": identifiers,
    }


def _severity_rank(severity: str | None) -> int:
    mapping = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    return mapping.get((severity or "").lower(), 1)


def _issue_impact_rank(issue: Dict[str, Any]) -> int:
    issue_type = issue.get("issue_type")
    severity = _severity_rank(issue.get("severity"))
    type_bonus = {
        "missing_values": 3,
        "duplicate_rows": 3,
        "outliers": 3,
        "constant_column": 2,
        "numeric_as_object": 2,
        "datetime_as_object": 2,
        "inconsistent_labels": 2,
        "high_cardinality": 1,
    }.get(issue_type, 1)
    return severity * 10 + type_bonus


def _issue_impact_text(issue: Dict[str, Any]) -> str:
    issue_type = issue.get("issue_type")
    severity = issue.get("severity", "medium").lower()
    if issue_type == "duplicate_rows":
        return f"Duplicates can bias aggregates and counts; severity {severity}."
    if issue_type == "missing_values":
        return f"Missing values can reduce reliability or remove records from analysis; severity {severity}."
    if issue_type == "outliers":
        return f"Outliers can distort summary statistics and relationships; severity {severity}."
    if issue_type == "constant_column":
        return f"A constant column adds no analytical value; severity {severity}."
    if issue_type == "numeric_as_object":
        return f"Numeric data stored as text can block analysis; severity {severity}."
    if issue_type == "datetime_as_object":
        return f"Time data stored as text can affect ordering and time-based analysis; severity {severity}."
    if issue_type == "inconsistent_labels":
        return f"Inconsistent labels can fragment grouping and reporting; severity {severity}."
    if issue_type == "high_cardinality":
        return f"Very granular categories may need domain-specific treatment; severity {severity}."
    return f"Potential data-quality issue detected; severity {severity}."


def _score_data_quality(profile: Dict[str, Any], issues: List[Dict[str, Any]], df: pd.DataFrame | None) -> int:
    if df is None:
        return 0
    score = 100.0
    missing_ratio = 0.0
    if profile.get("columns"):
        missing_ratio = sum(float(info.get("missing_ratio", 0.0)) for info in profile.get("columns", {}).values()) / max(len(profile["columns"]), 1)
    duplicate_ratio = float(df.duplicated().mean()) if len(df) else 0.0
    issue_penalty = sum(_severity_rank(issue.get("severity")) for issue in issues[:12])
    score -= missing_ratio * 45.0
    score -= duplicate_ratio * 35.0
    score -= issue_penalty * 3.0
    return int(max(0, min(100, round(score))))


def _alternative_actions_for_issue(issue: Dict[str, Any], profile: Dict[str, Any]) -> List[Dict[str, str]]:
    issue_type = issue.get("issue_type")
    recommended = issue.get("recommended_action")
    numeric_like_ratio = profile.get("numeric_like_ratio", 0.0)
    alternatives: List[Dict[str, str]] = []

    def _add(action: str, reason: str) -> None:
        if action == recommended:
            return
        if any(item["action"] == action for item in alternatives):
            return
        alternatives.append({"action": action, "reason": reason})

    if issue_type == "missing_values":
        if numeric_like_ratio >= 0.8:
            _add("impute_mean", "Suitable if the distribution is approximately symmetric and outliers are limited.")
            _add("impute_mode", "Useful if the column is categorical or label-like despite numeric storage.")
        else:
            _add("impute_mode", "Best when the column behaves like a categorical field.")
            _add("review_only", "Preserves original observations when there is no safe automatic fill.")
    elif issue_type == "outliers":
        _add("investigate_or_cap", "Reduces the influence of extreme values while retaining the record.")
        _add("review_only", "Useful when domain knowledge is needed before altering the values.")
    elif issue_type == "duplicate_rows":
        _add("review_only", "Lets an analyst confirm whether the duplicates are true duplicates or legitimate repeats.")
    elif issue_type in {"numeric_as_object", "datetime_as_object"}:
        _add("convert_to_numeric" if issue_type == "numeric_as_object" else "convert_to_datetime", "Normalizes the storage type for downstream analysis.")
        _add("review_only", "Useful when the values may encode special cases that should be checked first.")
    elif issue_type == "inconsistent_labels":
        _add("standardize_categories", "Aligns equivalent labels so grouping and reporting are consistent.")
        _add("review_only", "Allows manual review when label normalization might be too aggressive.")
    elif issue_type == "high_cardinality":
        _add("review_only", "High-cardinality fields often need analyst judgment before any automatic change.")
    else:
        _add("review_only", "Keeps the original data when automatic intervention is not clearly safe.")

    return alternatives[:3]


def _confidence_bundle(profile: Dict[str, Any], issues: List[Dict[str, Any]], validation: Dict[str, Any] | None = None) -> Dict[str, Any]:
    validation = validation or {}
    confidence_score = 100.0
    confidence_score -= min(40.0, len(issues) * 4.0)
    if validation.get("row_loss_ratio") is not None:
        confidence_score -= float(validation.get("row_loss_ratio", 0.0)) * 50.0
    if validation.get("anomalies"):
        confidence_score -= min(20.0, len(validation.get("anomalies", [])) * 8.0)
    if profile.get("column_count", 0) > 0:
        missing_mean = sum(float(info.get("missing_ratio", 0.0)) for info in profile.get("columns", {}).values()) / max(len(profile.get("columns", {})), 1)
        confidence_score -= missing_mean * 30.0
    confidence_score = max(0.0, min(100.0, confidence_score))
    category = "High" if confidence_score >= 80 else "Medium" if confidence_score >= 55 else "Low"
    factors = []
    if profile.get("column_count", 0):
        factors.append("Dataset profile is available")
    if not issues:
        factors.append("No major issues were detected")
    if validation.get("schema_stable", True):
        factors.append("Schema stayed stable during validation")
    if validation.get("row_loss_ratio", 0.0) == 0:
        factors.append("No row loss was introduced")
    return {
        "score": int(round(confidence_score)),
        "category": category,
        "factors": factors[:4] or ["Confidence is based on the currently available metadata."],
    }


def _impact_preview(stage: str, applied: bool, detail: str | None = None) -> Dict[str, Any]:
    if stage == "data_preparation":
        stages = ["Cleaning", "Validation", "Analysis dataset", "Planner", "Visualizations", "Executive report"]
    elif stage == "business_understanding":
        stages = ["Selected variables", "Analysis planner", "Statistical tests", "Visualizations", "Executive report"]
    elif stage == "analysis_strategy":
        stages = ["Analysis plan", "Tool execution", "Evidence synthesis", "Visualizations", "Executive report"]
    else:
        stages = ["Visualizations", "Executive report"]
    estimated_seconds = 6 + len(stages) * 2
    if not applied:
        estimated_seconds = max(3, estimated_seconds - 3)
    payload = {
        "affected_stages": stages,
        "estimated_recomputation_seconds": estimated_seconds,
    }
    if detail:
        payload["detail"] = detail
    return payload


def _print_impact_preview(preview: Dict[str, Any]) -> None:
    stages = preview.get("affected_stages", [])
    print("\n[Agent] Impact preview before modification:")
    print("[Agent] Affected stages:", ", ".join(stages) if stages else "None")
    print("[Agent] Estimated recomputation:", f"{preview.get('estimated_recomputation_seconds', 'unknown')} seconds")
    if preview.get("detail"):
        print("[Agent] Detail:", preview["detail"])


def _cleaning_action_intents(instruction_l: str) -> List[str]:
    intents: List[str] = []

    if any(term in instruction_l for term in ["mean imputation", "impute mean", "use mean"]) or (
        "mean" in instruction_l and any(term in instruction_l for term in ["fill", "impute", "replace", "with"])
    ):
        intents.append("impute_mean")

    if any(term in instruction_l for term in ["median imputation", "impute median", "use median"]) or (
        "median" in instruction_l and any(term in instruction_l for term in ["fill", "impute", "replace", "with"])
    ):
        intents.append("impute_median")

    if any(term in instruction_l for term in ["mode imputation", "impute mode", "use mode"]) or (
        "mode" in instruction_l and any(term in instruction_l for term in ["fill", "impute", "replace", "with"])
    ):
        intents.append("impute_mode")

    if any(term in instruction_l for term in ["forward fill", "fill forward", "ffill"]):
        intents.append("forward_fill")

    if any(term in instruction_l for term in ["backward fill", "fill backward", "bfill"]):
        intents.append("backward_fill")

    if any(term in instruction_l for term in ["remove duplicates", "drop duplicates", "deduplicate", "dedupe"]):
        intents.append("remove_duplicates")

    if any(term in instruction_l for term in ["keep duplicates", "preserve duplicates", "do not remove duplicates", "don't remove duplicates"]):
        intents.append("keep_duplicates")

    if any(term in instruction_l for term in ["keep outliers", "do not cap outliers", "do not remove outliers", "don't cap outliers"]):
        intents.append("keep_outliers")

    if any(term in instruction_l for term in ["cap outliers", "remove outliers", "trim outliers"]):
        intents.append("cap_outliers")

    if any(term in instruction_l for term in ["standardize categories", "normalize categories", "normalize labels", "fix labels", "clean labels", "harmonize labels"]):
        intents.append("standardize_categories")

    if any(term in instruction_l for term in ["convert to datetime", "parse as datetime", "datetime conversion"]):
        intents.append("convert_to_datetime")

    if any(term in instruction_l for term in ["convert to numeric", "parse as numeric", "numeric conversion"]):
        intents.append("convert_to_numeric")

    return intents


def _parse_restore_version(instruction_l: str, current_version: int) -> int | None:
    if not any(term in instruction_l for term in ["restore", "revert", "undo", "go back", "previous version", "version 1", "back to version"]):
        return None
    match = re.search(r"version\s*(\d+)", instruction_l)
    if match:
        requested = int(match.group(1))
        return requested if requested >= 1 else None
    if any(term in instruction_l for term in ["previous version", "go back", "undo", "revert"]):
        return max(1, current_version - 1)
    return None


def _version_history_lines(state: AnalystState, stage: str) -> List[str]:
    snapshots = (state.get("analysis_evidence") or {}).get("guided_version_snapshots") or {}
    stage_snapshots = snapshots.get(stage) or {}
    versions = sorted(int(version) for version in stage_snapshots.keys())
    if not versions:
        return ["No saved versions yet."]
    current = int((state.get("guided_checkpoint_versions") or {}).get(stage, versions[-1]))
    return [
        f"Current version: {current}",
        f"Available versions: {', '.join(map(str, versions))}",
    ]


def _workflow_supervision_lines(state: AnalystState, stage: str) -> List[str]:
    log = list(state.get("guided_decision_log") or [])
    completed_stages = [entry.get("stage") for entry in log if entry.get("stage")]
    current_version = int((state.get("guided_checkpoint_versions") or {}).get(stage, 1) or 1)
    completed_text = ", ".join(dict.fromkeys(map(str, completed_stages))) if completed_stages else "None yet"
    lines = [
        f"Current stage: {stage}",
        f"Current version: {current_version}",
        f"Completed stages so far: {completed_text}",
    ]
    if log:
        last = log[-1]
        lines.append(f"Last decision: {last.get('stage')} -> {last.get('user_decision')}")
    return lines


def _data_preparation_summary(state: AnalystState) -> Dict[str, List[str] | str]:
    evidence = state.setdefault("analysis_evidence", {})
    profile = _resolve_profile(state)
    df = _resolve_dataframe(state)
    issues = list(state.get("data_quality_issues", {}).get("issues", []))
    cleaning_plan = list(state.get("cleaning_plan", []))
    validation = state.get("cleaning_validation") or state.get("data_validation") or {}
    execution_log = evidence.get("cleaning_execution_log", [])
    selected, _, _ = _selected_columns_summary(state)
    groups = _profile_groups(profile)
    duplicate_count = int(df.duplicated().sum()) if df is not None else 0
    missing_count = int(df.isna().sum().sum()) if df is not None else 0
    quality_score = _score_data_quality(profile, issues, df)
    confidence = _confidence_bundle(profile, issues, validation)

    dataset_summary = [
        f"Rows: {profile.get('row_count', len(df) if df is not None else 0)}",
        f"Columns: {profile.get('column_count', len(df.columns) if df is not None else 0)}",
        f"Numeric columns: {_format_count(groups.get('numeric', []))}",
        f"Categorical columns: {_format_count(groups.get('categorical', []))}",
        f"Missing values: {missing_count}",
        f"Duplicates: {duplicate_count}",
        f"Data quality score: {quality_score}/100",
        f"Selected columns for analysis: {len(selected)}",
    ]

    ranked_issues = sorted(issues, key=_issue_impact_rank, reverse=True)
    issue_lines = []
    for issue in ranked_issues[:8]:
        column = issue.get("column")
        affected = [column] if column is not None else issue.get("affected_columns") or ["dataset"]
        suggested_action = issue.get("suggested_action") or issue.get("recommended_action", "review_only")
        alternatives = issue.get("alternatives") or _alternative_actions_for_issue(issue, profile)
        alternative_text = "; ".join(
            f"{alt.get('action')}: {alt.get('reason')}" for alt in alternatives[:2]
        ) or "No safe alternative identified."
        issue_lines.append(
            f"{issue.get('severity', 'medium').title()} | {issue.get('issue_type', 'issue')} | "
            f"Impact: {_issue_impact_text(issue)} | "
            f"Reason: {issue.get('reason', issue.get('reasoning', issue.get('explanation', 'Metadata-based detection')))} | "
            f"Affected columns: {_format_count([c for c in affected if c is not None])} | "
            f"Suggested action: {suggested_action} | "
            f"Alternatives: {alternative_text}"
        )
    if not issue_lines:
        issue_lines = ["No material data-quality issues were detected from the current dataset version."]

    cleaning_lines = []
    for step in cleaning_plan[:8]:
        column = step.get("column", "dataset")
        action = step.get("action", "review_only")
        explanation = step.get("explanation", "No explanation available.")
        cleaning_lines.append(f"{column}: {action} - {explanation}")
    if not cleaning_lines:
        cleaning_lines = ["No cleaning action was required for the current dataset version."]

    reasoning_lines = []
    for entry in execution_log[:8]:
        column = entry.get("column", "dataset")
        action = entry.get("action", "applied")
        status = entry.get("status", "applied")
        reasoning_lines.append(f"{column}: {action} ({status})")
    if not reasoning_lines:
        reasoning_lines = [
            f"Row loss ratio: {validation.get('row_loss_ratio', 0.0)}",
            f"Schema stable: {validation.get('schema_stable', True)}",
        ]
    stage_reasoning, reasoning_payload, reasoning_status = _explain_stage_decision(
        state,
        build_data_preparation_object(state),
    )

    main_recommendation = "Continue with the prepared cleaning version."
    if ranked_issues:
        top_issue = ranked_issues[0]
        main_recommendation = f"Prioritize {top_issue.get('issue_type', 'the top issue')} on {top_issue.get('column', 'the dataset')}."
    alternative_recommendations = [
        "Modify the cleaning plan if you want a different treatment for the top issue.",
        "Cancel if you want to stop before the cleaned dataset is reused downstream.",
    ]

    return {
        "What happened": dataset_summary,
        "What I found": issue_lines,
        "What was done": cleaning_lines,
        "Why this decision was made": stage_reasoning or reasoning_lines,
        "What I recommend": [main_recommendation, *alternative_recommendations],
        "How confident I am": [
            f"{confidence['score']}% ({confidence['category']})",
            *confidence["factors"],
        ],
        "Workflow supervision": _workflow_supervision_lines(state, "data_preparation"),
        "Version history": _version_history_lines(state, "data_preparation"),
        "What happens next": [
            "Continue accepts the current cleaned version.",
            "Modify recomputes the affected cleaning stages before analysis continues.",
            "Cancel stops the guided workflow at this checkpoint.",
        ],
        "LLM reasoning status": [reasoning_status],
    }


def _parse_cleaning_modification(state: AnalystState, instruction: str) -> ModificationOutcome:
    instruction_l = instruction.lower()
    df = state.get("dataframe")
    columns = list(df.columns) if df is not None else []
    named_columns = _extract_named_columns(instruction, columns)
    plan = [dict(step) for step in (state.get("cleaning_plan") or [])]
    unsupported: List[str] = []
    notes: List[str] = []
    applied = False
    current_version = int((state.get("guided_checkpoint_versions") or {}).get("data_preparation", 1) or 1)

    restore_version = _parse_restore_version(instruction_l, current_version)
    if restore_version is not None:
        restored = restore_guided_stage_snapshot(state, "data_preparation", restore_version)
        if restored is not None:
            snapshot_state = restored.get("state") or {}
            restored_plan = snapshot_state.get("cleaning_plan")
            if isinstance(restored_plan, list):
                plan = [dict(step) for step in restored_plan]
            state["cleaning_plan"] = plan
            notes.append(f"Restored cleaning version {restore_version}.")
            return ModificationOutcome(True, notes, unsupported, restored_version=restore_version)
        unsupported.append(f"version {restore_version}")
        return ModificationOutcome(False, notes, unsupported, "The requested cleaning version was not available.")

    def _replace_action(target_columns: List[str], predicate, new_action: str, explanation: str | None = None) -> None:
        nonlocal applied
        matched_columns: set[str] = set()
        for item in plan:
            column = item.get("column")
            if column in target_columns and predicate(item.get("action")):
                item["action"] = new_action
                if explanation:
                    item["explanation"] = explanation
                matched_columns.add(str(column))
        for column in target_columns:
            if str(column) in matched_columns:
                continue
            plan.append(
                {
                    "column": column,
                    "action": new_action,
                    "explanation": explanation or "User requested this cleaning change.",
                }
            )
        if target_columns:
            applied = True

    intents = _cleaning_action_intents(instruction_l)

    if "impute_mean" in intents:
        targets = named_columns or [
            item.get("column")
            for item in plan
            if item.get("action") in {"impute_median", "impute_mode", "forward_fill", "backward_fill", "review_only", "leave_unchanged"}
        ]
        targets = [col for col in targets if col]
        if targets:
            _replace_action(
                targets,
                lambda action: action in {"impute_median", "impute_mode", "forward_fill", "backward_fill", "review_only", "leave_unchanged"},
                "impute_mean",
                "User requested mean imputation.",
            )
            notes.append(f"Applied mean imputation to: {_format_count(targets)}")
        else:
            unsupported.append("mean imputation")

    if "impute_median" in intents:
        targets = named_columns or [
            item.get("column")
            for item in plan
            if item.get("action") in {"impute_mean", "impute_mode", "forward_fill", "backward_fill", "review_only", "leave_unchanged"}
        ]
        targets = [col for col in targets if col]
        if targets:
            _replace_action(
                targets,
                lambda action: action in {"impute_mean", "impute_mode", "forward_fill", "backward_fill", "review_only", "leave_unchanged"},
                "impute_median",
                "User requested median imputation.",
            )
            notes.append(f"Applied median imputation to: {_format_count(targets)}")
        else:
            unsupported.append("median imputation")

    if "impute_mode" in intents:
        targets = named_columns or [
            item.get("column")
            for item in plan
            if item.get("action") in {"impute_mean", "impute_median", "forward_fill", "backward_fill", "review_only", "leave_unchanged"}
        ]
        targets = [col for col in targets if col]
        if targets:
            _replace_action(
                targets,
                lambda action: action in {"impute_mean", "impute_median", "forward_fill", "backward_fill", "review_only", "leave_unchanged"},
                "impute_mode",
                "User requested mode imputation.",
            )
            notes.append(f"Applied mode imputation to: {_format_count(targets)}")
        else:
            unsupported.append("mode imputation")

    if "forward_fill" in intents:
        targets = named_columns or [item.get("column") for item in plan if item.get("action") in {"forward_fill", "backward_fill"}]
        targets = [col for col in targets if col]
        if targets:
            _replace_action(
                targets,
                lambda action: action in {"forward_fill", "backward_fill", "review_only", "leave_unchanged"},
                "forward_fill",
                "User requested forward fill.",
            )
            notes.append(f"Applied forward fill to: {_format_count(targets)}")
        else:
            unsupported.append("forward fill")

    if "backward_fill" in intents:
        targets = named_columns or [item.get("column") for item in plan if item.get("action") in {"forward_fill", "backward_fill"}]
        targets = [col for col in targets if col]
        if targets:
            _replace_action(
                targets,
                lambda action: action in {"forward_fill", "backward_fill", "review_only", "leave_unchanged"},
                "backward_fill",
                "User requested backward fill.",
            )
            notes.append(f"Applied backward fill to: {_format_count(targets)}")
        else:
            unsupported.append("backward fill")

    if "keep_duplicates" in intents:
        changed = False
        for item in plan:
            if item.get("action") == "remove_duplicates":
                item["action"] = "leave_unchanged"
                item["explanation"] = "User requested that duplicate rows be retained."
                changed = True
        if changed:
            applied = True
            notes.append("Kept duplicate rows in the cleaned dataset.")
        else:
            unsupported.append("duplicate retention")

    if "remove_duplicates" in intents:
        changed = False
        duplicate_step_found = any(item.get("action") in {"remove_duplicates", "leave_unchanged"} for item in plan)
        for item in plan:
            if item.get("action") in {"remove_duplicates", "leave_unchanged"}:
                item["action"] = "remove_duplicates"
                item["explanation"] = "User requested duplicate removal."
                changed = True
        if changed:
            applied = True
            notes.append("Removed duplicate rows.")
        elif not duplicate_step_found and df is not None and len(df) > 1:
            plan.append({"column": None, "action": "remove_duplicates", "explanation": "User requested duplicate removal."})
            applied = True
            notes.append("Added duplicate removal to the cleaning plan.")

    if "keep_outliers" in intents:
        changed = False
        for item in plan:
            if item.get("action") == "investigate_or_cap":
                item["action"] = "leave_unchanged"
                item["explanation"] = "User requested that outliers be retained."
                changed = True
        if changed:
            applied = True
            notes.append("Kept the outlier rows unchanged.")
        else:
            unsupported.append("outlier retention")

    if "cap_outliers" in intents:
        changed = False
        for item in plan:
            if item.get("action") in {"leave_unchanged", "investigate_or_cap"}:
                item["action"] = "investigate_or_cap"
                item["explanation"] = "User requested outlier capping."
                changed = True
        if changed:
            applied = True
            notes.append("Capped numeric outliers.")
        elif not any(item.get("action") == "investigate_or_cap" for item in plan):
            plan.append({"column": None, "action": "investigate_or_cap", "explanation": "User requested outlier capping."})
            applied = True
            notes.append("Added outlier capping to the cleaning plan.")

    if "standardize_categories" in intents:
        targets = named_columns or [item.get("column") for item in plan if item.get("action") in {"leave_unchanged", "review_only", "standardize_categories"}]
        targets = [col for col in targets if col]
        if targets:
            _replace_action(
                targets,
                lambda action: action in {"leave_unchanged", "review_only", "standardize_categories"},
                "standardize_categories",
                "User requested category normalization.",
            )
            notes.append(f"Standardized categories for: {_format_count(targets)}")
            applied = True
        else:
            unsupported.append("category standardization")

    if "convert_to_datetime" in intents:
        targets = named_columns or [item.get("column") for item in plan if item.get("action") in {"review_only", "leave_unchanged"}]
        targets = [col for col in targets if col]
        if targets:
            _replace_action(
                targets,
                lambda action: action in {"review_only", "leave_unchanged"},
                "convert_to_datetime",
                "User requested datetime conversion.",
            )
            notes.append(f"Converted to datetime for: {_format_count(targets)}")
        else:
            unsupported.append("datetime conversion")

    if "convert_to_numeric" in intents:
        targets = named_columns or [item.get("column") for item in plan if item.get("action") in {"review_only", "leave_unchanged"}]
        targets = [col for col in targets if col]
        if targets:
            _replace_action(
                targets,
                lambda action: action in {"review_only", "leave_unchanged"},
                "convert_to_numeric",
                "User requested numeric conversion.",
            )
            notes.append(f"Converted to numeric for: {_format_count(targets)}")
        else:
            unsupported.append("numeric conversion")

    if "mean" in instruction_l and "median" in instruction_l and not named_columns:
        unsupported.append("conflicting imputation instructions")

    if not applied and not notes:
        interpretation = interpret_modification_request(instruction, build_data_preparation_object(state))
        evidence_note = {
            "interpretation": interpretation,
            "raw_instruction": instruction,
        }
        state.setdefault("analysis_evidence", {})["guided_modification_interpretation"] = evidence_note
        if interpretation.get("best_matches"):
            notes.append(
                "Interpreted request as: "
                + ", ".join(
                    f"{item.get('name')}" for item in interpretation.get("best_matches", [])[:3] if item.get("name")
                )
            )
            applied = bool(interpretation.get("best_matches"))
        else:
            unsupported.append(instruction.strip() or "empty cleaning modification")

    state["cleaning_plan"] = plan
    state["analysis_evidence"]["guided_cleaning_notes"] = notes
    return ModificationOutcome(
        applied=applied,
        notes=notes,
        unsupported=unsupported,
    )


def _recompute_cleaning(state: AnalystState) -> None:
    df = state.get("dataframe")
    if df is None:
        raise ValueError("No dataframe found in state.")
    result = execute_cleaning_actions(
        df=df,
        actions=state.get("cleaning_plan", []),
        dataset_profile=state.get("dataset_profile") or state.get("analysis_evidence", {}).get("preclean_profile_json", {}),
        relationships=(state.get("relationship_signals") or {}).get("relationships", []),
    )
    state["cleaned_data"] = result["cleaned_df"]
    state.setdefault("analysis_evidence", {})["cleaning_execution_log"] = result["execution_log"]
    cleaning_audit_node(state)
    data_validation_node(state)


def _print_cleaning_recompute_summary(state: AnalystState) -> None:
    evidence = state.get("analysis_evidence", {})
    execution_log = evidence.get("cleaning_execution_log", [])
    validation = state.get("cleaning_validation") or {}

    if execution_log:
        print("\n[Agent] Updated cleaning output:")
        for entry in execution_log[:8]:
            column = entry.get("column", "dataset")
            action = entry.get("action", "unknown")
            status = entry.get("status", "unknown")
            print(f"[Agent] - {column}: {action} ({status})")
    else:
        print("\n[Agent] Updated cleaning output: no actions were executed.")

    if validation:
        print(
            "[Agent] Cleaning validation:",
            f"rows {validation.get('row_count_before', 'unknown')} -> {validation.get('row_count_after', 'unknown')}",
            f"row_loss_ratio={validation.get('row_loss_ratio', 'unknown')}",
            f"schema_stable={validation.get('schema_stable', 'unknown')}",
            f"anomalies={validation.get('anomalies', [])}",
        )


def guided_data_preparation_checkpoint_node(state: AnalystState) -> AnalystState:
    evidence = _ensure_guided_state(state)
    if state.get("mode", "autonomous") != "guided":
        return state

    version = _stage_version(state, "data_preparation")
    summary = _data_preparation_summary(state)
    _record_checkpoint_summary(state, "data_preparation", summary)
    _print_checkpoint("GUIDED CHECKPOINT 1 - DATA PREPARATION", summary)
    capture_guided_stage_snapshot(state, "data_preparation", version, summary=summary, note="pre_action_checkpoint")

    action = _prompt_user("data preparation")
    recommendation = "Continue with the current cleaning version."
    if action == "cancel":
        state["awaiting_user"] = True
        state["question_for_user"] = "Guided mode canceled at data preparation."
        evidence["final_output"] = ["Guided mode canceled during data preparation."]
        _append_log(state, "data_preparation", recommendation, "cancel", version, details={"summary": summary})
        return state

    if action == "modify":
        preview = _impact_preview("data_preparation", True, "Cleaning changes can affect validation, analysis inputs, visualizations, and the final report.")
        _print_impact_preview(preview)
        base_version = version
        while True:
            instruction = _prompt_modification(
                "data preparation",
                "Examples: use mean imputation, do not remove duplicates, keep outliers, standardize categories for Region",
            )
            outcome = _parse_cleaning_modification(state, instruction)
            _print_modification_review("data preparation", outcome)
            approval = _prompt_modification_approval("data preparation")
            if approval == "modify":
                restore_guided_stage_snapshot(state, "data_preparation", base_version)
                version = base_version
                continue
            if approval == "cancel":
                restore_guided_stage_snapshot(state, "data_preparation", base_version)
                state["awaiting_user"] = True
                state["question_for_user"] = "Guided mode canceled at data preparation."
                evidence["final_output"] = ["Guided mode canceled during data preparation."]
                _append_log(
                    state,
                    "data_preparation",
                    recommendation,
                    "cancel",
                    base_version,
                    reason=instruction,
                    details={"summary": summary, "notes": outcome.notes, "unsupported": outcome.unsupported, "impact_preview": preview},
                )
                return state

            if outcome.applied:
                details = {"summary": summary, "notes": outcome.notes, "unsupported": outcome.unsupported, "impact_preview": preview}
                if outcome.restored_version is not None:
                    version = outcome.restored_version
                    state.setdefault("guided_checkpoint_versions", {})["data_preparation"] = outcome.restored_version
                    recommendation = f"Restored cleaning version {version}."
                    _append_log(state, "data_preparation", recommendation, "restore", version, reason=instruction, details=details)
                    print(f"\n[Agent] Restored cleaning version {version}. Downstream stages will reuse the restored version.")
                    return state

                _recompute_cleaning(state)
                _print_cleaning_recompute_summary(state)
                version = _stage_version(state, "data_preparation", increment=True)
                updated_summary = _data_preparation_summary(state)
                capture_guided_stage_snapshot(state, "data_preparation", version, summary=updated_summary, note="post_action_checkpoint")
                recommendation = "Continue with the revised cleaning version."
                _append_log(state, "data_preparation", recommendation, "modify", version, reason=instruction, details=details)
                print("\n[Agent] Cleaning was updated and downstream stages will continue from the revised version.")
                return state

            _append_log(
                state,
                "data_preparation",
                recommendation,
                "modify_fallback",
                base_version,
                reason=instruction,
                details={"summary": summary, "unsupported": outcome.unsupported, "impact_preview": preview},
            )
            return state

    _append_log(state, "data_preparation", recommendation, "continue", version, details={"summary": summary})
    return state


def _business_understanding_summary(state: AnalystState) -> Dict[str, List[str] | str]:
    intent = state.get("intent") or {}
    selected = list(state.get("selected_columns") or intent.get("selected_columns") or [])
    full_df = state.get("cleaned_data")
    if full_df is None:
        full_df = state.get("dataframe")
    all_columns = list(full_df.columns) if full_df is not None else []
    rejected = [column for column in all_columns if column not in selected]
    column_registry = state.get("column_registry") or {}
    intent_columns = intent.get("intent_columns") or {}

    primary: List[str] = []
    supporting: List[str] = []
    for column in selected[:8]:
        registry = column_registry.get(column, {})
        reason_bits = []
        if column in intent.get("group_by_columns", []):
            reason_bits.append("grouping")
        if column in intent.get("aggregate_columns", []):
            reason_bits.append("aggregation")
        if column in intent_columns.get("comparison", []):
            reason_bits.append("comparison")
        if column in intent_columns.get("relationship", []):
            reason_bits.append("relationship")
        if not reason_bits and registry.get("semantic_role"):
            reason_bits.append(str(registry.get("semantic_role")))
        reason_text = ", ".join(reason_bits) if reason_bits else "explicitly retained"
        if any(tag in reason_bits for tag in ["grouping", "aggregation", "comparison", "relationship"]):
            primary.append(f"{column}: {reason_text}")
        else:
            supporting.append(f"{column}: {reason_text}")

    reasons_rejected = []
    for column in rejected[:8]:
        registry = column_registry.get(column, {})
        semantic_role = registry.get("semantic_role", "unknown")
        if semantic_role == "identifier":
            reason = "identifier-like column"
        elif column in all_columns and column not in selected:
            reason = f"not needed for the current analytic intent ({semantic_role})"
        else:
            reason = "not selected"
        reasons_rejected.append(f"{column}: {reason}")

    if not reasons_rejected:
        reasons_rejected = ["All available variables contribute to the current question."]

    confidence = intent.get("confidence")
    if isinstance(confidence, (int, float)):
        confidence_text = f"{round(float(confidence) * 100, 1)}%"
    else:
        confidence_text = "high" if selected else "medium"
    stage_reasoning, reasoning_payload, reasoning_status = _explain_stage_decision(
        state,
        build_business_understanding_object(state),
    )

    return {
        "Business question": [state.get("business_question", "N/A")],
        "Detected analytical intent": [str(intent.get("analytic_intent") or intent.get("type") or "unknown")],
        "Primary variables": primary if primary else ([f"{selected[0]}: primary analytical focus"] if selected else ["No primary variables were identified."]),
        "Supporting variables": supporting if supporting else ["No additional supporting variables were identified."],
        "Rejected variables": reasons_rejected,
        "Why these variables": stage_reasoning or ((primary + supporting) if (primary or supporting) else ["No explicit inclusion reasons were available."]),
        "Why these variables were excluded": reasons_rejected,
        "Confidence": confidence_text,
        "Workflow supervision": _workflow_supervision_lines(state, "business_understanding"),
        "Version history": _version_history_lines(state, "business_understanding"),
        "Recommendation": ["Continue with the current variable set unless you want to change what the current analysis will focus on."],
        "User actions": ["Continue", "Modify", "Cancel"],
        "LLM reasoning status": [reasoning_status],
    }


def _parse_selection_modification(state: AnalystState, instruction: str) -> ModificationOutcome:
    df = _current_df(state)
    if df is None:
        return ModificationOutcome(False, [], [instruction or "empty selection modification"], "No dataset available.")

    available = list(df.columns)
    selected = list(state.get("selected_columns") or [])
    lower_instruction = instruction.lower()
    named = _extract_named_columns(instruction, available)
    unsupported: List[str] = []
    notes: List[str] = []
    applied = False

    if "instead of" in lower_instruction and len(named) >= 2:
        old_col = named[-1]
        new_col = named[0]
        if old_col in selected and new_col in available:
            selected = [new_col if col == old_col else col for col in selected]
            applied = True
            notes.append(f"Replaced {old_col} with {new_col}.")
        else:
            unsupported.append(f"replace {old_col} with {new_col}")

    include_terms = ["include", "keep", "add", "use"]
    if "instead of" not in lower_instruction and any(term in lower_instruction for term in include_terms):
        for column in named:
            if column not in selected:
                selected.append(column)
                applied = True
                notes.append(f"Included {column}.")

    exclude_terms = ["remove", "exclude", "drop", "without", "ignore"]
    if any(term in lower_instruction for term in exclude_terms):
        for column in named:
            if column in selected:
                selected.remove(column)
                applied = True
                notes.append(f"Removed {column}.")

    if not named and "use" in lower_instruction and "instead of" not in lower_instruction:
        unsupported.append("unresolved column substitution")

    selected = [col for col in selected if col in available]
    if not selected:
        unsupported.append("no valid columns remained after modification")
        return ModificationOutcome(False, notes, unsupported, "The revised selection removed every usable column.")

    state["selected_columns"] = selected
    state["analysis_dataset"] = df[selected].copy()
    intent = state.setdefault("intent", {})
    intent["selected_columns"] = selected
    if intent.get("group_by") and intent["group_by"] not in selected:
        intent["group_by"] = selected[0] if selected else None
    if intent.get("aggregate_column") and intent["aggregate_column"] not in selected:
        numeric = [col for col in selected if pd.api.types.is_numeric_dtype(df[col])]
        intent["aggregate_column"] = numeric[0] if numeric else selected[0]
    evidence = state.setdefault("analysis_evidence", {})
    evidence["guided_selection_notes"] = notes
    return ModificationOutcome(applied or bool(notes), notes, unsupported)


def guided_business_understanding_checkpoint_node(state: AnalystState) -> AnalystState:
    evidence = _ensure_guided_state(state)
    if state.get("mode", "autonomous") != "guided":
        return state

    version = _stage_version(state, "business_understanding")
    summary = _business_understanding_summary(state)
    _record_checkpoint_summary(state, "business_understanding", summary)
    _print_checkpoint("GUIDED CHECKPOINT 2 - BUSINESS UNDERSTANDING", summary)
    capture_guided_stage_snapshot(state, "business_understanding", version, summary=summary, note="pre_action_checkpoint")

    action = _prompt_user("business understanding")
    recommendation = "Continue with the current variable selection."
    if action == "cancel":
        state["awaiting_user"] = True
        state["question_for_user"] = "Guided mode canceled at business understanding."
        evidence["final_output"] = ["Guided mode canceled during business understanding."]
        _append_log(state, "business_understanding", recommendation, "cancel", version, details={"summary": summary})
        return state

    if action == "modify":
        preview = _impact_preview("business_understanding", True, "Variable changes can affect the planner, statistical tests, visualizations, and the report.")
        _print_impact_preview(preview)
        base_version = version
        while True:
            instruction = _prompt_modification(
                "business understanding",
                "Examples: include Age, remove Region, use Profit instead of Revenue",
            )
            outcome = _parse_selection_modification(state, instruction)
            _print_modification_review("business understanding", outcome)
            approval = _prompt_modification_approval("business understanding")
            if approval == "modify":
                restore_guided_stage_snapshot(state, "business_understanding", base_version)
                version = base_version
                continue
            if approval == "cancel":
                restore_guided_stage_snapshot(state, "business_understanding", base_version)
                state["awaiting_user"] = True
                state["question_for_user"] = "Guided mode canceled at business understanding."
                evidence["final_output"] = ["Guided mode canceled during business understanding."]
                _append_log(
                    state,
                    "business_understanding",
                    recommendation,
                    "cancel",
                    base_version,
                    reason=instruction,
                    details={"summary": summary, "notes": outcome.notes, "unsupported": outcome.unsupported, "impact_preview": preview},
                )
                return state

            if outcome.applied:
                version = _stage_version(state, "business_understanding", increment=True)
                updated_summary = _business_understanding_summary(state)
                capture_guided_stage_snapshot(state, "business_understanding", version, summary=updated_summary, note="post_action_checkpoint")
                recommendation = "Continue with the revised variable selection."
                details = {"summary": summary, "notes": outcome.notes, "unsupported": outcome.unsupported, "impact_preview": preview}
                _append_log(state, "business_understanding", recommendation, "modify", version, reason=instruction, details=details)
                print("\n[Agent] Variable selection was updated and downstream stages will continue from the revised version.")
                return state

            _append_log(
                state,
                "business_understanding",
                recommendation,
                "modify_fallback",
                base_version,
                reason=instruction,
                details={"summary": summary, "unsupported": outcome.unsupported, "impact_preview": preview},
            )
            return state

    _append_log(state, "business_understanding", recommendation, "continue", version, details={"summary": summary})
    return state


def _analysis_strategy_summary(state: AnalystState) -> Dict[str, List[str] | str]:
    evidence = state.setdefault("analysis_evidence", {})
    intent = state.get("intent") or {}
    plan = evidence.get("analysis_plan") or state.get("analysis_plan") or []
    computation = evidence.get("computation_plan") or {}
    confidence = computation.get("confidence_score")
    if confidence is None and isinstance(intent.get("confidence"), (int, float)):
        confidence = intent.get("confidence")
    confidence_text = f"{round(float(confidence) * 100, 1)}%" if isinstance(confidence, (int, float)) else "unknown"
    steps = []
    for item in plan[:8]:
        columns = ", ".join(item.get("columns", []))
        steps.append(f"{item.get('tool')} on {columns or 'n/a'}")
    if not steps:
        steps = ["No analysis plan was generated."]
    assumption_checks = []
    for item in computation.get("steps", [])[:5]:
        op = item.get("operation")
        params = item.get("parameters", {})
        assumption_checks.append(f"{op}: {params or 'no special assumptions'}")
    if not assumption_checks:
        assumption_checks = ["No explicit assumption checks were available."]
    selected_reason = computation.get("justification", "The current plan best matches the resolved analytical intent and available evidence.")
    business_value = "This plan answers the current question with the available variables and preserves the strongest signal paths."
    if intent.get("analytic_intent"):
        business_value = f"The plan is aligned to the requested {intent.get('analytic_intent')} question."
    stage_reasoning, reasoning_payload, reasoning_status = _explain_stage_decision(
        state,
        build_analysis_strategy_object(state),
    )
    return {
        "What happened": steps,
        "Why this method was selected": stage_reasoning or [selected_reason],
        "Why not another method": ["Alternative methods were not selected because the current plan is the best fit for the available intent, variables, and assumptions."],
        "Assumptions and status": assumption_checks,
        "Expected outputs": [item.get("tool", "analysis result") for item in plan[:8]] or ["analysis result"],
        "Expected business value": [business_value],
        "Confidence": confidence_text,
        "Workflow supervision": _workflow_supervision_lines(state, "analysis_strategy"),
        "Version history": _version_history_lines(state, "analysis_strategy"),
        "Recommendation": ["Continue with the current analysis plan unless you want to change the analytical method."],
        "User actions": ["Continue", "Modify", "Cancel"],
        "LLM reasoning status": [reasoning_status],
    }


def _parse_analysis_modification(state: AnalystState, instruction: str) -> ModificationOutcome:
    plan = [dict(item) for item in (state.get("analysis_plan") or state.get("analysis_evidence", {}).get("analysis_plan") or [])]
    instruction_l = instruction.lower()
    df = _current_df(state)
    named_columns = _extract_named_columns(instruction, list(df.columns)) if df is not None else []
    unsupported: List[str] = []
    notes: List[str] = []
    applied = False

    supported_tools = {
        "correlation",
        "ttest",
        "anova",
        "summary_statistics",
        "detect_outliers",
        "categorical_analysis",
        "predictive_analysis",
        "prescriptive_analysis",
        "regression",
    }
    unsupported_tools = {
        "kruskal-wallis": "kruskal-wallis",
        "kruskal wallis": "kruskal-wallis",
        "mann-whitney": "mann-whitney",
        "wilcoxon": "wilcoxon",
    }

    for phrase, label in unsupported_tools.items():
        if phrase in instruction_l:
            unsupported.append(label)

    if "skip correlation" in instruction_l or "remove correlation" in instruction_l:
        before = len(plan)
        plan = [item for item in plan if item.get("tool") != "correlation"]
        if len(plan) != before:
            applied = True
            notes.append("Removed correlation analysis.")

    if any(phrase in instruction_l for phrase in ["add regression", "use regression"]):
        available = list(df.columns) if df is not None else []
        if len(named_columns) >= 2:
            columns = named_columns[:2]
        else:
            numeric = []
            if df is not None:
                numeric = [col for col in available if pd.api.types.is_numeric_dtype(df[col])]
            columns = numeric[:2]
        if len(columns) >= 2:
            plan.append({"tool": "regression", "columns": columns[:2]})
            applied = True
            notes.append(f"Added regression for {columns[0]} and {columns[1]}.")
        else:
            unsupported.append("regression")

    for tool in sorted(supported_tools):
        if tool in instruction_l and tool not in {"regression"}:
            if not any(item.get("tool") == tool for item in plan):
                available = list(df.columns) if df is not None else []
                if len(named_columns) >= 2 and tool in {"correlation", "anova", "ttest"}:
                    plan.append({"tool": tool, "columns": named_columns[:2]})
                    applied = True
                    notes.append(f"Added {tool} for {named_columns[0]} and {named_columns[1]}.")
                elif len(available) >= 2:
                    plan.append({"tool": tool, "columns": available[:2]})
                    applied = True
                    notes.append(f"Added {tool} using the first supported columns.")

    if not applied and unsupported:
        state["analysis_plan"] = plan
        state.setdefault("analysis_evidence", {})["analysis_plan"] = plan
        return ModificationOutcome(False, notes, unsupported, "The requested statistical method is not supported by the current engine.")

    if not plan:
        unsupported.append("empty analysis plan")
        return ModificationOutcome(False, notes, unsupported, "The revised analysis plan would remove every analysis step.")

    state["analysis_plan"] = plan
    state.setdefault("analysis_evidence", {})["analysis_plan"] = plan
    return ModificationOutcome(applied, notes, unsupported)


def guided_analysis_strategy_checkpoint(state: AnalystState) -> AnalystState:
    evidence = _ensure_guided_state(state)
    if state.get("mode", "autonomous") != "guided":
        return state

    version = _stage_version(state, "analysis_strategy")
    summary = _analysis_strategy_summary(state)
    _record_checkpoint_summary(state, "analysis_strategy", summary)
    _print_checkpoint("GUIDED CHECKPOINT 3 - ANALYSIS STRATEGY", summary)
    capture_guided_stage_snapshot(state, "analysis_strategy", version, summary=summary, note="pre_action_checkpoint")

    action = _prompt_user("analysis strategy")
    recommendation = "Continue with the current analysis strategy."
    if action == "cancel":
        state["awaiting_user"] = True
        state["question_for_user"] = "Guided mode canceled at analysis strategy."
        evidence["final_output"] = ["Guided mode canceled during analysis strategy."]
        _append_log(state, "analysis_strategy", recommendation, "cancel", version, details={"summary": summary})
        return state

    if action == "modify":
        preview = _impact_preview("analysis_strategy", True, "Analysis changes can alter tool execution, evidence synthesis, visualizations, and the report.")
        _print_impact_preview(preview)
        base_version = version
        while True:
            instruction = _prompt_modification(
                "analysis strategy",
                "Examples: skip correlation, add regression, use ANOVA, use Kruskal-Wallis",
            )
            outcome = _parse_analysis_modification(state, instruction)
            _print_modification_review("analysis strategy", outcome)
            approval = _prompt_modification_approval("analysis strategy")
            if approval == "modify":
                restore_guided_stage_snapshot(state, "analysis_strategy", base_version)
                version = base_version
                continue
            if approval == "cancel":
                restore_guided_stage_snapshot(state, "analysis_strategy", base_version)
                state["awaiting_user"] = True
                state["question_for_user"] = "Guided mode canceled at analysis strategy."
                evidence["final_output"] = ["Guided mode canceled during analysis strategy."]
                _append_log(
                    state,
                    "analysis_strategy",
                    recommendation,
                    "cancel",
                    base_version,
                    reason=instruction,
                    details={"summary": summary, "notes": outcome.notes, "unsupported": outcome.unsupported, "impact_preview": preview},
                )
                return state

            if outcome.applied:
                version = _stage_version(state, "analysis_strategy", increment=True)
                updated_summary = _analysis_strategy_summary(state)
                capture_guided_stage_snapshot(state, "analysis_strategy", version, summary=updated_summary, note="post_action_checkpoint")
                recommendation = "Continue with the revised analysis strategy."
                details = {"summary": summary, "notes": outcome.notes, "unsupported": outcome.unsupported, "impact_preview": preview}
                _append_log(state, "analysis_strategy", recommendation, "modify", version, reason=instruction, details=details)
                print("\n[Agent] Analysis strategy was updated and downstream execution will use the revised plan.")
                return state

            _append_log(
                state,
                "analysis_strategy",
                recommendation,
                "modify_fallback",
                base_version,
                reason=instruction,
                details={"summary": summary, "unsupported": outcome.unsupported, "impact_preview": preview},
            )
            return state

    _append_log(state, "analysis_strategy", recommendation, "continue", version, details={"summary": summary})
    return state


def _visualization_summary(state: AnalystState) -> Dict[str, List[str] | str]:
    evidence = state.setdefault("analysis_evidence", {})
    top_stories = evidence.get("top_stories", []) or []
    charts = evidence.get("visualizations", []) or []
    judgment = evidence.get("judgment_summary", {}) or {}
    confidence = judgment.get("global_confidence")
    if confidence is None and top_stories:
        confidence = top_stories[0].get("confidence")
    chart_lines = [
        f"{chart.get('type')} -> {chart.get('file_path')}"
        for chart in charts[:8]
    ]
    if not chart_lines:
        chart_lines = ["No visualizations were generated."]
    preview_lines = [
        story.get("insight", "No preview available.")
        for story in top_stories[:3]
    ] or ["No preview available."]
    stage_reasoning, reasoning_payload, reasoning_status = _explain_stage_decision(
        state,
        build_result_review_object(state),
    )
    return {
        "What happened": preview_lines,
        "Why this happened": stage_reasoning or [str(judgment.get("dominant_reasoning", "Evidence quality not yet finalized."))],
        "What I recommend": ["Continue to the reporting layer unless you want to change how the charts are rendered."],
        "How confident I am": [str(confidence) if confidence is not None else "unknown"],
        "Workflow supervision": _workflow_supervision_lines(state, "result_review"),
        "Version history": _version_history_lines(state, "result_review"),
        "What happens next": chart_lines,
        "User actions": ["Continue", "Modify", "Cancel"],
        "LLM reasoning status": [reasoning_status],
    }


def _parse_visualization_modification(state: AnalystState, instruction: str) -> ModificationOutcome:
    instruction_l = instruction.lower()
    preferences = state.setdefault("guided_visualization_preferences", {})
    chart_overrides = dict(preferences.get("chart_overrides", {}))
    notes: List[str] = []
    unsupported: List[str] = []
    applied = False

    if "boxplot" in instruction_l and "histogram" in instruction_l:
        chart_overrides["group_difference"] = "histogram"
        chart_overrides["inferential_group_difference"] = "histogram"
        applied = True
        notes.append("Replaced boxplot-style comparisons with histograms.")

    if "scatter" in instruction_l and "regression" in instruction_l:
        chart_overrides["correlation"] = "regression"
        chart_overrides["inferential_relationship"] = "regression"
        applied = True
        notes.append("Replaced scatter plots with regression charts.")

    if any(phrase in instruction_l for phrase in ["bar chart", "bar plot"]) and "histogram" in instruction_l:
        chart_overrides["category_frequency"] = "bar"
        applied = True
        notes.append("Kept categorical plots as bar charts.")

    if "remove chart" in instruction_l or "hide chart" in instruction_l:
        unsupported.append("chart removal")

    if not applied and not unsupported:
        unsupported.append(instruction.strip() or "empty visualization modification")

    preferences["chart_overrides"] = chart_overrides
    state["guided_visualization_preferences"] = preferences
    state["analysis_evidence"]["guided_visualization_notes"] = notes
    return ModificationOutcome(applied, notes, unsupported)


def guided_result_review_checkpoint_node(state: AnalystState) -> AnalystState:
    evidence = _ensure_guided_state(state)
    if state.get("mode", "autonomous") != "guided":
        return state

    version = _stage_version(state, "result_review")
    summary = _visualization_summary(state)
    _record_checkpoint_summary(state, "result_review", summary)
    _print_checkpoint("GUIDED CHECKPOINT 4 - RESULT REVIEW", summary)
    capture_guided_stage_snapshot(state, "result_review", version, summary=summary, note="pre_action_checkpoint")

    action = _prompt_user("result review")
    recommendation = "Continue to the reporting layer."
    if action == "cancel":
        state["awaiting_user"] = True
        state["question_for_user"] = "Guided mode canceled at result review."
        evidence["final_output"] = ["Guided mode canceled during result review."]
        _append_log(state, "result_review", recommendation, "cancel", version, details={"summary": summary})
        return state

    if action == "modify":
        preview = _impact_preview("result_review", True, "Visualization changes primarily affect charts and the final report.")
        _print_impact_preview(preview)
        base_version = version
        while True:
            instruction = _prompt_modification(
                "result review",
                "Examples: replace boxplot with histogram, use regression instead of scatter, hide a chart",
            )
            outcome = _parse_visualization_modification(state, instruction)
            _print_modification_review("result review", outcome)
            approval = _prompt_modification_approval("result review")
            if approval == "modify":
                restore_guided_stage_snapshot(state, "result_review", base_version)
                version = base_version
                continue
            if approval == "cancel":
                restore_guided_stage_snapshot(state, "result_review", base_version)
                state["awaiting_user"] = True
                state["question_for_user"] = "Guided mode canceled at result review."
                evidence["final_output"] = ["Guided mode canceled during result review."]
                _append_log(
                    state,
                    "result_review",
                    recommendation,
                    "cancel",
                    base_version,
                    reason=instruction,
                    details={"summary": summary, "notes": outcome.notes, "unsupported": outcome.unsupported, "impact_preview": preview},
                )
                return state

            if outcome.applied:
                visualization_generator_node(state)
                version = _stage_version(state, "result_review", increment=True)
                updated_summary = _visualization_summary(state)
                capture_guided_stage_snapshot(state, "result_review", version, summary=updated_summary, note="post_action_checkpoint")
                recommendation = "Continue to the reporting layer with the revised visualizations."
                details = {"summary": summary, "notes": outcome.notes, "unsupported": outcome.unsupported, "impact_preview": preview}
                _append_log(state, "result_review", recommendation, "modify", version, reason=instruction, details=details)
                print("\n[Agent] Visualizations were regenerated and downstream reporting will use the revised version.")
                return state

            _append_log(
                state,
                "result_review",
                recommendation,
                "modify_fallback",
                base_version,
                reason=instruction,
                details={"summary": summary, "unsupported": outcome.unsupported, "impact_preview": preview},
            )
            return state

    _append_log(state, "result_review", recommendation, "continue", version, details={"summary": summary})
    return state
