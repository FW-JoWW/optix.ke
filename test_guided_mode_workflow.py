from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Any, Callable

from core.reasoning_layer import explain_decision, format_reasoning_explanation, interpret_modification_request
from core.guided_versions import diff_guided_stage_snapshots, restore_guided_stage_snapshot
from core.reasoning_objects import (
    build_analysis_strategy_object,
    build_business_understanding_object,
    build_data_preparation_object,
    build_final_reasoning_object,
    build_result_review_object,
)
from scripts.guided_mode_harness import run_guided_workflow, summarize_guided_result
from utils.issue_detector import detect_issues


QUESTION = "What is the relationship between Revenue and Profit by Region?"


def test_guided_mode_full_workflow_continue() -> None:
    result = run_guided_workflow(
        question=QUESTION,
        responses=["continue", "continue", "continue", "continue"],
    )
    summary = summarize_guided_result(result)

    assert summary["final_report_available"] is True
    assert summary["awaiting_user"] is False
    assert len(summary["guided_decision_log"]) == 4
    assert result.final_state.get("final_report")
    print("GUIDED CONTINUE OK")


def test_guided_mode_full_workflow_modification_and_fallback() -> None:
    result = run_guided_workflow(
        question=QUESTION,
        responses=[
            "modify",
            "use mean imputation for Age, keep outliers, do not remove duplicates",
            "continue",
            "continue",
            "modify",
            "use Kruskal-Wallis",
            "continue",
            "continue",
        ],
    )
    summary = summarize_guided_result(result)
    log = result.final_state.get("guided_decision_log", [])

    assert summary["final_report_available"] is True
    assert summary["awaiting_user"] is False
    assert len(log) == 4
    assert any(
        entry.get("stage") == "analysis_strategy" and "Kruskal-Wallis" in entry.get("reason_for_modification", "")
        for entry in log
    )
    assert result.final_state.get("final_report")
    print("GUIDED MODIFICATION/FALLBACK OK")


def test_guided_mode_column_specific_cleaning_modification() -> None:
    result = run_guided_workflow(
        question=QUESTION,
        responses=[
            "modify",
            "use median imputation for Age, keep outliers, do not remove duplicates",
            "continue",
            "continue",
            "continue",
            "continue",
        ],
    )
    summary = summarize_guided_result(result)
    plan = result.final_state.get("cleaning_plan", [])
    execution_log = result.final_state.get("analysis_evidence", {}).get("cleaning_execution_log", [])

    assert summary["final_report_available"] is True
    assert any(step.get("column") == "Age" and step.get("action") == "impute_median" for step in plan)
    assert any(entry.get("column") == "Age" and entry.get("action") == "impute_median" for entry in execution_log)
    print("GUIDED COLUMN-SPECIFIC CLEANING OK")


def test_issue_detector_does_not_guess_datetime_for_plain_codes() -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "TransactionCode": ["202401", "202402", "202403", None],
            "Value": [1, 2, 3, 4],
        }
    )
    issues = detect_issues(df).get("detected_issues", [])

    assert not any(issue.get("issue_type") == "datetime_as_object" for issue in issues)
    print("DATETIME FALSE POSITIVE GUARD OK")


def test_reasoning_layer_returns_structured_explanation() -> None:
    result = run_guided_workflow(
        question=QUESTION,
        responses=["continue", "continue", "continue", "continue"],
    )
    decision_object = build_data_preparation_object(result.final_state)
    reasoning, status = explain_decision(decision_object, cache={})

    assert isinstance(reasoning, dict)
    assert reasoning.get("confidence", {}).get("score") == decision_object.get("confidence", {}).get("score")
    assert reasoning.get("summary")
    print(f"REASONING LAYER OK ({status})")


def test_guided_mode_cancel_gracefully() -> None:
    result = run_guided_workflow(
        question=QUESTION,
        responses=["continue", "continue", "continue", "cancel"],
    )
    summary = summarize_guided_result(result)

    assert summary["awaiting_user"] is True
    assert summary["final_report_available"] is False
    assert summary["final_output"]
    assert "canceled" in " ".join(map(str, summary["final_output"])).lower()
    print("GUIDED CANCEL OK")


def _plain_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_plain_text(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_plain_text(item) for item in value)
    return str(value)


def _deterministic_reasoning(decision_object: dict[str, Any]) -> tuple[dict[str, Any], list[str], str]:
    import core.reasoning_layer as reasoning_layer_module

    original_client_factory = reasoning_layer_module.get_openai_client
    reasoning_layer_module.get_openai_client = lambda: None
    try:
        reasoning, status = explain_decision(decision_object, cache={})
    finally:
        reasoning_layer_module.get_openai_client = original_client_factory
    return reasoning, format_reasoning_explanation(reasoning), status


def _assert_probe(name: str, condition: bool, detail: str) -> None:
    assert condition, f"{name}: {detail}"
    print(f"{name} -> {detail}")


def test_guided_mode_capability_question_bank() -> None:
    def _run_silent(responses: list[str]):
        with redirect_stdout(io.StringIO()):
            return run_guided_workflow(question=QUESTION, responses=responses)

    happy = _run_silent(["continue", "continue", "continue", "continue"])
    modified = _run_silent(
        [
            "modify",
            "use median imputation for Age, keep outliers, do not remove duplicates",
            "continue",
            "continue",
            "continue",
            "continue",
        ]
    )
    fallback = _run_silent(
        [
            "modify",
            "use mean imputation for Age, keep outliers, do not remove duplicates",
            "continue",
            "continue",
            "continue",
            "continue",
        ]
    )
    canceled = _run_silent(["continue", "continue", "continue", "cancel"])

    happy_state = happy.final_state
    modified_state = modified.final_state
    fallback_state = fallback.final_state
    canceled_state = canceled.final_state

    data_prep = build_data_preparation_object(happy_state)
    business = build_business_understanding_object(happy_state)
    analysis = build_analysis_strategy_object(happy_state)
    review = build_result_review_object(happy_state)
    final_reasoning = build_final_reasoning_object(happy_state)

    happy_evidence = happy_state.get("analysis_evidence", {})
    cleaning_plan = list(happy_state.get("cleaning_plan") or [])
    cleaning_validation = happy_state.get("cleaning_validation") or {}
    guided_log = list(happy_state.get("guided_decision_log") or [])
    question_text = _plain_text(happy_state.get("business_question"))
    issue_list = list((happy_state.get("data_quality_issues") or {}).get("issues", []))
    reasoning_summary = _deterministic_reasoning(data_prep)[1]
    business_reasoning = _deterministic_reasoning(business)[1]
    analysis_reasoning = _deterministic_reasoning(analysis)[1]
    review_reasoning = _deterministic_reasoning(review)[1]
    final_reasoning_lines = _deterministic_reasoning(final_reasoning)[1]

    # Data preparation capability checks.
    _assert_probe(
        "duplicate retention",
        not any(step.get("action") == "remove_duplicates" for step in cleaning_plan),
        "no duplicate-removal action was planned because the sample data has no duplicate rows",
    )
    _assert_probe(
        "missing value treatment",
        any(issue.get("column") == "Age" and issue.get("suggested_action") == "impute_median" for issue in issue_list),
        "Age is recommended for median imputation in the deterministic cleaning evidence",
    )
    _assert_probe(
        "which columns require cleaning",
        any(issue.get("column") == "Age" for issue in issue_list),
        "Age is the only column flagged by the current sample dataset",
    )
    _assert_probe(
        "critical issues",
        not any(str(issue.get("severity", "")).lower() == "critical" for issue in issue_list),
        "no critical issue was detected in the sample dataset",
    )
    _assert_probe(
        "cleaning assumptions",
        bool(data_prep.get("assumptions")),
        f"cleaning assumptions are exposed: {data_prep.get('assumptions')}",
    )
    _assert_probe(
        "alternative cleaning strategies",
        bool(data_prep.get("alternatives")),
        f"alternatives are exposed: {data_prep.get('alternatives')}",
    )
    _assert_probe(
        "cleaning confidence",
        data_prep.get("confidence", {}).get("score") is not None,
        f"confidence is tracked as {data_prep.get('confidence')}",
    )
    _assert_probe(
        "cleaning explanation",
        len(reasoning_summary) >= 4,
        f"reasoning layer returned {len(reasoning_summary)} explanation lines",
    )
    _assert_probe(
        "workflow supervision summary",
        "Current stage: data_preparation" in _plain_text(
            (happy_state.get("analysis_evidence", {}).get("guided_checkpoint_summaries") or {}).get("data_preparation", {}).get("Workflow supervision")
        ),
        f"workflow supervision is exposed for data preparation: {(happy_state.get('analysis_evidence', {}).get('guided_checkpoint_summaries') or {}).get('data_preparation', {}).get('Workflow supervision')}",
    )
    _assert_probe(
        "dataset alteration size",
        cleaning_validation.get("row_loss_ratio", 0) == 0,
        f"row loss ratio stayed at {cleaning_validation.get('row_loss_ratio', 0)}",
    )

    # Modification capability checks.
    interpreted_mean = interpret_modification_request(
        "What would happen if I choose mean instead of median imputation?",
        data_prep,
    )
    _assert_probe(
        "mean vs median fallback",
        interpreted_mean.get("fallback_recommendation") == data_prep.get("recommendation"),
        f"unsupported mean-vs-median request falls back to {interpreted_mean.get('fallback_recommendation')}",
    )
    _assert_probe(
        "leave untouched fallback",
        interpret_modification_request("Can you leave the missing values untouched?", data_prep).get("fallback_recommendation")
        == data_prep.get("recommendation"),
        "untouched request preserves the original deterministic recommendation",
    )
    _assert_probe(
        "reversible versioning",
        modified_state.get("guided_checkpoint_versions", {}).get("data_preparation", 0) >= 2,
        f"data-preparation version advanced to {modified_state.get('guided_checkpoint_versions', {}).get('data_preparation', 0)} after modification",
    )
    _assert_probe(
        "previous version retained",
        any(entry.get("stage") == "data_preparation" and entry.get("version") == 2 for entry in modified_state.get("guided_decision_log", [])),
        "the modification created a distinct second version in the decision log",
    )
    _assert_probe(
        "partial modification logging",
        any(
            entry.get("stage") == "data_preparation"
            and entry.get("user_decision") == "modify"
            and "duplicate retention" in _plain_text((entry.get("details") or {}).get("unsupported", []))
            and "outlier retention" in _plain_text((entry.get("details") or {}).get("unsupported", []))
            for entry in fallback_state.get("guided_decision_log", [])
        ),
        "the unsupported parts of the mean-imputation request are recorded while the supported part is applied",
    )

    # Business understanding capability checks.
    _assert_probe(
        "business question inference",
        "relationship" in str(happy_state.get("intent", {}).get("analytic_intent", happy_state.get("intent", {}).get("type", ""))).lower(),
        f"the inferred intent is {happy_state.get('intent', {}).get('analytic_intent', happy_state.get('intent', {}).get('type'))}",
    )
    _assert_probe(
        "most relevant variables",
        {"Revenue", "Profit", "Region"}.issubset(set(happy_state.get("selected_columns") or [])),
        f"selected columns are {happy_state.get('selected_columns')}",
    )
    _assert_probe(
        "rejected variables",
        "Age" not in (happy_state.get("selected_columns") or []) and "Age" in _plain_text(business.get("evidence")),
        f"Age is rejected in the business-understanding evidence: {business.get('evidence')}",
    )
    _assert_probe(
        "supporting variables",
        "Score" in (happy_state.get("selected_columns") or []),
        f"supporting variables include Score via {happy_state.get('selected_columns')}",
    )
    _assert_probe(
        "business assumptions",
        bool(business.get("assumptions")),
        f"business understanding assumptions are exposed: {business.get('assumptions')}",
    )
    _assert_probe(
        "business confidence",
        business.get("confidence", {}).get("score") is not None,
        f"business confidence is {business.get('confidence')}",
    )
    _assert_probe(
        "business reasoning",
        len(business_reasoning) >= 4,
        f"business reasoning produced {len(business_reasoning)} lines",
    )
    _assert_probe(
        "business supervision summary",
        "Current stage: business_understanding" in _plain_text(
            (happy_state.get("analysis_evidence", {}).get("guided_checkpoint_summaries") or {}).get("business_understanding", {}).get("Workflow supervision")
        ),
        f"workflow supervision is exposed for business understanding: {(happy_state.get('analysis_evidence', {}).get('guided_checkpoint_summaries') or {}).get('business_understanding', {}).get('Workflow supervision')}",
    )

    # Analysis planning capability checks.
    _assert_probe(
        "analysis method choice",
        "anova" not in _plain_text(analysis).lower() and "regression" not in _plain_text(analysis).lower(),
        f"the deterministic plan uses {analysis.get('evidence')}",
    )
    _assert_probe(
        "analysis assumptions",
        bool(analysis.get("assumptions")),
        f"analysis assumptions are exposed: {analysis.get('assumptions')}",
    )
    _assert_probe(
        "analysis alternatives",
        bool(analysis.get("alternatives")),
        f"analysis alternatives are exposed: {analysis.get('alternatives')}",
    )
    _assert_probe(
        "analysis confidence",
        analysis.get("confidence", {}).get("score") is not None,
        f"analysis confidence is {analysis.get('confidence')}",
    )
    _assert_probe(
        "analysis reasoning",
        len(analysis_reasoning) >= 4,
        f"analysis reasoning produced {len(analysis_reasoning)} lines",
    )
    _assert_probe(
        "analysis supervision summary",
        "Current stage: analysis_strategy" in _plain_text(
            (happy_state.get("analysis_evidence", {}).get("guided_checkpoint_summaries") or {}).get("analysis_strategy", {}).get("Workflow supervision")
        ),
        f"workflow supervision is exposed for analysis strategy: {(happy_state.get('analysis_evidence', {}).get('guided_checkpoint_summaries') or {}).get('analysis_strategy', {}).get('Workflow supervision')}",
    )

    # Workflow supervision and reporting checks.
    _assert_probe(
        "workflow progress",
        len(guided_log) == 4,
        f"the guided workflow completed four checkpoints: {[entry.get('stage') for entry in guided_log]}",
    )
    _assert_probe(
        "current workflow stage",
        guided_log[-1].get("stage") == "result_review",
        f"the current reviewed stage is {guided_log[-1].get('stage')}",
    )
    _assert_probe(
        "continue path",
        all(entry.get("user_decision") == "continue" for entry in guided_log),
        f"default run accepted each recommendation: {[entry.get('user_decision') for entry in guided_log]}",
    )
    _assert_probe(
        "cancel path",
        canceled_state.get("awaiting_user") is True and not canceled_state.get("final_report"),
        "cancel stops the workflow before final reporting",
    )
    _assert_probe(
        "decision log",
        bool(guided_log),
        f"decision log contains {len(guided_log)} entries",
    )
    _assert_probe(
        "accepted recommendations",
        any(entry.get("user_decision") == "continue" for entry in guided_log),
        "the happy-path run accepted the stage recommendations",
    )
    _assert_probe(
        "rejected recommendations",
        any(entry.get("user_decision") in {"modify", "modify_fallback", "cancel"} for entry in fallback_state.get("guided_decision_log", [])),
        "the fallback run records a rejected or modified recommendation",
    )
    _assert_probe(
        "stage summary",
        bool(final_reasoning.get("evidence")) and bool(review.get("evidence")),
        f"reporting evidence is available: {final_reasoning.get('evidence')} / {review.get('evidence')}",
    )
    _assert_probe(
        "explain reasoning",
        len(review_reasoning) >= 4,
        f"reporting reasoning produced {len(review_reasoning)} lines",
    )
    _assert_probe(
        "review supervision summary",
        "Current stage: result_review" in _plain_text(
            (happy_state.get("analysis_evidence", {}).get("guided_checkpoint_summaries") or {}).get("result_review", {}).get("Workflow supervision")
        ),
        f"workflow supervision is exposed for result review: {(happy_state.get('analysis_evidence', {}).get('guided_checkpoint_summaries') or {}).get('result_review', {}).get('Workflow supervision')}",
    )
    _assert_probe(
        "version review",
        modified_state.get("guided_checkpoint_versions", {}).get("result_review", 0) >= 1,
        f"result-review version is {modified_state.get('guided_checkpoint_versions', {}).get('result_review', 0)}",
    )
    _assert_probe(
        "version history available",
        bool((happy_state.get("analysis_evidence", {}).get("guided_version_snapshots") or {}).get("data_preparation")),
        "guided version snapshots are retained for data preparation",
    )
    version_diff = diff_guided_stage_snapshots(modified_state, "data_preparation", 1, 2)
    _assert_probe(
        "version diff",
        version_diff.get("from_version") == 1 and version_diff.get("to_version") == 2 and bool(version_diff.get("changes")),
        f"version diff contains {len(version_diff.get('changes') or [])} change(s)",
    )
    restored_state = _run_silent(
        [
            "modify",
            "use median imputation for Age, keep outliers, do not remove duplicates",
            "continue",
            "continue",
            "continue",
            "continue",
        ]
    ).final_state
    restored_before = restore_guided_stage_snapshot(restored_state, "data_preparation", 1)
    _assert_probe(
        "restore previous version",
        restored_before is not None and restored_state.get("guided_checkpoint_versions", {}).get("data_preparation") == 1,
        "version 1 of the data-preparation stage can be restored",
    )
    _assert_probe(
        "restore has effect",
        (
            restored_state.get("cleaned_data") is not None
            and happy_state.get("cleaned_data") is not None
            and restored_state.get("cleaned_data").equals(happy_state.get("cleaned_data"))
        ),
        "restoring the prior version returns the earlier cleaned dataset snapshot",
    )
    _assert_probe(
        "completed report",
        bool(happy_state.get("final_report")),
        "the final report is available after the happy-path run",
    )
    _assert_probe(
        "report appendix supervision",
        "Workflow supervision:" in _plain_text(happy_state.get("final_report")),
        "the final report appendix includes workflow supervision context",
    )

    print("GUIDED CAPABILITY QUESTION BANK OK")


if __name__ == "__main__":
    test_guided_mode_full_workflow_continue()
    test_guided_mode_full_workflow_modification_and_fallback()
    test_guided_mode_column_specific_cleaning_modification()
    test_issue_detector_does_not_guess_datetime_for_plain_codes()
    test_reasoning_layer_returns_structured_explanation()
    test_guided_mode_cancel_gracefully()
    test_guided_mode_capability_question_bank()
    print("All guided workflow tests passed.")
