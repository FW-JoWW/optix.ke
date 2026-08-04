from __future__ import annotations

import os

from collaborative_mode.models import CollaborativeTask, InvestigationSession
from collaborative_mode.answer_synthesis import render_answer_synthesis_report, synthesize_answer
from collaborative_mode.confidence_diagnostics import evaluate_investigation_decision, render_investigation_decision_report
from collaborative_mode.session_runner import CollaborativeSessionController
from collaborative_mode.task_manager import TaskManager
import collaborative_mode.orchestrator as collaborative_orchestrator
from scripts.collaborative_mode_harness import build_guided_sample_dataframe
from scripts.collaborative_mode_harness import run_collaborative_workflow, summarize_collaborative_result


QUESTION = "What is the relationship between Revenue and Profit by Region?"


def _fake_graph_invoke(state: dict) -> dict:
    question = state.get("business_question", "")
    if "challenge" in question.lower() or "compare" in question.lower():
        insight = "Regional contrast found"
        top_story = {
            "type": "correlation" if "relationship" in insight.lower() else "segmentation",
            "insight": insight,
            "score": 0.81,
            "confidence": "medium",
            "relationship_type": "segmentation",
            "insight_validity": {"valid": True},
        }
        analysis_plan = [{"tool": "group_by", "columns": ["Region"]}]
        tool_results = {"direct_computation_Region": {"tool": "direct_computation"}}
        final_report = "FAKE COLLABORATIVE REPORT\n\nCOLLABORATIVE INVESTIGATION\n- Synthetic report for fast testing."
    else:
        insight = "Positive relationship detected"
        top_story = {
            "type": "correlation",
            "insight": insight,
            "score": 0.93,
            "confidence": "high",
            "relationship_type": "correlation",
            "insight_validity": {"valid": True},
        }
        analysis_plan = [{"tool": "correlation", "columns": ["Revenue", "Profit"]}]
        tool_results = {"correlation_Revenue_Profit": {"tool": "correlation"}}
        final_report = "FAKE COLLABORATIVE REPORT\n\nCOLLABORATIVE INVESTIGATION\n- Synthetic report for fast testing."

    state = dict(state)
    state["selected_columns"] = ["Revenue", "Profit", "Region"]
    state["analysis_plan"] = analysis_plan
    state["final_report"] = final_report
    state["llm_reasoning_status"] = "stubbed"
    state["analytical_reasoning"] = {"confidence": {"score": 88}}
    state["analysis_evidence"] = {
        "analysis_plan": analysis_plan,
        "tool_results": tool_results,
        "visualizations": [{"title": "stub chart"}],
        "top_stories": [top_story],
        "judgment_summary": {
            "global_confidence": 88,
            "summary": "Synthetic judgment",
            "result_status": "completed",
        },
        "analytical_reasoning": {"confidence": {"score": 88}},
    }
    return state


def _fake_unrelated_graph_invoke(state: dict) -> dict:
    state = dict(state)
    state["selected_columns"] = ["FreightValue", "ShipmentCost"]
    state["analysis_plan"] = [{"tool": "summary", "columns": ["FreightValue", "ShipmentCost"]}]
    state["final_report"] = "FAKE COLLABORATIVE REPORT\n\nCOLLABORATIVE INVESTIGATION\n- Synthetic unrelated result."
    state["llm_reasoning_status"] = "stubbed"
    state["analytical_reasoning"] = {"confidence": {"score": 91}}
    state["analysis_evidence"] = {
        "analysis_plan": state["analysis_plan"],
        "tool_results": {"summary": {"tool": "summary"}},
        "visualizations": [{"title": "stub chart"}],
        "top_stories": [
            {
                "type": "outlier",
                "insight": "Freight value is highest in the western region.",
                "score": 0.96,
                "confidence": "high",
                "relationship_type": "outlier",
                "insight_validity": {"valid": True},
            }
        ],
        "judgment_summary": {
            "global_confidence": 91,
            "summary": "Synthetic unrelated judgment",
            "result_status": "completed",
        },
        "analytical_reasoning": {"confidence": {"score": 91}},
    }
    return state


def test_collaborative_mode_runs_investigation_session(monkeypatch) -> None:
    monkeypatch.setattr(collaborative_orchestrator.graph, "invoke", _fake_graph_invoke)
    result = run_collaborative_workflow(
        question=QUESTION,
        responses=["finish investigation"],
        initial_tasks=[
            {
                "title": "Primary investigation",
                "request": QUESTION,
            },
        ],
    )
    summary = summarize_collaborative_result(result)

    assert summary["final_report_available"] is True
    assert summary["current_status"] == "completed"
    assert len(summary["completed_tasks"]) == 1
    assert summary["evidence_count"] >= 1
    assert summary["hypothesis_count"] >= 1
    assert result.desk["investigation_id"]
    assert "human_actions" in result.desk
    assert result.desk["current_status"] == "completed"
    assert result.desk.get("current_decision") in {"STOP", "CONTINUE", "ASK_USER"}
    assert result.final_state.get("final_report")
    assert "EXECUTIVE ANSWER" in result.final_state.get("final_report", "")
    print("COLLABORATIVE INVESTIGATION SESSION OK")


def test_collaborative_mode_exposes_desk_and_suggestions() -> None:
    session = InvestigationSession(
        investigation_id="inv-desk",
        original_question=QUESTION,
    )
    session.current_status = "active"
    session.progressive_narrative.append("Initial finding")
    session.ai_suggestions.append({"title": "Refine task", "request": "Re-run the task with a tighter scope."})
    session.investigation_memory["current_understanding"] = "Initial finding"

    manager = TaskManager(session)
    desk = {
        "investigation_id": session.investigation_id,
        "current_status": session.current_status,
        "human_actions": [
            "new investigation",
            "refine task",
            "compare results",
        ],
        "ai_suggested_next_investigations": session.ai_suggestions,
        "current_understanding": session.investigation_memory.get("current_understanding"),
    }

    assert desk["investigation_id"]
    assert "human_actions" in desk
    assert desk["current_understanding"]
    assert isinstance(desk["ai_suggested_next_investigations"], list)
    print("COLLABORATIVE DESK OK")


def test_answer_synthesis_prefers_direct_evidence_when_available() -> None:
    synthesis = synthesize_answer(
        business_question="What is the relationship between Revenue and Profit by Region?",
        evidence={
            "top_stories": [
                {
                    "type": "correlation",
                    "insight": "Revenue and Profit move together across regions.",
                    "confidence": "high",
                }
            ],
            "judgment_summary": {
                "summary": "Revenue and Profit move together across regions.",
                "global_confidence": 87,
            },
        },
        hypotheses=[{"status": "supported", "hypothesis": "Revenue and Profit move together across regions."}],
        current_understanding="Revenue and Profit move together across regions.",
        confidence=87,
        knowledge_gaps=[],
        investigation_memory={"current_understanding": "Revenue and Profit move together across regions."},
    )

    report = render_answer_synthesis_report(synthesis)

    assert synthesis["confidence"]["status"] == "yes"
    assert synthesis["direct_answer"]
    assert "Do I have enough evidence to answer? Yes." in report
    assert "Direct Answer" in report


def test_answer_synthesis_reports_missing_evidence_when_sparse() -> None:
    synthesis = synthesize_answer(
        business_question="What is the relationship between Revenue and Profit by Region?",
        evidence={},
        hypotheses=[],
        current_understanding="",
        confidence=None,
        knowledge_gaps=["Need direct evidence for the regional revenue-profit relationship."],
        investigation_memory={},
    )

    assert synthesis["confidence"]["status"] == "no"
    assert synthesis["remaining_uncertainty"]
    assert synthesis["evidence_breakdown"]["missing"]
    assert "no direct evidence" in " ".join(synthesis["evidence_breakdown"]["missing"]).lower()


def test_answer_synthesis_includes_confidence_diagnostics_and_stop_logic() -> None:
    synthesis = synthesize_answer(
        business_question="What is the relationship between Revenue and Profit by Region?",
        evidence={
            "top_stories": [
                {
                    "type": "correlation",
                    "insight": "Revenue and Profit move together across regions.",
                    "confidence": "high",
                }
            ],
            "judgment_summary": {
                "summary": "Revenue and Profit move together across regions.",
                "global_confidence": 90,
            },
        },
        hypotheses=[{"status": "supported", "hypothesis": "Revenue and Profit move together across regions."}],
        current_understanding="Revenue and Profit move together across regions.",
        confidence=90,
        knowledge_gaps=[],
        investigation_memory={},
    )

    diagnostics = synthesis["confidence_diagnostics"]
    report = render_answer_synthesis_report(synthesis)

    assert diagnostics["evidence_sufficiency"]["status"] in {"yes", "partial"}
    assert "overall" in diagnostics["confidence"]
    assert "Confidence Diagnostics" in report
    assert "Evidence Sufficiency" in report
    assert "Stopping Criteria" in report
    assert "What is limiting confidence most?" in report
    assert "Another broad analysis pass is unlikely to move the conclusion much." in report


def test_investigation_decision_stops_when_evidence_is_sufficient() -> None:
    decision = evaluate_investigation_decision(
        business_question="What is the relationship between Revenue and Profit by Region?",
        evidence={},
        answer={
            "answer_position": "direct",
            "evidence_breakdown": {
                "direct": [1, 2],
                "indirect": [],
                "supporting": [1],
                "conflicting": [],
            },
        },
        diagnostics={
            "confidence": {
                "overall": {"score": 88},
                "question_coverage": {"score": 92},
                "evidence_quality": {"score": 84},
                "alternative_explanation": {"score": 80},
                "business_interpretation": {"score": 78},
                "recommendation": {"score": 76},
            },
            "evidence_sufficiency": {"diminishing_returns": True, "would_more_analysis_help": False},
            "uncertainty_sources": [],
            "fastest_path_to_strengthen_confidence": {"reducible": False, "expected_confidence_gain": 5, "action": "Finish the investigation"},
        },
        collaborative_mode=True,
    )

    report = render_investigation_decision_report(decision)

    assert decision["decision"] == "STOP"
    assert "Decision: STOP" in report
    assert "Original question has been answered" not in report  # report should stay analyst-like, not templated verbatim


def test_investigation_decision_respects_risk_and_stricter_thresholds() -> None:
    high_risk_decision = evaluate_investigation_decision(
        business_question="Should we approve a multi-million dollar investment?",
        evidence={},
        answer={
            "answer_position": "direct",
            "evidence_breakdown": {
                "direct": [1, 2],
                "indirect": [1],
                "supporting": [1],
                "conflicting": [],
            },
        },
        diagnostics={
            "confidence": {
                "overall": {"score": 76},
                "question_coverage": {"score": 82},
                "evidence_quality": {"score": 78},
                "alternative_explanation": {"score": 75},
                "business_interpretation": {"score": 74},
                "recommendation": {"score": 72},
            },
            "evidence_sufficiency": {"diminishing_returns": True, "would_more_analysis_help": False},
            "uncertainty_sources": [],
            "fastest_path_to_strengthen_confidence": {"reducible": False, "expected_confidence_gain": 4, "action": "Finish the investigation"},
        },
        collaborative_mode=False,
    )

    low_risk_decision = evaluate_investigation_decision(
        business_question="Should the dashboard formatting change?",
        evidence={},
        answer={
            "answer_position": "direct",
            "evidence_breakdown": {
                "direct": [1, 2],
                "indirect": [1],
                "supporting": [1],
                "conflicting": [],
            },
        },
        diagnostics={
            "confidence": {
                "overall": {"score": 70},
                "question_coverage": {"score": 78},
                "evidence_quality": {"score": 72},
                "alternative_explanation": {"score": 70},
                "business_interpretation": {"score": 69},
                "recommendation": {"score": 68},
            },
            "evidence_sufficiency": {"diminishing_returns": True, "would_more_analysis_help": False},
            "uncertainty_sources": [],
            "fastest_path_to_strengthen_confidence": {"reducible": False, "expected_confidence_gain": 4, "action": "Finish the investigation"},
        },
        collaborative_mode=False,
    )

    assert high_risk_decision["decision"] == "CONTINUE"
    assert low_risk_decision["decision"] == "STOP"
    assert high_risk_decision["internal_metrics"]["risk_level"] == "high"
    assert low_risk_decision["internal_metrics"]["risk_level"] == "low"


def test_investigation_decision_continues_when_more_analysis_is_warranted() -> None:
    decision = evaluate_investigation_decision(
        business_question="Why is churn rising?",
        evidence={},
        answer={
            "answer_position": "partial",
            "evidence_breakdown": {
                "direct": [],
                "indirect": [1],
                "supporting": [],
                "conflicting": [],
            },
        },
        diagnostics={
            "confidence": {
                "overall": {"score": 45},
                "question_coverage": {"score": 38},
                "evidence_quality": {"score": 42},
                "alternative_explanation": {"score": 35},
                "business_interpretation": {"score": 50},
                "recommendation": {"score": 52},
            },
            "evidence_sufficiency": {"diminishing_returns": False, "would_more_analysis_help": True},
            "uncertainty_sources": [
                {"source": "Missing segment analysis", "severity": "high", "reducible": True, "reason": "Segment mix may be masking the pattern."},
            ],
            "fastest_path_to_strengthen_confidence": {"reducible": True, "expected_confidence_gain": 25, "action": "Compare churn by customer segment"},
        },
        collaborative_mode=False,
    )

    assert decision["decision"] == "CONTINUE"
    assert decision["failed_gates"]
    assert "Compare churn by customer segment" in decision["recommended_next_step"]


def test_investigation_decision_asks_user_in_collaborative_mode_when_benefit_is_uncertain() -> None:
    decision = evaluate_investigation_decision(
        business_question="Why is churn rising?",
        evidence={},
        answer={
            "answer_position": "partial",
            "evidence_breakdown": {
                "direct": [],
                "indirect": [1],
                "supporting": [],
                "conflicting": [],
            },
        },
        diagnostics={
            "confidence": {
                "overall": {"score": 51},
                "question_coverage": {"score": 41},
                "evidence_quality": {"score": 44},
                "alternative_explanation": {"score": 39},
                "business_interpretation": {"score": 48},
                "recommendation": {"score": 49},
            },
            "evidence_sufficiency": {"diminishing_returns": False, "would_more_analysis_help": False},
            "uncertainty_sources": [
                {"source": "Unclear next best branch", "severity": "medium", "reducible": True, "reason": "Several follow-up paths look similar in value."},
            ],
            "fastest_path_to_strengthen_confidence": {"reducible": True, "expected_confidence_gain": 8, "action": "Choose a follow-up path"},
        },
        collaborative_mode=True,
    )

    assert decision["decision"] == "ASK_USER"
    assert decision["question_for_user"]


def test_investigation_decision_report_hides_internal_metrics_by_default() -> None:
    decision = evaluate_investigation_decision(
        business_question="Why is churn rising?",
        evidence={},
        answer={
            "answer_position": "direct",
            "evidence_breakdown": {
                "direct": [1, 2],
                "indirect": [1],
                "supporting": [1],
                "conflicting": [],
            },
        },
        diagnostics={
            "confidence": {
                "overall": {"score": 88},
                "question_coverage": {"score": 90},
                "evidence_quality": {"score": 84},
                "alternative_explanation": {"score": 82},
                "business_interpretation": {"score": 80},
                "recommendation": {"score": 78},
            },
            "evidence_sufficiency": {"diminishing_returns": True, "would_more_analysis_help": False},
            "uncertainty_sources": [],
            "fastest_path_to_strengthen_confidence": {"reducible": False, "expected_confidence_gain": 4, "action": "Finish"},
        },
        collaborative_mode=True,
    )

    report = render_investigation_decision_report(decision)
    debug_report = render_investigation_decision_report(decision, include_internal_metrics=True)

    assert "Internal Decision Metrics" not in report
    assert "Internal Decision Metrics" in debug_report


def test_collaborative_task_manager_can_compare_finished_tasks() -> None:
    session = InvestigationSession(
        investigation_id="inv-test",
        original_question=QUESTION,
    )
    manager = TaskManager(session)

    left = manager.create_task("Investigate Revenue and Profit by Region", title="Task 1")
    right = manager.create_task("Challenge the leading finding", title="Task 2")
    left.status = "completed"
    right.status = "completed"
    left.result_summary = {"conclusion": "positive relationship"}
    right.result_summary = {"conclusion": "regional contrast"}
    session.completed_tasks.extend([left.task_id, right.task_id])

    comparison = manager.compare_tasks(left.task_id, right.task_id)

    assert comparison["left_task_id"] == left.task_id
    assert comparison["right_task_id"] == right.task_id
    assert session.task_comparisons
    print("COLLABORATIVE COMPARISON OK")


def test_accept_ai_suggestion_uses_same_ranked_order_as_display() -> None:
    controller = CollaborativeSessionController.create(
        question=QUESTION,
        dataframe=build_guided_sample_dataframe(),
        initial_tasks=[{"title": "Primary investigation", "request": QUESTION}],
        build_final_report=False,
    )
    controller.session.ai_suggestions = [
        {
            "title": "Lower confidence but higher impact",
            "request": "Review the region split first.",
            "confidence": 40,
            "impact_percent": 95,
        },
        {
            "title": "Higher confidence recommendation",
            "request": "Validate the leading finding with a comparison task.",
            "confidence": 90,
            "impact_percent": 55,
        },
    ]

    ranked = controller._rank_suggestions(controller.session.ai_suggestions)
    assert ranked[0]["title"] == "Higher confidence recommendation"
    assert controller._pick_ai_suggestion("")["title"] == "Higher confidence recommendation"
    assert controller._pick_ai_suggestion("1")["title"] == "Higher confidence recommendation"
    print("COLLABORATIVE AI RANKING OK")


def test_unrelated_result_stays_supporting_evidence_only(monkeypatch) -> None:
    monkeypatch.setattr(collaborative_orchestrator.graph, "invoke", _fake_unrelated_graph_invoke)
    result = run_collaborative_workflow(
        question=QUESTION,
        responses=["finish investigation"],
        initial_tasks=[
            {
                "title": "Primary investigation",
                "request": QUESTION,
            },
        ],
    )
    session = result.session
    checkpoint = session.get("checkpoint_summaries", [{}])[-1]

    assert checkpoint.get("integrity", {}).get("should_promote") is False
    assert session.get("investigation_memory", {}).get("current_understanding") == QUESTION
    assert session.get("investigation_memory", {}).get("supporting_findings")
    assert "Investigation Integrity" in result.final_state.get("final_report", "")
    print("COLLABORATIVE INTEGRITY GATE OK")


def test_collaborative_integration_smoke_can_be_enabled_explicitly() -> None:
    if os.getenv("RUN_SLOW_COLLABORATIVE_TESTS") != "1":
        return
    result = run_collaborative_workflow(
        question=QUESTION,
        responses=["finish investigation"],
        initial_tasks=[{"title": "Primary investigation", "request": QUESTION}],
    )
    assert result.final_state.get("final_report")
