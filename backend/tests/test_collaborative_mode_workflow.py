from __future__ import annotations

import json
import os

from backend.collaborative_mode.models import CollaborativeTask, EvidenceRecord, InvestigationSession
from backend.collaborative_mode.answer_synthesis import render_answer_synthesis_report, synthesize_answer
from backend.collaborative_mode.confidence_diagnostics import evaluate_investigation_decision, render_investigation_decision_report
from backend.collaborative_mode.narrative_composer import render_analyst_report
from backend.collaborative_mode.session_runner import CollaborativeSessionController
from backend.collaborative_mode.session_runner import _infer_action_from_text
from backend.collaborative_mode.task_manager import TaskManager
import backend.collaborative_mode.orchestrator as collaborative_orchestrator
from backend.collaborative_mode.presentation import render_collaborative_handoff_view
from backend.scripts.collaborative_mode_harness import build_guided_sample_dataframe
from backend.scripts.collaborative_mode_harness import run_collaborative_workflow, summarize_collaborative_result


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


def test_collaborative_mode_handoff_view_reads_like_analyst_brief() -> None:
    session = InvestigationSession(
        investigation_id="inv-handoff",
        original_question=QUESTION,
    )
    session.current_status = "awaiting_user"
    session.investigation_memory["current_understanding"] = "Revenue and Profit appear to move together across regions."
    session.investigation_memory["investigation_decision"] = {
        "decision": "ASK_USER",
        "reasoning": [
            "The current evidence supports a closest-defensible answer, but it still falls short of a direct answer to the business question.",
            "Human guidance is needed to choose the most valuable next investigation path.",
        ],
        "remaining_uncertainties": [
            "The regional comparison still needs a direct validation pass.",
        ],
        "recommended_next_step": "Choose the most valuable next investigation path.",
    }
    session.checkpoint_summaries.append({"task_title": "Regional analysis"})
    session.completed_tasks.append("task-1")
    session.ai_suggestions.append({"title": "Compare segments", "request": "Compare the most important customer segments.", "confidence": 0.9})

    handoff = render_collaborative_handoff_view(session.to_dict())

    assert "ANALYST HANDOFF" in handoff
    assert "What we know:" in handoff
    assert "Why human input is needed now:" in handoff
    assert "What is still uncertain:" in handoff
    assert "Best next actions:" in handoff


def test_collaborative_mode_routes_key_capabilities() -> None:
    cases = {
        "Refine task": "refine task",
        "Accept AI suggestion": "accept ai suggestion",
        "Update our current understanding.": "query",
        "Create a new investigation.": "new investigation",
        "Finish investigation": "finish investigation",
        "What should we investigate next?": "query",
        "Show all active investigations.": "query",
    }

    for phrase, expected_action in cases.items():
        action, _ = _infer_action_from_text(phrase)
        assert action == expected_action, phrase


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
    missing_text = " ".join(synthesis["evidence_breakdown"]["missing"]).lower()
    assert missing_text
    assert "revenue and profit by region" in missing_text or "direct evidence" in missing_text


def test_answer_synthesis_uses_llm_semantic_fallback_when_deterministic_answer_is_indirect(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, content: str) -> None:
            self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    payload = {
                        "answer_status": "insufficient",
                        "direct_answer": "The current evidence does not directly provide a churn rate.",
                        "business_interpretation": "The available evidence describes inactivity gaps, which is related but not the same as churn rate.",
                        "supporting_evidence_summary": ["The evidence tracks customer inactivity and repeat gaps."],
                        "observed_facts": ["Median customer inactivity is 223.0 days."],
                        "analytical_interpretation": ["The evidence is directional only and does not define churn explicitly."],
                        "key_assumptions": ["A churn definition is still missing."],
                        "remaining_uncertainty": ["The dataset does not expose a churn rate or churn flag."],
                        "recommended_next_investigation": ["Measure churn directly using a churn-specific field or definition."],
                        "reasoning": "The evidence supports an inactivity pattern, but churn still needs a direct measure.",
                    }
                    return _FakeResponse(json.dumps(payload))

    monkeypatch.setattr("collaborative_mode.answer_synthesis.get_openai_client", lambda: _FakeClient())

    synthesis = synthesize_answer(
        business_question="What is the churn rate?",
        evidence={
            "top_stories": [
                {
                    "type": "summary_numeric",
                    "insight": "Median customer inactivity is 223.0 days; repeat gaps have a median of 29.0 days",
                    "confidence": "moderate",
                }
            ],
            "judgment_summary": {
                "summary": "Median customer inactivity is 223.0 days; repeat gaps have a median of 29.0 days",
                "global_confidence": 45,
            },
        },
        hypotheses=[],
        current_understanding="Median customer inactivity is 223.0 days; repeat gaps have a median of 29.0 days",
        confidence=45,
        knowledge_gaps=["Need direct evidence that speaks to churn rate."],
        investigation_memory={},
    )

    assert synthesis["semantic_reasoning_status"] == "live_llm"
    assert "does not directly provide a churn rate" in synthesis["direct_answer"].lower()
    assert "churn-specific field" in " ".join(synthesis["recommended_next_investigation"]).lower()


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


def test_investigation_decision_asks_user_when_diminishing_returns_do_not_yet_answer_the_question() -> None:
    decision = evaluate_investigation_decision(
        business_question="What is the relationship between Revenue and Profit by Region?",
        evidence={},
        answer={
            "answer_position": "closest_defensible",
            "evidence_breakdown": {
                "direct": [1],
                "indirect": [],
                "supporting": [1],
                "conflicting": [],
            },
        },
        diagnostics={
            "confidence": {
                "overall": {"score": 86},
                "question_coverage": {"score": 88},
                "evidence_quality": {"score": 82},
                "alternative_explanation": {"score": 80},
                "business_interpretation": {"score": 76},
                "recommendation": {"score": 74},
            },
            "evidence_sufficiency": {"diminishing_returns": True, "would_more_analysis_help": False},
            "uncertainty_sources": [],
            "fastest_path_to_strengthen_confidence": {"reducible": False, "expected_confidence_gain": 4, "action": "Finish the investigation"},
        },
        collaborative_mode=True,
    )

    assert decision["decision"] == "ASK_USER"
    assert "direct answer" in " ".join(decision["reasoning"]).lower()


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


def test_investigation_decision_asks_user_in_collaborative_mode_when_answer_is_still_partial() -> None:
    decision = evaluate_investigation_decision(
        business_question="Why is churn rising?",
        evidence={},
        answer={
            "answer_position": "closest_defensible",
            "evidence_breakdown": {
                "direct": [1],
                "indirect": [1],
                "supporting": [1],
                "conflicting": [],
            },
        },
        diagnostics={
            "confidence": {
                "overall": {"score": 62},
                "question_coverage": {"score": 58},
                "evidence_quality": {"score": 60},
                "alternative_explanation": {"score": 57},
                "business_interpretation": {"score": 59},
                "recommendation": {"score": 58},
            },
            "evidence_sufficiency": {"diminishing_returns": False, "would_more_analysis_help": True},
            "uncertainty_sources": [
                {"source": "Need a clearer business interpretation", "severity": "medium", "reducible": True, "reason": "The evidence still needs analyst judgment to become decision-ready."},
            ],
            "fastest_path_to_strengthen_confidence": {"reducible": True, "expected_confidence_gain": 18, "action": "Review the evidence with the analyst"},
        },
        collaborative_mode=True,
    )

    assert decision["decision"] == "ASK_USER"
    assert "direct answer" in " ".join(decision["reasoning"]).lower()


def test_investigation_decision_asks_user_when_answer_text_is_still_provisional() -> None:
    decision = evaluate_investigation_decision(
        business_question="What is the churn rate?",
        evidence={},
        answer={
            "answer_position": "direct",
            "direct_answer": "The churn rate cannot yet be answered.",
            "business_interpretation": "The evidence provided does not directly address the churn rate.",
            "remaining_uncertainty": ["The answer is still indirect."],
            "evidence_breakdown": {
                "direct": [1],
                "indirect": [1],
                "supporting": [1],
                "conflicting": [],
            },
        },
        diagnostics={
            "confidence": {
                "overall": {"score": 64},
                "question_coverage": {"score": 58},
                "evidence_quality": {"score": 61},
                "alternative_explanation": {"score": 58},
                "business_interpretation": {"score": 59},
                "recommendation": {"score": 57},
            },
            "evidence_sufficiency": {"diminishing_returns": False, "would_more_analysis_help": True},
            "uncertainty_sources": [
                {"source": "Indirect answer", "severity": "medium", "reducible": True, "reason": "The answer is still not direct enough for closure."},
            ],
            "fastest_path_to_strengthen_confidence": {"reducible": True, "expected_confidence_gain": 18, "action": "Ask the analyst whether the current proxy is acceptable"},
        },
        collaborative_mode=True,
    )

    assert decision["decision"] == "ASK_USER"
    assert any("provisional" in reason.lower() or "direct answer" in reason.lower() for reason in decision["reasoning"])


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


def test_collaborative_query_router_understands_more_capability_language() -> None:
    controller = CollaborativeSessionController.create(
        question=QUESTION,
        dataframe=build_guided_sample_dataframe(),
        initial_tasks=[{"title": "Primary investigation", "request": QUESTION}],
        build_final_report=False,
    )
    controller.session.evidence_store["task-1-evidence"] = EvidenceRecord(
        evidence_id="task-1-evidence",
        task_source="task-1",
        evidence_type="task_result",
        statement="Revenue and Profit move together across regions.",
        confidence=0.92,
        quality_score=0.91,
    )

    assert "What to investigate next" in controller.respond_to_query("Suggest three new analyses")
    assert "Supported hypotheses" in controller.respond_to_query("Rank the hypotheses by confidence")
    assert "Current understanding" in controller.respond_to_query("What did we conclude earlier?")
    assert "Evidence count" in controller.respond_to_query("Show the strongest evidence")


def test_collaborative_action_router_accepts_imperative_capability_text() -> None:
    assert _infer_action_from_text("Create a new investigation about churn")[0] == "new investigation"
    assert _infer_action_from_text("Resume previous investigation inv-20260804-123456")[0] == "resume previous investigation"
    assert _infer_action_from_text("Queue three investigations")[0] == "queue three investigations"


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


def test_final_report_prefers_best_answer_anchor_over_late_drift() -> None:
    best_answer = "fixed telephony has the highest payment_value_total and health_beauty the lowest."
    drift_answer = "2017-09-29 15:24:00 has the highest payment_value_total and 2018-03-03 17:11:00 the lowest."
    session = InvestigationSession(
        investigation_id="inv-anchor",
        original_question="Which product category has the highest and lowest payment value?",
    )
    session.investigation_memory["best_answer"] = {
        "task_id": "task-1",
        "task_title": "Primary investigation",
        "answer": best_answer,
        "question_relevance": 92,
        "overall_integrity": 91,
    }
    session.investigation_memory["current_understanding"] = drift_answer
    session.investigation_memory["investigation_decision"] = {
        "decision": "STOP",
        "recommended_next_step": "Finish the investigation.",
        "reasoning": ["The original question is already answered."],
        "remaining_uncertainties": [],
    }

    session_dict = session.to_dict()
    evidence = {
        "collaborative_session": session_dict,
        "top_stories": [
            {
                "type": "ranking",
                "insight": drift_answer,
                "confidence": "high",
            }
        ],
        "judgment_summary": {
            "summary": drift_answer,
            "global_confidence": 91,
        },
    }
    state = {
        "mode": "collaborative",
        "business_question": session.original_question,
        "collaborative_session": session_dict,
        "analysis_evidence": evidence,
    }
    report = render_analyst_report(state, evidence)

    assert best_answer in report
    assert f"Direct Answer\n- {best_answer}" in report
    print("COLLABORATIVE ANSWER ANCHOR OK")


def test_collaborative_integration_smoke_can_be_enabled_explicitly() -> None:
    if os.getenv("RUN_SLOW_COLLABORATIVE_TESTS") != "1":
        return
    result = run_collaborative_workflow(
        question=QUESTION,
        responses=["finish investigation"],
        initial_tasks=[{"title": "Primary investigation", "request": QUESTION}],
    )
    assert result.final_state.get("final_report")


