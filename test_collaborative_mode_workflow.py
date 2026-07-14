from __future__ import annotations

from scripts.collaborative_mode_harness import run_collaborative_workflow, summarize_collaborative_result


QUESTION = "What is the relationship between Revenue and Profit by Region?"


def test_collaborative_mode_runs_investigation_session() -> None:
    result = run_collaborative_workflow(
        question=QUESTION,
        responses=["new investigation", "finish investigation"],
        initial_tasks=[
            {
                "title": "Primary investigation",
                "request": QUESTION,
            },
            {
                "title": "Regional follow-up",
                "request": "Compare Revenue and Profit by Region and challenge the leading conclusion.",
            },
        ],
    )
    summary = summarize_collaborative_result(result)

    assert summary["final_report_available"] is True
    assert summary["current_status"] == "completed"
    assert len(summary["completed_tasks"]) == 2
    assert summary["evidence_count"] >= 2
    assert summary["hypothesis_count"] >= 2
    assert result.final_state.get("final_report")
    assert "COLLABORATIVE INVESTIGATION" in result.final_state.get("final_report", "")
    print("COLLABORATIVE INVESTIGATION SESSION OK")


def test_collaborative_mode_exposes_desk_and_suggestions() -> None:
    result = run_collaborative_workflow(
        question=QUESTION,
        responses=["new investigation", "finish investigation"],
    )
    summary = summarize_collaborative_result(result)
    desk = result.desk

    assert desk["investigation_id"]
    assert "human_actions" in desk
    assert summary["current_understanding"]
    assert isinstance(summary["ai_suggestions"], list)
    print("COLLABORATIVE DESK OK")
