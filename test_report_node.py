from __future__ import annotations

import pandas as pd

from nodes.report_node import report_node
from state.state import AnalystState


def test_report_node_builds_master_and_executive_reports() -> None:
    state: AnalystState = {
        "business_question": "Reduce churn in premium customers",
        "dataset_path": "data/sample.csv",
        "mode": "autonomous",
        "awaiting_user": False,
        "question_for_user": "",
        "user_response": "",
        "enable_llm_reasoning": False,
        "disable_llm_reasoning": True,
        "disable_semantic_matcher": True,
        "analysis_dataset": pd.DataFrame({"revenue": [100, 90, 80], "churn": [0, 1, 1]}),
        "selected_columns": ["revenue", "churn"],
        "dataset_profile": {
            "row_count": 3,
            "column_count": 2,
            "numeric_columns": ["revenue", "churn"],
            "categorical_columns": [],
        },
        "decision_context": {"decision": "reduce churn through retention interventions"},
        "analysis_evidence": {
            "analysis_plan": [{"tool": "correlation", "columns": ["revenue", "churn"]}],
            "computation_plan": {
                "steps": [
                    {
                        "operation": "correlation",
                        "columns": ["revenue", "churn"],
                        "parameters": {"method": "pearson"},
                    }
                ]
            },
            "tool_results": {
                "corr_1": {
                    "tool": "correlation",
                    "column_1": "revenue",
                    "column_2": "churn",
                    "correlation": -0.82,
                }
            },
            "top_stories": [
                {
                    "type": "correlation",
                    "insight": "revenue and churn show a strong negative pattern",
                    "columns": ["revenue", "churn"],
                    "value": -0.82,
                    "confidence": "high",
                    "score": 0.91,
                    "insight_validity": {"valid": True},
                    "semantic_reasoning": {"relationship_type": "observational"},
                    "score_components": {"base_score": 0.9, "relevance_multiplier": 1.0},
                }
            ],
            "visualizations": [
                {"type": "scatter", "file_path": "charts/revenue_churn.png", "caption": "Revenue vs churn"}
            ],
            "decision_recommendations": [
                {
                    "recommended_action": "Launch a retention pilot for premium customers",
                    "action_type": "optimization",
                    "priority": {"priority_score": 82, "priority_level": "high"},
                    "impact_assessment": {
                        "impact_level": "high",
                        "estimated_direction": "positive",
                    },
                    "confidence_in_action": 76,
                    "monitoring_kpis": ["churn_rate", "repeat_purchase_rate"],
                }
            ],
            "decision_priority_ranking": [
                {
                    "recommended_action": "Launch a retention pilot for premium customers",
                    "action_type": "optimization",
                    "priority": {"priority_score": 82, "priority_level": "high"},
                    "impact_assessment": {
                        "impact_level": "high",
                        "estimated_direction": "positive",
                    },
                    "confidence_in_action": 76,
                    "monitoring_kpis": ["churn_rate", "repeat_purchase_rate"],
                }
            ],
            "decision_recommended_first": {
                "recommended_action": "Launch a retention pilot for premium customers",
                "impact_assessment": {"impact_level": "high"},
            },
            "judgment_summary": {
                "dominant_reasoning": "A predictive signal is available for planning, but it should be used as decision support rather than interpreted as causal evidence.",
                "global_confidence": 79,
                "actionability": True,
                "allowed_actions": ["planning", "monitoring"],
                "blocked_actions": [],
                "contradictions_found": [],
                "recommended_first_action": "Launch a retention pilot for premium customers",
            },
            "llm_insights": "Revenue appears to be a useful planning signal for churn.",
            "clarification_questions": [],
        },
    }

    final_state = report_node(state)
    report = final_state["final_report"]

    assert "================ EXECUTIVE REPORT ================" in report
    assert "================ MASTER REPORT ================" in report
    assert "EXECUTIVE SUMMARY" in report
    assert "BUSINESS ANALYSIS REPORT" in report
    assert "OPPORTUNITY ASSESSMENT" in report
    assert "MONITORING AND SUCCESS METRICS" in report
    assert final_state["master_report"].startswith("================ MASTER REPORT ================")
    assert final_state["executive_report"].startswith("================ EXECUTIVE REPORT ================")
    assert final_state["analysis_evidence"]["report_package"]["traceability"]["business_question"] == "Reduce churn in premium customers"


if __name__ == "__main__":
    test_report_node_builds_master_and_executive_reports()
    print("Report node test passed.")
