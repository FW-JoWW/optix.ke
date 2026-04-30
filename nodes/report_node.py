from __future__ import annotations

from typing import Any, Dict, List

from state.state import AnalystState
from nodes.visualization_generator_node import derive_decision_from_top_stories


def _normalize_questions(questions: List[str]) -> List[str]:
    return [
        q for q in questions
        if isinstance(q, str) and q.strip().lower() not in {"", "none", "n/a", "no", "no questions"}
    ]


def _format_analysis_plan(analysis_plan: List[Any]) -> str:
    if not analysis_plan:
        return "None"
    return "\n".join(
        f"- {p}" if isinstance(p, str) else f"- {p.get('tool')} on columns: {', '.join(p.get('columns', []))}"
        for p in analysis_plan
    )


def _format_tool_results(tool_results: Dict[str, Any]) -> str:
    if not tool_results:
        return "None"

    lines: List[str] = []
    for key, result in tool_results.items():
        if not isinstance(result, dict):
            lines.append(f"- {key}: {result}")
            continue

        tool = result.get("tool", key)
        if tool == "correlation":
            lines.append(
                f"- {result.get('column_1')} vs {result.get('column_2')}: correlation={round(float(result.get('correlation', 0)), 4)}"
            )
        elif tool == "ttest":
            lines.append(
                f"- {result.get('column')} by {result.get('group_column')}: p-value={result.get('p_value')}"
            )
        elif tool == "anova":
            lines.append(
                f"- {result.get('numeric_column')} by {result.get('categorical_column')}: p-value={result.get('p_value')}"
            )
        elif tool == "detect_outliers":
            lines.append(
                f"- {result.get('column')}: {result.get('outlier_count', 0)} outliers detected"
            )
        elif tool == "inferential_analysis":
            hypothesis = result.get("hypothesis_test", {}) or {}
            effect = result.get("effect_size", {}) or {}
            causal = result.get("causal_evidence", {}) or {}
            next_step = result.get("recommended_next_step")
            lines.append(
                "- "
                + f"{result.get('method_selected')} on {', '.join(result.get('columns', []))}: "
                + f"p-value={hypothesis.get('p_value')}, decision={hypothesis.get('decision')}, "
                + f"effect={effect.get('metric')}={effect.get('value')}"
                + (f", causal_evidence={causal.get('grade')}({causal.get('score')})" if causal else "")
                + (f", next_step={next_step}" if next_step else "")
            )
        elif tool == "categorical_analysis":
            result_keys = list((result.get("results") or {}).keys())
            lines.append(f"- categorical analysis on: {', '.join(result_keys) if result_keys else 'no columns'}")
        else:
            lines.append(f"- {tool}: {result}")

    return "\n".join(lines)


def _format_computation_plan(computation_plan: Dict[str, Any]) -> str:
    if not computation_plan:
        return "None"
    steps = computation_plan.get("steps", [])
    if not steps:
        return "None"
    lines = []
    for step in steps:
        op = step.get("operation")
        col = step.get("column")
        cols = step.get("columns", [])
        params = step.get("parameters", {})
        target = col or ", ".join(cols)
        if params:
            lines.append(f"- {op} on {target}: {params}")
        else:
            lines.append(f"- {op} on {target}")
    return "\n".join(lines)


def _format_top_stories(top_stories: List[Dict[str, Any]]) -> str:
    if not top_stories:
        return "None"

    return "\n".join(
        f"- {story.get('insight')} "
        f"(score={story.get('score')}, confidence={story.get('confidence', 'unknown')}, "
        f"relationship_type={story.get('relationship_type', 'n/a')}, "
        f"valid={((story.get('insight_validity') or {}).get('valid'))})"
        for story in top_stories
    )


def _format_visualizations(visualizations: List[Dict[str, Any]]) -> str:
    if not visualizations:
        return "None"

    return "\n".join(
        f"- {viz.get('type')} -> {viz.get('file_path')} | {viz.get('caption') or 'No caption'}"
        for viz in visualizations
    )


def _format_decisions(decisions: List[Dict[str, Any]]) -> str:
    if not decisions:
        return "None"
    return "\n".join(
        f"- {item.get('recommended_action')} "
        f"(type={item.get('action_type')}, priority={((item.get('priority') or {}).get('priority_score'))}, "
        f"impact={((item.get('impact_assessment') or {}).get('impact_level'))}, "
        f"confidence={item.get('confidence_in_action')})"
        for item in decisions
    )


def _format_all_decision_records(decisions: List[Dict[str, Any]]) -> str:
    if not decisions:
        return "None"
    return "\n".join(
        f"- {item.get('action_type')}: {item.get('recommended_action')} "
        f"(validity={item.get('validity')}, summary={item.get('decision_summary')})"
        for item in decisions
    )


def report_node(state: AnalystState) -> AnalystState:
    """
    Generates a professional report that reads like a finished workflow output.
    """
    evidence = state.get("analysis_evidence", {})
    business_question = state.get("business_question", "N/A")
    llm_insights = state.get("llm_insights") or evidence.get("llm_insights") or "None"
    clarification_questions: List[str] = (
        state.get("clarification_questions")
        or evidence.get("clarification_questions")
        or []
    )
    clarification_questions = _normalize_questions(clarification_questions)
    analysis_plan = state.get("analysis_plan") or evidence.get("analysis_plan") or []
    computation_plan = evidence.get("computation_plan") or {}
    tool_results = evidence.get("tool_results", {})
    top_stories = evidence.get("top_stories", [])
    visualizations = evidence.get("visualizations", [])
    decision_recommendations = evidence.get("decision_recommendations", []) or []
    decision_ranking = evidence.get("decision_priority_ranking", []) or []
    recommended_first = evidence.get("decision_recommended_first")
    decision_context = state.get("decision_context")
    if not decision_context:
        decision_context = derive_decision_from_top_stories(top_stories)
    human_in_loop = evidence.get("human_in_loop")
    decision_notes = evidence.get("decision_notes", [])

    hitl_summary = "None"
    if human_in_loop:
        hitl_summary = f"{human_in_loop.get('mode')}: {human_in_loop.get('action')}"

    report = f"""
================ EXECUTIVE REPORT ================

BUSINESS QUESTION:
{business_question}

ANALYSIS PLAN:
{_format_analysis_plan(analysis_plan)}

COMPUTATION PLAN:
{_format_computation_plan(computation_plan)}

STATISTICAL OUTPUT SUMMARY:
{_format_tool_results(tool_results)}

TOP STORY CANDIDATES:
{_format_top_stories(top_stories)}

BUSINESS INTERPRETATION:
{llm_insights}

VISUALIZATIONS:
{_format_visualizations(visualizations)}

DECISION CONTEXT:
{decision_context}

ALL DECISION RECORDS:
{_format_all_decision_records(decision_recommendations)}

PRIORITIZED DECISIONS:
{_format_decisions(decision_ranking)}

RECOMMENDED FIRST ACTION:
{recommended_first if recommended_first else "None"}

DECISION ENGINE NOTES:
{chr(10).join(f"- {note}" for note in decision_notes) if decision_notes else "None"}

HUMAN IN LOOP:
{hitl_summary}

CLARIFICATION QUESTIONS:
{chr(10).join(f"- {q}" for q in clarification_questions) if clarification_questions else "None"}

=================================================
"""

    state["final_report"] = report

    print("\n===== FINAL REPORT =====")
    print(report)

    return state
