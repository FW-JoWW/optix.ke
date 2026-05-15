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
        elif tool == "predictive_analysis":
            metrics = ((result.get("metrics") or {}).get("values")) or {}
            confidence = result.get("confidence", {}) or {}
            monitoring = ((result.get("validation_summary") or {}).get("monitoring")) or {}
            lines.append(
                f"- predictive {result.get('problem_type')} for {result.get('target_column')} using {result.get('chosen_model')}: metrics={metrics}, confidence={confidence.get('label')}({confidence.get('score')}), health={monitoring.get('health_score')}"
            )
        elif tool == "prescriptive_analysis":
            lines.append(
                f"- prescriptive analysis for {result.get('based_on_target')}: estimated_upside={result.get('estimated_upside')}, objective={result.get('objective')}"
            )
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


def _format_judgment_summary(judgment: Dict[str, Any]) -> str:
    if not judgment:
        return "None"
    return "\n".join(
        [
            f"- Dominant interpretation: {judgment.get('dominant_reasoning')}",
            f"- Global confidence: {judgment.get('global_confidence')}",
            f"- Actionability: {judgment.get('actionability')}",
            f"- Allowed actions: {judgment.get('allowed_actions') or []}",
            f"- Blocked actions: {judgment.get('blocked_actions') or []}",
        ]
    )


def _executive_summary(evidence: Dict[str, Any], judgment: Dict[str, Any]) -> str:
    first_decision = evidence.get("decision_recommended_first") or {}
    top_story = (evidence.get("top_stories") or [{}])[0]
    return "\n".join(
        [
            f"- Best action: {first_decision.get('recommended_action', 'None')}",
            f"- Expected upside: {((first_decision.get('impact_assessment') or {}).get('impact_level', 'unknown'))}",
            f"- Confidence level: {judgment.get('global_confidence', 'unknown')}",
            f"- Core evidence: {top_story.get('insight', 'No strong story identified')}",
        ]
    )


def _first_tool_result(tool_results: Dict[str, Any], tool_name: str) -> Dict[str, Any] | None:
    return next(
        (
            value
            for value in tool_results.values()
            if isinstance(value, dict) and value.get("tool") == tool_name
        ),
        None,
    )


def _predictive_sections(tool_results: Dict[str, Any]) -> Dict[str, str]:
    predictive = _first_tool_result(tool_results, "predictive_analysis")
    prescriptive = _first_tool_result(tool_results, "prescriptive_analysis")
    if not predictive and not prescriptive:
        return {
            "drivers": "None",
            "risks": "None",
            "best_action": "None",
            "expected_upside": "None",
            "constraints": "None",
            "confidence": "None",
            "monitoring_plan": "None",
            "action_paths": "None",
            "assumption_stress": "None",
            "reliability": "None",
            "operational_sensitivity": "None",
            "dependency_risks": "None",
            "experiment_design": "None",
            "early_warnings": "None",
            "what_could_go_wrong": "None",
        }

    top_drivers = predictive.get("top_drivers", []) if predictive else []
    drivers = "\n".join(
        f"- {item.get('feature')}: importance={item.get('importance')}"
        for item in top_drivers[:5]
    ) if top_drivers else "None"
    limitations = (predictive.get("limitations", []) if predictive else []) or []
    truthfulness = (prescriptive.get("truthfulness_notes", []) if prescriptive else []) or []
    monitoring = ((predictive.get("validation_summary") or {}).get("monitoring") if predictive else {}) or {}
    performance_decay = ((predictive.get("validation_summary") or {}).get("performance_decay") if predictive else {}) or {}
    confidence = predictive.get("confidence", {}) if predictive else {}
    risks_payload: List[str] = []
    risks_payload.extend(limitations[:5])
    risks_payload.extend(truthfulness[:3])
    risks_payload.extend(monitoring.get("warnings", [])[:3])
    risks_payload.extend(performance_decay.get("warnings", [])[:2])
    if confidence:
        risks_payload.append(f"Confidence explanation: {confidence.get('explanation')}")
    risks = "\n".join(f"- {item}" for item in risks_payload) if risks_payload else "None"
    best_action = "None"
    expected_upside = "None"
    constraints = "None"
    action_paths = "None"
    if prescriptive:
        decision_paths = prescriptive.get("decision_paths", []) or []
        if decision_paths:
            best_action = decision_paths[0].get("recommended_action", "None")
            expected_upside = str(decision_paths[0].get("expected_impact_range", prescriptive.get("estimated_upside")))
            constraints = "\n".join(
                f"- {item}" for item in (decision_paths[0].get("constraint_explanations", []) or [])
            ) or "None"
            action_paths = "\n".join(
                f"- {item.get('scenario')}: {item.get('recommended_action')} | score={item.get('score')} | tradeoffs={item.get('tradeoffs')}"
                for item in decision_paths[:4]
            ) or "None"
        else:
            actions = prescriptive.get("recommended_actions", []) or []
            if actions:
                best_action = actions[0].get("action", "None")
                expected_upside = str(actions[0].get("estimated_uplift_range") or actions[0].get("estimated_uplift") or prescriptive.get("estimated_upside"))
                action_paths = "\n".join(
                    f"- {item.get('action')} | risk={item.get('risk_level')} | feasibility={item.get('feasibility')}"
                    for item in actions[:4]
                ) or "None"
    confidence = "None"
    if predictive and predictive.get("confidence"):
        confidence = (
            f"{predictive['confidence'].get('label')} "
            f"({predictive['confidence'].get('score')}) - "
            f"{predictive['confidence'].get('explanation')}"
        )
    monitoring_plan = "\n".join(f"- {item}" for item in monitoring.get("monitoring_plan", [])[:5]) if monitoring.get("monitoring_plan") else "None"
    driver_diag = ((predictive.get("validation_summary") or {}).get("driver_diagnostics") if predictive else {}) or {}
    assumption_stress = "\n".join(
        f"- {item}" for item in ((prescriptive.get("assumptions", []) if prescriptive else [])[:4])
    ) if prescriptive and prescriptive.get("assumptions") else "None"
    reliability = "None"
    operational_sensitivity = "None"
    dependency_risks = "None"
    experiment_design = "None"
    early_warnings = "None"
    what_could_go_wrong = "None"
    if prescriptive:
        operational_confidence = (prescriptive.get("operational_confidence") or {}) if isinstance(prescriptive, dict) else {}
        if operational_confidence:
            reliability = (
                f"- Predictive confidence: {confidence}\n"
                f"- Operational confidence: {operational_confidence.get('label')} ({operational_confidence.get('score')}) - {operational_confidence.get('explanation')}"
            )
        best_path = ((prescriptive.get("decision_paths") or [{}])[0]) or {}
        operational_sensitivity = "\n".join(
            f"- {item}" for item in ((best_path.get("constraint_explanations", []) or []) + (best_path.get("downside_risks", []) or []))[:6]
        ) or "None"
        experiment_design = (
            "- Start with a phased rollout on the affected segments.\n"
            "- Compare results with a holdout or pre-change baseline.\n"
            f"- Monitor {', '.join(best_path.get('monitoring_kpis', [])[:4]) if best_path.get('monitoring_kpis') else 'target KPI and customer response indicators'}."
        )
        early_warning_items = (best_path.get("failure_conditions", []) or []) + (best_path.get("monitoring_kpis", []) or [])
        early_warnings = "\n".join(f"- {item}" for item in early_warning_items[:6]) if early_warning_items else "None"
        what_could_go_wrong = "\n".join(f"- {item}" for item in (best_path.get("downside_risks", []) or [])[:6]) or "None"
    if predictive and driver_diag:
        dependency_lines = [
            f"- Top driver share: {driver_diag.get('top_driver_share')}",
            f"- Driver concentration score: {driver_diag.get('driver_concentration_score')}",
            f"- Signal diversity: {driver_diag.get('signal_diversity')}",
        ]
        dependency_lines.extend(f"- {item}" for item in driver_diag.get("warnings", [])[:4])
        dependency_risks = "\n".join(dependency_lines)
    return {
        "drivers": drivers,
        "risks": risks,
        "best_action": best_action,
        "expected_upside": expected_upside,
        "constraints": constraints,
        "confidence": confidence,
        "monitoring_plan": monitoring_plan,
        "action_paths": action_paths,
        "assumption_stress": assumption_stress,
        "reliability": reliability,
        "operational_sensitivity": operational_sensitivity,
        "dependency_risks": dependency_risks,
        "experiment_design": experiment_design,
        "early_warnings": early_warnings,
        "what_could_go_wrong": what_could_go_wrong,
    }


def _format_contradictions(judgment: Dict[str, Any]) -> str:
    contradictions = (judgment or {}).get("contradictions_found", []) or []
    if not contradictions:
        return "None"
    return "\n".join(f"- {item}" for item in contradictions)


def _format_recommendation_paths(tool_results: Dict[str, Any]) -> str:
    sections = _predictive_sections(tool_results)
    return sections.get("action_paths", "None")


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
    judgment = evidence.get("judgment_summary", {}) or {}
    decision_context = state.get("decision_context")
    if not decision_context:
        decision_context = derive_decision_from_top_stories(top_stories)
    human_in_loop = evidence.get("human_in_loop")
    decision_notes = evidence.get("decision_notes", [])
    predictive_sections = _predictive_sections(tool_results)

    hitl_summary = "None"
    if human_in_loop:
        hitl_summary = f"{human_in_loop.get('mode')}: {human_in_loop.get('action')}"

    report = f"""
================ EXECUTIVE REPORT ================

BUSINESS QUESTION:
{business_question}

EXECUTIVE SUMMARY:
{_executive_summary(evidence, judgment)}

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

KEY DRIVERS:
{predictive_sections["drivers"]}

RISKS:
{predictive_sections["risks"]}

BEST ACTION:
{predictive_sections["best_action"]}

EXPECTED UPSIDE:
{predictive_sections["expected_upside"]}

CONSTRAINTS:
{predictive_sections["constraints"]}

CONFIDENCE LEVEL:
{predictive_sections["confidence"]}

MONITORING PLAN:
{predictive_sections["monitoring_plan"]}

ACTION PATHS:
{_format_recommendation_paths(tool_results)}

EXECUTIVE RISKS:
{predictive_sections["risks"]}

ASSUMPTION STRESS POINTS:
{predictive_sections["assumption_stress"]}

RECOMMENDATION RELIABILITY:
{predictive_sections["reliability"]}

OPERATIONAL SENSITIVITY:
{predictive_sections["operational_sensitivity"]}

MODEL DEPENDENCY RISKS:
{predictive_sections["dependency_risks"]}

RECOMMENDED EXPERIMENT DESIGN:
{predictive_sections["experiment_design"]}

EARLY WARNING INDICATORS:
{predictive_sections["early_warnings"]}

WHAT COULD GO WRONG:
{predictive_sections["what_could_go_wrong"]}

JUDGMENT SUMMARY:
{_format_judgment_summary(judgment)}

CONTRADICTIONS RESOLVED:
{_format_contradictions(judgment)}

ALL DECISION RECORDS:
{_format_all_decision_records(decision_recommendations)}

PRIORITIZED DECISIONS:
{_format_decisions(decision_ranking)}

RECOMMENDED FIRST ACTION:
{judgment.get("recommended_first_action") if judgment else (recommended_first if recommended_first else "None")}

DECISION ENGINE NOTES:
{chr(10).join(f"- {note}" for note in ([*(judgment.get('judgment_notes', []) if judgment else []), *decision_notes])) if ((judgment.get('judgment_notes', []) if judgment else []) or decision_notes) else "None"}

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
