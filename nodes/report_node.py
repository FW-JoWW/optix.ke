from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from state.state import AnalystState
from nodes.visualization_generator_node import derive_decision_from_top_stories


_EMPTY_MARKERS = {"", "none", "n/a", "na", "null", "no", "no questions"}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        return "" if text.lower() in _EMPTY_MARKERS else text
    text = str(value).strip()
    return "" if text.lower() in _EMPTY_MARKERS else text


def _as_text_list(values: Any, limit: int | None = None) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        values = [values]

    output: List[str] = []
    for item in values:
        if isinstance(item, dict):
            text = ", ".join(
                f"{key}: {value}"
                for key, value in item.items()
                if _normalize_text(value)
            )
        else:
            text = _normalize_text(item)
        if text:
            output.append(text)
        if limit is not None and len(output) >= limit:
            break
    return output


def _to_lines(items: Any, limit: int | None = None) -> List[str]:
    """
    Backward-compatible alias for older report-building helpers.
    """
    return _as_text_list(items, limit=limit)


def _format_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, dict):
        if not value:
            return "None"
        flat_parts = []
        for key, item in list(value.items())[:8]:
            normalized = _normalize_text(item)
            if normalized:
                flat_parts.append(f"{key}: {normalized}")
        return "; ".join(flat_parts) if flat_parts else "None"
    if isinstance(value, (list, tuple, set)):
        items = _as_text_list(value, limit=8)
        return ", ".join(items) if items else "None"
    text = _normalize_text(value)
    return text or "None"


def _bullet_block(lines: Sequence[str]) -> str:
    clean = [line for line in lines if _normalize_text(line)]
    if not clean:
        return "None"
    return "\n".join(f"- {line}" for line in clean)


def _section(title: str, lines: Sequence[str]) -> str | None:
    body = _bullet_block(lines)
    if body == "None":
        return None
    return f"{title}\n{body}"


def _section_items(section_text: str | None, limit: int | None = None) -> List[str]:
    if not section_text:
        return []
    items = []
    for line in section_text.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        normalized = _normalize_text(line)
        if normalized:
            items.append(normalized)
        if limit is not None and len(items) >= limit:
            break
    return items


def _section_map(*sections: Tuple[str, Sequence[str]]) -> Dict[str, str]:
    mapped: Dict[str, str] = {}
    for title, lines in sections:
        section = _section(title, lines)
        if section:
            mapped[title] = section
    return mapped


def _normalize_questions(questions: List[str]) -> List[str]:
    return [
        q
        for q in questions
        if isinstance(q, str) and q.strip().lower() not in _EMPTY_MARKERS
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


def _confidence_label(score: Any, fallback: Any = None) -> str:
    numeric = None
    if isinstance(score, dict):
        numeric = score.get("score")
        label = score.get("label") or score.get("level") or score.get("confidence")
        if label and numeric is not None:
            return f"{label} ({numeric})"
        if label:
            return str(label)
    elif isinstance(score, (int, float)):
        numeric = float(score)
    elif isinstance(score, str) and score.strip():
        return score.strip()

    if numeric is None and isinstance(fallback, (int, float)):
        numeric = float(fallback)
    if numeric is None:
        return "unknown"

    if numeric <= 33:
        label = "low"
    elif numeric <= 66:
        label = "medium"
    else:
        label = "high"
    if numeric <= 1:
        numeric = round(numeric * 100, 2)
    return f"{label} ({numeric})"


def _story_confidence(story: Dict[str, Any]) -> str:
    if not story:
        return "unknown"
    value = story.get("confidence")
    score = story.get("score")
    if isinstance(score, (int, float)) and score <= 1:
        score = round(float(score) * 100, 2)
    elif isinstance(score, (int, float)):
        score = round(float(score), 2)
    return _confidence_label(value, score)


def _story_sample_size(story: Dict[str, Any]) -> int | None:
    candidates = [
        story.get("sample_size"),
        ((story.get("estimation") or {}).get("sample_size")) if isinstance(story.get("estimation"), dict) else None,
        ((story.get("confidence_assessment") or {}).get("sample_size")) if isinstance(story.get("confidence_assessment"), dict) else None,
        ((story.get("operational_confidence_assessment") or {}).get("sample_size")) if isinstance(story.get("operational_confidence_assessment"), dict) else None,
        (((story.get("semantic_reasoning") or {}).get("details") or {}).get("sample_size")) if isinstance(story.get("semantic_reasoning"), dict) else None,
        (((story.get("stats") or {}).get("sample_size")) if isinstance(story.get("stats"), dict) else None),
    ]
    for candidate in candidates:
        try:
            if candidate is not None:
                numeric = int(candidate)
                if numeric > 0:
                    return numeric
        except Exception:
            continue
    return None


def _confidence_context_lines(
    story: Dict[str, Any],
    evidence: Dict[str, Any],
    judgment: Dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    if not story:
        return ["Confidence: unknown", "Reason: no primary story was available to assess."]

    confidence_text = _story_confidence(story)
    lines.append(f"Confidence: {confidence_text}")

    reasons: List[str] = []
    score = story.get("confidence_assessment") or story.get("operational_confidence_assessment") or {}
    if isinstance(score, dict) and score.get("score") is not None:
        reasons.append(f"a structured confidence assessment was available with score {score.get('score')}")

    validity = story.get("insight_validity") or {}
    if validity:
        if validity.get("valid", True):
            reasons.append(f"the insight passed the validity gate with severity {validity.get('severity', 'unknown')}")
        else:
            reasons.append("the validity gate flagged the insight as not decision-ready")
        if validity.get("missing_ratio") is not None:
            reasons.append(f"missingness for the story-related fields was about {validity.get('missing_ratio')}")

    validation = evidence.get("data_validation") or evidence.get("cleaning_validation") or {}
    if isinstance(validation, dict):
        if validation.get("row_loss_ratio") is not None:
            reasons.append(f"row loss during cleaning was {validation.get('row_loss_ratio')}")
        if validation.get("schema_stable") is False:
            reasons.append("schema changes reduced confidence in downstream comparisons")
        anomalies = validation.get("anomalies") or []
        if anomalies:
            reasons.append(f"validation detected anomalies: {', '.join(_as_text_list(anomalies, limit=3))}")

    sample_size = _story_sample_size(story)
    if sample_size is not None:
        reasons.append(f"the supporting analysis touched roughly {sample_size} records")
    else:
        dataset_profile = evidence.get("dataset_profile_json") or evidence.get("preclean_profile_json") or {}
        row_count = dataset_profile.get("row_count") or dataset_profile.get("rows")
        if row_count:
            reasons.append(f"the dataset profile indicates about {row_count} records were available")

    p_value = story.get("p_value")
    if p_value is not None:
        try:
            p_value_float = float(p_value)
            if p_value_float <= 0.05:
                reasons.append(f"the relationship is statistically significant at p={round(p_value_float, 4)}")
            else:
                reasons.append(f"the relationship was tested but did not meet common significance thresholds (p={round(p_value_float, 4)})")
        except Exception:
            reasons.append(f"the relationship was tested with p-value {p_value}")

    causal = story.get("causal_evidence") or {}
    if causal.get("grade"):
        reasons.append(f"causal evidence is graded {causal.get('grade')}")

    if story.get("bias_risks"):
        reasons.append(f"bias risks remain: {', '.join(_as_text_list(story.get('bias_risks'), limit=3))}")
    if story.get("assumption_warnings"):
        reasons.append(f"assumptions were not fully satisfied: {', '.join(_as_text_list(story.get('assumption_warnings'), limit=3))}")
    if story.get("readiness_warnings"):
        reasons.append(f"model readiness warnings remain: {', '.join(_as_text_list(story.get('readiness_warnings'), limit=3))}")

    if judgment.get("contradictions_found"):
        reasons.append(f"the judgment layer resolved contradictions: {', '.join(_as_text_list(judgment.get('contradictions_found'), limit=3))}")

    if not reasons:
        reasons.append("confidence is based primarily on the relative story score because no stronger evidence signal was available.")

    lines.extend([f"Reason: {reason}" for reason in reasons[:8]])
    return lines


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
                expected_upside = str(
                    actions[0].get("estimated_uplift_range")
                    or actions[0].get("estimated_uplift")
                    or prescriptive.get("estimated_upside")
                )
                action_paths = "\n".join(
                    f"- {item.get('action')} | risk={item.get('risk_level')} | feasibility={item.get('feasibility')}"
                    for item in actions[:4]
                ) or "None"
    confidence_text = "None"
    if predictive and predictive.get("confidence"):
        confidence_text = (
            f"{predictive['confidence'].get('label')} "
            f"({predictive['confidence'].get('score')}) - "
            f"{predictive['confidence'].get('explanation')}"
        )
    monitoring_plan = "\n".join(
        f"- {item}" for item in monitoring.get("monitoring_plan", [])[:5]
    ) if monitoring.get("monitoring_plan") else "None"
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
                f"- Predictive confidence: {confidence_text}\n"
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
        "confidence": confidence_text,
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


def _reasoning_section_lines(reasoning: Dict[str, Any], key: str) -> List[str]:
    if not reasoning:
        return []
    sections = reasoning.get("sections") or {}
    items = sections.get(key) or []
    if isinstance(items, str):
        items = [items]
    return [line for line in _as_text_list(items, limit=20) if _normalize_text(line)]


def _dataset_documentation(
    state: AnalystState,
    evidence: Dict[str, Any],
    business_question: str,
    selected_columns: List[str],
) -> List[str]:
    dataset_profile = state.get("dataset_profile", {}) or evidence.get("dataset_profile_json", {}) or {}
    preclean_profile = evidence.get("preclean_profile_json", {}) or {}
    normalization = state.get("normalization_output", {}) or {}
    lines = [
        f"Business question: {business_question}",
        f"Dataset path: {state.get('dataset_path') or 'N/A'}",
        f"Active dataset: {state.get('active_dataset') or 'N/A'}",
        f"Selected analysis columns: {selected_columns if selected_columns else 'not specified'}",
        f"Row count: {dataset_profile.get('row_count') or preclean_profile.get('row_count') or 'unknown'}",
        f"Column count: {dataset_profile.get('column_count') or preclean_profile.get('column_count') or 'unknown'}",
        f"Numeric columns: {_format_value(dataset_profile.get('numeric_columns') or preclean_profile.get('numeric_columns'))}",
        f"Categorical columns: {_format_value(dataset_profile.get('categorical_columns') or preclean_profile.get('categorical_columns'))}",
        f"Normalization schema: {state.get('normalization_schema_name') or 'N/A'}",
        f"Normalization output: {_format_value(normalization) if normalization else 'None'}",
    ]
    if state.get("normalization_source_type"):
        lines.append(f"Normalization source type: {state.get('normalization_source_type')}")
    if state.get("normalization_synonyms"):
        lines.append(f"Synonym map: {_format_value(state.get('normalization_synonyms'))}")
    return [line for line in lines if _normalize_text(line)]


def _problem_definition_lines(
    business_question: str,
    decision_context: Any,
    selected_columns: List[str],
) -> List[str]:
    lines = [
        f"Business question: {business_question}",
        f"Decision context: {_format_value(decision_context)}",
        f"Primary columns in scope: {_format_value(selected_columns)}",
    ]
    return [line for line in lines if _normalize_text(line) and _normalize_text(line) != "None"]


def _quality_audit_lines(state: AnalystState, evidence: Dict[str, Any]) -> List[str]:
    validation = state.get("data_validation") or state.get("cleaning_validation") or {}
    cleaning_reasoning_status = evidence.get("cleaning_reasoning_status")
    structural_signals = evidence.get("structural_signals", {}) or {}
    cleaning_constraints = state.get("cleaning_constraints") or {}
    cleaning_audit = state.get("cleaning_audit") or []
    lines = [
        f"Validation summary: {_format_value(validation)}",
        f"Cleaning reasoning status: {_format_value(cleaning_reasoning_status)}",
        f"Structural signals: {_format_value(structural_signals)}",
        f"Cleaning constraints: {_format_value(cleaning_constraints)}",
        f"Cleaning audit: {_format_value(cleaning_audit)}",
    ]
    issues = validation.get("issues") if isinstance(validation, dict) else None
    warnings = validation.get("warnings") if isinstance(validation, dict) else None
    anomalies = validation.get("anomalies") if isinstance(validation, dict) else None
    if issues:
        lines.append(f"Issues identified: {_format_value(issues)}")
    if warnings:
        lines.append(f"Warnings: {_format_value(warnings)}")
    if anomalies:
        lines.append(f"Anomalies: {_format_value(anomalies)}")
    return [line for line in lines if _normalize_text(line) and _normalize_text(line) != "None"]


def _preparation_lines(state: AnalystState, evidence: Dict[str, Any]) -> List[str]:
    cleaning_plan = state.get("cleaning_plan") or []
    cleaning_execution = evidence.get("cleaning_execution_log") or []
    row_filtering = evidence.get("row_filtering") or {}
    selection = state.get("selected_columns") or []
    analysis_dataset = state.get("analysis_dataset")
    raw_analysis_dataset = state.get("raw_analysis_dataset")
    lines = [
        f"Cleaning plan: {_format_value(cleaning_plan)}",
        f"Cleaning execution log: {_format_value(cleaning_execution)}",
        f"Row filtering: {_format_value(row_filtering)}",
        f"Selected columns retained for analysis: {_format_value(selection)}",
        f"Analysis dataset prepared: {'yes' if analysis_dataset is not None else 'no'}",
        f"Raw analysis dataset preserved: {'yes' if raw_analysis_dataset is not None else 'no'}",
    ]
    return [line for line in lines if _normalize_text(line) and _normalize_text(line) != "None"]


def _analysis_lines(
    analysis_plan: List[Any],
    computation_plan: Dict[str, Any],
    tool_results: Dict[str, Any],
    top_stories: List[Dict[str, Any]],
    visualizations: List[Dict[str, Any]],
    llm_insights: str,
) -> List[str]:
    lines = [
        f"Analysis plan: {_format_analysis_plan(analysis_plan)}",
        f"Computation plan: {_format_computation_plan(computation_plan)}",
        f"Tool outputs: {_format_tool_results(tool_results)}",
        f"Top story candidates: {_format_top_stories(top_stories)}",
        f"Visualizations: {_format_visualizations(visualizations)}",
        f"LLM insights: {_normalize_text(llm_insights) or 'None'}",
    ]
    return [line for line in lines if _normalize_text(line) and _normalize_text(line) != "None"]


def _validation_lines(
    judgment: Dict[str, Any],
    predictive_sections: Dict[str, str],
    evidence: Dict[str, Any],
    top_stories: List[Dict[str, Any]],
    reasoning: Dict[str, Any] | None = None,
) -> List[str]:
    lines = [
        f"Judgment summary: {_format_judgment_summary(judgment)}",
        f"Contradictions resolved: {_format_contradictions(judgment)}",
        f"Confidence basis: {predictive_sections.get('confidence', 'None')}",
        f"Reliability notes: {predictive_sections.get('reliability', 'None')}",
        f"Model dependency risks: {predictive_sections.get('dependency_risks', 'None')}",
        f"Evidence chain: dataset -> preparation -> analysis -> story ranking -> decision review",
    ]
    if reasoning:
        confidence = reasoning.get("confidence") or {}
        if confidence:
            lines.append(f"Reasoning confidence: {confidence.get('level')} ({confidence.get('score')})")
            lines.append(f"Confidence rationale: {', '.join(_to_lines(confidence.get('reasons'), limit=4))}")
    if top_stories:
        lines.extend(_confidence_context_lines(top_stories[0], evidence, judgment))
    return [line for line in lines if _normalize_text(line) and _normalize_text(line) != "None"]


def _findings_lines(
    top_stories: List[Dict[str, Any]],
    llm_insights: str,
    evidence: Dict[str, Any],
    judgment: Dict[str, Any],
    reasoning: Dict[str, Any] | None = None,
) -> List[str]:
    reasoning_lines = _reasoning_section_lines(reasoning or {}, "findings")
    if reasoning_lines:
        return reasoning_lines[:10] + ([f"Analyst synthesis: {llm_insights}"] if llm_insights else [])
    lines: List[str] = []
    for index, story in enumerate(top_stories[:5]):
        insight = _normalize_text(story.get("insight"))
        if not insight:
            continue
        lines.append(
            f"{insight} | confidence={_story_confidence(story)} | score={story.get('score')} | validity={((story.get('insight_validity') or {}).get('valid', 'unknown'))}"
        )
        if index == 0:
            lines.extend(_confidence_context_lines(story, evidence, judgment)[:5])
    if llm_insights:
        lines.append(f"Analyst synthesis: {llm_insights}")
    return lines


def _root_cause_lines(
    top_stories: List[Dict[str, Any]],
    predictive_sections: Dict[str, str],
    evidence: Dict[str, Any],
    judgment: Dict[str, Any],
    reasoning: Dict[str, Any] | None = None,
) -> List[str]:
    reasoning_lines = _reasoning_section_lines(reasoning or {}, "root_cause")
    if reasoning_lines:
        return reasoning_lines[:10]
    lines: List[str] = []
    if predictive_sections.get("drivers") and predictive_sections["drivers"] != "None":
        lines.append(f"Driver evidence: {predictive_sections['drivers']}")
    if top_stories:
        primary = top_stories[0]
        lines.append(f"Primary pattern: {primary.get('insight')}")
        causal = primary.get("causal_evidence") or {}
        if causal:
            rationale = causal.get("rationale") or []
            lines.append(
                f"Supported cause evidence: grade={causal.get('grade', 'LOW')}, score={causal.get('score', 'unknown')}"
                + (f", rationale={'; '.join(_as_text_list(rationale, limit=3))}" if rationale else "")
            )
        if primary.get("confounders"):
            lines.append(f"Possible contributing factors: {_format_value(primary.get('confounders'))}")
        if primary.get("bias_risks"):
            lines.append(f"Evidence against a single-cause interpretation: {_format_value(primary.get('bias_risks'))}")
        if primary.get("assumption_warnings"):
            lines.append(f"Assumptions that may weaken the explanation: {_format_value(primary.get('assumption_warnings'))}")
        if primary.get("semantic_reasoning"):
            semantic = primary.get("semantic_reasoning") or {}
            relationship_type = semantic.get("relationship_type")
            if relationship_type:
                lines.append(f"Semantic interpretation: relationship_type={relationship_type}")
        if not causal or causal.get("grade") in {None, "LOW"}:
            lines.append("Alternative explanation: the observed pattern may be observational rather than causal, so root cause cannot be stated with high confidence.")
        if primary.get("score_components"):
            lines.append(f"Why this story ranked highly: {_format_value(primary.get('score_components'))}")
        lines.append(
            f"Overall root-cause confidence: {_story_confidence(primary)}"
        )
        if not primary.get("confounders") and not primary.get("bias_risks"):
            lines.append("Unknowns: the current analysis does not isolate a single driver, so the mechanism remains partially unresolved.")
    return lines


def _opportunity_lines(
    decision_recommendations: List[Dict[str, Any]],
    predictive_sections: Dict[str, str],
    top_stories: List[Dict[str, Any]],
    reasoning: Dict[str, Any] | None = None,
) -> List[str]:
    reasoning_lines = _reasoning_section_lines(reasoning or {}, "opportunities")
    if reasoning_lines:
        return reasoning_lines[:12]
    lines: List[str] = []
    if predictive_sections.get("best_action") and predictive_sections["best_action"] != "None":
        lines.append(f"Primary opportunity: {predictive_sections['best_action']}")
    if predictive_sections.get("expected_upside") and predictive_sections["expected_upside"] != "None":
        lines.append(f"Expected upside: {predictive_sections['expected_upside']}")
    for index, item in enumerate(decision_recommendations[:4]):
        action_type = item.get("action_type")
        if action_type in {"optimization", "strategic", "experiment"}:
            related_story = top_stories[index] if index < len(top_stories) else {}
            evidence_bits = []
            if related_story.get("insight"):
                evidence_bits.append(f"linked finding: {related_story.get('insight')}")
            if related_story.get("causal_evidence"):
                causal = related_story.get("causal_evidence") or {}
                evidence_bits.append(f"causal evidence grade={causal.get('grade', 'LOW')} score={causal.get('score', 'unknown')}")
            lines.append(
                f"{item.get('recommended_action')} | type={action_type} | confidence={item.get('confidence_in_action')} | impact={((item.get('impact_assessment') or {}).get('impact_level'))}"
            )
            if evidence_bits:
                lines.append(f"Evidence supporting opportunity: {'; '.join(evidence_bits)}")
    return lines


def _risk_lines(
    judgment: Dict[str, Any],
    predictive_sections: Dict[str, str],
    top_stories: List[Dict[str, Any]],
    evidence: Dict[str, Any],
    reasoning: Dict[str, Any] | None = None,
) -> List[str]:
    reasoning_lines = _reasoning_section_lines(reasoning or {}, "risks")
    if reasoning_lines:
        return reasoning_lines[:12]
    lines: List[str] = []
    for item in _as_text_list(predictive_sections.get("risks"), limit=5):
        lines.append(item)
    for item in _as_text_list(predictive_sections.get("what_could_go_wrong"), limit=5):
        lines.append(item)
    if judgment.get("contradictions_found"):
        lines.append(f"Contradictions: {_format_value(judgment.get('contradictions_found'))}")
    for story in top_stories[:3]:
        validity = story.get("insight_validity") or {}
        if not validity.get("valid", True):
            lines.append(f"Validation warning for {story.get('insight')}: {_format_value(validity)}")
        if story.get("bias_risks"):
            lines.append(f"Bias risk for {story.get('insight')}: {_format_value(story.get('bias_risks'))}")
        if story.get("assumption_warnings"):
            lines.append(f"Assumption risk for {story.get('insight')}: {_format_value(story.get('assumption_warnings'))}")
    validation = evidence.get("data_validation") or evidence.get("cleaning_validation") or {}
    if isinstance(validation, dict):
        if validation.get("row_loss_ratio") is not None and validation.get("row_loss_ratio", 0.0) > 0:
            lines.append(f"Data quality risk: cleaning changed {validation.get('row_loss_ratio')} of the rows, which can reduce confidence in the result.")
        anomalies = validation.get("anomalies") or []
        if anomalies:
            lines.append(f"Data quality warning: {', '.join(_as_text_list(anomalies, limit=3))}")
    return lines


def _recommendation_lines(
    decision_recommendations: List[Dict[str, Any]],
    decision_ranking: List[Dict[str, Any]],
    recommended_first: Dict[str, Any] | None,
    judgment: Dict[str, Any],
    top_stories: List[Dict[str, Any]],
    evidence: Dict[str, Any],
    reasoning: Dict[str, Any] | None = None,
) -> List[str]:
    reasoning_lines = _reasoning_section_lines(reasoning or {}, "recommendations")
    if reasoning_lines:
        return reasoning_lines[:18]
    lines: List[str] = []
    if recommended_first:
        lines.append(
            f"Recommended first action: {recommended_first.get('recommended_action')} | priority={((recommended_first.get('priority') or {}).get('priority_score'))} | confidence={recommended_first.get('confidence_in_action')}"
        )
    elif judgment.get("recommended_first_action"):
        lines.append(f"Recommended first action: {judgment.get('recommended_first_action')}")
    for index, item in enumerate(decision_ranking[:5]):
        impact = item.get("impact_assessment") or {}
        priority = item.get("priority") or {}
        related_story = top_stories[index] if index < len(top_stories) else {}
        assumptions = []
        if related_story.get("assumption_warnings"):
            assumptions.extend(_as_text_list(related_story.get("assumption_warnings"), limit=3))
        if related_story.get("bias_risks"):
            assumptions.extend(_as_text_list(related_story.get("bias_risks"), limit=3))
        if related_story.get("confidence_assessment") or related_story.get("operational_confidence_assessment"):
            conf_detail = related_story.get("confidence_assessment") or related_story.get("operational_confidence_assessment") or {}
            if conf_detail.get("explanation"):
                assumptions.append(f"confidence note: {conf_detail.get('explanation')}")
        monitoring_kpis = item.get("monitoring_kpis", []) or []
        if not monitoring_kpis and related_story.get("recommended_next_step"):
            monitoring_kpis = [related_story.get("recommended_next_step")]
        success_criteria = []
        if monitoring_kpis:
            success_criteria.append(f"monitor {', '.join(monitoring_kpis[:4])}")
        if related_story.get("insight_validity") and not (related_story.get("insight_validity") or {}).get("valid", True):
            success_criteria.append("only proceed if the insight becomes decision-ready")
        evidence_summary = []
        if related_story.get("insight"):
            evidence_summary.append(related_story.get("insight"))
        if related_story.get("causal_evidence"):
            causal = related_story.get("causal_evidence") or {}
            evidence_summary.append(f"causal grade {causal.get('grade', 'LOW')} (score {causal.get('score', 'unknown')})")
        if related_story.get("score") is not None:
            evidence_summary.append(f"story score {related_story.get('score')}")
        lines.append(
            f"{item.get('recommended_action')} | action_type={item.get('action_type')} | priority={priority.get('priority_score')} ({priority.get('priority_level')}) | impact={impact.get('impact_level')} | confidence={item.get('confidence_in_action')}"
        )
        if evidence_summary:
            lines.append(f"Evidence supporting recommendation: {'; '.join(evidence_summary)}")
        if assumptions:
            lines.append(f"Assumptions and risks: {'; '.join(assumptions[:4])}")
        if success_criteria:
            lines.append(f"Success measures: {'; '.join(success_criteria[:2])}")
        if impact.get("impact_level") not in {None, "none", "uncertain"}:
            lines.append(f"Expected business value: {impact.get('impact_level')} impact with {impact.get('estimated_direction')} directional effect.")
    if decision_recommendations:
        lines.append(f"Decision records available: {len(decision_recommendations)}")
    return lines


def _decision_matrix_lines(
    decision_ranking: List[Dict[str, Any]],
    top_stories: List[Dict[str, Any]],
    reasoning: Dict[str, Any] | None = None,
) -> List[str]:
    reasoning_lines = _reasoning_section_lines(reasoning or {}, "decision_matrix")
    if reasoning_lines:
        return reasoning_lines[:15]
    lines: List[str] = []
    for index, item in enumerate(decision_ranking[:5]):
        impact = item.get("impact_assessment") or {}
        priority = item.get("priority") or {}
        related_story = top_stories[index] if index < len(top_stories) else {}
        evidence_strength = _story_confidence(related_story) if related_story else "unknown"
        lines.append(
            f"{item.get('recommended_action')} | type={item.get('action_type')} | priority={priority.get('priority_score')} | impact={impact.get('impact_level')} | direction={impact.get('estimated_direction')} | confidence={item.get('confidence_in_action')} | evidence_strength={evidence_strength}"
        )
        if related_story.get("insight"):
            lines.append(f"Decision rationale: {related_story.get('insight')}")
        if related_story.get("bias_risks"):
            lines.append(f"Decision risk: {_format_value(related_story.get('bias_risks'))}")
    return lines


def _implementation_lines(
    predictive_sections: Dict[str, str],
    decision_ranking: List[Dict[str, Any]],
    top_stories: List[Dict[str, Any]],
    reasoning: Dict[str, Any] | None = None,
) -> List[str]:
    reasoning_lines = _reasoning_section_lines(reasoning or {}, "implementation")
    if reasoning_lines:
        return reasoning_lines[:12]
    lines: List[str] = []
    if predictive_sections.get("constraints") and predictive_sections["constraints"] != "None":
        lines.append(f"Constraints: {predictive_sections['constraints']}")
    if predictive_sections.get("experiment_design") and predictive_sections["experiment_design"] != "None":
        lines.append(f"Experiment design: {predictive_sections['experiment_design']}")
    if predictive_sections.get("assumption_stress") and predictive_sections["assumption_stress"] != "None":
        lines.append(f"Assumption stress: {predictive_sections['assumption_stress']}")
    for index, item in enumerate(decision_ranking[:3]):
        impact = item.get("impact_assessment") or {}
        related_story = top_stories[index] if index < len(top_stories) else {}
        blockers = []
        if related_story.get("readiness_warnings"):
            blockers.extend(_as_text_list(related_story.get("readiness_warnings"), limit=3))
        if related_story.get("assumption_warnings"):
            blockers.extend(_as_text_list(related_story.get("assumption_warnings"), limit=3))
        if related_story.get("bias_risks"):
            blockers.extend(_as_text_list(related_story.get("bias_risks"), limit=3))
        lines.append(
            f"Rollout guidance for {item.get('recommended_action')}: start with the smallest defensible scope, validate the supporting assumption set, and monitor for {impact.get('estimated_direction')} movement."
        )
        if blockers:
            lines.append(f"Potential blockers: {'; '.join(blockers[:4])}")
        monitoring_kpis = item.get("monitoring_kpis", []) or []
        if monitoring_kpis:
            lines.append(f"Required monitoring: {', '.join(monitoring_kpis[:4])}")
    return lines


def _monitoring_lines(
    predictive_sections: Dict[str, str],
    decision_ranking: List[Dict[str, Any]],
    top_stories: List[Dict[str, Any]],
    reasoning: Dict[str, Any] | None = None,
) -> List[str]:
    reasoning_lines = _reasoning_section_lines(reasoning or {}, "monitoring")
    if reasoning_lines:
        return reasoning_lines[:15]
    lines: List[str] = []
    if predictive_sections.get("monitoring_plan") and predictive_sections["monitoring_plan"] != "None":
        lines.append(f"Monitoring plan: {predictive_sections['monitoring_plan']}")
    if predictive_sections.get("early_warnings") and predictive_sections["early_warnings"] != "None":
        lines.append(f"Early warnings: {predictive_sections['early_warnings']}")
    for index, item in enumerate(decision_ranking[:4]):
        kpis = list(dict.fromkeys(item.get("monitoring_kpis", []) or []))
        related_story = top_stories[index] if index < len(top_stories) else {}
        if not kpis and related_story.get("recommended_next_step"):
            kpis = [related_story.get("recommended_next_step")]
        if kpis:
            lines.append(f"KPIs to monitor: {', '.join(kpis)}")
            lines.append("Baseline: not provided in the evidence.")
            lines.append("Target: improve the KPI relative to the current baseline; do not invent a numeric target without supporting business guidance.")
            if related_story.get("failure_conditions"):
                lines.append(f"Warning indicators: {_format_value(related_story.get('failure_conditions'))}")
            else:
                lines.append("Warning indicators: no explicit failure threshold was provided, so watch for directional deterioration.")
        else:
            lines.append("KPIs to monitor: not specified by the evidence.")
    return lines


def _limitations_lines(
    state: AnalystState,
    evidence: Dict[str, Any],
    predictive_sections: Dict[str, str],
    top_stories: List[Dict[str, Any]],
    reasoning: Dict[str, Any] | None = None,
) -> List[str]:
    reasoning_lines = _reasoning_section_lines(reasoning or {}, "limitations")
    if reasoning_lines:
        return reasoning_lines[:12]
    lines: List[str] = []
    data_validation = state.get("data_validation") or state.get("cleaning_validation") or {}
    limitations = []
    if isinstance(data_validation, dict):
        limitations.extend(_as_text_list(data_validation.get("warnings"), limit=5))
        limitations.extend(_as_text_list(data_validation.get("issues"), limit=5))
        if data_validation.get("row_loss_ratio") is not None:
            limitations.append(f"Row loss ratio: {data_validation.get('row_loss_ratio')}")
        if data_validation.get("anomalies"):
            limitations.extend(_as_text_list(data_validation.get("anomalies"), limit=5))
    limitations.extend(_as_text_list(predictive_sections.get("risks"), limit=5))
    limitations.extend(_as_text_list(predictive_sections.get("assumption_stress"), limit=5))
    assumptions = evidence.get("clarification_questions") or []
    if assumptions:
        limitations.append(f"Open assumptions/questions: {_format_value(assumptions)}")
    if top_stories:
        primary = top_stories[0]
        if primary.get("bias_risks"):
            limitations.append(f"Bias risks remain: {_format_value(primary.get('bias_risks'))}")
        if primary.get("confounders"):
            limitations.append(f"Potential confounders were identified: {_format_value(primary.get('confounders'))}")
        if primary.get("causal_evidence") and ((primary.get("causal_evidence") or {}).get("grade")) in {"LOW", None}:
            limitations.append("Causal strength is limited, so recommendations should be treated as evidence-based hypotheses rather than proven causes.")
    if limitations:
        lines.extend([f"{item}" for item in limitations if _normalize_text(item)])
    return lines


def _conclusion_lines(
    business_question: str,
    judgment: Dict[str, Any],
    recommended_first: Dict[str, Any] | None,
    predictive_sections: Dict[str, str],
    evidence: Dict[str, Any],
    top_stories: List[Dict[str, Any]],
    reasoning: Dict[str, Any] | None = None,
) -> List[str]:
    reasoning_lines = _reasoning_section_lines(reasoning or {}, "conclusion")
    if reasoning_lines:
        return reasoning_lines[:10]
    lines = [
        f"Business question: {business_question}",
        f"Dominant reasoning: {judgment.get('dominant_reasoning', 'None')}",
        f"Global confidence: {judgment.get('global_confidence', 'unknown')}",
    ]
    if recommended_first:
        lines.append(f"Recommended first action: {recommended_first.get('recommended_action')}")
    elif judgment.get("recommended_first_action"):
        lines.append(f"Recommended first action: {judgment.get('recommended_first_action')}")
    if predictive_sections.get("confidence") and predictive_sections["confidence"] != "None":
        lines.append(f"Decision confidence context: {predictive_sections['confidence']}")
    if top_stories:
        lines.extend(_confidence_context_lines(top_stories[0], evidence, judgment)[:4])
    return lines


def _appendix_lines(state: AnalystState, evidence: Dict[str, Any], top_stories: List[Dict[str, Any]]) -> List[str]:
    reasoning = evidence.get("analytical_reasoning") or state.get("analytical_reasoning") or {}
    lines = [
        f"Traceability chain: dataset -> preparation -> analysis -> stories -> decisions -> judgment",
        f"Dataset source: {state.get('dataset_path') or 'N/A'}",
        f"Prepared analysis dataset: {'available' if state.get('analysis_dataset') is not None else 'not available'}",
        f"Selected analysis columns: {_format_value(state.get('selected_columns') or [])}",
        f"Top story signatures: {_format_value([story.get('type') for story in top_stories[:5]])}",
        f"Primary evidence: {_format_value(top_stories[0].get('insight') if top_stories else None)}",
        f"Analytical reasoning traceability: {_format_value((reasoning or {}).get('traceability'))}",
        f"Human-in-the-loop: {_format_value(evidence.get('human_in_loop'))}",
        f"Clarification questions: {_format_value(evidence.get('clarification_questions'))}",
    ]
    return [line for line in lines if _normalize_text(line) and _normalize_text(line) != "None"]


def _render_report(title: str, sections: Dict[str, str]) -> str:
    ordered_sections = [sections[key] for key in sections]
    body = "\n\n".join(ordered_sections)
    return f"{title}\n\n{body}\n"


def _build_master_report(
    state: AnalystState,
    evidence: Dict[str, Any],
    business_question: str,
    selected_columns: List[str],
    analysis_plan: List[Any],
    computation_plan: Dict[str, Any],
    tool_results: Dict[str, Any],
    top_stories: List[Dict[str, Any]],
    visualizations: List[Dict[str, Any]],
    decision_recommendations: List[Dict[str, Any]],
    decision_ranking: List[Dict[str, Any]],
    recommended_first: Dict[str, Any] | None,
    judgment: Dict[str, Any],
    predictive_sections: Dict[str, str],
    llm_insights: str,
    reasoning: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    sections = _section_map(
        ("PROBLEM DEFINITION", _problem_definition_lines(business_question, state.get("decision_context"), selected_columns)),
        ("DATASET DOCUMENTATION", _dataset_documentation(state, evidence, business_question, selected_columns)),
        ("DATA PROFILING", [
            f"Dataset profile: {_format_value(state.get('dataset_profile') or evidence.get('dataset_profile_json'))}",
            f"Pre-clean profile: {_format_value(evidence.get('preclean_profile_json'))}",
            f"Column registry: {_format_value(state.get('column_registry'))}",
            f"Relationship signals: {_format_value(evidence.get('relationship_signals'))}",
        ]),
        ("DATA QUALITY AUDIT", _quality_audit_lines(state, evidence)),
        ("DATA PREPARATION AND CLEANING", _preparation_lines(state, evidence)),
        ("EXPLORATORY DATA ANALYSIS", _analysis_lines(analysis_plan, computation_plan, tool_results, top_stories, visualizations, llm_insights)),
        ("STATISTICAL AND ANALYTICAL VALIDATION", _validation_lines(judgment, predictive_sections, evidence, top_stories, reasoning)),
        ("KEY FINDINGS AND INSIGHTS", _findings_lines(top_stories, llm_insights, evidence, judgment, reasoning)),
        ("ROOT CAUSE ANALYSIS", _root_cause_lines(top_stories, predictive_sections, evidence, judgment, reasoning)),
        ("OPPORTUNITY ASSESSMENT", _opportunity_lines(decision_recommendations, predictive_sections, top_stories, reasoning)),
        ("RISK ASSESSMENT", _risk_lines(judgment, predictive_sections, top_stories, evidence, reasoning)),
        ("RECOMMENDATION ENGINE", _recommendation_lines(decision_recommendations, decision_ranking, recommended_first, judgment, top_stories, evidence, reasoning)),
        ("DECISION SUPPORT MATRIX", _decision_matrix_lines(decision_ranking, top_stories, reasoning)),
        ("IMPLEMENTATION CONSIDERATIONS", _implementation_lines(predictive_sections, decision_ranking, top_stories, reasoning)),
        ("MONITORING AND SUCCESS METRICS", _monitoring_lines(predictive_sections, decision_ranking, top_stories, reasoning)),
        ("LIMITATIONS AND ASSUMPTIONS", _limitations_lines(state, evidence, predictive_sections, top_stories, reasoning)),
        ("OVERALL CONCLUSION", _conclusion_lines(business_question, judgment, recommended_first, predictive_sections, evidence, top_stories, reasoning)),
        ("APPENDIX", _appendix_lines(state, evidence, top_stories)),
    )

    ordered_keys = [
        "PROBLEM DEFINITION",
        "DATASET DOCUMENTATION",
        "DATA PROFILING",
        "DATA QUALITY AUDIT",
        "DATA PREPARATION AND CLEANING",
        "EXPLORATORY DATA ANALYSIS",
        "STATISTICAL AND ANALYTICAL VALIDATION",
        "KEY FINDINGS AND INSIGHTS",
        "ROOT CAUSE ANALYSIS",
        "OPPORTUNITY ASSESSMENT",
        "RISK ASSESSMENT",
        "RECOMMENDATION ENGINE",
        "DECISION SUPPORT MATRIX",
        "IMPLEMENTATION CONSIDERATIONS",
        "MONITORING AND SUCCESS METRICS",
        "LIMITATIONS AND ASSUMPTIONS",
        "OVERALL CONCLUSION",
        "APPENDIX",
    ]
    return {key: sections[key] for key in ordered_keys if key in sections}


def _build_executive_report(
    master_sections: Dict[str, str],
    evidence: Dict[str, Any],
    judgment: Dict[str, Any],
    predictive_sections: Dict[str, str],
    decision_ranking: List[Dict[str, Any]],
    recommended_first: Dict[str, Any] | None,
    business_question: str,
) -> Dict[str, str]:
    overall_health = judgment.get("dominant_reasoning", "No judgment available.")
    top_findings = master_sections.get("KEY FINDINGS AND INSIGHTS", "None")
    top_risks = master_sections.get("RISK ASSESSMENT", "None")
    top_opportunities = master_sections.get("OPPORTUNITY ASSESSMENT", "None")
    action_plan = master_sections.get("IMPLEMENTATION CONSIDERATIONS", "None")
    monitoring = master_sections.get("MONITORING AND SUCCESS METRICS", "None")
    validation = master_sections.get("STATISTICAL AND ANALYTICAL VALIDATION", "None")
    monitoring_items = _section_items(monitoring, limit=1)
    action_plan_items = _section_items(action_plan, limit=1)
    top_finding_items = _section_items(top_findings, limit=1)
    top_risk_items = _section_items(top_risks, limit=1)
    top_opportunity_items = _section_items(top_opportunities, limit=1)
    confidence_items = _section_items(validation, limit=5)
    conclusion_confidence = predictive_sections.get("confidence")
    if not conclusion_confidence or conclusion_confidence == "None":
        conclusion_confidence = _confidence_label(judgment.get("global_confidence"))
    business_analysis_sections = _section_map(
        ("BUSINESS CONTEXT", [
            f"Business question: {business_question}",
            f"Dataset source: {evidence.get('dataset_path') or 'N/A'}",
            f"Judgment confidence: {judgment.get('global_confidence', 'unknown')}",
            f"Actionability: {judgment.get('actionability', 'unknown')}",
        ]),
        ("KEY FINDINGS", _section_items(top_findings, limit=6)),
        ("ROOT CAUSE", _section_items(master_sections.get("ROOT CAUSE ANALYSIS", "None"), limit=6)),
        ("OPPORTUNITIES", _section_items(top_opportunities, limit=6)),
        ("RISKS", _section_items(top_risks, limit=6)),
        ("RECOMMENDATIONS", _section_items(master_sections.get("RECOMMENDATION ENGINE", "None"), limit=6)),
        ("DECISION MATRIX", _section_items(master_sections.get("DECISION SUPPORT MATRIX", "None"), limit=6)),
        ("IMPLEMENTATION ROADMAP", _section_items(action_plan, limit=6)),
        ("MONITORING FRAMEWORK", _section_items(monitoring, limit=6)),
        ("LIMITATIONS", _section_items(master_sections.get("LIMITATIONS AND ASSUMPTIONS", "None"), limit=6)),
        ("FINAL BUSINESS CONCLUSION", [
            f"Dominant reasoning: {overall_health}",
            f"Recommended first action: {recommended_first.get('recommended_action') if recommended_first else judgment.get('recommended_first_action') or 'None'}",
            f"Confidence level: {conclusion_confidence}",
            f"Confidence rationale: {confidence_items[0] if confidence_items else 'Not enough evidence to explain confidence.'}",
        ]),
    )
    business_analysis_body = "\n\n".join(business_analysis_sections.values()) if business_analysis_sections else "None"
    summary_sections = _section_map(
        ("EXECUTIVE SUMMARY", [
            f"Business objective: {business_question}",
            f"Overall business health: {overall_health}",
            f"Most important findings: {top_finding_items[0] if top_finding_items else 'None'}",
            f"Key risks: {top_risk_items[0] if top_risk_items else 'None'}",
            f"Major opportunities: {top_opportunity_items[0] if top_opportunity_items else 'None'}",
            f"Executive decision: {recommended_first.get('recommended_action') if recommended_first else judgment.get('recommended_first_action') or 'None'}",
            f"Immediate action plan: {action_plan_items[0] if action_plan_items else 'None'}",
            f"KPIs to monitor: {monitoring_items[0].replace('KPIs to monitor: ', '') if monitoring_items and monitoring_items[0].startswith('KPIs to monitor: ') else (monitoring_items[0] if monitoring_items else 'None')}",
            f"Confidence rationale: {confidence_items[0] if confidence_items else 'Not enough evidence to explain confidence.'}",
        ]),
    )
    return {
        **summary_sections,
        "BUSINESS ANALYSIS REPORT": f"BUSINESS ANALYSIS REPORT\n\n{business_analysis_body}",
    }


def report_node(state: AnalystState) -> AnalystState:
    """
    Generates a professional report that reads like a finished workflow output.
    """
    evidence = state.setdefault("analysis_evidence", {})
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
    selected_columns = state.get("selected_columns", []) or []
    reasoning = evidence.get("analytical_reasoning") or state.get("analytical_reasoning") or {}

    hitl_summary = "None"
    if human_in_loop:
        hitl_summary = f"{human_in_loop.get('mode')}: {human_in_loop.get('action')}"

    master_sections = _build_master_report(
        state=state,
        evidence=evidence,
        business_question=business_question,
        selected_columns=selected_columns,
        analysis_plan=analysis_plan,
        computation_plan=computation_plan,
        tool_results=tool_results,
        top_stories=top_stories,
        visualizations=visualizations,
        decision_recommendations=decision_recommendations,
        decision_ranking=decision_ranking,
        recommended_first=recommended_first,
        judgment=judgment,
        predictive_sections=predictive_sections,
        llm_insights=llm_insights,
        reasoning=reasoning,
    )

    executive_sections = _build_executive_report(
        master_sections=master_sections,
        evidence=evidence,
        judgment=judgment,
        predictive_sections=predictive_sections,
        decision_ranking=decision_ranking,
        recommended_first=recommended_first,
        business_question=business_question,
    )

    master_report = _render_report("================ MASTER REPORT ================", master_sections)
    executive_report = _render_report("================ EXECUTIVE REPORT ================", executive_sections)
    combined_report = executive_report + "\n" + master_report

    report_package = {
        "master_report": master_report,
        "executive_report": executive_report,
        "sections": {
            "master": master_sections,
            "executive": executive_sections,
        },
        "analytical_reasoning": reasoning,
        "traceability": {
            "business_question": business_question,
            "selected_columns": selected_columns,
            "decision_context": decision_context,
            "judgment_summary": judgment,
            "human_in_loop": human_in_loop,
            "clarification_questions": clarification_questions,
        },
        "confidence": {
            "global_confidence": judgment.get("global_confidence"),
            "predictive_confidence": predictive_sections.get("confidence"),
            "primary_story_confidence": _story_confidence(top_stories[0]) if top_stories else "unknown",
            "analytical_reasoning_confidence": (reasoning.get("confidence") or {}).get("score") if reasoning else None,
        },
    }

    evidence["master_report"] = master_report
    evidence["executive_report"] = executive_report
    evidence["report_package"] = report_package

    state["master_report"] = master_report
    state["executive_report"] = executive_report
    state["final_report"] = combined_report

    print("\n===== FINAL REPORT =====")
    print(combined_report)

    return state
