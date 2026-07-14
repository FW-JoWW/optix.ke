from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _story_signature(story: Dict[str, Any]) -> str:
    columns = story.get("columns") or []
    if story.get("column"):
        columns = [story["column"], *columns]
    if story.get("group_column"):
        columns = [*columns, story["group_column"]]
    return f"{story.get('type', 'story')}|{'|'.join(str(col) for col in columns if col)}"


def _to_lines(items: Any, limit: int | None = None) -> List[str]:
    if not items:
        return []
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, (list, tuple, set)):
        items = [items]
    lines: List[str] = []
    for item in items:
        if isinstance(item, dict):
            text = ", ".join(
                f"{key}: {value}"
                for key, value in item.items()
                if value not in (None, "") and value != [] and value != {}
            )
        else:
            text = str(item).strip()
        if text:
            lines.append(text)
        if limit is not None and len(lines) >= limit:
            break
    return lines


def _story_detail_map(evidence: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    details = evidence.get("llm_insight_details", []) or []
    return {
        str(item.get("related_story_signature")): item
        for item in details
        if isinstance(item, dict) and item.get("related_story_signature")
    }


def _validation_payload(state: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    return state.get("data_validation") or state.get("cleaning_validation") or evidence.get("data_validation") or {}


def _confidence_from_story(story: Dict[str, Any], validation: Dict[str, Any], judgment: Dict[str, Any]) -> Dict[str, Any]:
    story_score = story.get("score")
    confidence_assessment = story.get("confidence_assessment") or story.get("operational_confidence_assessment") or {}
    causal = story.get("causal_evidence") or {}
    validity = story.get("insight_validity") or {}
    reasons: List[str] = []

    if isinstance(confidence_assessment, dict) and confidence_assessment.get("score") is not None:
        score = float(confidence_assessment.get("score"))
        reasons.append("the story already includes a structured confidence assessment")
    elif isinstance(story_score, (int, float)):
        score = float(story_score) * 100 if float(story_score) <= 1 else float(story_score)
        reasons.append("the ranking score provides the best available confidence proxy")
    elif causal.get("score") is not None:
        score = float(causal.get("score"))
        reasons.append("causal evidence provides the strongest available score")
    else:
        score = float(judgment.get("global_confidence", 0) or 0)
        reasons.append("judgment confidence is the remaining evidence proxy")

    missing_ratio = float(validity.get("missing_ratio", 0.0) or 0.0)
    row_loss_ratio = float(validation.get("row_loss_ratio", 0.0) or 0.0) if isinstance(validation, dict) else 0.0
    bias_count = len(story.get("bias_risks", []) or [])
    readiness_count = len(story.get("readiness_warnings", []) or [])
    contradictions = len(judgment.get("contradictions_found", []) or [])

    score -= missing_ratio * 25.0
    score -= row_loss_ratio * 20.0
    score -= min(bias_count * 6.0, 18.0)
    score -= min(readiness_count * 8.0, 24.0)
    score -= min(contradictions * 8.0, 24.0)

    if causal.get("grade") == "STRONG":
        score += 10.0
        reasons.append("causal evidence is graded STRONG")
    elif causal.get("grade") == "MODERATE":
        score += 5.0
        reasons.append("causal evidence is graded MODERATE")
    elif causal.get("grade") == "LOW":
        reasons.append("causal evidence remains limited")

    sample_size = story.get("sample_size")
    if sample_size is None:
        sample_size = (((story.get("estimation") or {}).get("sample_size")) if isinstance(story.get("estimation"), dict) else None)
    if sample_size is None:
        sample_size = (((story.get("semantic_reasoning") or {}).get("details") or {}).get("sample_size"))
    if sample_size:
        try:
            sample_size = int(sample_size)
            if sample_size >= 1000:
                score += 8.0
                reasons.append("sample size is large")
            elif sample_size >= 100:
                score += 4.0
                reasons.append("sample size is reasonably strong")
            elif sample_size < 30:
                score -= 8.0
                reasons.append("sample size is small")
        except Exception:
            pass

    if validation.get("schema_stable") is False:
        reasons.append("schema instability weakens confidence")
        score -= 6.0
    if validation.get("anomalies"):
        reasons.append("data validation detected anomalies")
        score -= 6.0
    if validity and not validity.get("valid", True):
        reasons.append("the insight is not decision-ready")
        score = min(score, 20.0)

    if isinstance(confidence_assessment, dict) and confidence_assessment.get("explanation"):
        reasons.append(str(confidence_assessment.get("explanation")))

    if score >= 75:
        level = "high"
    elif score >= 45:
        level = "medium"
    else:
        level = "low"

    return {
        "level": level,
        "score": round(max(0.0, min(score, 100.0)), 2),
        "reasons": _to_lines(reasons, limit=8),
    }


def _match_decision_for_story(story: Dict[str, Any], decision_ranking: List[Dict[str, Any]]) -> Dict[str, Any]:
    signature = _story_signature(story)
    for item in decision_ranking:
        if item.get("story_signature") == signature:
            return item
    return {}


def _build_findings(
    top_stories: List[Dict[str, Any]],
    detail_map: Dict[str, Dict[str, Any]],
    validation: Dict[str, Any],
    judgment: Dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    for index, story in enumerate(top_stories[:5]):
        signature = _story_signature(story)
        detail = detail_map.get(signature, {})
        headline = detail.get("plain_english") or story.get("insight") or "Observed pattern"
        implication = detail.get("business_implication") or "The pattern is useful for business interpretation."
        evidence_bits = [f"evidence: {story.get('insight')}"]
        if story.get("p_value") is not None:
            evidence_bits.append(f"p-value={story.get('p_value')}")
        if story.get("causal_evidence"):
            causal = story.get("causal_evidence") or {}
            evidence_bits.append(f"causal_grade={causal.get('grade', 'LOW')}")
        if story.get("insight_validity"):
            validity = story.get("insight_validity") or {}
            evidence_bits.append(f"valid={validity.get('valid', True)}")
        if detail.get("limitations"):
            evidence_bits.append(f"limitations: {detail.get('limitations')}")
        if index == 0:
            evidence_bits.append(f"confidence={_confidence_from_story(story, validation, judgment)['level']}")
        lines.append(f"{headline} | {implication} | {'; '.join(evidence_bits)}")
    return lines


def _build_root_cause(
    primary_story: Dict[str, Any],
    validation: Dict[str, Any],
    judgment: Dict[str, Any],
) -> List[str]:
    if not primary_story:
        return ["Root cause confidence is low because no dominant analytical story was identified."]

    causal = primary_story.get("causal_evidence") or {}
    semantic = primary_story.get("semantic_reasoning") or {}
    confounders = _to_lines(primary_story.get("confounders"), limit=4)
    bias_risks = _to_lines(primary_story.get("bias_risks"), limit=4)
    assumptions = _to_lines(primary_story.get("assumption_warnings"), limit=4)
    lines = [
        f"Observed issue: {primary_story.get('insight')}",
        f"Possible contributing factors: {', '.join(confounders) if confounders else 'No specific contributors were isolated.'}",
        f"Supporting evidence: grade={causal.get('grade', 'LOW')}, score={causal.get('score', 'unknown')}, rationale={'; '.join(_to_lines(causal.get('rationale'), limit=4)) or 'not provided'}",
        f"Contradicting evidence: {', '.join(bias_risks) if bias_risks else 'No explicit contradictions were identified.'}",
        f"Alternative explanations: {', '.join(assumptions) if assumptions else 'The data does not isolate a single cause.'}",
        f"Remaining unknowns: {validation.get('anomalies') or 'Causality is not established with the current evidence.'}",
        f"Overall confidence: {_confidence_from_story(primary_story, validation, judgment)['level']}",
    ]
    if semantic.get("relationship_type"):
        lines.insert(2, f"Relationship type: {semantic.get('relationship_type')}")
    return lines


def _build_opportunities(
    decision_ranking: List[Dict[str, Any]],
    top_stories: List[Dict[str, Any]],
    validation: Dict[str, Any],
    judgment: Dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    for index, decision in enumerate(decision_ranking[:4]):
        if decision.get("action_type") not in {"optimization", "strategic", "experiment"}:
            continue
        story = top_stories[index] if index < len(top_stories) else {}
        impact = decision.get("impact_assessment") or {}
        lines.append(
            f"Problem addressed: {story.get('insight') or decision.get('decision_summary')}"
        )
        lines.append(
            f"Evidence supporting the opportunity: {story.get('insight') or 'Decision evidence not available'}; impact={impact.get('impact_level', 'unknown')}; priority={((decision.get('priority') or {}).get('priority_score'))}"
        )
        lines.append(
            f"Why this is preferable: {decision.get('decision_summary') or 'This action aligns best with the current evidence.'}"
        )
        if story.get("recommendation_restrictions"):
            lines.append(f"Assumptions and constraints: {', '.join(_to_lines(story.get('recommendation_restrictions'), limit=4))}")
        lines.append(
            f"Expected business value: {impact.get('impact_level', 'uncertain')} {impact.get('estimated_direction', 'unclear')} impact."
        )
        lines.append(
            f"Confidence: {_confidence_from_story(story, validation, judgment)['level']}"
        )
    return lines


def _build_risks(
    top_stories: List[Dict[str, Any]],
    judgment: Dict[str, Any],
    validation: Dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    if judgment.get("contradictions_found"):
        lines.append(f"Contradictions resolved: {', '.join(_to_lines(judgment.get('contradictions_found'), limit=5))}")
    for story in top_stories[:4]:
        risks = _to_lines(story.get("bias_risks"), limit=4)
        assumptions = _to_lines(story.get("assumption_warnings"), limit=4)
        readiness = _to_lines(story.get("readiness_warnings"), limit=4)
        issues: List[str] = []
        if risks:
            issues.append(f"bias risks: {', '.join(risks)}")
        if assumptions:
            issues.append(f"assumptions: {', '.join(assumptions)}")
        if readiness:
            issues.append(f"readiness warnings: {', '.join(readiness)}")
        if validation.get("row_loss_ratio") is not None and float(validation.get("row_loss_ratio") or 0.0) > 0.1:
            issues.append(f"row loss ratio={validation.get('row_loss_ratio')}")
        if validation.get("anomalies"):
            issues.append(f"validation anomalies: {', '.join(_to_lines(validation.get('anomalies'), limit=4))}")
        if issues:
            lines.append(f"{story.get('insight')}: {'; '.join(issues)}")
    if not lines:
        lines.append("No material risk factors were isolated, but the absence of supporting evidence does not prove low risk.")
    return lines


def _build_recommendations(
    decision_ranking: List[Dict[str, Any]],
    top_stories: List[Dict[str, Any]],
    validation: Dict[str, Any],
    judgment: Dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    for index, decision in enumerate(decision_ranking[:5]):
        story = top_stories[index] if index < len(top_stories) else {}
        impact = decision.get("impact_assessment") or {}
        priority = decision.get("priority") or {}
        confidence = _confidence_from_story(story, validation, judgment)
        evidence_bits = [story.get("insight") or decision.get("decision_summary")]
        if story.get("causal_evidence"):
            causal = story.get("causal_evidence") or {}
            evidence_bits.append(f"causal_grade={causal.get('grade', 'LOW')}")
        if story.get("insight_validity") and not (story.get("insight_validity") or {}).get("valid", True):
            evidence_bits.append("not decision-ready")
        assumptions = _to_lines(story.get("assumption_warnings"), limit=4) + _to_lines(story.get("recommendation_restrictions"), limit=4)
        risks = _to_lines(story.get("bias_risks"), limit=4) + _to_lines(story.get("readiness_warnings"), limit=4)
        monitoring_kpis = _to_lines(decision.get("monitoring_kpis"), limit=4)
        lines.append(
            f"Problem addressed: {story.get('insight') or decision.get('decision_summary')}"
        )
        lines.append(f"Supporting evidence: {'; '.join([bit for bit in evidence_bits if bit])}")
        lines.append(f"Why this recommendation is appropriate: {decision.get('decision_summary') or 'It fits the current evidence profile.'}")
        lines.append(f"Assumptions: {', '.join(assumptions) if assumptions else 'No explicit assumptions were provided.'}")
        lines.append(f"Risks: {', '.join(risks) if risks else 'No explicit risks were provided.'}")
        lines.append(f"Expected business value: {impact.get('impact_level', 'uncertain')} impact, {impact.get('estimated_direction', 'unclear')} direction.")
        lines.append(f"Success measures: {', '.join(monitoring_kpis) if monitoring_kpis else 'No KPI could be inferred confidently.'}")
        lines.append(f"Confidence: {confidence['level']} ({confidence['score']})")
        lines.append(f"Priority: {priority.get('priority_level', 'unknown')} ({priority.get('priority_score', 'unknown')})")
    return lines


def _build_decision_matrix(
    decision_ranking: List[Dict[str, Any]],
    top_stories: List[Dict[str, Any]],
    validation: Dict[str, Any],
    judgment: Dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    for index, decision in enumerate(decision_ranking[:5]):
        story = top_stories[index] if index < len(top_stories) else {}
        confidence = _confidence_from_story(story, validation, judgment)
        impact = decision.get("impact_assessment") or {}
        effort = "low" if decision.get("action_type") == "investigation" else "medium" if decision.get("action_type") == "experiment" else "high"
        lines.append(
            f"{decision.get('recommended_action')} | benefits={impact.get('impact_level', 'uncertain')} | risks={', '.join(_to_lines(story.get('bias_risks'), limit=3)) or 'not explicit'} | effort={effort} | confidence={confidence['level']} ({confidence['score']}) | evidence={story.get('insight') or 'not explicit'}"
        )
    return lines


def _build_implementation(
    decision_ranking: List[Dict[str, Any]],
    top_stories: List[Dict[str, Any]],
    validation: Dict[str, Any],
    judgment: Dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    for index, decision in enumerate(decision_ranking[:3]):
        story = top_stories[index] if index < len(top_stories) else {}
        confidence = _confidence_from_story(story, validation, judgment)
        monitoring_kpis = _to_lines(decision.get("monitoring_kpis"), limit=4)
        failure_conditions = _to_lines(story.get("failure_conditions"), limit=4)
        downside_risks = _to_lines(story.get("downside_risks"), limit=4)
        lines.append(f"Execution order: start with {decision.get('recommended_action')} in the narrowest defensible scope.")
        lines.append(
            f"Dependencies: validate the supporting assumptions before scaling; confidence={confidence['level']} ({confidence['score']})."
        )
        lines.append(
            f"Potential blockers: {', '.join(failure_conditions + downside_risks) if (failure_conditions or downside_risks) else 'No explicit blocker was identified.'}"
        )
        lines.append(
            f"Monitoring approach: track {', '.join(monitoring_kpis) if monitoring_kpis else 'no KPI could be inferred confidently'}."
        )
    return lines


def _build_monitoring(
    decision_ranking: List[Dict[str, Any]],
    top_stories: List[Dict[str, Any]],
    validation: Dict[str, Any],
    judgment: Dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    for index, decision in enumerate(decision_ranking[:4]):
        story = top_stories[index] if index < len(top_stories) else {}
        kpis = _to_lines(decision.get("monitoring_kpis"), limit=4)
        if not kpis and story.get("recommended_next_step"):
            kpis = [str(story.get("recommended_next_step"))]
        if kpis:
            lines.append(f"Metric: {', '.join(kpis)}")
            lines.append("Baseline: not provided in the evidence.")
            lines.append("Target: improve relative to the current baseline; do not invent a numeric target.")
            failure_conditions = _to_lines(story.get("failure_conditions"), limit=4)
            if failure_conditions:
                lines.append(f"Warning indicators: {', '.join(failure_conditions)}")
            else:
                lines.append("Warning indicators: no explicit failure condition was provided, so watch for directional deterioration.")
            lines.append(f"Review frequency: frequent enough to detect directional movement before the full rollout matures.")
        else:
            lines.append("No KPI could be inferred confidently from the current evidence.")
        confidence = _confidence_from_story(story, validation, judgment)
        lines.append(f"Monitoring confidence: {confidence['level']} ({confidence['score']})")
    return lines


def _build_limitations(
    state: Dict[str, Any],
    evidence: Dict[str, Any],
    top_stories: List[Dict[str, Any]],
    validation: Dict[str, Any],
    judgment: Dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    dataset_profile = state.get("dataset_profile") or evidence.get("dataset_profile_json") or {}
    row_count = dataset_profile.get("row_count") or dataset_profile.get("rows")
    if row_count:
        lines.append(f"Dataset size context: about {row_count} rows were available for analysis.")
    if validation.get("row_loss_ratio") is not None:
        lines.append(f"Cleaning-induced row loss ratio: {validation.get('row_loss_ratio')}")
    if validation.get("schema_stable") is False:
        lines.append("Schema instability may weaken before/after comparisons.")
    if validation.get("anomalies"):
        lines.append(f"Validation anomalies: {', '.join(_to_lines(validation.get('anomalies'), limit=4))}")
    if evidence.get("clarification_questions"):
        lines.append(f"Open questions: {', '.join(_to_lines(evidence.get('clarification_questions'), limit=4))}")
    if top_stories:
        primary = top_stories[0]
        if primary.get("causal_evidence") and ((primary.get("causal_evidence") or {}).get("grade")) in {"LOW", None}:
            lines.append("Causality is not established, so recommendations should be treated as evidence-based hypotheses.")
        if primary.get("bias_risks"):
            lines.append(f"Bias risks remain: {', '.join(_to_lines(primary.get('bias_risks'), limit=4))}")
        if primary.get("assumption_warnings"):
            lines.append(f"Assumptions remain: {', '.join(_to_lines(primary.get('assumption_warnings'), limit=4))}")
    if judgment.get("contradictions_found"):
        lines.append(f"Conflicting outputs were suppressed for coherence: {', '.join(_to_lines(judgment.get('contradictions_found'), limit=4))}")
    if not lines:
        lines.append("No explicit limitation was captured, but absence of evidence is not evidence of absence.")
    return lines


def _build_conclusion(
    business_question: str,
    top_stories: List[Dict[str, Any]],
    judgment: Dict[str, Any],
    validation: Dict[str, Any],
) -> List[str]:
    lines = [
        f"Business question: {business_question}",
        f"Dominant reasoning: {judgment.get('dominant_reasoning', 'No dominant reasoning was identified.')}",
        f"Actionability: {judgment.get('actionability', 'unknown')}",
    ]
    if top_stories:
        confidence = _confidence_from_story(top_stories[0], validation, judgment)
        lines.append(f"Confidence: {confidence['level']} ({confidence['score']})")
        if confidence["reasons"]:
            lines.append(f"Confidence rationale: {', '.join(confidence['reasons'])}")
        if top_stories[0].get("insight"):
            lines.append(f"Primary evidence: {top_stories[0].get('insight')}")
    if judgment.get("recommended_first_action"):
        lines.append(f"Recommended first action: {judgment.get('recommended_first_action')}")
    return lines


def build_analytical_reasoning(state: Dict[str, Any]) -> Dict[str, Any]:
    evidence = state.setdefault("analysis_evidence", {})
    top_stories = evidence.get("top_stories", []) or []
    decision_ranking = evidence.get("decision_priority_ranking", []) or []
    judgment = evidence.get("judgment_summary", {}) or {}
    validation = _validation_payload(state, evidence)
    detail_map = _story_detail_map(evidence)
    primary_story = top_stories[0] if top_stories else {}

    sections = {
        "findings": _build_findings(top_stories, detail_map, validation, judgment),
        "root_cause": _build_root_cause(primary_story, validation, judgment),
        "opportunities": _build_opportunities(decision_ranking, top_stories, validation, judgment),
        "risks": _build_risks(top_stories, judgment, validation),
        "recommendations": _build_recommendations(decision_ranking, top_stories, validation, judgment),
        "decision_matrix": _build_decision_matrix(decision_ranking, top_stories, validation, judgment),
        "implementation": _build_implementation(decision_ranking, top_stories, validation, judgment),
        "monitoring": _build_monitoring(decision_ranking, top_stories, validation, judgment),
        "limitations": _build_limitations(state, evidence, top_stories, validation, judgment),
        "conclusion": _build_conclusion(state.get("business_question", "N/A"), top_stories, judgment, validation),
    }

    confidence = _confidence_from_story(primary_story, validation, judgment) if primary_story else {
        "level": "low",
        "score": float(judgment.get("global_confidence", 0) or 0),
        "reasons": ["no dominant story was available to assess confidence"],
    }

    traceability = {
        "business_question": state.get("business_question", "N/A"),
        "dataset_path": state.get("dataset_path"),
        "selected_columns": state.get("selected_columns", []) or [],
        "primary_story_signature": _story_signature(primary_story) if primary_story else None,
        "decision_signatures": [item.get("story_signature") for item in decision_ranking if item.get("story_signature")],
    }

    reasoning = {
        "sections": sections,
        "confidence": confidence,
        "traceability": traceability,
        "comparisons": {
            "available": bool(top_stories),
            "note": "Comparisons were used where the evidence provided a valid benchmark; otherwise the report explicitly notes the lack of a contextual baseline.",
        },
    }

    evidence["analytical_reasoning"] = reasoning
    state["analytical_reasoning"] = reasoning
    return reasoning
