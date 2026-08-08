from __future__ import annotations

from typing import Any, Dict, List, Sequence


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _confidence_label(value: Any) -> str:
    if isinstance(value, (int, float)):
        score = float(value)
        if score >= 75:
            return "High"
        if score >= 45:
            return "Moderate"
        return "Low"
    if value:
        return str(value).strip().capitalize()
    return "Unknown"


def _score_from_label(value: Any) -> int:
    if isinstance(value, (int, float)):
        numeric = float(value)
        return int(numeric if numeric > 1 else numeric * 100)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"high", "strong", "very high"}:
            return 85
        if lowered in {"moderate", "medium"}:
            return 60
        if lowered in {"low", "weak"}:
            return 35
    return 50


def _section_line(label: str, score: int, reason: str) -> str:
    return f"{label}: {_confidence_label(score)} ({max(0, min(100, score))}) - {reason}"


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _count_items(items: Any) -> int:
    return len(_as_list(items))


def _largest(items: Sequence[Dict[str, Any]], key: str = "severity_score") -> Dict[str, Any]:
    if not items:
        return {}
    return sorted(items, key=lambda item: float(item.get(key, 0) or 0), reverse=True)[0]


def _evidence_counts(answer: Dict[str, Any]) -> Dict[str, int]:
    breakdown = answer.get("evidence_breakdown") or {}
    return {
        "direct": _count_items(breakdown.get("direct")),
        "indirect": _count_items(breakdown.get("indirect")),
        "supporting": _count_items(breakdown.get("supporting")),
        "conflicting": _count_items(breakdown.get("conflicting")),
        "missing": _count_items(breakdown.get("missing")),
    }


def _uncertainty_sources(
    *,
    business_question: str,
    evidence: Dict[str, Any],
    answer: Dict[str, Any],
    hypotheses: Sequence[Dict[str, Any]],
    investigation_memory: Dict[str, Any],
) -> List[Dict[str, Any]]:
    counts = _evidence_counts(answer)
    diagnostics: List[Dict[str, Any]] = []
    confidence = answer.get("confidence") or {}
    status = _normalize_text(confidence.get("status") or "").lower()

    if counts["direct"] == 0:
        diagnostics.append({
            "source": "Insufficient direct evidence",
            "severity": "high",
            "severity_score": 90,
            "affected_conclusions": ["direct answer", "business conclusion"],
            "reducible": True,
            "expected_confidence_gain": 25,
            "reason": "The current evidence base does not contain enough direct support for the original question.",
        })

    if counts["conflicting"] > 0 or any(str(item.get("status", "")).lower() == "rejected" for item in hypotheses):
        diagnostics.append({
            "source": "Competing explanations",
            "severity": "high" if counts["conflicting"] > 1 else "medium",
            "severity_score": 85 if counts["conflicting"] > 1 else 65,
            "affected_conclusions": ["hypothesis confidence", "business conclusion"],
            "reducible": True,
            "expected_confidence_gain": 20,
            "reason": "There are still competing explanations that have not been fully ruled out.",
        })

    data_validation = evidence.get("data_validation") or evidence.get("cleaning_validation") or {}
    if isinstance(data_validation, dict):
        if data_validation.get("row_loss_ratio") not in (None, 0, 0.0):
            severity = "high" if float(data_validation.get("row_loss_ratio") or 0.0) > 0.15 else "medium"
            diagnostics.append({
                "source": "Data quality impact from cleaning",
                "severity": severity,
                "severity_score": 80 if severity == "high" else 60,
                "affected_conclusions": ["data quality confidence", "analytical validity"],
                "reducible": True,
                "expected_confidence_gain": 15,
                "reason": "Row loss during preparation reduces certainty in how representative the remaining data is.",
            })
        anomalies = data_validation.get("anomalies") or []
        if anomalies:
            diagnostics.append({
                "source": "Residual data anomalies",
                "severity": "medium",
                "severity_score": 60,
                "affected_conclusions": ["data quality confidence", "analytical validity"],
                "reducible": True,
                "expected_confidence_gain": 10,
                "reason": "Unresolved anomalies can still distort the confidence signal if they affect the main pattern.",
            })

    if answer.get("answer_position") == "needs_more_evidence" or status == "no":
        diagnostics.append({
            "source": "Missing direct coverage of the business question",
            "severity": "high",
            "severity_score": 88,
            "affected_conclusions": ["question coverage", "business conclusion"],
            "reducible": True,
            "expected_confidence_gain": 20,
            "reason": "The answer is still too indirect to close the question decisively.",
        })

    if investigation_memory.get("last_failure"):
        failure = investigation_memory.get("last_failure") or {}
        failure_reason = _normalize_text(failure.get("reason") or failure.get("message"))
        if failure_reason:
            diagnostics.append({
                "source": "Recent execution failure",
                "severity": "medium",
                "severity_score": 55,
                "affected_conclusions": ["analytical validity", "recommendation confidence"],
                "reducible": True,
                "expected_confidence_gain": 8,
                "reason": failure_reason,
            })

    if not evidence.get("top_stories") and not counts["direct"]:
        diagnostics.append({
            "source": "Sparse evidence store",
            "severity": "high",
            "severity_score": 92,
            "affected_conclusions": ["evidence quality", "business conclusion"],
            "reducible": True,
            "expected_confidence_gain": 25,
            "reason": "The investigation does not yet have enough structured evidence to support a strong conclusion.",
        })

    if not diagnostics:
        diagnostics.append({
            "source": "Evidence is already well aligned",
            "severity": "low",
            "severity_score": 20,
            "affected_conclusions": ["overall confidence"],
            "reducible": False,
            "expected_confidence_gain": 5,
            "reason": "No major uncertainty source is currently dominating the answer.",
        })

    diagnostics.sort(key=lambda item: float(item.get("severity_score", 0) or 0), reverse=True)
    return diagnostics


def evaluate_confidence_diagnostics(
    *,
    business_question: str,
    evidence: Dict[str, Any],
    answer: Dict[str, Any],
    hypotheses: Sequence[Dict[str, Any]] | None = None,
    investigation_memory: Dict[str, Any] | None = None,
    dataframe: Any = None,
) -> Dict[str, Any]:
    hypotheses = list(hypotheses or [])
    investigation_memory = dict(investigation_memory or {})
    counts = _evidence_counts(answer)
    confidence = answer.get("confidence") or {}
    sufficiency_status = _normalize_text(confidence.get("status") or "").lower()
    evidence_score = _score_from_label(confidence.get("evidence", {}).get("score"))
    interpretation_score = _score_from_label(confidence.get("interpretation", {}).get("score"))
    business_score = _score_from_label(confidence.get("business", {}).get("score"))
    recommendation_score = _score_from_label(confidence.get("recommendation", {}).get("score"))

    data_quality_score = 70
    data_validation = evidence.get("data_validation") or evidence.get("cleaning_validation") or {}
    if isinstance(data_validation, dict):
        row_loss = data_validation.get("row_loss_ratio")
        if row_loss is not None:
            row_loss = float(row_loss)
            data_quality_score -= int(min(40, row_loss * 100))
        if data_validation.get("anomalies"):
            data_quality_score -= min(20, len(_as_list(data_validation.get("anomalies"))) * 4)
        if data_validation.get("warnings"):
            data_quality_score -= min(15, len(_as_list(data_validation.get("warnings"))) * 3)
    if counts["direct"] == 0 and counts["indirect"] == 0:
        data_quality_score -= 10
    data_quality_score = max(0, min(100, data_quality_score))

    evidence_quality_score = max(0, min(100, evidence_score))
    if counts["direct"]:
        evidence_quality_score = min(100, evidence_quality_score + min(15, counts["direct"] * 5))
    if counts["conflicting"]:
        evidence_quality_score = max(0, evidence_quality_score - min(25, counts["conflicting"] * 8))

    analytical_validity_score = max(0, min(100, interpretation_score))
    if evidence.get("analytical_reasoning"):
        analytical_validity_score = min(100, analytical_validity_score + 8)
    if evidence.get("judgment_summary", {}).get("contradictions_found"):
        analytical_validity_score = max(0, analytical_validity_score - 10)

    question_coverage_score = 45
    if counts["direct"]:
        question_coverage_score += 30
    if counts["indirect"]:
        question_coverage_score += 10
    if answer.get("answer_position") == "direct":
        question_coverage_score += 10
    if sufficiency_status == "no":
        question_coverage_score -= 20
    question_coverage_score = max(0, min(100, question_coverage_score))

    supported = sum(1 for item in hypotheses if str(item.get("status", "")).lower() == "supported")
    rejected = sum(1 for item in hypotheses if str(item.get("status", "")).lower() == "rejected")
    hypothesis_score = 40 + min(25, supported * 10) - min(20, rejected * 8)
    if counts["conflicting"]:
        hypothesis_score -= min(15, counts["conflicting"] * 5)
    hypothesis_score = max(0, min(100, hypothesis_score))

    business_interpretation_score = max(0, min(100, business_score))
    if sufficiency_status == "partial":
        business_interpretation_score -= 5
    if sufficiency_status == "no":
        business_interpretation_score -= 15

    alternative_explanation_score = 60
    if counts["conflicting"]:
        alternative_explanation_score -= min(30, counts["conflicting"] * 10)
    if rejected:
        alternative_explanation_score -= min(15, rejected * 5)
    if counts["direct"] and not counts["conflicting"]:
        alternative_explanation_score += 10
    alternative_explanation_score = max(0, min(100, alternative_explanation_score))

    recommendation_confidence_score = max(0, min(100, recommendation_score))
    if sufficiency_status == "yes" and counts["conflicting"] == 0:
        recommendation_confidence_score = min(100, recommendation_confidence_score + 10)

    overall_score = round(
        (0.10 * data_quality_score)
        + (0.18 * evidence_quality_score)
        + (0.14 * analytical_validity_score)
        + (0.12 * question_coverage_score)
        + (0.12 * hypothesis_score)
        + (0.12 * business_interpretation_score)
        + (0.11 * alternative_explanation_score)
        + (0.11 * recommendation_confidence_score)
    )

    uncertainty_sources = _uncertainty_sources(
        business_question=business_question,
        evidence=evidence,
        answer=answer,
        hypotheses=hypotheses,
        investigation_memory=investigation_memory,
    )
    largest = _largest(uncertainty_sources)
    reducible_uncertainties = [item for item in uncertainty_sources if item.get("reducible")]
    expected_gain = max([int(item.get("expected_confidence_gain", 0) or 0) for item in reducible_uncertainties], default=0)
    more_analysis_helpful = bool(reducible_uncertainties and expected_gain >= 8)
    new_data_helpful = any(
        item.get("source") in {"Insufficient direct evidence", "Sparse evidence store", "Missing direct coverage of the business question"}
        for item in uncertainty_sources
    ) or data_quality_score < 55
    diminishing_returns = sufficiency_status == "yes" and expected_gain < 10 and not any(
        item.get("severity") in {"high", "medium"} for item in uncertainty_sources
    )

    if sufficiency_status == "yes" and not diminishing_returns:
        stopping_recommendation = "Continue only if the next task reduces a named uncertainty."
    elif sufficiency_status == "yes":
        stopping_recommendation = "Finish the investigation."
    elif sufficiency_status == "partial":
        stopping_recommendation = "Continue with a targeted investigation that attacks the largest remaining uncertainty."
    else:
        stopping_recommendation = "Keep investigating or gather more data before finalising the answer."

    if diminishing_returns:
        stopping_recommendation = "Finish the investigation."

    fastest_path = largest if largest else {}
    if fastest_path:
        if fastest_path["source"] in {"Competing explanations", "Alternative explanations"}:
            next_action = "Test the strongest alternative hypothesis."
        elif fastest_path["source"] in {"Insufficient direct evidence", "Missing direct coverage of the business question", "Sparse evidence store"}:
            next_action = "Run a more direct analysis or collect the missing variable/data."
        elif fastest_path["source"] in {"Data quality impact from cleaning", "Residual data anomalies"}:
            next_action = "Repair the data quality issue or re-run the analysis with a cleaner dataset."
        elif fastest_path["source"] == "Recent execution failure":
            next_action = "Correct the failed step and rerun the affected analysis."
        else:
            next_action = "Target the dominant uncertainty with the most direct available analysis."
    else:
        next_action = "Finish the investigation."

    why_confidence_high = []
    why_confidence_low = []
    if counts["direct"]:
        why_confidence_high.append(f"{counts['direct']} direct evidence item(s) answer the question head-on")
    if supported:
        why_confidence_high.append(f"{supported} hypothesis/hypotheses are already supported")
    if data_quality_score >= 75:
        why_confidence_high.append("data quality is not the primary limiter")
    if counts["conflicting"]:
        why_confidence_low.append(f"{counts['conflicting']} conflicting evidence item(s) still remain")
    if counts["direct"] == 0:
        why_confidence_low.append("the evidence is still indirect")
    if data_quality_score < 60:
        why_confidence_low.append("data quality or preparation is still constraining certainty")
    if rejected:
        why_confidence_low.append(f"{rejected} hypothesis/hypotheses are still rejected or unresolved")

    return {
        "business_question": business_question,
        "confidence": {
            "data_quality": {"score": data_quality_score, "label": _confidence_label(data_quality_score)},
            "evidence_quality": {"score": evidence_quality_score, "label": _confidence_label(evidence_quality_score)},
            "analytical_validity": {"score": analytical_validity_score, "label": _confidence_label(analytical_validity_score)},
            "question_coverage": {"score": question_coverage_score, "label": _confidence_label(question_coverage_score)},
            "hypothesis_confidence": {"score": hypothesis_score, "label": _confidence_label(hypothesis_score)},
            "business_interpretation": {"score": business_interpretation_score, "label": _confidence_label(business_interpretation_score)},
            "alternative_explanation": {"score": alternative_explanation_score, "label": _confidence_label(alternative_explanation_score)},
            "recommendation": {"score": recommendation_confidence_score, "label": _confidence_label(recommendation_confidence_score)},
            "overall": {"score": overall_score, "label": _confidence_label(overall_score)},
        },
        "evidence_sufficiency": {
            "status": sufficiency_status or "partial",
            "can_answer": sufficiency_status == "yes",
            "can_partially_answer": sufficiency_status in {"yes", "partial"},
            "would_more_analysis_help": more_analysis_helpful,
            "would_new_data_help": new_data_helpful,
            "diminishing_returns": diminishing_returns,
            "stopping_recommendation": stopping_recommendation,
        },
        "uncertainty_sources": uncertainty_sources,
        "largest_uncertainty": largest,
        "fastest_path_to_strengthen_confidence": {
            "action": next_action,
            "expected_confidence_gain": int(fastest_path.get("expected_confidence_gain", expected_gain) or expected_gain),
            "source": fastest_path.get("source") if fastest_path else "None",
            "reducible": bool(fastest_path.get("reducible", False)) if fastest_path else False,
        },
        "why_confidence": {
            "strongest_reasons": why_confidence_high[:4],
            "largest_risks": why_confidence_low[:4],
            "biggest_assumption": (answer.get("key_assumptions") or ["No major assumption identified."])[0],
            "biggest_evidence_gap": (answer.get("remaining_uncertainty") or ["No major gap identified."])[0],
        },
        "recommendation_alignment": {
            "should_continue": sufficiency_status != "yes" or not diminishing_returns,
            "should_stop": sufficiency_status == "yes" and diminishing_returns,
            "preferred_action": stopping_recommendation,
        },
    }


def build_confidence_diagnostics_sections(diagnostics: Dict[str, Any]) -> Dict[str, List[str]]:
    if not diagnostics:
        return {}

    confidence = diagnostics.get("confidence") or {}
    sufficiency = diagnostics.get("evidence_sufficiency") or {}
    uncertainty_sources = diagnostics.get("uncertainty_sources") or []
    largest = diagnostics.get("largest_uncertainty") or {}
    fastest = diagnostics.get("fastest_path_to_strengthen_confidence") or {}
    why = diagnostics.get("why_confidence") or {}
    recommendation = diagnostics.get("recommendation_alignment") or {}
    largest_source = _normalize_text(largest.get("source") or "No dominant uncertainty source identified.")
    largest_reason = _normalize_text(largest.get("reason") or "")
    continue_flag = bool(recommendation.get("should_continue"))
    stop_flag = bool(recommendation.get("should_stop"))
    more_analysis_helpful = bool(sufficiency.get("would_more_analysis_help"))
    new_data_helpful = bool(sufficiency.get("would_new_data_help"))
    diminishing_returns = bool(sufficiency.get("diminishing_returns"))

    confidence_lines = [
        _section_line("Data quality confidence", int(confidence.get("data_quality", {}).get("score", 0)), "Confidence in the input data after validation and cleaning."),
        _section_line("Evidence quality confidence", int(confidence.get("evidence_quality", {}).get("score", 0)), "How directly and consistently the evidence supports the answer."),
        _section_line("Analytical validity confidence", int(confidence.get("analytical_validity", {}).get("score", 0)), "How sound the analysis and judgment trail are."),
        _section_line("Question coverage confidence", int(confidence.get("question_coverage", {}).get("score", 0)), "How fully the evidence addresses the original question."),
        _section_line("Hypothesis confidence", int(confidence.get("hypothesis_confidence", {}).get("score", 0)), "How strongly the leading hypothesis is supported relative to alternatives."),
        _section_line("Business interpretation confidence", int(confidence.get("business_interpretation", {}).get("score", 0)), "How safe it is to turn the evidence into a business conclusion."),
        _section_line("Alternative explanation confidence", int(confidence.get("alternative_explanation", {}).get("score", 0)), "How well competing explanations have been ruled out."),
        _section_line("Recommendation confidence", int(confidence.get("recommendation", {}).get("score", 0)), "How reliable the next-step recommendation is."),
        _section_line("Overall confidence", int(confidence.get("overall", {}).get("score", 0)), "The combined confidence view across all dimensions."),
    ]

    sufficiency_lines = [
        f"Can we answer the question now? {'Yes' if sufficiency.get('can_answer') else 'Partially' if sufficiency.get('can_partially_answer') else 'No'}.",
        f"Would more analysis meaningfully improve the answer? {'Yes' if more_analysis_helpful else 'Not materially'}.",
        f"Would new data help more than another analysis pass? {'Yes' if new_data_helpful else 'Not clearly'}.",
        f"Has the investigation reached diminishing returns? {'Yes' if diminishing_returns else 'No'}.",
        f"Stopping recommendation: {sufficiency.get('stopping_recommendation') or 'Continue investigating.'}",
    ]

    uncertainty_lines = []
    if largest_source:
        uncertainty_lines.append(f"What is limiting confidence most? {largest_source}.")
        if largest_reason:
            uncertainty_lines.append(f"Why this is the main limiter: {largest_reason}")
    if stop_flag:
        uncertainty_lines.append("The strongest signal says we already know enough to stop, and further analysis is unlikely to change the conclusion.")
    elif continue_flag:
        uncertainty_lines.append("The strongest signal says we should continue, but only with a targeted step that reduces the named uncertainty.")
    else:
        uncertainty_lines.append("Confidence is still bounded by a few unresolved issues, so the answer remains provisional.")
    for item in uncertainty_sources[:5]:
        affected = ", ".join(item.get("affected_conclusions") or [])
        uncertainty_lines.append(
            f"{item.get('source')} | severity={item.get('severity')} | reducible={'yes' if item.get('reducible') else 'no'} | expected gain={item.get('expected_confidence_gain')} | affected={affected or 'unspecified'}"
        )
        reason = _normalize_text(item.get("reason"))
        if reason:
            uncertainty_lines.append(f"   why it matters: {reason}")
    if largest:
        uncertainty_lines.append(
            f"Largest uncertainty: {largest.get('source')} | severity={largest.get('severity')} | expected gain={largest.get('expected_confidence_gain')}"
        )

    stopping_lines = [
        f"Fastest path to stronger confidence: {fastest.get('action') or 'Finish the investigation.'}",
        f"Expected confidence gain: {fastest.get('expected_confidence_gain', 'unknown')}",
        f"Why confidence is not higher: {', '.join(why.get('largest_risks') or ['No dominant risk was identified.'])}",
        f"Why confidence is not lower: {', '.join(why.get('strongest_reasons') or ['The current evidence still supports a stable answer.'])}",
        f"Biggest assumption: {why.get('biggest_assumption') or 'None identified.'}",
        f"Biggest evidence gap: {why.get('biggest_evidence_gap') or 'None identified.'}",
    ]
    if more_analysis_helpful:
        stopping_lines.append("Another investigation is still likely to help, but only if it targets the main uncertainty directly.")
    else:
        stopping_lines.append("Another broad analysis pass is unlikely to move the conclusion much.")
    if recommendation:
        stopping_lines.append(f"Preferred action: {recommendation.get('preferred_action') or 'Continue investigating.'}")
        stopping_lines.append(f"Should stop now: {'Yes' if recommendation.get('should_stop') else 'No'}")

    return {
        "Confidence Diagnostics": confidence_lines,
        "Evidence Sufficiency": sufficiency_lines,
        "Uncertainty Diagnostics": uncertainty_lines or ["No major uncertainty source is currently visible."],
        "Stopping Criteria": stopping_lines,
    }


def render_confidence_diagnostics_report(diagnostics: Dict[str, Any]) -> str:
    sections = build_confidence_diagnostics_sections(diagnostics)
    if not sections:
        return ""
    lines: List[str] = ["================ CONFIDENCE DIAGNOSTICS ================"]
    for title, body_lines in sections.items():
        lines.append("")
        lines.append(title)
        for line in body_lines:
            lines.append(f"- {line}")
    return "\n".join(lines)


def _decision_gate(status: bool, name: str, reason: str, *, reducible: bool = False, severity: str = "medium") -> Dict[str, Any]:
    return {
        "gate": name,
        "passed": bool(status),
        "reason": reason,
        "reducible": reducible,
        "severity": severity,
    }


def _infer_risk_level(business_question: str, evidence: Dict[str, Any], answer: Dict[str, Any], diagnostics: Dict[str, Any]) -> str:
    text = " ".join(
        [
            _normalize_text(business_question),
            _normalize_text(answer.get("direct_answer") or answer.get("business_interpretation") or ""),
            _normalize_text(evidence.get("business_context") or ""),
        ]
    ).lower()
    high_risk_markers = [
        "multi-million",
        "million dollar",
        "investment",
        "capital allocation",
        "safety",
        "regulatory",
        "compliance",
        "litigation",
        "health",
        "fraud",
        "production outage",
        "loss",
        "shutdown",
    ]
    low_risk_markers = [
        "dashboard",
        "formatting",
        "layout",
        "label",
        "presentation",
        "visual polish",
        "report styling",
    ]
    if any(marker in text for marker in high_risk_markers):
        return "high"
    if any(marker in text for marker in low_risk_markers):
        return "low"
    decision_context = diagnostics.get("decision_context") or evidence.get("decision_context") or {}
    explicit = _normalize_text(decision_context.get("risk_level") or decision_context.get("stakes") or "")
    if explicit in {"high", "medium", "low"}:
        return explicit
    if explicit in {"critical", "severe", "very high"}:
        return "high"
    if explicit in {"minor", "limited", "very low"}:
        return "low"
    return "medium"


def _answer_needs_human_guidance(answer: Dict[str, Any]) -> bool:
    text = " ".join(
        [
            _normalize_text(answer.get("direct_answer") or ""),
            _normalize_text(answer.get("business_interpretation") or ""),
            _normalize_text(answer.get("reasoning") or ""),
            _normalize_text(answer.get("remaining_uncertainty") or ""),
        ]
    ).lower()
    if not text:
        return False
    triggers = [
        "cannot yet be answered",
        "does not directly address",
        "does not directly answer",
        "still falls short",
        "not yet sufficient",
        "provisional",
        "indirect only",
        "still indirect",
        "needs direct evidence",
        "needs human guidance",
        "not directly provide",
    ]
    return any(trigger in text for trigger in triggers)


def evaluate_investigation_decision(
    *,
    business_question: str,
    evidence: Dict[str, Any],
    answer: Dict[str, Any],
    diagnostics: Dict[str, Any] | None = None,
    hypotheses: Sequence[Dict[str, Any]] | None = None,
    investigation_memory: Dict[str, Any] | None = None,
    collaborative_mode: bool = False,
    dataframe: Any = None,
) -> Dict[str, Any]:
    diagnostics = diagnostics or evaluate_confidence_diagnostics(
        business_question=business_question,
        evidence=evidence,
        answer=answer,
        hypotheses=hypotheses,
        investigation_memory=investigation_memory,
        dataframe=dataframe,
    )
    hypotheses = list(hypotheses or [])
    investigation_memory = dict(investigation_memory or {})
    confidence = diagnostics.get("confidence") or {}
    sufficiency = diagnostics.get("evidence_sufficiency") or {}
    uncertainty_sources = diagnostics.get("uncertainty_sources") or []
    largest = diagnostics.get("largest_uncertainty") or {}
    fastest = diagnostics.get("fastest_path_to_strengthen_confidence") or {}
    counts = {
        "direct": len((answer.get("evidence_breakdown") or {}).get("direct") or []),
        "indirect": len((answer.get("evidence_breakdown") or {}).get("indirect") or []),
        "supporting": len((answer.get("evidence_breakdown") or {}).get("supporting") or []),
        "conflicting": len((answer.get("evidence_breakdown") or {}).get("conflicting") or []),
    }
    conflicting_items = list((answer.get("evidence_breakdown") or {}).get("conflicting") or [])
    risk_level = _infer_risk_level(business_question, evidence, answer, diagnostics)
    stop_threshold_by_risk = {"low": 65, "medium": 70, "high": 80}
    stop_threshold = stop_threshold_by_risk.get(risk_level, 70)

    coverage_gate = _decision_gate(
        status=(
            int(confidence.get("question_coverage", {}).get("score", 0) or 0) >= 60
            and (
                counts["direct"] >= 2
                or (counts["direct"] >= 1 and (counts["indirect"] + counts["supporting"]) >= 2)
                or (answer.get("answer_position") == "direct" and counts["direct"] >= 1 and counts["supporting"] >= 1)
            )
        ),
        name="Question Coverage",
        reason=(
            "The answer directly addresses the business question with enough breadth to support the conclusion."
            if int(confidence.get("question_coverage", {}).get("score", 0) or 0) >= 60
            and (
                counts["direct"] >= 2
                or (counts["direct"] >= 1 and (counts["indirect"] + counts["supporting"]) >= 2)
                or (answer.get("answer_position") == "direct" and counts["direct"] >= 1 and counts["supporting"] >= 1)
            )
            else "The current evidence answers only part of the question or rests on too narrow a slice of the analysis."
        ),
        reducible=True,
        severity="high" if int(confidence.get("question_coverage", {}).get("score", 0) or 0) < 60 else "low",
    )
    evidence_gate = _decision_gate(
        status=int(confidence.get("evidence_quality", {}).get("score", 0) or 0) >= 55,
        name="Evidence Quality",
        reason=(
            "The evidence base is strong enough to support a business conclusion."
            if int(confidence.get("evidence_quality", {}).get("score", 0) or 0) >= 55
            else "The evidence is still too weak or too narrow to support a reliable conclusion."
        ),
        reducible=True,
        severity="high" if int(confidence.get("evidence_quality", {}).get("score", 0) or 0) < 55 else "low",
    )
    consistency_gate = _decision_gate(
        status=int(confidence.get("alternative_explanation", {}).get("score", 0) or 0) >= 55 and not conflicting_items and not any(
            any(term in _normalize_text(item.get("source")).lower() for term in ("competing", "conflict", "contradict", "alternative", "disagreement"))
            and item.get("severity") == "high"
            for item in uncertainty_sources
        ),
        name="Evidence Consistency",
        reason=(
            "The major analytical methods are aligned and no material contradiction remains."
            if int(confidence.get("alternative_explanation", {}).get("score", 0) or 0) >= 55 and not conflicting_items
            else "The methods or hypotheses are still pulling in different directions."
        ),
        reducible=True,
        severity="high" if int(confidence.get("alternative_explanation", {}).get("score", 0) or 0) < 55 else "medium",
    )
    uncertainty_gate = _decision_gate(
        status=not any(item.get("severity") == "high" and item.get("reducible", False) for item in uncertainty_sources),
        name="Remaining Uncertainty",
        reason=(
            "The remaining uncertainty is manageable and no major unanswered question is blocking a decision."
            if not any(item.get("severity") == "high" and item.get("reducible", False) for item in uncertainty_sources)
            else "There is still a major unresolved question that could change the answer."
        ),
        reducible=True,
        severity="high" if any(item.get("severity") == "high" and item.get("reducible", False) for item in uncertainty_sources) else "low",
    )
    mandatory_gates = [coverage_gate, evidence_gate, consistency_gate, uncertainty_gate]
    failed_gates = [gate for gate in mandatory_gates if not gate["passed"]]
    mandatory_pass = not failed_gates

    information_gain_score = int(sufficiency.get("would_more_analysis_help", False)) * 50 + int(bool(fastest.get("expected_confidence_gain", 0) or 0))
    decision_impact_score = int(confidence.get("business_interpretation", {}).get("score", 0) or 0)
    actionability_score = int(confidence.get("recommendation", {}).get("score", 0) or 0)
    risk_score = max(
        0,
        100
        - int(confidence.get("overall", {}).get("score", 0) or 0)
        - min(20, len([item for item in uncertainty_sources if item.get("severity") == "high"]) * 5),
    )
    risk_pressure = max(0, 100 - risk_score)

    continuing_value = max(information_gain_score, decision_impact_score, actionability_score)
    expected_gain = int(fastest.get("expected_confidence_gain", 0) or 0)
    answer_ready = str(answer.get("answer_position") or "").strip().lower() == "direct"
    semantic_guidance_needed = _answer_needs_human_guidance(answer)
    should_stop = mandatory_pass and bool(sufficiency.get("diminishing_returns")) and int(confidence.get("overall", {}).get("score", 0) or 0) >= stop_threshold and answer_ready
    if mandatory_pass and risk_level == "high":
        should_stop = should_stop and int(confidence.get("overall", {}).get("score", 0) or 0) >= 80
    should_continue = not mandatory_pass and bool(fastest.get("reducible", False)) and expected_gain >= 15 and not should_stop
    answer_is_not_direct = not answer_ready
    should_ask_user = collaborative_mode and not should_stop and (
        answer_is_not_direct
        or semantic_guidance_needed
        or (not mandatory_pass and not should_continue)
        or (mandatory_pass and bool(sufficiency.get("diminishing_returns")) and not answer_ready)
    )
    if collaborative_mode and not mandatory_pass and expected_gain < 15 and not should_stop:
        should_ask_user = True

    if should_stop:
        decision = "STOP"
    elif should_ask_user:
        decision = "ASK_USER"
    elif should_continue:
        decision = "CONTINUE"
    else:
        decision = "ASK_USER" if collaborative_mode else "CONTINUE"

    if decision == "ASK_USER" and not collaborative_mode:
        decision = "CONTINUE"

    if decision == "STOP":
        decision_reasoning = [
            "The original question has been answered well enough for a reliable business recommendation.",
            "The evidence is consistent across the main analytical paths.",
            "No significant contradiction remains, and further analysis is unlikely to materially change the conclusion.",
        ]
    elif decision == "CONTINUE":
        if not mandatory_pass:
            decision_reasoning = [
                "The investigation is not yet ready to close because one or more mandatory evidence gates are still failing.",
                "The most important open issue should be targeted before the recommendation becomes decision-ready.",
            ]
        else:
            decision_reasoning = [
                "The investigation still has enough unresolved uncertainty that one more targeted analytical step is justified.",
                "The next step should be chosen to reduce the largest remaining uncertainty, not to add more analysis for its own sake.",
            ]
    else:
        decision_reasoning = [
            "The evidence base is promising, but the expected benefit of another analysis step is uncertain.",
            "Human guidance is needed to choose the most valuable next investigation path.",
        ]
        if not answer_ready or semantic_guidance_needed:
            decision_reasoning.insert(
                0,
                "The current answer is still provisional and does not yet give a direct answer to the business question.",
            )
        if mandatory_pass and bool(sufficiency.get("diminishing_returns")) and (not answer_ready or semantic_guidance_needed):
            decision_reasoning.insert(
                1 if decision_reasoning else 0,
                "The analysis has reached diminishing returns, but the current evidence still does not produce a decisive answer to the business question.",
            )

    if failed_gates:
        decision_reasoning.append("Mandatory gates that are still open: " + ", ".join(gate["gate"] for gate in failed_gates))

    if largest:
        remaining_uncertainties = [
            f"{largest.get('source')} is the main remaining uncertainty." if largest.get("source") else "A major uncertainty remains unresolved.",
        ]
    else:
        remaining_uncertainties = [
            "No major uncertainty source currently dominates the conclusion.",
        ]
    for item in uncertainty_sources[:3]:
        note = _normalize_text(item.get("reason"))
        if note and note not in remaining_uncertainties:
            remaining_uncertainties.append(note)

    if decision == "STOP":
        recommended_next_step = "Finish the investigation and present the recommendation."
    elif decision == "CONTINUE":
        recommended_next_step = fastest.get("action") or "Run the most targeted follow-up investigation."
    else:
        recommended_next_step = "Ask the analyst to choose the next investigation path or confirm that the current answer is sufficient."

    confidence_score = int(round(
        (0.35 * int(confidence.get("overall", {}).get("score", 0) or 0))
        + (0.20 * continuing_value)
        + (0.15 * (100 - risk_score))
        + (0.15 * (100 if mandatory_pass else 35))
        + (0.15 * (100 if decision == "STOP" else 55 if decision == "CONTINUE" else 45))
    ))
    confidence_score = max(0, min(100, confidence_score))

    audit_log = [
        f"question_coverage={coverage_gate['passed']}",
        f"evidence_quality={evidence_gate['passed']}",
        f"evidence_consistency={consistency_gate['passed']}",
        f"remaining_uncertainty={uncertainty_gate['passed']}",
        f"mandatory_pass={mandatory_pass}",
        f"collaborative_mode={collaborative_mode}",
    ]

    internal_metrics = {
        "question_coverage_score": confidence.get("question_coverage", {}).get("score", 0),
        "evidence_quality_score": confidence.get("evidence_quality", {}).get("score", 0),
        "consistency_score": confidence.get("alternative_explanation", {}).get("score", 0),
        "remaining_uncertainty_score": 100 - min(100, len(uncertainty_sources) * 10),
        "information_gain_score": information_gain_score,
        "decision_impact_score": decision_impact_score,
        "actionability_score": actionability_score,
        "risk_score": risk_score,
        "risk_pressure": risk_pressure,
        "risk_level": risk_level,
        "stop_threshold": stop_threshold,
        "decision_score": confidence_score,
    }

    return {
        "decision": decision,
        "confidence": {
            "score": confidence_score,
            "label": _confidence_label(confidence_score),
            "explanation": (
                "The decision is grounded in whether the mandatory evidence gates passed and how much unresolved uncertainty remains."
            ),
        },
        "reasoning": decision_reasoning,
        "failed_gates": [gate for gate in failed_gates],
        "remaining_uncertainties": remaining_uncertainties[:5],
        "recommended_next_step": recommended_next_step,
        "question_for_user": (
            "Please choose the next capability or confirm that the current answer is sufficient."
            if decision == "ASK_USER"
            else "No further user guidance is required yet."
        ),
        "internal_metrics": internal_metrics,
        "audit_log": audit_log,
        "gates": mandatory_gates,
    }


def build_investigation_decision_bundle(
    *,
    business_question: str,
    evidence: Dict[str, Any],
    answer: Dict[str, Any],
    hypotheses: Sequence[Dict[str, Any]] | None = None,
    investigation_memory: Dict[str, Any] | None = None,
    collaborative_mode: bool = False,
    dataframe: Any = None,
) -> Dict[str, Any]:
    """
    Build the full confidence and investigation decision package from one place.

    This keeps the decision logic centralized in the diagnostics layer so the
    synthesis, reporting, and workflow modules can consume a single result
    instead of independently reconstructing the same reasoning.
    """
    diagnostics = evaluate_confidence_diagnostics(
        business_question=business_question,
        evidence=evidence,
        answer=answer,
        hypotheses=hypotheses,
        investigation_memory=investigation_memory,
        dataframe=dataframe,
    )
    decision = evaluate_investigation_decision(
        business_question=business_question,
        evidence=evidence,
        answer=answer,
        diagnostics=diagnostics,
        hypotheses=hypotheses,
        investigation_memory=investigation_memory,
        collaborative_mode=collaborative_mode,
        dataframe=dataframe,
    )
    return {
        "confidence_diagnostics": diagnostics,
        "confidence_diagnostics_sections": build_confidence_diagnostics_sections(diagnostics),
        "investigation_decision": decision,
        "investigation_decision_sections": build_investigation_decision_sections(decision),
    }


def build_investigation_decision_sections(decision: Dict[str, Any], *, include_internal_metrics: bool = False) -> Dict[str, List[str]]:
    if not decision:
        return {}
    reasoning = decision.get("reasoning") or []
    failed_gates = decision.get("failed_gates") or []
    remaining = decision.get("remaining_uncertainties") or []
    internal_metrics = decision.get("internal_metrics") or {}
    sections = {
        "Investigation Decision": [
            f"Decision: {decision.get('decision') or 'CONTINUE'}",
            f"Confidence level: {decision.get('confidence', {}).get('label', 'Unknown')} ({decision.get('confidence', {}).get('score', 'n/a')})",
            f"Recommended next step: {decision.get('recommended_next_step') or 'Continue investigating.'}",
        ],
        "Decision Reasoning": list(reasoning[:5]) or ["No decision reasoning was recorded."],
        "Remaining Uncertainties": list(remaining[:5]) or ["No major uncertainty remains visible."],
    }
    if failed_gates:
        sections["Failed Gates"] = [
            f"{gate.get('gate')}: {'passed' if gate.get('passed') else 'failed'} - {gate.get('reason')}"
            for gate in failed_gates[:4]
        ]
    if include_internal_metrics and internal_metrics:
        sections["Internal Decision Metrics"] = [
            f"decision score: {internal_metrics.get('decision_score')}",
            f"information gain signal: {internal_metrics.get('information_gain_score')}",
            f"risk signal: {internal_metrics.get('risk_score')}",
        ]
    return sections


def render_investigation_decision_report(decision: Dict[str, Any], *, include_internal_metrics: bool = False) -> str:
    sections = build_investigation_decision_sections(decision, include_internal_metrics=include_internal_metrics)
    if not sections:
        return ""
    lines: List[str] = ["================ INVESTIGATION DECISION ================"]
    for title, body_lines in sections.items():
        lines.append("")
        lines.append(title)
        for line in body_lines:
            lines.append(f"- {line}")
    return "\n".join(lines)
