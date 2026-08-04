from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _tokens(value: Any) -> set[str]:
    text = _normalize_text(value).lower()
    tokens = set()
    for token in text.split():
        token = token.strip(".,:;!?()[]{}<>\"'")
        if len(token) >= 4:
            tokens.add(token)
    return tokens


def _overlap_score(*values: Any) -> int:
    if not values:
        return 0
    base = _tokens(values[0])
    if not base:
        return 0
    others: set[str] = set()
    for value in values[1:]:
        others |= _tokens(value)
    return len(base & others)


def _classify_level(score: int) -> str:
    if score >= 75:
        return "High"
    if score >= 45:
        return "Medium"
    if score >= 15:
        return "Low"
    return "None"


def _score_text_match(original_question: str, *texts: Any) -> int:
    question_tokens = _tokens(original_question)
    if not question_tokens:
        return 0
    matched = set()
    for text in texts:
        matched |= (question_tokens & _tokens(text))
    score = min(100, 20 + len(matched) * 18)
    if any(term in _normalize_text(" ".join(_normalize_text(text) for text in texts)).lower() for term in ["challenge", "compare", "refine", "rerun", "repeat", "investigate"]):
        score = min(100, score + 10)
    return score


def _score_continuity(task_request: str, current_hypothesis: str, prior_findings: Sequence[Any]) -> int:
    continuity = 20 + _overlap_score(task_request, current_hypothesis) * 20
    if prior_findings:
        continuity += min(20, len(prior_findings) * 3)
    request = _normalize_text(task_request).lower()
    if any(term in request for term in ["compare", "challenge", "refine", "rerun", "repeat"]):
        continuity += 10
    if any(term in request for term in ["new investigation", "new question"]) and not current_hypothesis:
        continuity += 5
    return max(0, min(100, continuity))


def _score_information_gain(task_request: str, summary: Dict[str, Any], prior_findings: Sequence[Any]) -> int:
    request = _normalize_text(task_request).lower()
    gain = 35
    if any(term in request for term in ["challenge", "test", "verify", "falsify"]):
        gain += 30
    elif any(term in request for term in ["compare", "contrast"]):
        gain += 25
    elif any(term in request for term in ["refine", "rerun", "repeat"]):
        gain += 20
    else:
        gain += 10
    if summary.get("analysis_plan"):
        gain += 10
    if summary.get("analysis_signals"):
        gain += 10
    if summary.get("current_understanding") and prior_findings:
        gain += 5 if _normalize_text(summary.get("current_understanding")) not in {_normalize_text(item) for item in prior_findings} else -5
    return max(0, min(100, gain))


def _score_validity(summary: Dict[str, Any], dataframe: Any = None) -> int:
    selected_columns = summary.get("selected_columns") or []
    if dataframe is None:
        return 60 if summary.get("analysis_plan") or summary.get("analysis_signals") else 40
    try:
        available = {str(column).strip().lower() for column in getattr(dataframe, "columns", [])}
    except Exception:
        available = set()
    if not selected_columns:
        return 50 if summary.get("analysis_plan") else 35
    matches = sum(1 for column in selected_columns if str(column).strip().lower() in available)
    if matches == len(selected_columns):
        return 90
    if matches:
        return 60 + int((matches / max(1, len(selected_columns))) * 20)
    return 20


def _expected_contribution(question_relevance: int, continuity: int, information_gain: int) -> str:
    if question_relevance >= 75 and continuity >= 60:
        return "Directly advances the primary business question."
    if question_relevance >= 45:
        return "Clarifies a related part of the investigation and reduces uncertainty."
    if continuity >= 60:
        return "Extends the current branch of analysis but may not directly answer the question."
    return "Acts as supporting context rather than a primary answer path."


def _actual_contribution(passed: bool, summary: Dict[str, Any]) -> str:
    if passed:
        return _normalize_text(summary.get("current_understanding") or summary.get("narrative") or "The task strengthened the active investigation.")
    return _normalize_text(summary.get("current_understanding") or summary.get("narrative") or "The task produced supporting evidence only.")


def evaluate_investigation_integrity(
    *,
    original_question: str,
    task_request: str,
    summary: Dict[str, Any],
    current_hypothesis: str = "",
    prior_findings: Sequence[Any] | None = None,
    dataframe: Any = None,
) -> Dict[str, Any]:
    prior_findings = list(prior_findings or [])
    finding_relevance = _score_text_match(
        original_question,
        summary.get("task_finding"),
        summary.get("current_understanding"),
        summary.get("narrative"),
        summary.get("analysis_story"),
        summary.get("report_excerpt"),
    )
    request_relevance = _score_text_match(original_question, task_request)
    question_relevance = max(0, min(100, round((0.75 * finding_relevance) + (0.25 * request_relevance))))
    continuity = _score_continuity(task_request, current_hypothesis, prior_findings)
    information_gain = _score_information_gain(task_request, summary, prior_findings)
    validity = _score_validity(summary, dataframe=dataframe)

    overall = round(
        (0.40 * question_relevance)
        + (0.20 * continuity)
        + (0.20 * information_gain)
        + (0.20 * validity)
    )

    relevance_level = _classify_level(question_relevance)
    continuity_level = _classify_level(continuity)
    information_gain_level = _classify_level(information_gain)
    validity_level = _classify_level(validity)
    overall_level = _classify_level(overall)

    branch_type = "primary"
    if relevance_level in {"Low", "None"}:
        branch_type = "supporting"
    if relevance_level == "None" or validity_level == "None":
        branch_type = "supporting"
    if continuity_level in {"Low", "None"} and relevance_level in {"Low", "None"}:
        branch_type = "new_branch"

    should_promote = (
        relevance_level in {"High", "Medium"}
        and continuity_level in {"High", "Medium"}
        and information_gain_level in {"High", "Medium"}
        and validity_level in {"High", "Medium"}
    )

    return {
        "original_question": original_question,
        "task_request": task_request,
        "current_hypothesis": current_hypothesis,
        "question_relevance": {"score": question_relevance, "level": relevance_level},
        "continuity": {"score": continuity, "level": continuity_level},
        "information_gain": {"score": information_gain, "level": information_gain_level},
        "analytical_validity": {"score": validity, "level": validity_level},
        "overall": {"score": overall, "level": overall_level},
        "branch_type": branch_type,
        "should_promote": should_promote,
        "expected_contribution": _expected_contribution(question_relevance, continuity, information_gain),
        "actual_contribution": _actual_contribution(should_promote, summary),
        "questions_answered": [item for item in [
            "Whether the task addresses the original question",
            "Whether the task follows the current hypothesis",
            "Whether the required variables are available",
        ] if item],
        "questions_remaining": [item for item in [
            "Whether this task closes the remaining uncertainty",
            "Whether alternative explanations still need testing",
        ] if item],
        "reason": (
            f"Question relevance is {relevance_level}, continuity is {continuity_level}, "
            f"information gain is {information_gain_level}, and analytical validity is {validity_level}."
        ),
        "promoted_understanding": _normalize_text(summary.get("current_understanding") or summary.get("narrative") or ""),
    }


def build_traceability_record(integrity: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "original_question": integrity.get("original_question"),
        "current_hypothesis": integrity.get("current_hypothesis"),
        "purpose_of_task": integrity.get("task_request"),
        "expected_contribution": integrity.get("expected_contribution"),
        "actual_contribution": integrity.get("actual_contribution"),
        "question_relevance": integrity.get("question_relevance"),
        "questions_answered": list(integrity.get("questions_answered") or []),
        "questions_remaining": list(integrity.get("questions_remaining") or []),
        "branch_type": integrity.get("branch_type"),
        "overall_integrity": integrity.get("overall"),
        "reason": integrity.get("reason"),
    }
