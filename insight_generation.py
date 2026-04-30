from __future__ import annotations

from typing import Any, Dict, List, Tuple

from llm.output_filter import filter_llm_output
from llm.guarded_reasoning import validate_explanation_text
from state.state import AnalystState
from utils.openai_runtime import get_openai_client


def _story_signature(story: Dict[str, Any]) -> str:
    columns = story.get("columns") or []
    if story.get("column"):
        columns = [story["column"], *columns]
    if story.get("group_column"):
        columns = [*columns, story["group_column"]]
    columns = [str(col) for col in columns if col]
    return f"{story.get('type', 'story')}|{'|'.join(columns)}"


def fallback_detail(story: Dict[str, Any]) -> Dict[str, Any]:
    story_type = story.get("type")
    insight = story.get("insight", "Pattern detected in the data.")
    explanation = insight
    implication = "This result is directionally useful, but should be considered alongside business context."
    action = "Review this pattern with the business team and confirm whether it matches operational expectations."
    relationship_type = story.get("relationship_type", "unknown")
    validity = story.get("insight_validity") or {}
    guarded_recommendation = story.get("guarded_recommendation")

    if relationship_type == "unit_conversion":
        explanation = "This pattern is expected because the two fields are mathematically linked by unit conversion."
        implication = "It is useful for data consistency validation, not for business decision-making."
        action = "No action required."
    elif validity and not validity.get("valid", True):
        explanation = "Detected relationship is not meaningful for decision-making."
        implication = validity.get("reason", "The current evidence is not reliable enough for action.")
        action = guarded_recommendation or "No actionable recommendation due to insufficient or non-meaningful relationship."

    if story_type == "summary_numeric":
        column = story.get("column", "metric")
        mean = story.get("mean")
        min_value = story.get("min")
        max_value = story.get("max")
        explanation = (
            f"{column} is centered around {round(float(mean), 2) if mean is not None else 'its typical level'}, "
            f"with values ranging from {round(float(min_value), 2) if min_value is not None else 'the low end'} "
            f"to {round(float(max_value), 2) if max_value is not None else 'the high end'}."
        )
        implication = f"This gives a baseline operating range for {column}."
        action = f"Use the observed range of {column} as a baseline for target-setting and anomaly monitoring."
    elif story_type == "inferential_relationship":
        columns = story.get("columns", ["metric_1", "metric_2"])
        p_value = story.get("p_value")
        effect = (story.get("effect_size") or {}).get("interpretation", "unknown")
        if relationship_type not in {"unit_conversion", "duplicate_feature"} and validity.get("valid", True):
            explanation = (
                f"The relationship between {columns[0]} and {columns[1]} holds up under formal testing"
                f"{'' if p_value is None else f' with p={round(float(p_value), 4)}'}."
            )
            implication = f"This makes the linkage between {columns[0]} and {columns[1]} credible enough to inform decision-making, although the practical impact is {effect}."
            action = guarded_recommendation or f"Treat {columns[0]} as a meaningful signal when monitoring or forecasting {columns[1]}."
    elif story_type == "inferential_group_difference":
        column = story.get("column", "metric")
        group_column = story.get("group_column", "group")
        top_group = story.get("top_group", "one group")
        bottom_group = story.get("bottom_group", "another group")
        p_value = story.get("p_value")
        effect = (story.get("effect_size") or {}).get("interpretation", "unknown")
        explanation = (
            f"{column} is not behaving the same way across {group_column}; {top_group} is meaningfully separated from {bottom_group}"
            f"{'' if p_value is None else f' with p={round(float(p_value), 4)}'}."
        )
        implication = f"A single shared policy for all {group_column} segments may hide real operational differences, and the practical impact appears {effect}."
        action = f"Set separate targets, pricing, or monitoring rules for the main {group_column} groups."
    elif story_type == "inferential_categorical_association":
        columns = story.get("columns", ["category_a", "category_b"])
        p_value = story.get("p_value")
        effect = (story.get("effect_size") or {}).get("interpretation", "unknown")
        explanation = (
            f"{columns[0]} and {columns[1]} are associated strongly enough to survive formal testing"
            f"{'' if p_value is None else f' with p={round(float(p_value), 4)}'}."
        )
        implication = f"These categories should be analyzed jointly because treating them as independent could distort conclusions, and the practical association looks {effect}."
        action = f"Report and monitor {columns[0]} together with {columns[1]} instead of summarizing them separately."

    return {
        "related_story_signature": _story_signature(story),
        "headline": insight,
        "plain_english": explanation,
        "business_implication": implication,
        "recommended_action": action,
        "limitations": "Interpret this finding alongside other relevant variables and process context.",
    }


def format_details(details: List[Dict[str, Any]]) -> str:
    sections = []
    for detail in details:
        sections.append(
            "\n".join(
                [
                    "INSIGHT:",
                    f"- {detail['plain_english']}",
                    "",
                    "BUSINESS IMPLICATION:",
                    f"- {detail['business_implication']}",
                    "",
                    "RECOMMENDATIONS:",
                    f"- {detail['recommended_action']}",
                    "",
                    "LIMITATIONS:",
                    f"- {detail['limitations']}",
                ]
            )
        )
    return "\n\n".join(sections) if sections else "No patterns detected to generate insights."


def parse_llm_output(raw_text: str, stories: List[Dict[str, Any]]) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    insights_text = raw_text
    questions: List[str] = []

    if "CLARIFICATION QUESTIONS:" in raw_text:
        parts = raw_text.split("CLARIFICATION QUESTIONS:")
        insights_text = parts[0].replace("INSIGHTS:", "").strip()
        questions = [
            q.replace("-", "").strip()
            for q in parts[1].strip().splitlines()
            if q.strip()
        ]

    questions = [
        q for q in questions
        if q and q.strip().lower() not in {"none", "n/a", "no", "no questions"}
    ]

    details = [fallback_detail(story) for story in stories]
    return insights_text, questions, details


def generate_insights(
    state: AnalystState,
    top_stories: List[Dict[str, Any]],
    prompt: str,
) -> Tuple[str, List[str], List[Dict[str, Any]], str]:
    if not top_stories:
        return "No patterns detected to generate insights.", [], [], "no_stories"

    if state.get("disable_llm_reasoning") or not state.get("enable_llm_reasoning", True):
        details = [fallback_detail(story) for story in top_stories]
        return format_details(details), [], details, "disabled"

    client = get_openai_client()
    if client is None:
        details = [fallback_detail(story) for story in top_stories]
        return format_details(details), [], details, "fallback_used: OPENAI_API_KEY not set"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = response.choices[0].message.content or ""
        primary_story = top_stories[0] if top_stories else {}
        semantic = primary_story.get("semantic_reasoning") or {}
        validity_payload = primary_story.get("insight_validity") or {"valid": True, "severity": "low"}
        recommendation_payload = {
            "recommendation_restrictions": primary_story.get("recommendation_restrictions", []),
            "allowed_actions": primary_story.get("guardrail_allowed_actions", []),
            "blocked_actions": primary_story.get("guardrail_blocked_actions", []),
            "final_recommendation": primary_story.get("guarded_recommendation"),
            "guardrail_triggered": primary_story.get("guardrail_triggered", False),
        }
        filtered = filter_llm_output(
            raw_text=raw,
            payload={"top_stories": top_stories, "causal_evidence": primary_story.get("causal_evidence", {})},
            semantic_output=semantic,
            validation_output=validity_payload,
            recommendation_output=recommendation_payload,
        )
        cleaned_raw = filtered["text"]
        causal_grade = ((primary_story.get("causal_evidence") or {}).get("grade")) or "LOW"
        numeric_validation = validate_explanation_text(cleaned_raw, {"top_stories": top_stories}, causal_grade)
        cleaned_raw = numeric_validation["text"]
        insights_text, questions, details = parse_llm_output(cleaned_raw, top_stories)
        combined_issues = filtered["issues"] + numeric_validation["issues"]
        status = "live_llm" if not combined_issues else f"live_llm_with_guardrail_warnings:{'; '.join(combined_issues)}"
        return insights_text, questions, details, status
    except Exception as exc:
        details = [fallback_detail(story) for story in top_stories]
        return format_details(details), [], details, f"fallback_used: {exc}"
