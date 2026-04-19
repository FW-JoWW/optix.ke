from __future__ import annotations

from typing import Any, Dict, List, Tuple

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
        insights_text, questions, details = parse_llm_output(raw, top_stories)
        return insights_text, questions, details, "live_llm"
    except Exception as exc:
        details = [fallback_detail(story) for story in top_stories]
        return format_details(details), [], details, f"fallback_used: {exc}"
