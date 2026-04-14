from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv

from state.state import AnalystState
from utils.openai_runtime import get_openai_client

load_dotenv()


def _story_signature(story: Dict[str, Any]) -> str:
    columns = story.get("columns") or []
    if story.get("column"):
        columns = [story["column"], *columns]
    if story.get("group_column"):
        columns = [*columns, story["group_column"]]
    columns = [str(col) for col in columns if col]
    return f"{story.get('type', 'story')}|{'|'.join(columns)}"


def _build_dataset_context(state: AnalystState) -> str:
    dataset_profile = state.get("dataset_profile", {}) or {}
    numeric_columns = dataset_profile.get("numeric_columns", []) or []
    categorical_columns = dataset_profile.get("categorical_columns", []) or []
    selected_columns = state.get("selected_columns", []) or []

    context_lines = [
        "Dataset context:",
        f"- Selected analysis columns: {selected_columns if selected_columns else 'not specified'}",
        f"- Numeric columns available: {numeric_columns if numeric_columns else 'none'}",
        f"- Categorical columns available: {categorical_columns if categorical_columns else 'none'}",
        "- Infer the business context only from the question and available columns.",
        "- Do not assume a specific industry, product, or marketplace unless it is directly supported by the inputs.",
    ]
    return "\n".join(context_lines)


def _fallback_detail(story: Dict[str, Any]) -> Dict[str, Any]:
    story_type = story.get("type")
    insight = story.get("insight", "Pattern detected in the data.")
    explanation = insight
    implication = "This result is directionally useful, but should be considered alongside business context."
    action = "Review this pattern with the business team and confirm whether it matches operational expectations."
    headline = insight

    if story_type == "correlation":
        columns = story.get("columns", ["metric_1", "metric_2"])
        direction = story.get("direction", "directional")
        strength = story.get("strength", "moderate")
        value = story.get("value")
        explanation = (
            f"In simple terms, when {columns[0]} changes, {columns[1]} tends to move in a "
            f"{direction} way. The relationship looks {strength}"
            f"{'' if value is None else f' with a correlation around {round(float(value), 3)}'}."
        )
        implication = (
            f"This suggests {columns[0]} and {columns[1]} are connected strongly enough to matter in analysis, "
            "though correlation alone does not prove causation."
        )
        action = f"Use {columns[0]} and {columns[1]} together when discussing drivers of performance."

    elif story_type in {"group_difference", "grouped_numeric"}:
        column = story.get("column", "metric")
        group_column = story.get("group_column", "group")
        top_group = story.get("top_group", "one group")
        bottom_group = story.get("bottom_group", "another group")
        explanation = (
            f"The average {column} is not evenly distributed across {group_column}. "
            f"{top_group} is leading while {bottom_group} is trailing."
        )
        implication = (
            f"This means the category split in {group_column} is useful for explaining differences in {column}."
        )
        action = f"Compare business practices or conditions behind {top_group} versus {bottom_group}."

    elif story_type == "categorical_relationship":
        columns = story.get("columns", ["category_a", "category_b"])
        p_value = story.get("p_value")
        explanation = (
            f"The distribution of {columns[0]} changes in a meaningful way across {columns[1]}. "
            f"{'' if p_value is None else f'The p-value of {round(float(p_value), 4)} suggests this is unlikely to be random.'}"
        )
        implication = "These categories should not be treated as independent in downstream decision-making."
        action = f"Segment reporting by both {columns[0]} and {columns[1]} instead of looking at either in isolation."

    elif story_type == "category_frequency":
        column = story.get("column", "category")
        category = story.get("category", "a category")
        share = story.get("share")
        explanation = (
            f"{category} appears most often in {column}"
            f"{'' if share is None else f', representing about {round(float(share), 2)}% of the records'}."
        )
        implication = f"This dominant category will heavily shape overall patterns seen in {column}."
        action = f"Make sure business conclusions for {column} are not being overly driven by {category}."

    elif story_type == "rare_categories":
        column = story.get("column", "category")
        categories = story.get("categories", [])
        explanation = f"{column} includes uncommon categories such as {', '.join(categories[:3])}."
        implication = "Rare categories may be important edge cases, but they can also make summary statistics unstable."
        action = f"Decide whether the rare values in {column} should be grouped, monitored separately, or investigated."

    elif story_type == "outliers":
        column = story.get("column", "metric")
        count = story.get("count", 0)
        explanation = f"There are {count} unusual values in {column}, meaning a small set of records sits far from the typical range."
        implication = f"These records can distort averages, correlations, and business summaries for {column}."
        action = f"Inspect the outliers in {column} before using it for high-stakes decisions."

    return {
        "related_story_signature": _story_signature(story),
        "headline": headline,
        "plain_english": explanation,
        "business_implication": implication,
        "recommended_action": action,
        "limitations": "This finding should be interpreted alongside other relevant variables and broader business context.",
    }


def _format_details(details: List[Dict[str, Any]]) -> str:
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


def _parse_llm_output(raw_text: str, stories: List[Dict[str, Any]]) -> tuple[str, List[str], List[Dict[str, Any]]]:
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

    details = [_fallback_detail(story) for story in stories]
    return insights_text, questions, details


def llm_insight_synthesizer_node(state: AnalystState) -> AnalystState:
    """
    Synthesizes top stories into plain-language business insights.
    Uses the LLM when available and enabled, otherwise falls back to a
    deterministic but still detailed explanation layer.
    """
    evidence = state.setdefault("analysis_evidence", {})
    top_stories = evidence.get("top_stories", [])
    business_question = state.get("business_question", "")
    dataset_profile = state.get("dataset_profile", {})
    dataset_context = _build_dataset_context(state)
    llm_disabled = state.get("disable_llm_reasoning") or not state.get("enable_llm_reasoning", True)

    if not top_stories:
        evidence["llm_insights"] = "No patterns detected to generate insights."
        evidence["llm_insight_details"] = []
        evidence["clarification_questions"] = []
        evidence["llm_synthesis_status"] = "no_stories"
        state["llm_insights"] = evidence["llm_insights"]
        state["clarification_questions"] = evidence["clarification_questions"]
        return state

    if llm_disabled:
        details = [_fallback_detail(story) for story in top_stories]
        evidence["llm_insight_details"] = details
        evidence["llm_insights"] = _format_details(details)
        evidence["clarification_questions"] = []
        evidence["llm_synthesis_status"] = "disabled"
        state["llm_insights"] = evidence["llm_insights"]
        state["clarification_questions"] = evidence["clarification_questions"]
        print("\n=== LLM SYNTHESIZED INSIGHTS ===")
        print(evidence["llm_insights"])
        return state

    prompt = f"""
You are a senior business analyst advising a decision-maker.

Your job is to convert statistical findings into business-level insight that helps someone act.

{dataset_context}

You MUST follow these rules:
- Use only the evidence provided below.
- Do not invent numbers, causes, segments, business models, or industry context.
- Translate findings into real-world business meaning.
- Explain what is happening operationally or commercially, not just statistically.
- State why the pattern matters for revenue, pricing, operations, strategy, risk, customer behavior, or resource allocation, whichever is actually supported.
- Give specific actions the decision-maker can take.
- Include limitations, especially when the evidence is weak, moderate, incomplete, or could be influenced by other variables.

STRICTLY DO NOT:
- Repeat or lightly paraphrase the raw statistical output.
- Say generic phrases like "X has a relationship with Y".
- Give vague recommendations such as "consider", "explore", or "use this in decision making".

Your tone must be:
- Direct
- Practical
- Specific
- Business-focused
- Written like an expert advisor speaking to a stakeholder

Business Question:
{business_question}

Dataset Profile:
{dataset_profile}

Top Stories:
{top_stories}

Return exactly in this format:

INSIGHT:
- Clear real-world explanation of the pattern

BUSINESS IMPLICATION:
- Why it matters commercially or operationally

RECOMMENDATIONS:
- Specific actions to take

LIMITATIONS:
- What to be careful about

CLARIFICATION QUESTIONS:
- bullet points only if needed, otherwise write "- None"
"""

    client = get_openai_client()
    if client is None:
        details = [_fallback_detail(story) for story in top_stories]
        insights_text = _format_details(details)
        questions = []
        evidence["llm_synthesis_status"] = "fallback_used: OPENAI_API_KEY not set"
        print("\n[INFO] LLM insight synthesis unavailable - using fallback: OPENAI_API_KEY not set")
    else:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            llm_result = response.choices[0].message.content or ""
            insights_text, questions, details = _parse_llm_output(llm_result, top_stories)
        except Exception as exc:
            details = [_fallback_detail(story) for story in top_stories]
            insights_text = _format_details(details)
            questions = []
            evidence["llm_synthesis_status"] = f"fallback_used: {exc}"
            print(f"\n[INFO] LLM insight synthesis unavailable - using fallback: {exc}")
        else:
            evidence["llm_synthesis_status"] = "live_llm"

    evidence["llm_insight_details"] = details
    evidence["llm_insights"] = insights_text
    evidence["clarification_questions"] = questions
    state["llm_insights"] = insights_text
    state["clarification_questions"] = questions

    print("\n=== LLM SYNTHESIZED INSIGHTS ===")
    print(insights_text)
    if questions:
        print("\nCLARIFICATION QUESTIONS:")
        for q in questions:
            print(f"- {q}")

    return state
