from state.state import AnalystState
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def llm_insight_synthesizer_node(state: AnalystState) -> AnalystState:
    """
    Synthesizes story candidates into business insights using an LLM.
    Consumes `story_candidates` and outputs:
    - llm_insights: structured business insights
    - clarification_questions: follow-up questions
    """
    evidence = state.setdefault("analysis_evidence", {})
    story_candidates = evidence.get("top_stories", [])
    business_question = state.get("business_question", "")
    dataset_profile = state.get("dataset_profile", {})

    if not story_candidates:
        evidence["llm_insights"] = "No patterns detected to generate insights."
        evidence["clarification_questions"] = []
        return state

    prompt = f"""
You are a senior data analyst.

Your task is to convert statistical patterns into clear, actionable business insights.

You MUST follow these rules:
- Only interpret the patterns provided.
- Do NOT invent numbers or calculations.
- Expand on the patterns in a business context.
- If the dataset is small or inconclusive, mention it.

Business Question:
{business_question}

Dataset Profile:
{dataset_profile}

Story Candidates:
{story_candidates}

Instructions:
1. Explain the patterns in business terms.
2. Highlight meaningful relationships.
3. Suggest possible business implications.
4. Ask clarification questions if more data is needed.

Return exactly in this format:

INSIGHTS:
- bullet points

CLARIFICATION QUESTIONS:
- bullet points
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    llm_result = response.choices[0].message.content

    # Parse LLM output
    insights_text = ""
    questions = []

    if "CLARIFICATION QUESTIONS:" in llm_result:
        parts = llm_result.split("CLARIFICATION QUESTIONS:")
        insights_text = parts[0].replace("INSIGHTS:", "").strip()
        questions_section = parts[1].strip().split("\n")
        questions = [q.replace("-", "").strip() for q in questions_section if q.strip()]
    else:
        insights_text = llm_result

    evidence["llm_insights"] = insights_text
    evidence["clarification_questions"] = questions

    print("\n=== LLM SYNTHESIZED INSIGHTS ===")
    print(insights_text)
    if questions:
        print("\nCLARIFICATION QUESTIONS:")
        for q in questions:
            print(f"- {q}")

    return state

