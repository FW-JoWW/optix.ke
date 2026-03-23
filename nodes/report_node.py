from state.state import AnalystState
from typing import List

def report_node(state: AnalystState) -> AnalystState:
    """
    Generates a professional report combining:
    - business question
    - rule-based insights
    - LLM insights
    - clarification questions
    - analysis plan
    """

    business_question = state.get("business_question", "N/A")
    insights: List[str] = state.get("insights") or []
    llm_insights: str = state.get("llm_insights") or "None"
    clarification_questions: List[str] = state.get("clarification_questions") or []
    analysis_plan = state.get("analysis_plan") or []

    # Format insights as bullet points
    def format_list(items: List[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "None"

    # Format analysis plan
    formatted_plan = "\n".join(
        f"- {p}" if isinstance(p, str) else f"- {p.get('tool')} on columns: {', '.join(p.get('columns', []))}" 
        for p in analysis_plan
        #f"- {p['tool']} on columns: {', '.join(p['columns'])}" for p in analysis_plan
    ) or "None"

    report = f"""
================ EXECUTIVE REPORT ================

BUSINESS QUESTION:
{business_question}

ANALYSIS PLAN:
{formatted_plan}

RULE-BASED INSIGHTS:
{format_list(insights)}

LLM INSIGHTS:
{llm_insights or 'None'}

CLARIFICATION QUESTIONS:
{format_list(clarification_questions)}

=================================================
"""

    state["final_report"] = report

    print("\n===== FINAL REPORT =====")
    print(report)

    return state

