from __future__ import annotations

from typing import Any, Dict, List

import re


def _extract_constraint(question: str, pattern: str) -> float | None:
    match = re.search(pattern, question, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def parse_constraints(question: str) -> Dict[str, Any]:
    return {
        "budget_cap": _extract_constraint(question, r"budget(?: cap)?\s*(?:of|under|below|<=)?\s*([\d\.]+)"),
        "max_price_increase_pct": _extract_constraint(question, r"max price increase\s*%?\s*(?:of|under|below|<=)?\s*([\d\.]+)"),
        "risk_tolerance": "low" if "low risk" in question.lower() else "high" if "high risk" in question.lower() else "medium",
        "inventory_limit": _extract_constraint(question, r"inventory(?: limit)?\s*(?:of|under|below|<=)?\s*([\d\.]+)"),
        "resource_capacity": _extract_constraint(question, r"capacity\s*(?:of|under|below|<=)?\s*([\d\.]+)"),
    }


def optimize_actions(
    predictive_result: Dict[str, Any],
    scenario_summary: List[Dict[str, Any]],
    question: str,
) -> Dict[str, Any]:
    constraints = parse_constraints(question)
    confidence = ((predictive_result.get("confidence") or {}).get("score")) or 40
    confidence_label = ((predictive_result.get("confidence") or {}).get("label")) or "low"
    truthfulness_flags = predictive_result.get("truthfulness_flags", []) or []
    driver_diagnostics = ((predictive_result.get("validation_summary") or {}).get("driver_diagnostics")) or {}
    dominance = float(driver_diagnostics.get("top_driver_share", 0.0) or 0.0)

    ranked_actions: List[Dict[str, Any]] = []
    for scenario in scenario_summary:
        midpoint = float(scenario.get("estimated_effect") or 0.0)
        effect_range = scenario.get("estimated_range", {}) or {}
        risk_level = scenario.get("risk_level", "medium")
        feasibility = scenario.get("feasibility_level", "medium")
        risk_penalty = {"low": 0.9, "medium": 0.75, "high": 0.55}.get(risk_level, 0.7)
        feasibility_bonus = {"low": 0.8, "medium": 1.0, "high": 1.1}.get(feasibility, 1.0)
        confidence_multiplier = max(float(confidence) / 100.0, 0.25)
        score = midpoint * risk_penalty * feasibility_bonus * confidence_multiplier
        explanation = []
        safety_grade = scenario.get("safety_grade", "guarded")
        if constraints.get("risk_tolerance") == "low" and risk_level == "high":
            score *= 0.7
            explanation.append("Penalized because it exceeds the requested risk tolerance.")
        if constraints.get("max_price_increase_pct") is not None and "price increase" in str(scenario.get("action", "")).lower():
            if 3.0 > float(constraints["max_price_increase_pct"]):
                score *= 0.5
                explanation.append("Penalized because the suggested price move exceeds the allowed maximum.")
        if truthfulness_flags:
            score *= 0.85
            explanation.append("Confidence-adjusted downward because predictive truthfulness flags remain active.")
        if dominance >= 0.6 and scenario.get("driver_family") == "pricing":
            score *= 0.82
            explanation.append("Downgraded because the recommendation depends heavily on one dominant feature without direct elasticity evidence.")
            safety_grade = "guarded"

        ranked_actions.append(
            {
                "scenario": scenario.get("scenario"),
                "recommended_action": scenario.get("action"),
                "objective": "maximize",
                "score": round(score, 4),
                "expected_impact_range": effect_range,
                "tradeoffs": {
                    "risk_level": risk_level,
                    "feasibility_level": feasibility,
                },
                "constraint_explanations": explanation,
                "affected_segments": scenario.get("affected_segments", []),
                "monitoring_kpis": scenario.get("monitoring_kpis", []),
                "downside_risks": scenario.get("downside_risks", []),
                "failure_conditions": scenario.get("failure_conditions", []),
                "reliability": confidence_label,
                "intervention_confidence": scenario.get("intervention_confidence", "low"),
                "safety_grade": safety_grade,
            }
        )

    ranked_actions.sort(key=lambda item: item["score"], reverse=True)
    return {
        "constraints": constraints,
        "ranked_actions": ranked_actions,
        "best_action": ranked_actions[0] if ranked_actions else None,
    }
