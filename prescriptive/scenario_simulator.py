from __future__ import annotations

from typing import Any, Dict, List


def _scenario_scale(predictive_result: Dict[str, Any]) -> float:
    metrics = ((predictive_result.get("metrics") or {}).get("values")) or {}
    confidence = ((predictive_result.get("confidence") or {}).get("score")) or 40
    base = float(metrics.get("mae") or metrics.get("rmse") or metrics.get("f1") or 0.1)
    return max(base, 0.1) * max(float(confidence) / 100.0, 0.25)


def _driver_action(feature: str, objective: str) -> str:
    feature_lower = str(feature).lower()
    if any(token in feature_lower for token in ["price", "discount"]):
        return "test a 3% price increase"
    if any(token in feature_lower for token in ["spend", "budget", "marketing", "ads", "ad_"]):
        return "shift 12% spend from the lowest-response segment to the strongest-response segment"
    if any(token in feature_lower for token in ["stock", "inventory", "qty", "quantity", "demand"]):
        return "increase stock by 8% for the strongest-demand segment"
    if any(token in feature_lower for token in ["churn", "risk", "retention"]):
        return "prioritize the top 15% highest-risk, highest-value records for intervention"
    return "run a controlled optimization on the highest-impact segment"


def simulate_scenarios(predictive_result: Dict[str, Any], objective: str) -> List[Dict[str, Any]]:
    top_drivers = predictive_result.get("top_drivers", []) or []
    if not top_drivers:
        return []

    scale = _scenario_scale(predictive_result)
    primary_driver = str(top_drivers[0].get("feature", "top_driver"))
    primary_importance = float(top_drivers[0].get("importance") or 0.1)
    action = _driver_action(primary_driver, objective)

    paths = [
        ("best_growth_option", 1.25, "high", "medium"),
        ("safest_option", 0.65, "low", "high"),
        ("balanced_option", 0.95, "medium", "medium"),
        ("high_risk_high_reward_option", 1.6, "high", "low"),
    ]

    scenarios: List[Dict[str, Any]] = []
    for name, multiplier, risk, feasibility in paths:
        midpoint = round(scale * max(primary_importance, 0.05) * multiplier, 4)
        scenarios.append(
            {
                "scenario": name,
                "driver": primary_driver,
                "action": action,
                "expected_direction": "increase" if objective in {"maximize", "grow", "improve"} else "decrease",
                "estimated_effect": midpoint,
                "estimated_range": {
                    "low": round(midpoint * 0.7, 4),
                    "high": round(midpoint * 1.3, 4),
                },
                "risk_level": risk,
                "feasibility_level": feasibility,
                "affected_segments": [primary_driver],
            }
        )
    return scenarios
