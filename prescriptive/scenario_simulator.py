from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _scenario_scale(predictive_result: Dict[str, Any]) -> float:
    metrics = ((predictive_result.get("metrics") or {}).get("values")) or {}
    confidence = ((predictive_result.get("confidence") or {}).get("score")) or 40
    base = float(metrics.get("mae") or metrics.get("rmse") or metrics.get("f1") or 0.1)
    return max(base, 0.1) * max(float(confidence) / 100.0, 0.25)


def _normalize_driver(driver: Dict[str, Any]) -> Tuple[str, float]:
    return str(driver.get("feature") or "top_driver"), float(driver.get("importance") or 0.0)


def _driver_family(feature: str) -> str:
    lower = feature.lower()
    if any(token in lower for token in ["price", "discount", "payment", "value"]):
        return "pricing"
    if any(token in lower for token in ["freight", "shipping", "delivery", "carrier", "logistics"]):
        return "logistics"
    if any(token in lower for token in ["stock", "inventory", "qty", "quantity", "demand"]):
        return "inventory"
    if any(token in lower for token in ["seller", "merchant", "vendor"]):
        return "seller"
    if any(token in lower for token in ["customer", "review", "retention", "risk", "churn"]):
        return "customer"
    return "operations"


def _affected_segments(feature: str, fallback_target: str) -> List[str]:
    family = _driver_family(feature)
    mapping = {
        "pricing": ["high-value SKUs", "premium segments"],
        "logistics": ["high-freight orders", "long-distance shipments"],
        "inventory": ["fast-moving SKUs", "demand-sensitive segments"],
        "seller": ["top sellers", "underperforming seller cohorts"],
        "customer": ["high-value customers", "high-friction customer cohorts"],
        "operations": [fallback_target],
    }
    return mapping.get(family, [fallback_target])


def _kpis_for_family(feature: str, target: str) -> List[str]:
    family = _driver_family(feature)
    mapping = {
        "pricing": [target, "conversion_rate", "review_score_avg", "repeat_purchase_rate"],
        "logistics": [target, "freight_value", "delivery_delay_rate", "cancel_rate"],
        "inventory": [target, "stockout_rate", "fill_rate", "order_volume"],
        "seller": [target, "seller_fill_rate", "late_shipment_rate", "seller_margin"],
        "customer": [target, "repeat_purchase_rate", "refund_rate", "review_score_avg"],
        "operations": [target, "cost_per_order", "exception_rate"],
    }
    return mapping.get(family, [target])


def _levers_from_top_drivers(top_drivers: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
    levers: List[Dict[str, Any]] = []
    seen_families: set[str] = set()
    for driver in top_drivers[:6]:
        feature, importance = _normalize_driver(driver)
        family = _driver_family(feature)
        if family in seen_families:
            continue
        seen_families.add(family)
        segments = _affected_segments(feature, target)
        kpis = _kpis_for_family(feature, target)
        if family == "pricing":
            levers.append(
                {
                    "family": family,
                    "feature": feature,
                    "importance": importance,
                    "action": "Run a controlled 2-3% price increase on higher-value SKUs with weaker elasticity risk.",
                    "evidence_summary": [
                        f"{feature} is a leading driver of {target}.",
                        "The model signal is concentrated around pricing behavior more than secondary operational variables.",
                    ],
                    "downside_risks": [
                        "Conversion could soften if customers are more price-sensitive than the model can observe.",
                        "Negative review or repeat-purchase movement could signal early elasticity failure.",
                    ],
                    "failure_conditions": [
                        "Conversion rate drops materially versus the pre-test baseline.",
                        "Review score or repeat purchase behavior deteriorates in the exposed group.",
                    ],
                    "monitoring_kpis": kpis,
                    "affected_segments": segments,
                    "risk_level": "medium",
                    "feasibility_level": "medium",
                }
            )
        elif family == "logistics":
            levers.append(
                {
                    "family": family,
                    "feature": feature,
                    "importance": importance,
                    "action": "Reduce freight-heavy order friction through carrier mix and packaging optimization before broad pricing changes.",
                    "evidence_summary": [
                        f"{feature} is materially influencing {target}.",
                        "Fulfillment cost signals remain meaningful behind the primary commercial drivers.",
                    ],
                    "downside_risks": [
                        "Carrier or packaging changes may raise operational complexity.",
                        "Savings may be offset if delivery reliability weakens.",
                    ],
                    "failure_conditions": [
                        "Delivery delays rise after the logistics change.",
                        "Freight cost savings do not translate into better realized order value.",
                    ],
                    "monitoring_kpis": kpis,
                    "affected_segments": segments,
                    "risk_level": "low",
                    "feasibility_level": "high",
                }
            )
        elif family == "inventory":
            levers.append(
                {
                    "family": family,
                    "feature": feature,
                    "importance": importance,
                    "action": "Prioritize inventory and availability for the strongest-demand SKU groups rather than spreading stock evenly.",
                    "evidence_summary": [
                        f"{feature} points to demand or availability sensitivity for {target}.",
                        "The signal suggests concentration in a subset of high-response products.",
                    ],
                    "downside_risks": [
                        "Over-allocation can increase carrying cost if demand softens.",
                        "Lower-priority segments may experience avoidable stockouts.",
                    ],
                    "failure_conditions": [
                        "Stockout rate does not improve in the prioritized groups.",
                        "Inventory turns weaken after the reallocation.",
                    ],
                    "monitoring_kpis": kpis,
                    "affected_segments": segments,
                    "risk_level": "medium",
                    "feasibility_level": "high",
                }
            )
        elif family == "seller":
            levers.append(
                {
                    "family": family,
                    "feature": feature,
                    "importance": importance,
                    "action": "Segment sellers by operational quality and focus commercial support on the highest-value cohorts with weaker fulfillment consistency.",
                    "evidence_summary": [
                        f"{feature} meaningfully contributes to {target}.",
                        "Seller-level variation appears operationally important, not just random noise.",
                    ],
                    "downside_risks": [
                        "Support resources can be wasted on low-response seller cohorts.",
                        "Operational attention may be diluted if segments are too broad.",
                    ],
                    "failure_conditions": [
                        "Seller service metrics do not improve after intervention.",
                        "Target KPI remains flat in the supported seller segment.",
                    ],
                    "monitoring_kpis": kpis,
                    "affected_segments": segments,
                    "risk_level": "medium",
                    "feasibility_level": "medium",
                }
            )
        elif family == "customer":
            levers.append(
                {
                    "family": family,
                    "feature": feature,
                    "importance": importance,
                    "action": "Protect high-value customer cohorts through service recovery or retention nudges before broad commercial changes.",
                    "evidence_summary": [
                        f"{feature} indicates customer behavior has material predictive influence on {target}.",
                        "Customer experience signals should be protected before scaling commercial interventions.",
                    ],
                    "downside_risks": [
                        "Broad retention spend can erode margin if targeted too loosely.",
                        "Short-term lifts may fade if customer friction is structural.",
                    ],
                    "failure_conditions": [
                        "Repeat purchase behavior does not improve in the targeted cohort.",
                        "Refund or complaint rates rise after the intervention.",
                    ],
                    "monitoring_kpis": kpis,
                    "affected_segments": segments,
                    "risk_level": "low",
                    "feasibility_level": "medium",
                }
            )
        else:
            levers.append(
                {
                    "family": family,
                    "feature": feature,
                    "importance": importance,
                    "action": "Run a controlled operational improvement on the highest-variance segment before broader deployment.",
                    "evidence_summary": [
                        f"{feature} contributes meaningfully to {target}.",
                        "The model signal suggests a concentrated operational lever, but the mechanism is not yet causal.",
                    ],
                    "downside_risks": ["The operational intervention may not generalize beyond the tested segment."],
                    "failure_conditions": ["The tested segment does not outperform the holdout after the change."],
                    "monitoring_kpis": kpis,
                    "affected_segments": segments,
                    "risk_level": "medium",
                    "feasibility_level": "medium",
                }
            )
    return levers


def _deduplicate_levers(levers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen_actions: set[str] = set()
    for lever in levers:
        action_key = lever["action"].split(".")[0].strip().lower()
        if action_key in seen_actions:
            continue
        seen_actions.add(action_key)
        deduped.append(lever)
    return deduped


def _scenario_variant_action(lever: Dict[str, Any], scenario_name: str) -> str:
    family = lever["family"]
    if family == "pricing":
        mapping = {
            "growth_option": "Run a controlled 2-3% price increase on higher-value SKUs with weaker elasticity risk.",
            "safest_option": "Tighten discount guardrails on premium SKUs before changing list prices more broadly.",
            "balanced_option": "Bundle moderate pricing adjustments with freight or service-value improvements on strong categories.",
            "aggressive_option": "Use segmented pricing tests on top-performing SKUs only after a control group is in place.",
        }
        return mapping.get(scenario_name, lever["action"])
    if family == "seller":
        mapping = {
            "growth_option": "Concentrate seller support on high-value cohorts with the strongest commercial upside.",
            "safest_option": "Raise fulfillment consistency thresholds for underperforming seller cohorts before scaling incentives.",
            "balanced_option": "Split seller support between top-volume and high-variance cohorts to improve reliability and value.",
            "aggressive_option": "Reallocate commercial exposure away from persistently weak seller cohorts toward higher-performing ones.",
        }
        return mapping.get(scenario_name, lever["action"])
    if family == "customer":
        mapping = {
            "growth_option": "Protect high-value customer cohorts through service recovery or retention nudges before broad commercial changes.",
            "safest_option": "Address the highest-friction customer journeys first before expanding commercial offers.",
            "balanced_option": "Combine retention outreach with experience fixes on the most valuable at-risk cohorts.",
            "aggressive_option": "Target high-value but unstable customer cohorts with a tightly monitored intervention pilot.",
        }
        return mapping.get(scenario_name, lever["action"])
    return lever["action"]


def _scenario_ordered_levers(levers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not levers:
        return []

    def pick(candidates: List[Dict[str, Any]], chosen: List[Dict[str, Any]]) -> Dict[str, Any]:
        chosen_actions = {item["action"] for item in chosen}
        for candidate in candidates:
            if candidate["action"] not in chosen_actions:
                return candidate
        return candidates[0]

    growth_candidates = sorted(levers, key=lambda item: (-item["importance"], item["risk_level"] == "high"))
    growth = pick(growth_candidates, [])

    safest_candidates = sorted(levers, key=lambda item: (item["risk_level"] != "low", -item["importance"]))
    safest = pick(safest_candidates, [growth])

    balanced_candidates = sorted(levers, key=lambda item: (item["risk_level"] != "medium", -item["importance"]))
    balanced = pick(balanced_candidates, [growth, safest])

    aggressive_candidates = sorted(
        levers,
        key=lambda item: (
            item["risk_level"] != "medium",
            item["feasibility_level"] == "low",
            -item["importance"],
        ),
    )
    aggressive = pick(aggressive_candidates, [growth, safest, balanced])
    return [growth, safest, balanced, aggressive]


def simulate_scenarios(predictive_result: Dict[str, Any], objective: str) -> List[Dict[str, Any]]:
    top_drivers = predictive_result.get("top_drivers", []) or []
    if not top_drivers:
        return []

    target = predictive_result.get("target_column", "target")
    scale = _scenario_scale(predictive_result)
    levers = _deduplicate_levers(_levers_from_top_drivers(top_drivers, target))
    if not levers:
        return []

    scenario_names = [
        ("growth_option", 1.15),
        ("safest_option", 0.65),
        ("balanced_option", 0.9),
        ("aggressive_option", 1.35),
    ]
    selected_levers = _scenario_ordered_levers(levers)

    scenarios: List[Dict[str, Any]] = []
    for (scenario_name, multiplier), lever in zip(scenario_names, selected_levers):
        midpoint = round(scale * max(float(lever["importance"]), 0.05) * multiplier, 4)
        scenarios.append(
            {
                "scenario": scenario_name,
                "driver": lever["feature"],
                "driver_family": lever["family"],
                "action": _scenario_variant_action(lever, scenario_name),
                "expected_direction": "increase" if objective in {"maximize", "grow", "improve"} else "decrease",
                "estimated_effect": midpoint,
                "estimated_range": {
                    "low": round(midpoint * 0.72, 4),
                    "high": round(midpoint * 1.28, 4),
                },
                "risk_level": lever["risk_level"],
                "feasibility_level": lever["feasibility_level"],
                "affected_segments": lever["affected_segments"],
                "evidence_summary": lever["evidence_summary"],
                "downside_risks": lever["downside_risks"],
                "failure_conditions": lever["failure_conditions"],
                "monitoring_kpis": lever["monitoring_kpis"],
                "intervention_confidence": "moderate" if scenario_name in {"safest_option", "balanced_option"} else "low",
                "safety_grade": "controlled" if scenario_name in {"safest_option", "balanced_option"} else "guarded",
            }
        )

    return scenarios
