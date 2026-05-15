from __future__ import annotations

from typing import Any, Dict, List

from core.predictive_contracts import ConfidenceAssessment, PrescriptiveAction, PrescriptiveResult
from predictive.recommendation_memory import calibrate_from_memory, store_recommendation_snapshot
from prescriptive.optimization_engine import optimize_actions
from prescriptive.scenario_simulator import simulate_scenarios


def _objective_from_question(question: str) -> str:
    query = (question or "").lower()
    if any(word in query for word in ["reduce", "decrease", "lower", "minimize"]):
        return "minimize"
    return "maximize"


def _action_templates(problem_type: str, target: str, confidence: str) -> List[PrescriptiveAction]:
    causal_safety_note = "Directional signal only. Validate through an experiment or controlled rollout before broad action."
    if problem_type == "classification":
        return [
            PrescriptiveAction(
                action=f"Prioritize the top 15% highest-risk records for intervention related to {target}.",
                rationale="Classification output is strongest when used for focused triage rather than blanket rollout.",
                feasibility="high",
                risk_level="medium",
                affected_segments=[target],
                requires_experiment=True,
                causal_safety_note=causal_safety_note,
                reliability="moderate" if confidence == "high" else "low",
                safety_grade="controlled",
            ),
            PrescriptiveAction(
                action="Run a controlled treatment on the top-risk segment before scaling.",
                rationale="A holdout or A/B design is needed before treating model lift as operational fact.",
                feasibility="medium",
                risk_level="low",
                affected_segments=[target],
                requires_experiment=True,
                causal_safety_note=causal_safety_note,
                reliability="moderate" if confidence == "high" else "low",
                safety_grade="controlled",
            ),
        ]
    if problem_type == "forecasting":
        return [
            PrescriptiveAction(
                action=f"Adjust inventory or capacity plans using the forecasted {target} baseline.",
                rationale="Forecasting is most actionable through supply and resource planning with explicit uncertainty bounds.",
                feasibility="high",
                risk_level="medium",
                affected_segments=[target],
                requires_experiment=False,
                causal_safety_note="Forecasts are scenario-based estimates, not guarantees. Use buffers for operational planning.",
                reliability="moderate" if confidence != "low" else "low",
                safety_grade="controlled",
            ),
            PrescriptiveAction(
                action="Stress-test the forecast under a downside demand scenario before committing full capacity.",
                rationale="Scenario planning improves resilience when forecast confidence is not perfect.",
                feasibility="medium",
                risk_level="low",
                affected_segments=[target],
                requires_experiment=False,
                causal_safety_note="Use operational buffers because forecast error can widen under shocks.",
                reliability="moderate" if confidence != "low" else "low",
                safety_grade="guarded",
            ),
        ]
    return [
        PrescriptiveAction(
            action=f"Shift 12% operational attention toward the strongest validated drivers of {target}.",
            rationale="Top model drivers are best used as directional prioritization, not as causal proof.",
            feasibility="medium",
            risk_level="medium",
            affected_segments=[target],
            requires_experiment=True,
            causal_safety_note=causal_safety_note,
            reliability="moderate" if confidence == "high" else "low",
            safety_grade="guarded",
        ),
        PrescriptiveAction(
            action="Run a limited controlled rollout before a broader policy shift.",
            rationale="Predictive patterns should be operationalized gradually unless backed by stronger causal proof.",
            feasibility="medium",
            risk_level="low",
            affected_segments=[target],
            requires_experiment=True,
            causal_safety_note=causal_safety_note,
            reliability="moderate" if confidence == "high" else "low",
            safety_grade="controlled",
        ),
    ]


def _operational_confidence(
    predictive_result: Dict[str, Any],
    scenarios: List[Dict[str, Any]],
) -> Dict[str, Any]:
    predictive_confidence = (predictive_result.get("confidence") or {}).copy()
    base = int(predictive_confidence.get("score", 40) or 40)
    reasons: List[str] = []
    driver_diagnostics = ((predictive_result.get("validation_summary") or {}).get("driver_diagnostics")) or {}
    top_share = float(driver_diagnostics.get("top_driver_share", 0.0) or 0.0)
    if top_share >= 0.6:
        base -= 18
        reasons.append("Operational confidence is lower because the recommendation depends heavily on one dominant driver.")
    truthfulness_flags = predictive_result.get("truthfulness_flags", []) or []
    if truthfulness_flags:
        base -= 10
        reasons.append("Operational confidence is reduced because causal certainty remains limited.")
    scenario_effects = [float(item.get("estimated_effect") or 0.0) for item in scenarios]
    spread_ratio = 0.0
    if scenario_effects:
        high = max(scenario_effects)
        low = min(scenario_effects)
        spread_ratio = abs(high - low) / max(abs(sum(scenario_effects) / len(scenario_effects)), 1e-6)
        if spread_ratio >= 1.0:
            base -= 12
            reasons.append("Scenario outcomes vary widely, so rollout confidence should remain controlled.")
        elif spread_ratio >= 0.5:
            base -= 6
            reasons.append("Scenario outcomes show meaningful spread, so operational estimates should be treated as ranges.")

    score = max(0, min(base, 100))
    label = "high" if score >= 75 else "moderate" if score >= 45 else "low"
    explanation = " ".join(reasons) if reasons else "Operational confidence remains close to predictive confidence because scenario risk is contained."
    return {
        "score": score,
        "label": label,
        "explanation": explanation,
        "factors": {
            "predictive_confidence": predictive_confidence,
            "feature_dominance": driver_diagnostics,
            "scenario_instability": {"spread_ratio": round(spread_ratio, 4)},
        },
    }


def run_prescriptive_analysis(predictive_result: Dict[str, Any], question: str) -> Dict[str, Any]:
    if predictive_result.get("error"):
        return {
            "tool": "prescriptive_analysis",
            "error": "Prescriptive analysis requires a successful predictive result first.",
            "upstream_error": predictive_result.get("error"),
        }

    problem_type = predictive_result.get("problem_type", "regression")
    target = predictive_result.get("target_column", "target")
    confidence = predictive_result.get("confidence_level", "low")
    objective = _objective_from_question(question)
    scenarios = simulate_scenarios(predictive_result, objective=objective)
    actions = _action_templates(problem_type, target, confidence)
    truthfulness_notes: List[str] = []
    memory_calibration = calibrate_from_memory(target)
    operational_confidence = _operational_confidence(predictive_result, scenarios)

    if predictive_result.get("no_reliable_recommendation"):
        truthfulness_notes.append("Predictive evidence is not stable enough for a reliable operational recommendation.")
        actions = [
            PrescriptiveAction(
                action="Collect more data or improve model quality before acting.",
                rationale="The current predictive signal is too weak or unstable for operational rollout.",
                feasibility="high",
                risk_level="low",
                affected_segments=[target],
                requires_experiment=False,
                causal_safety_note="No operational rollout is recommended from the current evidence base.",
                reliability="low",
                safety_grade="guarded",
            )
        ]

    for action in actions:
        if scenarios:
            best_match = scenarios[0]
            action.estimated_uplift = round(float(best_match.get("estimated_effect") or 0.0), 4)
            action.estimated_uplift_range = best_match.get("estimated_range", {})
            action.affected_segments = best_match.get("affected_segments", action.affected_segments or [])
            action.evidence_summary = best_match.get("evidence_summary", [])
            action.monitoring_kpis = best_match.get("monitoring_kpis", [])
            action.downside_risks = best_match.get("downside_risks", [])
            action.failure_conditions = best_match.get("failure_conditions", [])
            action.safety_grade = best_match.get("safety_grade", action.safety_grade)
            action.reliability = operational_confidence["label"]

    estimated_upside = None
    if scenarios:
        estimated_upside = round(sum(float(item.get("estimated_effect") or 0.0) for item in scenarios[:2]), 4)
        if memory_calibration.get("memory_adjustment"):
            estimated_upside = round(estimated_upside * (1.0 + float(memory_calibration["memory_adjustment"])), 4)

    optimization = optimize_actions(predictive_result, scenarios, question)
    if optimization.get("best_action"):
        best_action_text = optimization["best_action"]["recommended_action"]
        actions.sort(key=lambda item: 0 if item.action == best_action_text else 1)
        best_action = optimization["best_action"]
        if actions:
            actions[0].monitoring_kpis = best_action.get("monitoring_kpis", actions[0].monitoring_kpis)
            actions[0].downside_risks = best_action.get("downside_risks", actions[0].downside_risks)
            actions[0].failure_conditions = best_action.get("failure_conditions", actions[0].failure_conditions)
            actions[0].safety_grade = best_action.get("safety_grade", actions[0].safety_grade)
            actions[0].reliability = best_action.get("reliability", actions[0].reliability)

    store_recommendation_snapshot(
        {
            "target": target,
            "objective": objective,
            "predicted_uplift": estimated_upside,
            "recommended_action": actions[0].action if actions else None,
            "confidence_level": confidence,
        }
    )

    result = PrescriptiveResult(
        based_on_target=target,
        objective=objective,
        scenario_summary=scenarios,
        decision_paths=optimization.get("ranked_actions", []),
        recommended_actions=actions,
        estimated_upside=estimated_upside,
        assumptions=[
            "Scenario impacts are directional and derived from deterministic model signals, not randomized experimentation.",
            "Feature importance and predictive lift do not establish causal proof.",
            "Recommendations should be validated against business constraints before rollout.",
        ],
        confidence_level=confidence if confidence in {"low", "moderate", "high"} else "low",
        confidence=ConfidenceAssessment(**predictive_result["confidence"]) if predictive_result.get("confidence") else None,
        operational_confidence=ConfidenceAssessment(**operational_confidence),
        truthfulness_notes=[note for note in (
            truthfulness_notes
            or [
                "Use the recommendation as a directional planning input, not as causal proof.",
                "Validate any material rollout through controlled experimentation or phased deployment.",
                operational_confidence.get("explanation"),
                memory_calibration.get("note"),
            ]
        ) if note],
    )
    return result.model_dump()
