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
            ),
            PrescriptiveAction(
                action="Run a controlled treatment on the top-risk segment before scaling.",
                rationale="A holdout or A/B design is needed before treating model lift as operational fact.",
                feasibility="medium",
                risk_level="low",
                affected_segments=[target],
                requires_experiment=True,
                causal_safety_note=causal_safety_note,
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
            ),
            PrescriptiveAction(
                action="Stress-test the forecast under a downside demand scenario before committing full capacity.",
                rationale="Scenario planning improves resilience when forecast confidence is not perfect.",
                feasibility="medium",
                risk_level="low",
                affected_segments=[target],
                requires_experiment=False,
                causal_safety_note="Use operational buffers because forecast error can widen under shocks.",
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
        ),
        PrescriptiveAction(
            action="Run a limited controlled rollout before a broader policy shift.",
            rationale="Predictive patterns should be operationalized gradually unless backed by stronger causal proof.",
            feasibility="medium",
            risk_level="low",
            affected_segments=[target],
            requires_experiment=True,
            causal_safety_note=causal_safety_note,
        ),
    ]


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
            )
        ]

    for action in actions:
        if scenarios:
            best_match = scenarios[0]
            action.estimated_uplift = round(float(best_match.get("estimated_effect") or 0.0), 4)
            action.estimated_uplift_range = best_match.get("estimated_range", {})
            if not action.affected_segments:
                action.affected_segments = best_match.get("affected_segments", [])

    estimated_upside = None
    if scenarios:
        estimated_upside = round(sum(float(item.get("estimated_effect") or 0.0) for item in scenarios[:2]), 4)
        if memory_calibration.get("memory_adjustment"):
            estimated_upside = round(estimated_upside * (1.0 + float(memory_calibration["memory_adjustment"])), 4)

    optimization = optimize_actions(predictive_result, scenarios, question)
    if optimization.get("best_action"):
        best_action_text = optimization["best_action"]["recommended_action"]
        actions.sort(key=lambda item: 0 if item.action == best_action_text else 1)

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
        truthfulness_notes=[note for note in (
            truthfulness_notes
            or [
                "Use the recommendation as a directional planning input, not as causal proof.",
                "Validate any material rollout through controlled experimentation or phased deployment.",
                memory_calibration.get("note"),
            ]
        ) if note],
    )
    return result.model_dump()
