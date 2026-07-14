from __future__ import annotations

from typing import Any, Dict, List

from core.reasoning_layer import explain_decision, format_reasoning_explanation
from core.reasoning_objects import build_reasoning_objects
from state.state import AnalystState


def _stage_bucket(decision: Dict[str, Any]) -> str:
    return str(decision.get("stage") or decision.get("decision_type") or "unknown")


def reasoning_layer_node(state: AnalystState) -> AnalystState:
    evidence = state.setdefault("analysis_evidence", {})
    cache = evidence.setdefault("reasoning_cache", {})
    decision_objects = build_reasoning_objects(state)
    explanations: Dict[str, Dict[str, Any]] = {}
    status_map: Dict[str, str] = {}
    rendered_sections: Dict[str, List[str]] = {}

    for decision_object in decision_objects:
        reasoning, status = explain_decision(decision_object, cache=cache)
        stage = _stage_bucket(decision_object)
        explanations[stage] = {
            "decision_object": decision_object,
            "reasoning": reasoning,
            "status": status,
        }
        status_map[stage] = status
        rendered_sections[stage] = format_reasoning_explanation(reasoning)

    evidence["reasoning_layer"] = {
        "decision_objects": decision_objects,
        "explanations": explanations,
        "rendered_sections": rendered_sections,
        "status_map": status_map,
    }
    state["reasoning_objects"] = decision_objects
    state["llm_reasoning"] = explanations.get("report", {}).get("reasoning") or explanations.get("analysis_strategy", {}).get("reasoning") or explanations.get("data_preparation", {}).get("reasoning")
    state["llm_reasoning_status"] = "reasoning_layer:" + ",".join(sorted(set(status_map.values())))
    print("\n=== REASONING LAYER COMPLETE ===")
    print(
        {
            "stages": list(status_map.keys()),
            "statuses": status_map,
        }
    )
    return state
