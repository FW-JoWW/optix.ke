from __future__ import annotations

from typing import Any, Dict, List

from backend.core.reasoning_layer import explain_decision, format_reasoning_explanation
from backend.core.reasoning_objects import build_reasoning_objects
from backend.core.execution_state import record_execution_event, record_evidence_provenance
from backend.state.state import AnalystState


def _stage_bucket(decision: Dict[str, Any]) -> str:
    return str(decision.get("stage") or decision.get("decision_type") or "unknown")


def reasoning_layer_node(state: AnalystState) -> AnalystState:
    evidence = state.setdefault("analysis_evidence", {})
    cache = evidence.setdefault("reasoning_cache", {})
    decision_objects = build_reasoning_objects(state)
    allow_llm = not state.get("disable_llm_reasoning") and not state.get("fast_finalization")
    record_execution_event(
        state,
        phase="reasoning",
        message="Evaluating whether the evidence supports the current recommendation.",
        progress=72,
        operation="reasoning_layer",
        evidence_scope="sampled" if state.get("analysis_evidence", {}).get("profile_ambiguity", {}).get("requires_reasoning") else "exact",
    )
    explanations: Dict[str, Dict[str, Any]] = {}
    status_map: Dict[str, str] = {}
    rendered_sections: Dict[str, List[str]] = {}

    for decision_object in decision_objects:
        reasoning, status = explain_decision(decision_object, cache=cache, allow_llm=allow_llm)
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
    record_evidence_provenance(
        state,
        "reasoning_layer",
        scope="sampled" if not allow_llm else "exact",
        source="reasoning_layer_node",
        verified=allow_llm,
        method="live llm reasoning" if allow_llm else "deterministic fallback",
    )
    record_execution_event(
        state,
        phase="reasoning",
        message="Reasoning layer complete.",
        progress=78,
        operation="reasoning_layer",
        status="complete",
        evidence_scope="sampled" if not allow_llm else "exact",
    )
    print("\n=== REASONING LAYER COMPLETE ===")
    print(
        {
            "stages": list(status_map.keys()),
            "statuses": status_map,
        }
    )
    return state


